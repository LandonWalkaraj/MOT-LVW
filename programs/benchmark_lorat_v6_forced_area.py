from __future__ import annotations

import argparse
import csv
import math
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import benchmark_lorat_mot as bench
import benchmark_lorat_v5 as v6_bench
import bounding_box_v5_lorat_shared as v5
import bounding_box_v6_lorat_gated as v6
import exercise_lorat_mot as exercise


BBox = v5.BBox
DEFAULT_TARGET_AREAS = "128,256,512,1024,2048,4096,8192,16384,32768,65536,100000"
DEFAULT_OUTPUT_ROOT = v5.PROJECT_ROOT / "outputs" / "lorat-benchmarks" / "v6-forced-area"
DEFAULT_SEQUENCE = "dancetrack0065"
STOP_REQUESTED = False


@dataclass(frozen=True)
class ViewTransform:
    crop_x: float
    crop_y: float
    crop_w: int
    crop_h: int
    output_w: int
    output_h: int

    @property
    def scale_x(self) -> float:
        return float(self.output_w) / float(max(1, self.crop_w))

    @property
    def scale_y(self) -> float:
        return float(self.output_h) / float(max(1, self.crop_h))


@dataclass(frozen=True)
class ForcedAreaResult:
    sequence: str
    lorat_config: str
    execution_mode: str
    target_area_px: float
    source_init_area_px: float
    realized_init_area_px: float
    init_frame: int
    gt_track_id: int
    frames: int
    update_frames: int
    valid_samples: int
    sample_interval: int
    min_required_samples: int
    sufficient_samples: bool
    mean_iou: Optional[float]
    iou50: Optional[float]
    unreliable_rate: Optional[float]
    reliable: Optional[bool]
    total_seconds: float
    tracking_seconds: float
    fps_tracking: Optional[float]
    gpu_name: str
    gpu_peak_reserved_mb: Optional[float]
    crop_x: float
    crop_y: float
    crop_w: int
    crop_h: int
    scale_x: float
    scale_y: float
    preview_path: str


def request_stop(signum, _frame) -> None:
    del signum
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested; finishing the current forced-area case and flushing outputs.", flush=True)


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0:
            raise ValueError("Target areas must be positive.")
        values.append(value)
    if not values:
        raise ValueError("At least one target area is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V6 forced-area small-object benchmark. The initialized target is rendered at explicit "
            "pixel areas instead of relying on whatever GT sizes naturally occur in the sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--extract-zips", action="store_true")
    parser.add_argument("--init-frame", default="auto")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to use. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--target-areas", default=DEFAULT_TARGET_AREAS)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--max-samples-per-target",
        type=int,
        default=10,
        help="Stop each target-area case after this many valid samples. Use 0 to run through max-frames/full sequence.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until enough samples or sequence end.")
    parser.add_argument("--min-transformed-visibility", type=float, default=0.60)
    parser.add_argument("--reliable-iou50", type=float, default=0.80)
    parser.add_argument("--reliable-mean-iou", type=float, default=0.50)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--gpu-profile", default="hpc")

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lorat-root", type=Path, default=v5.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v5.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(v5.LORAT_WEIGHT_BY_CONFIG) + ("all",),
        help="Run one or more LoRAT configs. Use all to include every known config.",
    )
    parser.add_argument("--weight-path", type=Path)
    v6_bench.add_v5_runtime_args(parser)
    v6.add_v6_runtime_args(parser)
    return parser.parse_args()


def normalized_configs(args: argparse.Namespace) -> List[str]:
    if args.compare_configs:
        if args.weight_path:
            raise RuntimeError("--weight-path cannot be combined with --compare-configs.")
        return exercise.normalized_compare_configs(args.compare_configs)
    return [args.lorat_config]


def select_sequence(args: argparse.Namespace) -> Path:
    if args.extract_zips:
        exercise.extract_zips(args.dataset_root)
    sequences = exercise.find_sequences(args.dataset_root, args.split)
    for sequence in sequences:
        if sequence.name == args.sequence:
            return sequence
    raise RuntimeError(f"Requested sequence not found under {args.dataset_root}: {args.sequence}")


def class_ids(args: argparse.Namespace) -> List[int]:
    return list(dict.fromkeys(args.class_id or [1]))


def visible_rows(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    frame_number: int,
    wanted_class_ids: Sequence[int],
    min_visibility: float,
) -> List[exercise.GroundTruthRow]:
    wanted = set(wanted_class_ids)
    return [
        row
        for row in gt_by_frame.get(frame_number, [])
        if row.confidence != 0
        and row.visibility >= min_visibility
        and row.class_id in wanted
        and row.bbox[2] > 1
        and row.bbox[3] > 1
    ]


def choose_initial_row(
    args: argparse.Namespace,
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
) -> exercise.GroundTruthRow:
    wanted = class_ids(args)
    if args.init_frame != "auto":
        init_frame, rows = exercise.pick_initial_rows(
            gt_by_frame,
            args.init_frame,
            wanted,
            args.min_visibility,
            1,
            1,
        )
        if not rows:
            raise RuntimeError(f"No usable GT object at init frame {init_frame}.")
        return rows[0]

    frame_numbers = sorted(gt_by_frame)
    if not frame_numbers:
        raise RuntimeError("No GT rows found.")
    best: Optional[Tuple[int, float, exercise.GroundTruthRow]] = None
    for frame_number in frame_numbers:
        for row in visible_rows(gt_by_frame, frame_number, wanted, args.min_visibility):
            max_frame = frame_numbers[-1]
            if args.max_frames > 0:
                max_frame = min(max_frame, frame_number + args.max_frames - 1)
            sample_count = 0
            for sample_frame in range(frame_number, max_frame + 1, max(1, args.sample_interval)):
                if any(
                    candidate.track_id == row.track_id
                    for candidate in visible_rows(gt_by_frame, sample_frame, wanted, args.min_visibility)
                ):
                    sample_count += 1
            area = float(row.bbox[2]) * float(row.bbox[3])
            score = (sample_count, area)
            if best is None or score > (best[0], best[1]):
                best = (sample_count, area, row)
    if best is None:
        raise RuntimeError("No usable GT object found for forced-area benchmark.")
    if best[0] < args.min_samples:
        print(
            f"Warning: best GT track only has {best[0]} natural 10-frame samples; "
            f"results will mark insufficient targets if transformed visibility falls below {args.min_samples}.",
            flush=True,
        )
    return best[2]


def make_transform(frame_shape: Tuple[int, ...], init_bbox: BBox, target_area_px: float) -> ViewTransform:
    frame_h, frame_w = frame_shape[:2]
    source_area = max(1.0, float(init_bbox[2]) * float(init_bbox[3]))
    scale = math.sqrt(max(1.0, target_area_px) / source_area)
    crop_w = max(4, int(round(float(frame_w) / max(1e-6, scale))))
    crop_h = max(4, int(round(float(frame_h) / max(1e-6, scale))))
    center_x, center_y = v5.bbox_center(init_bbox)
    return ViewTransform(
        crop_x=float(center_x) - (float(crop_w) / 2.0),
        crop_y=float(center_y) - (float(crop_h) / 2.0),
        crop_w=crop_w,
        crop_h=crop_h,
        output_w=int(frame_w),
        output_h=int(frame_h),
    )


def render_transformed_frame(frame: np.ndarray, transform: ViewTransform) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    crop_x0 = int(math.floor(transform.crop_x))
    crop_y0 = int(math.floor(transform.crop_y))
    crop_x1 = crop_x0 + transform.crop_w
    crop_y1 = crop_y0 + transform.crop_h
    canvas = np.zeros((transform.crop_h, transform.crop_w, frame.shape[2]), dtype=frame.dtype)

    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(source_w, crop_x1)
    src_y1 = min(source_h, crop_y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - crop_x0
        dst_y0 = src_y0 - crop_y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return cv2.resize(canvas, (transform.output_w, transform.output_h), interpolation=cv2.INTER_LINEAR)


def transform_bbox(bbox: BBox, transform: ViewTransform) -> BBox:
    x, y, w, h = bbox
    crop_x0 = float(math.floor(transform.crop_x))
    crop_y0 = float(math.floor(transform.crop_y))
    return (
        (float(x) - crop_x0) * transform.scale_x,
        (float(y) - crop_y0) * transform.scale_y,
        float(w) * transform.scale_x,
        float(h) * transform.scale_y,
    )


def clip_bbox_to_shape(bbox: BBox, frame_shape: Tuple[int, ...]) -> Tuple[Optional[BBox], float]:
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = bbox
    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    x1 = min(float(frame_w), float(x) + max(0.0, float(w)))
    y1 = min(float(frame_h), float(y) + max(0.0, float(h)))
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    full_area = max(1.0, max(0.0, float(w)) * max(0.0, float(h)))
    visible_area = (x1 - x0) * (y1 - y0)
    return (x0, y0, x1 - x0, y1 - y0), visible_area / full_area


def backend_args(args: argparse.Namespace, lorat_config: str) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.lorat_config = lorat_config
    run_args.max_tracks = 1
    return run_args


def runtime_status(backend) -> v5.RuntimeStatus:
    if hasattr(backend, "runtime_status_snapshot"):
        return backend.runtime_status_snapshot()
    return v5.RuntimeStatus(active_objects=sum(1 for track in backend.tracks if track.ok))


def run_case(
    args: argparse.Namespace,
    sequence_path: Path,
    image_paths: Sequence[Path],
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    fps: float,
    sequence_length: int,
    init_row: exercise.GroundTruthRow,
    lorat_config: str,
    target_area_px: float,
    run_root: Path,
    observations: List[Dict[str, object]],
) -> ForcedAreaResult:
    init_index = exercise.frame_to_image_index(init_row.frame)
    init_frame = cv2.imread(str(image_paths[init_index]))
    if init_frame is None:
        raise RuntimeError(f"Unable to read frame: {image_paths[init_index]}")
    transform = make_transform(init_frame.shape, init_row.bbox, target_area_px)
    transformed_init = render_transformed_frame(init_frame, transform)
    transformed_init_bbox, init_visibility = clip_bbox_to_shape(transform_bbox(init_row.bbox, transform), transformed_init.shape)
    if transformed_init_bbox is None or init_visibility < args.min_transformed_visibility:
        raise RuntimeError(
            f"Target area {target_area_px:g} makes the initial GT box insufficiently visible "
            f"({init_visibility:.3f})."
        )

    source = SimpleNamespace(fps=fps, length=sequence_length, name=f"{sequence_path.name}-forced-area-{int(target_area_px)}")
    backend = v6.create_backend(backend_args(args, lorat_config), source, expected_tracks=1)
    preview_path = ""
    preview_writer = None
    if args.save_video:
        video_dir = run_root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        preview_path_obj = bench.unique_path(video_dir / f"{sequence_path.name}_{lorat_config}_area{int(target_area_px)}.mp4")
        preview_path = str(preview_path_obj)
        preview_writer = v5.make_video_writer(preview_path_obj, fps, transformed_init)

    end_index = len(image_paths) - 1
    if args.max_frames > 0:
        end_index = min(end_index, init_index + args.max_frames - 1)
    total_started = time.perf_counter()
    tracking_seconds = 0.0
    sampled_ious: List[float] = []
    valid_samples = 0
    frames = 1
    last_frame_number = init_row.frame

    try:
        backend.initialize(transformed_init, [transformed_init_bbox], init_row.frame)
        if preview_writer is not None:
            preview_writer.write(v5.draw_tracks(transformed_init, backend.tracks, init_row.frame, backend.backend_name))

        for image_index in range(init_index, end_index + 1):
            if STOP_REQUESTED:
                break
            frame_number = image_index + 1
            if image_index == init_index:
                transformed_frame = transformed_init
            else:
                frame = cv2.imread(str(image_paths[image_index]))
                if frame is None:
                    continue
                transformed_frame = render_transformed_frame(frame, transform)
                update_started = time.perf_counter()
                backend.update(transformed_frame, frame_number)
                tracking_seconds += time.perf_counter() - update_started
                frames += 1
                last_frame_number = frame_number
                if preview_writer is not None:
                    preview_writer.write(v5.draw_tracks(transformed_frame, backend.tracks, frame_number, backend.backend_name))

            if (frame_number - init_row.frame) % max(1, args.sample_interval) != 0:
                continue
            gt_row = next(
                (row for row in gt_by_frame.get(frame_number, []) if row.track_id == init_row.track_id),
                None,
            )
            if gt_row is None or gt_row.confidence == 0 or gt_row.visibility < args.min_visibility:
                continue
            transformed_gt, transformed_visibility = clip_bbox_to_shape(
                transform_bbox(gt_row.bbox, transform),
                transformed_frame.shape,
            )
            if transformed_gt is None or transformed_visibility < args.min_transformed_visibility:
                continue
            track = backend.tracks[0]
            own_iou = exercise.bbox_iou(track.bbox, transformed_gt)
            sampled_ious.append(own_iou)
            valid_samples += 1
            observations.append(
                {
                    "sequence": sequence_path.name,
                    "lorat_config": lorat_config,
                    "execution_mode": v6.V6_EXECUTION_MODE,
                    "target_area_px": f"{target_area_px:.0f}",
                    "frame": frame_number,
                    "sample_offset": frame_number - init_row.frame,
                    "gt_track_id": init_row.track_id,
                    "actual_transformed_gt_area_px": f"{transformed_gt[2] * transformed_gt[3]:.3f}",
                    "transformed_visibility": f"{transformed_visibility:.6f}",
                    "iou": f"{own_iou:.6f}",
                    "ok": int(bool(getattr(track, "ok", False))),
                    "state": str(getattr(track, "state", "")),
                    "bbox_x": f"{track.bbox[0]:.3f}",
                    "bbox_y": f"{track.bbox[1]:.3f}",
                    "bbox_w": f"{track.bbox[2]:.3f}",
                    "bbox_h": f"{track.bbox[3]:.3f}",
                    "gt_x": f"{transformed_gt[0]:.3f}",
                    "gt_y": f"{transformed_gt[1]:.3f}",
                    "gt_w": f"{transformed_gt[2]:.3f}",
                    "gt_h": f"{transformed_gt[3]:.3f}",
                }
            )
            if args.max_samples_per_target > 0 and valid_samples >= args.max_samples_per_target:
                break
            if args.progress_interval > 0 and valid_samples and valid_samples % args.progress_interval == 0:
                print(
                    f"{sequence_path.name} {lorat_config} target_area={target_area_px:g}: "
                    f"{valid_samples} valid samples collected",
                    flush=True,
                )
    finally:
        status = runtime_status(backend)
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    total_seconds = time.perf_counter() - total_started
    update_frames = max(0, frames - 1)
    fps_tracking = update_frames / tracking_seconds if tracking_seconds > 0 and update_frames else None
    mean_iou = statistics.fmean(sampled_ious) if sampled_ious else None
    iou50 = (sum(1 for value in sampled_ious if value >= 0.5) / len(sampled_ious)) if sampled_ious else None
    sufficient = valid_samples >= args.min_samples
    reliable = None
    if sufficient and mean_iou is not None and iou50 is not None:
        reliable = mean_iou >= args.reliable_mean_iou and iou50 >= args.reliable_iou50
    return ForcedAreaResult(
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        execution_mode=v6.V6_EXECUTION_MODE,
        target_area_px=target_area_px,
        source_init_area_px=float(init_row.bbox[2]) * float(init_row.bbox[3]),
        realized_init_area_px=float(transformed_init_bbox[2]) * float(transformed_init_bbox[3]),
        init_frame=init_row.frame,
        gt_track_id=init_row.track_id,
        frames=max(1, last_frame_number - init_row.frame + 1),
        update_frames=update_frames,
        valid_samples=valid_samples,
        sample_interval=args.sample_interval,
        min_required_samples=args.min_samples,
        sufficient_samples=sufficient,
        mean_iou=mean_iou,
        iou50=iou50,
        unreliable_rate=(1.0 - iou50) if iou50 is not None else None,
        reliable=reliable,
        total_seconds=total_seconds,
        tracking_seconds=tracking_seconds,
        fps_tracking=fps_tracking,
        gpu_name=status.gpu_name,
        gpu_peak_reserved_mb=status.gpu_peak_reserved_mb,
        crop_x=transform.crop_x,
        crop_y=transform.crop_y,
        crop_w=transform.crop_w,
        crop_h=transform.crop_h,
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        preview_path=preview_path,
    )


def write_results_csv(path: Path, rows: Sequence[ForcedAreaResult]) -> None:
    fieldnames = list(ForcedAreaResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        bench.optional_float(getattr(row, field), 6)
                        if isinstance(getattr(row, field), float)
                        else getattr(row, field)
                    )
                    for field in fieldnames
                }
            )


def write_observations_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "execution_mode",
        "target_area_px",
        "frame",
        "sample_offset",
        "gt_track_id",
        "actual_transformed_gt_area_px",
        "transformed_visibility",
        "iou",
        "ok",
        "state",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "gt_x",
        "gt_y",
        "gt_w",
        "gt_h",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, args: argparse.Namespace, rows: Sequence[ForcedAreaResult], observations_csv: Path, results_csv: Path) -> None:
    lines = [
        "# V6 Forced-Area Small-Object Benchmark",
        "",
        "## Method",
        "",
        "- This benchmark explicitly forces the initialized target object to known pixel areas instead of relying on naturally occurring GT object sizes.",
        "- For each target area, the video is rendered through one fixed viewport anchored at the selected first-frame GT box.",
        "- The viewport is resized back to the original frame size, making the initialized GT box area equal to the requested target area up to rounding.",
        f"- Reliability is sampled every `{args.sample_interval}` frames.",
        f"- A target area is judged only when it has at least `{args.min_samples}` valid sampled frames.",
        f"- Reliable rule: IoU@0.50 >= `{args.reliable_iou50}` and mean IoU >= `{args.reliable_mean_iou}`.",
        "",
        "## Output Files",
        "",
        f"- Per-target summary CSV: `{results_csv}`",
        f"- Per-sample observations CSV: `{observations_csv}`",
        "",
        "## Results",
        "",
        "| Config | Target Area px | Realized Init Area px | Samples | Sufficient | Mean IoU | IoU@0.50 | Reliable | FPS Tracking | Peak GPU Reserved MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.lorat_config} | {row.target_area_px:.0f} | {row.realized_init_area_px:.0f} | "
            f"{row.valid_samples} | {row.sufficient_samples} | {bench.optional_float(row.mean_iou, 3)} | "
            f"{bench.optional_float(row.iou50, 3)} | {'' if row.reliable is None else row.reliable} | "
            f"{bench.optional_float(row.fps_tracking, 3)} | {bench.optional_float(row.gpu_peak_reserved_mb, 1)} |"
        )
    lines.extend(["", "## Smallest Reliable Forced Area", ""])
    lines.append("| Config | Smallest Reliable Target Area px |")
    lines.append("|---|---:|")
    by_config: Dict[str, List[ForcedAreaResult]] = {}
    for row in rows:
        by_config.setdefault(row.lorat_config, []).append(row)
    for config, config_rows in sorted(by_config.items()):
        reliable_areas = [row.target_area_px for row in config_rows if row.reliable is True]
        lines.append(f"| {config} | {'' if not reliable_areas else int(min(reliable_areas))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flush_outputs(run_root: Path, args: argparse.Namespace, results: Sequence[ForcedAreaResult], observations: Sequence[Dict[str, object]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    results_csv = run_root / "forced_area_summary.csv"
    observations_csv = run_root / "forced_area_observations.csv"
    summary_md = run_root / "summary.md"
    write_results_csv(results_csv, results)
    write_observations_csv(observations_csv, observations)
    write_summary(summary_md, args, results, observations_csv, results_csv)


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parse_args()
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive.")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive.")
    if args.max_frames > 0:
        required_frames = 1 + ((args.min_samples - 1) * args.sample_interval)
        if args.max_frames < required_frames:
            raise ValueError(
                f"--max-frames {args.max_frames} cannot produce {args.min_samples} samples "
                f"at interval {args.sample_interval}; need at least {required_frames}."
            )

    sequence_path = select_sequence(args)
    image_paths = exercise.get_image_paths(sequence_path)
    if not image_paths:
        raise RuntimeError(f"No frames found in {sequence_path / 'img1'}")
    gt_by_frame = exercise.read_gt(sequence_path)
    fps, sequence_length = exercise.read_sequence_info(sequence_path)
    init_row = choose_initial_row(args, gt_by_frame)
    target_areas = parse_float_list(args.target_areas)
    configs = normalized_configs(args)
    label = bench.slugify(
        f"v6_forced_area_{sequence_path.name}_{'-'.join(configs)}_{int(min(target_areas))}-{int(max(target_areas))}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = args.output_root.resolve() / label
    results: List[ForcedAreaResult] = []
    observations: List[Dict[str, object]] = []
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"V6 forced-area run: {label}", flush=True)
    print(f"Output folder: {run_root}", flush=True)
    print(f"Sequence: {sequence_path.name}", flush=True)
    print(f"Selected GT track: {init_row.track_id} at frame {init_row.frame}", flush=True)
    print(f"Target areas: {', '.join(str(int(area)) for area in target_areas)}", flush=True)
    print(f"Sampling rule: every {args.sample_interval} frames, minimum {args.min_samples} valid samples", flush=True)

    for lorat_config in configs:
        for target_area in target_areas:
            if STOP_REQUESTED:
                flush_outputs(run_root, args, results, observations)
                return 0
            print(f"Starting {lorat_config} forced target area {target_area:g}px", flush=True)
            result = run_case(
                args,
                sequence_path,
                image_paths,
                gt_by_frame,
                fps,
                sequence_length or len(image_paths),
                init_row,
                lorat_config,
                target_area,
                run_root,
                observations,
            )
            results.append(result)
            flush_outputs(run_root, args, results, observations)
            print(
                f"{lorat_config} area={target_area:g}px samples={result.valid_samples} "
                f"mean_iou={bench.optional_float(result.mean_iou, 3)} "
                f"iou50={bench.optional_float(result.iou50, 3)} reliable={result.reliable}",
                flush=True,
            )
    flush_outputs(run_root, args, results, observations)
    print(f"Wrote {run_root / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
