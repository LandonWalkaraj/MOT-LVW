from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import benchmark_lorat_mot as bench
import benchmark_lorat_v4 as bench_v4
import exercise_lorat_mot as exercise
import bounding_box_v4_lorat_memory as v4


BBox = Tuple[float, float, float, float]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-benchmarks" / "v4-area-stress"
DEFAULT_SEQUENCE = "dancetrack0065"
DEFAULT_TARGET_AREAS = "128,256,384,512,768,1000,1500,2048,3072,4096,8192"
DEFAULT_MAX_FRAMES = 120


@dataclass(frozen=True)
class TrackSegment:
    sequence: str
    track_id: int
    start_frame: int
    end_frame: int
    rows: Tuple[exercise.GroundTruthRow, ...]


@dataclass(frozen=True)
class StressObservation:
    sequence: str
    lorat_config: str
    gt_track_id: int
    target_area_px: float
    frame: int
    actual_area_px: float
    iou: float
    ok: bool
    confidence: Optional[float]
    state: str
    elapsed_ms: float


@dataclass(frozen=True)
class StressResult:
    sequence: str
    lorat_config: str
    backbone: str
    input_size: int
    device: str
    gt_track_id: int
    target_area_px: float
    frames: int
    update_frames: int
    total_seconds: float
    init_seconds: float
    tracking_seconds: float
    tracking_ms_per_bbox: Optional[float]
    samples: int
    mean_actual_area_px: Optional[float]
    mean_iou: Optional[float]
    iou50: Optional[float]
    lost_rate: Optional[float]
    mean_confidence: Optional[float]
    reliable: Optional[bool]
    preview_path: Optional[Path]


@dataclass(frozen=True)
class OutputPaths:
    run_root: Path
    results_csv: Path
    observations_csv: Path
    summary_csv: Path
    summary_md: Path
    video_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled V4 LoRAT small-object benchmark. The script rescales a "
            "ground-truth target to fixed pixel-area levels and compares model sizes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--sequence",
        action="append",
        help=f"Sequence name to run. Repeat for multiple sequences. Defaults to {DEFAULT_SEQUENCE}.",
    )
    parser.add_argument("--list-sequences", action="store_true")
    parser.add_argument("--extract-zips", action="store_true")

    parser.add_argument("--device", default="cpu", help="LoRAT device, e.g. cpu, dml, directml, or cuda:0.")
    parser.add_argument("--lorat-root", type=Path, default=v4.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v4.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(v4.LORAT_WEIGHT_BY_CONFIG) + ("all",),
        help="Run multiple LoRAT model sizes/configs against the same controlled area levels.",
    )
    parser.add_argument("--weight-path", type=Path, help="Optional LoRAT weight override. Cannot be used with --compare-configs.")
    parser.add_argument("--skip-missing-weights", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--target-areas", default=DEFAULT_TARGET_AREAS, help="Comma-separated forced target bbox areas in pixels.")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Frames per area/config run; 0 means full selected segment.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-source-tracks", type=int, default=1, help="Number of GT tracks to stress per sequence.")
    parser.add_argument("--track-id", type=int, action="append", help="Specific GT track ID to stress. Repeat for multiple tracks.")
    parser.add_argument("--init-frame", default="auto")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to use. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--min-track-frames", type=int, default=30)

    parser.add_argument("--context-factor", type=float, default=4.0, help="Crop context around the GT object before resizing.")
    parser.add_argument("--background", choices=("blur", "solid"), default="blur")
    parser.add_argument("--background-blur-sigma", type=float, default=18.0)
    parser.add_argument("--solid-background", default="32,32,32", help="BGR color for --background solid.")

    parser.add_argument("--reliable-iou50", type=float, default=0.80)
    parser.add_argument("--reliable-mean-iou", type=float, default=0.50)
    parser.add_argument("--reliable-max-lost-rate", type=float, default=0.20)
    parser.add_argument("--min-area-samples", type=int, default=10)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--write-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-interval", type=int, default=10)

    bench_v4.add_v4_runtime_args(parser)
    parser.set_defaults(lorat_min_box_area=0.0)
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise ValueError("Target areas must be positive.")
        values.append(value)
    if not values:
        raise ValueError("At least one target area is required.")
    return sorted(dict.fromkeys(values))


def parse_bgr(text: str) -> Tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--solid-background must be a B,G,R triple like 32,32,32.")
    return tuple(max(0, min(255, value)) for value in parts)  # type: ignore[return-value]


def normalized_configs(args: argparse.Namespace) -> List[str]:
    if args.compare_configs:
        if args.weight_path:
            raise RuntimeError("--weight-path cannot be combined with --compare-configs.")
        return exercise.normalized_compare_configs(args.compare_configs)
    return [args.lorat_config]


def select_sequences(args: argparse.Namespace) -> List[Path]:
    if args.extract_zips:
        exercise.extract_zips(args.dataset_root)
    sequences = exercise.find_sequences(args.dataset_root, args.split)
    if args.list_sequences:
        for sequence in sequences:
            print(sequence)
        return []
    if not sequences:
        raise RuntimeError(f"No extracted sequences with img1 folders found under {args.dataset_root}.")
    wanted = set(args.sequence or [DEFAULT_SEQUENCE])
    sequences = [sequence for sequence in sequences if sequence.name in wanted]
    missing = wanted - {sequence.name for sequence in sequences}
    if missing:
        raise RuntimeError(f"Requested sequences not found: {sorted(missing)}")
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    return sequences


def make_run_label(args: argparse.Namespace, sequences: Sequence[Path], configs: Sequence[str], areas: Sequence[float]) -> str:
    sequence_part = "-".join(sequence.name for sequence in sequences)
    config_part = "-".join(configs)
    min_area = min(areas)
    max_area = max(areas)
    min_area_text = str(int(min_area)) if min_area.is_integer() else f"{min_area:g}"
    max_area_text = str(int(max_area)) if max_area.is_integer() else f"{max_area:g}"
    area_part = f"{min_area_text}-{max_area_text}_n{len(areas)}"
    frame_part = f"frames{args.max_frames}" if args.max_frames > 0 else "full"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return bench.slugify(f"v4_area_{sequence_part}_{config_part}_A{area_part}_{frame_part}_{timestamp}")


def default_output_paths(args: argparse.Namespace, run_label: str) -> OutputPaths:
    run_root = args.output_root.resolve() / run_label
    results_csv = args.results_csv.resolve() if args.results_csv else run_root / "area_stress_results.csv"
    observations_csv = (
        args.observations_csv.resolve()
        if args.observations_csv
        else run_root / "area_stress_observations.csv"
    )
    summary_csv = args.summary_csv.resolve() if args.summary_csv else run_root / "area_stress_summary.csv"
    summary_md = args.summary_md.resolve() if args.summary_md else run_root / "area_stress_summary.md"
    video_dir = run_root / "videos"
    for path in (results_csv, observations_csv, summary_csv, summary_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        run_root=run_root,
        results_csv=bench.unique_path(results_csv),
        observations_csv=bench.unique_path(observations_csv),
        summary_csv=bench.unique_path(summary_csv),
        summary_md=bench.unique_path(summary_md),
        video_dir=video_dir,
    )


def valid_gt_rows(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    class_ids: Optional[Sequence[int]],
    min_visibility: float,
) -> Dict[int, List[exercise.GroundTruthRow]]:
    class_id_set = set(class_ids or [1])
    filtered: Dict[int, List[exercise.GroundTruthRow]] = {}
    for frame, rows in gt_by_frame.items():
        usable = [
            row
            for row in rows
            if row.confidence != 0
            and row.class_id in class_id_set
            and row.visibility >= min_visibility
            and row.bbox[2] > 2
            and row.bbox[3] > 2
        ]
        if usable:
            filtered[frame] = usable
    return filtered


def contiguous_segments(rows: Sequence[exercise.GroundTruthRow]) -> List[Tuple[exercise.GroundTruthRow, ...]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: row.frame)
    segments: List[List[exercise.GroundTruthRow]] = [[sorted_rows[0]]]
    for row in sorted_rows[1:]:
        if row.frame == segments[-1][-1].frame + 1:
            segments[-1].append(row)
        else:
            segments.append([row])
    return [tuple(segment) for segment in segments]


def pick_track_segments(args: argparse.Namespace, sequence_path: Path) -> List[TrackSegment]:
    gt_by_frame = valid_gt_rows(exercise.read_gt(sequence_path), args.class_id, args.min_visibility)
    rows_by_track: Dict[int, List[exercise.GroundTruthRow]] = {}
    for rows in gt_by_frame.values():
        for row in rows:
            if args.track_id and row.track_id not in set(args.track_id):
                continue
            rows_by_track.setdefault(row.track_id, []).append(row)

    candidates: List[TrackSegment] = []
    for track_id, rows in rows_by_track.items():
        for segment in contiguous_segments(rows):
            if args.init_frame != "auto":
                init_frame = int(args.init_frame)
                segment = tuple(row for row in segment if row.frame >= init_frame)
                if not segment or segment[0].frame != init_frame:
                    continue
            if len(segment) < args.min_track_frames:
                continue
            if args.max_frames > 0:
                segment = segment[: args.max_frames]
            candidates.append(
                TrackSegment(
                    sequence=sequence_path.name,
                    track_id=track_id,
                    start_frame=segment[0].frame,
                    end_frame=segment[-1].frame,
                    rows=segment,
                )
            )

    if not candidates:
        raise RuntimeError(f"No usable GT track segments found in {sequence_path.name}.")

    candidates.sort(key=lambda item: (-len(item.rows), item.start_frame, item.track_id))
    if args.track_id:
        return candidates
    return candidates[: max(1, args.max_source_tracks)]


def crop_with_padding(frame: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    height, width = frame.shape[:2]
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(width, right)
    src_bottom = min(height, bottom)
    if src_right <= src_left or src_bottom <= src_top:
        raise RuntimeError("Synthetic crop is outside the source frame.")

    crop = frame[src_top:src_bottom, src_left:src_right]
    pad_left = src_left - left
    pad_top = src_top - top
    pad_right = right - src_right
    pad_bottom = bottom - src_bottom
    if any(value > 0 for value in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    return crop


def make_background(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.background == "solid":
        color = parse_bgr(args.solid_background)
        return np.full_like(frame, color)
    sigma = max(0.0, float(args.background_blur_sigma))
    if sigma <= 0:
        return frame.copy()
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


def paste_clipped(canvas: np.ndarray, patch: np.ndarray, left: int, top: int) -> None:
    canvas_h, canvas_w = canvas.shape[:2]
    patch_h, patch_w = patch.shape[:2]
    dst_left = max(0, left)
    dst_top = max(0, top)
    dst_right = min(canvas_w, left + patch_w)
    dst_bottom = min(canvas_h, top + patch_h)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        return
    src_left = dst_left - left
    src_top = dst_top - top
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    canvas[dst_top:dst_bottom, dst_left:dst_right] = patch[src_top:src_bottom, src_left:src_right]


def build_synthetic_frame(
    image_path: Path,
    row: exercise.GroundTruthRow,
    target_area_px: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, BBox, float]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Unable to read frame: {image_path}")

    x, y, width, height = row.bbox
    source_area = width * height
    if source_area <= 0:
        raise RuntimeError(f"Invalid GT bbox area for track {row.track_id} frame {row.frame}.")

    scale = math.sqrt(target_area_px / source_area)
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    crop_width = max(width * args.context_factor, width + 2.0)
    crop_height = max(height * args.context_factor, height + 2.0)
    crop_left = int(math.floor(center_x - crop_width * 0.5))
    crop_top = int(math.floor(center_y - crop_height * 0.5))
    crop_right = int(math.ceil(center_x + crop_width * 0.5))
    crop_bottom = int(math.ceil(center_y + crop_height * 0.5))

    crop = crop_with_padding(frame, crop_left, crop_top, crop_right, crop_bottom)
    scaled_width = max(1, int(round(crop.shape[1] * scale)))
    scaled_height = max(1, int(round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    patch = cv2.resize(crop, (scaled_width, scaled_height), interpolation=interpolation)

    box_x_in_patch = (x - crop_left) * scale
    box_y_in_patch = (y - crop_top) * scale
    box_w = width * scale
    box_h = height * scale
    paste_left = int(round(center_x - (box_x_in_patch + box_w * 0.5)))
    paste_top = int(round(center_y - (box_y_in_patch + box_h * 0.5)))
    synthetic_bbox = (
        paste_left + box_x_in_patch,
        paste_top + box_y_in_patch,
        box_w,
        box_h,
    )

    canvas = make_background(frame, args)
    paste_clipped(canvas, patch, paste_left, paste_top)
    return canvas, synthetic_bbox, box_w * box_h


def clip_bbox_to_frame_shape(frame: np.ndarray, bbox: BBox) -> Optional[BBox]:
    clipped = v4.clip_bbox_to_frame(frame, bbox)
    if clipped is None:
        return None
    return tuple(float(value) for value in clipped)


def preview_video_path(output_paths: OutputPaths, sequence: str, config: str, track_id: int, area: float) -> Path:
    area_text = str(int(area)) if area.is_integer() else f"{area:g}"
    filename = bench.slugify(f"{sequence}_{config}_gt{track_id}_area{area_text}_preview") + ".mp4"
    return bench.unique_path(output_paths.video_dir / filename)


def run_stress_case(
    args: argparse.Namespace,
    sequence_path: Path,
    image_paths: Sequence[Path],
    segment: TrackSegment,
    lorat_config: str,
    target_area_px: float,
    preview_path: Optional[Path],
) -> Optional[Tuple[StressResult, List[StressObservation]]]:
    weight_path = args.weight_path or v4.LORAT_WEIGHT_BY_CONFIG[lorat_config]
    if not weight_path.exists():
        message = f"Missing LoRAT weight for {lorat_config}: {weight_path}"
        if args.skip_missing_weights:
            print(f"Skipping {segment.sequence} {lorat_config} area={target_area_px:g}: {message}")
            return None
        raise FileNotFoundError(message)

    first_row = segment.rows[0]
    first_index = exercise.frame_to_image_index(first_row.frame)
    first_frame, first_bbox, _ = build_synthetic_frame(image_paths[first_index], first_row, target_area_px, args)
    initial_bbox = clip_bbox_to_frame_shape(first_frame, first_bbox)
    if initial_bbox is None:
        print(f"Skipping {segment.sequence} gt={segment.track_id} area={target_area_px:g}: initial box is too small.")
        return None

    fps, sequence_length = exercise.read_sequence_info(sequence_path)
    backend = bench_v4.build_v4_backend(args, lorat_config, 1, f"{segment.sequence}_area{target_area_px:g}", fps, sequence_length)
    observations: List[StressObservation] = []
    preview_writer = None
    tracking_seconds = 0.0
    total_started = time.perf_counter()
    init_started = time.perf_counter()
    try:
        backend.initialize(first_frame, [initial_bbox], first_row.frame)
        init_seconds = time.perf_counter() - init_started
        if args.save_video and preview_path is not None:
            preview_writer = v4.make_video_writer(preview_path, fps, first_frame)
            preview_writer.write(v4.draw_tracks(first_frame, backend.tracks, first_row.frame, backend.backend_name))

        if backend.tracks:
            track = backend.tracks[0]
            observations.append(
                StressObservation(
                    sequence=segment.sequence,
                    lorat_config=lorat_config,
                    gt_track_id=segment.track_id,
                    target_area_px=target_area_px,
                    frame=first_row.frame,
                    actual_area_px=initial_bbox[2] * initial_bbox[3],
                    iou=exercise.bbox_iou(track.bbox, initial_bbox),
                    ok=bool(track.ok),
                    confidence=track.confidence,
                    state=str(getattr(track, "state", "")),
                    elapsed_ms=init_seconds * 1000.0,
                )
            )

        for processed, row in enumerate(segment.rows[1:], start=2):
            image_index = exercise.frame_to_image_index(row.frame)
            frame, gt_bbox, actual_area = build_synthetic_frame(image_paths[image_index], row, target_area_px, args)
            gt_bbox = clip_bbox_to_frame_shape(frame, gt_bbox)
            if gt_bbox is None:
                continue
            update_started = time.perf_counter()
            backend.update(frame, row.frame)
            elapsed = time.perf_counter() - update_started
            tracking_seconds += elapsed
            track = backend.tracks[0] if backend.tracks else None
            if track is not None:
                observations.append(
                    StressObservation(
                        sequence=segment.sequence,
                        lorat_config=lorat_config,
                        gt_track_id=segment.track_id,
                        target_area_px=target_area_px,
                        frame=row.frame,
                        actual_area_px=actual_area,
                        iou=exercise.bbox_iou(track.bbox, gt_bbox),
                        ok=bool(track.ok),
                        confidence=track.confidence,
                        state=str(getattr(track, "state", "")),
                        elapsed_ms=elapsed * 1000.0,
                    )
                )
            if preview_writer is not None:
                preview_writer.write(v4.draw_tracks(frame, backend.tracks, row.frame, backend.backend_name))
            if args.progress_interval > 0 and (processed == len(segment.rows) or processed % args.progress_interval == 0):
                print(
                    f"[{segment.sequence} {lorat_config} gt={segment.track_id} area={target_area_px:g}] "
                    f"frame {processed}/{len(segment.rows)}",
                    flush=True,
                )
    finally:
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    total_seconds = time.perf_counter() - total_started
    update_frames = max(0, len(segment.rows) - 1)
    ious = [row.iou for row in observations]
    actual_areas = [row.actual_area_px for row in observations]
    confidences = [row.confidence for row in observations if row.confidence is not None]
    mean_iou = statistics.fmean(ious) if ious else None
    iou50 = sum(1 for value in ious if value >= 0.5) / len(ious) if ious else None
    lost_rate = sum(1 for row in observations if not row.ok) / len(observations) if observations else None
    reliable = None
    if len(observations) >= args.min_area_samples and mean_iou is not None and iou50 is not None and lost_rate is not None:
        reliable = (
            mean_iou >= args.reliable_mean_iou
            and iou50 >= args.reliable_iou50
            and lost_rate <= args.reliable_max_lost_rate
        )

    backbone, input_size = exercise.lorat_config_metadata(lorat_config)
    result = StressResult(
        sequence=segment.sequence,
        lorat_config=lorat_config,
        backbone=backbone,
        input_size=input_size,
        device=args.device,
        gt_track_id=segment.track_id,
        target_area_px=target_area_px,
        frames=len(segment.rows),
        update_frames=update_frames,
        total_seconds=total_seconds,
        init_seconds=init_seconds,
        tracking_seconds=tracking_seconds,
        tracking_ms_per_bbox=(tracking_seconds * 1000.0 / update_frames) if update_frames and tracking_seconds > 0 else None,
        samples=len(observations),
        mean_actual_area_px=statistics.fmean(actual_areas) if actual_areas else None,
        mean_iou=mean_iou,
        iou50=iou50,
        lost_rate=lost_rate,
        mean_confidence=statistics.fmean(confidences) if confidences else None,
        reliable=reliable,
        preview_path=preview_path if args.save_video else None,
    )
    print(
        f"{segment.sequence} {lorat_config} gt={segment.track_id} area={target_area_px:g}: "
        f"iou50={bench.optional_float(result.iou50, 3)}, "
        f"mean_iou={bench.optional_float(result.mean_iou, 3)}, "
        f"reliable={result.reliable}",
        flush=True,
    )
    return result, observations


def write_results_csv(path: Path, rows: Sequence[StressResult]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "backbone",
        "input_size",
        "device",
        "gt_track_id",
        "target_area_px",
        "frames",
        "update_frames",
        "total_seconds",
        "init_seconds",
        "tracking_seconds",
        "tracking_ms_per_bbox",
        "samples",
        "mean_actual_area_px",
        "mean_iou",
        "iou50",
        "lost_rate",
        "mean_confidence",
        "reliable",
        "preview_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "lorat_config": row.lorat_config,
                    "backbone": row.backbone,
                    "input_size": row.input_size,
                    "device": row.device,
                    "gt_track_id": row.gt_track_id,
                    "target_area_px": f"{row.target_area_px:.0f}",
                    "frames": row.frames,
                    "update_frames": row.update_frames,
                    "total_seconds": f"{row.total_seconds:.6f}",
                    "init_seconds": f"{row.init_seconds:.6f}",
                    "tracking_seconds": f"{row.tracking_seconds:.6f}",
                    "tracking_ms_per_bbox": bench.optional_float(row.tracking_ms_per_bbox),
                    "samples": row.samples,
                    "mean_actual_area_px": bench.optional_float(row.mean_actual_area_px, 2),
                    "mean_iou": bench.optional_float(row.mean_iou),
                    "iou50": bench.optional_float(row.iou50),
                    "lost_rate": bench.optional_float(row.lost_rate),
                    "mean_confidence": bench.optional_float(row.mean_confidence),
                    "reliable": "" if row.reliable is None else str(row.reliable),
                    "preview_path": str(row.preview_path or ""),
                }
            )


def write_observations_csv(path: Path, rows: Sequence[StressObservation]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "gt_track_id",
        "target_area_px",
        "frame",
        "actual_area_px",
        "iou",
        "ok",
        "confidence",
        "state",
        "elapsed_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "lorat_config": row.lorat_config,
                    "gt_track_id": row.gt_track_id,
                    "target_area_px": f"{row.target_area_px:.0f}",
                    "frame": row.frame,
                    "actual_area_px": f"{row.actual_area_px:.6f}",
                    "iou": f"{row.iou:.6f}",
                    "ok": int(row.ok),
                    "confidence": bench.optional_float(row.confidence),
                    "state": row.state,
                    "elapsed_ms": f"{row.elapsed_ms:.6f}",
                }
            )


def summarize_results(args: argparse.Namespace, results: Sequence[StressResult]) -> List[StressResult]:
    grouped: Dict[Tuple[str, str, float], List[StressResult]] = {}
    for row in results:
        grouped.setdefault((row.sequence, row.lorat_config, row.target_area_px), []).append(row)

    summaries: List[StressResult] = []
    for (sequence, config, area), rows in sorted(grouped.items(), key=lambda item: item[0]):
        first = rows[0]
        samples = sum(row.samples for row in rows)
        update_frames = sum(row.update_frames for row in rows)
        tracking_seconds = sum(row.tracking_seconds for row in rows)
        weighted_iou = sum((row.mean_iou or 0.0) * row.samples for row in rows if row.mean_iou is not None)
        weighted_iou50 = sum((row.iou50 or 0.0) * row.samples for row in rows if row.iou50 is not None)
        weighted_lost = sum((row.lost_rate or 0.0) * row.samples for row in rows if row.lost_rate is not None)
        weighted_conf = sum((row.mean_confidence or 0.0) * row.samples for row in rows if row.mean_confidence is not None)
        weighted_area = sum((row.mean_actual_area_px or 0.0) * row.samples for row in rows if row.mean_actual_area_px is not None)
        mean_iou = weighted_iou / samples if samples else None
        iou50 = weighted_iou50 / samples if samples else None
        lost_rate = weighted_lost / samples if samples else None
        mean_confidence = weighted_conf / samples if samples else None
        mean_area = weighted_area / samples if samples else None
        reliable = None
        if samples >= args.min_area_samples and mean_iou is not None and iou50 is not None and lost_rate is not None:
            reliable = (
                mean_iou >= args.reliable_mean_iou
                and iou50 >= args.reliable_iou50
                and lost_rate <= args.reliable_max_lost_rate
            )
        summaries.append(
            StressResult(
                sequence=sequence,
                lorat_config=config,
                backbone=first.backbone,
                input_size=first.input_size,
                device=first.device,
                gt_track_id=0,
                target_area_px=area,
                frames=sum(row.frames for row in rows),
                update_frames=update_frames,
                total_seconds=sum(row.total_seconds for row in rows),
                init_seconds=sum(row.init_seconds for row in rows),
                tracking_seconds=tracking_seconds,
                tracking_ms_per_bbox=(tracking_seconds * 1000.0 / update_frames) if update_frames and tracking_seconds > 0 else None,
                samples=samples,
                mean_actual_area_px=mean_area,
                mean_iou=mean_iou,
                iou50=iou50,
                lost_rate=lost_rate,
                mean_confidence=mean_confidence,
                reliable=reliable,
                preview_path=None,
            )
        )
    return summaries


def write_summary_csv(path: Path, rows: Sequence[StressResult]) -> None:
    write_results_csv(path, rows)


def smallest_reliable_area(rows: Sequence[StressResult]) -> Dict[Tuple[str, str], Optional[float]]:
    result: Dict[Tuple[str, str], Optional[float]] = {}
    grouped: Dict[Tuple[str, str], List[StressResult]] = {}
    for row in rows:
        grouped.setdefault((row.sequence, row.lorat_config), []).append(row)
    for key, group_rows in grouped.items():
        reliable_rows = [row for row in group_rows if row.reliable is True]
        result[key] = min((row.target_area_px for row in reliable_rows), default=None)
    return result


def write_summary_md(path: Path, args: argparse.Namespace, run_label: str, rows: Sequence[StressResult]) -> None:
    reliable_floor = smallest_reliable_area(rows)
    lines = [
        "# V4 LoRAT Controlled Small-Object Benchmark",
        "",
        "This benchmark forces the tracked target to fixed bounding-box pixel areas, then measures when tracking becomes unreliable.",
        "",
        f"- Run label: `{run_label}`",
        f"- Device: `{args.device}`",
        f"- Target areas: `{args.target_areas}`",
        f"- Reliable rule: IoU@0.50 >= `{args.reliable_iou50}`, mean IoU >= `{args.reliable_mean_iou}`, lost rate <= `{args.reliable_max_lost_rate}`, samples >= `{args.min_area_samples}`",
        f"- Synthetic frame background: `{args.background}`",
        "",
        "## Area Reliability",
        "",
        "| Sequence | Config | Forced Area px | Samples | Tracking ms/box | Mean IoU | IoU@0.50 | Lost Rate | Mean Confidence | Reliable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.target_area_px:.0f} | {row.samples} | "
            f"{bench.optional_float(row.tracking_ms_per_bbox, 3)} | {bench.optional_float(row.mean_iou, 3)} | "
            f"{bench.optional_float(row.iou50, 3)} | {bench.optional_float(row.lost_rate, 3)} | "
            f"{bench.optional_float(row.mean_confidence, 3)} | {'' if row.reliable is None else row.reliable} |"
        )

    lines.extend(["", "## Smallest Reliable Forced Area", ""])
    lines.append("| Sequence | Config | Smallest Reliable Area px |")
    lines.append("|---|---:|---:|")
    for (sequence, config), area in sorted(reliable_floor.items()):
        lines.append(f"| {sequence} | {config} | {'' if area is None else int(area)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flush_outputs(
    args: argparse.Namespace,
    output_paths: OutputPaths,
    run_label: str,
    results: Sequence[StressResult],
    observations: Sequence[StressObservation],
) -> List[StressResult]:
    summary_rows = summarize_results(args, results)
    write_results_csv(output_paths.results_csv, results)
    write_summary_csv(output_paths.summary_csv, summary_rows)
    if args.write_observations:
        write_observations_csv(output_paths.observations_csv, observations)
    write_summary_md(output_paths.summary_md, args, run_label, summary_rows)
    return summary_rows


def main() -> int:
    args = parse_args()
    target_areas = parse_float_list(args.target_areas)
    configs = normalized_configs(args)
    sequences = select_sequences(args)
    if args.list_sequences:
        return 0

    run_label = make_run_label(args, sequences, configs, target_areas)
    output_paths = default_output_paths(args, run_label)
    results: List[StressResult] = []
    observations: List[StressObservation] = []
    flush_outputs(args, output_paths, run_label, results, observations)

    print(f"V4 area-stress run label: {run_label}", flush=True)
    print(f"Output folder: {output_paths.run_root}", flush=True)
    print(f"Results CSV: {output_paths.results_csv}", flush=True)
    print(f"Summary CSV: {output_paths.summary_csv}", flush=True)
    if args.write_observations:
        print(f"Observations CSV: {output_paths.observations_csv}", flush=True)
    print(f"Summary Markdown: {output_paths.summary_md}", flush=True)
    print(f"Video previews: {output_paths.video_dir if args.save_video else 'disabled'}", flush=True)
    print(f"Configs: {', '.join(configs)}", flush=True)
    print(f"Forced target areas: {', '.join(f'{area:g}' for area in target_areas)}", flush=True)

    for sequence_path in sequences:
        image_paths = exercise.get_image_paths(sequence_path)
        if not image_paths:
            raise RuntimeError(f"No frames found in {sequence_path / 'img1'}")
        segments = pick_track_segments(args, sequence_path)
        print(
            f"{sequence_path.name}: selected GT tracks "
            f"{', '.join(str(segment.track_id) for segment in segments)}",
            flush=True,
        )
        for lorat_config in configs:
            for target_area_px in target_areas:
                for segment in segments:
                    preview_path = (
                        preview_video_path(output_paths, sequence_path.name, lorat_config, segment.track_id, target_area_px)
                        if args.save_video
                        else None
                    )
                    result = run_stress_case(
                        args,
                        sequence_path,
                        image_paths,
                        segment,
                        lorat_config,
                        target_area_px,
                        preview_path,
                    )
                    if result is None:
                        continue
                    case_result, case_observations = result
                    results.append(case_result)
                    observations.extend(case_observations)
                    flush_outputs(args, output_paths, run_label, results, observations)

    flush_outputs(args, output_paths, run_label, results, observations)
    print("Wrote V4 area-stress benchmark files:")
    print(f"  {output_paths.results_csv}")
    print(f"  {output_paths.summary_csv}")
    if args.write_observations:
        print(f"  {output_paths.observations_csv}")
    print(f"  {output_paths.summary_md}")
    if args.save_video:
        print(f"  video folder: {output_paths.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
