from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

import exercise_lorat_mot as exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-benchmarks" / "dancetrack"
DEFAULT_AREA_BINS = "0,256,512,1024,2048,4096,8192,16384,32768,inf"
DEFAULT_TRACK_COUNTS = "1,2,4,8"
DEFAULT_SEQUENCE = "dancetrack0065"
DEFAULT_MAX_FRAMES = 200


@dataclass(frozen=True)
class TimingResult:
    sequence: str
    lorat_config: str
    backbone: str
    input_size: int
    device: str
    checkpoint_mb: float
    target_tracks: int
    actual_tracks: int
    init_frame: int
    gt_track_ids: str
    frames: int
    update_frames: int
    boxes_total: int
    boxes_tracking: int
    total_seconds: float
    init_seconds: float
    tracking_seconds: float
    fps_total: float
    fps_tracking: Optional[float]
    total_ms_per_bbox: Optional[float]
    tracking_ms_per_bbox: Optional[float]
    mean_iou: Optional[float]
    iou50: Optional[float]
    preview_path: Optional[Path]


@dataclass(frozen=True)
class AreaObservation:
    sequence: str
    lorat_config: str
    target_tracks: int
    actual_tracks: int
    frame: int
    tracker_id: int
    gt_track_id: int
    area_px: float
    iou: float
    ok: bool
    state: str


@dataclass(frozen=True)
class AreaSummary:
    sequence: str
    lorat_config: str
    target_tracks: int
    actual_tracks: int
    area_bin: str
    min_area_px: float
    max_area_px: Optional[float]
    samples: int
    mean_area_px: Optional[float]
    mean_iou: Optional[float]
    iou50: Optional[float]
    unreliable_rate: Optional[float]
    reliable: Optional[bool]


@dataclass(frozen=True)
class OutputPaths:
    run_root: Path
    timing_csv: Path
    area_csv: Path
    observations_csv: Path
    summary_md: Path
    video_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LoRAT multi-object tracking speed by object count and "
            "tracking reliability by ground-truth object pixel area."
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
    parser.add_argument("--device", default="cpu", help="LoRAT device, e.g. cpu or cuda:0.")
    parser.add_argument("--lorat-root", type=Path, default=PROJECT_ROOT / "external" / "LoRAT-main")
    parser.add_argument("--lorat-config", default="B-224", choices=exercise.LORAT_CONFIG_CHOICES)
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=exercise.LORAT_CONFIG_CHOICES + ("all",),
        help="Run multiple LoRAT configs against the same timing and area benchmarks.",
    )
    parser.add_argument("--weight-path", type=Path, help="Optional LoRAT weight override. Cannot be used with --compare-configs.")
    parser.add_argument("--track-counts", default=DEFAULT_TRACK_COUNTS, help="Comma-separated object counts, e.g. 1,2,4,8.")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Frames per run; 0 means full sequence.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--init-frame", default="auto")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to initialize. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--min-init-tracks", type=int, default=1)
    parser.add_argument("--allow-fewer-tracks", action="store_true", help="Run even when a sequence cannot initialize the requested count.")
    parser.add_argument("--area-bins", default=DEFAULT_AREA_BINS, help="Comma-separated area bin edges. Use inf for the last edge.")
    parser.add_argument("--reliable-iou50", type=float, default=0.80, help="IoU@0.50 threshold for calling an area bin reliable.")
    parser.add_argument("--reliable-mean-iou", type=float, default=0.50, help="Mean-IoU threshold for calling an area bin reliable.")
    parser.add_argument("--min-area-samples", type=int, default=10, help="Minimum samples before an area bin is judged reliable/unreliable.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timing-csv", type=Path)
    parser.add_argument("--area-csv", type=Path)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument(
        "--write-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one row per track/frame area observation.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated MP4 previews.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print terminal progress every N processed frames. Use 0 to disable.",
    )
    parser.add_argument("--track-batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-coordinator", action="store_true")
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.30)
    parser.add_argument("--normal-proposal-weight", type=float, default=0.90)
    parser.add_argument("--occluded-proposal-weight", type=float, default=0.45)
    parser.add_argument("--suspicious-proposal-weight", type=float, default=0.25)
    parser.add_argument("--max-center-jump", type=float, default=0.45)
    parser.add_argument("--max-scale-change", type=float, default=0.65)
    parser.add_argument("--max-occlusion-frames", type=int, default=20)
    parser.add_argument("--detector", choices=("none", "hog"), default="none")
    parser.add_argument("--detector-interval", type=int, default=5)
    parser.add_argument("--max-detections", type=int, default=12)
    parser.add_argument("--association-min-score", type=float, default=0.15)
    parser.add_argument("--reid-weight", type=float, default=0.45)
    parser.add_argument("--motion-weight", type=float, default=0.30)
    parser.add_argument("--iou-weight", type=float, default=0.15)
    parser.add_argument("--source-weight", type=float, default=0.10)
    parser.add_argument("--appearance-update-rate", type=float, default=0.08)
    parser.add_argument("--appearance-bank-size", type=int, default=8)
    parser.add_argument("--appearance-commit-min-score", type=float, default=0.48)
    parser.add_argument("--appearance-commit-min-margin", type=float, default=0.04)
    parser.add_argument("--appearance-commit-min-similarity", type=float, default=0.35)
    parser.add_argument("--ambiguous-assignment-margin", type=float, default=0.04)
    parser.add_argument("--max-track-overlap-iou", type=float, default=0.70)
    parser.add_argument("--pair-memory-overlap-iou", type=float, default=0.35)
    parser.add_argument("--pair-memory-min-gap", type=float, default=6.0)
    parser.add_argument("--pair-memory-max-age", type=int, default=45)
    parser.add_argument("--pair-memory-strength", type=float, default=0.75)
    parser.add_argument("--source-switch-penalty", type=float, default=0.08)
    parser.add_argument("--reid-gate", type=float, default=0.28)
    parser.add_argument("--reid-competition-weight", type=float, default=0.20)
    parser.add_argument("--reid-competition-margin", type=float, default=0.04)
    parser.add_argument("--trajectory-history-size", type=int, default=12)
    parser.add_argument("--trajectory-weight", type=float, default=0.30)
    parser.add_argument("--trajectory-guard-min-score", type=float, default=0.25)
    parser.add_argument("--trajectory-guard-proposal-weight", type=float, default=0.50)
    parser.add_argument("--trajectory-penalty-weight", type=float, default=0.20)
    parser.add_argument("--recent-memory-frames", type=int, default=10)
    parser.add_argument("--recent-memory-max-shift-fraction", type=float, default=0.25)
    parser.add_argument("--recent-memory-min-shift-px", type=float, default=4.0)
    parser.add_argument("--recent-memory-weight", type=float, default=0.12)
    parser.add_argument("--recent-memory-competition-weight", type=float, default=0.25)
    parser.add_argument("--identity-reject-memory-score", type=float, default=0.12)
    parser.add_argument("--identity-reject-min-score", type=float, default=0.38)
    parser.add_argument("--identity-reject-motion-score", type=float, default=0.55)
    parser.add_argument("--identity-reject-competition-penalty", type=float, default=0.05)
    parser.add_argument("--guarded-max-scale-change", type=float, default=0.25)
    parser.add_argument("--direction-guard-min-score", type=float, default=0.35)
    parser.add_argument("--direction-penalty-weight", type=float, default=0.30)
    parser.add_argument("--bottom-guard-min-score", type=float, default=0.40)
    parser.add_argument("--bottom-penalty-weight", type=float, default=0.25)
    parser.add_argument("--center-anchor-max-shift-fraction", type=float, default=0.25)
    parser.add_argument("--center-anchor-min-shift-px", type=float, default=4.0)
    parser.add_argument("--extent-anchor-max-upward-fraction", type=float, default=0.12)
    parser.add_argument("--extent-anchor-min-scale", type=float, default=0.80)
    parser.add_argument("--enable-memory-recovery", dest="memory_recovery_enabled", action="store_true", default=True)
    parser.add_argument("--disable-memory-recovery", dest="memory_recovery_enabled", action="store_false")
    parser.add_argument("--memory-recovery-min-score", type=float, default=0.58)
    parser.add_argument("--memory-recovery-search-radius", type=float, default=0.90)
    parser.add_argument("--memory-recovery-scale-step", type=float, default=0.15)
    parser.add_argument("--enable-guard-resync", dest="resync_guarded_tracks", action="store_true", default=False)
    parser.add_argument("--disable-guard-resync", dest="resync_guarded_tracks", action="store_false")
    parser.add_argument("--guard-resync-min-interval", type=int, default=1)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("Track counts must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one track count is required.")
    return sorted(dict.fromkeys(values))


def parse_area_bins(text: str) -> List[float]:
    bins: List[float] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        bins.append(math.inf if part in {"inf", "infinity"} else float(part))
    if len(bins) < 2:
        raise ValueError("At least two area bin edges are required.")
    if bins[-1] != math.inf:
        bins.append(math.inf)
    for left, right in zip(bins, bins[1:]):
        if not left < right:
            raise ValueError("Area bins must be strictly increasing.")
    return bins


def area_bin_label(left: float, right: float) -> str:
    if math.isinf(right):
        return f">={int(left)}"
    return f"{int(left)}-{int(right)}"


def find_area_bin(area: float, bins: Sequence[float]) -> Optional[Tuple[float, float]]:
    for left, right in zip(bins, bins[1:]):
        if left <= area < right:
            return left, right
    return None


def optional_float(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def slugify(value: str) -> str:
    cleaned = []
    previous_was_separator = False
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            cleaned.append("_")
            previous_was_separator = True
    return "".join(cleaned).strip("_") or "benchmark"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for suffix in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create a non-overwriting output path near {path}.")


def make_run_label(
    args: argparse.Namespace,
    sequences: Sequence[Path],
    configs: Sequence[str],
    track_counts: Sequence[int],
) -> str:
    if len(sequences) == 1:
        sequence_part = sequences[0].name
    else:
        sequence_part = f"{len(sequences)}seq"
    config_part = "-".join(configs)
    count_part = "N" + "-".join(str(count) for count in track_counts)
    frames_part = f"frames{args.max_frames}" if args.max_frames > 0 else "full"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return slugify(f"{sequence_part}_{config_part}_{count_part}_{frames_part}_{stamp}")


def preview_video_path(
    output_paths: OutputPaths,
    sequence_name: str,
    lorat_config: str,
    target_tracks: int,
    max_frames: int,
) -> Path:
    frame_part = f"frames{max_frames}" if max_frames > 0 else "full"
    filename = slugify(f"{sequence_name}_{lorat_config}_N{target_tracks}_{frame_part}_preview") + ".mp4"
    return unique_path(output_paths.video_dir / filename)


def benchmark_args_for_run(args: argparse.Namespace, lorat_config: str, target_tracks: int) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.backend = "lorat"
    run_args.lorat_config = lorat_config
    run_args.max_tracks = target_tracks
    run_args.min_init_tracks = max(args.min_init_tracks, target_tracks)
    run_args.gt_init = True
    run_args.save_video = False
    run_args.interactive_playback = False
    run_args.debug_log = None
    run_args.debug_frame_start = 0
    run_args.debug_frame_end = 0
    run_args.output_by_config = False
    return run_args


def collect_area_observations(
    sequence: str,
    lorat_config: str,
    target_tracks: int,
    actual_tracks: int,
    frame_number: int,
    tracks,
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    min_visibility: float,
) -> List[AreaObservation]:
    gt_rows = {
        row.track_id: row
        for row in gt_by_frame.get(frame_number, [])
        if row.confidence != 0 and row.visibility >= min_visibility
    }
    observations: List[AreaObservation] = []
    for track in tracks:
        gt_track_id = tracker_to_gt_id.get(track.track_id)
        if gt_track_id is None:
            continue
        gt_row = gt_rows.get(gt_track_id)
        if gt_row is None:
            continue
        _, _, width, height = gt_row.bbox
        area = max(0.0, float(width) * float(height))
        iou = exercise.bbox_iou(track.bbox, gt_row.bbox)
        observations.append(
            AreaObservation(
                sequence=sequence,
                lorat_config=lorat_config,
                target_tracks=target_tracks,
                actual_tracks=actual_tracks,
                frame=frame_number,
                tracker_id=track.track_id,
                gt_track_id=gt_track_id,
                area_px=area,
                iou=iou,
                ok=bool(track.ok),
                state=track.coordinator_state,
            )
        )
    return observations


def run_benchmark_case(
    gui_v3,
    args: argparse.Namespace,
    sequence_path: Path,
    lorat_config: str,
    target_tracks: int,
    preview_path: Optional[Path],
) -> Optional[Tuple[TimingResult, List[AreaObservation]]]:
    image_paths = exercise.get_image_paths(sequence_path)
    if not image_paths:
        raise RuntimeError(f"No frames found in {sequence_path / 'img1'}")

    gt_by_frame = exercise.read_gt(sequence_path)
    fps, sequence_length = exercise.read_sequence_info(sequence_path)
    run_args = benchmark_args_for_run(args, lorat_config, target_tracks)
    init_frame, init_rows = exercise.pick_initial_rows(
        gt_by_frame,
        args.init_frame,
        args.class_id,
        args.min_visibility,
        target_tracks,
        max(args.min_init_tracks, target_tracks),
    )
    if len(init_rows) < target_tracks and not args.allow_fewer_tracks:
        print(
            f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: "
            f"only {len(init_rows)} usable GT tracks at selected init frame {init_frame}."
        )
        return None

    boxes = [row.bbox for row in init_rows]
    gt_track_ids = [row.track_id for row in init_rows]
    init_index = exercise.frame_to_image_index(init_frame)
    if init_index >= len(image_paths):
        raise RuntimeError(f"Init frame {init_frame} is outside image sequence length {len(image_paths)}.")

    init_frame_image = cv2.imread(str(image_paths[init_index]))
    if init_frame_image is None:
        raise RuntimeError(f"Unable to read frame: {image_paths[init_index]}")

    end_index = len(image_paths) - 1
    if args.max_frames > 0:
        end_index = min(end_index, init_index + args.max_frames - 1)
    total_frames_expected = max(1, end_index - init_index + 1)

    weight_path = args.weight_path or gui_v3.LORAT_WEIGHT_BY_CONFIG[lorat_config]
    checkpoint_mb = weight_path.stat().st_size / (1024 * 1024) if weight_path.exists() else 0.0
    backbone, input_size = exercise.lorat_config_metadata(lorat_config)
    observations: List[AreaObservation] = []
    metrics = {"count": 0.0, "iou_sum": 0.0, "hit50": 0.0}
    last_frame_number = init_frame
    preview_writer = None

    print(
        f"Starting {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"{total_frames_expected} frames, video={preview_path if args.save_video else 'disabled'}",
        flush=True,
    )
    backend = exercise.build_backend(gui_v3, run_args, sequence_path.name, fps, sequence_length or len(image_paths))
    total_started = time.perf_counter()
    init_started = time.perf_counter()
    try:
        backend.initialize(init_frame_image, boxes, init_frame)
        init_seconds = time.perf_counter() - init_started
        actual_tracks = len(backend.tracks)
        tracker_to_gt_id = {
            track.track_id: gt_track_id
            for track, gt_track_id in zip(backend.tracks, gt_track_ids)
            if gt_track_id is not None
        }
        observations.extend(
            collect_area_observations(
                sequence_path.name,
                lorat_config,
                target_tracks,
                actual_tracks,
                init_frame,
                backend.tracks,
                gt_by_frame,
                tracker_to_gt_id,
                args.min_visibility,
            )
        )
        exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, init_frame, tracker_to_gt_id, args.min_visibility)

        if args.save_video and preview_path is not None:
            preview_writer = gui_v3.make_video_writer(preview_path, fps, init_frame_image)
            preview_writer.write(gui_v3.draw_tracks(init_frame_image, backend.tracks, init_frame, backend.backend_name))

        print(
            f"[{sequence_path.name} {lorat_config} N={target_tracks}] "
            f"initialized {actual_tracks} tracks at source frame {init_frame}",
            flush=True,
        )
        if total_frames_expected == 1 and args.progress_interval > 0:
            elapsed = time.perf_counter() - total_started
            print(
                f"[{sequence_path.name} {lorat_config} N={target_tracks}] "
                f"frame 1/1 (source frame {init_frame}), elapsed {elapsed:.1f}s",
                flush=True,
            )

        tracking_seconds = 0.0
        for image_index in range(init_index + 1, end_index + 1):
            frame_number = image_index + 1
            frame = cv2.imread(str(image_paths[image_index]))
            if frame is None:
                print(f"Skipping unreadable frame: {image_paths[image_index]}")
                continue
            update_started = time.perf_counter()
            backend.update(frame, frame_number)
            tracking_seconds += time.perf_counter() - update_started
            last_frame_number = frame_number
            observations.extend(
                collect_area_observations(
                    sequence_path.name,
                    lorat_config,
                    target_tracks,
                    actual_tracks,
                    frame_number,
                    backend.tracks,
                    gt_by_frame,
                    tracker_to_gt_id,
                    args.min_visibility,
                )
            )
            exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, frame_number, tracker_to_gt_id, args.min_visibility)
            if preview_writer is not None:
                preview_writer.write(gui_v3.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name))
            processed_frames = image_index - init_index + 1
            if args.progress_interval > 0 and (
                processed_frames == total_frames_expected or processed_frames % args.progress_interval == 0
            ):
                elapsed = time.perf_counter() - total_started
                print(
                    f"[{sequence_path.name} {lorat_config} N={target_tracks}] "
                    f"frame {processed_frames}/{total_frames_expected} "
                    f"(source frame {frame_number}), elapsed {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    total_seconds = time.perf_counter() - total_started
    frames = max(1, last_frame_number - init_frame + 1)
    update_frames = max(0, frames - 1)
    boxes_total = frames * actual_tracks
    boxes_tracking = update_frames * actual_tracks
    fps_total = frames / total_seconds if total_seconds > 0 else 0.0
    fps_tracking = update_frames / tracking_seconds if tracking_seconds > 0 and update_frames else None
    total_ms_per_bbox = (total_seconds * 1000.0 / boxes_total) if boxes_total else None
    tracking_ms_per_bbox = (tracking_seconds * 1000.0 / boxes_tracking) if boxes_tracking else None
    mean_iou = metrics["iou_sum"] / metrics["count"] if metrics["count"] else None
    iou50 = metrics["hit50"] / metrics["count"] if metrics["count"] else None
    timing = TimingResult(
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        backbone=backbone,
        input_size=input_size,
        device=args.device,
        checkpoint_mb=checkpoint_mb,
        target_tracks=target_tracks,
        actual_tracks=actual_tracks,
        init_frame=init_frame,
        gt_track_ids=",".join(str(track_id) for track_id in gt_track_ids[:actual_tracks]),
        frames=frames,
        update_frames=update_frames,
        boxes_total=boxes_total,
        boxes_tracking=boxes_tracking,
        total_seconds=total_seconds,
        init_seconds=init_seconds,
        tracking_seconds=tracking_seconds,
        fps_total=fps_total,
        fps_tracking=fps_tracking,
        total_ms_per_bbox=total_ms_per_bbox,
        tracking_ms_per_bbox=tracking_ms_per_bbox,
        mean_iou=mean_iou,
        iou50=iou50,
        preview_path=preview_path if args.save_video else None,
    )
    print(
        f"{sequence_path.name} {lorat_config} N={target_tracks}: "
        f"actual={actual_tracks}, frames={frames}, "
        f"track_ms_per_box={optional_float(tracking_ms_per_bbox, 3)}, "
        f"iou50={optional_float(iou50, 3)}, "
        f"video={preview_path if args.save_video else 'disabled'}",
        flush=True,
    )
    return timing, observations


def summarize_area_observations(
    observations: Sequence[AreaObservation],
    area_bins: Sequence[float],
    reliable_iou50: float,
    reliable_mean_iou: float,
    min_area_samples: int,
) -> List[AreaSummary]:
    grouped: Dict[Tuple[str, str, int, int, float, float], List[AreaObservation]] = {}
    for observation in observations:
        bin_edges = find_area_bin(observation.area_px, area_bins)
        if bin_edges is None:
            continue
        left, right = bin_edges
        key = (
            observation.sequence,
            observation.lorat_config,
            observation.target_tracks,
            observation.actual_tracks,
            left,
            right,
        )
        grouped.setdefault(key, []).append(observation)

    summaries: List[AreaSummary] = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        sequence, lorat_config, target_tracks, actual_tracks, left, right = key
        ious = [row.iou for row in rows]
        areas = [row.area_px for row in rows]
        mean_iou = statistics.fmean(ious) if ious else None
        iou50 = sum(1 for score in ious if score >= 0.5) / len(ious) if ious else None
        reliable = None
        if len(rows) >= min_area_samples and mean_iou is not None and iou50 is not None:
            reliable = mean_iou >= reliable_mean_iou and iou50 >= reliable_iou50
        summaries.append(
            AreaSummary(
                sequence=sequence,
                lorat_config=lorat_config,
                target_tracks=target_tracks,
                actual_tracks=actual_tracks,
                area_bin=area_bin_label(left, right),
                min_area_px=left,
                max_area_px=None if math.isinf(right) else right,
                samples=len(rows),
                mean_area_px=statistics.fmean(areas) if areas else None,
                mean_iou=mean_iou,
                iou50=iou50,
                unreliable_rate=(1.0 - iou50) if iou50 is not None else None,
                reliable=reliable,
            )
        )
    return summaries


def default_output_paths(args: argparse.Namespace, run_label: str) -> OutputPaths:
    run_root = args.output_root.resolve() / run_label
    timing_csv = args.timing_csv.resolve() if args.timing_csv else run_root / f"{run_label}_timing_by_track_count.csv"
    area_csv = args.area_csv.resolve() if args.area_csv else run_root / f"{run_label}_area_reliability.csv"
    observations_csv = (
        args.observations_csv.resolve()
        if args.observations_csv
        else run_root / f"{run_label}_area_observations.csv"
    )
    summary_md = args.summary_md.resolve() if args.summary_md else run_root / f"{run_label}_summary.md"
    video_dir = run_root / "videos"
    timing_csv = unique_path(timing_csv)
    area_csv = unique_path(area_csv)
    observations_csv = unique_path(observations_csv)
    summary_md = unique_path(summary_md)
    for path in (timing_csv, area_csv, observations_csv, summary_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        run_root=run_root,
        timing_csv=timing_csv,
        area_csv=area_csv,
        observations_csv=observations_csv,
        summary_md=summary_md,
        video_dir=video_dir,
    )


def write_timing_csv(path: Path, rows: Sequence[TimingResult]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "backbone",
        "input_size",
        "device",
        "checkpoint_mb",
        "target_tracks",
        "actual_tracks",
        "init_frame",
        "gt_track_ids",
        "frames",
        "update_frames",
        "boxes_total",
        "boxes_tracking",
        "total_seconds",
        "init_seconds",
        "tracking_seconds",
        "fps_total",
        "fps_tracking",
        "total_ms_per_bbox",
        "tracking_ms_per_bbox",
        "mean_iou",
        "iou50",
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
                    "checkpoint_mb": f"{row.checkpoint_mb:.2f}",
                    "target_tracks": row.target_tracks,
                    "actual_tracks": row.actual_tracks,
                    "init_frame": row.init_frame,
                    "gt_track_ids": row.gt_track_ids,
                    "frames": row.frames,
                    "update_frames": row.update_frames,
                    "boxes_total": row.boxes_total,
                    "boxes_tracking": row.boxes_tracking,
                    "total_seconds": f"{row.total_seconds:.6f}",
                    "init_seconds": f"{row.init_seconds:.6f}",
                    "tracking_seconds": f"{row.tracking_seconds:.6f}",
                    "fps_total": f"{row.fps_total:.6f}",
                    "fps_tracking": optional_float(row.fps_tracking),
                    "total_ms_per_bbox": optional_float(row.total_ms_per_bbox),
                    "tracking_ms_per_bbox": optional_float(row.tracking_ms_per_bbox),
                    "mean_iou": optional_float(row.mean_iou),
                    "iou50": optional_float(row.iou50),
                    "preview_path": str(row.preview_path or ""),
                }
            )


def write_area_csv(path: Path, rows: Sequence[AreaSummary]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "target_tracks",
        "actual_tracks",
        "area_bin",
        "min_area_px",
        "max_area_px",
        "samples",
        "mean_area_px",
        "mean_iou",
        "iou50",
        "unreliable_rate",
        "reliable",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "lorat_config": row.lorat_config,
                    "target_tracks": row.target_tracks,
                    "actual_tracks": row.actual_tracks,
                    "area_bin": row.area_bin,
                    "min_area_px": f"{row.min_area_px:.0f}",
                    "max_area_px": "" if row.max_area_px is None else f"{row.max_area_px:.0f}",
                    "samples": row.samples,
                    "mean_area_px": optional_float(row.mean_area_px, 2),
                    "mean_iou": optional_float(row.mean_iou),
                    "iou50": optional_float(row.iou50),
                    "unreliable_rate": optional_float(row.unreliable_rate),
                    "reliable": "" if row.reliable is None else str(row.reliable),
                }
            )


def write_observations_csv(path: Path, rows: Sequence[AreaObservation]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "target_tracks",
        "actual_tracks",
        "frame",
        "tracker_id",
        "gt_track_id",
        "area_px",
        "iou",
        "ok",
        "state",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "lorat_config": row.lorat_config,
                    "target_tracks": row.target_tracks,
                    "actual_tracks": row.actual_tracks,
                    "frame": row.frame,
                    "tracker_id": row.tracker_id,
                    "gt_track_id": row.gt_track_id,
                    "area_px": f"{row.area_px:.6f}",
                    "iou": f"{row.iou:.6f}",
                    "ok": int(row.ok),
                    "state": row.state,
                }
            )


def smallest_reliable_area(rows: Sequence[AreaSummary]) -> Dict[Tuple[str, str, int], Optional[float]]:
    result: Dict[Tuple[str, str, int], Optional[float]] = {}
    grouped: Dict[Tuple[str, str, int], List[AreaSummary]] = {}
    for row in rows:
        grouped.setdefault((row.sequence, row.lorat_config, row.target_tracks), []).append(row)
    for key, group_rows in grouped.items():
        reliable_rows = [row for row in group_rows if row.reliable is True]
        result[key] = min((row.min_area_px for row in reliable_rows), default=None)
    return result


def write_summary_md(
    path: Path,
    args: argparse.Namespace,
    run_label: str,
    timing_rows: Sequence[TimingResult],
    area_rows: Sequence[AreaSummary],
) -> None:
    lines = [
        "# LoRAT Multi-Object Benchmarks",
        "",
        "This file summarizes Week 1 benchmark items (b) and (c): timing by number of tracked objects and reliability by ground-truth pixel area.",
        "",
        f"- Run label: `{run_label}`",
        f"- Device: `{args.device}`",
        f"- Track counts: `{args.track_counts}`",
        f"- Max frames per run: `{args.max_frames}`",
        f"- Area bins: `{args.area_bins}`",
        f"- Reliable bin rule: IoU@0.50 >= `{args.reliable_iou50}` and mean IoU >= `{args.reliable_mean_iou}` with at least `{args.min_area_samples}` samples",
        "",
        "## Timing By Object Count",
        "",
        "| Sequence | Config | Target N | Actual N | Frames | Tracking Seconds | Tracking ms/box | Total ms/box | FPS tracking | Mean IoU | IoU@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.target_tracks} | {row.actual_tracks} | "
            f"{row.frames} | {row.tracking_seconds:.3f} | "
            f"{optional_float(row.tracking_ms_per_bbox, 3)} | {optional_float(row.total_ms_per_bbox, 3)} | "
            f"{optional_float(row.fps_tracking, 3)} | {optional_float(row.mean_iou, 3)} | {optional_float(row.iou50, 3)} |"
        )

    preview_rows = [row for row in timing_rows if row.preview_path is not None]
    if preview_rows:
        lines.extend(
            [
                "",
                "## Preview Videos",
                "",
                "| Sequence | Config | Target N | Preview Path |",
                "|---|---:|---:|---|",
            ]
        )
        for row in preview_rows:
            lines.append(f"| {row.sequence} | {row.lorat_config} | {row.target_tracks} | `{row.preview_path}` |")

    reliable_floor = smallest_reliable_area(area_rows)
    lines.extend(
        [
            "",
            "## Small-Object Reliability",
            "",
            "| Sequence | Config | Target N | Area Bin px | Samples | Mean Area | Mean IoU | IoU@0.50 | Unreliable Rate | Reliable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in area_rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.target_tracks} | {row.area_bin} | "
            f"{row.samples} | {optional_float(row.mean_area_px, 1)} | {optional_float(row.mean_iou, 3)} | "
            f"{optional_float(row.iou50, 3)} | {optional_float(row.unreliable_rate, 3)} | "
            f"{'' if row.reliable is None else row.reliable} |"
        )

    lines.extend(["", "## Smallest Reliable Area", ""])
    lines.append("| Sequence | Config | Target N | Smallest Reliable Area px |")
    lines.append("|---|---:|---:|---:|")
    for key, value in sorted(reliable_floor.items()):
        sequence, config, target_tracks = key
        lines.append(f"| {sequence} | {config} | {target_tracks} | {'' if value is None else int(value)} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def main() -> int:
    args = parse_args()
    track_counts = parse_int_list(args.track_counts)
    area_bins = parse_area_bins(args.area_bins)
    sequences = select_sequences(args)
    if args.list_sequences:
        return 0

    configs = normalized_configs(args)
    run_label = make_run_label(args, sequences, configs, track_counts)
    output_paths = default_output_paths(args, run_label)
    gui_v3 = exercise.load_gui_v3_module()
    timing_rows: List[TimingResult] = []
    observations: List[AreaObservation] = []

    print(f"Benchmark run label: {run_label}", flush=True)
    print(f"Output folder: {output_paths.run_root}", flush=True)
    print(f"CSV timing output: {output_paths.timing_csv}", flush=True)
    print(f"CSV area output: {output_paths.area_csv}", flush=True)
    if args.write_observations:
        print(f"CSV observations output: {output_paths.observations_csv}", flush=True)
    print(f"Markdown summary output: {output_paths.summary_md}", flush=True)
    print(f"Video previews: {output_paths.video_dir if args.save_video else 'disabled'}", flush=True)
    print(f"Sequences: {', '.join(sequence.name for sequence in sequences)}", flush=True)
    print(f"Configs: {', '.join(configs)}", flush=True)
    print(f"Track counts: {', '.join(str(count) for count in track_counts)}", flush=True)
    print(f"Frames per run: {args.max_frames if args.max_frames > 0 else 'full sequence'}", flush=True)
    progress_text = f"every {args.progress_interval} frame(s)" if args.progress_interval > 0 else "disabled"
    print(f"Progress interval: {progress_text}", flush=True)

    for lorat_config in configs:
        for sequence_path in sequences:
            for target_tracks in track_counts:
                preview_path = (
                    preview_video_path(output_paths, sequence_path.name, lorat_config, target_tracks, args.max_frames)
                    if args.save_video
                    else None
                )
                result = run_benchmark_case(gui_v3, args, sequence_path, lorat_config, target_tracks, preview_path)
                if result is None:
                    continue
                timing, case_observations = result
                timing_rows.append(timing)
                observations.extend(case_observations)

    area_rows = summarize_area_observations(
        observations,
        area_bins,
        args.reliable_iou50,
        args.reliable_mean_iou,
        args.min_area_samples,
    )
    write_timing_csv(output_paths.timing_csv, timing_rows)
    write_area_csv(output_paths.area_csv, area_rows)
    if args.write_observations:
        write_observations_csv(output_paths.observations_csv, observations)
    write_summary_md(output_paths.summary_md, args, run_label, timing_rows, area_rows)

    print("Wrote benchmark files:")
    print(f"  {output_paths.timing_csv}")
    print(f"  {output_paths.area_csv}")
    if args.write_observations:
        print(f"  {output_paths.observations_csv}")
    if args.save_video:
        print(f"  video folder: {output_paths.video_dir}")
    print(f"  {output_paths.summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
