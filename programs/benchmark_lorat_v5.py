from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

import benchmark_lorat_mot as bench
import bounding_box_v4_lorat_memory as v4_baseline
import bounding_box_v5_lorat_shared as v5
import bounding_box_v6_lorat_gated as v6
import exercise_lorat_mot as exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-benchmarks" / "v6-dancetrack"
DEFAULT_SEQUENCE = "dancetrack0065"
DEFAULT_MAX_FRAMES = 200
BEFORE_LABEL = "v4-serial-baseline"
AFTER_LABEL = "v5-shared"
V6_LABEL = "v6-gated-sot-memory"
STOP_REQUESTED = False
DEBUG_CSV_FIELDS = [
    "timestamp",
    "event",
    "sequence",
    "lorat_config",
    "execution_mode",
    "target_tracks",
    "actual_tracks",
    "init_frame",
    "usable_tracks",
    "selected_gt_track_ids",
    "device",
    "gpu_name",
    "track_batch_size",
    "lorat_memory_slots",
    "lorat_active_slots_per_track",
    "v6_primary_slots_per_track",
    "v6_recovery_slots_per_track",
    "v6_recovery_interval",
    "v6_recovery_min_confidence",
    "v6_recovery_min_assignment_score",
    "v6_recovery_min_assignment_margin",
    "v6_recovery_stale_slot_frames",
    "frames_expected",
    "frames_completed",
    "update_frames",
    "boxes_tracking",
    "total_seconds",
    "init_seconds",
    "tracking_seconds",
    "total_ms_per_bbox",
    "tracking_ms_per_bbox",
    "fps_total",
    "fps_tracking",
    "evaluator_calls",
    "evaluator_tasks",
    "model_forward_calls",
    "model_forward_items",
    "max_model_forward_batch",
    "fusion_forward_calls",
    "fusion_forward_items",
    "max_fusion_forward_batch",
    "evaluator_tasks_per_update_frame",
    "model_forward_items_per_update_frame",
    "model_forward_items_per_bbox",
    "gating_decisions",
    "gating_primary_decisions",
    "gating_recovery_decisions",
    "gating_selected_slot_items",
    "gating_avg_slots_per_decision",
    "gating_recovery_reasons",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "gpu_memory_peak_allocated_mb",
    "gpu_memory_peak_reserved_mb",
    "max_evaluator_batch",
    "mean_iou",
    "iou50",
    "fps_sustains_threshold",
    "weight_path",
    "preview_path",
    "reason",
]


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; finishing the current benchmark case and flushing outputs.", flush=True)


def write_debug_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DEBUG_CSV_FIELDS})


def record_debug_event(
    rows: List[Dict[str, object]],
    path: Path,
    event: str,
    **values: object,
) -> None:
    row = {field: "" for field in DEBUG_CSV_FIELDS}
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    row["event"] = event
    for key, value in values.items():
        if key in row:
            row[key] = value
    rows.append(row)
    write_debug_csv(path, rows)


def tracker_debug_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "track_batch_size": args.track_batch_size,
        "lorat_memory_slots": args.lorat_memory_slots,
        "lorat_active_slots_per_track": args.lorat_active_slots_per_track,
        "v6_primary_slots_per_track": args.v6_primary_slots_per_track,
        "v6_recovery_slots_per_track": args.v6_recovery_slots_per_track,
        "v6_recovery_interval": args.v6_recovery_interval,
        "v6_recovery_min_confidence": args.v6_recovery_min_confidence,
        "v6_recovery_min_assignment_score": args.v6_recovery_min_assignment_score,
        "v6_recovery_min_assignment_margin": args.v6_recovery_min_assignment_margin,
        "v6_recovery_stale_slot_frames": args.v6_recovery_stale_slot_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LoRAT MOT versions: V4 serial baseline, V5 shared slots, and V6 gated SOT memory."
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
    parser.add_argument(
        "--gpu-profile",
        default="local",
        help="Label for the GPU being benchmarked, e.g. lab or hpc. Written to CSV and summary tables.",
    )
    parser.add_argument("--lorat-root", type=Path, default=v5.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v5.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(v5.LORAT_WEIGHT_BY_CONFIG) + ("all",),
        help="Run multiple LoRAT model sizes/configs against the same benchmarks.",
    )
    parser.add_argument("--weight-path", type=Path, help="Optional LoRAT weight override. Cannot be used with --compare-configs.")
    parser.add_argument(
        "--skip-missing-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip configs whose weight file is absent instead of failing fast.",
    )

    parser.add_argument("--track-counts", default=bench.DEFAULT_TRACK_COUNTS, help="Comma-separated object counts, e.g. 1,2,4,8.")
    parser.add_argument(
        "--max-track-count",
        type=int,
        default=0,
        help="If set, benchmark every object count from 1 through this N. Overrides --track-counts.",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Frames per run; 0 means full sequence.")
    parser.add_argument(
        "--benchmark-versions",
        default="v4,v5,v6",
        help=(
            "Comma-separated versions to benchmark. "
            "v4 is the before-refactor serial baseline; v5 is shared-backbone; "
            "v6 gates LoRAT memory slots from cached MOT state."
        ),
    )
    parser.add_argument(
        "--fps-threshold",
        type=float,
        default=25.0,
        help="FPS threshold used to report the maximum sustainable object count.",
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--init-frame", default="auto")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to initialize. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--min-init-tracks", type=int, default=1)
    parser.add_argument("--allow-fewer-tracks", action="store_true", help="Run even when a sequence cannot initialize the requested count.")

    parser.add_argument("--area-bins", default=bench.DEFAULT_AREA_BINS, help="Comma-separated area bin edges. Use inf for the last edge.")
    parser.add_argument("--reliable-iou50", type=float, default=0.80, help="IoU@0.50 threshold for calling an area bin reliable.")
    parser.add_argument("--reliable-mean-iou", type=float, default=0.50, help="Mean-IoU threshold for calling an area bin reliable.")
    parser.add_argument("--min-area-samples", type=int, default=10, help="Minimum samples before an area bin is judged reliable/unreliable.")
    parser.add_argument("--identity-sample-interval", type=int, default=10, help="Check object identity every N frames; 0 disables sampled identity flags.")
    parser.add_argument("--identity-correct-iou", type=float, default=0.30, help="Minimum IoU with the initialized GT object for a sampled frame to count as correct.")
    parser.add_argument("--identity-competitor-margin", type=float, default=0.05, help="Correctness margin over the best other visible GT object.")
    parser.add_argument("--identity-jump-factor", type=float, default=2.0, help="Sampled center movement larger than this times the GT diagonal is marked as a jump.")

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timing-csv", type=Path)
    parser.add_argument("--area-csv", type=Path)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--debug-csv", type=Path, help="Optional path for the case lifecycle debug CSV.")
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
    parser.add_argument("--progress-interval", type=int, default=10, help="Print progress every N processed frames. Use 0 to disable.")

    add_v5_runtime_args(parser)
    v6.add_v6_runtime_args(parser)
    return parser.parse_args()


def add_v5_runtime_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("V5 tracker runtime")
    group.add_argument("--track-batch-size", type=int, default=8)
    group.add_argument("--disable-amp", action="store_true")
    group.add_argument("--lorat-memory-slots", type=int, default=v5.DEFAULT_LORAT_MEMORY_SLOTS)
    group.add_argument("--lorat-memory-refresh-interval", type=int, default=v5.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL)
    group.add_argument("--lorat-active-slots-per-track", type=int, default=v5.DEFAULT_LORAT_ACTIVE_SLOTS_PER_TRACK)
    group.add_argument("--fixed-lorat-box-size", dest="lorat_fixed_box_size", action="store_true", default=v5.DEFAULT_LORAT_FIXED_BOX_SIZE)
    group.add_argument("--allow-lorat-size-change", dest="lorat_fixed_box_size", action="store_false")
    group.add_argument("--lorat-min-box-area", type=float, default=v5.DEFAULT_LORAT_MIN_BOX_AREA)
    group.add_argument("--lorat-max-area-change-per-frame", type=float, default=v5.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME)
    group.add_argument("--lorat-trusted-size-floor-scale", type=float, default=v5.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE)
    group.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    group.add_argument("--lorat-accept-min-score", type=float, default=v5.DEFAULT_LORAT_ACCEPT_MIN_SCORE)
    group.add_argument("--shrink-guard-window", type=int, default=v5.DEFAULT_SHRINK_GUARD_WINDOW)
    group.add_argument("--shrink-guard-area-ratio", type=float, default=v5.DEFAULT_SHRINK_GUARD_AREA_RATIO)
    group.add_argument("--shrink-guard-step-ratio", type=float, default=v5.DEFAULT_SHRINK_GUARD_STEP_RATIO)
    group.add_argument("--shrink-guard-min-confidence", type=float, default=v5.DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE)
    group.add_argument("--shrink-guard-min-reid", type=float, default=v5.DEFAULT_SHRINK_GUARD_MIN_REID)
    group.add_argument("--crop-information-min-score", type=float, default=v5.DEFAULT_CROP_INFORMATION_MIN_SCORE)
    group.add_argument("--crop-information-min-pixels", type=int, default=v5.DEFAULT_CROP_INFORMATION_MIN_PIXELS)
    group.add_argument("--lorat-slot-capacity", type=int, default=0)
    group.add_argument("--lorat-search-area-factor", type=float, default=v5.DEFAULT_LORAT_SEARCH_AREA_FACTOR)
    group.add_argument("--lorat-window-penalty", type=float, default=v5.DEFAULT_LORAT_WINDOW_PENALTY)
    group.add_argument("--lorat-state-update-min-score", type=float, default=v5.DEFAULT_LORAT_STATE_UPDATE_MIN_SCORE)
    group.add_argument("--lorat-state-update-max-center-shift", type=float, default=v5.DEFAULT_LORAT_STATE_UPDATE_MAX_CENTER_SHIFT)
    group.add_argument("--lorat-state-update-max-area-change", type=float, default=v5.DEFAULT_LORAT_STATE_UPDATE_MAX_AREA_CHANGE)
    group.add_argument("--disable-identity-arbitration", action="store_true")
    group.add_argument("--identity-min-score", type=float, default=v5.DEFAULT_IDENTITY_MIN_SCORE)
    group.add_argument("--identity-min-reid", type=float, default=v5.DEFAULT_IDENTITY_MIN_REID)
    group.add_argument("--identity-min-motion", type=float, default=v5.DEFAULT_IDENTITY_MIN_MOTION)
    group.add_argument("--identity-min-path", type=float, default=v5.DEFAULT_IDENTITY_MIN_PATH)
    group.add_argument("--identity-bank-size", type=int, default=12)
    group.add_argument("--identity-memory-min-confidence", type=float, default=v5.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE)
    group.add_argument("--occlusion-max-frames", type=int, default=v5.DEFAULT_OCCLUSION_MAX_FRAMES)
    group.add_argument("--occlusion-iou-threshold", type=float, default=v5.DEFAULT_OCCLUSION_IOU_THRESHOLD)
    group.add_argument("--occlusion-velocity-damping", type=float, default=v5.DEFAULT_OCCLUSION_VELOCITY_DAMPING)
    group.add_argument("--reid-recovery-min-score", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_SCORE)
    group.add_argument("--reid-recovery-min-reid", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_REID)
    group.add_argument("--reid-recovery-min-motion", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_MOTION)
    group.add_argument("--reid-recovery-min-confidence", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE)
    group.add_argument("--view-change-min-score", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_SCORE)
    group.add_argument("--view-change-min-motion", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_MOTION)
    group.add_argument("--view-change-min-confidence", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE)
    group.add_argument("--view-change-max-lost-frames", type=int, default=v5.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES)


def parse_track_counts(args: argparse.Namespace) -> List[int]:
    if args.max_track_count > 0:
        return list(range(1, args.max_track_count + 1))
    return bench.parse_int_list(args.track_counts)


def parse_benchmark_versions(args: argparse.Namespace) -> List[str]:
    versions = []
    for version in (part.strip().lower() for part in args.benchmark_versions.split(",")):
        if not version:
            continue
        if version not in {"v4", "v5", "v6"}:
            raise RuntimeError(f"Unsupported benchmark version {version!r}; choose v4, v5, v6, or a comma-separated combination.")
        if version not in versions:
            versions.append(version)
    if not versions:
        raise RuntimeError("--benchmark-versions must contain at least one version.")
    return versions


def execution_label_for_version(version: str) -> str:
    labels = {
        "v4": BEFORE_LABEL,
        "v5": AFTER_LABEL,
        "v6": V6_LABEL,
    }
    return labels[version]


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


def make_run_label(
    args: argparse.Namespace,
    sequences: Sequence[Path],
    configs: Sequence[str],
    track_counts: Sequence[int],
    benchmark_versions: Sequence[str],
) -> str:
    version_part = "versions-" + "-".join(benchmark_versions)
    prefix = "v6_" if "v6" in benchmark_versions else "v5_"
    return prefix + bench.make_run_label(args, sequences, configs, track_counts) + "_" + version_part


def tracker_args_for_run(
    args: argparse.Namespace,
    lorat_config: str,
    target_tracks: int,
    version: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        lorat_root=args.lorat_root,
        lorat_config=lorat_config,
        weight_path=args.weight_path,
        device=args.device,
        max_tracks=target_tracks,
        track_batch_size=1 if version == "v4" else args.track_batch_size,
        disable_amp=args.disable_amp,
        lorat_memory_slots=args.lorat_memory_slots,
        lorat_memory_refresh_interval=args.lorat_memory_refresh_interval,
        lorat_active_slots_per_track=args.lorat_active_slots_per_track,
        lorat_fixed_box_size=args.lorat_fixed_box_size,
        lorat_min_box_area=args.lorat_min_box_area,
        lorat_max_area_change_per_frame=args.lorat_max_area_change_per_frame,
        lorat_trusted_size_floor_scale=args.lorat_trusted_size_floor_scale,
        lorat_memory_min_score=args.lorat_memory_min_score,
        lorat_accept_min_score=args.lorat_accept_min_score,
        shrink_guard_window=args.shrink_guard_window,
        shrink_guard_area_ratio=args.shrink_guard_area_ratio,
        shrink_guard_step_ratio=args.shrink_guard_step_ratio,
        shrink_guard_min_confidence=args.shrink_guard_min_confidence,
        shrink_guard_min_reid=args.shrink_guard_min_reid,
        crop_information_min_score=args.crop_information_min_score,
        crop_information_min_pixels=args.crop_information_min_pixels,
        lorat_slot_capacity=args.lorat_slot_capacity,
        lorat_search_area_factor=args.lorat_search_area_factor,
        lorat_window_penalty=args.lorat_window_penalty,
        lorat_state_update_min_score=args.lorat_state_update_min_score,
        lorat_state_update_max_center_shift=args.lorat_state_update_max_center_shift,
        lorat_state_update_max_area_change=args.lorat_state_update_max_area_change,
        disable_identity_arbitration=args.disable_identity_arbitration,
        identity_min_score=args.identity_min_score,
        identity_min_reid=args.identity_min_reid,
        identity_min_motion=args.identity_min_motion,
        identity_min_path=args.identity_min_path,
        identity_bank_size=args.identity_bank_size,
        identity_memory_min_confidence=args.identity_memory_min_confidence,
        occlusion_max_frames=args.occlusion_max_frames,
        occlusion_iou_threshold=args.occlusion_iou_threshold,
        occlusion_velocity_damping=args.occlusion_velocity_damping,
        reid_recovery_min_score=args.reid_recovery_min_score,
        reid_recovery_min_reid=args.reid_recovery_min_reid,
        reid_recovery_min_motion=args.reid_recovery_min_motion,
        reid_recovery_min_confidence=args.reid_recovery_min_confidence,
        view_change_min_score=args.view_change_min_score,
        view_change_min_motion=args.view_change_min_motion,
        view_change_min_confidence=args.view_change_min_confidence,
        view_change_max_lost_frames=args.view_change_max_lost_frames,
        v6_primary_slots_per_track=args.v6_primary_slots_per_track,
        v6_recovery_slots_per_track=args.v6_recovery_slots_per_track,
        v6_recovery_interval=args.v6_recovery_interval,
        v6_recovery_min_confidence=args.v6_recovery_min_confidence,
        v6_recovery_min_assignment_score=args.v6_recovery_min_assignment_score,
        v6_recovery_min_assignment_margin=args.v6_recovery_min_assignment_margin,
        v6_recovery_stale_slot_frames=args.v6_recovery_stale_slot_frames,
    )


def build_backend(
    args: argparse.Namespace,
    lorat_config: str,
    target_tracks: int,
    version: str,
    sequence_name: str,
    fps: float,
    length: Optional[int],
):
    run_args = tracker_args_for_run(args, lorat_config, target_tracks, version)
    source = SimpleNamespace(fps=fps, length=length, name=sequence_name)
    module_by_version = {
        "v4": v4_baseline,
        "v5": v5,
        "v6": v6,
    }
    module = module_by_version[version]
    return module.create_backend(run_args, source, expected_tracks=target_tracks)


def collect_area_observations(
    sequence: str,
    lorat_config: str,
    execution_mode: str,
    target_tracks: int,
    actual_tracks: int,
    init_frame: int,
    frame_number: int,
    tracks,
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    min_visibility: float,
    identity_sample_interval: int,
    identity_correct_iou: float,
    identity_competitor_margin: float,
    identity_jump_factor: float,
    previous_sample_bboxes: Dict[int, v5.BBox],
) -> List[bench.AreaObservation]:
    visible_gt_rows = [
        row for row in gt_by_frame.get(frame_number, []) if row.confidence != 0 and row.visibility >= min_visibility
    ]
    gt_rows = {row.track_id: row for row in visible_gt_rows}
    observations: List[bench.AreaObservation] = []
    sampled = identity_sample_interval > 0 and (frame_number - init_frame) % identity_sample_interval == 0
    for track in tracks:
        gt_track_id = tracker_to_gt_id.get(track.track_id)
        if gt_track_id is None:
            continue
        gt_row = gt_rows.get(gt_track_id)
        if gt_row is None:
            continue
        _, _, width, height = gt_row.bbox
        area = max(0.0, float(width) * float(height))
        own_iou = exercise.bbox_iou(track.bbox, gt_row.bbox)
        best_other_iou = max(
            (exercise.bbox_iou(track.bbox, other.bbox) for other in visible_gt_rows if other.track_id != gt_track_id),
            default=0.0,
        )
        previous_sample_bbox = previous_sample_bboxes.get(track.track_id)
        center_jump_px = None
        identity_jump = None
        correct_object = None
        state = str(getattr(track, "state", ""))
        occluded = any(token in state.upper() for token in ("OCCLU", "LOST", "MISS", "LOWCONF", "ID_UNCERTAIN"))
        occluded = occluded or not bool(getattr(track, "ok", False))
        if sampled:
            correct_object = (
                bool(getattr(track, "ok", False))
                and own_iou >= identity_correct_iou
                and own_iou + identity_competitor_margin >= best_other_iou
            )
            if previous_sample_bbox is not None:
                center_jump_px = v5.center_distance(previous_sample_bbox, track.bbox)
                jump_threshold = identity_jump_factor * max(1.0, v5.bbox_diagonal(gt_row.bbox))
                identity_jump = center_jump_px > jump_threshold
            else:
                center_jump_px = 0.0
                identity_jump = False
            previous_sample_bboxes[track.track_id] = track.bbox
        observations.append(
            bench.AreaObservation(
                sequence=sequence,
                lorat_config=lorat_config,
                execution_mode=execution_mode,
                target_tracks=target_tracks,
                actual_tracks=actual_tracks,
                frame=frame_number,
                tracker_id=track.track_id,
                gt_track_id=gt_track_id,
                area_px=area,
                iou=own_iou,
                ok=bool(getattr(track, "ok", False)),
                state=state,
                sampled=sampled,
                correct_object=correct_object,
                identity_jump=identity_jump,
                occluded=occluded if sampled else None,
                center_jump_px=center_jump_px,
            )
        )
    return observations


def snapshot_runtime_status(backend, version: str) -> v5.RuntimeStatus:
    if hasattr(backend, "runtime_status_snapshot"):
        return backend.runtime_status_snapshot()

    status = v5.RuntimeStatus(
        active_objects=sum(1 for track in getattr(backend, "tracks", []) if getattr(track, "ok", False)),
        max_evaluator_batch=1 if version == "v4" else 0,
    )
    status.gpu_name = getattr(backend, "device_label", "")
    torch_module = getattr(backend, "torch", None)
    device = getattr(backend, "device", None)
    if torch_module is not None and device is not None and getattr(device, "type", None) == "cuda":
        status.gpu_name = torch_module.cuda.get_device_name(device)
        status.gpu_allocated_mb = v5.bytes_to_mb(torch_module.cuda.memory_allocated(device))
        status.gpu_reserved_mb = v5.bytes_to_mb(torch_module.cuda.memory_reserved(device))
        status.gpu_peak_allocated_mb = v5.bytes_to_mb(torch_module.cuda.max_memory_allocated(device))
        status.gpu_peak_reserved_mb = v5.bytes_to_mb(torch_module.cuda.max_memory_reserved(device))
    return status


def draw_backend_tracks(frame, backend, frame_number: int, version: str):
    if version in {"v5", "v6"} and hasattr(backend, "status_lines"):
        return v5.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines())
    return v4_baseline.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name)


def run_benchmark_case(
    args: argparse.Namespace,
    sequence_path: Path,
    lorat_config: str,
    version: str,
    target_tracks: int,
    preview_path: Optional[Path],
    debug_rows: List[Dict[str, object]],
    debug_csv: Path,
) -> Optional[Tuple[bench.TimingResult, List[bench.AreaObservation]]]:
    execution_mode = execution_label_for_version(version)
    image_paths = exercise.get_image_paths(sequence_path)
    if not image_paths:
        raise RuntimeError(f"No frames found in {sequence_path / 'img1'}")

    gt_by_frame = exercise.read_gt(sequence_path)
    fps, sequence_length = exercise.read_sequence_info(sequence_path)
    init_frame, init_rows = exercise.pick_initial_rows(
        gt_by_frame,
        args.init_frame,
        args.class_id,
        args.min_visibility,
        target_tracks,
        max(args.min_init_tracks, target_tracks),
    )
    if len(init_rows) < target_tracks and not args.allow_fewer_tracks:
        reason = f"only {len(init_rows)} usable GT tracks at selected init frame {init_frame}"
        print(f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: {reason}.")
        record_debug_event(
            debug_rows,
            debug_csv,
            "skip_insufficient_tracks",
            sequence=sequence_path.name,
            lorat_config=lorat_config,
            execution_mode=execution_mode,
            target_tracks=target_tracks,
            actual_tracks=len(init_rows),
            init_frame=init_frame,
            usable_tracks=len(init_rows),
            selected_gt_track_ids=",".join(str(row.track_id) for row in init_rows),
            device=args.device,
            **tracker_debug_config(args),
            reason=reason,
        )
        return None

    weight_path = args.weight_path or v5.LORAT_WEIGHT_BY_CONFIG[lorat_config]
    if not weight_path.exists():
        message = f"Missing LoRAT weight for {lorat_config}: {weight_path}"
        if args.skip_missing_weights:
            print(f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: {message}")
            record_debug_event(
                debug_rows,
                debug_csv,
                "skip_missing_weight",
                sequence=sequence_path.name,
                lorat_config=lorat_config,
                execution_mode=execution_mode,
                target_tracks=target_tracks,
                actual_tracks=len(init_rows),
                init_frame=init_frame,
                usable_tracks=len(init_rows),
                selected_gt_track_ids=",".join(str(row.track_id) for row in init_rows),
                device=args.device,
                **tracker_debug_config(args),
                weight_path=weight_path,
                reason=message,
            )
            return None
        raise FileNotFoundError(message)

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

    checkpoint_mb = weight_path.stat().st_size / (1024 * 1024)
    backbone, input_size = exercise.lorat_config_metadata(lorat_config)
    observations: List[bench.AreaObservation] = []
    metrics = {"count": 0.0, "iou_sum": 0.0, "hit50": 0.0}
    last_frame_number = init_frame
    preview_writer = None
    previous_sample_bboxes: Dict[int, v5.BBox] = {}
    runtime_status = v5.RuntimeStatus()

    print(
        f"Starting {execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"{total_frames_expected} frames, video={preview_path if args.save_video else 'disabled'}",
        flush=True,
    )
    record_debug_event(
        debug_rows,
        debug_csv,
        "case_start",
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        execution_mode=execution_mode,
        target_tracks=target_tracks,
        actual_tracks=len(init_rows),
        init_frame=init_frame,
        usable_tracks=len(init_rows),
        selected_gt_track_ids=",".join(str(track_id) for track_id in gt_track_ids),
        device=args.device,
        **tracker_debug_config(args),
        frames_expected=total_frames_expected,
        weight_path=weight_path,
        preview_path=preview_path if args.save_video else "",
    )
    backend = build_backend(
        args,
        lorat_config,
        target_tracks,
        version,
        sequence_path.name,
        fps,
        sequence_length or len(image_paths),
    )
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
                execution_mode,
                target_tracks,
                actual_tracks,
                init_frame,
                init_frame,
                backend.tracks,
                gt_by_frame,
                tracker_to_gt_id,
                args.min_visibility,
                args.identity_sample_interval,
                args.identity_correct_iou,
                args.identity_competitor_margin,
                args.identity_jump_factor,
                previous_sample_bboxes,
            )
        )
        exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, init_frame, tracker_to_gt_id, args.min_visibility)

        if args.save_video and preview_path is not None:
            preview_writer = v5.make_video_writer(preview_path, fps, init_frame_image)
            preview_writer.write(draw_backend_tracks(init_frame_image, backend, init_frame, version))

        print(
            f"[{execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}] "
            f"initialized {actual_tracks} tracks at source frame {init_frame}",
            flush=True,
        )
        record_debug_event(
            debug_rows,
            debug_csv,
            "case_initialized",
            sequence=sequence_path.name,
            lorat_config=lorat_config,
            execution_mode=execution_mode,
            target_tracks=target_tracks,
            actual_tracks=actual_tracks,
            init_frame=init_frame,
            usable_tracks=len(init_rows),
            selected_gt_track_ids=",".join(str(track_id) for track_id in gt_track_ids[:actual_tracks]),
            device=args.device,
            **tracker_debug_config(args),
            frames_expected=total_frames_expected,
            frames_completed=1,
            init_seconds=init_seconds,
            weight_path=weight_path,
            preview_path=preview_path if args.save_video else "",
        )

        tracking_seconds = 0.0
        for image_index in range(init_index + 1, end_index + 1):
            if STOP_REQUESTED:
                print(
                    f"[{execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}] "
                    "stop requested before the next frame; saving partial case.",
                    flush=True,
                )
                break
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
                    execution_mode,
                    target_tracks,
                    actual_tracks,
                    init_frame,
                    frame_number,
                    backend.tracks,
                    gt_by_frame,
                    tracker_to_gt_id,
                    args.min_visibility,
                    args.identity_sample_interval,
                    args.identity_correct_iou,
                    args.identity_competitor_margin,
                    args.identity_jump_factor,
                    previous_sample_bboxes,
                )
            )
            exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, frame_number, tracker_to_gt_id, args.min_visibility)

            if preview_writer is not None:
                preview_writer.write(draw_backend_tracks(frame, backend, frame_number, version))

            processed_frames = image_index - init_index + 1
            if args.progress_interval > 0 and (
                processed_frames == total_frames_expected or processed_frames % args.progress_interval == 0
            ):
                elapsed = time.perf_counter() - total_started
                print(
                    f"[{execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}] "
                    f"frame {processed_frames}/{total_frames_expected} "
                    f"(source frame {frame_number}), elapsed {elapsed:.1f}s",
                    flush=True,
                )
            if STOP_REQUESTED:
                print(
                    f"[{execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}] "
                    f"stop requested after source frame {frame_number}; saving partial case.",
                    flush=True,
                )
                break
        runtime_status = snapshot_runtime_status(backend, version)
    finally:
        runtime_status = snapshot_runtime_status(backend, version)
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
    evaluator_tasks_per_update_frame = (
        runtime_status.evaluator_tasks / update_frames
        if update_frames
        else None
    )
    model_forward_items_per_update_frame = (
        runtime_status.model_forward_items / update_frames
        if update_frames
        else None
    )
    model_forward_items_per_bbox = (
        runtime_status.model_forward_items / boxes_tracking
        if boxes_tracking
        else None
    )
    mean_iou = metrics["iou_sum"] / metrics["count"] if metrics["count"] else None
    iou50 = metrics["hit50"] / metrics["count"] if metrics["count"] else None

    timing = bench.TimingResult(
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
        execution_mode=execution_mode,
        gpu_profile=args.gpu_profile,
        gpu_name=runtime_status.gpu_name,
        gpu_memory_allocated_mb=runtime_status.gpu_allocated_mb,
        gpu_memory_reserved_mb=runtime_status.gpu_reserved_mb,
        gpu_memory_peak_allocated_mb=runtime_status.gpu_peak_allocated_mb,
        gpu_memory_peak_reserved_mb=runtime_status.gpu_peak_reserved_mb,
        max_evaluator_batch=runtime_status.max_evaluator_batch,
        evaluator_calls=runtime_status.evaluator_calls,
        evaluator_tasks=runtime_status.evaluator_tasks,
        model_forward_calls=runtime_status.model_forward_calls,
        model_forward_items=runtime_status.model_forward_items,
        max_model_forward_batch=runtime_status.max_model_forward_batch,
        fusion_forward_calls=runtime_status.fusion_forward_calls,
        fusion_forward_items=runtime_status.fusion_forward_items,
        max_fusion_forward_batch=runtime_status.max_fusion_forward_batch,
        evaluator_tasks_per_update_frame=evaluator_tasks_per_update_frame,
        model_forward_items_per_update_frame=model_forward_items_per_update_frame,
        model_forward_items_per_bbox=model_forward_items_per_bbox,
        gating_decisions=runtime_status.gating_decisions,
        gating_primary_decisions=runtime_status.gating_primary_decisions,
        gating_recovery_decisions=runtime_status.gating_recovery_decisions,
        gating_selected_slot_items=runtime_status.gating_selected_slot_items,
        gating_avg_slots_per_decision=runtime_status.gating_avg_slots_per_decision,
        gating_recovery_reasons=runtime_status.gating_recovery_reasons,
        fps_sustains_25=(fps_tracking is not None and fps_tracking >= args.fps_threshold),
    )
    print(
        f"{execution_mode} {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"actual={actual_tracks}, frames={frames}, "
        f"track_ms_per_box={bench.optional_float(tracking_ms_per_bbox, 3)}, "
        f"iou50={bench.optional_float(iou50, 3)}, "
        f"peak_gpu_reserved_mb={bench.optional_float(runtime_status.gpu_peak_reserved_mb, 2)}, "
        f"video={preview_path if args.save_video else 'disabled'}",
        flush=True,
    )
    record_debug_event(
        debug_rows,
        debug_csv,
        "case_complete" if not STOP_REQUESTED else "case_partial_stop",
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        execution_mode=execution_mode,
        target_tracks=target_tracks,
        actual_tracks=actual_tracks,
        init_frame=init_frame,
        usable_tracks=len(init_rows),
        selected_gt_track_ids=",".join(str(track_id) for track_id in gt_track_ids[:actual_tracks]),
        device=args.device,
        gpu_name=runtime_status.gpu_name,
        **tracker_debug_config(args),
        frames_expected=total_frames_expected,
        frames_completed=frames,
        update_frames=update_frames,
        boxes_tracking=boxes_tracking,
        total_seconds=total_seconds,
        init_seconds=init_seconds,
        tracking_seconds=tracking_seconds,
        total_ms_per_bbox="" if total_ms_per_bbox is None else total_ms_per_bbox,
        tracking_ms_per_bbox="" if tracking_ms_per_bbox is None else tracking_ms_per_bbox,
        fps_total=fps_total,
        fps_tracking="" if fps_tracking is None else fps_tracking,
        evaluator_calls=runtime_status.evaluator_calls,
        evaluator_tasks=runtime_status.evaluator_tasks,
        model_forward_calls=runtime_status.model_forward_calls,
        model_forward_items=runtime_status.model_forward_items,
        max_model_forward_batch=runtime_status.max_model_forward_batch,
        fusion_forward_calls=runtime_status.fusion_forward_calls,
        fusion_forward_items=runtime_status.fusion_forward_items,
        max_fusion_forward_batch=runtime_status.max_fusion_forward_batch,
        evaluator_tasks_per_update_frame="" if evaluator_tasks_per_update_frame is None else evaluator_tasks_per_update_frame,
        model_forward_items_per_update_frame="" if model_forward_items_per_update_frame is None else model_forward_items_per_update_frame,
        model_forward_items_per_bbox="" if model_forward_items_per_bbox is None else model_forward_items_per_bbox,
        gating_decisions=runtime_status.gating_decisions,
        gating_primary_decisions=runtime_status.gating_primary_decisions,
        gating_recovery_decisions=runtime_status.gating_recovery_decisions,
        gating_selected_slot_items=runtime_status.gating_selected_slot_items,
        gating_avg_slots_per_decision=runtime_status.gating_avg_slots_per_decision,
        gating_recovery_reasons=runtime_status.gating_recovery_reasons,
        gpu_memory_allocated_mb="" if runtime_status.gpu_allocated_mb is None else runtime_status.gpu_allocated_mb,
        gpu_memory_reserved_mb="" if runtime_status.gpu_reserved_mb is None else runtime_status.gpu_reserved_mb,
        gpu_memory_peak_allocated_mb="" if runtime_status.gpu_peak_allocated_mb is None else runtime_status.gpu_peak_allocated_mb,
        gpu_memory_peak_reserved_mb="" if runtime_status.gpu_peak_reserved_mb is None else runtime_status.gpu_peak_reserved_mb,
        max_evaluator_batch=runtime_status.max_evaluator_batch,
        mean_iou="" if mean_iou is None else mean_iou,
        iou50="" if iou50 is None else iou50,
        fps_sustains_threshold=fps_tracking is not None and fps_tracking >= args.fps_threshold,
        weight_path=weight_path,
        preview_path=preview_path if args.save_video else "",
        reason="stop requested" if STOP_REQUESTED else "",
    )
    return timing, observations


def smallest_reliable_area_by_key(rows: Sequence[bench.AreaSummary]) -> Dict[Tuple[str, str, str, int], Optional[float]]:
    return bench.smallest_reliable_area(rows)


def write_model_comparison_csv(
    path: Path,
    timing_rows: Sequence[bench.TimingResult],
    area_rows: Sequence[bench.AreaSummary],
) -> None:
    reliable_floor = smallest_reliable_area_by_key(area_rows)
    fieldnames = [
        "sequence",
        "lorat_config",
        "execution_mode",
        "backbone",
        "input_size",
        "target_tracks",
        "actual_tracks",
        "tracking_ms_per_bbox",
        "fps_tracking",
        "gpu_profile",
        "gpu_name",
        "gpu_memory_peak_reserved_mb",
        "max_evaluator_batch",
        "model_forward_calls",
        "model_forward_items",
        "model_forward_items_per_update_frame",
        "model_forward_items_per_bbox",
        "max_model_forward_batch",
        "fusion_forward_calls",
        "max_fusion_forward_batch",
        "gating_decisions",
        "gating_recovery_decisions",
        "gating_avg_slots_per_decision",
        "gating_recovery_reasons",
        "fps_sustains_25",
        "mean_iou",
        "iou50",
        "smallest_reliable_area_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in timing_rows:
            reliable_area = reliable_floor.get((row.sequence, row.lorat_config, row.execution_mode, row.target_tracks))
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "lorat_config": row.lorat_config,
                    "execution_mode": row.execution_mode,
                    "backbone": row.backbone,
                    "input_size": row.input_size,
                    "target_tracks": row.target_tracks,
                    "actual_tracks": row.actual_tracks,
                    "tracking_ms_per_bbox": bench.optional_float(row.tracking_ms_per_bbox),
                    "fps_tracking": bench.optional_float(row.fps_tracking),
                    "gpu_profile": row.gpu_profile,
                    "gpu_name": row.gpu_name,
                    "gpu_memory_peak_reserved_mb": bench.optional_float(row.gpu_memory_peak_reserved_mb, 2),
                    "max_evaluator_batch": row.max_evaluator_batch,
                    "model_forward_calls": row.model_forward_calls,
                    "model_forward_items": row.model_forward_items,
                    "model_forward_items_per_update_frame": bench.optional_float(row.model_forward_items_per_update_frame),
                    "model_forward_items_per_bbox": bench.optional_float(row.model_forward_items_per_bbox),
                    "max_model_forward_batch": row.max_model_forward_batch,
                    "fusion_forward_calls": row.fusion_forward_calls,
                    "max_fusion_forward_batch": row.max_fusion_forward_batch,
                    "gating_decisions": row.gating_decisions,
                    "gating_recovery_decisions": row.gating_recovery_decisions,
                    "gating_avg_slots_per_decision": bench.optional_float(row.gating_avg_slots_per_decision),
                    "gating_recovery_reasons": row.gating_recovery_reasons,
                    "fps_sustains_25": "" if row.fps_sustains_25 is None else str(row.fps_sustains_25),
                    "mean_iou": bench.optional_float(row.mean_iou),
                    "iou50": bench.optional_float(row.iou50),
                    "smallest_reliable_area_px": "" if reliable_area is None else f"{reliable_area:.0f}",
                }
            )


def build_speedup_rows(timing_rows: Sequence[bench.TimingResult]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, bench.TimingResult]] = {}
    for row in timing_rows:
        key = (row.sequence, row.lorat_config, row.target_tracks)
        grouped.setdefault(key, {})[row.execution_mode] = row
    rows: List[Dict[str, object]] = []
    for (sequence, lorat_config, target_tracks), by_mode in sorted(grouped.items()):
        serial = by_mode.get(BEFORE_LABEL)
        shared = by_mode.get(AFTER_LABEL)
        gated = by_mode.get(V6_LABEL)
        if serial is None or (shared is None and gated is None):
            continue
        serial_fps = serial.fps_tracking
        shared_fps = shared.fps_tracking if shared is not None else None
        gated_fps = gated.fps_tracking if gated is not None else None
        shared_speedup = None
        gated_speedup = None
        if serial_fps is not None and serial_fps > 0:
            if shared_fps is not None:
                shared_speedup = shared_fps / serial_fps
            if gated_fps is not None:
                gated_speedup = gated_fps / serial_fps
        rows.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "target_tracks": target_tracks,
                "serial_fps": serial_fps,
                "shared_fps": shared_fps,
                "gated_fps": gated_fps,
                "shared_fps_speedup": shared_speedup,
                "gated_fps_speedup": gated_speedup,
                "serial_ms_per_box": serial.tracking_ms_per_bbox,
                "shared_ms_per_box": shared.tracking_ms_per_bbox if shared is not None else None,
                "gated_ms_per_box": gated.tracking_ms_per_bbox if gated is not None else None,
                "shared_model_items_per_bbox": shared.model_forward_items_per_bbox if shared is not None else None,
                "gated_model_items_per_bbox": gated.model_forward_items_per_bbox if gated is not None else None,
            }
        )
    return rows


def max_sustainable_rows(
    timing_rows: Sequence[bench.TimingResult],
    fps_threshold: float,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[bench.TimingResult]] = {}
    for row in timing_rows:
        key = (row.gpu_profile, row.gpu_name or row.device, row.lorat_config, row.execution_mode)
        grouped.setdefault(key, []).append(row)
    rows: List[Dict[str, object]] = []
    for (gpu_profile, gpu_name, lorat_config, execution_mode), group_rows in sorted(grouped.items()):
        passing = [
            row.target_tracks
            for row in group_rows
            if row.fps_tracking is not None and row.fps_tracking >= fps_threshold
        ]
        if not passing:
            continue
        rows.append(
            {
                "gpu_profile": gpu_profile,
                "gpu_name": gpu_name,
                "lorat_config": lorat_config,
                "execution_mode": execution_mode,
                "max_n": max(passing),
            }
        )
    return rows


def summarize_identity_observations(
    observations: Sequence[bench.AreaObservation],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int], List[bench.AreaObservation]] = {}
    for observation in observations:
        if observation.sampled:
            key = (
                observation.sequence,
                observation.lorat_config,
                observation.execution_mode,
                observation.target_tracks,
            )
            grouped.setdefault(key, []).append(observation)
    rows: List[Dict[str, object]] = []
    for (sequence, lorat_config, execution_mode, target_tracks), group_rows in sorted(grouped.items()):
        samples = len(group_rows)
        correct_values = [row.correct_object for row in group_rows if row.correct_object is not None]
        jump_values = [row.identity_jump for row in group_rows if row.identity_jump is not None]
        occluded_values = [row.occluded for row in group_rows if row.occluded is not None]
        rows.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "execution_mode": execution_mode,
                "target_tracks": target_tracks,
                "samples": samples,
                "correct_rate": (sum(1 for value in correct_values if value) / len(correct_values)) if correct_values else None,
                "jump_rate": (sum(1 for value in jump_values if value) / len(jump_values)) if jump_values else None,
                "occluded_rate": (sum(1 for value in occluded_values if value) / len(occluded_values)) if occluded_values else None,
            }
        )
    return rows


def write_summary_md(
    path: Path,
    args: argparse.Namespace,
    run_label: str,
    timing_rows: Sequence[bench.TimingResult],
    area_rows: Sequence[bench.AreaSummary],
    observations: Sequence[bench.AreaObservation],
    model_comparison_csv: Path,
    debug_csv: Path,
    debug_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# LoRAT Multi-Object Benchmarks",
        "",
        "This run covers serial before-refactor timing, shared-backbone timing, V6 gated SOT-memory timing, GPU memory by object count, sampled object-correctness checks, and live-status instrumentation.",
        "",
        f"- Run label: `{run_label}`",
        f"- Device: `{args.device}`",
        f"- GPU profile label: `{args.gpu_profile}`",
        f"- Benchmark versions: `{args.benchmark_versions}`",
        f"- Track counts: `{','.join(str(value) for value in parse_track_counts(args))}`",
        f"- Max frames per run: `{args.max_frames}`",
        f"- 25 FPS capacity rule: max N where tracking FPS >= `{args.fps_threshold}`",
        f"- Area bins: `{args.area_bins}`",
        f"- Reliable bin rule: IoU@0.50 >= `{args.reliable_iou50}` and mean IoU >= `{args.reliable_mean_iou}` with at least `{args.min_area_samples}` samples",
        f"- Every-{args.identity_sample_interval}-frame correctness rule: own initialized-GT IoU >= `{args.identity_correct_iou}`, and no other visible GT exceeds it by more than `{args.identity_competitor_margin}`",
        f"- Jump rule: sampled center shift > `{args.identity_jump_factor}` x current GT diagonal",
        f"- V5 learning guard: shrink window `{args.shrink_guard_window}`, cumulative area ratio `< {args.shrink_guard_area_ratio}`, single-step ratio `< {args.shrink_guard_step_ratio}`, strong evidence confidence `>= {args.shrink_guard_min_confidence}` and ReID `>= {args.shrink_guard_min_reid}`",
        f"- V5 crop-information guard: score `< {args.crop_information_min_score}` holds template/ReID/size learning; full pixel credit starts at `{args.crop_information_min_pixels}` px",
        f"- V6 gated SOT memory: primary slots/track `{args.v6_primary_slots_per_track}`, recovery slots/track `{args.v6_recovery_slots_per_track}`, periodic recovery interval `{args.v6_recovery_interval}` frames",
        f"- V6 recovery triggers: confidence `< {args.v6_recovery_min_confidence}`, assignment score `< {args.v6_recovery_min_assignment_score}`, assignment margin `< {args.v6_recovery_min_assignment_margin}`, active slot stale after `{args.v6_recovery_stale_slot_frames}` frames",
        f"- Model comparison CSV: `{model_comparison_csv}`",
        f"- Debug case log CSV: `{debug_csv}`",
        "",
        "## Timing By Object Count",
        "",
        "| Sequence | Config | Mode | Target N | Actual N | Frames | Tracking Seconds | Tracking ms/box | FPS tracking | Eval tasks/frame | Model items/frame | Model items/box | Avg gated slots | >=25 FPS | Mean IoU | IoU@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.execution_mode} | {row.target_tracks} | {row.actual_tracks} | "
            f"{row.frames} | {row.tracking_seconds:.3f} | "
            f"{bench.optional_float(row.tracking_ms_per_bbox, 3)} | "
            f"{bench.optional_float(row.fps_tracking, 3)} | "
            f"{bench.optional_float(row.evaluator_tasks_per_update_frame, 3)} | "
            f"{bench.optional_float(row.model_forward_items_per_update_frame, 3)} | "
            f"{bench.optional_float(row.model_forward_items_per_bbox, 3)} | "
            f"{bench.optional_float(row.gating_avg_slots_per_decision, 3)} | "
            f"{'' if row.fps_sustains_25 is None else row.fps_sustains_25} | "
            f"{bench.optional_float(row.mean_iou, 3)} | {bench.optional_float(row.iou50, 3)} |"
        )

    speedup_rows = build_speedup_rows(timing_rows)
    if speedup_rows:
        lines.extend(
            [
                "",
                "## Before/After Speedup",
                "",
                "| Sequence | Config | Target N | Serial FPS | V5 FPS | V6 FPS | V5 Speedup | V6 Speedup | Serial ms/box | V5 ms/box | V6 ms/box | V5 model items/box | V6 model items/box |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in speedup_rows:
            lines.append(
                "| "
                f"{row['sequence']} | {row['lorat_config']} | {row['target_tracks']} | "
                f"{bench.optional_float(row['serial_fps'], 3)} | {bench.optional_float(row['shared_fps'], 3)} | "
                f"{bench.optional_float(row['gated_fps'], 3)} | "
                f"{bench.optional_float(row['shared_fps_speedup'], 3)} | {bench.optional_float(row['gated_fps_speedup'], 3)} | "
                f"{bench.optional_float(row['serial_ms_per_box'], 3)} | {bench.optional_float(row['shared_ms_per_box'], 3)} | "
                f"{bench.optional_float(row['gated_ms_per_box'], 3)} | "
                f"{bench.optional_float(row['shared_model_items_per_bbox'], 3)} | "
                f"{bench.optional_float(row['gated_model_items_per_bbox'], 3)} |"
            )

    lines.extend(
        [
            "",
            "## GPU Memory And 25 FPS Capacity",
            "",
            "| GPU Profile | GPU Name | Config | Mode | Target N | FPS tracking | Peak Reserved MB | Max Batch | >=25 FPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            "| "
            f"{row.gpu_profile} | {row.gpu_name or row.device} | {row.lorat_config} | {row.execution_mode} | "
            f"{row.target_tracks} | {bench.optional_float(row.fps_tracking, 3)} | "
            f"{bench.optional_float(row.gpu_memory_peak_reserved_mb, 2)} | {row.max_evaluator_batch} | "
            f"{'' if row.fps_sustains_25 is None else row.fps_sustains_25} |"
        )

    capacity_rows = max_sustainable_rows(timing_rows, args.fps_threshold)
    lines.extend(
        [
            "",
            "## Maximum N At 25 FPS",
            "",
            "| GPU Profile | GPU Name | Config | Mode | Max N >= Threshold | Threshold FPS |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    if capacity_rows:
        for row in capacity_rows:
            lines.append(
                "| "
                f"{row['gpu_profile']} | {row['gpu_name']} | {row['lorat_config']} | {row['execution_mode']} | "
                f"{row['max_n']} | {args.fps_threshold:.1f} |"
            )
    else:
        lines.append(f"| {args.gpu_profile} | {args.device} |  |  | none measured | {args.fps_threshold:.1f} |")

    preview_rows = [row for row in timing_rows if row.preview_path is not None]
    if preview_rows:
        lines.extend(["", "## Preview Videos", "", "| Sequence | Config | Mode | Target N | Preview Path |", "|---|---:|---:|---:|---|"])
        for row in preview_rows:
            lines.append(f"| {row.sequence} | {row.lorat_config} | {row.execution_mode} | {row.target_tracks} | `{row.preview_path}` |")

    notable_debug_rows = [
        row
        for row in debug_rows
        if (
            str(row.get("event", "")).startswith("skip_")
            or str(row.get("event", "")).startswith("case_partial")
            or str(row.get("event", "")) == "case_error"
        )
    ]
    if notable_debug_rows:
        lines.extend(
            [
                "",
                "## Skipped Or Partial Cases",
                "",
                "| Event | Sequence | Config | Mode | Target N | Usable Tracks | Init Frame | Selected GT IDs | Reason |",
                "|---|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in notable_debug_rows:
            lines.append(
                "| "
                f"{row.get('event', '')} | {row.get('sequence', '')} | {row.get('lorat_config', '')} | "
                f"{row.get('execution_mode', '')} | {row.get('target_tracks', '')} | {row.get('usable_tracks', '')} | "
                f"{row.get('init_frame', '')} | `{row.get('selected_gt_track_ids', '')}` | {row.get('reason', '')} |"
            )

    lines.extend(
        [
            "",
            "## Small-Object Reliability",
            "",
            "| Sequence | Config | Mode | Target N | Area Bin px | Samples | Mean Area | Mean IoU | IoU@0.50 | Unreliable Rate | Reliable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in area_rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.execution_mode} | {row.target_tracks} | {row.area_bin} | "
            f"{row.samples} | {bench.optional_float(row.mean_area_px, 1)} | {bench.optional_float(row.mean_iou, 3)} | "
            f"{bench.optional_float(row.iou50, 3)} | {bench.optional_float(row.unreliable_rate, 3)} | "
            f"{'' if row.reliable is None else row.reliable} |"
        )

    identity_rows = summarize_identity_observations(observations)
    if identity_rows:
        lines.extend(
            [
                "",
                f"## Every-{args.identity_sample_interval}-Frame Object Check",
                "",
                "| Sequence | Config | Mode | Target N | Samples | Correct Rate | Jump Rate | Occluded Rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in identity_rows:
            lines.append(
                "| "
                f"{row['sequence']} | {row['lorat_config']} | {row['execution_mode']} | {row['target_tracks']} | "
                f"{row['samples']} | {bench.optional_float(row['correct_rate'], 3)} | "
                f"{bench.optional_float(row['jump_rate'], 3)} | {bench.optional_float(row['occluded_rate'], 3)} |"
            )

    reliable_floor = smallest_reliable_area_by_key(area_rows)
    lines.extend(["", "## Model Size Comparison", ""])
    lines.append("| Sequence | Config | Mode | Backbone | Target N | Tracking ms/box | IoU@0.50 | Smallest Reliable Area px |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in timing_rows:
        reliable_area = reliable_floor.get((row.sequence, row.lorat_config, row.execution_mode, row.target_tracks))
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.execution_mode} | {row.backbone} | {row.target_tracks} | "
            f"{bench.optional_float(row.tracking_ms_per_bbox, 3)} | {bench.optional_float(row.iou50, 3)} | "
            f"{'' if reliable_area is None else int(reliable_area)} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_comparison_csv_path(output_paths: bench.OutputPaths, run_label: str) -> Path:
    path = output_paths.run_root / "model_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return bench.unique_path(path)


def debug_csv_path(args: argparse.Namespace, output_paths: bench.OutputPaths) -> Path:
    path = args.debug_csv.resolve() if args.debug_csv else output_paths.run_root / "benchmark_debug.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return bench.unique_path(path)


def default_output_paths(args: argparse.Namespace, run_label: str) -> bench.OutputPaths:
    run_root = args.output_root.resolve() / run_label
    timing_csv = args.timing_csv.resolve() if args.timing_csv else run_root / "timing_by_track_count.csv"
    area_csv = args.area_csv.resolve() if args.area_csv else run_root / "area_reliability.csv"
    observations_csv = args.observations_csv.resolve() if args.observations_csv else run_root / "area_observations.csv"
    summary_md = args.summary_md.resolve() if args.summary_md else run_root / "summary.md"
    video_dir = run_root / "videos"
    for path in (timing_csv, area_csv, observations_csv, summary_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    return bench.OutputPaths(
        run_root=run_root,
        timing_csv=bench.unique_path(timing_csv),
        area_csv=bench.unique_path(area_csv),
        observations_csv=bench.unique_path(observations_csv),
        summary_md=bench.unique_path(summary_md),
        video_dir=video_dir,
    )


def preview_video_path(
    output_paths: bench.OutputPaths,
    sequence_name: str,
    lorat_config: str,
    execution_label: str,
    target_tracks: int,
    max_frames: int,
) -> Path:
    frame_part = f"frames{max_frames}" if max_frames > 0 else "full"
    filename = bench.slugify(
        f"{sequence_name}_{lorat_config}_{execution_label}_N{target_tracks}_{frame_part}_preview"
    ) + ".mp4"
    return bench.unique_path(output_paths.video_dir / filename)


def flush_outputs(
    args: argparse.Namespace,
    area_bins: Sequence[float],
    output_paths: bench.OutputPaths,
    comparison_csv: Path,
    debug_csv: Path,
    run_label: str,
    timing_rows: Sequence[bench.TimingResult],
    observations: Sequence[bench.AreaObservation],
    debug_rows: Sequence[Dict[str, object]],
) -> List[bench.AreaSummary]:
    area_rows = bench.summarize_area_observations(
        observations,
        area_bins,
        args.reliable_iou50,
        args.reliable_mean_iou,
        args.min_area_samples,
    )
    bench.write_timing_csv(output_paths.timing_csv, timing_rows)
    bench.write_area_csv(output_paths.area_csv, area_rows)
    if args.write_observations:
        bench.write_observations_csv(output_paths.observations_csv, observations)
    write_model_comparison_csv(comparison_csv, timing_rows, area_rows)
    write_debug_csv(debug_csv, debug_rows)
    write_summary_md(
        output_paths.summary_md,
        args,
        run_label,
        timing_rows,
        area_rows,
        observations,
        comparison_csv,
        debug_csv,
        debug_rows,
    )
    return area_rows


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    args = parse_args()
    track_counts = parse_track_counts(args)
    benchmark_versions = parse_benchmark_versions(args)
    area_bins = bench.parse_area_bins(args.area_bins)
    sequences = select_sequences(args)
    if args.list_sequences:
        return 0

    configs = normalized_configs(args)
    run_label = make_run_label(args, sequences, configs, track_counts, benchmark_versions)
    output_paths = default_output_paths(args, run_label)
    comparison_csv = model_comparison_csv_path(output_paths, run_label)
    debug_csv = debug_csv_path(args, output_paths)
    timing_rows: List[bench.TimingResult] = []
    observations: List[bench.AreaObservation] = []
    debug_rows: List[Dict[str, object]] = []

    print(f"LoRAT benchmark run label: {run_label}", flush=True)
    print(f"Output folder: {output_paths.run_root}", flush=True)
    print(f"CSV timing output: {output_paths.timing_csv}", flush=True)
    print(f"CSV area output: {output_paths.area_csv}", flush=True)
    if args.write_observations:
        print(f"CSV observations output: {output_paths.observations_csv}", flush=True)
    print(f"CSV model comparison output: {comparison_csv}", flush=True)
    print(f"CSV debug output: {debug_csv}", flush=True)
    print(f"Markdown summary output: {output_paths.summary_md}", flush=True)
    print(f"Video previews: {output_paths.video_dir if args.save_video else 'disabled'}", flush=True)
    print(f"Sequences: {', '.join(sequence.name for sequence in sequences)}", flush=True)
    print(f"Configs: {', '.join(configs)}", flush=True)
    print(f"Benchmark versions: {', '.join(benchmark_versions)}", flush=True)
    print(f"Track counts: {', '.join(str(count) for count in track_counts)}", flush=True)
    print(f"Frames per run: {args.max_frames if args.max_frames > 0 else 'full sequence'}", flush=True)
    print(f"GPU profile label: {args.gpu_profile}", flush=True)

    record_debug_event(
        debug_rows,
        debug_csv,
        "run_start",
        device=args.device,
        **tracker_debug_config(args),
        reason=(
            f"sequences={','.join(sequence.name for sequence in sequences)}; "
            f"configs={','.join(configs)}; versions={','.join(benchmark_versions)}; "
            f"track_counts={','.join(str(count) for count in track_counts)}; "
            f"max_frames={args.max_frames if args.max_frames > 0 else 'full'}"
        ),
    )

    area_rows = flush_outputs(
        args,
        area_bins,
        output_paths,
        comparison_csv,
        debug_csv,
        run_label,
        timing_rows,
        observations,
        debug_rows,
    )

    for lorat_config in configs:
        for sequence_path in sequences:
            for version in benchmark_versions:
                execution_label = execution_label_for_version(version)
                for target_tracks in track_counts:
                    if STOP_REQUESTED:
                        print("Stop requested before starting the next benchmark case.", flush=True)
                        flush_outputs(
                            args,
                            area_bins,
                            output_paths,
                            comparison_csv,
                            debug_csv,
                            run_label,
                            timing_rows,
                            observations,
                            debug_rows,
                        )
                        return 0
                    preview_path = (
                        preview_video_path(
                            output_paths,
                            sequence_path.name,
                            lorat_config,
                            execution_label,
                            target_tracks,
                            args.max_frames,
                        )
                        if args.save_video
                        else None
                    )
                    try:
                        result = run_benchmark_case(
                            args,
                            sequence_path,
                            lorat_config,
                            version,
                            target_tracks,
                            preview_path,
                            debug_rows,
                            debug_csv,
                        )
                    except Exception as exc:
                        record_debug_event(
                            debug_rows,
                            debug_csv,
                            "case_error",
                            sequence=sequence_path.name,
                            lorat_config=lorat_config,
                            execution_mode=execution_label,
                            target_tracks=target_tracks,
                            device=args.device,
                            **tracker_debug_config(args),
                            preview_path=preview_path if args.save_video else "",
                            reason=repr(exc),
                        )
                        flush_outputs(
                            args,
                            area_bins,
                            output_paths,
                            comparison_csv,
                            debug_csv,
                            run_label,
                            timing_rows,
                            observations,
                            debug_rows,
                        )
                        raise
                    if result is None:
                        area_rows = flush_outputs(
                            args,
                            area_bins,
                            output_paths,
                            comparison_csv,
                            debug_csv,
                            run_label,
                            timing_rows,
                            observations,
                            debug_rows,
                        )
                        continue
                    timing, case_observations = result
                    timing_rows.append(timing)
                    observations.extend(case_observations)
                    area_rows = flush_outputs(
                        args,
                        area_bins,
                        output_paths,
                        comparison_csv,
                        debug_csv,
                        run_label,
                        timing_rows,
                        observations,
                        debug_rows,
                    )
                    if STOP_REQUESTED:
                        print("Stop requested; flushed partial benchmark outputs and exiting cleanly.", flush=True)
                        return 0

    record_debug_event(
        debug_rows,
        debug_csv,
        "run_complete",
        device=args.device,
        **tracker_debug_config(args),
        reason=f"completed_cases={len(timing_rows)}; observations={len(observations)}",
    )

    area_rows = flush_outputs(
        args,
        area_bins,
        output_paths,
        comparison_csv,
        debug_csv,
        run_label,
        timing_rows,
        observations,
        debug_rows,
    )

    print("Wrote LoRAT benchmark files:")
    print(f"  {output_paths.timing_csv}")
    print(f"  {output_paths.area_csv}")
    if args.write_observations:
        print(f"  {output_paths.observations_csv}")
    print(f"  {comparison_csv}")
    print(f"  {debug_csv}")
    if args.save_video:
        print(f"  video folder: {output_paths.video_dir}")
    print(f"  {output_paths.summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
