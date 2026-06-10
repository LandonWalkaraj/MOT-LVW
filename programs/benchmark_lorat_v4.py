from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

import benchmark_lorat_mot as bench
import bounding_box_v4_lorat_memory as v4
import exercise_lorat_mot as exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-benchmarks" / "v4-dancetrack"
DEFAULT_SEQUENCE = "dancetrack0065"
DEFAULT_MAX_FRAMES = 200
V4_EXECUTION_MODE = "v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark V4 LoRAT MOT speed by object count, small-object reliability, "
            "and model-size tradeoffs."
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
    parser.add_argument("--lorat-root", type=Path, default=v4.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v4.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(v4.LORAT_WEIGHT_BY_CONFIG) + ("all",),
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
        "--execution-modes",
        default=V4_EXECUTION_MODE,
        help="Compatibility label written to CSV. V4 itself has a single execution path.",
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

    add_v4_runtime_args(parser)
    return parser.parse_args()


def add_v4_runtime_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("V4 tracker runtime")
    group.add_argument("--track-batch-size", type=int, default=8)
    group.add_argument("--disable-amp", action="store_true")
    group.add_argument("--lorat-memory-slots", type=int, default=v4.DEFAULT_LORAT_MEMORY_SLOTS)
    group.add_argument("--lorat-memory-refresh-interval", type=int, default=v4.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL)
    group.add_argument("--lorat-active-slots-per-track", type=int, default=v4.DEFAULT_LORAT_ACTIVE_SLOTS_PER_TRACK)
    group.add_argument("--fixed-lorat-box-size", dest="lorat_fixed_box_size", action="store_true", default=v4.DEFAULT_LORAT_FIXED_BOX_SIZE)
    group.add_argument("--allow-lorat-size-change", dest="lorat_fixed_box_size", action="store_false")
    group.add_argument("--lorat-min-box-area", type=float, default=v4.DEFAULT_LORAT_MIN_BOX_AREA)
    group.add_argument("--lorat-max-area-change-per-frame", type=float, default=v4.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME)
    group.add_argument("--lorat-trusted-size-floor-scale", type=float, default=v4.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE)
    group.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    group.add_argument("--lorat-accept-min-score", type=float, default=v4.DEFAULT_LORAT_ACCEPT_MIN_SCORE)
    group.add_argument("--lorat-slot-capacity", type=int, default=0)
    group.add_argument("--lorat-search-area-factor", type=float, default=v4.DEFAULT_LORAT_SEARCH_AREA_FACTOR)
    group.add_argument("--lorat-window-penalty", type=float, default=v4.DEFAULT_LORAT_WINDOW_PENALTY)
    group.add_argument("--lorat-state-update-min-score", type=float, default=v4.DEFAULT_LORAT_STATE_UPDATE_MIN_SCORE)
    group.add_argument("--lorat-state-update-max-center-shift", type=float, default=v4.DEFAULT_LORAT_STATE_UPDATE_MAX_CENTER_SHIFT)
    group.add_argument("--lorat-state-update-max-area-change", type=float, default=v4.DEFAULT_LORAT_STATE_UPDATE_MAX_AREA_CHANGE)
    group.add_argument("--disable-identity-arbitration", action="store_true")
    group.add_argument("--identity-min-score", type=float, default=v4.DEFAULT_IDENTITY_MIN_SCORE)
    group.add_argument("--identity-min-reid", type=float, default=v4.DEFAULT_IDENTITY_MIN_REID)
    group.add_argument("--identity-min-motion", type=float, default=v4.DEFAULT_IDENTITY_MIN_MOTION)
    group.add_argument("--identity-min-path", type=float, default=v4.DEFAULT_IDENTITY_MIN_PATH)
    group.add_argument("--identity-bank-size", type=int, default=12)
    group.add_argument("--identity-memory-min-confidence", type=float, default=v4.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE)
    group.add_argument("--occlusion-max-frames", type=int, default=v4.DEFAULT_OCCLUSION_MAX_FRAMES)
    group.add_argument("--occlusion-iou-threshold", type=float, default=v4.DEFAULT_OCCLUSION_IOU_THRESHOLD)
    group.add_argument("--occlusion-velocity-damping", type=float, default=v4.DEFAULT_OCCLUSION_VELOCITY_DAMPING)
    group.add_argument("--reid-recovery-min-score", type=float, default=v4.DEFAULT_REID_RECOVERY_MIN_SCORE)
    group.add_argument("--reid-recovery-min-reid", type=float, default=v4.DEFAULT_REID_RECOVERY_MIN_REID)
    group.add_argument("--reid-recovery-min-motion", type=float, default=v4.DEFAULT_REID_RECOVERY_MIN_MOTION)
    group.add_argument("--reid-recovery-min-confidence", type=float, default=v4.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE)
    group.add_argument("--view-change-min-score", type=float, default=v4.DEFAULT_VIEW_CHANGE_MIN_SCORE)
    group.add_argument("--view-change-min-motion", type=float, default=v4.DEFAULT_VIEW_CHANGE_MIN_MOTION)
    group.add_argument("--view-change-min-confidence", type=float, default=v4.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE)
    group.add_argument("--view-change-max-lost-frames", type=int, default=v4.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES)


def parse_track_counts(args: argparse.Namespace) -> List[int]:
    if args.max_track_count > 0:
        return list(range(1, args.max_track_count + 1))
    return bench.parse_int_list(args.track_counts)


def parse_execution_modes(args: argparse.Namespace) -> List[str]:
    modes = []
    for mode in (part.strip() for part in args.execution_modes.split(",")):
        if not mode:
            continue
        if mode != V4_EXECUTION_MODE:
            raise RuntimeError(f"Unsupported V4 execution label {mode!r}; use {V4_EXECUTION_MODE!r}.")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise RuntimeError("--execution-modes must contain at least one mode.")
    return modes


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
    execution_modes: Sequence[str],
) -> str:
    mode_part = "modes-" + "-".join(mode.replace("-", "") for mode in execution_modes)
    return "v4_" + bench.make_run_label(args, sequences, configs, track_counts) + "_" + mode_part


def v4_args_for_run(
    args: argparse.Namespace,
    lorat_config: str,
    target_tracks: int,
    execution_mode: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        lorat_root=args.lorat_root,
        lorat_config=lorat_config,
        weight_path=args.weight_path,
        device=args.device,
        max_tracks=target_tracks,
        track_batch_size=args.track_batch_size,
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
    )


def build_v4_backend(
    args: argparse.Namespace,
    lorat_config: str,
    target_tracks: int,
    execution_mode: str,
    sequence_name: str,
    fps: float,
    length: Optional[int],
):
    run_args = v4_args_for_run(args, lorat_config, target_tracks, execution_mode)
    source = SimpleNamespace(fps=fps, length=length, name=sequence_name)
    return v4.create_backend(run_args, source, expected_tracks=target_tracks)


def empty_runtime_status(max_batch: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_name="",
        gpu_allocated_mb=None,
        gpu_reserved_mb=None,
        gpu_peak_allocated_mb=None,
        gpu_peak_reserved_mb=None,
        max_evaluator_batch=max_batch,
        evaluator_calls=0,
        evaluator_tasks=0,
    )


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
    previous_sample_bboxes: Dict[int, v4.BBox],
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
                center_jump_px = v4.center_distance(previous_sample_bbox, track.bbox)
                jump_threshold = identity_jump_factor * max(1.0, v4.bbox_diagonal(gt_row.bbox))
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


def run_benchmark_case(
    args: argparse.Namespace,
    sequence_path: Path,
    lorat_config: str,
    execution_mode: str,
    target_tracks: int,
    preview_path: Optional[Path],
) -> Optional[Tuple[bench.TimingResult, List[bench.AreaObservation]]]:
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
        print(
            f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: "
            f"only {len(init_rows)} usable GT tracks at selected init frame {init_frame}."
        )
        return None

    weight_path = args.weight_path or v4.LORAT_WEIGHT_BY_CONFIG[lorat_config]
    if not weight_path.exists():
        message = f"Missing LoRAT weight for {lorat_config}: {weight_path}"
        if args.skip_missing_weights:
            print(f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: {message}")
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
    previous_sample_bboxes: Dict[int, v4.BBox] = {}
    runtime_status = empty_runtime_status()

    print(
        f"Starting V4 {sequence_path.name} {lorat_config} {execution_mode} N={target_tracks}: "
        f"{total_frames_expected} frames, video={preview_path if args.save_video else 'disabled'}",
        flush=True,
    )
    backend = build_v4_backend(
        args,
        lorat_config,
        target_tracks,
        execution_mode,
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
            preview_writer = v4.make_video_writer(preview_path, fps, init_frame_image)
            preview_writer.write(v4.draw_tracks(init_frame_image, backend.tracks, init_frame, backend.backend_name))

        print(
            f"[V4 {sequence_path.name} {lorat_config} {execution_mode} N={target_tracks}] "
            f"initialized {actual_tracks} tracks at source frame {init_frame}",
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
                preview_writer.write(v4.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name))

            processed_frames = image_index - init_index + 1
            if args.progress_interval > 0 and (
                processed_frames == total_frames_expected or processed_frames % args.progress_interval == 0
            ):
                elapsed = time.perf_counter() - total_started
                print(
                    f"[V4 {sequence_path.name} {lorat_config} {execution_mode} N={target_tracks}] "
                    f"frame {processed_frames}/{total_frames_expected} "
                    f"(source frame {frame_number}), elapsed {elapsed:.1f}s",
                    flush=True,
                )
        runtime_status = empty_runtime_status(max_batch=getattr(backend, "track_batch_size", 0))
    finally:
        runtime_status = empty_runtime_status(max_batch=getattr(backend, "track_batch_size", 0))
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
        fps_sustains_25=(fps_tracking is not None and fps_tracking >= args.fps_threshold),
    )
    print(
        f"V4 {sequence_path.name} {lorat_config} {execution_mode} N={target_tracks}: "
        f"actual={actual_tracks}, frames={frames}, "
        f"track_ms_per_box={bench.optional_float(tracking_ms_per_bbox, 3)}, "
        f"iou50={bench.optional_float(iou50, 3)}, "
        f"peak_gpu_reserved_mb={bench.optional_float(runtime_status.gpu_peak_reserved_mb, 2)}, "
        f"video={preview_path if args.save_video else 'disabled'}",
        flush=True,
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
        serial = by_mode.get(V4_EXECUTION_MODE)
        shared = None
        if serial is None or shared is None:
            continue
        serial_fps = serial.fps_tracking
        shared_fps = shared.fps_tracking
        speedup = None
        if serial_fps is not None and shared_fps is not None and serial_fps > 0:
            speedup = shared_fps / serial_fps
        rows.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "target_tracks": target_tracks,
                "serial_fps": serial_fps,
                "shared_fps": shared_fps,
                "fps_speedup": speedup,
                "serial_ms_per_box": serial.tracking_ms_per_bbox,
                "shared_ms_per_box": shared.tracking_ms_per_bbox,
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
) -> None:
    lines = [
        "# V4 LoRAT Multi-Object Benchmarks",
        "",
        "This run covers the Week 2 tracker benchmark: serial before-refactor timing, shared-backbone after-refactor timing, GPU memory by object count, sampled object-correctness checks, and live-status instrumentation.",
        "",
        f"- Run label: `{run_label}`",
        f"- Device: `{args.device}`",
        f"- GPU profile label: `{args.gpu_profile}`",
        f"- Execution modes: `{args.execution_modes}`",
        f"- Track counts: `{','.join(str(value) for value in parse_track_counts(args))}`",
        f"- Max frames per run: `{args.max_frames}`",
        f"- 25 FPS capacity rule: max N where tracking FPS >= `{args.fps_threshold}`",
        f"- Area bins: `{args.area_bins}`",
        f"- Reliable bin rule: IoU@0.50 >= `{args.reliable_iou50}` and mean IoU >= `{args.reliable_mean_iou}` with at least `{args.min_area_samples}` samples",
        f"- Every-{args.identity_sample_interval}-frame correctness rule: own initialized-GT IoU >= `{args.identity_correct_iou}`, and no other visible GT exceeds it by more than `{args.identity_competitor_margin}`",
        f"- Jump rule: sampled center shift > `{args.identity_jump_factor}` x current GT diagonal",
        f"- Model comparison CSV: `{model_comparison_csv}`",
        "",
        "## Timing By Object Count",
        "",
        "| Sequence | Config | Mode | Target N | Actual N | Frames | Tracking Seconds | Tracking ms/box | FPS tracking | Max Batch | >=25 FPS | Mean IoU | IoU@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            "| "
            f"{row.sequence} | {row.lorat_config} | {row.execution_mode} | {row.target_tracks} | {row.actual_tracks} | "
            f"{row.frames} | {row.tracking_seconds:.3f} | "
            f"{bench.optional_float(row.tracking_ms_per_bbox, 3)} | "
            f"{bench.optional_float(row.fps_tracking, 3)} | {row.max_evaluator_batch} | "
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
                "| Sequence | Config | Target N | Serial FPS | Shared FPS | FPS Speedup | Serial ms/box | Shared ms/box |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in speedup_rows:
            lines.append(
                "| "
                f"{row['sequence']} | {row['lorat_config']} | {row['target_tracks']} | "
                f"{bench.optional_float(row['serial_fps'], 3)} | {bench.optional_float(row['shared_fps'], 3)} | "
                f"{bench.optional_float(row['fps_speedup'], 3)} | "
                f"{bench.optional_float(row['serial_ms_per_box'], 3)} | {bench.optional_float(row['shared_ms_per_box'], 3)} |"
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


def v4_preview_video_path(
    output_paths: bench.OutputPaths,
    sequence_name: str,
    lorat_config: str,
    execution_mode: str,
    target_tracks: int,
    max_frames: int,
) -> Path:
    frame_part = f"frames{max_frames}" if max_frames > 0 else "full"
    filename = bench.slugify(
        f"{sequence_name}_{lorat_config}_{execution_mode}_N{target_tracks}_{frame_part}_preview"
    ) + ".mp4"
    return bench.unique_path(output_paths.video_dir / filename)


def flush_outputs(
    args: argparse.Namespace,
    area_bins: Sequence[float],
    output_paths: bench.OutputPaths,
    comparison_csv: Path,
    run_label: str,
    timing_rows: Sequence[bench.TimingResult],
    observations: Sequence[bench.AreaObservation],
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
    write_summary_md(output_paths.summary_md, args, run_label, timing_rows, area_rows, observations, comparison_csv)
    return area_rows


def main() -> int:
    args = parse_args()
    track_counts = parse_track_counts(args)
    execution_modes = parse_execution_modes(args)
    area_bins = bench.parse_area_bins(args.area_bins)
    sequences = select_sequences(args)
    if args.list_sequences:
        return 0

    configs = normalized_configs(args)
    run_label = make_run_label(args, sequences, configs, track_counts, execution_modes)
    output_paths = default_output_paths(args, run_label)
    comparison_csv = model_comparison_csv_path(output_paths, run_label)
    timing_rows: List[bench.TimingResult] = []
    observations: List[bench.AreaObservation] = []

    print(f"V4 benchmark run label: {run_label}", flush=True)
    print(f"Output folder: {output_paths.run_root}", flush=True)
    print(f"CSV timing output: {output_paths.timing_csv}", flush=True)
    print(f"CSV area output: {output_paths.area_csv}", flush=True)
    if args.write_observations:
        print(f"CSV observations output: {output_paths.observations_csv}", flush=True)
    print(f"CSV model comparison output: {comparison_csv}", flush=True)
    print(f"Markdown summary output: {output_paths.summary_md}", flush=True)
    print(f"Video previews: {output_paths.video_dir if args.save_video else 'disabled'}", flush=True)
    print(f"Sequences: {', '.join(sequence.name for sequence in sequences)}", flush=True)
    print(f"Configs: {', '.join(configs)}", flush=True)
    print(f"Execution modes: {', '.join(execution_modes)}", flush=True)
    print(f"Track counts: {', '.join(str(count) for count in track_counts)}", flush=True)
    print(f"Frames per run: {args.max_frames if args.max_frames > 0 else 'full sequence'}", flush=True)
    print(f"GPU profile label: {args.gpu_profile}", flush=True)

    area_rows = flush_outputs(args, area_bins, output_paths, comparison_csv, run_label, timing_rows, observations)

    for lorat_config in configs:
        for sequence_path in sequences:
            for execution_mode in execution_modes:
                for target_tracks in track_counts:
                    preview_path = (
                        v4_preview_video_path(
                            output_paths,
                            sequence_path.name,
                            lorat_config,
                            execution_mode,
                            target_tracks,
                            args.max_frames,
                        )
                        if args.save_video
                        else None
                    )
                    result = run_benchmark_case(
                        args,
                        sequence_path,
                        lorat_config,
                        execution_mode,
                        target_tracks,
                        preview_path,
                    )
                    if result is None:
                        continue
                    timing, case_observations = result
                    timing_rows.append(timing)
                    observations.extend(case_observations)
                    area_rows = flush_outputs(args, area_bins, output_paths, comparison_csv, run_label, timing_rows, observations)

    print("Wrote V4 benchmark files:")
    print(f"  {output_paths.timing_csv}")
    print(f"  {output_paths.area_csv}")
    if args.write_observations:
        print(f"  {output_paths.observations_csv}")
    print(f"  {comparison_csv}")
    if args.save_video:
        print(f"  video folder: {output_paths.video_dir}")
    print(f"  {output_paths.summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
