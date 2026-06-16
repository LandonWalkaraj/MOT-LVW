from __future__ import annotations

import argparse
import copy
import csv
import io
import signal
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

import benchmark_lorat_mot as bench
import bounding_box_v8_lorat_quality_batched as v8
import exercise_lorat_mot as exercise
import mot_common as mot


DEFAULT_SEQUENCE = "dancetrack0065"
DEFAULT_TRACK_COUNTS = "1,2,3,4,5"
DEFAULT_OUTPUT_ROOT = mot.PROJECT_ROOT / "outputs" / "benchmarks" / "lorat-v8"
EXECUTION_MODE = v8.V8_EXECUTION_MODE

STOP_REQUESTED = False


@dataclass(frozen=True)
class V8TimingResult:
    sequence: str
    lorat_config: str
    backbone: str
    input_size: int
    device: str
    reid_mode: str
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
    execution_mode: str
    head_mode: str
    gpu_profile: str
    gpu_name: str
    gpu_memory_allocated_mb: Optional[float]
    gpu_memory_reserved_mb: Optional[float]
    gpu_memory_peak_allocated_mb: Optional[float]
    gpu_memory_peak_reserved_mb: Optional[float]
    shared_frame_backbone_calls: int
    shared_frame_backbone_items: int
    shared_backbone_calls_per_frame: Optional[float]
    object_head_batches: int
    object_head_items: int
    max_object_head_batch: int
    object_head_roi_tokens: int
    object_head_batches_per_update_frame: Optional[float]
    object_head_items_per_update_frame: Optional[float]
    object_head_items_per_bbox: Optional[float]
    selected_head_items: int
    selected_head_items_per_update_frame: Optional[float]
    profile_candidate_transfer_ms_per_update: Optional[float]
    profile_candidate_extract_ms_per_update: Optional[float]
    profile_template_match_ms_per_update: Optional[float]
    profile_candidate_fusion_ms_per_update: Optional[float]
    profile_reid_appearance_ms_per_update: Optional[float]
    profile_identity_resolve_ms_per_update: Optional[float]
    profile_identity_score_ms_per_update: Optional[float]
    profile_debug_output_ms_per_update: Optional[float]
    profile_accept_ms_per_update: Optional[float]
    profile_hold_ms_per_update: Optional[float]
    profile_appearance_refresh_ms_per_update: Optional[float]
    profile_proof_output_ms_per_update: Optional[float]
    profile_unbucketed_ms_per_update: Optional[float]
    proof_track_frames: int
    proof_shared_backbone_ok_frames: int
    proof_batched_head_ok_frames: int
    proof_shared_backbone_ok_rate: Optional[float]
    proof_batched_head_ok_rate: Optional[float]
    fps_sustains_25: Optional[bool]
    preview_path: Optional[Path]


@dataclass(frozen=True)
class IdentityObservation:
    sequence: str
    lorat_config: str
    execution_mode: str
    reid_mode: str
    target_tracks: int
    actual_tracks: int
    init_frame: int
    frame: int
    sample_offset: int
    tracker_id: int
    gt_track_id: int
    gt_visible: bool
    gt_visibility: float
    matched_gt_id: Optional[int]
    matched_gt_iou: float
    area_px: float
    own_iou: float
    best_other_iou: float
    correct_object: bool
    identity_jump: bool
    identity_switch: bool
    track_lost: bool
    center_jump_px: float
    occluded: bool
    ok: bool
    lost_frames: int
    lifecycle_state: str
    state: str


@dataclass
class V8OutputPaths:
    run_root: Path
    timing_csv: Path
    area_csv: Path
    observations_csv: Path
    full_observations_csv: Path
    identity_csv: Path
    identity_summary_csv: Path
    occlusion_survival_csv: Path
    week2_proof_csv: Path
    candidate_diagnostics_csv: Path
    debug_csv: Path
    summary_md: Path
    video_dir: Path


def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested; the benchmark will flush partial outputs at the next safe point.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v8-only LoRAT benchmark for Week 1/2/3 metrics: bounding-box timing, "
            "small-object area reliability, FPS/memory scaling, shared-frame ViT proof, "
            "ReID ablation, identity switches, track loss, and occlusion survival."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="val", help="Dataset split folder to search under --dataset-root.")
    parser.add_argument("--extract-zips", action="store_true")
    parser.add_argument("--sequence", action="append", help="Sequence name to benchmark. Repeat for multiple.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--list-sequences", action="store_true")
    parser.add_argument("--init-frame", default="auto", help="GT init frame number, or auto.")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to initialize. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--min-init-tracks", type=int, default=1)
    parser.add_argument("--allow-fewer-tracks", action="store_true")

    parser.add_argument("--track-counts", default=DEFAULT_TRACK_COUNTS)
    parser.add_argument("--max-track-count", type=int, default=0, help="Use 1..N instead of --track-counts when positive.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full sequence.")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--fps-threshold", type=float, default=25.0)
    parser.add_argument("--gpu-profile", default="local")

    parser.add_argument("--area-bins", default=bench.DEFAULT_AREA_BINS)
    parser.add_argument("--area-sample-interval", type=int, default=10, help="Sample small-object reliability every N frames.")
    parser.add_argument("--full-area-observations", action="store_true", help="Also write every-frame visible area observations.")
    parser.add_argument("--reliable-iou50", type=float, default=0.80)
    parser.add_argument("--reliable-mean-iou", type=float, default=0.50)
    parser.add_argument("--min-area-samples", type=int, default=10)
    parser.add_argument("--identity-sample-interval", type=int, default=10)
    parser.add_argument("--identity-correct-iou", type=float, default=0.30)
    parser.add_argument("--identity-competitor-margin", type=float, default=0.05)
    parser.add_argument("--identity-jump-factor", type=float, default=2.0)
    parser.add_argument(
        "--reid-ablation",
        "--week3-reid-ablation",
        dest="reid_ablation",
        action="store_true",
        help="Run each case twice: ReID on and ReID off.",
    )
    parser.add_argument("--disable-reid", action="store_true", help="Disable ReID/identity arbitration.")

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timing-csv", type=Path)
    parser.add_argument("--area-csv", type=Path)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--full-observations-csv", type=Path)
    parser.add_argument("--identity-csv", type=Path)
    parser.add_argument("--week2-proof-csv", type=Path)
    parser.add_argument("--candidate-diagnostics-csv", type=Path)
    parser.add_argument("--debug-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--save-video", action="store_true")

    parser.add_argument("--lorat-config", default="B-224", choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG) + ("all",),
        help="Run one or more LoRAT configs under v8. Use all to include B/L/g and 224/378.",
    )
    parser.add_argument("--weight-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lorat-root", type=Path, default=mot.DEFAULT_LORAT_ROOT)
    parser.add_argument("--disable-amp", action="store_true")

    parser.add_argument("--v8-frame-size", type=int, default=0)
    parser.add_argument("--v8-head-rank", type=int, default=mot.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--v8-head-hidden-dim", type=int, default=256)
    parser.add_argument("--v8-head-lora-rank", type=int, default=16)
    parser.add_argument("--v8-head-weights", type=Path)
    parser.add_argument(
        "--v8-head-weights-root",
        type=Path,
        help=(
            "Directory containing per-config trained V8 heads, e.g. "
            "B_224/v8_head_B_224_best_by_val_iou.pt or B_224/v8_head_B_224_latest.pt."
        ),
    )
    parser.add_argument(
        "--v8-head-checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Which checkpoint to load from --v8-head-weights-root. Direct --v8-head-weights overrides this.",
    )
    parser.add_argument("--v8-search-radius-factor", type=float, default=2.25)
    parser.add_argument("--v8-min-confidence", type=float, default=0.48)
    parser.add_argument("--v8-template-update-rate", type=float, default=0.08)
    parser.add_argument("--v8-template-update-min-confidence", type=float, default=0.58)
    parser.add_argument("--v8-score-reduction", choices=("max", "mean"), default="max")
    parser.add_argument("--lorat-memory-slots", type=int, default=mot.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--lorat-memory-refresh-interval", type=int, default=mot.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL)
    parser.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    parser.add_argument("--lorat-accept-min-score", type=float, default=mot.DEFAULT_LORAT_ACCEPT_MIN_SCORE)
    parser.add_argument("--fixed-lorat-box-size", dest="lorat_fixed_box_size", action="store_true", default=mot.DEFAULT_LORAT_FIXED_BOX_SIZE)
    parser.add_argument("--allow-lorat-size-change", dest="lorat_fixed_box_size", action="store_false")
    parser.add_argument("--lorat-min-box-area", type=float, default=mot.DEFAULT_LORAT_MIN_BOX_AREA)
    parser.add_argument("--lorat-max-area-change-per-frame", type=float, default=mot.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME)
    parser.add_argument("--lorat-trusted-size-floor-scale", type=float, default=mot.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE)
    parser.add_argument("--shrink-guard-window", type=int, default=mot.DEFAULT_SHRINK_GUARD_WINDOW)
    parser.add_argument("--shrink-guard-area-ratio", type=float, default=mot.DEFAULT_SHRINK_GUARD_AREA_RATIO)
    parser.add_argument("--shrink-guard-step-ratio", type=float, default=mot.DEFAULT_SHRINK_GUARD_STEP_RATIO)
    parser.add_argument("--shrink-guard-min-confidence", type=float, default=mot.DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE)
    parser.add_argument("--shrink-guard-min-reid", type=float, default=mot.DEFAULT_SHRINK_GUARD_MIN_REID)
    parser.add_argument("--crop-information-min-score", type=float, default=mot.DEFAULT_CROP_INFORMATION_MIN_SCORE)
    parser.add_argument("--crop-information-min-pixels", type=int, default=mot.DEFAULT_CROP_INFORMATION_MIN_PIXELS)
    parser.add_argument("--disable-identity-arbitration", action="store_true")
    parser.add_argument("--identity-min-score", type=float, default=mot.DEFAULT_IDENTITY_MIN_SCORE)
    parser.add_argument("--identity-min-reid", type=float, default=mot.DEFAULT_IDENTITY_MIN_REID)
    parser.add_argument("--identity-min-motion", type=float, default=mot.DEFAULT_IDENTITY_MIN_MOTION)
    parser.add_argument("--identity-min-path", type=float, default=mot.DEFAULT_IDENTITY_MIN_PATH)
    parser.add_argument("--identity-bank-size", type=int, default=12)
    parser.add_argument("--identity-memory-min-confidence", type=float, default=mot.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE)
    parser.add_argument("--occlusion-max-frames", type=int, default=mot.DEFAULT_OCCLUSION_MAX_FRAMES)
    parser.add_argument("--occlusion-iou-threshold", type=float, default=mot.DEFAULT_OCCLUSION_IOU_THRESHOLD)
    parser.add_argument("--occlusion-velocity-damping", type=float, default=mot.DEFAULT_OCCLUSION_VELOCITY_DAMPING)
    parser.add_argument("--reid-recovery-min-score", type=float, default=mot.DEFAULT_REID_RECOVERY_MIN_SCORE)
    parser.add_argument("--reid-recovery-min-reid", type=float, default=mot.DEFAULT_REID_RECOVERY_MIN_REID)
    parser.add_argument("--reid-recovery-min-motion", type=float, default=mot.DEFAULT_REID_RECOVERY_MIN_MOTION)
    parser.add_argument("--reid-recovery-min-confidence", type=float, default=mot.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE)
    parser.add_argument("--view-change-min-score", type=float, default=mot.DEFAULT_VIEW_CHANGE_MIN_SCORE)
    parser.add_argument("--view-change-min-motion", type=float, default=mot.DEFAULT_VIEW_CHANGE_MIN_MOTION)
    parser.add_argument("--view-change-min-confidence", type=float, default=mot.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE)
    parser.add_argument("--view-change-max-lost-frames", type=int, default=mot.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES)
    parser.add_argument("--v8-primary-heads-per-track", type=int, default=v8.DEFAULT_V8_PRIMARY_HEADS_PER_TRACK)
    parser.add_argument("--v8-recovery-heads-per-track", type=int, default=v8.DEFAULT_V8_RECOVERY_HEADS_PER_TRACK)
    parser.add_argument("--v8-recovery-interval", type=int, default=v8.DEFAULT_V8_RECOVERY_INTERVAL)
    parser.add_argument("--v8-recovery-min-confidence", type=float, default=v8.DEFAULT_V8_RECOVERY_MIN_CONFIDENCE)
    parser.add_argument("--v8-recovery-min-assignment-score", type=float, default=v8.DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_SCORE)
    parser.add_argument("--v8-recovery-min-assignment-margin", type=float, default=v8.DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_MARGIN)
    parser.add_argument("--v8-recovery-stale-head-frames", type=int, default=v8.DEFAULT_V8_RECOVERY_STALE_HEAD_FRAMES)
    parser.add_argument(
        "--disable-v8-template-match",
        dest="v8_template_match",
        action="store_false",
        default=v8.DEFAULT_V8_TEMPLATE_MATCH_ENABLED,
        help="Disable V8 shared-feature template recovery after the batched head.",
    )
    parser.add_argument("--v8-template-match-min-score", type=float, default=v8.DEFAULT_V8_TEMPLATE_MATCH_MIN_SCORE)
    parser.add_argument("--v8-template-match-prefer-margin", type=float, default=v8.DEFAULT_V8_TEMPLATE_MATCH_PREFER_MARGIN)
    parser.add_argument(
        "--v8-template-match-every-frame",
        dest="v8_template_match_on_uncertain_only",
        action="store_false",
        default=v8.DEFAULT_V8_TEMPLATE_MATCH_ON_UNCERTAIN_ONLY,
        help="Run V8 shared-feature template recovery every frame instead of only when a trained head is uncertain.",
    )
    parser.add_argument(
        "--v8-template-match-uncertain-only",
        dest="v8_template_match_on_uncertain_only",
        action="store_true",
        help="Only run V8 shared-feature template recovery when a trained head is uncertain.",
    )
    parser.add_argument(
        "--v8-template-match-head-confidence-gate",
        type=float,
        default=v8.DEFAULT_V8_TEMPLATE_MATCH_HEAD_CONFIDENCE_GATE,
    )
    parser.add_argument(
        "--v8-template-match-margin-gate",
        type=float,
        default=v8.DEFAULT_V8_TEMPLATE_MATCH_MARGIN_GATE,
    )
    parser.add_argument("--v8-head-template-blend", type=float, default=v8.DEFAULT_V8_HEAD_TEMPLATE_BLEND)
    parser.add_argument("--v8-memory-min-motion", type=float, default=v8.DEFAULT_V8_MEMORY_MIN_MOTION)
    parser.add_argument("--v8-memory-min-path", type=float, default=v8.DEFAULT_V8_MEMORY_MIN_PATH)
    parser.add_argument("--v8-memory-min-appearance", type=float, default=v8.DEFAULT_V8_MEMORY_MIN_APPEARANCE)
    parser.add_argument("--v8-memory-min-stable-updates", type=int, default=v8.DEFAULT_V8_MEMORY_MIN_STABLE_UPDATES)
    parser.add_argument("--v8-accept-min-initial-anchor", type=float, default=v8.DEFAULT_V8_ACCEPT_MIN_INITIAL_ANCHOR)
    parser.add_argument("--v8-accept-min-identity-margin", type=float, default=v8.DEFAULT_V8_ACCEPT_MIN_IDENTITY_MARGIN)
    parser.add_argument("--v8-memory-min-initial-anchor", type=float, default=v8.DEFAULT_V8_MEMORY_MIN_INITIAL_ANCHOR)
    parser.add_argument("--v8-memory-min-identity-margin", type=float, default=v8.DEFAULT_V8_MEMORY_MIN_IDENTITY_MARGIN)
    parser.add_argument("--v8-window-penalty-ratio", type=float, default=v8.DEFAULT_V8_WINDOW_PENALTY_RATIO)
    return parser.parse_args()


def parse_track_counts(args: argparse.Namespace) -> List[int]:
    if args.max_track_count > 0:
        return list(range(1, args.max_track_count + 1))
    return bench.parse_int_list(args.track_counts)


def normalized_configs(args: argparse.Namespace) -> List[str]:
    if args.compare_configs:
        if args.weight_path:
            raise RuntimeError("--weight-path cannot be combined with --compare-configs.")
        return exercise.normalized_compare_configs(args.compare_configs)
    return [args.lorat_config]


def reid_case_modes(args: argparse.Namespace) -> List[Tuple[str, bool]]:
    if getattr(args, "reid_ablation", False):
        return [("reid_on", False), ("reid_off", True)]
    disabled = bool(getattr(args, "disable_reid", False) or getattr(args, "disable_identity_arbitration", False))
    return [("reid_off" if disabled else "reid_on", disabled)]


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
    selected = [sequence for sequence in sequences if sequence.name in wanted]
    missing = wanted - {sequence.name for sequence in selected}
    if missing:
        raise RuntimeError(f"Requested sequences not found: {sorted(missing)}")
    if args.max_sequences > 0:
        selected = selected[: args.max_sequences]
    return selected


def run_label(args: argparse.Namespace, sequences: Sequence[Path], configs: Sequence[str], track_counts: Sequence[int]) -> str:
    base = bench.make_run_label(args, sequences, configs, track_counts)
    return bench.slugify(f"v8_{base}")


def default_output_paths(args: argparse.Namespace, label: str) -> V8OutputPaths:
    run_root = args.output_root.resolve() / label
    run_root.mkdir(parents=True, exist_ok=True)
    video_dir = run_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = bench.unique_path(args.timing_csv.resolve() if args.timing_csv else run_root / "timing_by_object_count.csv")
    area_csv = bench.unique_path(args.area_csv.resolve() if args.area_csv else run_root / "area_reliability.csv")
    observations_csv = bench.unique_path(
        args.observations_csv.resolve() if args.observations_csv else run_root / "area_observations_sampled.csv"
    )
    full_observations_csv = bench.unique_path(
        args.full_observations_csv.resolve() if args.full_observations_csv else run_root / "area_observations_every_frame.csv"
    )
    identity_csv = bench.unique_path(args.identity_csv.resolve() if args.identity_csv else run_root / "identity_observations_sampled.csv")
    identity_summary_csv = bench.unique_path(run_root / "identity_recovery_summary.csv")
    occlusion_survival_csv = bench.unique_path(run_root / "occlusion_survival.csv")
    week2_proof_csv = bench.unique_path(args.week2_proof_csv.resolve() if args.week2_proof_csv else run_root / "week2_shared_backbone_proof.csv")
    candidate_diagnostics_csv = bench.unique_path(
        args.candidate_diagnostics_csv.resolve()
        if args.candidate_diagnostics_csv
        else run_root / "candidate_diagnostics.csv"
    )
    debug_csv = bench.unique_path(args.debug_csv.resolve() if args.debug_csv else run_root / "debug_log.csv")
    summary_md = bench.unique_path(args.summary_md.resolve() if args.summary_md else run_root / "summary.md")
    for path in (
        timing_csv,
        area_csv,
        observations_csv,
        full_observations_csv,
        identity_csv,
        identity_summary_csv,
        occlusion_survival_csv,
        week2_proof_csv,
        candidate_diagnostics_csv,
        debug_csv,
        summary_md,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    return V8OutputPaths(
        run_root=run_root,
        timing_csv=timing_csv,
        area_csv=area_csv,
        observations_csv=observations_csv,
        full_observations_csv=full_observations_csv,
        identity_csv=identity_csv,
        identity_summary_csv=identity_summary_csv,
        occlusion_survival_csv=occlusion_survival_csv,
        week2_proof_csv=week2_proof_csv,
        candidate_diagnostics_csv=candidate_diagnostics_csv,
        debug_csv=debug_csv,
        summary_md=summary_md,
        video_dir=video_dir,
    )


def preview_video_path(paths: V8OutputPaths, sequence: str, config: str, target_tracks: int, max_frames: int) -> Path:
    frame_part = f"frames{max_frames}" if max_frames > 0 else "full"
    name = bench.slugify(f"{sequence}_{config}_v8_N{target_tracks}_{frame_part}_preview") + ".mp4"
    return bench.unique_path(paths.video_dir / name)


def v8_head_config_key(lorat_config: str) -> str:
    return lorat_config.replace("-", "_")


def resolve_v8_head_weights(args: argparse.Namespace, lorat_config: str) -> Optional[Path]:
    if args.v8_head_weights is not None:
        return args.v8_head_weights
    if args.v8_head_weights_root is None:
        return None
    root = args.v8_head_weights_root
    config_key = v8_head_config_key(lorat_config)
    checkpoint_suffix = "best_by_val_iou" if getattr(args, "v8_head_checkpoint", "best") == "best" else "latest"
    fallback_suffix = "latest" if checkpoint_suffix == "best_by_val_iou" else "best_by_val_iou"
    candidates = [
        root / config_key / f"v8_head_{config_key}_{checkpoint_suffix}.pt",
        root / f"v8_head_{config_key}_{checkpoint_suffix}.pt",
        root / config_key / f"v8_head_{config_key}_{fallback_suffix}.pt",
        root / f"v8_head_{config_key}_{fallback_suffix}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"No trained V8 head found for {lorat_config}. Checked: {candidate_text}")


def tracker_args_for_run(args: argparse.Namespace, lorat_config: str, target_tracks: int) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.lorat_config = lorat_config
    run_args.max_tracks = target_tracks
    run_args.v8_head_weights = resolve_v8_head_weights(args, lorat_config)
    run_args.output = None
    run_args.debug_log = None
    run_args.slot_debug_log = None
    run_args.week2_proof_log = None
    run_args.collect_week2_proof = True
    run_args.no_week2_proof_log = False
    run_args.no_slot_debug_log = True
    run_args.no_save_video = True
    run_args.save_video = None
    run_args.no_display = True
    run_args.initial_boxes = ""
    return run_args


def sample_due(frame_number: int, init_frame: int, interval: int) -> bool:
    return interval > 0 and (frame_number - init_frame) % interval == 0


def visible_gt_rows(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    frame_number: int,
    min_visibility: float,
) -> List[exercise.GroundTruthRow]:
    return [
        row
        for row in gt_by_frame.get(frame_number, [])
        if row.confidence != 0 and row.visibility >= min_visibility
    ]


def collect_sampled_observations(
    sequence: str,
    lorat_config: str,
    reid_mode: str,
    target_tracks: int,
    actual_tracks: int,
    init_frame: int,
    frame_number: int,
    tracks: Sequence[mot.TrackState],
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    min_visibility: float,
    area_sample_interval: int,
    identity_sample_interval: int,
    identity_correct_iou: float,
    identity_competitor_margin: float,
    identity_jump_factor: float,
    previous_identity_bboxes: Dict[int, mot.BBox],
    collect_full_area: bool,
) -> Tuple[List[bench.AreaObservation], List[bench.AreaObservation], List[IdentityObservation]]:
    rows = visible_gt_rows(gt_by_frame, frame_number, min_visibility)
    all_rows = [row for row in gt_by_frame.get(frame_number, []) if row.confidence != 0]
    gt_rows = {row.track_id: row for row in rows}
    all_gt_rows = {row.track_id: row for row in all_rows}
    area_sampled = sample_due(frame_number, init_frame, area_sample_interval)
    identity_sampled = sample_due(frame_number, init_frame, identity_sample_interval)
    sampled_area: List[bench.AreaObservation] = []
    full_area: List[bench.AreaObservation] = []
    identity_rows: List[IdentityObservation] = []

    for track in tracks:
        gt_track_id = tracker_to_gt_id.get(track.track_id)
        if gt_track_id is None:
            continue
        visible_gt_row = gt_rows.get(gt_track_id)
        gt_row = visible_gt_row or all_gt_rows.get(gt_track_id)
        gt_visible = visible_gt_row is not None
        if gt_row is not None:
            _, _, width, height = gt_row.bbox
            area = max(0.0, float(width) * float(height))
            own_iou = exercise.bbox_iou(track.bbox, gt_row.bbox)
            gt_visibility = float(gt_row.visibility)
            reference_bbox = gt_row.bbox
        else:
            area = 0.0
            own_iou = 0.0
            gt_visibility = 0.0
            reference_bbox = track.bbox
        best_other_iou = max(
            (exercise.bbox_iou(track.bbox, other.bbox) for other in rows if other.track_id != gt_track_id),
            default=0.0,
        )
        other_ious = [
            (other.track_id, exercise.bbox_iou(track.bbox, other.bbox))
            for other in rows
            if other.track_id != gt_track_id
        ]
        matched_gt_id, matched_gt_iou = mot.matched_gt_id_from_ious(
            gt_track_id,
            own_iou,
            other_ious,
            identity_correct_iou,
        )
        state = str(getattr(track, "state", ""))
        lifecycle_state = mot.set_track_lifecycle(track)
        occluded = not gt_visible
        occluded = occluded or any(token in state.upper() for token in ("OCCLU", "LOST", "MISS", "LOWCONF", "ID_UNCERTAIN"))
        occluded = occluded or not bool(getattr(track, "ok", False))
        correct_object: Optional[bool] = None
        identity_jump: Optional[bool] = None
        identity_switch: Optional[bool] = None
        track_lost: Optional[bool] = None
        center_jump_px: Optional[float] = None

        if identity_sampled:
            correct_object = (
                gt_visible
                and gt_row is not None
                and bool(getattr(track, "ok", False))
                and own_iou >= identity_correct_iou
                and own_iou + identity_competitor_margin >= best_other_iou
            )
            identity_switch = gt_visible and matched_gt_id is not None and matched_gt_id != gt_track_id
            track_lost = (
                not bool(getattr(track, "ok", False))
                or lifecycle_state == mot.TrackLifecycle.LOST
                or (gt_visible and matched_gt_id is None)
            )
            previous_bbox = previous_identity_bboxes.get(track.track_id)
            if previous_bbox is None:
                center_jump_px = 0.0
                identity_jump = False
            else:
                center_jump_px = mot.center_distance(previous_bbox, track.bbox)
                jump_threshold = identity_jump_factor * max(1.0, mot.bbox_diagonal(reference_bbox))
                identity_jump = center_jump_px > jump_threshold
            previous_identity_bboxes[track.track_id] = track.bbox
            identity_rows.append(
                IdentityObservation(
                    sequence=sequence,
                    lorat_config=lorat_config,
                    execution_mode=EXECUTION_MODE,
                    reid_mode=reid_mode,
                    target_tracks=target_tracks,
                    actual_tracks=actual_tracks,
                    init_frame=init_frame,
                    frame=frame_number,
                    sample_offset=frame_number - init_frame,
                    tracker_id=track.track_id,
                    gt_track_id=gt_track_id,
                    gt_visible=gt_visible,
                    gt_visibility=gt_visibility,
                    matched_gt_id=matched_gt_id,
                    matched_gt_iou=matched_gt_iou,
                    area_px=area,
                    own_iou=own_iou,
                    best_other_iou=best_other_iou,
                    correct_object=bool(correct_object),
                    identity_jump=bool(identity_jump),
                    identity_switch=bool(identity_switch),
                    track_lost=bool(track_lost),
                    center_jump_px=float(center_jump_px or 0.0),
                    occluded=occluded,
                    ok=bool(getattr(track, "ok", False)),
                    lost_frames=int(getattr(track, "lost_frames", 0) or 0),
                    lifecycle_state=lifecycle_state,
                    state=state,
                )
            )

        if visible_gt_row is None:
            continue
        observation = bench.AreaObservation(
            sequence=sequence,
            lorat_config=lorat_config,
            execution_mode=EXECUTION_MODE,
            target_tracks=target_tracks,
            actual_tracks=actual_tracks,
            frame=frame_number,
            tracker_id=track.track_id,
            gt_track_id=gt_track_id,
            area_px=area,
            iou=exercise.bbox_iou(track.bbox, visible_gt_row.bbox),
            ok=bool(getattr(track, "ok", False)),
            state=state,
            sampled=area_sampled,
            correct_object=correct_object,
            identity_jump=identity_jump,
            occluded=occluded if identity_sampled else None,
            center_jump_px=center_jump_px,
        )
        if area_sampled:
            sampled_area.append(observation)
        if collect_full_area:
            full_area.append(observation)
    return sampled_area, full_area, identity_rows


def coerce_bbox(value: object) -> Optional[mot.BBox]:
    if value is None:
        return None
    try:
        items = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(items) != 4:
        return None
    return items  # type: ignore[return-value]


def bbox_text(value: object) -> str:
    bbox = coerce_bbox(value)
    if bbox is None:
        return ""
    return ";".join(f"{item:.3f}" for item in bbox)


def optional_iou(candidate: object, gt_bbox: mot.BBox) -> Optional[float]:
    bbox = coerce_bbox(candidate)
    if bbox is None:
        return None
    return exercise.bbox_iou(bbox, gt_bbox)


def optional_center_error(candidate: object, gt_bbox: mot.BBox) -> Optional[float]:
    bbox = coerce_bbox(candidate)
    if bbox is None:
        return None
    return mot.center_distance(bbox, gt_bbox)


def bbox_area(bbox: Optional[mot.BBox]) -> Optional[float]:
    if bbox is None:
        return None
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def optional_area_ratio(candidate: object, gt_bbox: mot.BBox) -> Optional[float]:
    bbox = coerce_bbox(candidate)
    candidate_area = bbox_area(bbox)
    gt_area = bbox_area(gt_bbox)
    if candidate_area is None or gt_area is None:
        return None
    return candidate_area / max(1.0, gt_area)


def optional_center_error_gt_size(candidate: object, gt_bbox: mot.BBox) -> Optional[float]:
    center_error = optional_center_error(candidate, gt_bbox)
    if center_error is None:
        return None
    _, _, gt_width, gt_height = gt_bbox
    return center_error / max(1.0, max(float(gt_width), float(gt_height)))


def classify_iou_failure(
    final_iou: Optional[float],
    final_best_other_iou: Optional[float],
    head_iou: Optional[float],
    head_top5_best_iou: float,
    template_iou: Optional[float],
    fused_iou: Optional[float],
    assigned_iou: Optional[float],
    final_center_error_gt_size: Optional[float],
    final_area_ratio: Optional[float],
) -> Tuple[str, str]:
    if final_iou is not None and final_iou >= 0.50:
        return "ok_iou50", "final"
    if final_iou is not None and final_iou >= 0.30:
        return "partial_overlap_low_precision", "final"
    if final_best_other_iou is not None and final_iou is not None and final_best_other_iou > final_iou + 0.05:
        return "identity_swap_or_assignment_conflict", "identity"
    if head_iou is not None and head_iou >= 0.50 and (final_iou or 0.0) < 0.50:
        return "post_head_fusion_or_hold_degraded_good_head_box", "fusion_or_state"
    if assigned_iou is not None and assigned_iou >= 0.50 and (final_iou or 0.0) < 0.50:
        return "assignment_found_good_box_but_final_degraded", "assignment"
    if fused_iou is not None and fused_iou >= 0.50 and (final_iou or 0.0) < 0.50:
        return "fusion_found_good_box_but_final_degraded", "fusion_or_state"
    if template_iou is not None and template_iou >= 0.50 and (final_iou or 0.0) < 0.50:
        return "template_recovery_found_good_box_but_final_degraded", "template_or_fusion"
    if head_top5_best_iou >= 0.50 and (head_iou or 0.0) < 0.50:
        return "head_ranking_missed_good_top5_candidate", "head_scoring"
    if head_top5_best_iou >= 0.30 and (head_iou or 0.0) < 0.30:
        return "head_has_partial_top5_candidate_but_ranked_poorly", "head_scoring"
    if final_center_error_gt_size is not None and final_center_error_gt_size > 1.0:
        return "large_center_drift", "localization"
    if final_area_ratio is not None and (final_area_ratio < 0.35 or final_area_ratio > 2.75):
        return "box_scale_mismatch", "box_geometry"
    if (head_iou or 0.0) < 0.30 and head_top5_best_iou < 0.30:
        return "head_no_good_candidate_in_top5", "head_geometry"
    return "low_iou_unclassified", "unknown"


def collect_candidate_diagnostics(
    sequence: str,
    lorat_config: str,
    reid_mode: str,
    target_tracks: int,
    actual_tracks: int,
    init_frame: int,
    frame_number: int,
    tracks: Sequence[mot.TrackState],
    raw_diagnostics: Sequence[Dict[str, object]],
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    min_visibility: float,
) -> List[Dict[str, object]]:
    rows = visible_gt_rows(gt_by_frame, frame_number, min_visibility)
    gt_rows = {row.track_id: row for row in rows}
    track_by_id = {track.track_id: track for track in tracks}
    output_rows: List[Dict[str, object]] = []

    for diagnostic in raw_diagnostics:
        track_id = int(diagnostic.get("track_id", -1))
        gt_track_id = tracker_to_gt_id.get(track_id)
        if gt_track_id is None:
            continue
        gt_row = gt_rows.get(gt_track_id)
        if gt_row is None:
            continue

        gt_bbox = gt_row.bbox
        _, _, gt_width, gt_height = gt_bbox
        area_px = max(0.0, float(gt_width) * float(gt_height))
        top_candidates = tuple(diagnostic.get("head_top_candidates") or ())
        top_iou_rows: List[Tuple[int, float, str, float]] = []
        for candidate in top_candidates:
            candidate_bbox = coerce_bbox(getattr(candidate, "bbox", None))
            if candidate_bbox is None:
                continue
            iou = exercise.bbox_iou(candidate_bbox, gt_bbox)
            top_iou_rows.append(
                (
                    int(getattr(candidate, "rank", 0) or 0),
                    iou,
                    bbox_text(candidate_bbox),
                    float(getattr(candidate, "confidence", 0.0) or 0.0),
                )
            )
        best_top = max(top_iou_rows, key=lambda item: item[1], default=(0, 0.0, "", 0.0))
        final_bbox = coerce_bbox(diagnostic.get("final_bbox"))
        if final_bbox is None and track_id in track_by_id:
            final_bbox = track_by_id[track_id].bbox
        final_iou = exercise.bbox_iou(final_bbox, gt_bbox) if final_bbox is not None else None
        final_best_other_iou = (
            max((exercise.bbox_iou(final_bbox, other.bbox) for other in rows if other.track_id != gt_track_id), default=0.0)
            if final_bbox is not None
            else None
        )
        final_correct_object = (
            final_iou is not None
            and final_iou >= 0.30
            and (final_best_other_iou is None or final_iou + 0.05 >= final_best_other_iou)
        )
        head_iou = optional_iou(diagnostic.get("head_bbox"), gt_bbox)
        head_center_error = optional_center_error(diagnostic.get("head_bbox"), gt_bbox)
        head_area_ratio = optional_area_ratio(diagnostic.get("head_bbox"), gt_bbox)
        template_iou = optional_iou(diagnostic.get("template_bbox"), gt_bbox)
        template_center_error = optional_center_error(diagnostic.get("template_bbox"), gt_bbox)
        template_area_ratio = optional_area_ratio(diagnostic.get("template_bbox"), gt_bbox)
        fused_iou = optional_iou(diagnostic.get("fused_bbox"), gt_bbox)
        fused_area_ratio = optional_area_ratio(diagnostic.get("fused_bbox"), gt_bbox)
        assigned_iou = optional_iou(diagnostic.get("assigned_bbox"), gt_bbox)
        final_center_error = optional_center_error(final_bbox, gt_bbox)
        final_center_error_norm = optional_center_error_gt_size(final_bbox, gt_bbox)
        final_area_ratio = optional_area_ratio(final_bbox, gt_bbox)
        failure_bucket, failure_stage = classify_iou_failure(
            final_iou,
            final_best_other_iou,
            head_iou,
            best_top[1],
            template_iou,
            fused_iou,
            assigned_iou,
            final_center_error_norm,
            final_area_ratio,
        )

        output_rows.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "execution_mode": EXECUTION_MODE,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "actual_tracks": actual_tracks,
                "init_frame": init_frame,
                "frame": frame_number,
                "sample_offset": frame_number - init_frame,
                "tracker_id": track_id,
                "gt_track_id": gt_track_id,
                "gt_area_px": area_px,
                "gt_bbox": bbox_text(gt_bbox),
                "previous_bbox": bbox_text(diagnostic.get("previous_bbox")),
                "predicted_bbox": bbox_text(diagnostic.get("predicted_bbox")),
                "head_bbox": bbox_text(diagnostic.get("head_bbox")),
                "head_confidence": diagnostic.get("head_confidence"),
                "head_margin": diagnostic.get("head_margin"),
                "head_roi_tokens": diagnostic.get("head_roi_tokens"),
                "head_iou": head_iou,
                "head_center_error_px": head_center_error,
                "head_area_ratio": head_area_ratio,
                "head_top5_count": len(top_iou_rows),
                "head_top5_best_iou": best_top[1],
                "head_top5_best_rank": best_top[0],
                "head_top5_best_confidence": best_top[3],
                "head_top5_best_bbox": best_top[2],
                "head_top5_iou50": best_top[1] >= 0.50,
                "head_top5_iou30": best_top[1] >= 0.30,
                "template_attempted": diagnostic.get("template_attempted"),
                "template_bbox": bbox_text(diagnostic.get("template_bbox")),
                "template_confidence": diagnostic.get("template_confidence"),
                "template_iou": template_iou,
                "template_center_error_px": template_center_error,
                "template_area_ratio": template_area_ratio,
                "fused_bbox": bbox_text(diagnostic.get("fused_bbox")),
                "fused_confidence": diagnostic.get("fused_confidence"),
                "fused_margin": diagnostic.get("fused_margin"),
                "fused_iou": fused_iou,
                "fused_area_ratio": fused_area_ratio,
                "candidate_source": diagnostic.get("candidate_source"),
                "assigned_source_track_id": diagnostic.get("assigned_source_track_id"),
                "assigned_bbox": bbox_text(diagnostic.get("assigned_bbox")),
                "assigned_confidence": diagnostic.get("assigned_confidence"),
                "assignment_score": diagnostic.get("assignment_score"),
                "assignment_margin": diagnostic.get("assignment_margin"),
                "assigned_iou": assigned_iou,
                "reject_state": diagnostic.get("reject_state"),
                "accepted": diagnostic.get("accepted"),
                "held": diagnostic.get("held"),
                "final_bbox": bbox_text(final_bbox),
                "final_confidence": diagnostic.get("final_confidence"),
                "final_iou": final_iou,
                "final_center_error_px": final_center_error,
                "final_center_error_gt_size": final_center_error_norm,
                "final_area_ratio": final_area_ratio,
                "final_best_other_iou": final_best_other_iou,
                "final_correct_object": final_correct_object,
                "final_state": diagnostic.get("final_state"),
                "assigned_source": diagnostic.get("assigned_source"),
                "iou_failure_bucket": failure_bucket,
                "iou_failure_stage": failure_stage,
            }
        )
    return output_rows


def summarize_identity(identity_rows: Sequence[IdentityObservation]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, int], List[IdentityObservation]] = {}
    for row in identity_rows:
        key = (row.sequence, row.lorat_config, row.execution_mode, row.reid_mode, row.target_tracks)
        grouped.setdefault(key, []).append(row)
    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, execution_mode, reid_mode, target_tracks), rows in sorted(grouped.items()):
        visible_rows = [row for row in rows if row.gt_visible]
        summary_rows = visible_rows or list(rows)
        switch_count = sum(1 for row in summary_rows if row.identity_switch)
        lost_count = sum(1 for row in summary_rows if row.track_lost)
        summaries.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "execution_mode": execution_mode,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "samples": len(rows),
                "visible_samples": len(visible_rows),
                "hidden_samples": len(rows) - len(visible_rows),
                "correct_rate": sum(1 for row in summary_rows if row.correct_object) / len(summary_rows) if summary_rows else None,
                "jump_rate": sum(1 for row in summary_rows if row.identity_jump) / len(summary_rows) if summary_rows else None,
                "identity_switches": switch_count,
                "identity_switches_per_1000_samples": (switch_count * 1000.0 / len(summary_rows)) if summary_rows else None,
                "track_loss_rate": (lost_count / len(summary_rows)) if summary_rows else None,
                "occluded_rate": sum(1 for row in rows if row.occluded) / len(rows) if rows else None,
                "mean_iou": statistics.fmean(row.own_iou for row in summary_rows) if summary_rows else None,
            }
        )
    return summaries


def summarize_occlusion_survival(identity_rows: Sequence[IdentityObservation]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int, int], List[IdentityObservation]] = {}
    for row in identity_rows:
        key = (row.sequence, row.lorat_config, row.reid_mode, row.target_tracks, row.tracker_id)
        grouped.setdefault(key, []).append(row)
    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, reid_mode, target_tracks, tracker_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row.frame)
        longest_occluded_frames = 0
        longest_survived_frames = 0
        active_start: Optional[int] = None
        previous_frame: Optional[int] = None
        recovered_after_gap = False
        lost_after_gap = False
        for row in ordered:
            in_gap = row.occluded or row.track_lost or row.lifecycle_state in (mot.TrackLifecycle.UNCERTAIN, mot.TrackLifecycle.LOST)
            if in_gap and active_start is None:
                active_start = row.frame
            if not in_gap and active_start is not None:
                duration = max(0, (previous_frame or row.frame) - active_start + 1)
                longest_occluded_frames = max(longest_occluded_frames, duration)
                if row.correct_object:
                    longest_survived_frames = max(longest_survived_frames, duration)
                    recovered_after_gap = True
                active_start = None
            if row.track_lost:
                lost_after_gap = True
            previous_frame = row.frame
        if active_start is not None and previous_frame is not None:
            longest_occluded_frames = max(longest_occluded_frames, max(0, previous_frame - active_start + 1))
        summaries.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "tracker_id": tracker_id,
                "samples": len(ordered),
                "longest_occluded_frames_observed": longest_occluded_frames,
                "longest_occlusion_survived_frames": longest_survived_frames,
                "recovered_after_gap": int(recovered_after_gap),
                "lost_after_gap": int(lost_after_gap),
                "final_lifecycle_state": ordered[-1].lifecycle_state if ordered else "",
            }
        )
    return summaries


def parse_week2_proof_lines(lines: Sequence[str]) -> List[Dict[str, str]]:
    if not lines:
        return []
    text = v8.WEEK2_PROOF_LOG_HEADER + "".join(lines)
    return list(csv.DictReader(io.StringIO(text)))


def summarize_week2_proof(lines: Sequence[str]) -> Dict[str, object]:
    rows = parse_week2_proof_lines(lines)
    track_rows = [row for row in rows if row.get("phase") == "track"]
    shared_ok = sum(1 for row in track_rows if row.get("week2_shared_backbone_ok") == "1")
    head_ok = sum(1 for row in track_rows if row.get("week2_batched_head_ok") == "1")
    head_modes = sorted({row.get("head_mode", "") for row in track_rows if row.get("head_mode")})

    def mean_float(field: str) -> Optional[float]:
        values = []
        for row in track_rows:
            value = row.get(field, "")
            if value == "":
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        return statistics.fmean(values) if values else None

    return {
        "track_frames": len(track_rows),
        "shared_ok_frames": shared_ok,
        "head_ok_frames": head_ok,
        "shared_ok_rate": (shared_ok / len(track_rows)) if track_rows else None,
        "head_ok_rate": (head_ok / len(track_rows)) if track_rows else None,
        "head_mode": ",".join(head_modes),
        "profile_unbucketed_ms_per_update": mean_float("profile_unbucketed_ms"),
    }


def append_week2_proof_rows(
    output_rows: List[Dict[str, object]],
    sequence: str,
    lorat_config: str,
    reid_mode: str,
    target_tracks: int,
    actual_tracks: int,
    lines: Sequence[str],
) -> None:
    for row in parse_week2_proof_lines(lines):
        full_row: Dict[str, object] = {
            "sequence": sequence,
            "lorat_config": lorat_config,
            "reid_mode": reid_mode,
            "target_tracks": target_tracks,
            "actual_tracks": actual_tracks,
        }
        full_row.update(row)
        output_rows.append(full_row)


def record_debug(rows: List[Dict[str, object]], path: Path, event: str, **values: object) -> None:
    row: Dict[str, object] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event}
    row.update(values)
    rows.append(row)
    write_debug_csv(path, rows)


def optional_ratio(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def profile_ms_per_update(runtime_status: mot.RuntimeStatus, bucket: str) -> Optional[float]:
    value = getattr(runtime_status, f"v8_profile_{bucket}_ms_per_update", None)
    return None if value is None else float(value)


def run_benchmark_case(
    args: argparse.Namespace,
    sequence_path: Path,
    lorat_config: str,
    target_tracks: int,
    preview_path: Optional[Path],
    debug_rows: List[Dict[str, object]],
    proof_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    debug_csv: Path,
) -> Optional[Tuple[V8TimingResult, List[bench.AreaObservation], List[bench.AreaObservation], List[IdentityObservation]]]:
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
        reason = f"only {len(init_rows)} usable GT tracks at init frame {init_frame}"
        print(f"Skipping {sequence_path.name} {lorat_config} N={target_tracks}: {reason}.", flush=True)
        record_debug(
            debug_rows,
            debug_csv,
            "case_skip",
            sequence=sequence_path.name,
            lorat_config=lorat_config,
            target_tracks=target_tracks,
            reason=reason,
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
    weight_path = args.weight_path or mot.LORAT_WEIGHT_BY_CONFIG[lorat_config]
    checkpoint_mb = weight_path.stat().st_size / (1024 * 1024) if weight_path.exists() else 0.0
    backbone, input_size = exercise.lorat_config_metadata(lorat_config)
    reid_mode = str(getattr(args, "reid_mode", "reid_off" if getattr(args, "disable_identity_arbitration", False) else "reid_on"))
    run_args = tracker_args_for_run(args, lorat_config, target_tracks)
    head_weight_path = run_args.v8_head_weights
    source = SimpleNamespace(name=sequence_path.name, fps=fps, length=sequence_length or len(image_paths))
    backend = v8.create_backend(run_args, source, target_tracks)
    preview_writer = None
    sampled_area: List[bench.AreaObservation] = []
    full_area: List[bench.AreaObservation] = []
    identity_rows: List[IdentityObservation] = []
    previous_identity_bboxes: Dict[int, mot.BBox] = {}
    metrics = {"count": 0.0, "iou_sum": 0.0, "hit50": 0.0}
    last_frame_number = init_frame
    tracking_seconds = 0.0

    record_debug(
        debug_rows,
        debug_csv,
        "case_start",
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        reid_mode=reid_mode,
        target_tracks=target_tracks,
        frames_expected=total_frames_expected,
        init_frame=init_frame,
        area_sample_interval=args.area_sample_interval,
        identity_sample_interval=args.identity_sample_interval,
        v8_head_weights=head_weight_path,
    )
    print(
        f"Starting v8 {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"{total_frames_expected} frames, area sample every {args.area_sample_interval}, "
        f"reid={reid_mode}, "
        f"head_weights={head_weight_path if head_weight_path is not None else 'none'}, "
        f"video={preview_path if args.save_video else 'disabled'}",
        flush=True,
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
        area_rows, full_rows, id_rows = collect_sampled_observations(
            sequence_path.name,
            lorat_config,
            reid_mode,
            target_tracks,
            actual_tracks,
            init_frame,
            init_frame,
            backend.tracks,
            gt_by_frame,
            tracker_to_gt_id,
            args.min_visibility,
            args.area_sample_interval,
            args.identity_sample_interval,
            args.identity_correct_iou,
            args.identity_competitor_margin,
            args.identity_jump_factor,
            previous_identity_bboxes,
            args.full_area_observations,
        )
        sampled_area.extend(area_rows)
        full_area.extend(full_rows)
        identity_rows.extend(id_rows)
        exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, init_frame, tracker_to_gt_id, args.min_visibility)

        if args.save_video and preview_path is not None:
            preview_writer = mot.make_video_writer(preview_path, fps, init_frame_image)
            preview_writer.write(mot.draw_tracks(init_frame_image, backend.tracks, init_frame, backend.backend_name))

        print(
            f"[v8 {sequence_path.name} {lorat_config} N={target_tracks}] "
            f"initialized {actual_tracks} tracks at source frame {init_frame}",
            flush=True,
        )

        for image_index in range(init_index + 1, end_index + 1):
            if STOP_REQUESTED:
                print("Stop requested before the next frame; saving partial case.", flush=True)
                break
            frame_number = image_index + 1
            frame = cv2.imread(str(image_paths[image_index]))
            if frame is None:
                print(f"Skipping unreadable frame: {image_paths[image_index]}", flush=True)
                continue
            update_started = time.perf_counter()
            backend.update(frame, frame_number)
            tracking_seconds += time.perf_counter() - update_started
            last_frame_number = frame_number
            area_rows, full_rows, id_rows = collect_sampled_observations(
                sequence_path.name,
                lorat_config,
                reid_mode,
                target_tracks,
                actual_tracks,
                init_frame,
                frame_number,
                backend.tracks,
                gt_by_frame,
                tracker_to_gt_id,
                args.min_visibility,
                args.area_sample_interval,
                args.identity_sample_interval,
                args.identity_correct_iou,
                args.identity_competitor_margin,
                args.identity_jump_factor,
                previous_identity_bboxes,
                args.full_area_observations,
            )
            sampled_area.extend(area_rows)
            full_area.extend(full_rows)
            identity_rows.extend(id_rows)
            candidate_rows.extend(
                collect_candidate_diagnostics(
                    sequence_path.name,
                    lorat_config,
                    reid_mode,
                    target_tracks,
                    actual_tracks,
                    init_frame,
                    frame_number,
                    backend.tracks,
                    getattr(backend, "last_candidate_diagnostics", ()),
                    gt_by_frame,
                    tracker_to_gt_id,
                    args.min_visibility,
                )
            )
            exercise.update_overlap_metrics(metrics, backend.tracks, gt_by_frame, frame_number, tracker_to_gt_id, args.min_visibility)
            if preview_writer is not None:
                status = backend.runtime_status_snapshot()
                status_lines = [
                    f"FPS {status.fps:.2f} | objects {status.active_objects} | {EXECUTION_MODE}",
                    f"shared ViT {status.shared_frame_backbone_calls} | head batches {status.object_head_batches} | head items {status.object_head_items}",
                    f"GPU peak reserved {bench.optional_float(status.gpu_peak_reserved_mb, 1)} MB",
                ]
                preview_writer.write(mot.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, status_lines))
            processed_frames = image_index - init_index + 1
            if args.progress_interval > 0 and (
                processed_frames == total_frames_expected or processed_frames % args.progress_interval == 0
            ):
                elapsed = time.perf_counter() - total_started
                print(
                    f"[v8 {sequence_path.name} {lorat_config} N={target_tracks}] "
                    f"frame {processed_frames}/{total_frames_expected} "
                    f"(source frame {frame_number}), elapsed {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        runtime_status = backend.runtime_status_snapshot()
        proof_summary = summarize_week2_proof(backend.week2_proof_lines)
        append_week2_proof_rows(
            proof_rows,
            sequence_path.name,
            lorat_config,
            reid_mode,
            target_tracks,
            len(backend.tracks),
            backend.week2_proof_lines,
        )
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    total_seconds = time.perf_counter() - total_started
    frames = max(1, last_frame_number - init_frame + 1)
    update_frames = max(0, frames - 1)
    actual_tracks = len(backend.tracks)
    boxes_total = frames * actual_tracks
    boxes_tracking = update_frames * actual_tracks
    fps_total = frames / total_seconds if total_seconds > 0 else 0.0
    fps_tracking = update_frames / tracking_seconds if tracking_seconds > 0 and update_frames else None
    total_ms_per_bbox = (total_seconds * 1000.0 / boxes_total) if boxes_total else None
    tracking_ms_per_bbox = (tracking_seconds * 1000.0 / boxes_tracking) if boxes_tracking else None
    mean_iou = metrics["iou_sum"] / metrics["count"] if metrics["count"] else None
    iou50 = metrics["hit50"] / metrics["count"] if metrics["count"] else None
    shared_calls_per_frame = optional_ratio(runtime_status.shared_frame_backbone_calls, frames)
    object_batches_per_update = optional_ratio(runtime_status.object_head_batches, update_frames)
    object_items_per_update = optional_ratio(runtime_status.object_head_items, update_frames)
    object_items_per_bbox = optional_ratio(runtime_status.object_head_items, boxes_tracking)
    selected_head_items = runtime_status.gating_selected_slot_items
    selected_items_per_update = optional_ratio(selected_head_items, update_frames)

    timing = V8TimingResult(
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        backbone=backbone,
        input_size=input_size,
        device=args.device,
        reid_mode=reid_mode,
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
        execution_mode=EXECUTION_MODE,
        head_mode=str(proof_summary.get("head_mode", "")),
        gpu_profile=args.gpu_profile,
        gpu_name=runtime_status.gpu_name,
        gpu_memory_allocated_mb=runtime_status.gpu_allocated_mb,
        gpu_memory_reserved_mb=runtime_status.gpu_reserved_mb,
        gpu_memory_peak_allocated_mb=runtime_status.gpu_peak_allocated_mb,
        gpu_memory_peak_reserved_mb=runtime_status.gpu_peak_reserved_mb,
        shared_frame_backbone_calls=runtime_status.shared_frame_backbone_calls,
        shared_frame_backbone_items=runtime_status.shared_frame_backbone_items,
        shared_backbone_calls_per_frame=shared_calls_per_frame,
        object_head_batches=runtime_status.object_head_batches,
        object_head_items=runtime_status.object_head_items,
        max_object_head_batch=runtime_status.max_object_head_batch,
        object_head_roi_tokens=runtime_status.object_head_roi_tokens,
        object_head_batches_per_update_frame=object_batches_per_update,
        object_head_items_per_update_frame=object_items_per_update,
        object_head_items_per_bbox=object_items_per_bbox,
        selected_head_items=selected_head_items,
        selected_head_items_per_update_frame=selected_items_per_update,
        profile_candidate_transfer_ms_per_update=profile_ms_per_update(runtime_status, "candidate_transfer"),
        profile_candidate_extract_ms_per_update=profile_ms_per_update(runtime_status, "candidate_extract"),
        profile_template_match_ms_per_update=profile_ms_per_update(runtime_status, "template_match"),
        profile_candidate_fusion_ms_per_update=profile_ms_per_update(runtime_status, "candidate_fusion"),
        profile_reid_appearance_ms_per_update=profile_ms_per_update(runtime_status, "reid_appearance"),
        profile_identity_resolve_ms_per_update=profile_ms_per_update(runtime_status, "identity_resolve"),
        profile_identity_score_ms_per_update=profile_ms_per_update(runtime_status, "identity_score"),
        profile_debug_output_ms_per_update=profile_ms_per_update(runtime_status, "debug_output"),
        profile_accept_ms_per_update=profile_ms_per_update(runtime_status, "accept"),
        profile_hold_ms_per_update=profile_ms_per_update(runtime_status, "hold"),
        profile_appearance_refresh_ms_per_update=profile_ms_per_update(runtime_status, "appearance_refresh"),
        profile_proof_output_ms_per_update=profile_ms_per_update(runtime_status, "proof_output"),
        profile_unbucketed_ms_per_update=proof_summary.get("profile_unbucketed_ms_per_update"),  # type: ignore[arg-type]
        proof_track_frames=int(proof_summary["track_frames"]),
        proof_shared_backbone_ok_frames=int(proof_summary["shared_ok_frames"]),
        proof_batched_head_ok_frames=int(proof_summary["head_ok_frames"]),
        proof_shared_backbone_ok_rate=proof_summary["shared_ok_rate"],  # type: ignore[arg-type]
        proof_batched_head_ok_rate=proof_summary["head_ok_rate"],  # type: ignore[arg-type]
        fps_sustains_25=(fps_tracking is not None and fps_tracking >= args.fps_threshold),
        preview_path=preview_path if args.save_video else None,
    )
    record_debug(
        debug_rows,
        debug_csv,
        "case_complete",
        sequence=sequence_path.name,
        lorat_config=lorat_config,
        reid_mode=reid_mode,
        target_tracks=target_tracks,
        actual_tracks=actual_tracks,
        frames=frames,
        tracking_seconds=f"{tracking_seconds:.6f}",
        fps_tracking=bench.optional_float(fps_tracking),
        mean_iou=bench.optional_float(mean_iou),
        iou50=bench.optional_float(iou50),
        shared_backbone_calls=runtime_status.shared_frame_backbone_calls,
        object_head_batches=runtime_status.object_head_batches,
        object_head_items=runtime_status.object_head_items,
        profile_candidate_transfer_ms_per_update=bench.optional_float(timing.profile_candidate_transfer_ms_per_update),
        profile_candidate_extract_ms_per_update=bench.optional_float(timing.profile_candidate_extract_ms_per_update),
        profile_template_match_ms_per_update=bench.optional_float(timing.profile_template_match_ms_per_update),
        profile_candidate_fusion_ms_per_update=bench.optional_float(timing.profile_candidate_fusion_ms_per_update),
        profile_reid_appearance_ms_per_update=bench.optional_float(timing.profile_reid_appearance_ms_per_update),
        profile_identity_resolve_ms_per_update=bench.optional_float(timing.profile_identity_resolve_ms_per_update),
        profile_identity_score_ms_per_update=bench.optional_float(timing.profile_identity_score_ms_per_update),
        profile_debug_output_ms_per_update=bench.optional_float(timing.profile_debug_output_ms_per_update),
        profile_accept_ms_per_update=bench.optional_float(timing.profile_accept_ms_per_update),
        profile_hold_ms_per_update=bench.optional_float(timing.profile_hold_ms_per_update),
        profile_appearance_refresh_ms_per_update=bench.optional_float(timing.profile_appearance_refresh_ms_per_update),
        profile_proof_output_ms_per_update=bench.optional_float(timing.profile_proof_output_ms_per_update),
        profile_unbucketed_ms_per_update=bench.optional_float(timing.profile_unbucketed_ms_per_update),
        proof_shared_ok_rate=bench.optional_float(timing.proof_shared_backbone_ok_rate),
        proof_head_ok_rate=bench.optional_float(timing.proof_batched_head_ok_rate),
    )
    print(
        f"v8 {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"actual={actual_tracks}, frames={frames}, "
        f"fps={bench.optional_float(fps_tracking, 3)}, "
        f"track_ms_per_box={bench.optional_float(tracking_ms_per_bbox, 3)}, "
        f"iou50={bench.optional_float(iou50, 3)}, "
        f"shared_calls/frame={bench.optional_float(shared_calls_per_frame, 3)}, "
        f"head_items/update={bench.optional_float(object_items_per_update, 3)}",
        flush=True,
    )
    return timing, sampled_area, full_area, identity_rows


def write_timing_csv(path: Path, rows: Sequence[V8TimingResult]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "backbone",
        "input_size",
        "device",
        "reid_mode",
        "execution_mode",
        "head_mode",
        "gpu_profile",
        "gpu_name",
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
        "gpu_memory_allocated_mb",
        "gpu_memory_reserved_mb",
        "gpu_memory_peak_allocated_mb",
        "gpu_memory_peak_reserved_mb",
        "shared_frame_backbone_calls",
        "shared_frame_backbone_items",
        "shared_backbone_calls_per_frame",
        "object_head_batches",
        "object_head_items",
        "max_object_head_batch",
        "object_head_roi_tokens",
        "object_head_batches_per_update_frame",
        "object_head_items_per_update_frame",
        "object_head_items_per_bbox",
        "selected_head_items",
        "selected_head_items_per_update_frame",
        "profile_candidate_transfer_ms_per_update",
        "profile_candidate_extract_ms_per_update",
        "profile_template_match_ms_per_update",
        "profile_candidate_fusion_ms_per_update",
        "profile_reid_appearance_ms_per_update",
        "profile_identity_resolve_ms_per_update",
        "profile_identity_score_ms_per_update",
        "profile_debug_output_ms_per_update",
        "profile_accept_ms_per_update",
        "profile_hold_ms_per_update",
        "profile_appearance_refresh_ms_per_update",
        "profile_proof_output_ms_per_update",
        "profile_unbucketed_ms_per_update",
        "proof_track_frames",
        "proof_shared_backbone_ok_frames",
        "proof_batched_head_ok_frames",
        "proof_shared_backbone_ok_rate",
        "proof_batched_head_ok_rate",
        "fps_sustains_25",
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
                    "reid_mode": row.reid_mode,
                    "execution_mode": row.execution_mode,
                    "head_mode": row.head_mode,
                    "gpu_profile": row.gpu_profile,
                    "gpu_name": row.gpu_name,
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
                    "fps_tracking": bench.optional_float(row.fps_tracking),
                    "total_ms_per_bbox": bench.optional_float(row.total_ms_per_bbox),
                    "tracking_ms_per_bbox": bench.optional_float(row.tracking_ms_per_bbox),
                    "mean_iou": bench.optional_float(row.mean_iou),
                    "iou50": bench.optional_float(row.iou50),
                    "gpu_memory_allocated_mb": bench.optional_float(row.gpu_memory_allocated_mb, 2),
                    "gpu_memory_reserved_mb": bench.optional_float(row.gpu_memory_reserved_mb, 2),
                    "gpu_memory_peak_allocated_mb": bench.optional_float(row.gpu_memory_peak_allocated_mb, 2),
                    "gpu_memory_peak_reserved_mb": bench.optional_float(row.gpu_memory_peak_reserved_mb, 2),
                    "shared_frame_backbone_calls": row.shared_frame_backbone_calls,
                    "shared_frame_backbone_items": row.shared_frame_backbone_items,
                    "shared_backbone_calls_per_frame": bench.optional_float(row.shared_backbone_calls_per_frame),
                    "object_head_batches": row.object_head_batches,
                    "object_head_items": row.object_head_items,
                    "max_object_head_batch": row.max_object_head_batch,
                    "object_head_roi_tokens": row.object_head_roi_tokens,
                    "object_head_batches_per_update_frame": bench.optional_float(row.object_head_batches_per_update_frame),
                    "object_head_items_per_update_frame": bench.optional_float(row.object_head_items_per_update_frame),
                    "object_head_items_per_bbox": bench.optional_float(row.object_head_items_per_bbox),
                    "selected_head_items": row.selected_head_items,
                    "selected_head_items_per_update_frame": bench.optional_float(row.selected_head_items_per_update_frame),
                    "profile_candidate_transfer_ms_per_update": bench.optional_float(row.profile_candidate_transfer_ms_per_update),
                    "profile_candidate_extract_ms_per_update": bench.optional_float(row.profile_candidate_extract_ms_per_update),
                    "profile_template_match_ms_per_update": bench.optional_float(row.profile_template_match_ms_per_update),
                    "profile_candidate_fusion_ms_per_update": bench.optional_float(row.profile_candidate_fusion_ms_per_update),
                    "profile_reid_appearance_ms_per_update": bench.optional_float(row.profile_reid_appearance_ms_per_update),
                    "profile_identity_resolve_ms_per_update": bench.optional_float(row.profile_identity_resolve_ms_per_update),
                    "profile_identity_score_ms_per_update": bench.optional_float(row.profile_identity_score_ms_per_update),
                    "profile_debug_output_ms_per_update": bench.optional_float(row.profile_debug_output_ms_per_update),
                    "profile_accept_ms_per_update": bench.optional_float(row.profile_accept_ms_per_update),
                    "profile_hold_ms_per_update": bench.optional_float(row.profile_hold_ms_per_update),
                    "profile_appearance_refresh_ms_per_update": bench.optional_float(row.profile_appearance_refresh_ms_per_update),
                    "profile_proof_output_ms_per_update": bench.optional_float(row.profile_proof_output_ms_per_update),
                    "profile_unbucketed_ms_per_update": bench.optional_float(row.profile_unbucketed_ms_per_update),
                    "proof_track_frames": row.proof_track_frames,
                    "proof_shared_backbone_ok_frames": row.proof_shared_backbone_ok_frames,
                    "proof_batched_head_ok_frames": row.proof_batched_head_ok_frames,
                    "proof_shared_backbone_ok_rate": bench.optional_float(row.proof_shared_backbone_ok_rate),
                    "proof_batched_head_ok_rate": bench.optional_float(row.proof_batched_head_ok_rate),
                    "fps_sustains_25": "" if row.fps_sustains_25 is None else str(row.fps_sustains_25),
                    "preview_path": str(row.preview_path or ""),
                }
            )


def write_identity_csv(path: Path, rows: Sequence[IdentityObservation]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "execution_mode",
        "reid_mode",
        "target_tracks",
        "actual_tracks",
        "init_frame",
        "frame",
        "sample_offset",
        "tracker_id",
        "gt_track_id",
        "gt_visible",
        "gt_visibility",
        "matched_gt_id",
        "matched_gt_iou",
        "area_px",
        "own_iou",
        "best_other_iou",
        "correct_object",
        "identity_jump",
        "identity_switch",
        "track_lost",
        "center_jump_px",
        "occluded",
        "ok",
        "lost_frames",
        "lifecycle_state",
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
                    "execution_mode": row.execution_mode,
                    "reid_mode": row.reid_mode,
                    "target_tracks": row.target_tracks,
                    "actual_tracks": row.actual_tracks,
                    "init_frame": row.init_frame,
                    "frame": row.frame,
                    "sample_offset": row.sample_offset,
                    "tracker_id": row.tracker_id,
                    "gt_track_id": row.gt_track_id,
                    "gt_visible": int(row.gt_visible),
                    "gt_visibility": f"{row.gt_visibility:.6f}",
                    "matched_gt_id": "" if row.matched_gt_id is None else row.matched_gt_id,
                    "matched_gt_iou": f"{row.matched_gt_iou:.6f}",
                    "area_px": f"{row.area_px:.6f}",
                    "own_iou": f"{row.own_iou:.6f}",
                    "best_other_iou": f"{row.best_other_iou:.6f}",
                    "correct_object": int(row.correct_object),
                    "identity_jump": int(row.identity_jump),
                    "identity_switch": int(row.identity_switch),
                    "track_lost": int(row.track_lost),
                    "center_jump_px": f"{row.center_jump_px:.3f}",
                    "occluded": int(row.occluded),
                    "ok": int(row.ok),
                    "lost_frames": row.lost_frames,
                    "lifecycle_state": row.lifecycle_state,
                    "state": row.state,
                }
            )


def write_dict_rows_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_week2_proof_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    proof_fields = next(csv.reader([v8.WEEK2_PROOF_LOG_HEADER.strip()]))
    fieldnames = ["sequence", "lorat_config", "reid_mode", "target_tracks", "actual_tracks"] + proof_fields
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_candidate_diagnostics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "sequence",
        "lorat_config",
        "execution_mode",
        "reid_mode",
        "target_tracks",
        "actual_tracks",
        "init_frame",
        "frame",
        "sample_offset",
        "tracker_id",
        "gt_track_id",
        "gt_area_px",
        "gt_bbox",
        "previous_bbox",
        "predicted_bbox",
        "head_bbox",
        "head_confidence",
        "head_margin",
        "head_roi_tokens",
        "head_iou",
        "head_center_error_px",
        "head_area_ratio",
        "head_top5_count",
        "head_top5_best_iou",
        "head_top5_best_rank",
        "head_top5_best_confidence",
        "head_top5_best_bbox",
        "head_top5_iou50",
        "head_top5_iou30",
        "template_attempted",
        "template_bbox",
        "template_confidence",
        "template_iou",
        "template_center_error_px",
        "template_area_ratio",
        "fused_bbox",
        "fused_confidence",
        "fused_margin",
        "fused_iou",
        "fused_area_ratio",
        "candidate_source",
        "assigned_source_track_id",
        "assigned_bbox",
        "assigned_confidence",
        "assignment_score",
        "assignment_margin",
        "assigned_iou",
        "reject_state",
        "accepted",
        "held",
        "final_bbox",
        "final_confidence",
        "final_iou",
        "final_center_error_px",
        "final_center_error_gt_size",
        "final_area_ratio",
        "final_best_other_iou",
        "final_correct_object",
        "final_state",
        "assigned_source",
        "iou_failure_bucket",
        "iou_failure_stage",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_debug_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "time",
        "event",
        "sequence",
        "lorat_config",
        "target_tracks",
        "actual_tracks",
        "frames_expected",
        "frames",
        "init_frame",
        "area_sample_interval",
        "identity_sample_interval",
        "tracking_seconds",
        "fps_tracking",
        "mean_iou",
        "iou50",
        "shared_backbone_calls",
        "object_head_batches",
        "object_head_items",
        "profile_candidate_transfer_ms_per_update",
        "profile_candidate_extract_ms_per_update",
        "profile_template_match_ms_per_update",
        "profile_candidate_fusion_ms_per_update",
        "profile_reid_appearance_ms_per_update",
        "profile_identity_resolve_ms_per_update",
        "profile_identity_score_ms_per_update",
        "profile_debug_output_ms_per_update",
        "profile_accept_ms_per_update",
        "profile_hold_ms_per_update",
        "profile_appearance_refresh_ms_per_update",
        "profile_proof_output_ms_per_update",
        "profile_unbucketed_ms_per_update",
        "proof_shared_ok_rate",
        "proof_head_ok_rate",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reliable_floor(rows: Sequence[bench.AreaSummary]) -> Dict[Tuple[str, str, str, int], Optional[float]]:
    return bench.smallest_reliable_area(rows)


def capacity_rows(timing_rows: Sequence[V8TimingResult]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[V8TimingResult]] = {}
    for row in timing_rows:
        grouped.setdefault((row.sequence, row.lorat_config, row.reid_mode, row.gpu_profile), []).append(row)
    result: List[Dict[str, object]] = []
    for (sequence, lorat_config, reid_mode, gpu_profile), rows in sorted(grouped.items()):
        sustaining = [row for row in rows if row.fps_sustains_25]
        max_n = max((row.actual_tracks for row in sustaining), default=None)
        result.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "reid_mode": reid_mode,
                "gpu_profile": gpu_profile,
                "max_n_sustaining_25_fps": max_n,
            }
        )
    return result


def mean_numeric(rows: Sequence[Dict[str, object]], field: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.fmean(values) if values else None


def true_rate(rows: Sequence[Dict[str, object]], field: str) -> Optional[float]:
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get(field))) / float(len(rows))


def summarize_candidate_diagnostics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("sequence", "")),
            str(row.get("lorat_config", "")),
            str(row.get("reid_mode", "")),
            int(row.get("target_tracks", 0) or 0),
        )
        grouped.setdefault(key, []).append(row)
    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, reid_mode, target_tracks), group_rows in sorted(grouped.items()):
        source_counts = Counter(str(row.get("candidate_source", "")) for row in group_rows)
        failure_counts = Counter(str(row.get("iou_failure_bucket", "")) for row in group_rows)
        stage_counts = Counter(str(row.get("iou_failure_stage", "")) for row in group_rows)
        summaries.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "samples": len(group_rows),
                "mean_head_iou": mean_numeric(group_rows, "head_iou"),
                "head_iou50": sum(1 for row in group_rows if float(row.get("head_iou") or 0.0) >= 0.50)
                / float(len(group_rows))
                if group_rows
                else None,
                "mean_top5_best_iou": mean_numeric(group_rows, "head_top5_best_iou"),
                "top5_iou50": true_rate(group_rows, "head_top5_iou50"),
                "top5_iou30": true_rate(group_rows, "head_top5_iou30"),
                "mean_template_iou": mean_numeric(group_rows, "template_iou"),
                "template_attempt_rate": true_rate(group_rows, "template_attempted"),
                "mean_fused_iou": mean_numeric(group_rows, "fused_iou"),
                "mean_assigned_iou": mean_numeric(group_rows, "assigned_iou"),
                "mean_final_iou": mean_numeric(group_rows, "final_iou"),
                "final_iou50": sum(1 for row in group_rows if float(row.get("final_iou") or 0.0) >= 0.50)
                / float(len(group_rows))
                if group_rows
                else None,
                "final_correct_rate": true_rate(group_rows, "final_correct_object"),
                "candidate_sources": ";".join(f"{source}:{count}" for source, count in sorted(source_counts.items())),
                "iou_failure_buckets": ";".join(f"{bucket}:{count}" for bucket, count in sorted(failure_counts.items())),
                "iou_failure_stages": ";".join(f"{stage}:{count}" for stage, count in sorted(stage_counts.items())),
            }
        )
    return summaries


def write_summary_md(
    path: Path,
    args: argparse.Namespace,
    label: str,
    paths: V8OutputPaths,
    timing_rows: Sequence[V8TimingResult],
    area_rows: Sequence[bench.AreaSummary],
    identity_rows: Sequence[IdentityObservation],
    candidate_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# LoRAT v8 Benchmark Summary",
        "",
        "## Run",
        "",
        f"- Run label: `{label}`",
        f"- Execution mode: `{EXECUTION_MODE}`",
        f"- Device: `{args.device}`",
        f"- GPU profile label: `{args.gpu_profile}`",
        f"- Max frames per case: `{args.max_frames if args.max_frames > 0 else 'full sequence'}`",
        f"- Track counts: `{','.join(str(value) for value in parse_track_counts(args))}`",
        "",
        "## Metric Definitions",
        "",
        "- Small-object reliability is sampled every "
        f"`{args.area_sample_interval}` frame(s). Each sampled tracker box is compared with the same object's current ground-truth box, not the first-frame box.",
        "- Pixel area is `ground_truth_width * ground_truth_height` for the current sampled frame.",
        "- IoU is `area(predicted_box intersection ground_truth_box) / area(predicted_box union ground_truth_box)`.",
        "- IoU@0.50 is the fraction of sampled boxes whose IoU is at least `0.50`.",
        f"- Reliable area-bin rule: IoU@0.50 >= `{args.reliable_iou50}`, mean IoU >= `{args.reliable_mean_iou}`, and samples >= `{args.min_area_samples}`.",
        f"- Every-`{args.identity_sample_interval}`-frame identity check: current GT IoU >= `{args.identity_correct_iou}` and no other visible GT beats it by more than `{args.identity_competitor_margin}`.",
        "- Identity switch counts a sampled frame where the tracker box matches a different visible GT object than the initialized GT object.",
        "- Track loss counts a sampled frame where the track is marked lost/not-ok or no visible GT object reaches the identity IoU threshold.",
        "- Occlusion survival reports the longest sampled uncertain/occluded gap that later returns to a correct object match.",
        f"- 25 FPS capacity rule: maximum actual N with tracking FPS >= `{args.fps_threshold}`.",
        "- Week 2 proof columns show one shared frame-backbone call per tracked frame and one batched object-head operation whose item count scales with object count.",
        "- v8 FPS profile values are elapsed milliseconds per tracked update frame, aggregated from tracker-side timers.",
        "",
        "## Output Files",
        "",
        f"- Timing and memory: `{paths.timing_csv}`",
        f"- Area reliability: `{paths.area_csv}`",
        f"- Sampled area observations: `{paths.observations_csv}`",
        f"- Identity observations: `{paths.identity_csv}`",
        f"- Identity/ReID summary: `{paths.identity_summary_csv}`",
        f"- Occlusion survival: `{paths.occlusion_survival_csv}`",
        f"- Week 2 shared-backbone proof: `{paths.week2_proof_csv}`",
        f"- V8 candidate/oracle diagnostics: `{paths.candidate_diagnostics_csv}`",
        f"- Debug log: `{paths.debug_csv}`",
    ]
    if args.full_area_observations:
        lines.append(f"- Every-frame area observations: `{paths.full_observations_csv}`")
    lines.extend(["", "## Timing / Memory", ""])
    lines.append(
        "| Sequence | Config | ReID Mode | N Target | N Actual | FPS Track | Track ms/box | Mean IoU | IoU@0.50 | Peak GPU Reserved MB | Shared Calls/Frame | Head Items/Update | Week2 Shared OK | Week2 Head OK | 25 FPS |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in timing_rows:
        lines.append(
            f"| {row.sequence} | {row.lorat_config} | {row.reid_mode} | {row.target_tracks} | {row.actual_tracks} | "
            f"{bench.optional_float(row.fps_tracking, 3)} | {bench.optional_float(row.tracking_ms_per_bbox, 3)} | "
            f"{bench.optional_float(row.mean_iou, 3)} | {bench.optional_float(row.iou50, 3)} | "
            f"{bench.optional_float(row.gpu_memory_peak_reserved_mb, 1)} | "
            f"{bench.optional_float(row.shared_backbone_calls_per_frame, 3)} | "
            f"{bench.optional_float(row.object_head_items_per_update_frame, 3)} | "
            f"{bench.optional_float(row.proof_shared_backbone_ok_rate, 3)} | "
            f"{bench.optional_float(row.proof_batched_head_ok_rate, 3)} | "
            f"{row.fps_sustains_25} |"
        )
    lines.extend(["", "## v8 FPS Profile", ""])
    lines.append(
        "| Sequence | Config | ReID Mode | N Target | Candidate transfer ms/update | Candidate decode ms/update | Feature-template ms/update | Fusion ms/update | ReID feature ms/update | Identity ms/update | Accept ms/update | Hold ms/update | Feature refresh ms/update | Debug/proof ms/update | Unbucketed ms/update |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in timing_rows:
        identity_ms = None
        if row.profile_identity_resolve_ms_per_update is not None or row.profile_identity_score_ms_per_update is not None:
            identity_ms = (row.profile_identity_resolve_ms_per_update or 0.0) + (
                row.profile_identity_score_ms_per_update or 0.0
            )
        debug_ms = None
        if row.profile_debug_output_ms_per_update is not None or row.profile_proof_output_ms_per_update is not None:
            debug_ms = (row.profile_debug_output_ms_per_update or 0.0) + (
                row.profile_proof_output_ms_per_update or 0.0
            )
        lines.append(
            f"| {row.sequence} | {row.lorat_config} | {row.reid_mode} | {row.target_tracks} | "
            f"{bench.optional_float(row.profile_candidate_transfer_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_candidate_extract_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_template_match_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_candidate_fusion_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_reid_appearance_ms_per_update, 3)} | "
            f"{bench.optional_float(identity_ms, 3)} | "
            f"{bench.optional_float(row.profile_accept_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_hold_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_appearance_refresh_ms_per_update, 3)} | "
            f"{bench.optional_float(debug_ms, 3)} | "
            f"{bench.optional_float(row.profile_unbucketed_ms_per_update, 3)} |"
        )
    candidate_summary = summarize_candidate_diagnostics(candidate_rows)
    if candidate_summary:
        lines.extend(["", "## V8 Candidate Oracle Diagnostics", ""])
        lines.append(
            "| Sequence | Config | ReID Mode | N Target | Samples | Head Mean IoU | Head IoU@0.50 | Top-5 Best Mean IoU | Top-5 IoU@0.50 | Top-5 IoU@0.30 | Template Attempt Rate | Template Mean IoU | Fused Mean IoU | Assigned Mean IoU | Final Mean IoU | Final IoU@0.50 | Final Correct Object | Failure Buckets | Failure Stages | Candidate Sources |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
        )
        for row in candidate_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | {row['target_tracks']} | {row['samples']} | "
                f"{bench.optional_float(row['mean_head_iou'], 3)} | {bench.optional_float(row['head_iou50'], 3)} | "
                f"{bench.optional_float(row['mean_top5_best_iou'], 3)} | {bench.optional_float(row['top5_iou50'], 3)} | "
                f"{bench.optional_float(row['top5_iou30'], 3)} | {bench.optional_float(row['template_attempt_rate'], 3)} | "
                f"{bench.optional_float(row['mean_template_iou'], 3)} | {bench.optional_float(row['mean_fused_iou'], 3)} | "
                f"{bench.optional_float(row['mean_assigned_iou'], 3)} | {bench.optional_float(row['mean_final_iou'], 3)} | "
                f"{bench.optional_float(row['final_iou50'], 3)} | {bench.optional_float(row['final_correct_rate'], 3)} | "
                f"{row['iou_failure_buckets']} | {row['iou_failure_stages']} | "
                f"{row['candidate_sources']} |"
            )
    lines.extend(["", "## Small-Object Reliability", ""])
    lines.append(
        "| Sequence | Config | N Target | Area Bin px | Samples | Mean Area | Mean IoU | IoU@0.50 | Unreliable Rate | Reliable |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in area_rows:
        lines.append(
            f"| {row.sequence} | {row.lorat_config} | {row.target_tracks} | {row.area_bin} | "
            f"{row.samples} | {bench.optional_float(row.mean_area_px, 1)} | "
            f"{bench.optional_float(row.mean_iou, 3)} | {bench.optional_float(row.iou50, 3)} | "
            f"{bench.optional_float(row.unreliable_rate, 3)} | {'' if row.reliable is None else row.reliable} |"
        )
    floors = reliable_floor(area_rows)
    lines.extend(["", "## Smallest Reliable Area", ""])
    lines.append("| Sequence | Config | N Target | Smallest Reliable Area px |")
    lines.append("|---|---:|---:|---:|")
    for key, value in sorted(floors.items()):
        sequence, lorat_config, execution_mode, target_tracks = key
        del execution_mode
        lines.append(f"| {sequence} | {lorat_config} | {target_tracks} | {'' if value is None else int(value)} |")
    identity_summary = summarize_identity(identity_rows)
    if identity_summary:
        lines.extend(["", "## Identity Sampling", ""])
        lines.append("| Sequence | Config | ReID Mode | N Target | Samples | Visible | Hidden | Correct Rate | ID Switches | Track-Loss Rate | Jump Rate | Occluded Rate | Mean IoU |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in identity_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | "
                f"{row['target_tracks']} | {row['samples']} | {row['visible_samples']} | {row['hidden_samples']} | "
                f"{bench.optional_float(row['correct_rate'], 3)} | {row['identity_switches']} | "
                f"{bench.optional_float(row['track_loss_rate'], 3)} | {bench.optional_float(row['jump_rate'], 3)} | "
                f"{bench.optional_float(row['occluded_rate'], 3)} | {bench.optional_float(row['mean_iou'], 3)} |"
            )
    occlusion_summary = summarize_occlusion_survival(identity_rows)
    if occlusion_summary:
        lines.extend(["", "## Occlusion Survival", ""])
        lines.append("| Sequence | Config | ReID Mode | N Target | Tracker | Longest Observed Occlusion Frames | Longest Survived Occlusion Frames | Recovered After Gap | Lost After Gap | Final State |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in occlusion_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | {row['target_tracks']} | "
                f"{row['tracker_id']} | {row['longest_occluded_frames_observed']} | "
                f"{row['longest_occlusion_survived_frames']} | {row['recovered_after_gap']} | "
                f"{row['lost_after_gap']} | {row['final_lifecycle_state']} |"
            )
    lines.extend(["", "## 25 FPS Capacity", ""])
    lines.append("| Sequence | Config | ReID Mode | GPU Profile | Max N Sustaining 25 FPS |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in capacity_rows(timing_rows):
        lines.append(
            f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | {row['gpu_profile']} | "
            f"{'' if row['max_n_sustaining_25_fps'] is None else row['max_n_sustaining_25_fps']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flush_outputs(
    args: argparse.Namespace,
    paths: V8OutputPaths,
    label: str,
    area_bins: Sequence[float],
    timing_rows: Sequence[V8TimingResult],
    area_observations: Sequence[bench.AreaObservation],
    full_area_observations: Sequence[bench.AreaObservation],
    identity_rows: Sequence[IdentityObservation],
    proof_rows: Sequence[Dict[str, object]],
    candidate_rows: Sequence[Dict[str, object]],
    debug_rows: Sequence[Dict[str, object]],
) -> List[bench.AreaSummary]:
    area_rows = bench.summarize_area_observations(
        area_observations,
        area_bins,
        args.reliable_iou50,
        args.reliable_mean_iou,
        args.min_area_samples,
    )
    write_timing_csv(paths.timing_csv, timing_rows)
    bench.write_area_csv(paths.area_csv, area_rows)
    bench.write_observations_csv(paths.observations_csv, area_observations)
    if args.full_area_observations:
        bench.write_observations_csv(paths.full_observations_csv, full_area_observations)
    write_identity_csv(paths.identity_csv, identity_rows)
    write_dict_rows_csv(
        paths.identity_summary_csv,
        summarize_identity(identity_rows),
        [
            "sequence",
            "lorat_config",
            "execution_mode",
            "reid_mode",
            "target_tracks",
            "samples",
            "visible_samples",
            "hidden_samples",
            "correct_rate",
            "jump_rate",
            "identity_switches",
            "identity_switches_per_1000_samples",
            "track_loss_rate",
            "occluded_rate",
            "mean_iou",
        ],
    )
    write_dict_rows_csv(
        paths.occlusion_survival_csv,
        summarize_occlusion_survival(identity_rows),
        [
            "sequence",
            "lorat_config",
            "reid_mode",
            "target_tracks",
            "tracker_id",
            "samples",
            "longest_occluded_frames_observed",
            "longest_occlusion_survived_frames",
            "recovered_after_gap",
            "lost_after_gap",
            "final_lifecycle_state",
        ],
    )
    write_week2_proof_csv(paths.week2_proof_csv, proof_rows)
    write_candidate_diagnostics_csv(paths.candidate_diagnostics_csv, candidate_rows)
    write_debug_csv(paths.debug_csv, debug_rows)
    write_summary_md(paths.summary_md, args, label, paths, timing_rows, area_rows, identity_rows, candidate_rows)
    return area_rows


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    args = parse_args()
    if args.disable_reid:
        args.disable_identity_arbitration = True
    sequences = select_sequences(args)
    if args.list_sequences:
        return 0
    configs = normalized_configs(args)
    track_counts = parse_track_counts(args)
    reid_modes = reid_case_modes(args)
    area_bins = bench.parse_area_bins(args.area_bins)
    label = run_label(args, sequences, configs, track_counts)
    paths = default_output_paths(args, label)

    timing_rows: List[V8TimingResult] = []
    area_observations: List[bench.AreaObservation] = []
    full_area_observations: List[bench.AreaObservation] = []
    identity_rows: List[IdentityObservation] = []
    proof_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []

    print(f"LoRAT v8 benchmark run label: {label}", flush=True)
    print(f"Output folder: {paths.run_root}", flush=True)
    print(f"Timing CSV: {paths.timing_csv}", flush=True)
    print(f"Area reliability CSV: {paths.area_csv}", flush=True)
    print(f"Sampled observations CSV: {paths.observations_csv}", flush=True)
    print(f"Identity CSV: {paths.identity_csv}", flush=True)
    print(f"Identity/ReID summary CSV: {paths.identity_summary_csv}", flush=True)
    print(f"Occlusion survival CSV: {paths.occlusion_survival_csv}", flush=True)
    print(f"Week 2 proof CSV: {paths.week2_proof_csv}", flush=True)
    print(f"Candidate diagnostics CSV: {paths.candidate_diagnostics_csv}", flush=True)
    print(f"Debug CSV: {paths.debug_csv}", flush=True)
    print(f"Summary: {paths.summary_md}", flush=True)
    print(f"Sequences: {', '.join(sequence.name for sequence in sequences)}", flush=True)
    print(f"Configs: {', '.join(configs)}", flush=True)
    print(f"Track counts: {', '.join(str(count) for count in track_counts)}", flush=True)
    print(f"ReID modes: {', '.join(mode for mode, _ in reid_modes)}", flush=True)
    print(f"Frames per run: {args.max_frames if args.max_frames > 0 else 'full sequence'}", flush=True)
    print(f"Area reliability sampling: every {args.area_sample_interval} frame(s)", flush=True)
    print(f"Identity/ReID sampling: every {args.identity_sample_interval} frame(s)", flush=True)

    record_debug(
        debug_rows,
        paths.debug_csv,
        "run_start",
        reason=(
            f"sequences={','.join(sequence.name for sequence in sequences)}; "
            f"configs={','.join(configs)}; track_counts={','.join(str(count) for count in track_counts)}; "
            f"reid_modes={','.join(mode for mode, _ in reid_modes)}"
        ),
        area_sample_interval=args.area_sample_interval,
        identity_sample_interval=args.identity_sample_interval,
    )
    flush_outputs(
        args,
        paths,
        label,
        area_bins,
        timing_rows,
        area_observations,
        full_area_observations,
        identity_rows,
        proof_rows,
        candidate_rows,
        debug_rows,
    )

    for lorat_config in configs:
        for sequence_path in sequences:
            for target_tracks in track_counts:
                for reid_mode, disable_reid in reid_modes:
                    if STOP_REQUESTED:
                        print("Stop requested before starting the next benchmark case.", flush=True)
                        flush_outputs(
                            args,
                            paths,
                            label,
                            area_bins,
                            timing_rows,
                            area_observations,
                            full_area_observations,
                            identity_rows,
                            proof_rows,
                            candidate_rows,
                            debug_rows,
                        )
                        return 0
                    case_args = copy.copy(args)
                    case_args.reid_mode = reid_mode
                    case_args.disable_identity_arbitration = bool(disable_reid)
                    preview_path = (
                        preview_video_path(paths, sequence_path.name, f"{lorat_config}_{reid_mode}", target_tracks, args.max_frames)
                        if args.save_video
                        else None
                    )
                    try:
                        result = run_benchmark_case(
                            case_args,
                            sequence_path,
                            lorat_config,
                            target_tracks,
                            preview_path,
                            debug_rows,
                            proof_rows,
                            candidate_rows,
                            paths.debug_csv,
                        )
                    except Exception as exc:
                        record_debug(
                            debug_rows,
                            paths.debug_csv,
                            "case_error",
                            sequence=sequence_path.name,
                            lorat_config=lorat_config,
                            reid_mode=reid_mode,
                            target_tracks=target_tracks,
                            reason=repr(exc),
                        )
                        flush_outputs(
                            args,
                            paths,
                            label,
                            area_bins,
                            timing_rows,
                            area_observations,
                            full_area_observations,
                            identity_rows,
                            proof_rows,
                            candidate_rows,
                            debug_rows,
                        )
                        raise
                    if result is None:
                        flush_outputs(
                            args,
                            paths,
                            label,
                            area_bins,
                            timing_rows,
                            area_observations,
                            full_area_observations,
                            identity_rows,
                            proof_rows,
                            candidate_rows,
                            debug_rows,
                        )
                        continue
                    timing, sampled_area, full_area, identity = result
                    timing_rows.append(timing)
                    area_observations.extend(sampled_area)
                    full_area_observations.extend(full_area)
                    identity_rows.extend(identity)
                    flush_outputs(
                        args,
                        paths,
                        label,
                        area_bins,
                        timing_rows,
                        area_observations,
                        full_area_observations,
                        identity_rows,
                        proof_rows,
                        candidate_rows,
                        debug_rows,
                    )
                    if STOP_REQUESTED:
                        print("Stop requested; flushed partial v8 benchmark outputs and exiting cleanly.", flush=True)
                        return 0

    record_debug(
        debug_rows,
        paths.debug_csv,
        "run_complete",
        reason=f"completed_cases={len(timing_rows)}; area_samples={len(area_observations)}; identity_samples={len(identity_rows)}",
    )
    flush_outputs(
        args,
        paths,
        label,
        area_bins,
        timing_rows,
        area_observations,
        full_area_observations,
        identity_rows,
        proof_rows,
        candidate_rows,
        debug_rows,
    )
    print("Wrote LoRAT v8 benchmark files:", flush=True)
    print(f"  {paths.timing_csv}", flush=True)
    print(f"  {paths.area_csv}", flush=True)
    print(f"  {paths.observations_csv}", flush=True)
    print(f"  {paths.identity_csv}", flush=True)
    print(f"  {paths.identity_summary_csv}", flush=True)
    print(f"  {paths.occlusion_survival_csv}", flush=True)
    print(f"  {paths.week2_proof_csv}", flush=True)
    print(f"  {paths.candidate_diagnostics_csv}", flush=True)
    print(f"  {paths.debug_csv}", flush=True)
    print(f"  {paths.summary_md}", flush=True)
    if args.full_area_observations:
        print(f"  {paths.full_observations_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


