from __future__ import annotations

import argparse
import copy
import csv
import io
import math
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
import bounding_box_v9_runtime_base as v8
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
    profile_dinov2_crop_reid_ms_per_update: Optional[float]
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
    dinov2_crop_reid_forward_calls: int
    dinov2_crop_reid_forward_items: int
    max_dinov2_crop_reid_batch: int
    assignment_conflict_rejections: int
    assignment_conflict_reasons: str
    assignment_alt_rescue_attempts: int
    assignment_alt_rescue_hits: int
    assignment_alt_rescue_rejects: str
    fps_sustains_25: Optional[bool]
    preview_path: Optional[Path]


@dataclass(frozen=True)
class IdentityObservation:
    sequence: str
    lorat_config: str
    execution_mode: str
    reid_mode: str
    v9_diagnostic_mode: str
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
    mot_metrics_csv: Path
    occlusion_survival_csv: Path
    controlled_occlusion_trials_csv: Path
    controlled_occlusion_survival_csv: Path
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
            "V9-owned LoRAT-MOT benchmark harness for timing, "
            "small-object area reliability, FPS/memory scaling, shared-frame ViT proof, "
            "ReID ablation, identity switches, track loss, and controlled occlusion survival."
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
    parser.add_argument(
        "--init-selection",
        choices=("largest", "smallest", "area-window", "middle"),
        default="largest",
        help=(
            "How to choose GT initialization boxes when no interactive selection is possible. "
            "largest preserves the original benchmark behavior; smallest/area-window are useful "
            "for small-object demo runs on Theia."
        ),
    )
    parser.add_argument("--init-min-area", type=float, default=0.0, help="Minimum GT box area for initialization selection.")
    parser.add_argument("--init-max-area", type=float, default=0.0, help="Maximum GT box area for initialization selection; 0 disables.")
    parser.add_argument(
        "--init-track-id",
        type=int,
        action="append",
        help="Specific GT track id to initialize. Repeat for multiple. Overrides area/size ordering after filtering.",
    )

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
        "--controlled-occlusion-durations",
        default="",
        help=(
            "Comma-separated forced occlusion lengths in frames, e.g. 0,5,10,20,40. "
            "Empty disables the controlled occlusion-duration benchmark."
        ),
    )
    parser.add_argument("--controlled-occlusion-trials-per-duration", type=int, default=3)
    parser.add_argument("--controlled-occlusion-warmup-frames", type=int, default=10)
    parser.add_argument("--controlled-occlusion-recovery-frames", type=int, default=30)
    parser.add_argument(
        "--controlled-occlusion-mask",
        choices=("black", "blur", "mean"),
        default="black",
        help="How to hide the target during forced occlusion windows.",
    )
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
    parser.add_argument(
        "--draw-candidate-diagnostics",
        action="store_true",
        help="Overlay search windows, previous boxes, top candidates, and final boxes when backend diagnostics are available.",
    )
    parser.add_argument(
        "--draw-ground-truth",
        dest="draw_ground_truth",
        action="store_true",
        default=True,
        help=(
            "Draw benchmark ground-truth boxes in saved videos. Yellow boxes are the GT objects "
            "paired with initialized tracks; magenta center lines show prediction-vs-GT error."
        ),
    )
    parser.add_argument(
        "--no-draw-ground-truth",
        dest="draw_ground_truth",
        action="store_false",
        help="Disable ground-truth overlays in saved benchmark videos.",
    )
    parser.add_argument(
        "--draw-all-ground-truth",
        action="store_true",
        help="Also draw visible GT boxes that were not selected as initialized benchmark tracks.",
    )

    parser.add_argument("--lorat-config", default="B-224", choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG) + ("all",),
        help="Run one or more LoRAT configs. Use all to include B/L/g and 224/378.",
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
            "Directory containing per-config trained head checkpoints. "
            "V9's front-end resolves v9_local_head_* files while keeping this legacy option name."
        ),
    )
    parser.add_argument(
        "--v8-head-checkpoint",
        choices=("best", "best_by_val_iou", "best_by_rollout_identity", "latest"),
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
    parser.add_argument(
        "--disable-v8-dinov2-crop-reid",
        dest="v8_dinov2_crop_reid",
        action="store_false",
        default=v8.DEFAULT_V8_DINOV2_CROP_REID,
        help="Disable literal DINOv2 crop embeddings for Week 3 ReID.",
    )
    parser.add_argument("--v8-dinov2-crop-reid-batch", type=int, default=v8.DEFAULT_V8_DINOV2_CROP_REID_BATCH)
    parser.add_argument("--v8-dinov2-crop-reid-min-area", type=float, default=v8.DEFAULT_V8_DINOV2_CROP_REID_MIN_AREA)
    parser.add_argument("--v8-assignment-conflict-iou", type=float, default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_IOU)
    parser.add_argument("--v8-assignment-conflict-hard-iou", type=float, default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_HARD_IOU)
    parser.add_argument("--v8-assignment-conflict-score-margin", type=float, default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_SCORE_MARGIN)
    parser.add_argument(
        "--v8-assignment-conflict-center-ratio",
        type=float,
        default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_CENTER_RATIO,
    )
    parser.add_argument(
        "--v8-assignment-conflict-containment",
        type=float,
        default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_CONTAINMENT,
    )
    parser.add_argument(
        "--v8-assignment-conflict-ownership-margin",
        type=float,
        default=v8.DEFAULT_V8_ASSIGNMENT_CONFLICT_OWNERSHIP_MARGIN,
    )
    parser.add_argument(
        "--disable-v8-assignment-alt-rescue",
        dest="v8_assignment_alt_rescue",
        action="store_false",
        default=v8.DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_ENABLED,
    )
    parser.add_argument(
        "--v8-assignment-alt-rescue-max-candidates",
        type=int,
        default=v8.DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MAX_CANDIDATES,
    )
    parser.add_argument(
        "--v8-assignment-alt-rescue-min-confidence",
        type=float,
        default=v8.DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MIN_CONFIDENCE,
    )
    parser.add_argument(
        "--disable-v8-small-target-mode",
        dest="v8_small_target_mode",
        action="store_false",
        default=v8.DEFAULT_V8_SMALL_TARGET_MODE,
        help="Disable small-target scale locking/template rescue behavior.",
    )
    parser.add_argument("--v8-small-target-area", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_AREA)
    parser.add_argument("--v8-small-target-max-side", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_MAX_SIDE)
    parser.add_argument("--v8-small-target-max-scale-change", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE)
    parser.add_argument("--v8-small-target-template-min-score", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE)
    parser.add_argument("--v8-small-target-template-min-motion", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION)
    parser.add_argument("--v8-small-target-template-min-path", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH)
    parser.add_argument("--v8-small-target-confidence-floor", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR)
    parser.add_argument(
        "--v9-diagnostic-mode",
        choices=("normal", "gt_window", "gt_identity"),
        default="normal",
        help="V9-only diagnostic mode. gt_window centers local windows on GT; gt_identity keeps runtime boxes but labels identity failures.",
    )
    parser.add_argument("--v9-scale-gate-min-ratio", type=float, default=0.35)
    parser.add_argument("--v9-scale-gate-max-ratio", type=float, default=2.50)
    parser.add_argument("--v9-scale-gate-override-confidence", type=float, default=0.92)
    parser.add_argument("--v9-scale-gate-fallback-confidence-scale", type=float, default=0.65)
    parser.add_argument(
        "--v9-protective-reid",
        dest="v9_protective_reid",
        action=argparse.BooleanOptionalAction,
        default=getattr(v8, "DEFAULT_V9_PROTECTIVE_REID", True),
        help="Use DINOv2 crop ReID as a recovery/conflict mechanism instead of steering healthy local tracks.",
    )
    parser.add_argument(
        "--v9-protective-reid-confidence-gate",
        type=float,
        default=getattr(v8, "DEFAULT_V9_PROTECTIVE_REID_CONFIDENCE_GATE", 0.58),
    )
    parser.add_argument(
        "--v9-protective-reid-margin-gate",
        type=float,
        default=getattr(v8, "DEFAULT_V9_PROTECTIVE_REID_MARGIN_GATE", 0.055),
    )
    parser.add_argument(
        "--v9-protective-reid-overlap-iou",
        type=float,
        default=getattr(v8, "DEFAULT_V9_PROTECTIVE_REID_OVERLAP_IOU", 0.10),
    )
    parser.add_argument(
        "--v9-stage1-local-min-confidence",
        type=float,
        default=getattr(v8, "DEFAULT_V9_STAGE1_LOCAL_MIN_CONFIDENCE", 0.62),
    )
    parser.add_argument(
        "--v9-stage1-local-min-margin",
        type=float,
        default=getattr(v8, "DEFAULT_V9_STAGE1_LOCAL_MIN_MARGIN", 0.055),
    )
    parser.add_argument(
        "--v9-stage1-local-min-motion",
        type=float,
        default=getattr(v8, "DEFAULT_V9_STAGE1_LOCAL_MIN_MOTION", 0.36),
    )
    parser.add_argument(
        "--v9-stage1-local-min-path",
        type=float,
        default=getattr(v8, "DEFAULT_V9_STAGE1_LOCAL_MIN_PATH", 0.36),
    )
    parser.add_argument(
        "--v9-next-best-min-confidence",
        type=float,
        default=getattr(v8, "DEFAULT_V9_NEXT_BEST_MIN_CONFIDENCE", 0.24),
    )
    parser.add_argument(
        "--v9-next-best-max-candidates",
        type=int,
        default=getattr(v8, "DEFAULT_V9_NEXT_BEST_MAX_CANDIDATES", 4),
    )
    parser.add_argument(
        "--v9-local-rescue",
        dest="v9_local_rescue",
        action=argparse.BooleanOptionalAction,
        default=getattr(v8, "DEFAULT_V9_LOCAL_RESCUE", True),
        help="Allow sane V9-local owned candidates to override LOWCONF/ID_UNCERTAIN hold.",
    )
    parser.add_argument(
        "--v9-local-rescue-min-assignment-score",
        type=float,
        default=getattr(v8, "DEFAULT_V9_LOCAL_RESCUE_MIN_ASSIGNMENT_SCORE", 0.30),
    )
    parser.add_argument(
        "--v9-local-rescue-max-scale-error",
        type=float,
        default=getattr(v8, "DEFAULT_V9_LOCAL_RESCUE_MAX_SCALE_ERROR", 2.75),
    )
    parser.add_argument(
        "--v9-accept-max-center-ratio",
        type=float,
        default=getattr(v8, "DEFAULT_V9_ACCEPT_MAX_CENTER_RATIO", 2.40),
    )
    parser.add_argument(
        "--v9-accept-max-healthy-center-ratio",
        type=float,
        default=getattr(v8, "DEFAULT_V9_ACCEPT_MAX_HEALTHY_CENTER_RATIO", 1.45),
    )
    parser.add_argument(
        "--v9-local-hold-min-confidence",
        type=float,
        default=getattr(v8, "DEFAULT_V9_LOCAL_HOLD_MIN_CONFIDENCE", 0.28),
    )
    parser.add_argument(
        "--v9-local-hold-max-center-ratio",
        type=float,
        default=getattr(v8, "DEFAULT_V9_LOCAL_HOLD_MAX_CENTER_RATIO", 1.90),
    )
    parser.add_argument(
        "--v9-local-hold-max-lost-center-ratio",
        type=float,
        default=getattr(v8, "DEFAULT_V9_LOCAL_HOLD_MAX_LOST_CENTER_RATIO", 3.00),
    )
    return parser.parse_args()


def parse_track_counts(args: argparse.Namespace) -> List[int]:
    if args.max_track_count > 0:
        return list(range(1, args.max_track_count + 1))
    return bench.parse_int_list(args.track_counts)


def parse_controlled_occlusion_durations(args: argparse.Namespace) -> List[int]:
    text = str(getattr(args, "controlled_occlusion_durations", "") or "").strip()
    if not text:
        return []
    durations = sorted(
        {
            max(0, int(value.strip()))
            for value in text.replace(";", ",").split(",")
            if value.strip()
        }
    )
    return durations


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
    selection_parts: List[str] = []
    if getattr(args, "init_selection", "largest") != "largest":
        selection_parts.append(f"init-{args.init_selection}")
    if getattr(args, "init_min_area", 0.0) > 0:
        selection_parts.append(f"minarea-{int(args.init_min_area)}")
    if getattr(args, "init_max_area", 0.0) > 0:
        selection_parts.append(f"maxarea-{int(args.init_max_area)}")
    if getattr(args, "init_track_id", None):
        selection_parts.append("ids-" + "-".join(str(value) for value in args.init_track_id))
    if selection_parts:
        base = f"{base}_{'_'.join(selection_parts)}"
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
    mot_metrics_csv = bench.unique_path(run_root / "mot_paper_metrics_summary.csv")
    occlusion_survival_csv = bench.unique_path(run_root / "occlusion_survival.csv")
    controlled_occlusion_trials_csv = bench.unique_path(run_root / "controlled_occlusion_trials.csv")
    controlled_occlusion_survival_csv = bench.unique_path(run_root / "controlled_occlusion_survival.csv")
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
        mot_metrics_csv,
        occlusion_survival_csv,
        controlled_occlusion_trials_csv,
        controlled_occlusion_survival_csv,
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
        mot_metrics_csv=mot_metrics_csv,
        occlusion_survival_csv=occlusion_survival_csv,
        controlled_occlusion_trials_csv=controlled_occlusion_trials_csv,
        controlled_occlusion_survival_csv=controlled_occlusion_survival_csv,
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


def diagnostic_gt_boxes_for_backend(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    frame_number: int,
    min_visibility: float,
) -> Dict[int, mot.BBox]:
    rows = {row.track_id: row for row in visible_gt_rows(gt_by_frame, frame_number, min_visibility)}
    boxes: Dict[int, mot.BBox] = {}
    for track_id, gt_track_id in tracker_to_gt_id.items():
        row = rows.get(gt_track_id)
        if row is not None:
            boxes[int(track_id)] = row.bbox
    return boxes


def set_backend_diagnostic_gt_boxes(
    backend: object,
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    frame_number: int,
    min_visibility: float,
) -> None:
    setter = getattr(backend, "set_v9_diagnostic_gt_boxes", None)
    if setter is None:
        return
    setter(diagnostic_gt_boxes_for_backend(gt_by_frame, tracker_to_gt_id, frame_number, min_visibility))


def _bbox_center(bbox: mot.BBox) -> Tuple[float, float]:
    x, y, width, height = bbox
    return float(x) + float(width) * 0.5, float(y) + float(height) * 0.5


def _clamped_box_points(bbox: mot.BBox, frame_shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    height, width = frame_shape[:2]
    x, y, box_width, box_height = [float(value) for value in bbox]
    left = max(0, min(width - 1, int(round(x))))
    top = max(0, min(height - 1, int(round(y))))
    right = max(0, min(width - 1, int(round(x + max(0.0, box_width)))))
    bottom = max(0, min(height - 1, int(round(y + max(0.0, box_height)))))
    return (left, top), (right, bottom)


def _draw_text_with_outline(
    frame,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _coerce_draw_bbox(value: object) -> Optional[mot.BBox]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            return None
        try:
            return tuple(float(part) for part in parts)  # type: ignore[return-value]
        except ValueError:
            return None
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(values) != 4:
        return None
    try:
        return tuple(float(part) for part in values)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def draw_candidate_diagnostics_overlay(
    output,
    raw_diagnostics: Optional[Sequence[Dict[str, object]]],
) -> None:
    if not raw_diagnostics:
        return
    search_color = (255, 255, 0)
    previous_color = (255, 140, 0)
    anchor_color = (180, 100, 255)
    top_color = (0, 165, 255)
    continuity_color = (255, 0, 180)
    final_color = (0, 0, 255)
    for diagnostic in raw_diagnostics:
        track_id = diagnostic.get("track_id", "?")
        search_window = _coerce_draw_bbox(diagnostic.get("search_window"))
        if search_window is not None:
            top_left, bottom_right = _clamped_box_points(search_window, output.shape)
            cv2.rectangle(output, top_left, bottom_right, search_color, 1)
            _draw_text_with_outline(output, f"ID {track_id} search", (top_left[0], max(18, top_left[1] - 6)), search_color, 0.42, 1)

        previous_bbox = _coerce_draw_bbox(diagnostic.get("previous_bbox"))
        if previous_bbox is not None:
            top_left, bottom_right = _clamped_box_points(previous_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, previous_color, 1)
            _draw_text_with_outline(output, "prev", (top_left[0], min(output.shape[0] - 8, bottom_right[1] + 13)), previous_color, 0.36, 1)

        accepted_bbox = _coerce_draw_bbox(diagnostic.get("v9_last_accepted_bbox"))
        if accepted_bbox is not None:
            top_left, bottom_right = _clamped_box_points(accepted_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, anchor_color, 1)
            _draw_text_with_outline(output, "accepted", (top_left[0], min(output.shape[0] - 8, bottom_right[1] + 13)), anchor_color, 0.36, 1)

        anchor_bbox = _coerce_draw_bbox(diagnostic.get("v9_search_anchor_bbox"))
        if anchor_bbox is not None:
            top_left, bottom_right = _clamped_box_points(anchor_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, anchor_color, 1)
            _draw_text_with_outline(output, str(diagnostic.get("v9_search_anchor_source") or "anchor"), (top_left[0], max(18, top_left[1] - 4)), anchor_color, 0.36, 1)

        for candidate in tuple(diagnostic.get("head_top_candidates") or ())[:5]:
            candidate_bbox = _coerce_draw_bbox(getattr(candidate, "bbox", None))
            if candidate_bbox is None:
                continue
            top_left, bottom_right = _clamped_box_points(candidate_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, top_color, 1)
            rank = getattr(candidate, "rank", "?")
            confidence = float(getattr(candidate, "confidence", 0.0) or 0.0)
            _draw_text_with_outline(output, f"t{rank} {confidence:.2f}", (top_left[0], max(18, top_left[1] - 4)), top_color, 0.38, 1)

        continuity_bbox = _coerce_draw_bbox(diagnostic.get("v9_continuity_selected_bbox"))
        if continuity_bbox is not None:
            top_left, bottom_right = _clamped_box_points(continuity_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, continuity_color, 2)
            reason = str(diagnostic.get("v9_continuity_reason") or "continuity")
            _draw_text_with_outline(output, reason, (top_left[0], max(18, top_left[1] - 4)), continuity_color, 0.38, 1)

        final_bbox = _coerce_draw_bbox(diagnostic.get("final_bbox"))
        if final_bbox is not None:
            top_left, bottom_right = _clamped_box_points(final_bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, final_color, 2)
            reason = str(diagnostic.get("diagnostic_failure_reason") or diagnostic.get("v9_scale_gate_state") or "")
            label = f"ID {track_id} final"
            if reason:
                label = f"{label} {reason}"
            _draw_text_with_outline(output, label, (top_left[0], min(output.shape[0] - 8, bottom_right[1] + 16)), final_color, 0.42, 1)


def draw_tracks_with_ground_truth(
    frame,
    tracks: Sequence[mot.TrackState],
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    tracker_to_gt_id: Dict[int, int],
    frame_number: int,
    min_visibility: float,
    backend_label: str,
    status_lines: Optional[Sequence[str]] = None,
    draw_ground_truth: bool = True,
    draw_all_ground_truth: bool = False,
    candidate_diagnostics: Optional[Sequence[Dict[str, object]]] = None,
    draw_candidate_diagnostics: bool = False,
):
    output = mot.draw_tracks(frame, tracks, frame_number, backend_label, status_lines)
    if draw_candidate_diagnostics:
        draw_candidate_diagnostics_overlay(output, candidate_diagnostics)
    if not draw_ground_truth:
        return output

    gt_rows = visible_gt_rows(gt_by_frame, frame_number, min_visibility)
    gt_by_id = {row.track_id: row for row in gt_rows}
    track_by_id = {track.track_id: track for track in tracks}
    initialized_gt_ids = set(tracker_to_gt_id.values())

    gt_color = (0, 255, 255)
    unmatched_gt_color = (170, 170, 170)
    center_error_color = (255, 0, 255)

    for track_id, gt_track_id in tracker_to_gt_id.items():
        gt_row = gt_by_id.get(gt_track_id)
        if gt_row is None:
            continue
        top_left, bottom_right = _clamped_box_points(gt_row.bbox, output.shape)
        cv2.rectangle(output, top_left, bottom_right, gt_color, 2)

        track = track_by_id.get(track_id)
        label = f"GT {gt_track_id}"
        if track is not None:
            iou = exercise.bbox_iou(track.bbox, gt_row.bbox)
            center_error = mot.center_distance(track.bbox, gt_row.bbox)
            label = f"GT {gt_track_id} vs ID {track_id} IoU {iou:.2f} d {center_error:.0f}px"
            pred_center = _bbox_center(track.bbox)
            gt_center = _bbox_center(gt_row.bbox)
            cv2.line(
                output,
                (int(round(pred_center[0])), int(round(pred_center[1]))),
                (int(round(gt_center[0])), int(round(gt_center[1]))),
                center_error_color,
                2,
                cv2.LINE_AA,
            )
        label_x = top_left[0]
        label_y = min(output.shape[0] - 8, max(18, bottom_right[1] + 18))
        _draw_text_with_outline(output, label, (label_x, label_y), gt_color, scale=0.5, thickness=1)

    if draw_all_ground_truth:
        for gt_row in gt_rows:
            if gt_row.track_id in initialized_gt_ids:
                continue
            top_left, bottom_right = _clamped_box_points(gt_row.bbox, output.shape)
            cv2.rectangle(output, top_left, bottom_right, unmatched_gt_color, 1)
            _draw_text_with_outline(
                output,
                f"GT {gt_row.track_id}",
                (top_left[0], max(18, top_left[1] - 6)),
                unmatched_gt_color,
                scale=0.45,
                thickness=1,
            )

    legend = "GT yellow | model colored | magenta line = center error"
    _draw_text_with_outline(output, legend, (20, output.shape[0] - 18), gt_color, scale=0.52, thickness=1)
    return output


def collect_sampled_observations(
    sequence: str,
    lorat_config: str,
    reid_mode: str,
    v9_diagnostic_mode: str,
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
                    v9_diagnostic_mode=v9_diagnostic_mode,
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


def gt_row_for_track(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    frame_number: int,
    gt_track_id: int,
    min_visibility: float,
) -> Optional[exercise.GroundTruthRow]:
    for row in gt_by_frame.get(frame_number, []):
        if row.track_id == gt_track_id and row.confidence != 0 and row.visibility >= min_visibility:
            return row
    return None


def mask_bbox_in_frame(frame, bbox: mot.BBox, mode: str) -> None:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(0, min(width, int(round(x + w))))
    y2 = max(0, min(height, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if mode == "blur" and region.size:
        kernel = max(9, int(min(x2 - x1, y2 - y1) // 3) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    elif mode == "mean" and region.size:
        frame[y1:y2, x1:x2] = region.reshape(-1, region.shape[-1]).mean(axis=0)
    else:
        frame[y1:y2, x1:x2] = 0


def controlled_occlusion_candidate_starts(
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    gt_track_ids: Sequence[int],
    init_frame: int,
    last_frame: int,
    duration: int,
    warmup_frames: int,
    recovery_frames: int,
    min_visibility: float,
) -> List[Tuple[int, int]]:
    start_min = init_frame + max(1, warmup_frames)
    start_max = last_frame - duration - recovery_frames
    if start_max < start_min:
        return []
    candidates: List[Tuple[int, int]] = []
    for gt_track_id in gt_track_ids:
        for start_frame in range(start_min, start_max + 1):
            pre_frame = max(init_frame, start_frame - 1)
            reappear_frame = start_frame + duration
            recovery_end = start_frame + duration + recovery_frames
            if (
                gt_row_for_track(gt_by_frame, pre_frame, gt_track_id, min_visibility) is not None
                and gt_row_for_track(gt_by_frame, reappear_frame, gt_track_id, min_visibility) is not None
                and gt_row_for_track(gt_by_frame, recovery_end, gt_track_id, min_visibility) is not None
            ):
                candidates.append((start_frame, gt_track_id))
    return candidates


def evenly_sample_trials(candidates: Sequence[Tuple[int, int]], limit: int) -> List[Tuple[int, int]]:
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return list(candidates)
    if limit == 1:
        return [candidates[len(candidates) // 2]]
    selected: List[Tuple[int, int]] = []
    for index in range(limit):
        candidate_index = round(index * (len(candidates) - 1) / (limit - 1))
        selected.append(candidates[candidate_index])
    return selected


def controlled_identity_state(
    track: mot.TrackState,
    gt_track_id: int,
    gt_by_frame: Dict[int, List[exercise.GroundTruthRow]],
    frame_number: int,
    min_visibility: float,
    identity_correct_iou: float,
    identity_competitor_margin: float,
) -> Tuple[float, float, Optional[int], float, bool, bool, bool]:
    visible_rows = visible_gt_rows(gt_by_frame, frame_number, min_visibility)
    gt_row = next((row for row in visible_rows if row.track_id == gt_track_id), None)
    own_iou = exercise.bbox_iou(track.bbox, gt_row.bbox) if gt_row is not None else 0.0
    other_ious = [
        (row.track_id, exercise.bbox_iou(track.bbox, row.bbox))
        for row in visible_rows
        if row.track_id != gt_track_id
    ]
    best_other_iou = max((iou for _, iou in other_ious), default=0.0)
    matched_gt_id, matched_gt_iou = mot.matched_gt_id_from_ious(
        gt_track_id,
        own_iou,
        other_ious,
        identity_correct_iou,
    )
    ok = bool(getattr(track, "ok", False))
    correct = (
        gt_row is not None
        and ok
        and own_iou >= identity_correct_iou
        and own_iou + identity_competitor_margin >= best_other_iou
    )
    identity_switch = gt_row is not None and matched_gt_id is not None and matched_gt_id != gt_track_id
    lifecycle_state = mot.set_track_lifecycle(track)
    track_lost = (not ok) or lifecycle_state == mot.TrackLifecycle.LOST or (gt_row is not None and matched_gt_id is None)
    return own_iou, best_other_iou, matched_gt_id, matched_gt_iou, correct, identity_switch, track_lost


def run_controlled_occlusion_trial(
    args: argparse.Namespace,
    sequence_path: Path,
    lorat_config: str,
    target_tracks: int,
    reid_mode: str,
    disable_reid: bool,
    duration_frames: int,
    trial_index: int,
    occlusion_start_frame: int,
    occluded_gt_id: int,
) -> Dict[str, object]:
    image_paths = exercise.get_image_paths(sequence_path)
    gt_by_frame = exercise.read_gt(sequence_path)
    fps, sequence_length = exercise.read_sequence_info(sequence_path)
    init_frame, init_rows = exercise.pick_initial_rows(
        gt_by_frame,
        args.init_frame,
        args.class_id,
        args.min_visibility,
        target_tracks,
        max(args.min_init_tracks, target_tracks),
        args.init_selection,
        args.init_min_area,
        args.init_max_area,
        args.init_track_id,
    )
    boxes = [row.bbox for row in init_rows]
    gt_track_ids = [row.track_id for row in init_rows]
    init_index = exercise.frame_to_image_index(init_frame)
    init_frame_image = cv2.imread(str(image_paths[init_index]))
    if init_frame_image is None:
        raise RuntimeError(f"Unable to read frame: {image_paths[init_index]}")

    case_args = copy.copy(args)
    case_args.reid_mode = reid_mode
    case_args.disable_identity_arbitration = bool(disable_reid)
    run_args = tracker_args_for_run(case_args, lorat_config, target_tracks)
    source = SimpleNamespace(name=sequence_path.name, fps=fps, length=sequence_length or len(image_paths))
    backend = v8.create_backend(run_args, source, target_tracks)
    occlusion_end_frame = occlusion_start_frame + duration_frames - 1
    recovery_start_frame = occlusion_start_frame + duration_frames
    recovery_end_frame = min(len(image_paths), recovery_start_frame + args.controlled_occlusion_recovery_frames)
    pre_frame = max(init_frame, occlusion_start_frame - 1)
    pre_iou = 0.0
    pre_correct = False
    post_ious: List[float] = []
    post_iou50_count = 0
    recovered = False
    first_recovery_frame: Optional[int] = None
    identity_switch_after = False
    track_lost_after = False
    final_iou = 0.0
    final_state = ""
    frames_to_recover: Optional[int] = None

    try:
        backend.initialize(init_frame_image, boxes, init_frame)
        tracker_to_gt_id = {
            track.track_id: gt_track_id
            for track, gt_track_id in zip(backend.tracks, gt_track_ids)
            if gt_track_id is not None
        }
        occluded_tracker_id = next(
            (track_id for track_id, gt_track_id in tracker_to_gt_id.items() if gt_track_id == occluded_gt_id),
            None,
        )
        if occluded_tracker_id is None:
            raise RuntimeError(f"Controlled occlusion GT id {occluded_gt_id} was not initialized.")

        for image_index in range(init_index + 1, recovery_end_frame):
            frame_number = image_index + 1
            frame = cv2.imread(str(image_paths[image_index]))
            if frame is None:
                continue
            if duration_frames > 0 and occlusion_start_frame <= frame_number <= occlusion_end_frame:
                gt_row = gt_row_for_track(gt_by_frame, frame_number, occluded_gt_id, 0.0)
                if gt_row is not None:
                    mask_bbox_in_frame(frame, gt_row.bbox, args.controlled_occlusion_mask)
            backend.update(frame, frame_number)
            track = next((item for item in backend.tracks if item.track_id == occluded_tracker_id), None)
            if track is None:
                continue
            own_iou, _, matched_gt_id, _, correct, identity_switch, track_lost = controlled_identity_state(
                track,
                occluded_gt_id,
                gt_by_frame,
                frame_number,
                args.min_visibility,
                args.identity_correct_iou,
                args.identity_competitor_margin,
            )
            if frame_number == pre_frame:
                pre_iou = own_iou
                pre_correct = correct
            if recovery_start_frame <= frame_number <= recovery_end_frame:
                post_ious.append(own_iou)
                if own_iou >= args.identity_correct_iou:
                    post_iou50_count += int(own_iou >= 0.50)
                identity_switch_after = identity_switch_after or identity_switch
                track_lost_after = track_lost_after or track_lost
                if correct and not recovered:
                    recovered = True
                    first_recovery_frame = frame_number
                    frames_to_recover = max(0, frame_number - recovery_start_frame)
                final_iou = own_iou
                final_state = str(getattr(track, "state", ""))
    finally:
        backend.close()

    valid_trial = bool(pre_correct)
    survived = bool(valid_trial and recovered and not identity_switch_after)
    return {
        "sequence": sequence_path.name,
        "lorat_config": lorat_config,
        "execution_mode": EXECUTION_MODE,
        "reid_mode": reid_mode,
        "target_tracks": target_tracks,
        "duration_frames": duration_frames,
        "trial_index": trial_index,
        "occluded_gt_id": occluded_gt_id,
        "occluded_tracker_id": occluded_tracker_id,
        "occlusion_start_frame": occlusion_start_frame,
        "occlusion_end_frame": occlusion_end_frame if duration_frames > 0 else "",
        "recovery_start_frame": recovery_start_frame,
        "recovery_end_frame": recovery_end_frame,
        "pre_iou": pre_iou,
        "pre_correct": int(pre_correct),
        "valid_trial": int(valid_trial),
        "recovered": int(recovered),
        "survived": int(survived),
        "identity_lost": int(not survived),
        "first_recovery_frame": first_recovery_frame if first_recovery_frame is not None else "",
        "frames_to_recover": frames_to_recover if frames_to_recover is not None else "",
        "post_mean_iou": statistics.fmean(post_ious) if post_ious else None,
        "post_iou50": (post_iou50_count / len(post_ious)) if post_ious else None,
        "identity_switch_after": int(identity_switch_after),
        "track_lost_after": int(track_lost_after),
        "final_iou": final_iou,
        "final_state": final_state,
        "rule": (
            f"valid if pre-occlusion IoU >= {args.identity_correct_iou}; "
            f"survived if same GT recovered within {args.controlled_occlusion_recovery_frames} frames "
            "without an identity switch"
        ),
    }


def run_controlled_occlusion_benchmark(
    args: argparse.Namespace,
    sequences: Sequence[Path],
    configs: Sequence[str],
    track_counts: Sequence[int],
    reid_modes: Sequence[Tuple[str, bool]],
    debug_rows: List[Dict[str, object]],
    debug_csv: Path,
) -> List[Dict[str, object]]:
    durations = parse_controlled_occlusion_durations(args)
    if not durations:
        return []
    trial_rows: List[Dict[str, object]] = []
    for lorat_config in configs:
        for sequence_path in sequences:
            image_paths = exercise.get_image_paths(sequence_path)
            gt_by_frame = exercise.read_gt(sequence_path)
            last_frame = len(image_paths)
            for target_tracks in track_counts:
                init_frame, init_rows = exercise.pick_initial_rows(
                    gt_by_frame,
                    args.init_frame,
                    args.class_id,
                    args.min_visibility,
                    target_tracks,
                    max(args.min_init_tracks, target_tracks),
                    args.init_selection,
                    args.init_min_area,
                    args.init_max_area,
                    args.init_track_id,
                )
                if len(init_rows) < target_tracks and not args.allow_fewer_tracks:
                    continue
                gt_track_ids = [row.track_id for row in init_rows]
                for duration in durations:
                    candidates = controlled_occlusion_candidate_starts(
                        gt_by_frame,
                        gt_track_ids,
                        init_frame,
                        last_frame,
                        duration,
                        args.controlled_occlusion_warmup_frames,
                        args.controlled_occlusion_recovery_frames,
                        args.min_visibility,
                    )
                    starts = evenly_sample_trials(candidates, args.controlled_occlusion_trials_per_duration)
                    if not starts:
                        record_debug(
                            debug_rows,
                            debug_csv,
                            "controlled_occlusion_skip",
                            sequence=sequence_path.name,
                            lorat_config=lorat_config,
                            target_tracks=target_tracks,
                            reason=f"duration={duration}; no visible before/after windows",
                        )
                        continue
                    for reid_mode, disable_reid in reid_modes:
                        for trial_index, (start_frame, occluded_gt_id) in enumerate(starts, start=1):
                            if STOP_REQUESTED:
                                return trial_rows
                            row = run_controlled_occlusion_trial(
                                args,
                                sequence_path,
                                lorat_config,
                                target_tracks,
                                reid_mode,
                                disable_reid,
                                duration,
                                trial_index,
                                start_frame,
                                occluded_gt_id,
                            )
                            trial_rows.append(row)
                            print(
                                f"Controlled occlusion {sequence_path.name} {lorat_config} "
                                f"N={target_tracks} {reid_mode} duration={duration} "
                                f"trial={trial_index}: survived={row['survived']} valid={row['valid_trial']}",
                                flush=True,
                            )
    return trial_rows


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


def optional_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def bbox_contains_center(container: object, target: mot.BBox) -> bool:
    bbox = coerce_bbox(container)
    if bbox is None:
        return False
    cx, cy = mot.bbox_center(target)
    x, y, width, height = mot.clamp_bbox_size(bbox)
    return x <= cx <= x + width and y <= cy <= y + height


def classify_diagnostic_failure_reason(
    diagnostic: Dict[str, object],
    track_id: int,
    final_iou: Optional[float],
    final_correct_object: bool,
    final_best_other_iou: Optional[float],
    head_iou: Optional[float],
    head_top5_best_iou: float,
    assigned_iou: Optional[float],
    final_area_ratio: Optional[float],
    gt_bbox: mot.BBox,
    reid_mode: str,
) -> str:
    if final_correct_object or (final_iou is not None and final_iou >= 0.50):
        return "ok"
    if bool(diagnostic.get("reid_caused_hold")):
        return "reid_caused_hold"
    if bool(diagnostic.get("reid_wrong_reattach")):
        return "reid_reinforced_wrong_crop"
    if diagnostic.get("search_window") is not None and not bbox_contains_center(diagnostic.get("search_window"), gt_bbox):
        return "window_missed_target"
    assigned_source = diagnostic.get("assigned_source_track_id")
    if assigned_source not in (None, "", track_id, str(track_id)) and not final_correct_object:
        return "identity_arbitration_wrong"
    if final_best_other_iou is not None and final_iou is not None and final_best_other_iou > final_iou + 0.05:
        if str(reid_mode).lower() == "reid_on" or "reid" in str(diagnostic.get("assigned_source") or "").lower():
            return "reid_reinforced_wrong_crop"
        if bool(diagnostic.get("v9_continuity_applied")):
            return "continuity_selected_wrong_candidate"
        return "identity_arbitration_wrong"
    if bool(diagnostic.get("v9_continuity_applied")) and final_iou is not None and final_iou < 0.30:
        return "continuity_selected_wrong_candidate"
    scale_gate_state = str(diagnostic.get("v9_scale_gate_state") or "")
    if scale_gate_state == "override_high_confidence" or scale_gate_state.startswith("reference_size_lock"):
        return "decode_geometry_bad"
    if final_area_ratio is not None and (final_area_ratio < 0.35 or final_area_ratio > 2.75):
        return "decode_geometry_bad"
    if head_top5_best_iou >= 0.50 and (head_iou or 0.0) < 0.50:
        return "head_selected_wrong_candidate"
    if assigned_iou is not None and assigned_iou < 0.30 and (head_iou or 0.0) >= 0.30:
        return "identity_arbitration_wrong"
    if (head_iou or 0.0) < 0.30:
        return "head_selected_wrong_candidate"
    return "unclassified"


def best_oracle_candidate(
    diagnostic: Dict[str, object],
    final_bbox: Optional[mot.BBox],
    gt_bbox: mot.BBox,
) -> Dict[str, object]:
    candidates: List[Tuple[str, Optional[mot.BBox], Optional[float]]] = [
        ("head", coerce_bbox(diagnostic.get("head_bbox")), optional_float(diagnostic.get("head_confidence"))),
        (
            "head_original",
            coerce_bbox(diagnostic.get("head_original_bbox")),
            optional_float(diagnostic.get("head_original_confidence")),
        ),
        ("template", coerce_bbox(diagnostic.get("template_bbox")), optional_float(diagnostic.get("template_confidence"))),
        ("fused", coerce_bbox(diagnostic.get("fused_bbox")), optional_float(diagnostic.get("fused_confidence"))),
        (
            "continuity_current",
            coerce_bbox(diagnostic.get("v9_continuity_current_bbox")),
            optional_float(diagnostic.get("v9_continuity_current_score")),
        ),
        (
            "continuity_selected",
            coerce_bbox(diagnostic.get("v9_continuity_selected_bbox")),
            optional_float(diagnostic.get("v9_continuity_score")),
        ),
        ("assigned", coerce_bbox(diagnostic.get("assigned_bbox")), optional_float(diagnostic.get("assigned_confidence"))),
        ("final", final_bbox, optional_float(diagnostic.get("final_confidence"))),
    ]
    for candidate in tuple(diagnostic.get("head_top_candidates") or ()):
        candidates.append(
            (
                f"top{int(getattr(candidate, 'rank', 0) or 0)}",
                coerce_bbox(getattr(candidate, "bbox", None)),
                optional_float(getattr(candidate, "confidence", None)),
            )
        )

    scored: List[Tuple[float, str, mot.BBox, Optional[float]]] = []
    for source, bbox, confidence in candidates:
        if bbox is None:
            continue
        scored.append((exercise.bbox_iou(bbox, gt_bbox), source, bbox, confidence))
    if not scored:
        return {
            "oracle_candidate_count": 0,
            "oracle_best_source": "",
            "oracle_best_iou": None,
            "oracle_best_confidence": None,
            "oracle_best_bbox": "",
            "oracle_runtime_iou_gap": None,
            "oracle_best_iou50": False,
            "oracle_best_iou30": False,
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, source, bbox, confidence = scored[0]
    final_iou = exercise.bbox_iou(final_bbox, gt_bbox) if final_bbox is not None else 0.0
    return {
        "oracle_candidate_count": len(scored),
        "oracle_best_source": source,
        "oracle_best_iou": best_iou,
        "oracle_best_confidence": confidence,
        "oracle_best_bbox": bbox_text(bbox),
        "oracle_runtime_iou_gap": max(0.0, best_iou - final_iou),
        "oracle_best_iou50": best_iou >= 0.50,
        "oracle_best_iou30": best_iou >= 0.30,
    }


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
        reid_outcome = str(diagnostic.get("reid_outcome") or "")
        assigned_source_value = diagnostic.get("assigned_source_track_id")
        assigned_from_other = assigned_source_value not in (None, "", track_id, str(track_id))
        reid_wrong_reattach = bool(
            not final_correct_object
            and final_best_other_iou is not None
            and final_iou is not None
            and final_best_other_iou > final_iou + 0.05
            and (
                str(reid_mode).lower() == "reid_on"
                or "reid" in reid_outcome.lower()
                or "reid" in str(diagnostic.get("assigned_source") or "").lower()
                or assigned_from_other
            )
        )
        diagnostic["reid_wrong_reattach"] = reid_wrong_reattach
        diagnostic_failure_reason = classify_diagnostic_failure_reason(
            diagnostic,
            track_id,
            final_iou,
            final_correct_object,
            final_best_other_iou,
            head_iou,
            best_top[1],
            assigned_iou,
            final_area_ratio,
            gt_bbox,
            reid_mode,
        )
        oracle = best_oracle_candidate(diagnostic, final_bbox, gt_bbox)

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
                "v9_diagnostic_mode": diagnostic.get("v9_diagnostic_mode"),
                "diagnostic_failure_reason": diagnostic_failure_reason,
                "search_window": bbox_text(diagnostic.get("search_window")),
                "v9_search_anchor_bbox": bbox_text(diagnostic.get("v9_search_anchor_bbox")),
                "v9_search_anchor_source": diagnostic.get("v9_search_anchor_source"),
                "v9_search_anchor_age": diagnostic.get("v9_search_anchor_age"),
                "v9_last_accepted_bbox": bbox_text(diagnostic.get("v9_last_accepted_bbox")),
                "v9_last_accepted_frame": diagnostic.get("v9_last_accepted_frame"),
                "v9_last_accepted_source": diagnostic.get("v9_last_accepted_source"),
                "v9_search_window_contains_final_center": diagnostic.get("v9_search_window_contains_final_center"),
                "v9_search_window_contains_head_center": diagnostic.get("v9_search_window_contains_head_center"),
                "previous_bbox": bbox_text(diagnostic.get("previous_bbox")),
                "predicted_bbox": bbox_text(diagnostic.get("predicted_bbox")),
                "head_original_bbox": bbox_text(diagnostic.get("head_original_bbox")),
                "head_original_confidence": diagnostic.get("head_original_confidence"),
                "head_bbox": bbox_text(diagnostic.get("head_bbox")),
                "head_confidence": diagnostic.get("head_confidence"),
                "head_margin": diagnostic.get("head_margin"),
                "head_visibility": diagnostic.get("head_visibility"),
                "candidate_visibility": diagnostic.get("candidate_visibility"),
                "head_roi_tokens": diagnostic.get("head_roi_tokens"),
                "v9_scale_gate_state": diagnostic.get("v9_scale_gate_state"),
                "v9_scale_gate_reason": diagnostic.get("v9_scale_gate_reason"),
                "v9_scale_gate_width_ratio": diagnostic.get("v9_scale_gate_width_ratio"),
                "v9_scale_gate_height_ratio": diagnostic.get("v9_scale_gate_height_ratio"),
                "v9_scale_gate_locked_bbox": bbox_text(diagnostic.get("v9_scale_gate_locked_bbox")),
                "v9_scale_gate_locked_width_ratio": diagnostic.get("v9_scale_gate_locked_width_ratio"),
                "v9_scale_gate_locked_height_ratio": diagnostic.get("v9_scale_gate_locked_height_ratio"),
                "v9_scale_gate_suppressed_original": diagnostic.get("v9_scale_gate_suppressed_original"),
                "v9_scale_gate_confidence_preserved": diagnostic.get("v9_scale_gate_confidence_preserved"),
                "v9_scale_candidate_original_score": diagnostic.get("v9_scale_candidate_original_score"),
                "v9_scale_candidate_locked_score": diagnostic.get("v9_scale_candidate_locked_score"),
                "v9_scale_candidate_selected": diagnostic.get("v9_scale_candidate_selected"),
                "v9_crop_reid_allowed": diagnostic.get("v9_crop_reid_allowed"),
                "v9_accept_guard_state": diagnostic.get("v9_accept_guard_state"),
                "v9_local_owner_override": diagnostic.get("v9_local_owner_override"),
                "v9_hold_source": diagnostic.get("v9_hold_source"),
                "v9_local_health_ok": diagnostic.get("v9_local_health_ok"),
                "v9_local_health_reason": diagnostic.get("v9_local_health_reason"),
                "v9_local_health_tier": diagnostic.get("v9_local_health_tier"),
                "v9_local_health_confidence_threshold": diagnostic.get("v9_local_health_confidence_threshold"),
                "v9_local_health_margin_threshold": diagnostic.get("v9_local_health_margin_threshold"),
                "v9_local_health_motion": diagnostic.get("v9_local_health_motion"),
                "v9_local_health_accepted_anchor_motion": diagnostic.get("v9_local_health_accepted_anchor_motion"),
                "v9_local_health_path": diagnostic.get("v9_local_health_path"),
                "v9_local_health_continuity_score": diagnostic.get("v9_local_health_continuity_score"),
                "v9_local_health_identity_risk": diagnostic.get("v9_local_health_identity_risk"),
                "v9_local_health_visibility": diagnostic.get("v9_local_health_visibility"),
                "v9_association_stage": diagnostic.get("v9_association_stage"),
                "reid_outcome": reid_outcome,
                "reid_prevented_switch": diagnostic.get("reid_prevented_switch"),
                "reid_caused_hold": diagnostic.get("reid_caused_hold"),
                "reid_recovered_lost": diagnostic.get("reid_recovered_lost"),
                "reid_wrong_reattach": reid_wrong_reattach,
                "reid_noop_bad_candidate_pool": diagnostic.get("reid_noop_bad_candidate_pool"),
                "reid_skipped_healthy_local": diagnostic.get("reid_skipped_healthy_local"),
                "reid_next_best_attempted": diagnostic.get("reid_next_best_attempted"),
                "reid_next_best_accepted": diagnostic.get("reid_next_best_accepted"),
                "reid_next_best_source": diagnostic.get("reid_next_best_source"),
                "reid_next_best_reason": diagnostic.get("reid_next_best_reason"),
                "v9_local_rescue_accept": diagnostic.get("v9_local_rescue_accept"),
                "v9_local_rescue_reject_state": diagnostic.get("v9_local_rescue_reject_state"),
                "v9_continuity_enabled": diagnostic.get("v9_continuity_enabled"),
                "v9_continuity_candidate_count": diagnostic.get("v9_continuity_candidate_count"),
                "v9_continuity_applied": diagnostic.get("v9_continuity_applied"),
                "v9_continuity_reason": diagnostic.get("v9_continuity_reason"),
                "v9_continuity_current_score": diagnostic.get("v9_continuity_current_score"),
                "v9_continuity_best_score": diagnostic.get("v9_continuity_best_score"),
                "v9_continuity_score": diagnostic.get("v9_continuity_score"),
                "v9_continuity_score_margin": diagnostic.get("v9_continuity_score_margin"),
                "v9_continuity_selected_rank": diagnostic.get("v9_continuity_selected_rank"),
                "v9_continuity_selected_source": diagnostic.get("v9_continuity_selected_source"),
                "v9_continuity_selected_bbox": bbox_text(diagnostic.get("v9_continuity_selected_bbox")),
                "v9_continuity_current_bbox": bbox_text(diagnostic.get("v9_continuity_current_bbox")),
                "v9_continuity_current_source": diagnostic.get("v9_continuity_current_source"),
                "v9_continuity_head_score": diagnostic.get("v9_continuity_head_score"),
                "v9_continuity_margin_score": diagnostic.get("v9_continuity_margin_score"),
                "v9_continuity_motion_score": diagnostic.get("v9_continuity_motion_score"),
                "v9_continuity_accepted_anchor_motion": diagnostic.get("v9_continuity_accepted_anchor_motion"),
                "v9_continuity_path_score": diagnostic.get("v9_continuity_path_score"),
                "v9_continuity_current_visibility": diagnostic.get("v9_continuity_current_visibility"),
                "v9_continuity_best_visibility": diagnostic.get("v9_continuity_best_visibility"),
                "v9_continuity_anchor_score": diagnostic.get("v9_continuity_anchor_score"),
                "v9_continuity_appearance_score": diagnostic.get("v9_continuity_appearance_score"),
                "v9_visibility_score": diagnostic.get("v9_visibility_score"),
                "v9_visibility_absent_penalty": diagnostic.get("v9_visibility_absent_penalty"),
                "v9_continuity_other_anchor_score": diagnostic.get("v9_continuity_other_anchor_score"),
                "v9_continuity_negative_anchor_score": diagnostic.get("v9_continuity_negative_anchor_score"),
                "v9_continuity_other_anchor_pressure": diagnostic.get("v9_continuity_other_anchor_pressure"),
                "v9_continuity_negative_anchor_pressure": diagnostic.get("v9_continuity_negative_anchor_pressure"),
                "v9_continuity_identity_risk": diagnostic.get("v9_continuity_identity_risk"),
                "v9_continuity_identity_margin": diagnostic.get("v9_continuity_identity_margin"),
                "v9_continuity_scale_score": diagnostic.get("v9_continuity_scale_score"),
                "v9_continuity_center_jump_penalty": diagnostic.get("v9_continuity_center_jump_penalty"),
                "v9_continuity_local_reject": diagnostic.get("v9_continuity_local_reject"),
                "v9_continuity_current_local_reject": diagnostic.get("v9_continuity_current_local_reject"),
                "v9_continuity_best_local_reject": diagnostic.get("v9_continuity_best_local_reject"),
                "v9_continuity_scale_gate_state": diagnostic.get("v9_continuity_scale_gate_state"),
                "v9_continuity_current_drift_risk": diagnostic.get("v9_continuity_current_drift_risk"),
                "v9_continuity_drift_risk": diagnostic.get("v9_continuity_drift_risk"),
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
                "v9_final_minus_head_iou_gap": final_iou - head_iou if head_iou is not None and final_iou is not None else None,
                "final_center_error_px": final_center_error,
                "final_center_error_gt_size": final_center_error_norm,
                "final_area_ratio": final_area_ratio,
                "final_best_other_iou": final_best_other_iou,
                "final_correct_object": final_correct_object,
                "final_state": diagnostic.get("final_state"),
                "assigned_source": diagnostic.get("assigned_source"),
                "iou_failure_bucket": failure_bucket,
                "iou_failure_stage": failure_stage,
                **oracle,
            }
        )
    return output_rows


def summarize_identity(identity_rows: Sequence[IdentityObservation]) -> List[Dict[str, object]]:
    def selected_target_ok(row: IdentityObservation) -> bool:
        return row.correct_object and not row.track_lost and not row.identity_switch

    def steady_tracking_stats(rows: Sequence[IdentityObservation]) -> Dict[str, object]:
        by_tracker: Dict[int, List[IdentityObservation]] = {}
        for row in rows:
            by_tracker.setdefault(row.tracker_id, []).append(row)

        per_tracker: List[Dict[str, object]] = []
        for tracker_rows in by_tracker.values():
            visible_rows = sorted((row for row in tracker_rows if row.gt_visible), key=lambda row: (row.frame, row.sample_offset))
            if not visible_rows:
                continue

            frames_until_failure = 0
            first_failure_seen = False
            current_correct = 0
            longest_correct = 0
            correct_visible_frames = 0
            for row in visible_rows:
                if selected_target_ok(row):
                    correct_visible_frames += 1
                    current_correct += 1
                    longest_correct = max(longest_correct, current_correct)
                    if not first_failure_seen:
                        frames_until_failure += 1
                else:
                    current_correct = 0
                    first_failure_seen = True
            per_tracker.append(
                {
                    "visible_frames": len(visible_rows),
                    "correct_visible_frames": correct_visible_frames,
                    "frames_until_failure": frames_until_failure,
                    "longest_correct_run": longest_correct,
                    "survived_span": int(not first_failure_seen),
                }
            )

        if not per_tracker:
            return {
                "steady_tracker_count": 0,
                "steady_mean_frames_until_failure": None,
                "steady_min_frames_until_failure": None,
                "steady_max_frames_until_failure": None,
                "steady_mean_longest_correct_run": None,
                "steady_min_longest_correct_run": None,
                "steady_max_longest_correct_run": None,
                "steady_survived_span_rate": None,
            }

        frames_until = [int(row["frames_until_failure"]) for row in per_tracker]
        longest_runs = [int(row["longest_correct_run"]) for row in per_tracker]
        return {
            "steady_tracker_count": len(per_tracker),
            "steady_mean_frames_until_failure": statistics.fmean(frames_until),
            "steady_min_frames_until_failure": min(frames_until),
            "steady_max_frames_until_failure": max(frames_until),
            "steady_mean_longest_correct_run": statistics.fmean(longest_runs),
            "steady_min_longest_correct_run": min(longest_runs),
            "steady_max_longest_correct_run": max(longest_runs),
            "steady_survived_span_rate": statistics.fmean(int(row["survived_span"]) for row in per_tracker),
        }

    def single_object_survival_stats(rows: Sequence[IdentityObservation]) -> Dict[str, object]:
        if not rows:
            return {}
        by_tracker: Dict[int, List[IdentityObservation]] = {}
        for row in rows:
            by_tracker.setdefault(row.tracker_id, []).append(row)
        tracker_rows = max(
            by_tracker.values(),
            key=lambda group: (sum(1 for row in group if row.gt_visible), len(group)),
        )
        visible_rows = sorted((row for row in tracker_rows if row.gt_visible), key=lambda row: (row.frame, row.sample_offset))
        if not visible_rows:
            return {
                "single_object_visible_frames": 0,
                "single_object_correct_visible_frames": 0,
                "single_object_frames_until_first_failure": None,
                "single_object_first_failure_frame": None,
                "single_object_first_failure_offset": None,
                "single_object_first_failure_reason": "",
                "single_object_longest_correct_run": 0,
                "single_object_longest_failure_run": 0,
                "single_object_survived_visible_span": None,
            }

        frames_until_failure = 0
        correct_visible_frames = 0
        first_failure_frame: Optional[int] = None
        first_failure_offset: Optional[int] = None
        first_failure_reason = ""
        current_correct = 0
        current_failure = 0
        longest_correct = 0
        longest_failure = 0
        for row in visible_rows:
            if selected_target_ok(row):
                correct_visible_frames += 1
                current_correct += 1
                current_failure = 0
                longest_correct = max(longest_correct, current_correct)
                if first_failure_frame is None:
                    frames_until_failure += 1
                continue

            current_correct = 0
            current_failure += 1
            longest_failure = max(longest_failure, current_failure)
            if first_failure_frame is None:
                first_failure_frame = row.frame
                first_failure_offset = row.sample_offset
                if row.identity_switch:
                    first_failure_reason = "identity_switch"
                elif row.track_lost:
                    first_failure_reason = "track_lost"
                elif not row.correct_object:
                    first_failure_reason = "wrong_object"
                elif row.identity_jump:
                    first_failure_reason = "identity_jump"
                else:
                    first_failure_reason = "unknown_failure"

        survived_visible_span = first_failure_frame is None
        return {
            "single_object_visible_frames": len(visible_rows),
            "single_object_correct_visible_frames": correct_visible_frames,
            "single_object_frames_until_first_failure": frames_until_failure,
            "single_object_first_failure_frame": first_failure_frame,
            "single_object_first_failure_offset": first_failure_offset,
            "single_object_first_failure_reason": first_failure_reason,
            "single_object_longest_correct_run": longest_correct,
            "single_object_longest_failure_run": longest_failure,
            "single_object_survived_visible_span": int(survived_visible_span),
        }

    grouped: Dict[Tuple[str, str, str, str, str, int], List[IdentityObservation]] = {}
    for row in identity_rows:
        key = (
            row.sequence,
            row.lorat_config,
            row.execution_mode,
            row.reid_mode,
            row.v9_diagnostic_mode,
            row.target_tracks,
        )
        grouped.setdefault(key, []).append(row)
    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, execution_mode, reid_mode, diagnostic_mode, target_tracks), rows in sorted(grouped.items()):
        visible_rows = [row for row in rows if row.gt_visible]
        summary_rows = visible_rows or list(rows)
        switch_count = sum(1 for row in summary_rows if row.identity_switch)
        lost_count = sum(1 for row in summary_rows if row.track_lost)
        summary: Dict[str, object] = {
            "sequence": sequence,
            "lorat_config": lorat_config,
            "execution_mode": execution_mode,
            "reid_mode": reid_mode,
            "v9_diagnostic_mode": diagnostic_mode,
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
        summary.update(steady_tracking_stats(rows))
        if target_tracks == 1:
            summary.update(single_object_survival_stats(rows))
        else:
            summary.update(
                {
                    "single_object_visible_frames": "",
                    "single_object_correct_visible_frames": "",
                    "single_object_frames_until_first_failure": "",
                    "single_object_first_failure_frame": "",
                    "single_object_first_failure_offset": "",
                    "single_object_first_failure_reason": "",
                    "single_object_longest_correct_run": "",
                    "single_object_longest_failure_run": "",
                    "single_object_survived_visible_span": "",
                }
            )
        summaries.append(summary)
    return summaries


def summarize_mot_paper_metrics(identity_rows: Sequence[IdentityObservation]) -> List[Dict[str, object]]:
    thresholds = [round(value / 100.0, 2) for value in range(5, 100, 5)]
    summary_threshold = 0.50

    def prediction_present(row: IdentityObservation) -> bool:
        lifecycle = str(row.lifecycle_state).lower()
        state = str(row.state).lower()
        return bool(row.ok) and lifecycle != "lost" and state not in {"lost", "inactive"}

    def correct_at_threshold(row: IdentityObservation, threshold: float) -> bool:
        return row.gt_visible and row.own_iou >= threshold and row.own_iou >= row.best_other_iou

    def id_switch_at_threshold(row: IdentityObservation, threshold: float) -> bool:
        return row.gt_visible and prediction_present(row) and row.best_other_iou >= threshold and row.best_other_iou > row.own_iou

    def threshold_metrics(rows: Sequence[IdentityObservation], threshold: float) -> Dict[str, object]:
        visible_rows = [row for row in rows if row.gt_visible]
        correct_rows = [row for row in rows if correct_at_threshold(row, threshold)]
        tp = len(correct_rows)
        fn = max(0, len(visible_rows) - tp)
        fp = sum(1 for row in rows if prediction_present(row) and not correct_at_threshold(row, threshold))
        idsw = sum(1 for row in rows if id_switch_at_threshold(row, threshold))
        gt_count = len(visible_rows)

        mota = 1.0 - ((fn + fp + idsw) / gt_count) if gt_count else None
        id_precision = tp / (tp + fp) if (tp + fp) else None
        id_recall = tp / (tp + fn) if (tp + fn) else None
        idf1 = (2.0 * tp / ((2.0 * tp) + fp + fn)) if ((2.0 * tp) + fp + fn) else None
        deta = tp / (tp + fp + fn) if (tp + fp + fn) else None
        loca = statistics.fmean(row.own_iou for row in correct_rows) if correct_rows else None

        pair_rows: Dict[Tuple[int, int], List[IdentityObservation]] = {}
        for row in rows:
            pair_rows.setdefault((row.tracker_id, row.gt_track_id), []).append(row)
        weighted_assoc = 0.0
        assoc_weight = 0
        for pair_group in pair_rows.values():
            pair_tp = sum(1 for row in pair_group if correct_at_threshold(row, threshold))
            if pair_tp == 0:
                continue
            pair_fn = sum(1 for row in pair_group if row.gt_visible and not correct_at_threshold(row, threshold))
            pair_fp = sum(1 for row in pair_group if prediction_present(row) and not correct_at_threshold(row, threshold))
            pair_assoc = pair_tp / (pair_tp + pair_fp + pair_fn) if (pair_tp + pair_fp + pair_fn) else 0.0
            weighted_assoc += pair_tp * pair_assoc
            assoc_weight += pair_tp
        assa = weighted_assoc / assoc_weight if assoc_weight else None
        hota = math.sqrt(deta * assa) if deta is not None and assa is not None else None
        return {
            "threshold": threshold,
            "gt_count": gt_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "id_switches": idsw,
            "mota": mota,
            "id_precision": id_precision,
            "id_recall": id_recall,
            "idf1": idf1,
            "deta": deta,
            "assa": assa,
            "hota": hota,
            "loca": loca,
        }

    grouped: Dict[Tuple[str, str, str, str, str, int], List[IdentityObservation]] = {}
    for row in identity_rows:
        key = (
            row.sequence,
            row.lorat_config,
            row.execution_mode,
            row.reid_mode,
            row.v9_diagnostic_mode,
            row.target_tracks,
        )
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, execution_mode, reid_mode, diagnostic_mode, target_tracks), rows in sorted(grouped.items()):
        threshold_rows = [threshold_metrics(rows, threshold) for threshold in thresholds]
        iou50 = threshold_metrics(rows, summary_threshold)

        hota_values = [float(row["hota"]) for row in threshold_rows if row.get("hota") is not None]
        deta_values = [float(row["deta"]) for row in threshold_rows if row.get("deta") is not None]
        assa_values = [float(row["assa"]) for row in threshold_rows if row.get("assa") is not None]
        loca_values = [float(row["loca"]) for row in threshold_rows if row.get("loca") is not None]
        summaries.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "execution_mode": execution_mode,
                "reid_mode": reid_mode,
                "v9_diagnostic_mode": diagnostic_mode,
                "target_tracks": target_tracks,
                "actual_tracks": max((row.actual_tracks for row in rows), default=0),
                "samples": len(rows),
                "visible_gt_count": iou50["gt_count"],
                "mot_iou_threshold": summary_threshold,
                "mota_iou50": iou50["mota"],
                "idf1_iou50": iou50["idf1"],
                "id_precision_iou50": iou50["id_precision"],
                "id_recall_iou50": iou50["id_recall"],
                "tp_iou50": iou50["tp"],
                "fp_iou50": iou50["fp"],
                "fn_iou50": iou50["fn"],
                "id_switches_iou50": iou50["id_switches"],
                "hota": statistics.fmean(hota_values) if hota_values else None,
                "deta": statistics.fmean(deta_values) if deta_values else None,
                "assa": statistics.fmean(assa_values) if assa_values else None,
                "loca": statistics.fmean(loca_values) if loca_values else None,
                "hota_iou50": iou50["hota"],
                "deta_iou50": iou50["deta"],
                "assa_iou50": iou50["assa"],
                "loca_iou50": iou50["loca"],
                "hota_thresholds": ",".join(f"{threshold:.2f}" for threshold in thresholds),
                "metric_scope": "selected_initialized_targets",
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


def summarize_controlled_occlusion(trial_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int, int], List[Dict[str, object]]] = {}
    for row in trial_rows:
        key = (
            str(row.get("sequence", "")),
            str(row.get("lorat_config", "")),
            str(row.get("reid_mode", "")),
            int(row.get("target_tracks", 0) or 0),
            int(row.get("duration_frames", 0) or 0),
        )
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for (sequence, lorat_config, reid_mode, target_tracks, duration_frames), rows in sorted(grouped.items()):
        valid_rows = [row for row in rows if int(row.get("valid_trial", 0) or 0)]
        denominator = valid_rows or rows
        survived_rows = [row for row in denominator if int(row.get("survived", 0) or 0)]
        recovered_rows = [row for row in denominator if int(row.get("recovered", 0) or 0)]
        recovery_delays = [
            float(row["frames_to_recover"])
            for row in recovered_rows
            if row.get("frames_to_recover") not in ("", None)
        ]
        post_ious = [
            float(row["post_mean_iou"])
            for row in denominator
            if row.get("post_mean_iou") not in ("", None)
        ]
        summaries.append(
            {
                "sequence": sequence,
                "lorat_config": lorat_config,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "duration_frames": duration_frames,
                "trials": len(rows),
                "valid_trials": len(valid_rows),
                "survived_trials": len(survived_rows),
                "recovered_trials": len(recovered_rows),
                "survival_rate": (len(survived_rows) / len(denominator)) if denominator else None,
                "recovery_rate": (len(recovered_rows) / len(denominator)) if denominator else None,
                "identity_lost_rate": (
                    sum(1 for row in denominator if int(row.get("identity_lost", 0) or 0)) / len(denominator)
                    if denominator
                    else None
                ),
                "mean_frames_to_recover": statistics.fmean(recovery_delays) if recovery_delays else None,
                "mean_post_occlusion_iou": statistics.fmean(post_ious) if post_ious else None,
                "identity_switch_after_count": sum(int(row.get("identity_switch_after", 0) or 0) for row in denominator),
                "track_lost_after_count": sum(int(row.get("track_lost_after", 0) or 0) for row in denominator),
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
    diagnostic_mode: str,
    target_tracks: int,
    actual_tracks: int,
    lines: Sequence[str],
) -> None:
    for row in parse_week2_proof_lines(lines):
        full_row: Dict[str, object] = {
            "sequence": sequence,
            "lorat_config": lorat_config,
            "reid_mode": reid_mode,
            "diagnostic_mode": diagnostic_mode,
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
        args.init_selection,
        args.init_min_area,
        args.init_max_area,
        args.init_track_id,
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
            str(getattr(args, "v9_diagnostic_mode", "normal") or "normal"),
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
            preview_writer.write(
                draw_tracks_with_ground_truth(
                    init_frame_image,
                    backend.tracks,
                    gt_by_frame,
                    tracker_to_gt_id,
                    init_frame,
                    args.min_visibility,
                    backend.backend_name,
                    draw_ground_truth=args.draw_ground_truth,
                    draw_all_ground_truth=args.draw_all_ground_truth,
                    draw_candidate_diagnostics=args.draw_candidate_diagnostics,
                )
            )

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
            set_backend_diagnostic_gt_boxes(
                backend,
                gt_by_frame,
                tracker_to_gt_id,
                frame_number,
                args.min_visibility,
            )
            update_started = time.perf_counter()
            backend.update(frame, frame_number)
            tracking_seconds += time.perf_counter() - update_started
            last_frame_number = frame_number
            area_rows, full_rows, id_rows = collect_sampled_observations(
                sequence_path.name,
                lorat_config,
                reid_mode,
                str(getattr(args, "v9_diagnostic_mode", "normal") or "normal"),
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
                preview_writer.write(
                    draw_tracks_with_ground_truth(
                        frame,
                        backend.tracks,
                        gt_by_frame,
                        tracker_to_gt_id,
                        frame_number,
                        args.min_visibility,
                        backend.backend_name,
                        status_lines,
                        draw_ground_truth=args.draw_ground_truth,
                        draw_all_ground_truth=args.draw_all_ground_truth,
                        candidate_diagnostics=getattr(backend, "last_candidate_diagnostics", ()),
                        draw_candidate_diagnostics=args.draw_candidate_diagnostics,
                    )
                )
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
            str(getattr(args, "v9_diagnostic_mode", "normal") or "normal"),
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
        profile_dinov2_crop_reid_ms_per_update=profile_ms_per_update(runtime_status, "dinov2_crop_reid"),
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
        dinov2_crop_reid_forward_calls=int(getattr(runtime_status, "crop_reid_forward_calls", 0)),
        dinov2_crop_reid_forward_items=int(getattr(runtime_status, "crop_reid_forward_items", 0)),
        max_dinov2_crop_reid_batch=int(getattr(runtime_status, "max_crop_reid_batch", 0)),
        assignment_conflict_rejections=int(getattr(runtime_status, "v8_assignment_conflict_rejections", 0)),
        assignment_conflict_reasons=str(getattr(runtime_status, "v8_assignment_conflict_reasons", "")),
        assignment_alt_rescue_attempts=int(getattr(runtime_status, "v8_assignment_alt_rescue_attempts", 0)),
        assignment_alt_rescue_hits=int(getattr(runtime_status, "v8_assignment_alt_rescue_hits", 0)),
        assignment_alt_rescue_rejects=str(getattr(runtime_status, "v8_assignment_alt_rescue_rejects", "")),
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
        profile_dinov2_crop_reid_ms_per_update=bench.optional_float(timing.profile_dinov2_crop_reid_ms_per_update),
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
        dinov2_crop_reid_forward_calls=timing.dinov2_crop_reid_forward_calls,
        dinov2_crop_reid_forward_items=timing.dinov2_crop_reid_forward_items,
        max_dinov2_crop_reid_batch=timing.max_dinov2_crop_reid_batch,
        assignment_conflict_rejections=timing.assignment_conflict_rejections,
        assignment_conflict_reasons=timing.assignment_conflict_reasons,
        assignment_alt_rescue_attempts=timing.assignment_alt_rescue_attempts,
        assignment_alt_rescue_hits=timing.assignment_alt_rescue_hits,
        assignment_alt_rescue_rejects=timing.assignment_alt_rescue_rejects,
    )
    print(
        f"v8 {sequence_path.name} {lorat_config} N={target_tracks}: "
        f"actual={actual_tracks}, frames={frames}, "
        f"fps={bench.optional_float(fps_tracking, 3)}, "
        f"track_ms_per_box={bench.optional_float(tracking_ms_per_bbox, 3)}, "
        f"iou50={bench.optional_float(iou50, 3)}, "
        f"shared_calls/frame={bench.optional_float(shared_calls_per_frame, 3)}, "
        f"head_items/update={bench.optional_float(object_items_per_update, 3)}, "
        f"conflicts={timing.assignment_conflict_rejections}, "
        f"alt_rescue={timing.assignment_alt_rescue_hits}/{timing.assignment_alt_rescue_attempts}",
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
        "profile_dinov2_crop_reid_ms_per_update",
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
        "dinov2_crop_reid_forward_calls",
        "dinov2_crop_reid_forward_items",
        "max_dinov2_crop_reid_batch",
        "assignment_conflict_rejections",
        "assignment_conflict_reasons",
        "assignment_alt_rescue_attempts",
        "assignment_alt_rescue_hits",
        "assignment_alt_rescue_rejects",
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
                    "profile_dinov2_crop_reid_ms_per_update": bench.optional_float(row.profile_dinov2_crop_reid_ms_per_update),
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
                    "dinov2_crop_reid_forward_calls": row.dinov2_crop_reid_forward_calls,
                    "dinov2_crop_reid_forward_items": row.dinov2_crop_reid_forward_items,
                    "max_dinov2_crop_reid_batch": row.max_dinov2_crop_reid_batch,
                    "assignment_conflict_rejections": row.assignment_conflict_rejections,
                    "assignment_conflict_reasons": row.assignment_conflict_reasons,
                    "assignment_alt_rescue_attempts": row.assignment_alt_rescue_attempts,
                    "assignment_alt_rescue_hits": row.assignment_alt_rescue_hits,
                    "assignment_alt_rescue_rejects": row.assignment_alt_rescue_rejects,
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
        "v9_diagnostic_mode",
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
                    "v9_diagnostic_mode": row.v9_diagnostic_mode,
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
        "v9_diagnostic_mode",
        "diagnostic_failure_reason",
        "search_window",
        "v9_search_anchor_bbox",
        "v9_search_anchor_source",
        "v9_search_anchor_age",
        "v9_last_accepted_bbox",
        "v9_last_accepted_frame",
        "v9_last_accepted_source",
        "v9_search_window_contains_final_center",
        "v9_search_window_contains_head_center",
        "previous_bbox",
        "predicted_bbox",
        "head_original_bbox",
        "head_original_confidence",
        "head_bbox",
        "head_confidence",
        "head_margin",
        "head_visibility",
        "candidate_visibility",
        "head_roi_tokens",
        "v9_scale_gate_state",
        "v9_scale_gate_reason",
        "v9_scale_gate_width_ratio",
        "v9_scale_gate_height_ratio",
        "v9_scale_gate_locked_bbox",
        "v9_scale_gate_locked_width_ratio",
        "v9_scale_gate_locked_height_ratio",
        "v9_scale_gate_suppressed_original",
        "v9_scale_gate_confidence_preserved",
        "v9_scale_candidate_original_score",
        "v9_scale_candidate_locked_score",
        "v9_scale_candidate_selected",
        "v9_crop_reid_allowed",
        "v9_accept_guard_state",
        "v9_local_owner_override",
        "v9_hold_source",
        "v9_local_health_ok",
        "v9_local_health_reason",
        "v9_local_health_tier",
        "v9_local_health_confidence_threshold",
        "v9_local_health_margin_threshold",
        "v9_local_health_motion",
        "v9_local_health_accepted_anchor_motion",
        "v9_local_health_path",
        "v9_local_health_continuity_score",
        "v9_local_health_identity_risk",
        "v9_local_health_visibility",
        "v9_association_stage",
        "reid_outcome",
        "reid_prevented_switch",
        "reid_caused_hold",
        "reid_recovered_lost",
        "reid_wrong_reattach",
        "reid_noop_bad_candidate_pool",
        "reid_skipped_healthy_local",
        "reid_next_best_attempted",
        "reid_next_best_accepted",
        "reid_next_best_source",
        "reid_next_best_reason",
        "v9_local_rescue_accept",
        "v9_local_rescue_reject_state",
        "v9_continuity_enabled",
        "v9_continuity_candidate_count",
        "v9_continuity_applied",
        "v9_continuity_reason",
        "v9_continuity_current_score",
        "v9_continuity_best_score",
        "v9_continuity_score",
        "v9_continuity_score_margin",
        "v9_continuity_selected_rank",
        "v9_continuity_selected_source",
        "v9_continuity_selected_bbox",
        "v9_continuity_current_bbox",
        "v9_continuity_current_source",
        "v9_continuity_head_score",
        "v9_continuity_margin_score",
        "v9_continuity_motion_score",
        "v9_continuity_accepted_anchor_motion",
        "v9_continuity_path_score",
        "v9_continuity_current_visibility",
        "v9_continuity_best_visibility",
        "v9_continuity_anchor_score",
        "v9_continuity_appearance_score",
        "v9_visibility_score",
        "v9_visibility_absent_penalty",
        "v9_continuity_other_anchor_score",
        "v9_continuity_negative_anchor_score",
        "v9_continuity_other_anchor_pressure",
        "v9_continuity_negative_anchor_pressure",
        "v9_continuity_identity_risk",
        "v9_continuity_identity_margin",
        "v9_continuity_scale_score",
        "v9_continuity_center_jump_penalty",
        "v9_continuity_local_reject",
        "v9_continuity_current_local_reject",
        "v9_continuity_best_local_reject",
        "v9_continuity_scale_gate_state",
        "v9_continuity_current_drift_risk",
        "v9_continuity_drift_risk",
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
        "v9_final_minus_head_iou_gap",
        "final_center_error_px",
        "final_center_error_gt_size",
        "final_area_ratio",
        "final_best_other_iou",
        "final_correct_object",
        "final_state",
        "assigned_source",
        "iou_failure_bucket",
        "iou_failure_stage",
        "oracle_candidate_count",
        "oracle_best_source",
        "oracle_best_iou",
        "oracle_best_confidence",
        "oracle_best_bbox",
        "oracle_runtime_iou_gap",
        "oracle_best_iou50",
        "oracle_best_iou30",
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
        "profile_dinov2_crop_reid_ms_per_update",
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
        "dinov2_crop_reid_forward_calls",
        "dinov2_crop_reid_forward_items",
        "max_dinov2_crop_reid_batch",
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
        association_stage_counts = Counter(str(row.get("v9_association_stage", "")) for row in group_rows)
        reid_outcome_counts = Counter(str(row.get("reid_outcome", "")) for row in group_rows if row.get("reid_outcome"))
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
                "mean_oracle_best_iou": mean_numeric(group_rows, "oracle_best_iou"),
                "oracle_iou50": true_rate(group_rows, "oracle_best_iou50"),
                "oracle_iou30": true_rate(group_rows, "oracle_best_iou30"),
                "mean_oracle_runtime_gap": mean_numeric(group_rows, "oracle_runtime_iou_gap"),
                "candidate_sources": ";".join(f"{source}:{count}" for source, count in sorted(source_counts.items())),
                "iou_failure_buckets": ";".join(f"{bucket}:{count}" for bucket, count in sorted(failure_counts.items())),
                "iou_failure_stages": ";".join(f"{stage}:{count}" for stage, count in sorted(stage_counts.items())),
                "v9_association_stages": ";".join(
                    f"{stage}:{count}" for stage, count in sorted(association_stage_counts.items())
                ),
                "reid_outcomes": ";".join(
                    f"{outcome}:{count}" for outcome, count in sorted(reid_outcome_counts.items())
                ),
                "reid_prevented_switch_count": sum(1 for row in group_rows if bool(row.get("reid_prevented_switch"))),
                "reid_caused_hold_count": sum(1 for row in group_rows if bool(row.get("reid_caused_hold"))),
                "reid_recovered_lost_count": sum(1 for row in group_rows if bool(row.get("reid_recovered_lost"))),
                "reid_wrong_reattach_count": sum(1 for row in group_rows if bool(row.get("reid_wrong_reattach"))),
                "reid_skipped_healthy_local_count": sum(
                    1 for row in group_rows if bool(row.get("reid_skipped_healthy_local"))
                ),
                "reid_next_best_attempt_count": sum(
                    1 for row in group_rows if bool(row.get("reid_next_best_attempted"))
                ),
                "reid_next_best_accept_count": sum(
                    1 for row in group_rows if bool(row.get("reid_next_best_accepted"))
                ),
                "v9_local_rescue_accept_count": sum(
                    1 for row in group_rows if bool(row.get("v9_local_rescue_accept"))
                ),
                "v9_continuity_applied_count": sum(
                    1 for row in group_rows if bool(row.get("v9_continuity_applied"))
                ),
                "v9_continuity_applied_rate": true_rate(group_rows, "v9_continuity_applied"),
                "mean_v9_continuity_score": mean_numeric(group_rows, "v9_continuity_score"),
                "mean_v9_continuity_margin": mean_numeric(group_rows, "v9_continuity_score_margin"),
            }
        )
    return summaries


def reid_changed_decision(row: Dict[str, object]) -> bool:
    if any(
        bool(row.get(field))
        for field in (
            "reid_prevented_switch",
            "reid_caused_hold",
            "reid_recovered_lost",
            "reid_wrong_reattach",
            "reid_next_best_attempted",
            "reid_next_best_accepted",
        )
    ):
        return True
    outcome = str(row.get("reid_outcome") or "").strip()
    return bool(outcome and outcome not in {"reid_skipped_healthy_local", "skipped_healthy_local"})


def summarize_v9_diagnostic_comparison(
    candidate_rows: Sequence[Dict[str, object]],
    identity_rows: Sequence[IdentityObservation],
) -> List[Dict[str, object]]:
    candidate_groups: Dict[Tuple[str, str, str, str, int], List[Dict[str, object]]] = {}
    for row in candidate_rows:
        key = (
            str(row.get("sequence", "")),
            str(row.get("lorat_config", "")),
            str(row.get("reid_mode", "")),
            str(row.get("v9_diagnostic_mode") or "normal"),
            int(row.get("target_tracks", 0) or 0),
        )
        candidate_groups.setdefault(key, []).append(row)

    identity_summary = {
        (
            str(row.get("sequence", "")),
            str(row.get("lorat_config", "")),
            str(row.get("reid_mode", "")),
            str(row.get("v9_diagnostic_mode") or "normal"),
            int(row.get("target_tracks", 0) or 0),
        ): row
        for row in summarize_identity(identity_rows)
    }
    normal_by_base: Dict[Tuple[str, str, str, int], Dict[str, object]] = {}
    rows_by_base: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, object]]] = {}
    for (sequence, config, reid_mode, mode, target_tracks), group in sorted(candidate_groups.items()):
        high_conf_bad_head = sum(
            1
            for row in group
            if float(row.get("head_confidence") or 0.0) >= 0.70 and float(row.get("head_iou") or 0.0) < 0.30
        )
        low_conf_good_head = sum(
            1
            for row in group
            if float(row.get("head_confidence") or 0.0) < 0.55 and float(row.get("head_iou") or 0.0) >= 0.50
        )
        base = (sequence, config, reid_mode, target_tracks)
        summary = {
            "sequence": sequence,
            "lorat_config": config,
            "reid_mode": reid_mode,
            "v9_diagnostic_mode": mode,
            "target_tracks": target_tracks,
            "samples": len(group),
            "mean_final_iou": mean_numeric(group, "final_iou"),
            "mean_head_iou": mean_numeric(group, "head_iou"),
            "mean_oracle_best_iou": mean_numeric(group, "oracle_best_iou"),
            "correct_object_rate": true_rate(group, "final_correct_object"),
            "head_minus_final_iou_gap": mean_numeric(
                [
                    {"gap": float(row.get("head_iou") or 0.0) - float(row.get("final_iou") or 0.0)}
                    for row in group
                    if row.get("head_iou") not in (None, "") and row.get("final_iou") not in (None, "")
                ],
                "gap",
            ),
            "high_conf_bad_head_count": high_conf_bad_head,
            "low_conf_good_head_count": low_conf_good_head,
            "reid_changed_decision_count": sum(1 for row in group if reid_changed_decision(row)),
            "identity_switches": identity_summary.get((sequence, config, reid_mode, mode, target_tracks), {}).get("identity_switches"),
            "track_loss_rate": identity_summary.get((sequence, config, reid_mode, mode, target_tracks), {}).get("track_loss_rate"),
        }
        rows_by_base.setdefault(base, {})[mode] = summary
        if mode == "normal":
            normal_by_base[base] = summary

    comparison_rows: List[Dict[str, object]] = []
    for base, mode_rows in sorted(rows_by_base.items()):
        normal = normal_by_base.get(base)
        gt_window = mode_rows.get("gt_window")
        for mode, summary in sorted(mode_rows.items()):
            row = dict(summary)
            row["gt_window_minus_normal_final_iou"] = None
            row["gt_window_minus_normal_correct_rate"] = None
            if normal is not None and gt_window is not None:
                final_normal = normal.get("mean_final_iou")
                final_gt = gt_window.get("mean_final_iou")
                correct_normal = normal.get("correct_object_rate")
                correct_gt = gt_window.get("correct_object_rate")
                if final_normal is not None and final_gt is not None:
                    row["gt_window_minus_normal_final_iou"] = float(final_gt) - float(final_normal)
                if correct_normal is not None and correct_gt is not None:
                    row["gt_window_minus_normal_correct_rate"] = float(correct_gt) - float(correct_normal)
            comparison_rows.append(row)
    return comparison_rows


def write_summary_md(
    path: Path,
    args: argparse.Namespace,
    label: str,
    paths: V8OutputPaths,
    timing_rows: Sequence[V8TimingResult],
    area_rows: Sequence[bench.AreaSummary],
    identity_rows: Sequence[IdentityObservation],
    candidate_rows: Sequence[Dict[str, object]],
    controlled_occlusion_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# LoRAT V9 Benchmark Summary",
        "",
        "## Run",
        "",
        f"- Run label: `{label}`",
        f"- Execution mode: `{EXECUTION_MODE}`",
        f"- Device: `{args.device}`",
        f"- GPU profile label: `{args.gpu_profile}`",
        f"- Max frames per case: `{args.max_frames if args.max_frames > 0 else 'full sequence'}`",
        f"- Track counts: `{','.join(str(value) for value in parse_track_counts(args))}`",
        f"- Initialization selection: `{args.init_selection}`",
        f"- Initialization area window: min `{args.init_min_area}` px, max `{args.init_max_area if args.init_max_area > 0 else 'disabled'}` px",
        f"- Explicit initialization GT IDs: `{','.join(str(value) for value in args.init_track_id) if args.init_track_id else 'none'}`",
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
        "- ReID-on mode extracts explicit DINOv2 crop embeddings by batching candidate image crops through the LoRA-adapted DINOv2 backbone, then compares them with per-track crop-memory banks.",
        "- Identity switch counts a sampled frame where the tracker box matches a different visible GT object than the initialized GT object.",
        "- Track loss counts a sampled frame where the track is marked lost/not-ok or no visible GT object reaches the identity IoU threshold.",
        "- Steady tracking frames-until-failure is computed per tracker as visible sampled frames correctly matched to the initialized GT identity before the first wrong-object, identity-switch, or track-lost event.",
        "- N1 frames until first failure counts visible sampled frames where the single initialized object remains correctly matched before the first wrong-object, identity-switch, or track-lost event.",
        "- Paper-style MOT metrics are computed over initialized selected targets. `MOTA` and `IDF1` are reported at IoU 0.50; `HOTA` is averaged over IoU thresholds 0.05 through 0.95.",
        "- Natural occlusion diagnostics report the longest sampled uncertain/occluded gap that later returns to a correct object match.",
        "- Controlled occlusion survival masks the initialized object's current GT box for fixed durations, then measures whether the same track reattaches to the same GT identity within the recovery window.",
        f"- 25 FPS capacity rule: maximum actual N with tracking FPS >= `{args.fps_threshold}`.",
        "- Week 2 proof columns show one shared frame-backbone call per tracked frame and one batched object-head operation whose item count scales with object count.",
        "- Tracker FPS profile values are elapsed milliseconds per tracked update frame, aggregated from tracker-side timers.",
        "",
        "## Output Files",
        "",
        f"- Timing and memory: `{paths.timing_csv}`",
        f"- Area reliability: `{paths.area_csv}`",
        f"- Sampled area observations: `{paths.observations_csv}`",
        f"- Identity observations: `{paths.identity_csv}`",
        f"- Identity/ReID summary: `{paths.identity_summary_csv}`",
        f"- Paper-style MOT metrics: `{paths.mot_metrics_csv}`",
        f"- Natural occlusion diagnostics: `{paths.occlusion_survival_csv}`",
        f"- Controlled occlusion trials: `{paths.controlled_occlusion_trials_csv}`",
        f"- Controlled occlusion survival: `{paths.controlled_occlusion_survival_csv}`",
        f"- Week 2 shared-backbone proof: `{paths.week2_proof_csv}`",
        f"- Candidate/oracle diagnostics: `{paths.candidate_diagnostics_csv}`",
        f"- Debug log: `{paths.debug_csv}`",
    ]
    if args.full_area_observations:
        lines.append(f"- Every-frame area observations: `{paths.full_observations_csv}`")
    lines.extend(["", "## Timing / Memory", ""])
    lines.append(
        "| Sequence | Config | ReID Mode | N Target | N Actual | FPS Track | Track ms/box | Mean IoU | IoU@0.50 | Peak GPU Reserved MB | Shared Calls/Frame | Head Items/Update | DINOv2 Crop Calls | DINOv2 Crop Items | Week2 Shared OK | Week2 Head OK | 25 FPS |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in timing_rows:
        lines.append(
            f"| {row.sequence} | {row.lorat_config} | {row.reid_mode} | {row.target_tracks} | {row.actual_tracks} | "
            f"{bench.optional_float(row.fps_tracking, 3)} | {bench.optional_float(row.tracking_ms_per_bbox, 3)} | "
            f"{bench.optional_float(row.mean_iou, 3)} | {bench.optional_float(row.iou50, 3)} | "
            f"{bench.optional_float(row.gpu_memory_peak_reserved_mb, 1)} | "
            f"{bench.optional_float(row.shared_backbone_calls_per_frame, 3)} | "
            f"{bench.optional_float(row.object_head_items_per_update_frame, 3)} | "
            f"{row.dinov2_crop_reid_forward_calls} | "
            f"{row.dinov2_crop_reid_forward_items} | "
            f"{bench.optional_float(row.proof_shared_backbone_ok_rate, 3)} | "
            f"{bench.optional_float(row.proof_batched_head_ok_rate, 3)} | "
            f"{row.fps_sustains_25} |"
        )
    lines.extend(["", "## v8 FPS Profile", ""])
    lines.append(
        "| Sequence | Config | ReID Mode | N Target | Candidate transfer ms/update | Candidate decode ms/update | Feature-template ms/update | Fusion ms/update | Shared ROI ReID ms/update | DINOv2 crop ReID ms/update | Identity ms/update | Accept ms/update | Hold ms/update | Feature refresh ms/update | Debug/proof ms/update | Unbucketed ms/update |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
            f"{bench.optional_float(row.profile_dinov2_crop_reid_ms_per_update, 3)} | "
            f"{bench.optional_float(identity_ms, 3)} | "
            f"{bench.optional_float(row.profile_accept_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_hold_ms_per_update, 3)} | "
            f"{bench.optional_float(row.profile_appearance_refresh_ms_per_update, 3)} | "
            f"{bench.optional_float(debug_ms, 3)} | "
            f"{bench.optional_float(row.profile_unbucketed_ms_per_update, 3)} |"
        )
    candidate_summary = summarize_candidate_diagnostics(candidate_rows)
    if candidate_summary:
        lines.extend(["", "## Candidate Oracle Diagnostics", ""])
        lines.append(
            "| Sequence | Config | ReID Mode | N Target | Samples | Head Mean IoU | Head IoU@0.50 | Top-5 Best Mean IoU | Top-5 IoU@0.50 | Top-5 IoU@0.30 | Template Attempt Rate | Template Mean IoU | Fused Mean IoU | Assigned Mean IoU | Final Mean IoU | Final IoU@0.50 | Final Correct Object | Oracle Best IoU | Oracle IoU@0.50 | Oracle IoU@0.30 | Oracle-Final Gap | ReID Outcomes | Prevented Switch | Caused Hold | Recovered Lost | Wrong Reattach | Skipped Healthy | Next-Best Accept | V9 Local Rescue | Continuity Applies | Continuity Rate | Continuity Score | Continuity Margin | Association Stages | Failure Buckets | Failure Stages | Candidate Sources |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"
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
                f"{bench.optional_float(row['mean_oracle_best_iou'], 3)} | "
                f"{bench.optional_float(row['oracle_iou50'], 3)} | {bench.optional_float(row['oracle_iou30'], 3)} | "
                f"{bench.optional_float(row['mean_oracle_runtime_gap'], 3)} | "
                f"{row['reid_outcomes']} | {row['reid_prevented_switch_count']} | "
                f"{row['reid_caused_hold_count']} | {row['reid_recovered_lost_count']} | "
                f"{row['reid_wrong_reattach_count']} | {row['reid_skipped_healthy_local_count']} | "
                f"{row['reid_next_best_accept_count']} | {row['v9_local_rescue_accept_count']} | "
                f"{row['v9_continuity_applied_count']} | {bench.optional_float(row['v9_continuity_applied_rate'], 3)} | "
                f"{bench.optional_float(row['mean_v9_continuity_score'], 3)} | "
                f"{bench.optional_float(row['mean_v9_continuity_margin'], 3)} | "
                f"{row['v9_association_stages']} | "
                f"{row['iou_failure_buckets']} | {row['iou_failure_stages']} | "
                f"{row['candidate_sources']} |"
            )
    v9_comparison = summarize_v9_diagnostic_comparison(candidate_rows, identity_rows)
    if v9_comparison:
        lines.extend(["", "## V9 Selected-Target Diagnostic Comparison", ""])
        lines.append(
            "| Sequence | Config | ReID Mode | Diagnostic Mode | N Target | Samples | Final IoU | Head IoU | Oracle IoU | Correct Object | ID Switches | Track-Loss Rate | GT Window - Normal IoU | GT Window - Normal Correct | Head-Final Gap | High-Conf Bad Head | Low-Conf Good Head | ReID Changed Decisions |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in v9_comparison:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | "
                f"{row['v9_diagnostic_mode']} | {row['target_tracks']} | {row['samples']} | "
                f"{bench.optional_float(row['mean_final_iou'], 3)} | "
                f"{bench.optional_float(row['mean_head_iou'], 3)} | "
                f"{bench.optional_float(row['mean_oracle_best_iou'], 3)} | "
                f"{bench.optional_float(row['correct_object_rate'], 3)} | "
                f"{'' if row['identity_switches'] is None else row['identity_switches']} | "
                f"{bench.optional_float(row['track_loss_rate'], 3)} | "
                f"{bench.optional_float(row['gt_window_minus_normal_final_iou'], 3)} | "
                f"{bench.optional_float(row['gt_window_minus_normal_correct_rate'], 3)} | "
                f"{bench.optional_float(row['head_minus_final_iou_gap'], 3)} | "
                f"{row['high_conf_bad_head_count']} | {row['low_conf_good_head_count']} | "
                f"{row['reid_changed_decision_count']} |"
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
        lines.append("| Sequence | Config | ReID Mode | Diagnostic Mode | N Target | Samples | Visible | Hidden | Correct Rate | ID Switches | Track-Loss Rate | Jump Rate | Occluded Rate | Mean IoU | Steady Mean Frames Until Failure | Steady Min Frames Until Failure | Steady Max Longest Correct Run | N1 Frames Until First Failure | N1 First Failure | N1 Longest Correct Run |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in identity_summary:
            first_failure = ""
            if row.get("single_object_first_failure_reason"):
                first_failure = f"{row['single_object_first_failure_reason']}@{row.get('single_object_first_failure_offset', '')}"
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | "
                f"{row['v9_diagnostic_mode']} | {row['target_tracks']} | {row['samples']} | {row['visible_samples']} | {row['hidden_samples']} | "
                f"{bench.optional_float(row['correct_rate'], 3)} | {row['identity_switches']} | "
                f"{bench.optional_float(row['track_loss_rate'], 3)} | {bench.optional_float(row['jump_rate'], 3)} | "
                f"{bench.optional_float(row['occluded_rate'], 3)} | {bench.optional_float(row['mean_iou'], 3)} | "
                f"{bench.optional_float(row.get('steady_mean_frames_until_failure'), 1)} | "
                f"{row.get('steady_min_frames_until_failure', '')} | "
                f"{row.get('steady_max_longest_correct_run', '')} | "
                f"{row.get('single_object_frames_until_first_failure', '')} | {first_failure} | "
                f"{row.get('single_object_longest_correct_run', '')} |"
            )
    mot_metrics_summary = summarize_mot_paper_metrics(identity_rows)
    if mot_metrics_summary:
        lines.extend(["", "## Paper-Style MOT Metrics", ""])
        lines.append("| Sequence | Config | ReID Mode | Diagnostic Mode | N Target | MOTA@0.50 | IDF1@0.50 | HOTA | DetA | AssA | LocA | TP@0.50 | FP@0.50 | FN@0.50 | IDSW@0.50 | Scope |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in mot_metrics_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | "
                f"{row['v9_diagnostic_mode']} | {row['target_tracks']} | "
                f"{bench.optional_float(row['mota_iou50'], 3)} | {bench.optional_float(row['idf1_iou50'], 3)} | "
                f"{bench.optional_float(row['hota'], 3)} | {bench.optional_float(row['deta'], 3)} | "
                f"{bench.optional_float(row['assa'], 3)} | {bench.optional_float(row['loca'], 3)} | "
                f"{row['tp_iou50']} | {row['fp_iou50']} | {row['fn_iou50']} | {row['id_switches_iou50']} | "
                f"{row['metric_scope']} |"
            )
    occlusion_summary = summarize_occlusion_survival(identity_rows)
    if occlusion_summary:
        lines.extend(["", "## Natural Occlusion Diagnostic", ""])
        lines.append("| Sequence | Config | ReID Mode | N Target | Tracker | Longest Observed Occlusion Frames | Longest Survived Occlusion Frames | Recovered After Gap | Lost After Gap | Final State |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in occlusion_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | {row['target_tracks']} | "
                f"{row['tracker_id']} | {row['longest_occluded_frames_observed']} | "
                f"{row['longest_occlusion_survived_frames']} | {row['recovered_after_gap']} | "
                f"{row['lost_after_gap']} | {row['final_lifecycle_state']} |"
            )
    controlled_occlusion_summary = summarize_controlled_occlusion(controlled_occlusion_rows)
    if controlled_occlusion_summary:
        lines.extend(["", "## Controlled Occlusion Survival", ""])
        lines.append(
            "| Sequence | Config | ReID Mode | N Target | Forced Occlusion Frames | Trials | Valid Trials | Survival Rate | Recovery Rate | Identity Lost Rate | Mean Frames To Recover | Mean Post-IoU |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in controlled_occlusion_summary:
            lines.append(
                f"| {row['sequence']} | {row['lorat_config']} | {row['reid_mode']} | {row['target_tracks']} | "
                f"{row['duration_frames']} | {row['trials']} | {row['valid_trials']} | "
                f"{bench.optional_float(row['survival_rate'], 3)} | {bench.optional_float(row['recovery_rate'], 3)} | "
                f"{bench.optional_float(row['identity_lost_rate'], 3)} | "
                f"{bench.optional_float(row['mean_frames_to_recover'], 2)} | "
                f"{bench.optional_float(row['mean_post_occlusion_iou'], 3)} |"
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
    controlled_occlusion_rows: Sequence[Dict[str, object]],
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
            "v9_diagnostic_mode",
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
            "steady_tracker_count",
            "steady_mean_frames_until_failure",
            "steady_min_frames_until_failure",
            "steady_max_frames_until_failure",
            "steady_mean_longest_correct_run",
            "steady_min_longest_correct_run",
            "steady_max_longest_correct_run",
            "steady_survived_span_rate",
            "single_object_visible_frames",
            "single_object_correct_visible_frames",
            "single_object_frames_until_first_failure",
            "single_object_first_failure_frame",
            "single_object_first_failure_offset",
            "single_object_first_failure_reason",
            "single_object_longest_correct_run",
            "single_object_longest_failure_run",
            "single_object_survived_visible_span",
        ],
    )
    write_dict_rows_csv(
        paths.mot_metrics_csv,
        summarize_mot_paper_metrics(identity_rows),
        [
            "sequence",
            "lorat_config",
            "execution_mode",
            "reid_mode",
            "v9_diagnostic_mode",
            "target_tracks",
            "actual_tracks",
            "samples",
            "visible_gt_count",
            "mot_iou_threshold",
            "mota_iou50",
            "idf1_iou50",
            "id_precision_iou50",
            "id_recall_iou50",
            "tp_iou50",
            "fp_iou50",
            "fn_iou50",
            "id_switches_iou50",
            "hota",
            "deta",
            "assa",
            "loca",
            "hota_iou50",
            "deta_iou50",
            "assa_iou50",
            "loca_iou50",
            "hota_thresholds",
            "metric_scope",
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
    write_dict_rows_csv(
        paths.controlled_occlusion_trials_csv,
        controlled_occlusion_rows,
        [
            "sequence",
            "lorat_config",
            "execution_mode",
            "reid_mode",
            "target_tracks",
            "duration_frames",
            "trial_index",
            "occluded_gt_id",
            "occluded_tracker_id",
            "occlusion_start_frame",
            "occlusion_end_frame",
            "recovery_start_frame",
            "recovery_end_frame",
            "pre_iou",
            "pre_correct",
            "valid_trial",
            "recovered",
            "survived",
            "identity_lost",
            "first_recovery_frame",
            "frames_to_recover",
            "post_mean_iou",
            "post_iou50",
            "identity_switch_after",
            "track_lost_after",
            "final_iou",
            "final_state",
            "rule",
        ],
    )
    write_dict_rows_csv(
        paths.controlled_occlusion_survival_csv,
        summarize_controlled_occlusion(controlled_occlusion_rows),
        [
            "sequence",
            "lorat_config",
            "reid_mode",
            "target_tracks",
            "duration_frames",
            "trials",
            "valid_trials",
            "survived_trials",
            "recovered_trials",
            "survival_rate",
            "recovery_rate",
            "identity_lost_rate",
            "mean_frames_to_recover",
            "mean_post_occlusion_iou",
            "identity_switch_after_count",
            "track_lost_after_count",
        ],
    )
    write_week2_proof_csv(paths.week2_proof_csv, proof_rows)
    write_candidate_diagnostics_csv(paths.candidate_diagnostics_csv, candidate_rows)
    write_debug_csv(paths.debug_csv, debug_rows)
    write_summary_md(
        paths.summary_md,
        args,
        label,
        paths,
        timing_rows,
        area_rows,
        identity_rows,
        candidate_rows,
        controlled_occlusion_rows,
    )
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
    controlled_occlusion_rows: List[Dict[str, object]] = []
    proof_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []

    print(f"LoRAT V9 benchmark run label: {label}", flush=True)
    print(f"Output folder: {paths.run_root}", flush=True)
    print(f"Timing CSV: {paths.timing_csv}", flush=True)
    print(f"Area reliability CSV: {paths.area_csv}", flush=True)
    print(f"Sampled observations CSV: {paths.observations_csv}", flush=True)
    print(f"Identity CSV: {paths.identity_csv}", flush=True)
    print(f"Identity/ReID summary CSV: {paths.identity_summary_csv}", flush=True)
    print(f"Natural occlusion diagnostic CSV: {paths.occlusion_survival_csv}", flush=True)
    print(f"Controlled occlusion trials CSV: {paths.controlled_occlusion_trials_csv}", flush=True)
    print(f"Controlled occlusion survival CSV: {paths.controlled_occlusion_survival_csv}", flush=True)
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
    print(
        "Controlled occlusion durations: "
        f"{','.join(str(value) for value in parse_controlled_occlusion_durations(args)) or 'disabled'}",
        flush=True,
    )

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
        controlled_occlusion_rows,
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
                            controlled_occlusion_rows,
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
                            controlled_occlusion_rows,
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
                            controlled_occlusion_rows,
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
                        controlled_occlusion_rows,
                        proof_rows,
                        candidate_rows,
                        debug_rows,
                    )
                    if STOP_REQUESTED:
                        print("Stop requested; flushed partial V9 benchmark outputs and exiting cleanly.", flush=True)
                        return 0

    controlled_occlusion_rows.extend(
        run_controlled_occlusion_benchmark(
            args,
            sequences,
            configs,
            track_counts,
            reid_modes,
            debug_rows,
            paths.debug_csv,
        )
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
        controlled_occlusion_rows,
        proof_rows,
        candidate_rows,
        debug_rows,
    )
    if STOP_REQUESTED:
        print("Stop requested; flushed partial controlled occlusion outputs and exiting cleanly.", flush=True)
        return 0

    record_debug(
        debug_rows,
        paths.debug_csv,
        "run_complete",
        reason=(
            f"completed_cases={len(timing_rows)}; area_samples={len(area_observations)}; "
            f"identity_samples={len(identity_rows)}; controlled_occlusion_trials={len(controlled_occlusion_rows)}"
        ),
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
        controlled_occlusion_rows,
        proof_rows,
        candidate_rows,
        debug_rows,
    )
    print("Wrote LoRAT V9 benchmark files:", flush=True)
    print(f"  {paths.timing_csv}", flush=True)
    print(f"  {paths.area_csv}", flush=True)
    print(f"  {paths.observations_csv}", flush=True)
    print(f"  {paths.identity_csv}", flush=True)
    print(f"  {paths.identity_summary_csv}", flush=True)
    print(f"  {paths.occlusion_survival_csv}", flush=True)
    print(f"  {paths.controlled_occlusion_trials_csv}", flush=True)
    print(f"  {paths.controlled_occlusion_survival_csv}", flush=True)
    print(f"  {paths.week2_proof_csv}", flush=True)
    print(f"  {paths.candidate_diagnostics_csv}", flush=True)
    print(f"  {paths.debug_csv}", flush=True)
    print(f"  {paths.summary_md}", flush=True)
    if args.full_area_observations:
        print(f"  {paths.full_observations_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
