from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
VideoSource = Union[int, str]

# ---------------------------------------------------------------------------
# V5 editable defaults and shared data models
# ---------------------------------------------------------------------------

# Edit this block when tuning V5. Launch profiles should avoid passing these
# knobs so these constants remain the default source of truth.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lorat-gui-v5"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
DEFAULT_LORAT_ROOT = PROJECT_ROOT / "external" / "LoRAT-main"
DEFAULT_DANCETRACK_SEQUENCE = PROJECT_ROOT / "data" / "raw" / "DanceTrack" / "val" / "val" / "dancetrack0065"
DEFAULT_LORAT_SEARCH_AREA_FACTOR = 3.0
DEFAULT_LORAT_WINDOW_PENALTY = 0.60
DEFAULT_LORAT_STATE_UPDATE_MIN_SCORE = 0.30
DEFAULT_LORAT_STATE_UPDATE_MAX_CENTER_SHIFT = 0.85
DEFAULT_LORAT_STATE_UPDATE_MAX_AREA_CHANGE = 1.80
DEFAULT_LORAT_MEMORY_SLOTS = 5
DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL = 1
DEFAULT_LORAT_ACTIVE_SLOTS_PER_TRACK = 5
DEFAULT_LORAT_FIXED_BOX_SIZE = False
DEFAULT_LORAT_MIN_BOX_AREA = 1.0
DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME = 1.05
DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE = 0.30
DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE = 12
DEFAULT_LORAT_ACCEPT_MIN_SCORE = 0.20
DEFAULT_SHRINK_GUARD_WINDOW = 6
DEFAULT_SHRINK_GUARD_AREA_RATIO = 0.72
DEFAULT_SHRINK_GUARD_STEP_RATIO = 0.94
DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE = 0.70
DEFAULT_SHRINK_GUARD_MIN_REID = 0.50
DEFAULT_CROP_INFORMATION_MIN_SCORE = 0.12
DEFAULT_CROP_INFORMATION_MIN_PIXELS = 64
DEFAULT_IDENTITY_MIN_SCORE = 0.50
DEFAULT_IDENTITY_MIN_REID = 0.28
DEFAULT_IDENTITY_MIN_MOTION = 0.18
DEFAULT_IDENTITY_MIN_PATH = 0.40
DEFAULT_PATH_GATE_MIN_RELIABLE_POINTS = 10
DEFAULT_PATH_GATE_MIN_FRAME_SPAN = 10
DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU = 0.18
DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER = 0.68
DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN = 0.06
DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY = 0.55
DEFAULT_CENTER_PATH_DIRECTION_MIN_SPEED = 4.0
DEFAULT_CENTER_PATH_STATIONARY_RADIUS = 32.0
DEFAULT_CENTER_PATH_STATIONARY_STEP_FACTOR = 4.5
DEFAULT_CENTER_PATH_STATIONARY_BOX_FACTOR = 0.06
DEFAULT_CENTER_PATH_REVERSAL_STEP_FACTOR = 2.0
DEFAULT_CENTER_PATH_REVERSAL_MIN_COSINE = -0.45
DEFAULT_CENTER_PATH_REVERSAL_PENALTY = 0.65
DEFAULT_PATH_RECOVERY_AFTER_FRAMES = 2
DEFAULT_PATH_RECOVERY_MIN_CONFIDENCE = 0.45
DEFAULT_PATH_RECOVERY_MIN_REID = 0.72
DEFAULT_PATH_RECOVERY_MIN_MOTION = 0.55
DEFAULT_RECOVERY_INITIAL_ANCHOR_MIN = 0.0
DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE = 0.55
DEFAULT_OCCLUSION_MAX_FRAMES = 30
DEFAULT_OCCLUSION_IOU_THRESHOLD = 0.28
DEFAULT_OCCLUSION_VELOCITY_DAMPING = 0.65
DEFAULT_REID_RECOVERY_MIN_SCORE = 0.55
DEFAULT_REID_RECOVERY_MIN_REID = 0.68
DEFAULT_REID_RECOVERY_MIN_MOTION = 0.45
DEFAULT_REID_RECOVERY_MIN_CONFIDENCE = 0.02
DEFAULT_VIEW_CHANGE_MIN_SCORE = 0.40
DEFAULT_VIEW_CHANGE_MIN_MOTION = 0.60
DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE = 0.16
DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES = 2
MAX_LORAT_MEMORY_SLOTS = 16
V5_EXECUTION_MODE = "shared-backbone"

LORAT_WEIGHT_BY_CONFIG = {
    "B-224": PROJECT_ROOT / "models" / "lorat" / "base.bin",
    "B-378": PROJECT_ROOT / "models" / "lorat" / "base-378.bin",
    "L-224": PROJECT_ROOT / "models" / "lorat" / "large.bin",
    "L-378": PROJECT_ROOT / "models" / "lorat" / "large-378.bin",
    "g-224": PROJECT_ROOT / "models" / "lorat" / "giant.bin",
    "g-378": PROJECT_ROOT / "models" / "lorat" / "giant-378.bin",
}


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    color: Color
    ok: bool = True
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    confidence_baseline: Optional[float] = None
    lost_frames: int = 0
    tracker: Optional[object] = None
    previous_bbox: Optional[BBox] = None
    predicted_bbox: Optional[BBox] = None
    raw_bbox: Optional[BBox] = None
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)
    trajectory: List[Tuple[int, BBox]] = field(default_factory=list)
    reliable_trajectory: List[Tuple[int, BBox]] = field(default_factory=list)
    state: str = ""
    active_template_frame: Optional[int] = None
    assigned_source: str = ""
    active_lorat_slot: str = "initial"
    lorat_memory_slot_count: int = 0
    initial_bbox: Optional[BBox] = None
    trusted_size_bank: List[BBox] = field(default_factory=list)
    appearance_hist: Optional[np.ndarray] = None
    initial_appearance_hist: Optional[np.ndarray] = None
    appearance_bank: List[np.ndarray] = field(default_factory=list)
    appearance_updates: int = 0
    assignment_score: Optional[float] = None
    assignment_margin: Optional[float] = None
    reid_score: Optional[float] = None
    motion_score: Optional[float] = None
    path_score: Optional[float] = None
    source_score: Optional[float] = None
    initial_anchor_score: Optional[float] = None
    other_anchor_score: Optional[float] = None
    other_anchor_track_id: Optional[int] = None
    identity_margin: Optional[float] = None
    occlusion_track_id: Optional[int] = None
    occlusion_iou: Optional[float] = None
    kalman: Optional["BBoxKalmanFilter"] = None
    occluded_frames: int = 0
    last_reliable_bbox: Optional[BBox] = None
    last_reliable_frame: int = 0
    size_history: List[Tuple[int, BBox]] = field(default_factory=list)
    shrink_risk_frames: int = 0
    learning_held_frames: int = 0
    learning_block_reason: str = ""
    last_area_ratio: Optional[float] = None
    last_window_area_ratio: Optional[float] = None
    last_crop_info_score: Optional[float] = None
    last_crop_edge_density: Optional[float] = None
    last_crop_laplacian_var: Optional[float] = None
    last_crop_contrast: Optional[float] = None


@dataclass
class LoRATMemorySlot:
    task_id: int
    track_id: int
    label: str
    frame_number: int
    bbox: BBox
    confidence: Optional[float] = None
    raw_confidence: Optional[float] = None
    confidence_baseline: Optional[float] = None
    last_refresh_frame: int = 0
    active: bool = True

    anchor_frame_number: int = 0
    anchor_bbox: Optional[BBox] = None


@dataclass
class LoRATSlotOutput:
    source_track_id: int
    slot: LoRATMemorySlot
    bbox: BBox
    confidence: Optional[float]
    appearance_hist: Optional[np.ndarray] = None


@dataclass
class RuntimeStatus:
    fps: float = 0.0
    last_frame_seconds: float = 0.0
    active_objects: int = 0
    evaluator_calls: int = 0
    evaluator_tasks: int = 0
    max_evaluator_batch: int = 0
    model_forward_calls: int = 0
    model_forward_items: int = 0
    max_model_forward_batch: int = 0
    fusion_forward_calls: int = 0
    fusion_forward_items: int = 0
    max_fusion_forward_batch: int = 0
    gpu_name: str = ""
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_peak_allocated_mb: Optional[float] = None
    gpu_peak_reserved_mb: Optional[float] = None
    gating_decisions: int = 0
    gating_primary_decisions: int = 0
    gating_recovery_decisions: int = 0
    gating_selected_slot_items: int = 0
    gating_avg_slots_per_decision: float = 0.0
    gating_recovery_reasons: str = ""
    shared_frame_backbone_calls: int = 0
    shared_frame_backbone_items: int = 0
    object_head_batches: int = 0
    object_head_items: int = 0
    max_object_head_batch: int = 0
    object_head_roi_tokens: int = 0


@dataclass(frozen=True)
class CropInformation:
    score: float = 0.0
    edge_density: float = 0.0
    laplacian_var: float = 0.0
    contrast: float = 0.0
    pixel_count: int = 0


@dataclass(frozen=True)
class IdentityScore:
    total: float
    appearance: float
    motion: float
    path: float
    source: float
    confidence: float
    iou: float
    initial_anchor: float
    other_anchor: float
    other_track_id: Optional[int]
    identity_margin: float
    occlusion_track_id: Optional[int]
    occlusion_iou: float


@dataclass(frozen=True)
class IdentityAssignment:
    track: TrackState
    output: LoRATSlotOutput
    score: IdentityScore
    assignment_margin: float


class BBoxKalmanFilter:
    """Constant-velocity Kalman filter over box center, size, and their velocities."""

    def __init__(self, bbox: BBox):
        x, y, w, h = clamp_bbox_size(bbox)
        center_x, center_y = bbox_center((x, y, w, h))
        self.state = np.array(
            [center_x, center_y, w, h, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.covariance = np.diag([25.0, 25.0, 16.0, 16.0, 100.0, 100.0, 36.0, 36.0]).astype(np.float32)
        self.transition = np.eye(8, dtype=np.float32)
        self.transition[0, 4] = 1.0
        self.transition[1, 5] = 1.0
        self.transition[2, 6] = 1.0
        self.transition[3, 7] = 1.0
        self.observation = np.zeros((4, 8), dtype=np.float32)
        self.observation[0, 0] = 1.0
        self.observation[1, 1] = 1.0
        self.observation[2, 2] = 1.0
        self.observation[3, 3] = 1.0
        self.process_noise = np.diag([2.0, 2.0, 1.0, 1.0, 8.0, 8.0, 3.0, 3.0]).astype(np.float32)

    def predict(self) -> BBox:
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.to_bbox()

    def update(self, bbox: BBox, confidence: Optional[float] = None) -> BBox:
        x, y, w, h = clamp_bbox_size(bbox)
        center_x, center_y = bbox_center((x, y, w, h))
        measurement = np.array([center_x, center_y, w, h], dtype=np.float32)
        confidence_value = 0.5 if confidence is None else max(0.05, min(1.0, float(confidence)))
        measurement_noise_scale = 1.0 / confidence_value
        measurement_noise = (
            np.diag([18.0, 18.0, 10.0, 10.0]).astype(np.float32) * measurement_noise_scale
        )
        innovation = measurement - (self.observation @ self.state)
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T + measurement_noise
        )
        kalman_gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + (kalman_gain @ innovation)
        self.covariance = (np.eye(8, dtype=np.float32) - (kalman_gain @ self.observation)) @ self.covariance
        return self.to_bbox()

    def to_bbox(self) -> BBox:
        center_x, center_y, w, h = [float(value) for value in self.state[:4]]
        w = max(1.0, w)
        h = max(1.0, h)
        return center_x - (w / 2.0), center_y - (h / 2.0), w, h


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class FrameSource:
    name: str
    fps: float
    length: Optional[int]

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class VideoCaptureSource(FrameSource):
    def __init__(self, source: VideoSource):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps if fps and fps > 1 else 30.0
        self.length = frame_count if frame_count > 0 else None
        self.name = str(source)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


class ImageSequenceSource(FrameSource):
    def __init__(self, sequence_path: Path, fps: float):
        image_dir = sequence_path / "img1" if (sequence_path / "img1").is_dir() else sequence_path
        image_paths = sorted(
            path
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in image_dir.glob(suffix)
        )
        if not image_paths:
            raise RuntimeError(f"No image frames found in: {image_dir}")

        self.sequence_path = sequence_path
        self.image_paths = image_paths
        self.index = 0
        self.fps = fps
        self.length = len(image_paths)
        self.name = sequence_path.name

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= len(self.image_paths):
            return False, None

        frame = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1
        return frame is not None, frame


# ---------------------------------------------------------------------------
# Lightweight identity memory and assignment
# ---------------------------------------------------------------------------

class LightweightIdentityArbitrator:
    """Small ReID/Hungarian layer that protects identity without steering LoRAT every frame."""

    def __init__(
        self,
        enabled: bool = True,
        min_score: float = DEFAULT_IDENTITY_MIN_SCORE,
        min_reid: float = DEFAULT_IDENTITY_MIN_REID,
        min_motion: float = DEFAULT_IDENTITY_MIN_MOTION,
        min_path: float = DEFAULT_IDENTITY_MIN_PATH,
        appearance_update_rate: float = 0.06,
        appearance_bank_size: int = 12,
        memory_min_confidence: float = DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE,
        view_change_min_score: float = DEFAULT_VIEW_CHANGE_MIN_SCORE,
        view_change_min_motion: float = DEFAULT_VIEW_CHANGE_MIN_MOTION,
        view_change_min_confidence: float = DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE,
        view_change_max_lost_frames: int = DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES,
    ):
        self.enabled = enabled
        self.min_score = max(0.0, float(min_score))
        self.min_reid = max(0.0, float(min_reid))
        self.min_motion = max(0.0, float(min_motion))
        self.min_path = max(0.0, min(1.0, float(min_path)))
        self.appearance_update_rate = max(0.0, min(1.0, float(appearance_update_rate)))
        self.appearance_bank_size = max(1, int(appearance_bank_size))
        self.memory_min_confidence = max(0.0, min(1.0, float(memory_min_confidence)))
        self.view_change_min_score = max(0.0, min(1.0, float(view_change_min_score)))
        self.view_change_min_motion = max(0.0, min(1.0, float(view_change_min_motion)))
        self.view_change_min_confidence = max(0.0, min(1.0, float(view_change_min_confidence)))
        self.view_change_max_lost_frames = max(0, int(view_change_max_lost_frames))

    def initialize_track(self, track: TrackState, frame: np.ndarray) -> None:
        hist = extract_reid_histogram(frame, track.bbox)
        if hist is None:
            return
        track.initial_appearance_hist = hist.copy()
        track.appearance_hist = hist.copy()
        track.appearance_bank = [hist.copy()]
        track.appearance_updates = 1

    def resolve(
        self,
        tracks: Sequence[TrackState],
        outputs: Sequence[LoRATSlotOutput],
        frame: Optional[np.ndarray],
    ) -> List[IdentityAssignment]:
        if not self.enabled:
            return self._owned_only_assignments(tracks, outputs)
        if not tracks or not outputs:
            return []

        prepared_outputs = [
            self._with_appearance(output, frame)
            for output in outputs
        ]
        score_details = [
            [self.score(track, output, tracks) for output in prepared_outputs]
            for track in tracks
        ]
        score_matrix = [[score.total for score in row] for row in score_details]
        assignments = []
        assignment_floor = min(self.min_score, self.view_change_min_score)
        for row, col, score in solve_assignment(score_matrix, assignment_floor):
            score_parts = score_details[row][col]
            track = tracks[row]
            output = prepared_outputs[col]
            is_view_change = self.is_view_change_candidate(track, output, score_parts)
            if score_parts.total < self.min_score and not is_view_change:
                continue
            if track_has_appearance(track) and score_parts.appearance < self.min_reid and not is_view_change:
                continue
            assignments.append(
                IdentityAssignment(
                    track=track,
                    output=output,
                    score=score_parts,
                    assignment_margin=assignment_margin(score_matrix[row], col),
                )
            )
        return assignments

    def score(
        self,
        track: TrackState,
        output: LoRATSlotOutput,
        all_tracks: Sequence[TrackState] = (),
    ) -> IdentityScore:
        confidence = 0.5 if output.confidence is None else max(0.0, min(1.0, float(output.confidence)))
        appearance = 0.5
        initial_anchor = 0.5
        other_anchor = 0.0
        other_track_id: Optional[int] = None
        if output.appearance_hist is not None:
            appearance = track_appearance_similarity(track, output.appearance_hist)
            if track.initial_appearance_hist is not None:
                initial_anchor = histogram_similarity(track.initial_appearance_hist, output.appearance_hist)
            else:
                initial_anchor = appearance
            for other in all_tracks:
                if other.track_id == track.track_id or other.initial_appearance_hist is None:
                    continue
                other_score = histogram_similarity(other.initial_appearance_hist, output.appearance_hist)
                if other_score > other_anchor:
                    other_anchor = other_score
                    other_track_id = other.track_id

        predicted = kalman_prediction_reference(track)
        reference_diagonal = max(1.0, bbox_diagonal(track.bbox))
        motion = motion_affinity(predicted, output.bbox, reference_diagonal)
        path = center_path_affinity(track, output.bbox)
        iou = max(bbox_iou(track.bbox, output.bbox), bbox_iou(predicted, output.bbox))
        occlusion_track_id, occlusion_iou = strongest_track_overlap(track, output.bbox, all_tracks)
        source = 1.0 if output.source_track_id == track.track_id else 0.24

        identity_margin = initial_anchor - other_anchor
        anchor_conflict = (
            other_track_id is not None
            and occlusion_iou >= DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and other_anchor >= DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and identity_margin <= -DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )
        if anchor_conflict:
            source *= DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY
            motion *= DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY

        if output.source_track_id != track.track_id and appearance < max(0.30, self.min_reid):
            source *= 0.35
        if track_has_appearance(track) and appearance < self.min_reid:
            motion *= 0.45
        if motion < self.min_motion and confidence < 0.70:
            source *= 0.65
        if path < self.min_path and confidence < 0.75:
            source *= 0.55

        total = (
            (0.38 * appearance)
            + (0.22 * motion)
            + (0.18 * path)
            + (0.08 * source)
            + (0.10 * confidence)
            + (0.04 * iou)
        )
        return IdentityScore(
            total=max(0.0, min(1.0, float(total))),
            appearance=appearance,
            motion=motion,
            path=path,
            source=source,
            confidence=confidence,
            iou=iou,
            initial_anchor=initial_anchor,
            other_anchor=other_anchor,
            other_track_id=other_track_id,
            identity_margin=identity_margin,
            occlusion_track_id=occlusion_track_id,
            occlusion_iou=occlusion_iou,
        )

    def commit_track_memory(
        self,
        track: TrackState,
        output: LoRATSlotOutput,
        assignment: IdentityAssignment,
        frame: Optional[np.ndarray],
    ) -> None:
        hist = output.appearance_hist
        if hist is None and frame is not None:
            hist = extract_reid_histogram(frame, track.bbox)
        if hist is None:
            return
        is_view_change = self.is_view_change_candidate(track, output, assignment.score)
        if assignment.score.confidence < self.memory_min_confidence:
            return
        if track_has_appearance(track) and assignment.score.appearance < max(self.min_reid, 0.30) and not is_view_change:
            return
        if assignment.score.motion < self.min_motion and not is_view_change:
            return
        if assignment.score.total < max(self.min_score, 0.50) and not is_view_change:
            return

        if track.initial_appearance_hist is None:
            track.initial_appearance_hist = hist.copy()
        if track.appearance_hist is None:
            track.appearance_hist = hist.copy()
        else:
            update_rate = self.appearance_update_rate * (0.5 if is_view_change else 1.0)
            track.appearance_hist = (
                ((1.0 - update_rate) * track.appearance_hist)
                + (update_rate * hist)
            ).astype(np.float32)
            norm = float(np.linalg.norm(track.appearance_hist))
            if norm > 0:
                track.appearance_hist /= norm

        if not track.appearance_bank or histogram_similarity(track.appearance_bank[-1], hist) < 0.985:
            track.appearance_bank.append(hist.copy())
            if len(track.appearance_bank) > self.appearance_bank_size:
                del track.appearance_bank[: len(track.appearance_bank) - self.appearance_bank_size]
        track.appearance_updates += 1

    def is_view_change_candidate(
        self,
        track: TrackState,
        output: LoRATSlotOutput,
        score: IdentityScore,
    ) -> bool:
        return (
            output.source_track_id == track.track_id
            and track.lost_frames <= self.view_change_max_lost_frames
            and score.total >= self.view_change_min_score
            and score.motion >= self.view_change_min_motion
            and score.path >= self.min_path
            and score.confidence >= self.view_change_min_confidence
        )

    def _with_appearance(self, output: LoRATSlotOutput, frame: Optional[np.ndarray]) -> LoRATSlotOutput:
        if output.appearance_hist is not None or frame is None:
            return output
        return LoRATSlotOutput(
            source_track_id=output.source_track_id,
            slot=output.slot,
            bbox=output.bbox,
            confidence=output.confidence,
            appearance_hist=extract_reid_histogram(frame, output.bbox),
        )

    def _owned_only_assignments(
        self,
        tracks: Sequence[TrackState],
        outputs: Sequence[LoRATSlotOutput],
    ) -> List[IdentityAssignment]:
        assignments = []
        outputs_by_track: Dict[int, List[LoRATSlotOutput]] = {}
        for output in outputs:
            outputs_by_track.setdefault(output.source_track_id, []).append(output)

        for track in tracks:
            owned_outputs = outputs_by_track.get(track.track_id)
            if not owned_outputs:
                continue
            output = max(
                owned_outputs,
                key=lambda item: 0.5 if item.confidence is None else max(0.0, min(1.0, float(item.confidence))),
            )
            score = self.score(track, output, tracks)
            assignments.append(
                IdentityAssignment(track=track, output=output, score=score, assignment_margin=1.0)
            )
        return assignments


# ---------------------------------------------------------------------------
# V5 LoRAT-backed MOT engine
# ---------------------------------------------------------------------------

class LoRATMultiObjectTracker:
    backend_name = "LoRAT-v5-shared"

    def __init__(
        self,
        lorat_root: Path,
        config_name: str,
        weight_path: Path,
        device: str,
        max_tracks: int,
        track_batch_size: int,
        fps: Optional[float],
        sequence_length: Optional[int],
        sequence_name: str,
        disable_amp: bool,
        lorat_memory_slots: int = DEFAULT_LORAT_MEMORY_SLOTS,
        lorat_memory_refresh_interval: int = DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL,
        lorat_active_slots_per_track: int = DEFAULT_LORAT_ACTIVE_SLOTS_PER_TRACK,
        lorat_fixed_box_size: bool = DEFAULT_LORAT_FIXED_BOX_SIZE,
        lorat_min_box_area: float = DEFAULT_LORAT_MIN_BOX_AREA,
        lorat_max_area_change_per_frame: float = DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME,
        lorat_trusted_size_floor_scale: float = DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE,
        lorat_memory_min_score: float = 0.55,
        lorat_accept_min_score: float = DEFAULT_LORAT_ACCEPT_MIN_SCORE,
        shrink_guard_window: int = DEFAULT_SHRINK_GUARD_WINDOW,
        shrink_guard_area_ratio: float = DEFAULT_SHRINK_GUARD_AREA_RATIO,
        shrink_guard_step_ratio: float = DEFAULT_SHRINK_GUARD_STEP_RATIO,
        shrink_guard_min_confidence: float = DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE,
        shrink_guard_min_reid: float = DEFAULT_SHRINK_GUARD_MIN_REID,
        crop_information_min_score: float = DEFAULT_CROP_INFORMATION_MIN_SCORE,
        crop_information_min_pixels: int = DEFAULT_CROP_INFORMATION_MIN_PIXELS,
        lorat_slot_capacity: int = 0,
        expected_tracks: int = 0,
        lorat_search_area_factor: Optional[float] = DEFAULT_LORAT_SEARCH_AREA_FACTOR,
        lorat_window_penalty: Optional[float] = DEFAULT_LORAT_WINDOW_PENALTY,
        lorat_state_update_min_score: Optional[float] = DEFAULT_LORAT_STATE_UPDATE_MIN_SCORE,
        lorat_state_update_max_center_shift: Optional[float] = DEFAULT_LORAT_STATE_UPDATE_MAX_CENTER_SHIFT,
        lorat_state_update_max_area_change: Optional[float] = DEFAULT_LORAT_STATE_UPDATE_MAX_AREA_CHANGE,
        identity_arbitration: bool = True,
        identity_min_score: float = DEFAULT_IDENTITY_MIN_SCORE,
        identity_min_reid: float = DEFAULT_IDENTITY_MIN_REID,
        identity_min_motion: float = DEFAULT_IDENTITY_MIN_MOTION,
        identity_min_path: float = DEFAULT_IDENTITY_MIN_PATH,
        identity_bank_size: int = 12,
        identity_memory_min_confidence: float = DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE,
        occlusion_max_frames: int = DEFAULT_OCCLUSION_MAX_FRAMES,
        occlusion_iou_threshold: float = DEFAULT_OCCLUSION_IOU_THRESHOLD,
        occlusion_velocity_damping: float = DEFAULT_OCCLUSION_VELOCITY_DAMPING,
        reid_recovery_min_score: float = DEFAULT_REID_RECOVERY_MIN_SCORE,
        reid_recovery_min_reid: float = DEFAULT_REID_RECOVERY_MIN_REID,
        reid_recovery_min_motion: float = DEFAULT_REID_RECOVERY_MIN_MOTION,
        reid_recovery_min_confidence: float = DEFAULT_REID_RECOVERY_MIN_CONFIDENCE,
        view_change_min_score: float = DEFAULT_VIEW_CHANGE_MIN_SCORE,
        view_change_min_motion: float = DEFAULT_VIEW_CHANGE_MIN_MOTION,
        view_change_min_confidence: float = DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE,
        view_change_max_lost_frames: int = DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES,
    ):
        self.lorat_root = lorat_root.resolve()
        self.config_name = config_name
        self.weight_path = weight_path.resolve()
        self.device_string = device
        self.max_tracks = max_tracks
        self.track_batch_size = max(1, track_batch_size)
        self.fps = fps
        self.sequence_length = sequence_length
        self.sequence_name = sequence_name
        self.disable_amp = disable_amp
        self.trajectory_history_size = 12
        self.lorat_memory_slots = max(1, min(MAX_LORAT_MEMORY_SLOTS, lorat_memory_slots))
        self.lorat_memory_refresh_interval = max(1, lorat_memory_refresh_interval)
        self.lorat_active_slots_per_track = max(0, lorat_active_slots_per_track)
        self.lorat_fixed_box_size = bool(lorat_fixed_box_size)
        self.lorat_min_box_area = max(0.0, float(lorat_min_box_area))
        self.lorat_max_area_change_per_frame = max(0.0, float(lorat_max_area_change_per_frame))
        self.lorat_trusted_size_floor_scale = max(0.0, min(1.0, lorat_trusted_size_floor_scale))
        self.lorat_memory_min_score = max(0.0, lorat_memory_min_score)
        self.lorat_accept_min_score = max(0.0, min(1.0, float(lorat_accept_min_score)))
        self.shrink_guard_window = max(0, int(shrink_guard_window))
        self.shrink_guard_area_ratio = max(0.0, min(1.0, float(shrink_guard_area_ratio)))
        self.shrink_guard_step_ratio = max(0.0, min(1.0, float(shrink_guard_step_ratio)))
        self.shrink_guard_min_confidence = max(0.0, min(1.0, float(shrink_guard_min_confidence)))
        self.shrink_guard_min_reid = max(0.0, min(1.0, float(shrink_guard_min_reid)))
        self.crop_information_min_score = max(0.0, float(crop_information_min_score))
        self.crop_information_min_pixels = max(1, int(crop_information_min_pixels))
        self.occlusion_max_frames = max(0, int(occlusion_max_frames))
        self.occlusion_iou_threshold = max(0.0, min(1.0, float(occlusion_iou_threshold)))
        self.occlusion_velocity_damping = max(0.0, min(1.0, float(occlusion_velocity_damping)))
        self.reid_recovery_min_score = max(0.0, min(1.0, float(reid_recovery_min_score)))
        self.reid_recovery_min_reid = max(0.0, min(1.0, float(reid_recovery_min_reid)))
        self.reid_recovery_min_motion = max(0.0, min(1.0, float(reid_recovery_min_motion)))
        self.reid_recovery_min_confidence = max(0.0, min(1.0, float(reid_recovery_min_confidence)))
        if lorat_search_area_factor is not None and lorat_search_area_factor <= 0:
            raise ValueError("--lorat-search-area-factor must be greater than 0.")
        if lorat_window_penalty is not None and lorat_window_penalty < 0:
            raise ValueError("--lorat-window-penalty must be greater than or equal to 0.")
        if lorat_state_update_min_score is not None and lorat_state_update_min_score < 0:
            raise ValueError("--lorat-state-update-min-score must be greater than or equal to 0.")
        if lorat_state_update_max_center_shift is not None and lorat_state_update_max_center_shift < 0:
            raise ValueError("--lorat-state-update-max-center-shift must be greater than or equal to 0.")
        if lorat_state_update_max_area_change is not None and lorat_state_update_max_area_change < 0:
            raise ValueError("--lorat-state-update-max-area-change must be greater than or equal to 0.")
        self.lorat_search_area_factor = lorat_search_area_factor
        self.lorat_window_penalty = lorat_window_penalty
        self.lorat_state_update_min_score = lorat_state_update_min_score
        self.lorat_state_update_max_center_shift = lorat_state_update_max_center_shift
        self.lorat_state_update_max_area_change = lorat_state_update_max_area_change
        self.lorat_runtime_overrides: List[str] = []
        self.identity_arbitrator = LightweightIdentityArbitrator(
            enabled=identity_arbitration,
            min_score=identity_min_score,
            min_reid=identity_min_reid,
            min_motion=identity_min_motion,
            min_path=identity_min_path,
            appearance_bank_size=identity_bank_size,
            memory_min_confidence=identity_memory_min_confidence,
            view_change_min_score=view_change_min_score,
            view_change_min_motion=view_change_min_motion,
            view_change_min_confidence=view_change_min_confidence,
            view_change_max_lost_frames=view_change_max_lost_frames,
        )
        auto_slot_capacity = max(
            self.track_batch_size,
            (
                self.max_tracks
                if self.max_tracks > 0
                else max(self.track_batch_size, expected_tracks)
            )
            * self.lorat_memory_slots,
        )
        self.lorat_slot_capacity = max(auto_slot_capacity, lorat_slot_capacity)
        self.tracks: List[TrackState] = []
        self.track_by_id: Dict[int, TrackState] = {}
        self.lorat_slots_by_task_id: Dict[int, LoRATMemorySlot] = {}
        self.lorat_slots_by_track_id: Dict[int, List[int]] = {}
        self.slot_debug_lines: List[str] = []
        self.next_track_id = 1
        self.next_lorat_task_id = 1
        self.closed = False
        self.using_directml = False
        self.device_label = self.device_string
        self.gpu_name = ""
        self.runtime_status = RuntimeStatus()
        self._fps_smoothing = 0.15
        self.backend_name = "LoRAT-v5-shared"

        self._load_lorat()
        self._build_runtime()

    def _load_lorat(self) -> None:
        if not self.lorat_root.exists():
            raise RuntimeError(f"LoRAT checkout not found: {self.lorat_root}")
        if not self.weight_path.exists():
            raise RuntimeError(f"LoRAT weight not found: {self.weight_path}")

        lorat_root_str = str(self.lorat_root)
        if lorat_root_str not in sys.path:
            sys.path.insert(0, lorat_root_str)

        import torch

        self.torch = torch
        requested_device = self.device_string.lower()
        if requested_device in {"dml", "directml"} or requested_device.startswith(("dml:", "directml:")):
            try:
                import torch_directml
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "LoRAT was asked to use DirectML, but torch-directml is not installed. "
                    "Install it with: .\\.venv\\Scripts\\python.exe -m pip install torch-directml"
                ) from exc
            device_index = 0
            if ":" in requested_device:
                device_index = int(requested_device.rsplit(":", 1)[1])
            self.device = torch_directml.device(device_index)
            self.using_directml = True
            try:
                device_name = torch_directml.device_name(device_index).rstrip("\x00")
            except Exception:
                device_name = str(self.device)
            self.device_label = f"DirectML {device_index} ({device_name}) [{self.device}]"
            if self.track_batch_size > 1:
                print(
                    "DirectML currently showed unstable batched LoRAT outputs here; "
                    "forcing --track-batch-size 1 for this backend."
                )
                self.track_batch_size = 1
        else:
            self.device = torch.device(self.device_string)
            self.device_label = str(self.device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LoRAT was asked to use cuda, but this PyTorch build reports no CUDA/HIP device. "
                "Use --device cpu on this laptop, or install a working CUDA/ROCm PyTorch build."
            )
        if self.device.type == "cuda":
            self.gpu_name = torch.cuda.get_device_name(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.using_directml:
            self.gpu_name = self.device_label
        self.runtime_status.gpu_name = self.gpu_name

    def _build_runtime(self) -> None:
        try:
            from trackit.core.boot.funcs.main.load_config import load_config
            from trackit.core.runtime.global_constant import get_global_constant
            from trackit.data.methods.siamese_tracker_eval.transform.builder import build_data_transform
            from trackit.models import ModelManager
            from trackit.models.compiling.plain.builder import build_plain_inference_engine
            from trackit.models.methods.builder import create_model_build_context
            from trackit.runners.evaluation.distributed.tracker_evaluator import EvaluatorContext
            from trackit.runners.evaluation.distributed.tracker_evaluator.default.evaluator import (
                DefaultTrackerEvaluator,
            )
            from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.builder import (
                build_tracker_evaluator_pipeline,
            )
        except ModuleNotFoundError as exc:
            package_name = exc.name or "unknown"
            raise RuntimeError(
                f"LoRAT dependency '{package_name}' is missing from this Python interpreter: "
                f"{sys.executable}. In VS Code, select the project interpreter at "
                f"{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'}, or run "
                "scripts/setup-lorat-env.ps1."
            ) from exc

        if get_global_constant("TIMM_USE_OLD_CACHE", default=True):
            os.environ["TIMM_USE_OLD_CACHE"] = "1"

        runtime_vars = SimpleNamespace(
            root_path=str(self.lorat_root),
            config_path=str(self.lorat_root / "config"),
            method_name="LoRAT",
            config_name=self.config_name,
            mixin_config=None,
        )
        config = load_config(runtime_vars)
        self._apply_lorat_runtime_overrides(config)
        self.config = config

        self.dtype = self.torch.float32
        transform_config = config["run"]["data"]["eval"]["transform"]
        self.transform = build_data_transform(transform_config, config, self.device, self.dtype)

        model_manager = ModelManager(create_model_build_context(config), rng_fixed_seed=42)
        model_manager.load_state_dict_from_file(str(self.weight_path), strict=False, print_missing=False)
        self.model_manager = model_manager

        inference_config = copy.deepcopy(config["run"]["runner"]["test"]["inference_engine"])
        if self.device.type != "cuda" or self.disable_amp:
            inference_config["auto_mixed_precision"]["enabled"] = False
        inference_config["torch_compile"]["enabled"] = False
        inference_engine = build_plain_inference_engine(inference_config, self.device)

        self.optimized_model = inference_engine(
            model_manager,
            self.device,
            self.dtype,
            self.track_batch_size,
            1,
        )
        self._install_lorat_forward_profile()

        pipeline_config = config["run"]["runner"]["test"]["evaluator"]["pipeline"]
        pipeline = build_tracker_evaluator_pipeline(
            pipeline_config,
            config,
            self.device,
            config["run"]["num_epochs"],
        )
        self.evaluator = DefaultTrackerEvaluator(pipeline)
        self.evaluator_context = EvaluatorContext(
            epoch=0,
            max_batch_size=self.lorat_slot_capacity,
            num_input_data_streams=1,
            dtype=self.dtype,
            auto_mixed_precision_dtype=self.optimized_model.auto_mixed_precision_dtype,
            model=self.optimized_model.raw_model,
        )
        self.evaluator.start(self.evaluator_context)
        print(
            f"Loaded LoRAT {self.config_name} on {self.device_label} with weight {self.weight_path.name}. "
            f"Track batch size: {self.track_batch_size}; LoRAT slot capacity: {self.lorat_slot_capacity}; "
            f"execution mode: {V5_EXECUTION_MODE}; "
            f"active slots/track: {'all' if self.lorat_active_slots_per_track == 0 else self.lorat_active_slots_per_track}; "
            f"fixed box size: {self.lorat_fixed_box_size}; "
            f"min box area: {self.lorat_min_box_area:.1f}; "
            f"max area change/frame: {self.lorat_max_area_change_per_frame:.2f}; "
            f"size floor scale: {self.lorat_trusted_size_floor_scale:.2f}; "
            f"shrink guard: window={self.shrink_guard_window}, area_ratio={self.shrink_guard_area_ratio:.2f}, "
            f"step_ratio={self.shrink_guard_step_ratio:.2f}; "
            f"crop info min: {self.crop_information_min_score:.2f}/{self.crop_information_min_pixels}px; "
            f"accept score: {self.lorat_accept_min_score:.2f}; "
            f"occlusion hold: {self.occlusion_max_frames} frames; "
            f"occlusion velocity damping: {self.occlusion_velocity_damping:.2f}"
        )
        if self.lorat_runtime_overrides:
            print("LoRAT runtime overrides: " + "; ".join(self.lorat_runtime_overrides))

    @staticmethod
    def _tensor_batch_size(value: object) -> Optional[int]:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None

    def _forward_batch_size(self, args: Sequence[object], kwargs: Dict[str, object]) -> int:
        for key in ("z", "x", "z_feat", "x_feat"):
            if key in kwargs:
                batch_size = self._tensor_batch_size(kwargs[key])
                if batch_size is not None:
                    return batch_size
        for value in args:
            batch_size = self._tensor_batch_size(value)
            if batch_size is not None:
                return batch_size
        return 0

    def _install_lorat_forward_profile(self) -> None:
        raw_model = getattr(self.optimized_model, "raw_model", None)
        if raw_model is None or getattr(raw_model, "_v5_forward_profile_installed", False):
            return

        original_forward = raw_model.forward

        def profiled_forward(*args, **kwargs):
            batch_size = self._forward_batch_size(args, kwargs)
            self.runtime_status.model_forward_calls += 1
            self.runtime_status.model_forward_items += batch_size
            self.runtime_status.max_model_forward_batch = max(
                self.runtime_status.max_model_forward_batch,
                batch_size,
            )
            return original_forward(*args, **kwargs)

        raw_model.forward = profiled_forward

        if hasattr(raw_model, "_fusion"):
            original_fusion = raw_model._fusion

            def profiled_fusion(*args, **kwargs):
                batch_size = self._forward_batch_size(args, kwargs)
                self.runtime_status.fusion_forward_calls += 1
                self.runtime_status.fusion_forward_items += batch_size
                self.runtime_status.max_fusion_forward_batch = max(
                    self.runtime_status.max_fusion_forward_batch,
                    batch_size,
                )
                return original_fusion(*args, **kwargs)

            raw_model._fusion = profiled_fusion

        raw_model._v5_forward_profile_installed = True

    def _apply_lorat_runtime_overrides(self, config: dict) -> None:
        pipeline_config = config["run"]["runner"]["test"]["evaluator"]["pipeline"]
        if self.lorat_search_area_factor is not None:
            search_config = pipeline_config["search_region_cropping"]
            previous = search_config.get("area_factor")
            search_config["area_factor"] = float(self.lorat_search_area_factor)
            self.lorat_runtime_overrides.append(
                f"search_region_cropping.area_factor {previous} -> {self.lorat_search_area_factor}"
            )

        if self.lorat_window_penalty is not None:
            post_process_config = pipeline_config["post_process"]
            previous = post_process_config.get("window_penalty")
            post_process_config["window_penalty"] = float(self.lorat_window_penalty)
            self.lorat_runtime_overrides.append(
                f"post_process.window_penalty {previous} -> {self.lorat_window_penalty}"
            )

        state_update_config = pipeline_config.setdefault("state_update", {})
        if self.lorat_state_update_min_score is not None:
            previous = state_update_config.get("min_score")
            state_update_config["min_score"] = float(self.lorat_state_update_min_score)
            self.lorat_runtime_overrides.append(
                f"state_update.min_score {previous} -> {self.lorat_state_update_min_score}"
            )
        if self.lorat_state_update_max_center_shift is not None:
            previous = state_update_config.get("max_center_shift_factor")
            state_update_config["max_center_shift_factor"] = float(self.lorat_state_update_max_center_shift)
            self.lorat_runtime_overrides.append(
                "state_update.max_center_shift_factor "
                f"{previous} -> {self.lorat_state_update_max_center_shift}"
            )
        if self.lorat_state_update_max_area_change is not None:
            previous = state_update_config.get("max_area_change_factor")
            state_update_config["max_area_change_factor"] = float(self.lorat_state_update_max_area_change)
            self.lorat_runtime_overrides.append(
                f"state_update.max_area_change_factor {previous} -> {self.lorat_state_update_max_area_change}"
            )
        previous_fixed_size = pipeline_config.get("fixed_box_size")
        pipeline_config["fixed_box_size"] = bool(self.lorat_fixed_box_size)
        self.lorat_runtime_overrides.append(
            f"fixed_box_size {previous_fixed_size} -> {self.lorat_fixed_box_size}"
        )
        scale_limit_config = pipeline_config.setdefault("scale_limit", {})
        previous_min_area = scale_limit_config.get("min_box_area")
        scale_limit_config["min_box_area"] = float(self.lorat_min_box_area)
        self.lorat_runtime_overrides.append(
            f"scale_limit.min_box_area {previous_min_area} -> {self.lorat_min_box_area}"
        )
        previous_max_area_change = scale_limit_config.get("max_area_change_factor")
        scale_limit_config["max_area_change_factor"] = float(self.lorat_max_area_change_per_frame)
        self.lorat_runtime_overrides.append(
            "scale_limit.max_area_change_factor "
            f"{previous_max_area_change} -> {self.lorat_max_area_change_per_frame}"
        )
        previous_min_initial_size = scale_limit_config.get("min_initial_size_factor")
        scale_limit_config["min_initial_size_factor"] = float(self.lorat_trusted_size_floor_scale)
        self.lorat_runtime_overrides.append(
            "scale_limit.min_initial_size_factor "
            f"{previous_min_initial_size} -> {self.lorat_trusted_size_floor_scale}"
        )

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int = 1) -> None:
        self.add_tracks(frame, boxes, frame_number)

    def _allocate_lorat_task_id(self) -> int:
        while self.next_lorat_task_id in self.lorat_slots_by_task_id:
            self.next_lorat_task_id += 1
        task_id = self.next_lorat_task_id
        self.next_lorat_task_id += 1
        return task_id

    def _make_slot_task(
        self,
        track: TrackState,
        slot: LoRATMemorySlot,
        frame_rgb: np.ndarray,
        bbox: BBox,
        frame_number: int,
        create_task: bool,
        track_after_init: bool = False,
    ):
        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )
        from trackit.data.protocol import SequenceInfo

        init_context = SiameseTrackerEvalDataWorker_FrameContext(
            frame_number,
            make_frame_getter(frame_rgb),
            xywh_to_xyxy_np(bbox),
            None,
        )
        sequence_info = None
        if create_task:
            sequence_info = SequenceInfo(
                dataset_name="user",
                data_split=None,
                dataset_full_name=None,
                sequence_name=f"{self.sequence_name}-track-{track.track_id}-{slot.label}-{slot.task_id}",
                length=self.sequence_length,
                fps=self.fps,
            )
        return SiameseTrackerEvalDataWorker_Task(
            task_index=slot.task_id,
            do_task_creation=sequence_info,
            do_tracker_init=init_context,
            do_tracker_track=init_context if track_after_init else None,
            do_task_finalization=False,
        )

    def _create_lorat_slot(
        self,
        track: TrackState,
        label: str,
        frame: np.ndarray,
        frame_rgb: np.ndarray,
        bbox: BBox,
        frame_number: int,
    ) -> Optional[object]:
        clipped = clip_bbox_to_frame(frame, bbox)
        if clipped is None:
            return None
        if len(self.lorat_slots_by_task_id) >= self.lorat_slot_capacity:
            if label != "initial":
                return None
            raise RuntimeError(
                "LoRAT slot capacity exceeded while creating an initial slot. "
                "Increase --lorat-slot-capacity, set --max-tracks, or reduce --lorat-memory-slots."
            )

        slot = LoRATMemorySlot(
            task_id=self._allocate_lorat_task_id(),
            track_id=track.track_id,
            label=label,
            frame_number=frame_number,
            bbox=tuple(float(value) for value in clipped),
            confidence=track.confidence,
            raw_confidence=track.raw_confidence,
            confidence_baseline=track.confidence_baseline,
            last_refresh_frame=frame_number,
            anchor_frame_number=frame_number,
            anchor_bbox=tuple(float(value) for value in clipped),
        )
        self.lorat_slots_by_task_id[slot.task_id] = slot
        self.lorat_slots_by_track_id.setdefault(track.track_id, []).append(slot.task_id)
        track.active_template_frame = frame_number
        self._sync_track_slot_count(track)
        return self._make_slot_task(track, slot, frame_rgb, slot.bbox, frame_number, create_task=True)

    def _refresh_lorat_slot(
        self,
        track: TrackState,
        slot: LoRATMemorySlot,
        frame: np.ndarray,
        frame_rgb: np.ndarray,
        bbox: BBox,
        frame_number: int,
    ) -> Optional[object]:
        clipped = clip_bbox_to_frame(frame, bbox)
        if clipped is None:
            return None

        slot.frame_number = frame_number
        slot.bbox = tuple(float(value) for value in clipped)
        slot.confidence = track.confidence
        slot.raw_confidence = track.raw_confidence
        slot.confidence_baseline = track.confidence_baseline
        slot.last_refresh_frame = frame_number
        slot.anchor_frame_number = frame_number
        slot.anchor_bbox = tuple(float(value) for value in clipped)
        slot.active = True
        track.active_template_frame = frame_number
        self._sync_track_slot_count(track)
        return self._make_slot_task(track, slot, frame_rgb, slot.bbox, frame_number, create_task=False)

    def _get_track_slots(self, track: TrackState) -> List[LoRATMemorySlot]:
        return [
            slot
            for task_id in self.lorat_slots_by_track_id.get(track.track_id, [])
            if (slot := self.lorat_slots_by_task_id.get(task_id)) is not None and slot.active
        ]

    def _get_track_slot(self, track: TrackState, label: str) -> Optional[LoRATMemorySlot]:
        for slot in self._get_track_slots(track):
            if slot.label == label:
                return slot
        return None

    def _select_lorat_tracking_slots(self, track: TrackState, frame_number: int) -> List[LoRATMemorySlot]:
        slots = self._get_track_slots(track)
        active_limit = self.lorat_active_slots_per_track
        if active_limit <= 0 or len(slots) <= active_limit:
            return slots

        by_label = {slot.label: slot for slot in slots}
        selected: List[LoRATMemorySlot] = []

        def add(slot: Optional[LoRATMemorySlot]) -> None:
            if slot is None or len(selected) >= active_limit:
                return
            if any(existing.task_id == slot.task_id for existing in selected):
                return
            selected.append(slot)

        add(by_label.get("initial"))

        recent_slots = sorted(
            (slot for slot in slots if slot.label != "initial"),
            key=lambda slot: slot.label,
        )
        if recent_slots:
            primary_recent_slot = by_label.get(track.active_lorat_slot) if track.active_lorat_slot != "initial" else None
            add(primary_recent_slot or max(recent_slots, key=lambda slot: (slot.last_refresh_frame, slot.task_id)))
            start_frame = track.trajectory[0][0] if track.trajectory else frame_number
            rotating_index = max(0, frame_number - start_frame - 1) % len(recent_slots)
            for offset in range(len(recent_slots)):
                add(recent_slots[(rotating_index + offset) % len(recent_slots)])
                if len(selected) >= active_limit:
                    break
            for slot in sorted(recent_slots, key=lambda item: (item.last_refresh_frame, item.task_id), reverse=True):
                add(slot)
                if len(selected) >= active_limit:
                    break

        return selected

    def _sync_track_slot_count(self, track: TrackState) -> None:
        track.lorat_memory_slot_count = len(self._get_track_slots(track))

    def _recent_lorat_memory_label(self, track: TrackState, frame_number: int) -> Optional[str]:
        recent_slot_count = self.lorat_memory_slots - 1
        if recent_slot_count <= 0:
            return None
        start_frame = track.trajectory[0][0] if track.trajectory else frame_number
        refresh_index = max(0, (frame_number - start_frame - 1) // self.lorat_memory_refresh_interval)
        slot_index = (refresh_index % recent_slot_count) + 1
        return f"recent-{slot_index:02d}"

    def _should_refresh_lorat_memory_slot(self, track: TrackState, frame_number: int) -> bool:
        if self.lorat_memory_slots <= 1:
            return False
        if track.learning_block_reason:
            return False
        if track.confidence is not None and track.confidence < self.lorat_memory_min_score:
            return False
        if track.assignment_score is not None and track.assignment_score < self.identity_arbitrator.min_score:
            if "VIEWCHANGE" not in track.state:
                return False
        if track.reid_score is not None and track.reid_score < self.identity_arbitrator.min_reid:
            if "VIEWCHANGE" not in track.state:
                return False
            if track.motion_score is None or track.motion_score < self.identity_arbitrator.view_change_min_motion:
                return False
        if track.path_score is not None and track.path_score < self.identity_arbitrator.min_path:
            return False
        if track.lost_frames > 0:
            return False
        if track.occluded_frames > 0:
            return False
        return bool(track.trajectory) and frame_number > track.trajectory[0][0]

    def _commit_trusted_size(self, track: TrackState, bbox: BBox) -> None:
        track.trusted_size_bank.append(clamp_bbox_size(bbox))
        if len(track.trusted_size_bank) > DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:
            del track.trusted_size_bank[: len(track.trusted_size_bank) - DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE]

    def _record_size_history(self, track: TrackState, frame_number: int, bbox: BBox) -> None:
        track.size_history.append((frame_number, clamp_bbox_size(bbox)))
        max_history = max(DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE, self.shrink_guard_window + 2)
        if len(track.size_history) > max_history:
            del track.size_history[: len(track.size_history) - max_history]

    def _learning_evidence_is_strong(
        self,
        track: TrackState,
        confidence: Optional[float],
        identity_assignment: Optional[IdentityAssignment],
    ) -> bool:
        confidence_value = 0.5 if confidence is None else max(0.0, min(1.0, float(confidence)))
        if confidence_value < self.shrink_guard_min_confidence:
            return False
        if identity_assignment is None or not track_has_appearance(track):
            return True
        return identity_assignment.score.appearance >= self.shrink_guard_min_reid

    def _assess_learning_hold(
        self,
        track: TrackState,
        bbox: BBox,
        confidence: Optional[float],
        identity_assignment: Optional[IdentityAssignment],
        frame: Optional[np.ndarray],
    ) -> Tuple[bool, List[str], CropInformation, float, float, int]:
        crop_info = measure_crop_information(frame, bbox, self.crop_information_min_pixels)
        previous_area = bbox_area(track.bbox)
        current_area = bbox_area(bbox)
        step_ratio = current_area / max(1.0, previous_area)

        recent_history = track.size_history[-self.shrink_guard_window :] if self.shrink_guard_window > 0 else []
        reference_area = max([previous_area] + [bbox_area(sample_bbox) for _, sample_bbox in recent_history])
        window_ratio = current_area / max(1.0, reference_area)

        projected_shrink_frames = track.shrink_risk_frames + 1 if step_ratio < 0.995 else 0
        shrink_reasons: List[str] = []
        if self.shrink_guard_step_ratio > 0 and step_ratio < self.shrink_guard_step_ratio:
            shrink_reasons.append("SHRINKSTEP")
        if self.shrink_guard_area_ratio > 0 and window_ratio < self.shrink_guard_area_ratio:
            shrink_reasons.append("SHRINKWINDOW")
        if projected_shrink_frames >= 3 and window_ratio < 0.90:
            shrink_reasons.append("SHRINKRATCHET")

        reasons: List[str] = []
        if crop_info.score < self.crop_information_min_score:
            reasons.append("LOWINFO")
        if shrink_reasons and not self._learning_evidence_is_strong(track, confidence, identity_assignment):
            reasons.append("SHRINKRISK")

        return bool(reasons), reasons, crop_info, step_ratio, window_ratio, projected_shrink_frames

    def _trusted_size_floor(self, track: TrackState) -> Optional[Tuple[float, float]]:
        if self.lorat_trusted_size_floor_scale <= 0:
            return None

        initial_floor: Optional[Tuple[float, float]] = None
        if track.initial_bbox is not None:
            _, _, initial_w, initial_h = clamp_bbox_size(track.initial_bbox)
            initial_floor = (
                max(1.0, initial_w * self.lorat_trusted_size_floor_scale),
                max(1.0, initial_h * self.lorat_trusted_size_floor_scale),
            )

        samples: List[BBox] = []
        samples.extend(track.trusted_size_bank[-DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:])
        if not samples:
            return initial_floor

        widths = np.asarray([max(1.0, float(sample[2])) for sample in samples], dtype=np.float32)
        heights = np.asarray([max(1.0, float(sample[3])) for sample in samples], dtype=np.float32)
        reference_width = float(np.median(widths))
        reference_height = float(np.median(heights))
        memory_floor = (
            max(1.0, reference_width * self.lorat_trusted_size_floor_scale),
            max(1.0, reference_height * self.lorat_trusted_size_floor_scale),
        )
        if initial_floor is None:
            return memory_floor
        return (
            max(initial_floor[0], memory_floor[0]),
            max(initial_floor[1], memory_floor[1]),
        )

    def _apply_trusted_size_floor(
        self,
        track: TrackState,
        bbox: BBox,
        frame: Optional[np.ndarray],
    ) -> Tuple[BBox, bool]:
        x, y, w, h = clamp_bbox_size(bbox)
        floor = self._trusted_size_floor(track)
        if floor is None:
            return clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False

        min_w, min_h = floor
        if w >= min_w and h >= min_h:
            return clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False

        center_x, center_y = bbox_center((x, y, w, h))
        guarded_w = max(w, min_w)
        guarded_h = max(h, min_h)
        guarded = clamp_bbox_to_frame_bounds(
            frame,
            (
                center_x - (guarded_w / 2.0),
                center_y - (guarded_h / 2.0),
                guarded_w,
                guarded_h,
            ),
        )
        return guarded, True

    def _apply_fixed_box_size(
        self,
        track: TrackState,
        bbox: BBox,
        frame: Optional[np.ndarray],
    ) -> Tuple[BBox, bool]:
        if not self.lorat_fixed_box_size or track.initial_bbox is None:
            return clamp_bbox_to_frame_bounds(frame, bbox), False

        _, _, fixed_w, fixed_h = clamp_bbox_size(track.initial_bbox)
        x, y, w, h = clamp_bbox_size(bbox)
        center_x, center_y = bbox_center((x, y, w, h))
        fixed = clamp_bbox_to_frame_bounds(
            frame,
            (
                center_x - (fixed_w / 2.0),
                center_y - (fixed_h / 2.0),
                fixed_w,
                fixed_h,
            ),
        )
        changed = abs(fixed[2] - w) > 0.01 or abs(fixed[3] - h) > 0.01
        return fixed, changed

    def _scale_bbox_to_area(self, bbox: BBox, target_area: float, frame: Optional[np.ndarray]) -> BBox:
        x, y, w, h = clamp_bbox_size(bbox)
        current_area = max(1.0, w * h)
        target_area = max(1.0, float(target_area))
        scale = float(np.sqrt(target_area / current_area))
        center_x, center_y = bbox_center((x, y, w, h))
        scaled_w = max(1.0, w * scale)
        scaled_h = max(1.0, h * scale)
        return clamp_bbox_to_frame_bounds(
            frame,
            (
                center_x - (scaled_w / 2.0),
                center_y - (scaled_h / 2.0),
                scaled_w,
                scaled_h,
            ),
        )

    def _apply_scale_limits(
        self,
        track: TrackState,
        bbox: BBox,
        frame: Optional[np.ndarray],
    ) -> Tuple[BBox, List[str]]:
        limited = clamp_bbox_to_frame_bounds(frame, bbox)
        tokens: List[str] = []

        if self.lorat_min_box_area > 0 and bbox_area(limited) < self.lorat_min_box_area:
            limited = self._scale_bbox_to_area(limited, self.lorat_min_box_area, frame)
            tokens.append("MINAREA")

        if self.lorat_max_area_change_per_frame > 1.0 and track.bbox is not None:
            previous_area = bbox_area(track.bbox)
            current_area = bbox_area(limited)
            min_area = max(self.lorat_min_box_area, previous_area / self.lorat_max_area_change_per_frame)
            max_area = previous_area * self.lorat_max_area_change_per_frame
            max_area = max(max_area, min_area)
            target_area = min(max(current_area, min_area), max_area)
            if abs(target_area - current_area) > 0.5:
                limited = self._scale_bbox_to_area(limited, target_area, frame)
                tokens.append("SCALELIMIT")

        limited, size_floor_applied = self._apply_trusted_size_floor(track, limited, frame)
        if size_floor_applied:
            tokens.append("SIZEFLOOR")
        return limited, tokens

    def _predict_active_tracks(self, tracks: Sequence[TrackState], frame: Optional[np.ndarray]) -> None:
        for track in tracks:
            if track.kalman is None:
                track.kalman = BBoxKalmanFilter(track.bbox)
            predicted = track.kalman.predict()
            track.predicted_bbox = clamp_bbox_to_frame_bounds(frame, predicted)

    def _candidate_is_occluded(self, track: TrackState, bbox: BBox) -> bool:
        return self._candidate_occlusion_info(track, bbox)[0] is not None

    def _candidate_occlusion_info(self, track: TrackState, bbox: BBox) -> Tuple[Optional[int], float]:
        if self.occlusion_iou_threshold <= 0:
            return None, 0.0
        other_track_id, overlap = strongest_track_overlap(track, bbox, self.tracks)
        if other_track_id is None or overlap < self.occlusion_iou_threshold:
            return None, overlap
        return other_track_id, overlap

    def _calibrate_lorat_confidence(
        self,
        track: TrackState,
        slot: LoRATMemorySlot,
        raw_confidence: Optional[float],
    ) -> Optional[float]:
        if raw_confidence is None:
            return None
        raw_value = max(0.0, min(1.0, float(raw_confidence)))
        baseline = slot.confidence_baseline or track.confidence_baseline
        if baseline is None or baseline <= 0:
            baseline = max(raw_value, 0.05)
            self._update_confidence_baseline(track, slot, raw_value, force=True)
        return max(0.0, min(1.0, raw_value / max(0.05, float(baseline))))

    def _update_confidence_baseline(
        self,
        track: TrackState,
        slot: LoRATMemorySlot,
        raw_confidence: Optional[float],
        force: bool = False,
    ) -> None:
        if raw_confidence is None:
            return
        raw_value = max(0.0, min(1.0, float(raw_confidence)))
        if force or track.confidence_baseline is None:
            track.confidence_baseline = max(raw_value, 0.05)
        elif raw_value > track.confidence_baseline:
            track.confidence_baseline = raw_value

        if force or slot.confidence_baseline is None:
            slot.confidence_baseline = max(raw_value, 0.05)
        elif raw_value > slot.confidence_baseline:
            slot.confidence_baseline = raw_value

    def _candidate_reject_state(
        self,
        track: TrackState,
        bbox: BBox,
        confidence: Optional[float],
        identity_assignment: Optional[IdentityAssignment],
    ) -> Optional[str]:
        confidence_value = 0.5 if confidence is None else max(0.0, min(1.0, float(confidence)))
        if confidence_value < self.lorat_accept_min_score:
            if self._is_reid_recovery(track, confidence_value, identity_assignment):
                return None
            return "LOWCONF"

        if identity_assignment is None:
            return None

        score = identity_assignment.score
        is_view_change = self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            score,
        )
        if track.lost_frames > 0 and not self._has_recovery_initial_anchor(identity_assignment):
            return "ANCHORLOW"
        if self._is_initial_anchor_steal(score):
            return "OTHERID"
        if (
            path_gate_ready(track)
            and score.path < self.identity_arbitrator.min_path
            and confidence_value < 0.85
            and not is_view_change
            and not self._is_path_recovery(track, confidence_value, identity_assignment)
        ):
            return "PATHLOW"
        if score.total < self.identity_arbitrator.min_score:
            if not is_view_change:
                return "ID_UNCERTAIN"
        if track_has_appearance(track) and score.appearance < self.identity_arbitrator.min_reid:
            if not is_view_change:
                return "REIDLOW"
        if score.motion < self.identity_arbitrator.min_motion and confidence_value < 0.70:
            return "MOTIONLOW"
        if track.lost_frames > 0:
            reacquire_confidence = min(0.95, max(self.lorat_accept_min_score + 0.10, 0.40))
            if (
                confidence_value < reacquire_confidence
                and score.appearance < max(0.55, self.identity_arbitrator.min_reid)
            ):
                return "REACQUIRE_LOWCONF"
        return None

    def _is_initial_anchor_steal(self, score: IdentityScore) -> bool:
        return (
            score.other_track_id is not None
            and score.occlusion_iou >= DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and score.other_anchor >= DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and score.identity_margin <= -DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )

    def _is_path_recovery(
        self,
        track: TrackState,
        confidence_value: float,
        identity_assignment: Optional[IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames < DEFAULT_PATH_RECOVERY_AFTER_FRAMES:
            return False
        score = identity_assignment.score
        return (
            identity_assignment.output.source_track_id == track.track_id
            and confidence_value >= DEFAULT_PATH_RECOVERY_MIN_CONFIDENCE
            and score.appearance >= DEFAULT_PATH_RECOVERY_MIN_REID
            and self._has_recovery_initial_anchor(identity_assignment)
            and score.motion >= DEFAULT_PATH_RECOVERY_MIN_MOTION
            and score.total >= self.identity_arbitrator.min_score
        )

    def _has_recovery_initial_anchor(self, identity_assignment: Optional[IdentityAssignment]) -> bool:
        if DEFAULT_RECOVERY_INITIAL_ANCHOR_MIN <= 0:
            return True
        if identity_assignment is None:
            return False
        return identity_assignment.score.initial_anchor >= DEFAULT_RECOVERY_INITIAL_ANCHOR_MIN

    def _is_reid_recovery(
        self,
        track: TrackState,
        confidence_value: float,
        identity_assignment: Optional[IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames <= 0:
            return False
        score = identity_assignment.score
        return (
            confidence_value >= self.reid_recovery_min_confidence
            and score.total >= self.reid_recovery_min_score
            and score.appearance >= self.reid_recovery_min_reid
            and self._has_recovery_initial_anchor(identity_assignment)
            and score.motion >= self.reid_recovery_min_motion
            and (
                score.path >= self.identity_arbitrator.min_path
                or self._is_path_recovery(track, confidence_value, identity_assignment)
            )
        )

    def _build_lorat_memory_refresh_tasks(
        self,
        tracks: Sequence[TrackState],
        frame: np.ndarray,
        frame_number: int,
    ) -> List[object]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tasks: List[object] = []
        for track in tracks:
            if not self._should_refresh_lorat_memory_slot(track, frame_number):
                continue

            label = self._recent_lorat_memory_label(track, frame_number)
            if label is None:
                continue

            slot = self._get_track_slot(track, label)
            if slot is None:
                task = self._create_lorat_slot(track, label, frame, frame_rgb, track.bbox, frame_number)
            elif frame_number - slot.last_refresh_frame >= self.lorat_memory_refresh_interval:
                task = self._refresh_lorat_slot(track, slot, frame, frame_rgb, track.bbox, frame_number)
            else:
                task = None
            if task is not None:
                tasks.append(task)

        return tasks

    def add_tracks(
        self,
        frame: np.ndarray,
        boxes: Sequence[BBox],
        frame_number: int = 1,
    ) -> List[TrackState]:
        if self.max_tracks > 0 and len(self.tracks) + len(boxes) > self.max_tracks:
            raise RuntimeError(
                f"Too many tracks selected. Current={len(self.tracks)}, requested={len(boxes)}, "
                f"max={self.max_tracks}. Increase --max-tracks or set it to 0 for no cap."
            )

        added_tracks = []
        tasks = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for bbox in boxes:
            clipped = clip_bbox_to_frame(frame, bbox)
            if clipped is None:
                continue
            if len(self.lorat_slots_by_task_id) >= self.lorat_slot_capacity:
                raise RuntimeError(
                    "LoRAT slot capacity exceeded while adding tracks. "
                    "Increase --lorat-slot-capacity, set --max-tracks, or reduce --lorat-memory-slots."
                )

            track_id = self.next_track_id
            self.next_track_id += 1
            track = TrackState(
                track_id=track_id,
                bbox=tuple(float(value) for value in clipped),
                color=color_for_track(track_id),
                confidence=1.0,
                raw_confidence=1.0,
                previous_bbox=tuple(float(value) for value in clipped),
                predicted_bbox=tuple(float(value) for value in clipped),
                raw_bbox=tuple(float(value) for value in clipped),
                kalman=BBoxKalmanFilter(clipped),
                last_reliable_bbox=tuple(float(value) for value in clipped),
                last_reliable_frame=frame_number,
            )
            track.initial_bbox = track.bbox
            self._commit_trusted_size(track, track.bbox)
            self._record_size_history(track, frame_number, track.bbox)
            self.tracks.append(track)
            self.track_by_id[track_id] = track
            added_tracks.append(track)
            record_track_trajectory(track, frame_number, track.bbox, self.trajectory_history_size)
            record_reliable_track_trajectory(track, frame_number, track.bbox, self.trajectory_history_size)
            track.active_template_frame = frame_number
            track.active_lorat_slot = "initial"
            self.identity_arbitrator.initialize_track(track, frame)

            initial_task = self._create_lorat_slot(
                track,
                "initial",
                frame,
                rgb_frame,
                track.bbox,
                frame_number,
            )
            if initial_task is not None:
                tasks.append(initial_task)

        if tasks:
            outputs = self._run_worker_tasks(tasks)
            self._apply_evaluated_frames(outputs, frame, frame_number)
        return added_tracks

    def update(self, frame: np.ndarray, frame_number: int) -> Sequence[TrackState]:
        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )

        update_started = time.perf_counter()
        active_tracks = [track for track in self.tracks if self._get_track_slots(track)]
        if not active_tracks:
            self._record_frame_status(time.perf_counter() - update_started)
            return self.tracks
        self._predict_active_tracks(active_tracks, frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        get_rgb_frame = make_frame_getter(rgb_frame)
        tasks = []
        for track in active_tracks:
            for slot in self._select_lorat_tracking_slots(track, frame_number):
                track_context = SiameseTrackerEvalDataWorker_FrameContext(
                    frame_number,
                    get_rgb_frame,
                    None,
                    None,
                )
                tasks.append(
                    SiameseTrackerEvalDataWorker_Task(
                        task_index=slot.task_id,
                        do_task_creation=None,
                        do_tracker_init=None,
                        do_tracker_track=track_context,
                        do_task_finalization=False,
                    )
                )

        if not tasks:
            return self.tracks

        outputs = self._run_worker_tasks(tasks)
        self._apply_evaluated_frames(outputs, frame, frame_number)
        self._record_frame_status(time.perf_counter() - update_started)
        return self.tracks

    def _apply_evaluated_frames(
        self,
        outputs: Optional[dict],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> None:
        evaluated_frames = outputs.get("evaluated_frames", []) if outputs is not None else []
        slot_outputs: Dict[int, List[Tuple[LoRATMemorySlot, BBox, Optional[float]]]] = {}
        evaluated_track_ids: set[int] = set()

        for result in evaluated_frames:
            slot = self.lorat_slots_by_task_id.get(result.id)
            if slot is None:
                continue
            track = self.track_by_id.get(slot.track_id)
            if track is None:
                continue
            evaluated_track_ids.add(track.track_id)
            if result.output_box is None:
                continue

            confidence = float(result.output_confidence) if result.output_confidence is not None else None
            bbox = xyxy_to_xywh_tuple(result.output_box)
            slot.raw_confidence = confidence
            slot_outputs.setdefault(track.track_id, []).append((slot, bbox, confidence))

        self._apply_lorat_owned_slot_outputs(slot_outputs, evaluated_track_ids, frame, frame_number)

    def _apply_lorat_owned_slot_outputs(
        self,
        slot_outputs: Dict[int, List[Tuple[LoRATMemorySlot, BBox, Optional[float]]]],
        evaluated_track_ids: set[int],
        frame: Optional[np.ndarray],
        frame_number: int,
    ) -> None:
        updated_tracks: List[TrackState] = []
        evaluated_tracks = [
            track
            for track_id in sorted(evaluated_track_ids)
            if (track := self.track_by_id.get(track_id)) is not None
        ]
        candidate_outputs = [
            LoRATSlotOutput(
                source_track_id=track_id,
                slot=slot,
                bbox=bbox,
                confidence=confidence,
            )
            for track_id, outputs_for_track in slot_outputs.items()
            for slot, bbox, confidence in outputs_for_track
        ]
        prepared_candidate_outputs = [
            self.identity_arbitrator._with_appearance(output, frame)
            for output in candidate_outputs
        ]
        self._append_slot_debug_rows(frame_number, evaluated_tracks, prepared_candidate_outputs)
        assignments = self.identity_arbitrator.resolve(evaluated_tracks, prepared_candidate_outputs, frame)
        assigned_track_ids = set()
        assigned_output_keys = {
            (assignment.output.source_track_id, assignment.output.slot.task_id)
            for assignment in assignments
        }

        for assignment in assignments:
            track = assignment.track
            assigned_track_ids.add(track.track_id)
            output = assignment.output
            if self._accept_lorat_output(
                track,
                output.slot,
                output.bbox,
                output.confidence,
                frame,
                frame_number,
                assignment,
            ):
                updated_tracks.append(track)
            else:
                self._mark_lorat_miss(
                    track,
                    frame_number,
                    state=track.state or "LORAT_REJECT",
                    source="lorat-reject",
                    raw_bbox=track.raw_bbox,
                    confidence=track.confidence,
                    raw_confidence=track.raw_confidence,
                    preserve_scores=True,
                )

        for track in evaluated_tracks:
            if track.track_id in assigned_track_ids:
                continue
            if slot_outputs.get(track.track_id):
                fallback_assignment = self._best_owned_fallback_assignment(
                    track,
                    slot_outputs[track.track_id],
                    assigned_output_keys,
                    evaluated_tracks,
                    frame,
                )
                if fallback_assignment is not None:
                    output = fallback_assignment.output
                    if self._accept_lorat_output(
                        track,
                        output.slot,
                        output.bbox,
                        output.confidence,
                        frame,
                        frame_number,
                        fallback_assignment,
                    ):
                        updated_tracks.append(track)
                    else:
                        self._mark_lorat_miss(
                            track,
                            frame_number,
                            state=track.state or "ID_UNCERTAIN",
                            source="identity-gate",
                            raw_bbox=track.raw_bbox,
                            confidence=track.confidence,
                            raw_confidence=track.raw_confidence,
                            preserve_scores=True,
                        )
                else:
                    self._mark_lorat_miss(track, frame_number, state="ID_UNCERTAIN", source="identity-gate")
            else:
                self._mark_lorat_miss(track, frame_number)

        if frame is not None and updated_tracks:
            refresh_tasks = self._build_lorat_memory_refresh_tasks(updated_tracks, frame, frame_number)
            if refresh_tasks:
                self._run_worker_tasks(refresh_tasks)

    def _append_slot_debug_rows(
        self,
        frame_number: int,
        evaluated_tracks: Sequence[TrackState],
        candidate_outputs: Sequence[LoRATSlotOutput],
    ) -> None:
        for output in candidate_outputs:
            track = self.track_by_id.get(output.source_track_id)
            if track is None:
                continue
            score = self.identity_arbitrator.score(track, output, evaluated_tracks)
            calibrated_confidence = preview_lorat_confidence(track, output.slot, output.confidence)
            fields = [
                str(frame_number),
                str(track.track_id),
                csv_text(output.slot.label),
                str(output.slot.task_id),
                str(output.slot.frame_number),
                str(output.slot.anchor_frame_number),
                str(output.slot.last_refresh_frame),
                "1" if output.slot.active else "0",
                csv_text(track.active_lorat_slot),
                csv_float(output.confidence),
                csv_float(calibrated_confidence),
                csv_float(output.slot.confidence_baseline),
                csv_float(track.confidence_baseline),
                *csv_bbox(output.bbox),
                *csv_bbox(track.bbox),
                *csv_bbox(track.predicted_bbox),
                csv_float(score.total),
                csv_float(score.appearance),
                csv_float(score.motion),
                csv_float(score.path),
                csv_float(score.source),
                csv_float(score.confidence),
                csv_float(score.iou),
                csv_float(score.initial_anchor),
                csv_float(score.other_anchor),
                str(score.other_track_id) if score.other_track_id is not None else "",
                csv_float(score.identity_margin),
                str(score.occlusion_track_id) if score.occlusion_track_id is not None else "",
                csv_float(score.occlusion_iou),
            ]
            self.slot_debug_lines.append(",".join(fields) + "\n")

    def _best_owned_fallback_assignment(
        self,
        track: TrackState,
        owned_outputs: Sequence[Tuple[LoRATMemorySlot, BBox, Optional[float]]],
        assigned_output_keys: set[Tuple[int, int]],
        evaluated_tracks: Sequence[TrackState],
        frame: Optional[np.ndarray],
    ) -> Optional[IdentityAssignment]:
        candidates: List[IdentityAssignment] = []
        for slot, bbox, confidence in owned_outputs:
            if (track.track_id, slot.task_id) in assigned_output_keys:
                continue
            output = LoRATSlotOutput(
                source_track_id=track.track_id,
                slot=slot,
                bbox=bbox,
                confidence=confidence,
            )
            output = self.identity_arbitrator._with_appearance(output, frame)
            score = self.identity_arbitrator.score(track, output, evaluated_tracks)
            candidates.append(
                IdentityAssignment(
                    track=track,
                    output=output,
                    score=score,
                    assignment_margin=0.0,
                )
            )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda assignment: (
                assignment.score.total,
                assignment.score.confidence,
                assignment.score.appearance,
            ),
        )

    def _accept_lorat_output(
        self,
        track: TrackState,
        slot: LoRATMemorySlot,
        bbox: BBox,
        confidence: Optional[float],
        frame: Optional[np.ndarray],
        frame_number: int,
        identity_assignment: Optional[IdentityAssignment] = None,
    ) -> bool:
        clipped = clip_bbox_to_frame(frame, bbox) if frame is not None else clamp_bbox_size(bbox)
        if clipped is None:
            return False

        raw_bbox = tuple(float(value) for value in clipped)
        raw_confidence = confidence
        calibrated_confidence = self._calibrate_lorat_confidence(track, slot, raw_confidence)
        if self.lorat_fixed_box_size:
            accepted_bbox, fixed_size_applied = self._apply_fixed_box_size(track, raw_bbox, frame)
            scale_tokens: List[str] = []
        else:
            accepted_bbox, scale_tokens = self._apply_scale_limits(track, raw_bbox, frame)
            fixed_size_applied = False
        reject_state = self._candidate_reject_state(
            track,
            accepted_bbox,
            calibrated_confidence,
            identity_assignment,
        )
        if reject_state is not None:
            track.raw_bbox = raw_bbox
            track.raw_confidence = raw_confidence
            track.confidence = calibrated_confidence
            track.state = reject_state
            if identity_assignment is not None:
                track.assignment_score = identity_assignment.score.total
                track.assignment_margin = identity_assignment.assignment_margin
                track.reid_score = identity_assignment.score.appearance
                track.motion_score = identity_assignment.score.motion
                track.path_score = identity_assignment.score.path
                track.source_score = identity_assignment.score.source
                track.initial_anchor_score = identity_assignment.score.initial_anchor
                track.other_anchor_score = identity_assignment.score.other_anchor
                track.other_anchor_track_id = identity_assignment.score.other_track_id
                track.identity_margin = identity_assignment.score.identity_margin
                track.occlusion_track_id = identity_assignment.score.occlusion_track_id
                track.occlusion_iou = identity_assignment.score.occlusion_iou
            return False

        (
            learning_held,
            learning_hold_reasons,
            crop_info,
            area_ratio,
            window_area_ratio,
            projected_shrink_frames,
        ) = self._assess_learning_hold(
            track,
            accepted_bbox,
            calibrated_confidence,
            identity_assignment,
            frame,
        )
        occlusion_track_id, occlusion_iou = self._candidate_occlusion_info(track, accepted_bbox)
        candidate_occluded = occlusion_track_id is not None
        previous = track.bbox
        track.previous_bbox = previous
        track.raw_bbox = raw_bbox
        track.bbox = accepted_bbox
        track.raw_confidence = raw_confidence
        track.confidence = calibrated_confidence
        track.velocity = bbox_delta(previous, accepted_bbox)
        track.ok = True
        track.lost_frames = 0
        track.occluded_frames = track.occluded_frames + 1 if candidate_occluded else 0
        track.occlusion_track_id = occlusion_track_id
        track.occlusion_iou = occlusion_iou
        track.shrink_risk_frames = projected_shrink_frames
        track.learning_held_frames = track.learning_held_frames + 1 if learning_held else 0
        track.learning_block_reason = ",".join(learning_hold_reasons) if learning_held else ""
        track.last_area_ratio = area_ratio
        track.last_window_area_ratio = window_area_ratio
        track.last_crop_info_score = crop_info.score
        track.last_crop_edge_density = crop_info.edge_density
        track.last_crop_laplacian_var = crop_info.laplacian_var
        track.last_crop_contrast = crop_info.contrast
        state_prefix = "LORAT" if slot.label == "initial" else f"LORAT-{slot.label.upper()}"
        if identity_assignment is not None and identity_assignment.output.source_track_id != track.track_id:
            state_prefix = f"REID-{state_prefix}"
            track.assigned_source = f"lorat-track-{identity_assignment.output.source_track_id}-{slot.label}"
        else:
            track.assigned_source = f"lorat-{slot.label}"
        if fixed_size_applied:
            state_prefix = append_state_token(state_prefix, "FIXEDSIZE")
        for token in scale_tokens:
            state_prefix = append_state_token(state_prefix, token)
        if identity_assignment is not None and self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            identity_assignment.score,
        ):
            state_prefix = append_state_token(state_prefix, "VIEWCHANGE")
        if self._is_reid_recovery(
            track,
            0.0 if calibrated_confidence is None else float(calibrated_confidence),
            identity_assignment,
        ):
            state_prefix = append_state_token(state_prefix, "REIDRECOVERY")
        if candidate_occluded:
            state_prefix = append_state_token(state_prefix, "OCCLUSION")
        if learning_held:
            state_prefix = append_state_token(state_prefix, "NOLEARN")
            for reason in learning_hold_reasons:
                state_prefix = append_state_token(state_prefix, reason)
        track.state = state_prefix
        track.active_lorat_slot = slot.label
        track.active_template_frame = slot.frame_number
        slot.bbox = accepted_bbox
        slot.raw_confidence = raw_confidence
        slot.confidence = calibrated_confidence
        if identity_assignment is not None:
            track.assignment_score = identity_assignment.score.total
            track.assignment_margin = identity_assignment.assignment_margin
            track.reid_score = identity_assignment.score.appearance
            track.motion_score = identity_assignment.score.motion
            track.path_score = identity_assignment.score.path
            track.source_score = identity_assignment.score.source
            track.initial_anchor_score = identity_assignment.score.initial_anchor
            track.other_anchor_score = identity_assignment.score.other_anchor
            track.other_anchor_track_id = identity_assignment.score.other_track_id
            track.identity_margin = identity_assignment.score.identity_margin
            track.occlusion_track_id = identity_assignment.score.occlusion_track_id
            track.occlusion_iou = identity_assignment.score.occlusion_iou
            if not candidate_occluded and not learning_held:
                self.identity_arbitrator.commit_track_memory(
                    track,
                    identity_assignment.output,
                    identity_assignment,
                    frame,
                )
        if track.kalman is None:
            track.kalman = BBoxKalmanFilter(accepted_bbox)
        track.kalman.update(accepted_bbox, calibrated_confidence)
        if not candidate_occluded and not learning_held:
            self._update_confidence_baseline(track, slot, raw_confidence)
            track.last_reliable_bbox = accepted_bbox
            track.last_reliable_frame = frame_number
            record_reliable_track_trajectory(track, frame_number, accepted_bbox, self.trajectory_history_size)
            self._commit_trusted_size(track, accepted_bbox)
        self._record_size_history(track, frame_number, accepted_bbox)
        record_track_trajectory(track, frame_number, accepted_bbox, self.trajectory_history_size)
        return True

    def _mark_lorat_miss(
        self,
        track: TrackState,
        frame_number: int = 0,
        state: str = "LORAT_MISS",
        source: str = "lorat-miss",
        raw_bbox: Optional[BBox] = None,
        confidence: Optional[float] = None,
        raw_confidence: Optional[float] = None,
        preserve_scores: bool = False,
    ) -> None:
        previous = track.bbox
        previous_confidence = track.confidence
        predicted = track.predicted_bbox or predict_bbox(track.bbox, track.velocity)
        predicted = clamp_bbox_to_frame_bounds(None, predicted)
        track.previous_bbox = track.bbox
        track.raw_bbox = raw_bbox
        track.raw_confidence = raw_confidence
        if confidence is None:
            track.confidence = max(0.0, float(previous_confidence or 0.0) * self.occlusion_velocity_damping)
        else:
            track.confidence = confidence
        track.bbox = predicted
        predicted_delta = bbox_delta(previous, predicted)
        track.velocity = tuple(float(delta * self.occlusion_velocity_damping) for delta in predicted_delta)
        if track.kalman is not None:
            track.kalman.state[4:] *= self.occlusion_velocity_damping
        track.lost_frames += 1
        track.occluded_frames += 1
        track.ok = self.occlusion_max_frames > 0 and track.lost_frames <= self.occlusion_max_frames
        track.state = append_state_token(state, "OCCLUDED") if track.ok else append_state_token(state, "LOST")
        track.assigned_source = source
        if not preserve_scores:
            track.assignment_score = None
            track.assignment_margin = None
            track.reid_score = None
            track.motion_score = None
            track.path_score = None
            track.source_score = None
            track.initial_anchor_score = None
            track.other_anchor_score = None
            track.other_anchor_track_id = None
            track.identity_margin = None
            track.occlusion_track_id = None
            track.occlusion_iou = None
        if frame_number > 0:
            record_track_trajectory(track, frame_number, track.bbox, self.trajectory_history_size)

    def _run_worker_tasks(self, worker_tasks: Sequence[object]):
        from trackit.data.protocol.eval_input import TrackerEvalData

        merged_outputs = {"evaluated_frames": []}
        for task_chunk in chunk_sequence(worker_tasks, self.track_batch_size):
            task_count = len(task_chunk)
            self.runtime_status.evaluator_calls += 1
            self.runtime_status.evaluator_tasks += task_count
            self.runtime_status.max_evaluator_batch = max(self.runtime_status.max_evaluator_batch, task_count)
            transformed_tasks = tuple(self.transform(task) for task in task_chunk)
            data = TrackerEvalData(transformed_tasks, {})
            outputs = self.evaluator.run(
                data,
                self.optimized_model.model,
                self.optimized_model.raw_model,
            )
            if outputs is None:
                continue
            for key, value in outputs.items():
                if key == "evaluated_frames":
                    merged_outputs.setdefault(key, []).extend(value)
                elif key not in merged_outputs:
                    merged_outputs[key] = value
            self._update_gpu_memory_status()
        return merged_outputs

    def _record_frame_status(self, elapsed_seconds: float) -> None:
        self.runtime_status.last_frame_seconds = elapsed_seconds
        instant_fps = 1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0
        if self.runtime_status.fps <= 0:
            self.runtime_status.fps = instant_fps
        else:
            alpha = self._fps_smoothing
            self.runtime_status.fps = (alpha * instant_fps) + ((1.0 - alpha) * self.runtime_status.fps)
        self.runtime_status.active_objects = sum(1 for track in self.tracks if track.ok)
        self._update_gpu_memory_status()

    def _update_gpu_memory_status(self) -> None:
        if not hasattr(self, "torch"):
            return
        if getattr(self, "device", None) is None or self.device.type != "cuda":
            return
        allocated = self.torch.cuda.memory_allocated(self.device)
        reserved = self.torch.cuda.memory_reserved(self.device)
        peak_allocated = self.torch.cuda.max_memory_allocated(self.device)
        peak_reserved = self.torch.cuda.max_memory_reserved(self.device)
        self.runtime_status.gpu_allocated_mb = bytes_to_mb(allocated)
        self.runtime_status.gpu_reserved_mb = bytes_to_mb(reserved)
        self.runtime_status.gpu_peak_allocated_mb = bytes_to_mb(peak_allocated)
        self.runtime_status.gpu_peak_reserved_mb = bytes_to_mb(peak_reserved)

    def runtime_status_snapshot(self) -> RuntimeStatus:
        self._update_gpu_memory_status()
        return copy.copy(self.runtime_status)

    def status_lines(self) -> List[str]:
        status = self.runtime_status_snapshot()
        memory = "GPU n/a"
        if status.gpu_reserved_mb is not None:
            memory = (
                f"GPU {status.gpu_allocated_mb:.0f}/{status.gpu_reserved_mb:.0f} MB "
                f"(peak {status.gpu_peak_reserved_mb:.0f} MB)"
            )
        lines = [
            f"FPS {status.fps:.1f} | Objects {status.active_objects} | {memory}",
            f"Mode {V5_EXECUTION_MODE} | Eval calls {status.evaluator_calls} | max batch {status.max_evaluator_batch}",
        ]
        if status.model_forward_calls > 0:
            lines.append(
                f"Model forwards {status.model_forward_calls} | max model batch {status.max_model_forward_batch} | "
                f"max fusion batch {status.max_fusion_forward_batch}"
            )
        held_tracks = [track for track in self.tracks if track.learning_block_reason]
        if held_tracks:
            reasons = sorted({reason for track in held_tracks for reason in track.learning_block_reason.split(",") if reason})
            lines.append(f"No-learn {len(held_tracks)} | {','.join(reasons)}")
        return lines

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.lorat_slots_by_task_id:
            from trackit.data.methods.siamese_tracker_eval import SiameseTrackerEvalDataWorker_Task

            tasks = [
                SiameseTrackerEvalDataWorker_Task(
                    task_index=slot.task_id,
                    do_task_creation=None,
                    do_tracker_init=None,
                    do_tracker_track=None,
                    do_task_finalization=True,
                )
                for slot in self.lorat_slots_by_task_id.values()
            ]
            self._run_worker_tasks(tasks)

        self.evaluator.stop(self.evaluator_context)
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI and input source setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version 5 LoRAT-backed multi-object GUI with shared batched LoRAT inference."
    )
    parser.add_argument("--video", help="Path to a video file or camera index. Use 0 for webcam.")
    parser.add_argument(
        "--sequence",
        type=Path,
        help="Path to a DanceTrack/MOT17-style sequence folder, usually one containing img1.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root containing DanceTrack or MOT17 sequences. Use with --dataset and --sequence-name.",
    )
    parser.add_argument("--dataset", choices=("dancetrack", "mot17"), help="Dataset layout for --dataset-root.")
    parser.add_argument("--sequence-name", help="Sequence folder name to run from --dataset-root.")
    parser.add_argument("--list-sequences", action="store_true", help="List resolved dataset sequences and exit.")
    parser.add_argument("--sequence-fps", type=float, default=30.0, help="Playback FPS for image sequence inputs.")
    parser.add_argument("--device", default="cpu", help="LoRAT device: cpu, dml/directml for AMD/Windows, cuda:0 for NVIDIA.")
    parser.add_argument("--lorat-root", type=Path, default=DEFAULT_LORAT_ROOT, help="Local LoRAT checkout.")
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT weight. Defaults from --lorat-config.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Optional track cap. 0 means no cap.")
    parser.add_argument("--track-batch-size", type=int, default=8, help="Internal LoRAT task chunk size.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable LoRAT automatic mixed precision.")
    parser.add_argument(
        "--lorat-memory-slots",
        type=int,
        default=DEFAULT_LORAT_MEMORY_SLOTS,
        help=(
            "Internal LoRAT slots per visible track. "
            "5 means one first-frame anchor plus four rolling recent LoRAT templates; capped at 16."
        ),
    )
    parser.add_argument(
        "--lorat-memory-refresh-interval",
        type=int,
        default=DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL,
        help="Frames between rolling LoRAT memory refreshes. 1 keeps the recent slots frame-by-frame.",
    )
    parser.add_argument(
        "--lorat-active-slots-per-track",
        type=int,
        default=DEFAULT_LORAT_ACTIVE_SLOTS_PER_TRACK,
        help=(
            "Maximum LoRAT memory slots evaluated per track per frame. "
            "0 evaluates the full memory bank."
        ),
    )
    parser.add_argument(
        "--fixed-lorat-box-size",
        dest="lorat_fixed_box_size",
        action="store_true",
        default=DEFAULT_LORAT_FIXED_BOX_SIZE,
        help="Keep accepted boxes at the initial selected width/height.",
    )
    parser.add_argument(
        "--allow-lorat-size-change",
        dest="lorat_fixed_box_size",
        action="store_false",
        help="Allow LoRAT to change accepted box width/height. This is the default now.",
    )
    parser.add_argument(
        "--lorat-min-box-area",
        type=float,
        default=DEFAULT_LORAT_MIN_BOX_AREA,
        help="Hard minimum accepted LoRAT box area in pixels. 0 disables this limit.",
    )
    parser.add_argument(
        "--lorat-max-area-change-per-frame",
        type=float,
        default=DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME,
        help="Maximum accepted per-frame box area ratio. 1.05 means +/- about 5 percent area per frame; 0 disables.",
    )
    parser.add_argument(
        "--lorat-trusted-size-floor-scale",
        type=float,
        default=DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE,
        help=(
            "Minimum accepted box width/height as a fraction of trusted size memory. "
            "V5 keeps this permissive by default; 0 disables the geometric floor."
        ),
    )
    parser.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    parser.add_argument(
        "--lorat-accept-min-score",
        type=float,
        default=DEFAULT_LORAT_ACCEPT_MIN_SCORE,
        help=(
            "Minimum LoRAT confidence required before a box can update the visible track. "
            "Lower-confidence boxes are held with Kalman occlusion prediction instead."
        ),
    )
    parser.add_argument(
        "--shrink-guard-window",
        type=int,
        default=DEFAULT_SHRINK_GUARD_WINDOW,
        help="Recent accepted-box window used to detect cumulative area shrink. 0 disables the window check.",
    )
    parser.add_argument(
        "--shrink-guard-area-ratio",
        type=float,
        default=DEFAULT_SHRINK_GUARD_AREA_RATIO,
        help="Hold learning when current area falls below this ratio of the recent max area unless evidence is strong.",
    )
    parser.add_argument(
        "--shrink-guard-step-ratio",
        type=float,
        default=DEFAULT_SHRINK_GUARD_STEP_RATIO,
        help="Hold learning on a single-frame area shrink below this previous-area ratio unless evidence is strong.",
    )
    parser.add_argument(
        "--shrink-guard-min-confidence",
        type=float,
        default=DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE,
        help="Minimum calibrated LoRAT confidence required to learn from a shrink-risk update.",
    )
    parser.add_argument(
        "--shrink-guard-min-reid",
        type=float,
        default=DEFAULT_SHRINK_GUARD_MIN_REID,
        help="Minimum ReID appearance score required to learn from a shrink-risk update when appearance memory exists.",
    )
    parser.add_argument(
        "--crop-information-min-score",
        type=float,
        default=DEFAULT_CROP_INFORMATION_MIN_SCORE,
        help="Minimum crop information score before an accepted box can refresh template/ReID/size memory.",
    )
    parser.add_argument(
        "--crop-information-min-pixels",
        type=int,
        default=DEFAULT_CROP_INFORMATION_MIN_PIXELS,
        help="Pixel count at which the crop-information score stops being penalized for tiny crops.",
    )
    parser.add_argument(
        "--lorat-search-area-factor",
        type=float,
        default=DEFAULT_LORAT_SEARCH_AREA_FACTOR,
        help=(
            "LoRAT search crop area factor. Smaller values reduce drift to nearby similar targets; "
            "upstream LoRAT uses 4.0 for 224 configs and 5.0 for 378 configs."
        ),
    )
    parser.add_argument(
        "--lorat-window-penalty",
        type=float,
        default=DEFAULT_LORAT_WINDOW_PENALTY,
        help=(
            "LoRAT post-process window penalty. Higher values bias the tracker toward staying near the "
            "current target; upstream LoRAT uses 0.45."
        ),
    )
    parser.add_argument(
        "--lorat-state-update-min-score",
        type=float,
        default=DEFAULT_LORAT_STATE_UPDATE_MIN_SCORE,
        help="Minimum LoRAT confidence required before LoRAT updates its own search/crop state.",
    )
    parser.add_argument(
        "--lorat-state-update-max-center-shift",
        type=float,
        default=DEFAULT_LORAT_STATE_UPDATE_MAX_CENTER_SHIFT,
        help="Maximum center jump, measured as a multiple of the previous box diagonal, before LoRAT state update is held.",
    )
    parser.add_argument(
        "--lorat-state-update-max-area-change",
        type=float,
        default=DEFAULT_LORAT_STATE_UPDATE_MAX_AREA_CHANGE,
        help="Maximum area growth/shrink factor before LoRAT state update is held.",
    )
    parser.add_argument("--disable-identity-arbitration", action="store_true")
    parser.add_argument("--identity-min-score", type=float, default=DEFAULT_IDENTITY_MIN_SCORE)
    parser.add_argument("--identity-min-reid", type=float, default=DEFAULT_IDENTITY_MIN_REID)
    parser.add_argument("--identity-min-motion", type=float, default=DEFAULT_IDENTITY_MIN_MOTION)
    parser.add_argument(
        "--identity-min-path",
        type=float,
        default=DEFAULT_IDENTITY_MIN_PATH,
        help="Minimum reliable center-path score before lower-confidence crossing candidates are rejected.",
    )
    parser.add_argument("--identity-bank-size", type=int, default=12)
    parser.add_argument(
        "--identity-memory-min-confidence",
        type=float,
        default=DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE,
        help="Minimum accepted LoRAT confidence before updating the appearance/ReID memory bank.",
    )
    parser.add_argument(
        "--occlusion-max-frames",
        type=int,
        default=DEFAULT_OCCLUSION_MAX_FRAMES,
        help="Frames to keep a missed/rejected target alive with Kalman prediction. 0 disables occlusion holding.",
    )
    parser.add_argument(
        "--occlusion-iou-threshold",
        type=float,
        default=DEFAULT_OCCLUSION_IOU_THRESHOLD,
        help="If an accepted box overlaps another active track this much, skip memory refresh for that frame.",
    )
    parser.add_argument(
        "--occlusion-velocity-damping",
        type=float,
        default=DEFAULT_OCCLUSION_VELOCITY_DAMPING,
        help="Velocity multiplier applied each held occlusion frame so Kalman coasting does not run away.",
    )
    parser.add_argument(
        "--reid-recovery-min-score",
        type=float,
        default=DEFAULT_REID_RECOVERY_MIN_SCORE,
        help="Minimum total identity score for accepting a low-confidence LoRAT box as a ReID recovery.",
    )
    parser.add_argument(
        "--reid-recovery-min-reid",
        type=float,
        default=DEFAULT_REID_RECOVERY_MIN_REID,
        help="Minimum appearance score for accepting a low-confidence LoRAT box as a ReID recovery.",
    )
    parser.add_argument(
        "--reid-recovery-min-motion",
        type=float,
        default=DEFAULT_REID_RECOVERY_MIN_MOTION,
        help="Minimum motion score for accepting a low-confidence LoRAT box as a ReID recovery.",
    )
    parser.add_argument(
        "--reid-recovery-min-confidence",
        type=float,
        default=DEFAULT_REID_RECOVERY_MIN_CONFIDENCE,
        help="Lowest LoRAT confidence that can still be accepted when ReID and motion are strong.",
    )
    parser.add_argument(
        "--view-change-min-score",
        type=float,
        default=DEFAULT_VIEW_CHANGE_MIN_SCORE,
        help="Minimum identity score for same-slot smooth-motion pose/view adaptation.",
    )
    parser.add_argument(
        "--view-change-min-motion",
        type=float,
        default=DEFAULT_VIEW_CHANGE_MIN_MOTION,
        help="Minimum motion score for accepting an appearance-changing same-target update.",
    )
    parser.add_argument(
        "--view-change-min-confidence",
        type=float,
        default=DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE,
        help="Minimum raw LoRAT confidence for pose/view adaptation before ReID agrees.",
    )
    parser.add_argument(
        "--view-change-max-lost-frames",
        type=int,
        default=DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES,
        help="Maximum recent miss count where smooth-motion view adaptation is still allowed.",
    )
    parser.add_argument(
        "--lorat-slot-capacity",
        type=int,
        default=0,
        help="Maximum active internal LoRAT slots. 0 auto-scales from --track-batch-size/selected tracks.",
    )
    parser.add_argument("--output", type=Path, help="MOTChallenge-format result file.")
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Annotated MP4 output path. Defaults to outputs/lorat-gui-v5/<source>_lorat_v5_annotated.mp4.",
    )
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated MP4 writing.")
    parser.add_argument("--debug-log", type=Path, help="Tracking debug CSV output path. Defaults to outputs/debug.")
    parser.add_argument(
        "--slot-debug-log",
        type=Path,
        help="Per-LoRAT-memory-slot debug CSV output path. Defaults to outputs/debug.",
    )
    parser.add_argument("--no-slot-debug-log", action="store_true", help="Disable per-slot LoRAT debug CSV writing.")
    parser.add_argument("--debug-frame-start", type=int, default=0, help="First frame to include in --debug-log; 0 means all.")
    parser.add_argument("--debug-frame-end", type=int, default=0, help="Last frame to include in --debug-log; 0 means all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for smoke tests.")
    return parser.parse_args()

def resolve_dataset_sequence(args: argparse.Namespace) -> Optional[Path]:
    if args.dataset_root is None:
        return args.sequence

    sequence_dirs = find_dataset_sequences(args.dataset_root, args.dataset)
    if args.list_sequences:
        for path in sequence_dirs:
            print(path)
        return None

    if not args.sequence_name:
        raise RuntimeError("--sequence-name is required when using --dataset-root without --list-sequences.")

    matches = [path for path in sequence_dirs if path.name == args.sequence_name]
    if not matches:
        raise RuntimeError(f"Sequence {args.sequence_name!r} not found under {args.dataset_root}")
    return matches[0]


def find_dataset_sequences(root: Path, dataset: Optional[str]) -> List[Path]:
    if not root.exists():
        raise RuntimeError(f"Dataset root does not exist: {root}")

    candidates = []
    search_roots = [root]
    if dataset == "dancetrack":
        search_roots.extend(path for name in ("train", "val", "test") if (path := root / name).is_dir())
    elif dataset == "mot17":
        search_roots.extend(path for name in ("train", "test") if (path := root / name).is_dir())

    for search_root in search_roots:
        for path in search_root.iterdir():
            if path.is_dir() and (path / "img1").is_dir():
                candidates.append(path)

    return sorted(set(candidates), key=lambda path: str(path).lower())


def parse_video_source(value: str) -> VideoSource:
    try:
        return int(value)
    except ValueError:
        return value


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    sequence = resolve_dataset_sequence(args)
    if args.list_sequences:
        raise SystemExit(0)
    if sequence is not None:
        return ImageSequenceSource(sequence, args.sequence_fps)
    if args.video is None and DEFAULT_DANCETRACK_SEQUENCE.exists():
        print(f"No --video or --sequence was provided. Opening default DanceTrack sequence: {DEFAULT_DANCETRACK_SEQUENCE}")
        return ImageSequenceSource(DEFAULT_DANCETRACK_SEQUENCE, args.sequence_fps)
    if args.video is None:
        print("No --video or --sequence was provided. Falling back to webcam 0.")
        args.video = "0"
    return VideoCaptureSource(parse_video_source(args.video))


def create_backend(args: argparse.Namespace, source: FrameSource, expected_tracks: int = 0):
    weight_path = args.weight_path or LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return LoRATMultiObjectTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        args.max_tracks,
        args.track_batch_size,
        source.fps,
        source.length,
        source.name,
        args.disable_amp,
        args.lorat_memory_slots,
        args.lorat_memory_refresh_interval,
        args.lorat_active_slots_per_track,
        args.lorat_fixed_box_size,
        args.lorat_min_box_area,
        args.lorat_max_area_change_per_frame,
        args.lorat_trusted_size_floor_scale,
        args.lorat_memory_min_score,
        args.lorat_accept_min_score,
        args.shrink_guard_window,
        args.shrink_guard_area_ratio,
        args.shrink_guard_step_ratio,
        args.shrink_guard_min_confidence,
        args.shrink_guard_min_reid,
        args.crop_information_min_score,
        args.crop_information_min_pixels,
        args.lorat_slot_capacity,
        expected_tracks,
        args.lorat_search_area_factor,
        args.lorat_window_penalty,
        args.lorat_state_update_min_score,
        args.lorat_state_update_max_center_shift,
        args.lorat_state_update_max_area_change,
        not args.disable_identity_arbitration,
        args.identity_min_score,
        args.identity_min_reid,
        args.identity_min_motion,
        args.identity_min_path,
        args.identity_bank_size,
        args.identity_memory_min_confidence,
        args.occlusion_max_frames,
        args.occlusion_iou_threshold,
        args.occlusion_velocity_damping,
        args.reid_recovery_min_score,
        args.reid_recovery_min_reid,
        args.reid_recovery_min_motion,
        args.reid_recovery_min_confidence,
        args.view_change_min_score,
        args.view_change_min_motion,
        args.view_change_min_confidence,
        args.view_change_max_lost_frames,
    )

def make_frame_getter(frame: np.ndarray):
    return lambda: frame


# ---------------------------------------------------------------------------
# Geometry, ReID features, and small utilities
# ---------------------------------------------------------------------------

def color_for_track(track_id: int) -> Color:
    palette = (
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 0, 128),
    )
    return palette[(track_id - 1) % len(palette)]


def clip_bbox_to_frame(frame: np.ndarray, bbox: BBox) -> Optional[Tuple[int, int, int, int]]:
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = [int(round(value)) for value in bbox]
    left = max(0, min(x, frame_width - 1))
    top = max(0, min(y, frame_height - 1))
    right = max(left + 1, min(x + w, frame_width))
    bottom = max(top + 1, min(y + h, frame_height))
    width = right - left
    height = bottom - top
    if width <= 2 or height <= 2:
        return None
    return left, top, width, height








def xywh_to_xyxy_np(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array((x, y, x + w, y + h), dtype=np.float64)


def xyxy_to_xywh_tuple(bbox: np.ndarray) -> BBox:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)




def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def bbox_iou(left: BBox, right: BBox) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(lx + lw, rx + rw)
    inter_y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = (lw * lh) + (rw * rh) - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def preview_lorat_confidence(
    track: TrackState,
    slot: LoRATMemorySlot,
    raw_confidence: Optional[float],
) -> Optional[float]:
    if raw_confidence is None:
        return None
    raw_value = max(0.0, min(1.0, float(raw_confidence)))
    baseline = slot.confidence_baseline or track.confidence_baseline
    if baseline is None or baseline <= 0:
        baseline = max(raw_value, 0.05)
    return max(0.0, min(1.0, raw_value / max(0.05, float(baseline))))


def strongest_track_overlap(
    track: TrackState,
    bbox: BBox,
    tracks: Sequence[TrackState],
) -> Tuple[Optional[int], float]:
    best_track_id: Optional[int] = None
    best_iou = 0.0
    for other in tracks:
        if other.track_id == track.track_id or not other.ok:
            continue
        other_reference = other.predicted_bbox or other.bbox
        overlap = bbox_iou(bbox, other_reference)
        if overlap > best_iou:
            best_iou = overlap
            best_track_id = other.track_id
    return best_track_id, best_iou


def bbox_diagonal(bbox: BBox) -> float:
    _, _, w, h = bbox
    return float(np.hypot(w, h))


def bbox_area(bbox: BBox) -> float:
    _, _, w, h = bbox
    return max(1.0, float(w) * float(h))


def bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def bbox_aspect_ratio(bbox: BBox) -> float:
    _, _, w, h = bbox
    return max(0.01, float(w) / max(1.0, float(h)))


def center_distance(left: BBox, right: BBox) -> float:
    left_x, left_y = bbox_center(left)
    right_x, right_y = bbox_center(right)
    return float(np.hypot(left_x - right_x, left_y - right_y))


def scale_similarity(left: BBox, right: BBox) -> float:
    change = abs(np.log(bbox_area(right) / bbox_area(left)))
    return max(0.0, 1.0 - min(1.0, float(change)))


def aspect_similarity(left: BBox, right: BBox) -> float:
    change = abs(np.log(bbox_aspect_ratio(right) / bbox_aspect_ratio(left)))
    return max(0.0, 1.0 - min(1.0, float(change)))


def center_affinity(predicted: BBox, candidate: BBox, reference_diagonal: float) -> float:
    normalized_distance = center_distance(predicted, candidate) / max(1.0, reference_diagonal)
    return max(0.0, 1.0 - min(1.0, normalized_distance))


def motion_affinity(predicted: BBox, candidate: BBox, reference_diagonal: float) -> float:
    center_score = center_affinity(predicted, candidate, reference_diagonal)
    scale_score = scale_similarity(predicted, candidate)
    aspect_score = aspect_similarity(predicted, candidate)
    return (0.68 * center_score) + (0.22 * scale_score) + (0.10 * aspect_score)


def predict_bbox(bbox: BBox, velocity: BBox) -> BBox:
    return tuple(float(value + delta) for value, delta in zip(bbox, velocity))


def kalman_prediction_reference(track: TrackState) -> BBox:
    if track.predicted_bbox is not None:
        return track.predicted_bbox
    return predict_bbox(track.bbox, track.velocity)


def reliable_path_point_count(track: TrackState) -> int:
    return len(track.reliable_trajectory)


def path_gate_ready(track: TrackState) -> bool:
    if len(track.reliable_trajectory) < DEFAULT_PATH_GATE_MIN_RELIABLE_POINTS:
        return False
    first_frame = int(track.reliable_trajectory[0][0])
    last_frame = int(track.reliable_trajectory[-1][0])
    return (last_frame - first_frame) >= DEFAULT_PATH_GATE_MIN_FRAME_SPAN


def center_path_affinity(track: TrackState, candidate: BBox) -> float:
    if len(track.reliable_trajectory) < 2:
        return 0.50

    samples = track.reliable_trajectory[-6:]
    first_frame, first_bbox = samples[0]
    last_frame, last_bbox = samples[-1]
    dt = max(1, int(last_frame) - int(first_frame))
    first_center = np.asarray(bbox_center(first_bbox), dtype=np.float32)
    last_center = np.asarray(bbox_center(last_bbox), dtype=np.float32)
    velocity = (last_center - first_center) / float(dt)
    speed = float(np.linalg.norm(velocity))
    candidate_center = np.asarray(bbox_center(candidate), dtype=np.float32)
    reference_diagonal = max(1.0, bbox_diagonal(last_bbox))

    recent_centers = [np.asarray(bbox_center(bbox), dtype=np.float32) for _, bbox in samples]
    recent_steps = [
        float(np.linalg.norm(right - left))
        for left, right in zip(recent_centers, recent_centers[1:])
    ]
    median_step = float(np.median(np.asarray(recent_steps, dtype=np.float32))) if recent_steps else 0.0
    is_directional = speed >= DEFAULT_CENTER_PATH_DIRECTION_MIN_SPEED and median_step >= 1.0

    if not is_directional:
        distance = float(np.linalg.norm(candidate_center - last_center))
        local_radius = max(
            8.0,
            min(
                DEFAULT_CENTER_PATH_STATIONARY_RADIUS,
                (median_step * DEFAULT_CENTER_PATH_STATIONARY_STEP_FACTOR)
                + (reference_diagonal * DEFAULT_CENTER_PATH_STATIONARY_BOX_FACTOR),
            ),
        )
        path_score = max(0.0, 1.0 - min(1.0, distance / local_radius))
        previous_vector = recent_centers[-1] - recent_centers[-2]
        candidate_vector = candidate_center - last_center
        previous_distance = float(np.linalg.norm(previous_vector))
        candidate_distance = float(np.linalg.norm(candidate_vector))
        reversal_distance = max(6.0, median_step * DEFAULT_CENTER_PATH_REVERSAL_STEP_FACTOR)
        if previous_distance >= 2.0 and candidate_distance >= reversal_distance:
            cosine = float(
                np.dot(previous_vector, candidate_vector)
                / max(previous_distance * candidate_distance, 1e-6)
            )
            if cosine <= DEFAULT_CENTER_PATH_REVERSAL_MIN_COSINE:
                path_score *= DEFAULT_CENTER_PATH_REVERSAL_PENALTY
        return max(0.0, min(1.0, path_score))

    frames_since_reliable = max(1, track.lost_frames + 1)
    expected_center = last_center + (velocity * float(frames_since_reliable))
    distance = float(np.linalg.norm(candidate_center - expected_center))
    distance_score = max(0.0, 1.0 - min(1.0, distance / reference_diagonal))

    candidate_vector = candidate_center - last_center
    candidate_distance = float(np.linalg.norm(candidate_vector))
    if candidate_distance < 1.0:
        direction_score = 1.0
        lateral_score = 1.0
    else:
        unit_velocity = velocity / max(speed, 1e-6)
        unit_candidate = candidate_vector / max(candidate_distance, 1e-6)
        direction_score = max(0.0, float(np.dot(unit_velocity, unit_candidate)))
        lateral_distance = abs(float((unit_velocity[0] * candidate_vector[1]) - (unit_velocity[1] * candidate_vector[0])))
        lateral_score = max(0.0, 1.0 - min(1.0, lateral_distance / reference_diagonal))

    return max(
        0.0,
        min(
            1.0,
            (0.55 * distance_score) + (0.25 * direction_score) + (0.20 * lateral_score),
        ),
    )
























def bbox_delta(previous: BBox, current: BBox) -> BBox:
    return tuple(float(current_value - previous_value) for previous_value, current_value in zip(previous, current))









def record_track_trajectory(track: TrackState, frame_number: int, bbox: BBox, max_history: int) -> None:
    bbox = clamp_bbox_size(bbox)
    frame_number = int(frame_number)
    if track.trajectory and track.trajectory[-1][0] == frame_number:
        track.trajectory[-1] = (frame_number, bbox)
    else:
        track.trajectory.append((frame_number, bbox))

    max_history = max(2, max_history)
    if len(track.trajectory) > max_history:
        del track.trajectory[: len(track.trajectory) - max_history]


def record_reliable_track_trajectory(track: TrackState, frame_number: int, bbox: BBox, max_history: int) -> None:
    bbox = clamp_bbox_size(bbox)
    frame_number = int(frame_number)
    if track.reliable_trajectory and track.reliable_trajectory[-1][0] == frame_number:
        track.reliable_trajectory[-1] = (frame_number, bbox)
    else:
        track.reliable_trajectory.append((frame_number, bbox))

    max_history = max(2, max_history)
    if len(track.reliable_trajectory) > max_history:
        del track.reliable_trajectory[: len(track.reliable_trajectory) - max_history]












def clamp_bbox_size(bbox: BBox) -> BBox:
    x, y, w, h = bbox
    return float(x), float(y), max(1.0, float(w)), max(1.0, float(h))


def clamp_bbox_to_frame_bounds(frame: Optional[np.ndarray], bbox: BBox) -> BBox:
    bbox = clamp_bbox_size(bbox)
    if frame is None:
        return bbox

    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = bbox
    w = min(max(1.0, w), max(1.0, float(frame_width)))
    h = min(max(1.0, h), max(1.0, float(frame_height)))
    x = max(0.0, min(float(x), max(0.0, float(frame_width) - w)))
    y = max(0.0, min(float(y), max(0.0, float(frame_height) - h)))
    return x, y, w, h


def append_state_token(state: str, token: str) -> str:
    if not state:
        return token
    tokens = state.split("+")
    if token in tokens:
        return state
    return f"{state}+{token}"
























REGION_FEATURE_LENGTH = (12 * 6 * 4) + (8 * 8 * 4) + 16 + 9 + 1 + 12


def measure_crop_information(
    frame: Optional[np.ndarray],
    bbox: BBox,
    min_pixels: int = DEFAULT_CROP_INFORMATION_MIN_PIXELS,
) -> CropInformation:
    if frame is None:
        return CropInformation(score=1.0, pixel_count=min_pixels)
    clipped = clip_bbox_to_frame(frame, bbox)
    if clipped is None:
        return CropInformation()
    x, y, w, h = clipped
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return CropInformation()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pixel_count = int(gray.size)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if pixel_count > 0 else 0.0
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / max(1, pixel_count)
    contrast = float(gray.std())

    pixel_score = min(1.0, pixel_count / max(1.0, float(min_pixels)))
    texture_score = min(1.0, laplacian_var / 80.0)
    edge_score = min(1.0, edge_density / 0.08)
    contrast_score = min(1.0, contrast / 32.0)
    score = pixel_score * ((0.50 * texture_score) + (0.35 * edge_score) + (0.15 * contrast_score))
    return CropInformation(
        score=max(0.0, min(1.0, float(score))),
        edge_density=edge_density,
        laplacian_var=laplacian_var,
        contrast=contrast,
        pixel_count=pixel_count,
    )


def extract_reid_histogram(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    clipped = clip_bbox_to_frame(frame, bbox)
    if clipped is None:
        return None
    x, y, w, h = clipped
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    regions = [crop]
    half_h = max(1, crop.shape[0] // 2)
    half_w = max(1, crop.shape[1] // 2)
    regions.extend(
        (
            crop[:half_h, :half_w],
            crop[:half_h, half_w:],
            crop[half_h:, :half_w],
            crop[half_h:, half_w:],
        )
    )

    hist = np.concatenate([region_appearance_features(region) for region in regions]).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    return None if norm <= 0 else hist / norm


def region_appearance_features(region: np.ndarray) -> np.ndarray:
    if region.size == 0:
        return np.zeros(REGION_FEATURE_LENGTH, dtype=np.float32)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 4], [0, 180, 0, 256, 0, 256])

    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    lab_hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 4], [0, 256, 0, 256, 0, 256])

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([float(np.count_nonzero(edges)) / max(1, edges.size)], dtype=np.float32)
    gradient_hist = gradient_orientation_histogram(gray, 9)

    hsv_pixels = hsv.reshape(-1, 3).astype(np.float32)
    lab_pixels = lab.reshape(-1, 3).astype(np.float32)
    color_moments = np.concatenate(
        [
            hsv_pixels.mean(axis=0) / np.array([180.0, 256.0, 256.0], dtype=np.float32),
            hsv_pixels.std(axis=0) / np.array([180.0, 256.0, 256.0], dtype=np.float32),
            lab_pixels.mean(axis=0) / 256.0,
            lab_pixels.std(axis=0) / 256.0,
        ]
    ).astype(np.float32)

    feature = np.concatenate(
        [
            hsv_hist.flatten(),
            lab_hist.flatten(),
            gray_hist.flatten(),
            gradient_hist,
            edge_density,
            color_moments,
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def gradient_orientation_histogram(gray: np.ndarray, bins: int) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    bin_indices = np.floor(angle * (bins / 180.0)).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    hist = np.bincount(bin_indices.ravel(), weights=magnitude.ravel(), minlength=bins).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 0 else hist


def histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right))))


def track_has_appearance(track: TrackState) -> bool:
    return (
        track.initial_appearance_hist is not None
        or track.appearance_hist is not None
        or bool(track.appearance_bank)
    )


def track_appearance_similarity(track: TrackState, candidate_hist: np.ndarray) -> float:
    scores = []
    initial_score: Optional[float] = None
    if track.initial_appearance_hist is not None:
        initial_score = histogram_similarity(track.initial_appearance_hist, candidate_hist)
        scores.append(initial_score)
    if track.appearance_hist is not None:
        scores.append(histogram_similarity(track.appearance_hist, candidate_hist))
    scores.extend(histogram_similarity(memory, candidate_hist) for memory in track.appearance_bank)
    if not scores:
        return 0.50
    scores.sort(reverse=True)
    best_pair = scores[0] if len(scores) == 1 else (0.72 * scores[0]) + (0.28 * scores[1])
    if initial_score is None:
        return best_pair
    top_count = min(4, len(scores))
    top_average = float(sum(scores[:top_count]) / top_count)
    return (0.54 * best_pair) + (0.30 * initial_score) + (0.16 * top_average)


def assignment_margin(row: Sequence[float], assigned_col: int) -> float:
    assigned_score = float(row[assigned_col])
    alternatives = [float(score) for index, score in enumerate(row) if index != assigned_col]
    if not alternatives:
        return 1.0
    return assigned_score - max(alternatives)


def solve_assignment(score_matrix: Sequence[Sequence[float]], min_score: float) -> List[Tuple[int, int, float]]:
    if not score_matrix or not score_matrix[0]:
        return []
    scores = np.asarray(score_matrix, dtype=np.float32)
    row_indices, col_indices = linear_sum_assignment(-scores)
    assignments = []
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        score = float(scores[row_index, col_index])
        if score >= min_score:
            assignments.append((row_index, col_index, score))
    assignments.sort(key=lambda item: item[0])
    return assignments




















def chunk_sequence(items: Sequence[object], chunk_size: int):
    chunk_size = max(1, chunk_size)
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


# ---------------------------------------------------------------------------
# Selection UI, output writers, and playback loop
# ---------------------------------------------------------------------------

def bbox_measurement_text(width: float, length: float) -> str:
    width = max(0.0, float(width))
    length = max(0.0, float(length))
    area = width * length
    return f"W {width:.0f}px | L {length:.0f}px | A {area:.0f}px"


def draw_selection_label(
    image: np.ndarray,
    lines: Sequence[str],
    anchor: Tuple[int, int],
    color: Color,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    padding = 5
    line_gap = 4
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    label_width = max(width for width, _ in text_sizes) + (padding * 2)
    line_height = max(height for _, height in text_sizes)
    label_height = (line_height * len(lines)) + (line_gap * (len(lines) - 1)) + (padding * 2)

    frame_height, frame_width = image.shape[:2]
    x = max(0, min(anchor[0], frame_width - label_width))
    y = anchor[1] - label_height - 4
    if y < 0:
        y = min(frame_height - label_height, anchor[1] + 4)
    y = max(0, y)

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + label_width, y + label_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (x, y), (x + label_width, y + label_height), color, 1)

    text_y = y + padding + line_height
    for line in lines:
        cv2.putText(image, line, (x + padding, text_y), font, scale, color, thickness, cv2.LINE_AA)
        text_y += line_height + line_gap


def draw_selection_center(image: np.ndarray, bbox: BBox, color: Color) -> None:
    center_x, center_y = bbox_center(bbox)
    center = (int(round(center_x)), int(round(center_y)))
    cv2.drawMarker(
        image,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    label = f"C {center[0]},{center[1]}"
    cv2.putText(
        image,
        label,
        (center[0] + 8, max(16, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (center[0] + 8, max(16, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def select_boxes(frame: np.ndarray, title: str = "Select Objects") -> List[BBox]:
    print("Drag boxes with the mouse. ENTER/SPACE = start tracking. c/q/ESC = cancel. r = reset boxes.")
    boxes: List[BBox] = []
    drawing = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal drawing, start_point, current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing and start_point is not None:
            drawing = False
            current_point = (x, y)
            x1, y1 = start_point
            x2, y2 = x, y
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            width = right - left
            height = bottom - top
            if width > 2 and height > 2:
                boxes.append((float(left), float(top), float(width), float(height)))
            start_point = None
            current_point = None

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_callback)
    while True:
        preview = frame.copy()
        for index, bbox in enumerate(boxes, start=1):
            x, y, w, h = [int(round(value)) for value in bbox]
            color = color_for_track(index)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            draw_selection_label(preview, (f"Box {index}", bbox_measurement_text(w, h)), (x, y), color)
            draw_selection_center(preview, bbox, color)

        if drawing and start_point is not None and current_point is not None:
            x1, y1 = start_point
            x2, y2 = current_point
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            width = right - left
            length = bottom - top
            cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 255), 2)
            draw_selection_center(preview, (float(left), float(top), float(width), float(length)), (0, 255, 255))
            draw_selection_label(
                preview,
                ("Drawing", bbox_measurement_text(width, length)),
                (left, top),
                (0, 255, 255),
            )

        cv2.putText(
            preview,
            f"{len(boxes)} boxes | ENTER/SPACE done | c cancel | r reset",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"{len(boxes)} boxes | ENTER/SPACE done | c cancel | r reset",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(title, preview)
        key = cv2.waitKey(16) & 0xFF
        if key in (13, 32):
            cv2.destroyWindow(title)
            return boxes
        if key in (27, ord("c"), ord("q")):
            cv2.destroyWindow(title)
            return []
        if key == ord("r"):
            boxes.clear()
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return []
        except cv2.error:
            return []


def append_mot_results(lines: List[str], frame_number: int, tracks: Sequence[TrackState]) -> None:
    for track in tracks:
        if not track.ok:
            continue
        x, y, w, h = track.bbox
        confidence = track.confidence if track.confidence is not None else 1.0
        lines.append(
            f"{frame_number},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{confidence:.4f},-1,-1,-1\n"
        )


DEBUG_LOG_HEADER = (
    "frame,track_id,ok,state,confidence,raw_confidence,confidence_baseline,"
    "x,y,w,h,width_px,length_px,area_px,raw_x,raw_y,raw_w,raw_h,pred_x,pred_y,pred_w,pred_h,prev_x,prev_y,prev_w,prev_h,"
    "vel_x,vel_y,vel_w,vel_h,assignment_score,assignment_margin,reid_score,motion_score,path_score,source_score,"
    "initial_anchor_score,other_anchor_score,other_anchor_track_id,identity_margin,occlusion_track_id,occlusion_iou,"
    "assigned_source,lost_frames,occluded_frames,last_reliable_frame,template_frame,"
    "active_lorat_slot,lorat_memory_slot_count,appearance_bank_size\n"
)


SLOT_DEBUG_LOG_HEADER = (
    "frame,track_id,slot_label,slot_task_id,slot_template_frame,slot_anchor_frame,slot_last_refresh_frame,"
    "slot_active,active_lorat_slot_before,raw_confidence,calibrated_confidence,slot_confidence_baseline,"
    "track_confidence_baseline,raw_x,raw_y,raw_w,raw_h,track_x,track_y,track_w,track_h,"
    "pred_x,pred_y,pred_w,pred_h,score_total,reid_score,motion_score,path_score,source_score,"
    "score_confidence,iou_score,initial_anchor_score,other_anchor_score,other_anchor_track_id,"
    "identity_margin,occlusion_track_id,occlusion_iou\n"
)


def append_debug_rows(
    lines: List[str],
    frame_number: int,
    tracks: Sequence[TrackState],
    start_frame: int = 0,
    end_frame: int = 0,
) -> None:
    if start_frame > 0 and frame_number < start_frame:
        return
    if end_frame > 0 and frame_number > end_frame:
        return

    for track in tracks:
        fields = [
            str(frame_number),
            str(track.track_id),
            "1" if track.ok else "0",
            csv_text(track.state),
            csv_float(track.confidence),
            csv_float(track.raw_confidence),
            csv_float(track.confidence_baseline),
            *csv_bbox(track.bbox),
            *csv_bbox_measurements(track.bbox),
            *csv_bbox(track.raw_bbox),
            *csv_bbox(track.predicted_bbox),
            *csv_bbox(track.previous_bbox),
            *csv_bbox(track.velocity),
            csv_float(track.assignment_score),
            csv_float(track.assignment_margin),
            csv_float(track.reid_score),
            csv_float(track.motion_score),
            csv_float(track.path_score),
            csv_float(track.source_score),
            csv_float(track.initial_anchor_score),
            csv_float(track.other_anchor_score),
            str(track.other_anchor_track_id) if track.other_anchor_track_id is not None else "",
            csv_float(track.identity_margin),
            str(track.occlusion_track_id) if track.occlusion_track_id is not None else "",
            csv_float(track.occlusion_iou),
            csv_text(track.assigned_source),
            str(track.lost_frames),
            str(track.occluded_frames),
            str(track.last_reliable_frame) if track.last_reliable_frame else "",
            str(track.active_template_frame) if track.active_template_frame is not None else "",
            csv_text(track.active_lorat_slot),
            str(track.lorat_memory_slot_count),
            str(len(track.appearance_bank)),
        ]
        lines.append(",".join(fields) + "\n")


def write_debug_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEBUG_LOG_HEADER + "".join(lines), encoding="utf-8")


def write_slot_debug_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SLOT_DEBUG_LOG_HEADER + "".join(lines), encoding="utf-8")


def csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def csv_bbox(bbox: Optional[BBox]) -> List[str]:
    if bbox is None:
        return ["", "", "", ""]
    return [f"{float(value):.6f}" for value in bbox]


def csv_bbox_measurements(bbox: Optional[BBox]) -> List[str]:
    if bbox is None:
        return ["", "", ""]
    width = max(0.0, float(bbox[2]))
    length = max(0.0, float(bbox[3]))
    area = width * length
    return [f"{width:.6f}", f"{length:.6f}", f"{area:.6f}"]


def csv_text(value: object) -> str:
    text = "" if value is None else str(value)
    return '"' + text.replace('"', '""') + '"' if "," in text or '"' in text else text


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    frame_number: int,
    backend_label: str,
    status_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    output = frame.copy()
    header = f"Frame {frame_number} | {backend_label} | q quit | a add boxes | p pause"
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    if status_lines:
        y = 56
        for line in status_lines:
            cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 22

    for track in tracks:
        x, y, w, h = [int(round(value)) for value in track.bbox]
        color = track.color if track.ok else (0, 0, 255)
        label = f"ID {track.track_id}"
        if track.confidence is not None:
            label += f" {track.confidence:.2f}"
        if track.state:
            label += f" {track.state}"
        if not track.ok:
            label += " LOST"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return output


def default_output_path(source_name: str, backend: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_OUTPUT_DIR / f"{safe_name}_{backend}_tracks.txt"


def default_debug_log_path(source_name: str, backend: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_DEBUG_DIR / f"{safe_name}_{backend}_debug.csv"


def default_slot_debug_log_path(source_name: str, backend: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_DEBUG_DIR / f"{safe_name}_{backend}_slot_debug.csv"


def default_video_path(source_name: str, backend: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_OUTPUT_DIR / f"{safe_name}_{backend}_annotated.mp4"


def make_video_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open annotated video writer: {path}")
    return writer


def main() -> int:
    args = parse_args()
    frame_source = open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name, "lorat_v5")
    debug_log_path = args.debug_log or default_debug_log_path(frame_source.name, "lorat_v5")
    slot_debug_log_path = (
        None
        if args.no_slot_debug_log
        else (args.slot_debug_log or default_slot_debug_log_path(frame_source.name, "lorat_v5"))
    )
    save_video_path = None if args.no_save_video else (args.save_video or default_video_path(frame_source.name, "lorat_v5"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok, first_frame = frame_source.read()
    if not ok or first_frame is None:
        print("Unable to read the first frame.")
        return 1

    boxes = select_boxes(first_frame)
    if not boxes:
        print("No bounding boxes selected. Exiting.")
        frame_source.release()
        cv2.destroyAllWindows()
        return 0

    backend = create_backend(args, frame_source, len(boxes))
    writer = make_video_writer(save_video_path, frame_source.fps, first_frame) if save_video_path is not None else None
    mot_lines: List[str] = []
    debug_lines: List[str] = []
    frame_number = 1
    paused = False
    outputs_written = False

    def flush_outputs() -> None:
        nonlocal outputs_written
        if outputs_written:
            return
        output_path.write_text("".join(mot_lines), encoding="utf-8")
        print(f"Wrote MOTChallenge-format tracks to: {output_path}")
        write_debug_log(debug_log_path, debug_lines)
        print(f"Wrote debug CSV to: {debug_log_path}")
        if slot_debug_log_path is not None:
            write_slot_debug_log(slot_debug_log_path, backend.slot_debug_lines)
            print(f"Wrote LoRAT slot debug CSV to: {slot_debug_log_path}")
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        append_mot_results(mot_lines, frame_number, backend.tracks)
        append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
        if writer is not None:
            writer.write(draw_tracks(first_frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines()))

        while True:
            if not paused:
                ok, frame = frame_source.read()
                if not ok or frame is None:
                    break

                frame_number += 1
                backend.update(frame, frame_number)
                append_mot_results(mot_lines, frame_number, backend.tracks)
                append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
            else:
                frame = frame.copy()

            shown = draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines())
            if writer is not None and not paused:
                writer.write(shown)

            cv2.imshow("LoRAT Multi-Object Tracker v5", shown)
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("a"):
                paused = True
                new_boxes = select_boxes(frame, "Add Objects")
                if new_boxes:
                    added_tracks = backend.add_tracks(frame, new_boxes, frame_number)
                    append_mot_results(mot_lines, frame_number, added_tracks)
                    append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
                paused = False

            if args.max_frames > 0 and frame_number >= args.max_frames:
                break
    finally:
        backend.close()
        frame_source.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        flush_outputs()

    if save_video_path is not None:
        print(f"Wrote annotated video to: {save_video_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
