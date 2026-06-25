from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
VideoSource = Union[int, str]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lorat-gui"
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
    v8_initial_feature: Optional[object] = None
    v8_appearance_feature: Optional[object] = None
    v8_feature_bank: List[object] = field(default_factory=list)
    v8_initial_crop_feature: Optional[object] = None
    v8_appearance_crop_feature: Optional[object] = None
    v8_crop_feature_bank: List[object] = field(default_factory=list)


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
    v8_feature: Optional[object] = None
    v8_crop_feature: Optional[object] = None


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
    crop_reid_forward_calls: int = 0
    crop_reid_forward_items: int = 0
    max_crop_reid_batch: int = 0


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
        measurement_noise = np.diag([18.0, 18.0, 10.0, 10.0]).astype(np.float32) * measurement_noise_scale
        innovation = measurement - (self.observation @ self.state)
        innovation_covariance = self.observation @ self.covariance @ self.observation.T + measurement_noise
        kalman_gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + (kalman_gain @ innovation)
        self.covariance = (np.eye(8, dtype=np.float32) - (kalman_gain @ self.observation)) @ self.covariance
        return self.to_bbox()

    def to_bbox(self) -> BBox:
        center_x, center_y, w, h = [float(value) for value in self.state[:4]]
        w = max(1.0, w)
        h = max(1.0, h)
        return center_x - (w / 2.0), center_y - (h / 2.0), w, h


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


class LightweightIdentityArbitrator:
    """Small ReID/Hungarian layer that protects identity without steering the tracker every frame."""

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

        prepared_outputs = [self._with_appearance(output, frame) for output in outputs]
        score_details = [[self.score(track, output, tracks) for output in prepared_outputs] for track in tracks]
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
            track.appearance_hist = (((1.0 - update_rate) * track.appearance_hist) + (update_rate * hist)).astype(np.float32)
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
            assignments.append(IdentityAssignment(track=track, output=output, score=score, assignment_margin=1.0))
        return assignments


class TrackLifecycle:
    HEALTHY = "HEALTHY"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    MANUAL_REANCHOR = "MANUAL_REANCHOR"


UNCERTAIN_TOKENS = (
    "MISS",
    "LOWCONF",
    "ID_UNCERTAIN",
    "OCCLU",
    "NOLEARN",
    "SHRINK",
    "MEMHELD",
)


def bbox_text(bbox: Optional[BBox]) -> str:
    if bbox is None:
        return ""
    return ";".join(f"{float(value):.3f}" for value in bbox)


def classify_track_lifecycle(track: object) -> str:
    """Map tracker-specific state text into a shared lifecycle vocabulary."""
    state = str(getattr(track, "state", "") or "").upper()
    ok = bool(getattr(track, "ok", True))
    lost_frames = int(getattr(track, "lost_frames", 0) or 0)

    if "MANUAL_REANCHOR" in state:
        return TrackLifecycle.MANUAL_REANCHOR
    if "REIDRECOVERY" in state or "REACQUIRED" in state:
        return TrackLifecycle.REACQUIRED
    if "LOST" in state or not ok:
        return TrackLifecycle.LOST
    if lost_frames > 0 or any(token in state for token in UNCERTAIN_TOKENS):
        return TrackLifecycle.UNCERTAIN
    return TrackLifecycle.HEALTHY


def set_track_lifecycle(track: object, lifecycle: Optional[str] = None) -> str:
    value = lifecycle or classify_track_lifecycle(track)
    setattr(track, "track_lifecycle_state", value)
    return value


def feature_bank_size(track: object) -> int:
    bank = getattr(track, "v8_feature_bank", None)
    if bank is None:
        return 0
    try:
        return len(bank)
    except TypeError:
        return 0


def track_lifecycle_counts(tracks: Iterable[object]) -> Dict[str, int]:
    counts = {
        TrackLifecycle.HEALTHY: 0,
        TrackLifecycle.UNCERTAIN: 0,
        TrackLifecycle.LOST: 0,
        TrackLifecycle.REACQUIRED: 0,
        TrackLifecycle.MANUAL_REANCHOR: 0,
    }
    for track in tracks:
        lifecycle = set_track_lifecycle(track)
        counts[lifecycle] = counts.get(lifecycle, 0) + 1
    return counts


def choose_manual_reanchor_track(tracks: Sequence[object]) -> Optional[object]:
    """Choose the most urgent track for a one-action reanchor command."""
    if not tracks:
        return None
    priority = {
        TrackLifecycle.LOST: 0,
        TrackLifecycle.UNCERTAIN: 1,
        TrackLifecycle.REACQUIRED: 2,
        TrackLifecycle.HEALTHY: 3,
        TrackLifecycle.MANUAL_REANCHOR: 4,
    }
    return min(
        tracks,
        key=lambda track: (
            priority.get(classify_track_lifecycle(track), 9),
            -int(getattr(track, "lost_frames", 0) or 0),
            int(getattr(track, "track_id", 0) or 0),
        ),
    )


@dataclass(frozen=True)
class ManualReanchorEvent:
    event_type: str
    frame: int
    track_id: int
    old_bbox: Optional[BBox]
    new_bbox: BBox
    previous_state: str
    previous_lifecycle: str
    seconds_spent: Optional[float]
    source: str
    timestamp: str

    def to_row(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "frame": self.frame,
            "track_id": self.track_id,
            "old_bbox": bbox_text(self.old_bbox),
            "new_bbox": bbox_text(self.new_bbox),
            "previous_state": self.previous_state,
            "previous_lifecycle": self.previous_lifecycle,
            "seconds_spent": "" if self.seconds_spent is None else f"{self.seconds_spent:.3f}",
            "source": self.source,
        }


MANUAL_EVENT_FIELDS = [
    "timestamp",
    "event_type",
    "frame",
    "track_id",
    "old_bbox",
    "new_bbox",
    "previous_state",
    "previous_lifecycle",
    "seconds_spent",
    "source",
]


@dataclass
class ObjectProposal:
    proposal_id: int
    frame: int
    bbox: BBox
    score: float
    source: str = "class_agnostic"
    status: str = "pending"
    spawned_track_id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_row(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "proposal_id": self.proposal_id,
            "frame": self.frame,
            "status": self.status,
            "bbox": bbox_text(self.bbox),
            "bbox_x": f"{float(self.bbox[0]):.6f}",
            "bbox_y": f"{float(self.bbox[1]):.6f}",
            "bbox_w": f"{float(self.bbox[2]):.6f}",
            "bbox_h": f"{float(self.bbox[3]):.6f}",
            "bbox_area": f"{bbox_area(self.bbox):.6f}",
            "score": f"{float(self.score):.6f}",
            "source": self.source,
            "spawned_track_id": "" if self.spawned_track_id is None else self.spawned_track_id,
        }


PROPOSAL_EVENT_FIELDS = [
    "timestamp",
    "proposal_id",
    "frame",
    "status",
    "bbox",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "score",
    "source",
    "spawned_track_id",
]


class ProposalQueue:
    def __init__(self, max_pending: int = 64, iou_suppression: float = 0.70) -> None:
        self.max_pending = max(1, int(max_pending))
        self.iou_suppression = max(0.0, min(1.0, float(iou_suppression)))
        self._next_id = 1
        self._pending: List[ObjectProposal] = []
        self._history: List[ObjectProposal] = []
        self._cursor = 0

    @property
    def pending(self) -> List[ObjectProposal]:
        return list(self._pending)

    @property
    def history(self) -> List[ObjectProposal]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._pending)

    def current(self) -> Optional[ObjectProposal]:
        if not self._pending:
            return None
        self._cursor %= len(self._pending)
        return self._pending[self._cursor]

    def next(self) -> Optional[ObjectProposal]:
        if not self._pending:
            return None
        self._cursor = (self._cursor + 1) % len(self._pending)
        return self.current()

    def previous(self) -> Optional[ObjectProposal]:
        if not self._pending:
            return None
        self._cursor = (self._cursor - 1) % len(self._pending)
        return self.current()

    def add_frame_proposals(self, frame_number: int, proposals: Sequence[Tuple[BBox, float, str]]) -> List[ObjectProposal]:
        added: List[ObjectProposal] = []
        for bbox, score, source in proposals:
            if any(bbox_iou(bbox, existing.bbox) >= self.iou_suppression for existing in self._pending):
                continue
            proposal = ObjectProposal(
                proposal_id=self._next_id,
                frame=int(frame_number),
                bbox=bbox,
                score=max(0.0, min(1.0, float(score))),
                source=source,
            )
            self._next_id += 1
            self._pending.append(proposal)
            self._history.append(proposal)
            added.append(proposal)
        self._pending.sort(key=lambda proposal: proposal.score, reverse=True)
        if len(self._pending) > self.max_pending:
            overflow = self._pending[self.max_pending :]
            for proposal in overflow:
                proposal.status = "expired"
            self._pending = self._pending[: self.max_pending]
        if self._pending:
            self._cursor %= len(self._pending)
        else:
            self._cursor = 0
        return added

    def accept_current(self, spawned_track_id: Optional[int]) -> Optional[ObjectProposal]:
        proposal = self.current()
        if proposal is None:
            return None
        proposal.status = "accepted"
        proposal.spawned_track_id = spawned_track_id
        self._pending.pop(self._cursor)
        self._cursor = min(self._cursor, max(0, len(self._pending) - 1))
        return proposal

    def reject_current(self) -> Optional[ObjectProposal]:
        proposal = self.current()
        if proposal is None:
            return None
        proposal.status = "rejected"
        self._pending.pop(self._cursor)
        self._cursor = min(self._cursor, max(0, len(self._pending) - 1))
        return proposal


def write_proposal_event_csv(path: Path, rows: Sequence[ObjectProposal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROPOSAL_EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def make_manual_reanchor_event(
    frame: int,
    track_id: int,
    old_bbox: Optional[BBox],
    new_bbox: BBox,
    previous_state: str,
    previous_lifecycle: str,
    seconds_spent: Optional[float] = None,
    source: str = "manual",
) -> ManualReanchorEvent:
    return ManualReanchorEvent(
        event_type="manual_reanchor",
        frame=int(frame),
        track_id=int(track_id),
        old_bbox=old_bbox,
        new_bbox=new_bbox,
        previous_state=previous_state,
        previous_lifecycle=previous_lifecycle,
        seconds_spent=seconds_spent,
        source=source,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def write_manual_event_csv(path: Path, rows: Sequence[ManualReanchorEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANUAL_EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def matched_gt_id_from_ious(
    own_gt_id: int,
    own_iou: float,
    other_ious: Sequence[Tuple[int, float]],
    min_match_iou: float,
) -> Tuple[Optional[int], float]:
    candidates = [(own_gt_id, own_iou), *other_ious]
    matched_id, matched_iou = max(candidates, key=lambda item: item[1], default=(None, 0.0))  # type: ignore[arg-type]
    if matched_id is None or matched_iou < min_match_iou:
        return None, float(matched_iou)
    return int(matched_id), float(matched_iou)


def resolve_dataset_sequence(args: object) -> Optional[Path]:
    if getattr(args, "dataset_root", None) is None:
        return getattr(args, "sequence", None)

    dataset_root = getattr(args, "dataset_root")
    sequence_dirs = find_dataset_sequences(dataset_root, getattr(args, "dataset", None))
    if getattr(args, "list_sequences", False):
        for path in sequence_dirs:
            print(path)
        return None

    sequence_name = getattr(args, "sequence_name", None)
    if not sequence_name:
        raise RuntimeError("--sequence-name is required when using --dataset-root without --list-sequences.")

    matches = [path for path in sequence_dirs if path.name == sequence_name]
    if not matches:
        raise RuntimeError(f"Sequence {sequence_name!r} not found under {dataset_root}")
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


def open_frame_source(args: object) -> FrameSource:
    sequence = resolve_dataset_sequence(args)
    if getattr(args, "list_sequences", False):
        raise SystemExit(0)
    if sequence is not None:
        return ImageSequenceSource(sequence, getattr(args, "sequence_fps", 30.0))
    if getattr(args, "video", None) is None and DEFAULT_DANCETRACK_SEQUENCE.exists():
        print(f"No --video or --sequence was provided. Opening default DanceTrack sequence: {DEFAULT_DANCETRACK_SEQUENCE}")
        return ImageSequenceSource(DEFAULT_DANCETRACK_SEQUENCE, getattr(args, "sequence_fps", 30.0))
    if getattr(args, "video", None) is None:
        print("No --video or --sequence was provided. Falling back to webcam 0.")
        setattr(args, "video", "0")
    return VideoCaptureSource(parse_video_source(getattr(args, "video")))


def make_frame_getter(frame: np.ndarray):
    return lambda: frame


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
    recent_steps = [float(np.linalg.norm(right - left)) for left, right in zip(recent_centers, recent_centers[1:])]
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
            cosine = float(np.dot(previous_vector, candidate_vector) / max(previous_distance * candidate_distance, 1e-6))
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

    return max(0.0, min(1.0, (0.55 * distance_score) + (0.25 * direction_score) + (0.20 * lateral_score)))


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


def _proposal_score(frame: np.ndarray, bbox: BBox, min_area: float) -> float:
    info = measure_crop_information(frame, bbox, max(16, int(min_area // 4)))
    area = bbox_area(bbox)
    area_score = min(1.0, area / max(1.0, float(min_area) * 4.0))
    return max(0.0, min(1.0, (0.70 * info.score) + (0.30 * area_score)))


def _filter_proposal_bbox(
    frame: np.ndarray,
    bbox: BBox,
    min_area: float,
    max_area_fraction: float,
    min_side: int,
    max_aspect_ratio: float,
) -> Optional[BBox]:
    clipped = clamp_bbox_to_frame_bounds(frame, bbox)
    if clipped is None:
        return None
    _, _, width, height = clipped
    area = bbox_area(clipped)
    frame_area = float(frame.shape[0] * frame.shape[1])
    if width < min_side or height < min_side:
        return None
    if area < min_area or area > (frame_area * max_area_fraction):
        return None
    aspect = bbox_aspect_ratio(clipped)
    if aspect > max_aspect_ratio or (1.0 / aspect) > max_aspect_ratio:
        return None
    return clipped


def _nms_scored_bboxes(candidates: Sequence[Tuple[BBox, float, str]], iou_threshold: float, limit: int) -> List[Tuple[BBox, float, str]]:
    selected: List[Tuple[BBox, float, str]] = []
    for bbox, score, source in sorted(candidates, key=lambda item: item[1], reverse=True):
        if any(bbox_iou(bbox, chosen_bbox) >= iou_threshold for chosen_bbox, _, _ in selected):
            continue
        selected.append((bbox, score, source))
        if len(selected) >= limit:
            break
    return selected


def _selective_search_proposals(
    frame: np.ndarray,
    min_area: float,
    max_area_fraction: float,
    min_side: int,
    max_aspect_ratio: float,
    candidate_limit: int,
) -> List[Tuple[BBox, float, str]]:
    ximgproc = getattr(cv2, "ximgproc", None)
    segmentation = getattr(ximgproc, "segmentation", None) if ximgproc is not None else None
    if segmentation is None or not hasattr(segmentation, "createSelectiveSearchSegmentation"):
        return []

    max_dimension = 720.0
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(1.0, float(max(height, width))))
    search_frame = frame if scale >= 0.999 else cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))))

    search = segmentation.createSelectiveSearchSegmentation()
    search.setBaseImage(search_frame)
    search.switchToSelectiveSearchFast()
    rects = search.process()
    candidates: List[Tuple[BBox, float, str]] = []
    max_rects = max(candidate_limit * 500, 5000)
    inv_scale = 1.0 / max(scale, 1e-6)
    for x, y, width, height in rects[:max_rects]:
        bbox = _filter_proposal_bbox(
            frame,
            (float(x) * inv_scale, float(y) * inv_scale, float(width) * inv_scale, float(height) * inv_scale),
            min_area,
            max_area_fraction,
            min_side,
            max_aspect_ratio,
        )
        if bbox is None:
            continue
        candidates.append((bbox, _proposal_score(frame, bbox, min_area), "selective_search"))
    return candidates


def _contour_proposals(
    frame: np.ndarray,
    min_area: float,
    max_area_fraction: float,
    min_side: int,
    max_aspect_ratio: float,
) -> List[Tuple[BBox, float, str]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[BBox, float, str]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        bbox = _filter_proposal_bbox(
            frame,
            (float(x), float(y), float(width), float(height)),
            min_area,
            max_area_fraction,
            min_side,
            max_aspect_ratio,
        )
        if bbox is None:
            continue
        score = _proposal_score(frame, bbox, min_area)
        contour_area = float(cv2.contourArea(contour))
        fill_score = min(1.0, contour_area / max(1.0, bbox_area(bbox)))
        candidates.append((bbox, max(0.0, min(1.0, (0.75 * score) + (0.25 * fill_score))), "contour"))
    return candidates


def generate_class_agnostic_proposals(
    frame: np.ndarray,
    source: str = "auto",
    max_proposals: int = 12,
    min_area: float = 256.0,
    nms_iou: float = 0.70,
    max_area_fraction: float = 0.80,
    min_side: int = 8,
    max_aspect_ratio: float = 8.0,
) -> List[Tuple[BBox, float, str]]:
    """Return class-agnostic candidate boxes as (bbox, score, source)."""
    max_proposals = max(1, int(max_proposals))
    candidate_limit = max(max_proposals * 4, max_proposals)
    candidates: List[Tuple[BBox, float, str]] = []

    if source in ("auto", "selective_search"):
        candidates.extend(
            _selective_search_proposals(
                frame,
                min_area=min_area,
                max_area_fraction=max_area_fraction,
                min_side=min_side,
                max_aspect_ratio=max_aspect_ratio,
                candidate_limit=candidate_limit,
            )
        )
        if source == "selective_search":
            return _nms_scored_bboxes(candidates, nms_iou, max_proposals)

    if source in ("auto", "contour") and (source == "contour" or len(candidates) < max_proposals):
        candidates.extend(
            _contour_proposals(
                frame,
                min_area=min_area,
                max_area_fraction=max_area_fraction,
                min_side=min_side,
                max_aspect_ratio=max_aspect_ratio,
            )
        )

    return _nms_scored_bboxes(candidates, nms_iou, max_proposals)


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
    cv2.putText(image, label, (center[0] + 8, max(16, center[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, label, (center[0] + 8, max(16, center[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def select_boxes(frame: np.ndarray, title: str = "Select Objects") -> List[BBox]:
    print("Drag boxes with the mouse. ENTER/SPACE = start tracking. c/q/ESC = cancel. r = reset boxes.")
    boxes: List[BBox] = []
    drawing = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
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
            draw_selection_label(preview, ("Drawing", bbox_measurement_text(width, length)), (left, top), (0, 255, 255))

        status = f"{len(boxes)} boxes | ENTER/SPACE done | c cancel | r reset"
        cv2.putText(preview, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(preview, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

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
        lines.append(f"{frame_number},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{confidence:.4f},-1,-1,-1\n")


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


def draw_proposals(
    frame: np.ndarray,
    proposals: Sequence[ObjectProposal],
    active_proposal_id: Optional[int] = None,
    max_draw: int = 20,
) -> np.ndarray:
    output = frame.copy()
    for proposal in list(proposals)[: max(0, int(max_draw))]:
        x, y, w, h = [int(round(value)) for value in proposal.bbox]
        active = proposal.proposal_id == active_proposal_id
        color = (255, 255, 0) if active else (255, 180, 0)
        thickness = 3 if active else 1
        label = f"P{proposal.proposal_id} {proposal.score:.2f}"
        if active:
            label += " current"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    return output


def _safe_source_name(source_name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)


def default_output_path(source_name: str, backend: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{_safe_source_name(source_name)}_{backend}_tracks.txt"


def default_debug_log_path(source_name: str, backend: str) -> Path:
    return DEFAULT_DEBUG_DIR / f"{_safe_source_name(source_name)}_{backend}_debug.csv"


def default_slot_debug_log_path(source_name: str, backend: str) -> Path:
    return DEFAULT_DEBUG_DIR / f"{_safe_source_name(source_name)}_{backend}_slot_debug.csv"


def default_video_path(source_name: str, backend: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{_safe_source_name(source_name)}_{backend}_annotated.mp4"


def make_video_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open annotated video writer: {path}")
    return writer
