from __future__ import annotations

import argparse
import copy
import os
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lorat-gui"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
DEFAULT_LORAT_ROOT = PROJECT_ROOT / "external" / "LoRAT-main"
DEFAULT_DANCETRACK_SEQUENCE = PROJECT_ROOT / "data" / "raw" / "DanceTrack" / "val" / "val" / "dancetrack0065"

LORAT_WEIGHT_BY_CONFIG = {
    "B-224": PROJECT_ROOT / "models" / "lorat" / "base.bin",
    "B-378": PROJECT_ROOT / "models" / "lorat" / "base-378.bin",
    "L-224": PROJECT_ROOT / "models" / "lorat" / "large.bin",
    "L-378": PROJECT_ROOT / "models" / "lorat" / "large-378.bin",
    "g-224": PROJECT_ROOT / "models" / "lorat" / "giant.bin",
    "g-378": PROJECT_ROOT / "models" / "lorat" / "giant-378.bin",
}


@dataclass
class TrackVisualMemory:
    frame_number: int
    image_rgb: np.ndarray
    bbox: BBox
    confidence: Optional[float]
    state: str
    initial: bool = False
    appearance_hist: Optional[np.ndarray] = None


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    color: Color
    ok: bool = True
    confidence: Optional[float] = None
    lost_frames: int = 0
    tracker: Optional[object] = None
    previous_bbox: Optional[BBox] = None
    predicted_bbox: Optional[BBox] = None
    raw_bbox: Optional[BBox] = None
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)
    trajectory: List[Tuple[int, BBox]] = field(default_factory=list)
    occluded_frames: int = 0
    coordinator_state: str = ""
    appearance_hist: Optional[np.ndarray] = None
    initial_appearance_hist: Optional[np.ndarray] = None
    appearance_bank: List[np.ndarray] = field(default_factory=list)
    appearance_updates: int = 0
    initial_visual_memory: Optional[TrackVisualMemory] = None
    visual_memory: List[TrackVisualMemory] = field(default_factory=list)
    active_template_frame: Optional[int] = None
    assignment_score: Optional[float] = None
    assignment_margin: Optional[float] = None
    reid_score: Optional[float] = None
    motion_score: Optional[float] = None
    trajectory_score: Optional[float] = None
    memory_score: Optional[float] = None
    memory_penalty: Optional[float] = None
    direction_score: Optional[float] = None
    bottom_score: Optional[float] = None
    iou_score: Optional[float] = None
    assigned_source: str = ""
    last_resync_frame: int = -1
    resync_count: int = 0


@dataclass
class AssociationCandidate:
    bbox: BBox
    confidence: Optional[float]
    source: str
    source_track_id: Optional[int] = None
    appearance_hist: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AssociationScore:
    total: float
    appearance: float
    motion: float
    trajectory: float
    memory: float
    direction: float
    bottom: float
    iou: float
    source: float
    competition_penalty: float
    memory_penalty: float
    switch_penalty: float
    gate_penalty: float
    trajectory_penalty: float
    direction_penalty: float
    bottom_penalty: float


@dataclass
class ResolvedTrackUpdate:
    track: TrackState
    bbox: BBox
    confidence: Optional[float]
    state: str
    assigned_source: str
    previous_bbox: BBox
    predicted_bbox: BBox
    raw_bbox: BBox
    in_conflict: bool
    suspicious: bool
    ambiguous: bool
    trajectory_guarded: bool
    direction_guarded: bool
    bottom_guarded: bool
    center_guarded: bool
    memory_guarded: bool
    extent_guarded: bool
    assignment_score: float
    assignment_margin: float
    score_parts: AssociationScore
    overlap_guarded: bool = False


@dataclass
class PairOcclusionMemory:
    frame_number: int
    center_gap_x: float
    center_gap_y: float
    bottom_gap: float


class MultiObjectCoordinator:
    def __init__(
        self,
        enabled: bool = True,
        overlap_iou_threshold: float = 0.30,
        normal_proposal_weight: float = 0.90,
        occluded_proposal_weight: float = 0.55,
        suspicious_proposal_weight: float = 0.35,
        max_center_jump: float = 0.45,
        max_scale_change: float = 0.65,
        max_occlusion_frames: int = 20,
        velocity_smoothing: float = 0.70,
        detector_method: str = "none",
        detector_interval: int = 5,
        max_detections: int = 12,
        association_min_score: float = 0.15,
        reid_weight: float = 0.45,
        motion_weight: float = 0.30,
        iou_weight: float = 0.15,
        source_weight: float = 0.10,
        appearance_update_rate: float = 0.08,
        appearance_bank_size: int = 8,
        appearance_commit_min_score: float = 0.48,
        appearance_commit_min_margin: float = 0.04,
        appearance_commit_min_similarity: float = 0.35,
        ambiguous_assignment_margin: float = 0.04,
        max_track_overlap_iou: float = 0.70,
        pair_memory_overlap_iou: float = 0.35,
        pair_memory_min_gap: float = 6.0,
        pair_memory_max_age: int = 45,
        pair_memory_strength: float = 0.75,
        source_switch_penalty: float = 0.08,
        reid_gate: float = 0.28,
        reid_competition_weight: float = 0.20,
        reid_competition_margin: float = 0.04,
        trajectory_history_size: int = 12,
        trajectory_weight: float = 0.30,
        trajectory_guard_min_score: float = 0.25,
        trajectory_guard_proposal_weight: float = 0.50,
        trajectory_penalty_weight: float = 0.20,
        recent_memory_frames: int = 10,
        recent_memory_max_shift_fraction: float = 0.25,
        recent_memory_min_shift_px: float = 4.0,
        recent_memory_weight: float = 0.12,
        recent_memory_competition_weight: float = 0.25,
        identity_reject_memory_score: float = 0.12,
        identity_reject_min_score: float = 0.38,
        identity_reject_motion_score: float = 0.55,
        identity_reject_competition_penalty: float = 0.05,
        guarded_max_scale_change: float = 0.25,
        direction_guard_min_score: float = 0.35,
        direction_penalty_weight: float = 0.30,
        bottom_guard_min_score: float = 0.40,
        bottom_penalty_weight: float = 0.25,
        center_anchor_max_shift_fraction: float = 0.25,
        center_anchor_min_shift_px: float = 4.0,
        extent_anchor_max_upward_fraction: float = 0.12,
        extent_anchor_min_scale: float = 0.80,
        memory_recovery_enabled: bool = True,
        memory_recovery_min_score: float = 0.58,
        memory_recovery_search_radius: float = 0.90,
        memory_recovery_scale_step: float = 0.15,
    ):
        self.enabled = enabled
        self.overlap_iou_threshold = overlap_iou_threshold
        self.normal_proposal_weight = normal_proposal_weight
        self.occluded_proposal_weight = occluded_proposal_weight
        self.suspicious_proposal_weight = suspicious_proposal_weight
        self.max_center_jump = max_center_jump
        self.max_scale_change = max_scale_change
        self.max_occlusion_frames = max_occlusion_frames
        self.velocity_smoothing = velocity_smoothing
        self.detector_method = detector_method
        self.detector_interval = max(1, detector_interval)
        self.max_detections = max(0, max_detections)
        self.association_min_score = association_min_score
        self.reid_weight = reid_weight
        self.motion_weight = motion_weight
        self.iou_weight = iou_weight
        self.source_weight = source_weight
        self.appearance_update_rate = appearance_update_rate
        self.appearance_bank_size = max(1, appearance_bank_size)
        self.appearance_commit_min_score = appearance_commit_min_score
        self.appearance_commit_min_margin = appearance_commit_min_margin
        self.appearance_commit_min_similarity = appearance_commit_min_similarity
        self.ambiguous_assignment_margin = ambiguous_assignment_margin
        self.max_track_overlap_iou = max(0.0, min(1.0, max_track_overlap_iou))
        self.pair_memory_overlap_iou = max(0.0, min(1.0, pair_memory_overlap_iou))
        self.pair_memory_min_gap = max(0.0, pair_memory_min_gap)
        self.pair_memory_max_age = max(1, pair_memory_max_age)
        self.pair_memory_strength = max(0.0, min(1.0, pair_memory_strength))
        self.source_switch_penalty = source_switch_penalty
        self.reid_gate = reid_gate
        self.reid_competition_weight = reid_competition_weight
        self.reid_competition_margin = reid_competition_margin
        self.recent_memory_frames = max(3, recent_memory_frames)
        self.recent_memory_max_shift_fraction = max(0.0, recent_memory_max_shift_fraction)
        self.recent_memory_min_shift_px = max(0.0, recent_memory_min_shift_px)
        self.recent_memory_weight = max(0.0, recent_memory_weight)
        self.recent_memory_competition_weight = max(0.0, recent_memory_competition_weight)
        self.identity_reject_memory_score = max(0.0, identity_reject_memory_score)
        self.identity_reject_min_score = max(0.0, identity_reject_min_score)
        self.identity_reject_motion_score = max(0.0, identity_reject_motion_score)
        self.identity_reject_competition_penalty = max(0.0, identity_reject_competition_penalty)
        self.trajectory_history_size = max(2, trajectory_history_size, self.recent_memory_frames)
        self.trajectory_weight = max(0.0, min(1.0, trajectory_weight))
        self.trajectory_guard_min_score = trajectory_guard_min_score
        self.trajectory_guard_proposal_weight = max(0.0, min(1.0, trajectory_guard_proposal_weight))
        self.trajectory_penalty_weight = max(0.0, trajectory_penalty_weight)
        self.guarded_max_scale_change = max(0.0, guarded_max_scale_change)
        self.direction_guard_min_score = direction_guard_min_score
        self.direction_penalty_weight = max(0.0, direction_penalty_weight)
        self.bottom_guard_min_score = bottom_guard_min_score
        self.bottom_penalty_weight = max(0.0, bottom_penalty_weight)
        self.center_anchor_max_shift_fraction = max(0.0, center_anchor_max_shift_fraction)
        self.center_anchor_min_shift_px = max(0.0, center_anchor_min_shift_px)
        self.extent_anchor_max_upward_fraction = max(0.0, extent_anchor_max_upward_fraction)
        self.extent_anchor_min_scale = max(0.05, min(1.0, extent_anchor_min_scale))
        self.memory_recovery_enabled = memory_recovery_enabled
        self.memory_recovery_min_score = max(0.0, min(1.0, memory_recovery_min_score))
        self.memory_recovery_search_radius = max(0.0, memory_recovery_search_radius)
        self.memory_recovery_scale_step = max(0.0, memory_recovery_scale_step)
        self.hog = None
        if detector_method == "hog":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.pair_memories: Dict[Tuple[int, int], PairOcclusionMemory] = {}
        self.rejected_track_states: Dict[int, str] = {}
        self.rejected_track_scores: Dict[int, Tuple[float, float, AssociationScore, str]] = {}

    def initialize_tracks(self, frame: np.ndarray, tracks: Sequence[TrackState]) -> None:
        for track in tracks:
            track.initial_appearance_hist = extract_reid_histogram(frame, track.bbox)
            track.appearance_hist = track.initial_appearance_hist.copy() if track.initial_appearance_hist is not None else None
            track.appearance_bank = []
            if track.initial_appearance_hist is not None:
                track.appearance_bank.append(track.initial_appearance_hist.copy())
            track.appearance_updates = 1 if track.appearance_hist is not None else 0

    def resolve(
        self,
        tracks: Sequence[TrackState],
        proposals: Dict[int, Tuple[BBox, Optional[float]]],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> Dict[int, Tuple[BBox, Optional[float], str]]:
        if not self.enabled:
            return {
                track_id: (proposal, confidence, "")
                for track_id, (proposal, confidence) in proposals.items()
            }

        self.rejected_track_states = {}
        self.rejected_track_scores = {}
        active_tracks = [
            track
            for track in tracks
            if track.ok or track.track_id in proposals or track.lost_frames <= self.max_occlusion_frames
        ]
        candidates = self._build_candidates(active_tracks, proposals, frame, frame_number)
        if not candidates:
            return {}

        candidate_boxes = {index: candidate.bbox for index, candidate in enumerate(candidates)}
        conflict_candidate_ids = self._find_candidate_conflicts(candidate_boxes)
        assignments = self._assign_tracks(active_tracks, candidates, frame, frame_number)
        pending_updates: List[ResolvedTrackUpdate] = []

        for track, candidate_index, assignment_score, assignment_margin, score_parts in assignments:
            candidate = candidates[candidate_index]
            proposal = candidate.bbox
            confidence = candidate.confidence
            previous = track.bbox
            predicted = predict_track_bbox(track, frame_number, self.trajectory_weight)
            suspicious = self._is_suspicious(previous, predicted, proposal)
            in_conflict = candidate_index in conflict_candidate_ids
            ambiguous = assignment_margin < self.ambiguous_assignment_margin
            trajectory_guarded = (
                track_has_trajectory(track)
                and score_parts.trajectory < self.trajectory_guard_min_score
            )
            direction_guarded = (
                track_has_trajectory(track)
                and score_parts.direction < self.direction_guard_min_score
            )
            bottom_guarded = (
                track_has_trajectory(track)
                and score_parts.bottom < self.bottom_guard_min_score
            )
            memory_guarded = (
                track_has_trajectory(track)
                and score_parts.memory < 0.35
                and (
                    in_conflict
                    or suspicious
                    or ambiguous
                    or trajectory_guarded
                    or direction_guarded
                    or bottom_guarded
                    or score_parts.memory_penalty > 0.0
                )
            )
            source_switch = (
                candidate.source == "lorat"
                and candidate.source_track_id is not None
                and candidate.source_track_id != track.track_id
            )
            weak_reid_switch = source_switch and score_parts.appearance < self.appearance_commit_min_similarity

            if self._should_reject_identity_break(
                track,
                candidate,
                all_tracks=active_tracks,
                predicted=predicted,
                score_parts=score_parts,
                assignment_score=assignment_score,
                in_conflict=in_conflict,
                suspicious=suspicious,
                ambiguous=ambiguous,
                trajectory_guarded=trajectory_guarded,
                direction_guarded=direction_guarded,
                bottom_guarded=bottom_guarded,
                memory_guarded=memory_guarded,
                frame_number=frame_number,
            ):
                reject_state = self._identity_reject_state(
                    in_conflict,
                    suspicious,
                    ambiguous,
                    trajectory_guarded,
                    direction_guarded,
                    bottom_guarded,
                    memory_guarded,
                    score_parts,
                )
                self.rejected_track_states[track.track_id] = reject_state
                self.rejected_track_scores[track.track_id] = (
                    assignment_score,
                    assignment_margin,
                    score_parts,
                    candidate.source,
                )
                continue

            if in_conflict:
                proposal_weight = self.occluded_proposal_weight
                state = "OCC"
            else:
                proposal_weight = self.normal_proposal_weight
                state = "REID"

            if candidate.source == "detector":
                state = "DET" if not in_conflict else "DET_OCC"
            elif candidate.source == "memory":
                state = "MEMREC" if not in_conflict else "MEMREC_OCC"
            elif candidate.source_track_id == track.track_id:
                state = "" if not in_conflict else state

            if suspicious:
                proposal_weight = min(proposal_weight, self.suspicious_proposal_weight)
                state = "SMOOTH" if not state else state

            if ambiguous:
                proposal_weight = min(
                    proposal_weight,
                    self.suspicious_proposal_weight if in_conflict else self.occluded_proposal_weight,
                )
                state = "AMBIG"

            if trajectory_guarded:
                proposal_weight = min(proposal_weight, self.trajectory_guard_proposal_weight)
                state = append_state_token(state, "PATH")

            if direction_guarded:
                proposal_weight = min(proposal_weight, self.trajectory_guard_proposal_weight)
                state = append_state_token(state, "DIR")

            if bottom_guarded:
                proposal_weight = min(proposal_weight, self.trajectory_guard_proposal_weight)
                state = append_state_token(state, "BOTTOM")

            if memory_guarded:
                proposal_weight = min(proposal_weight, self.trajectory_guard_proposal_weight)
                state = append_state_token(state, "MEM")

            if weak_reid_switch:
                proposal_weight = min(proposal_weight, self.suspicious_proposal_weight)
                state = append_state_token(state, "REID?")

            if track.occluded_frames >= self.max_occlusion_frames:
                proposal_weight = max(proposal_weight, self.normal_proposal_weight)
                state = ""

            final_bbox = clamp_bbox_size(blend_bbox(predicted, proposal, proposal_weight))
            if (
                suspicious
                or in_conflict
                or ambiguous
                or trajectory_guarded
                or direction_guarded
                or bottom_guarded
                or weak_reid_switch
            ):
                final_bbox = clamp_bbox_scale_change(previous, final_bbox, self.guarded_max_scale_change)
            center_guarded = self._should_center_anchor_guard(
                predicted,
                proposal,
                final_bbox,
                score_parts,
                in_conflict,
                suspicious,
                ambiguous,
                trajectory_guarded,
                direction_guarded,
                bottom_guarded,
            )
            if center_guarded:
                final_bbox = self._apply_center_anchor_guard(predicted, final_bbox, frame)
                state = append_state_token(state, "CENTER")
            extent_guarded = self._should_extent_anchor_guard(
                previous,
                predicted,
                proposal,
                score_parts,
                in_conflict,
                suspicious,
                trajectory_guarded,
                direction_guarded,
                bottom_guarded,
                center_guarded,
                memory_guarded,
            )
            if extent_guarded:
                final_bbox = self._apply_extent_anchor_guard(previous, predicted, final_bbox, frame)
                state = append_state_token(state, "EXTENT")
            size_guarded_bbox = clamp_bbox_to_trusted_size_floor(track, final_bbox, frame)
            if size_guarded_bbox != final_bbox:
                final_bbox = size_guarded_bbox
                state = append_state_token(state, "SIZE")
            pending_updates.append(
                ResolvedTrackUpdate(
                    track=track,
                    bbox=final_bbox,
                    confidence=confidence,
                    state=state,
                    assigned_source=candidate.source,
                    previous_bbox=previous,
                    predicted_bbox=predicted,
                    raw_bbox=proposal,
                    in_conflict=in_conflict,
                    suspicious=suspicious,
                    ambiguous=ambiguous,
                    trajectory_guarded=trajectory_guarded,
                    direction_guarded=direction_guarded,
                    bottom_guarded=bottom_guarded,
                    center_guarded=center_guarded,
                    memory_guarded=memory_guarded,
                    extent_guarded=extent_guarded,
                    assignment_score=assignment_score,
                    assignment_margin=assignment_margin,
                    score_parts=score_parts,
                )
            )

        self._refresh_pair_order_memories(pending_updates, frame_number)
        self._apply_pair_order_memory(pending_updates, frame, frame_number)
        self._apply_track_overlap_guard(pending_updates, frame)

        updates: Dict[int, Tuple[BBox, Optional[float], str]] = {}
        for update in pending_updates:
            track = update.track
            size_guarded_bbox = clamp_bbox_to_trusted_size_floor(track, update.bbox, frame)
            if size_guarded_bbox != update.bbox:
                update.bbox = size_guarded_bbox
                update.state = append_state_token(update.state, "SIZE")
            self._update_track_motion(
                track,
                update.previous_bbox,
                update.bbox,
                update.predicted_bbox,
                update.raw_bbox,
                update.in_conflict,
                update.state,
                frame_number,
            )
            track.assignment_score = update.assignment_score
            track.assignment_margin = update.assignment_margin
            track.reid_score = update.score_parts.appearance
            track.motion_score = update.score_parts.motion
            track.trajectory_score = update.score_parts.trajectory
            track.memory_score = update.score_parts.memory
            track.memory_penalty = update.score_parts.memory_penalty
            track.direction_score = update.score_parts.direction
            track.bottom_score = update.score_parts.bottom
            track.iou_score = update.score_parts.iou
            track.assigned_source = "guarded" if update.overlap_guarded else update.assigned_source
            if self._should_update_appearance(
                track,
                frame,
                update.bbox,
                update.in_conflict,
                update.suspicious,
                update.ambiguous,
                update.trajectory_guarded,
                update.direction_guarded,
                update.bottom_guarded,
                update.center_guarded,
                update.memory_guarded,
                update.extent_guarded,
                update.assignment_score,
                update.assignment_margin,
                update.score_parts.appearance,
                update.overlap_guarded,
            ):
                self._update_appearance(track, frame, update.bbox)
            updates[track.track_id] = (update.bbox, update.confidence, update.state)

        return updates

    def _build_candidates(
        self,
        tracks: Sequence[TrackState],
        proposals: Dict[int, Tuple[BBox, Optional[float]]],
        frame: Optional[np.ndarray],
        frame_number: int,
    ) -> List[AssociationCandidate]:
        candidates = [
            AssociationCandidate(bbox=bbox, confidence=confidence, source="lorat", source_track_id=track_id)
            for track_id, (bbox, confidence) in proposals.items()
        ]

        if frame is not None and self.memory_recovery_enabled:
            candidates.extend(self._build_memory_recovery_candidates(tracks, proposals, frame, frame_number))

        if frame is not None and self.detector_method != "none" and frame_number % self.detector_interval == 0:
            detections = self._detect_people(frame)
            candidates.extend(
                AssociationCandidate(bbox=bbox, confidence=confidence, source="detector")
                for bbox, confidence in detections
            )

        for candidate in candidates:
            if frame is not None:
                candidate.appearance_hist = extract_reid_histogram(frame, candidate.bbox)
        return candidates

    def _build_memory_recovery_candidates(
        self,
        tracks: Sequence[TrackState],
        proposals: Dict[int, Tuple[BBox, Optional[float]]],
        frame: np.ndarray,
        frame_number: int,
    ) -> List[AssociationCandidate]:
        candidates: List[AssociationCandidate] = []
        for track in tracks:
            if not track_has_appearance(track):
                continue

            predicted = predict_track_bbox(
                track,
                frame_number,
                self.trajectory_weight,
                preserve_size=True,
            )
            predicted = clamp_bbox_to_trusted_size_floor(track, predicted, frame)
            proposal_tuple = proposals.get(track.track_id)
            proposal_bbox = proposal_tuple[0] if proposal_tuple is not None else None
            proposal_confidence = proposal_tuple[1] if proposal_tuple is not None else None
            if not self._should_try_memory_recovery(
                track,
                predicted,
                proposal_bbox,
                proposal_confidence,
                tracks,
                frame_number,
            ):
                continue

            candidate = self._best_memory_recovery_candidate(track, predicted, frame)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _should_try_memory_recovery(
        self,
        track: TrackState,
        predicted: BBox,
        proposal: Optional[BBox],
        confidence: Optional[float],
        all_tracks: Sequence[TrackState],
        frame_number: int,
    ) -> bool:
        if track.lost_frames > 0 or proposal is None:
            return True
        if confidence is not None and confidence < 0.35:
            return True
        if self._is_suspicious(track.bbox, predicted, proposal):
            return True
        if center_distance(predicted, proposal) > self._center_anchor_max_shift(predicted):
            return True
        if self._candidate_near_other_track(track, proposal, all_tracks, frame_number):
            return True

        recent_memory_predicted = predict_recent_memory_bbox(track, frame_number, self.recent_memory_frames)
        if recent_memory_predicted is None:
            return False
        memory_score = recent_memory_affinity(
            recent_memory_predicted,
            proposal,
            self.recent_memory_max_shift_fraction,
            self.recent_memory_min_shift_px,
        )
        return memory_score < 0.25

    def _best_memory_recovery_candidate(
        self,
        track: TrackState,
        predicted: BBox,
        frame: np.ndarray,
    ) -> Optional[AssociationCandidate]:
        search_boxes = memory_recovery_search_boxes(
            predicted,
            frame,
            self.memory_recovery_search_radius,
            self.memory_recovery_scale_step,
        )
        if not search_boxes:
            return None

        reference_diag = max(1.0, bbox_diagonal(predicted))
        best: Optional[Tuple[float, BBox, np.ndarray, float]] = None
        for bbox in search_boxes:
            hist = extract_reid_histogram(frame, bbox)
            if hist is None:
                continue
            appearance_score = track_appearance_similarity(track, hist)
            if appearance_score < self.memory_recovery_min_score:
                continue
            motion_score = motion_affinity(predicted, bbox, reference_diag)
            bottom_score = bottom_anchor_affinity(predicted, bbox)
            score = (0.70 * appearance_score) + (0.20 * motion_score) + (0.10 * bottom_score)
            if best is None or score > best[0]:
                best = (score, bbox, hist, appearance_score)

        if best is None:
            return None
        score, bbox, hist, appearance_score = best
        return AssociationCandidate(
            bbox=bbox,
            confidence=float(score),
            source="memory",
            source_track_id=track.track_id,
            appearance_hist=hist,
        )

    def _detect_people(self, frame: np.ndarray) -> List[Tuple[BBox, Optional[float]]]:
        if self.hog is None or self.max_detections <= 0:
            return []

        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections = []
        for box, weight in zip(boxes, weights):
            x, y, w, h = [float(value) for value in box]
            if w <= 2 or h <= 2:
                continue
            detections.append(((x, y, w, h), float(weight)))
        detections.sort(key=lambda item: item[1] if item[1] is not None else 0.0, reverse=True)
        return nms_detections(detections, 0.45)[: self.max_detections]

    def _assign_tracks(
        self,
        tracks: Sequence[TrackState],
        candidates: Sequence[AssociationCandidate],
        frame: Optional[np.ndarray],
        frame_number: int,
    ) -> List[Tuple[TrackState, int, float, float, AssociationScore]]:
        score_details = [
            [self._association_score(track, candidate, tracks, frame, frame_number) for candidate in candidates]
            for track in tracks
        ]
        score_matrix = [[score.total for score in row] for row in score_details]
        assignments = solve_assignment(score_matrix, self.association_min_score)
        return [
            (
                tracks[row],
                col,
                score,
                assignment_margin(score_matrix[row], col),
                score_details[row][col],
            )
            for row, col, score in assignments
        ]

    def _association_score(
        self,
        track: TrackState,
        candidate: AssociationCandidate,
        all_tracks: Sequence[TrackState],
        frame: Optional[np.ndarray],
        frame_number: int,
    ) -> AssociationScore:
        if candidate.source in {"lorat", "memory"} and candidate.source_track_id is not None and candidate.source_track_id != track.track_id:
            return AssociationScore(
                total=0.0,
                appearance=0.0,
                motion=0.0,
                trajectory=0.0,
                memory=0.0,
                direction=0.0,
                bottom=0.0,
                iou=0.0,
                source=0.0,
                competition_penalty=0.0,
                memory_penalty=0.0,
                switch_penalty=1.0,
                gate_penalty=0.0,
                trajectory_penalty=0.0,
                direction_penalty=0.0,
                bottom_penalty=0.0,
            )

        instant_predicted = predict_bbox(track.bbox, track.velocity)
        trajectory_predicted = predict_trajectory_bbox(track, frame_number)
        predicted = (
            blend_bbox(instant_predicted, trajectory_predicted, self.trajectory_weight)
            if trajectory_predicted is not None
            else instant_predicted
        )
        diag = max(1.0, bbox_diagonal(track.bbox))
        instant_score = motion_affinity(instant_predicted, candidate.bbox, diag)
        trajectory_motion_score = (
            motion_affinity(trajectory_predicted, candidate.bbox, diag)
            if trajectory_predicted is not None
            else instant_score
        )
        trajectory_score = (
            center_affinity(trajectory_predicted, candidate.bbox, diag)
            if trajectory_predicted is not None
            else center_affinity(instant_predicted, candidate.bbox, diag)
        )
        recent_memory_predicted = predict_recent_memory_bbox(track, frame_number, self.recent_memory_frames)
        memory_score = (
            recent_memory_affinity(
                recent_memory_predicted,
                candidate.bbox,
                self.recent_memory_max_shift_fraction,
                self.recent_memory_min_shift_px,
            )
            if recent_memory_predicted is not None
            else 0.50
        )
        direction_score = direction_consistency_score(track, candidate.bbox)
        bottom_score = bottom_anchor_affinity(predicted, candidate.bbox)
        motion_score = (
            ((1.0 - self.trajectory_weight) * instant_score)
            + (self.trajectory_weight * trajectory_motion_score)
        )
        iou_score = max(bbox_iou(predicted, candidate.bbox), bbox_iou(track.bbox, candidate.bbox))

        appearance_score = 0.50
        if candidate.appearance_hist is not None:
            appearance_score = track_appearance_similarity(track, candidate.appearance_hist)
        elif frame is not None and (track.appearance_hist is not None or track.initial_appearance_hist is not None):
            candidate_hist = extract_reid_histogram(frame, candidate.bbox)
            if candidate_hist is not None:
                appearance_score = track_appearance_similarity(track, candidate_hist)

        source_score = 0.55
        switch_penalty = 0.0
        if candidate.source == "lorat" and candidate.source_track_id == track.track_id:
            source_score = 1.0
        elif candidate.source == "memory" and candidate.source_track_id == track.track_id:
            source_score = 0.82
        elif candidate.source == "lorat" and candidate.source_track_id is not None:
            source_score = 0.20
            switch_penalty = self.source_switch_penalty
        elif candidate.source == "detector":
            source_score = 0.70

        competition_penalty = self._reid_competition_penalty(track, candidate, all_tracks, appearance_score)
        memory_penalty = self._recent_memory_competition_penalty(
            track,
            candidate,
            all_tracks,
            frame_number,
            memory_score,
        )
        gate_penalty = 0.0
        if self._needs_reid_gate(track, candidate) and appearance_score < self.reid_gate:
            gate_penalty = 0.10 + ((self.reid_gate - appearance_score) * 0.75)
        trajectory_penalty = 0.0
        if track_has_trajectory(track) and trajectory_score < self.trajectory_guard_min_score:
            trajectory_penalty = (
                (self.trajectory_guard_min_score - trajectory_score)
                * self.trajectory_penalty_weight
            )
        direction_penalty = 0.0
        if track_has_trajectory(track) and direction_score < self.direction_guard_min_score:
            direction_penalty = (
                (self.direction_guard_min_score - direction_score)
                * self.direction_penalty_weight
            )
        bottom_penalty = 0.0
        if track_has_trajectory(track) and bottom_score < self.bottom_guard_min_score:
            bottom_penalty = (
                (self.bottom_guard_min_score - bottom_score)
                * self.bottom_penalty_weight
            )

        total = (
            (self.reid_weight * appearance_score)
            + (self.motion_weight * motion_score)
            + (self.iou_weight * iou_score)
            + (self.source_weight * source_score)
            + (self.recent_memory_weight * memory_score)
            - competition_penalty
            - memory_penalty
            - switch_penalty
            - gate_penalty
            - trajectory_penalty
            - direction_penalty
            - bottom_penalty
        )
        return AssociationScore(
            total=max(0.0, float(total)),
            appearance=appearance_score,
            motion=motion_score,
            trajectory=trajectory_score,
            memory=memory_score,
            direction=direction_score,
            bottom=bottom_score,
            iou=iou_score,
            source=source_score,
            competition_penalty=competition_penalty,
            memory_penalty=memory_penalty,
            switch_penalty=switch_penalty,
            gate_penalty=gate_penalty,
            trajectory_penalty=trajectory_penalty,
            direction_penalty=direction_penalty,
            bottom_penalty=bottom_penalty,
        )

    def _reid_competition_penalty(
        self,
        track: TrackState,
        candidate: AssociationCandidate,
        all_tracks: Sequence[TrackState],
        appearance_score: float,
    ) -> float:
        if (
            candidate.appearance_hist is None
            or self.reid_competition_weight <= 0
            or not track_has_appearance(track)
        ):
            return 0.0

        best_other_score = 0.0
        for other in all_tracks:
            if other.track_id == track.track_id or not track_has_appearance(other):
                continue
            best_other_score = max(
                best_other_score,
                track_appearance_similarity(other, candidate.appearance_hist),
            )

        advantage = best_other_score - appearance_score - self.reid_competition_margin
        if advantage <= 0:
            return 0.0
        return min(0.35, advantage * self.reid_competition_weight)

    def _recent_memory_competition_penalty(
        self,
        track: TrackState,
        candidate: AssociationCandidate,
        all_tracks: Sequence[TrackState],
        frame_number: int,
        memory_score: float,
    ) -> float:
        if self.recent_memory_competition_weight <= 0.0:
            return 0.0

        best_other_score = 0.0
        for other in all_tracks:
            if other.track_id == track.track_id or not track_has_trajectory(other):
                continue
            other_memory = predict_recent_memory_bbox(other, frame_number, self.recent_memory_frames)
            if other_memory is None:
                continue
            best_other_score = max(
                best_other_score,
                recent_memory_affinity(
                    other_memory,
                    candidate.bbox,
                    self.recent_memory_max_shift_fraction,
                    self.recent_memory_min_shift_px,
                ),
            )

        advantage = best_other_score - memory_score - 0.08
        if advantage <= 0.0:
            return 0.0
        return min(0.35, advantage * self.recent_memory_competition_weight)

    def _should_reject_identity_break(
        self,
        track: TrackState,
        candidate: AssociationCandidate,
        all_tracks: Sequence[TrackState],
        predicted: BBox,
        score_parts: AssociationScore,
        assignment_score: float,
        in_conflict: bool,
        suspicious: bool,
        ambiguous: bool,
        trajectory_guarded: bool,
        direction_guarded: bool,
        bottom_guarded: bool,
        memory_guarded: bool,
        frame_number: int,
    ) -> bool:
        if (
            candidate.source != "lorat"
            or candidate.source_track_id != track.track_id
            or not track_has_trajectory(track)
        ):
            return False

        near_other_track = self._candidate_near_other_track(
            track,
            candidate.bbox,
            all_tracks,
            frame_number,
        )
        contested = (
            in_conflict
            or near_other_track
            or score_parts.memory_penalty >= self.identity_reject_competition_penalty
        )
        if not contested:
            return False

        weak_memory = score_parts.memory <= self.identity_reject_memory_score
        weak_assignment = assignment_score < self.identity_reject_min_score
        weak_motion = score_parts.motion < self.identity_reject_motion_score
        weak_path = trajectory_guarded or score_parts.trajectory < self.trajectory_guard_min_score
        weak_direction = direction_guarded or score_parts.direction < self.direction_guard_min_score
        jump_too_far = center_distance(predicted, candidate.bbox) > self._center_anchor_max_shift(predicted)
        competing_memory = score_parts.memory_penalty >= self.identity_reject_competition_penalty

        identity_disagreement = (
            weak_assignment
            or (weak_motion and (weak_path or weak_direction or memory_guarded))
            or (competing_memory and (weak_path or weak_motion))
            or suspicious
            or ambiguous
            or (jump_too_far and weak_path)
        )
        return weak_memory and identity_disagreement

    def _candidate_near_other_track(
        self,
        track: TrackState,
        candidate_bbox: BBox,
        all_tracks: Sequence[TrackState],
        frame_number: int,
    ) -> bool:
        candidate_diag = bbox_diagonal(candidate_bbox)
        near_distance = max(6.0, candidate_diag * 0.85)
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            if not other.ok and other.lost_frames > self.max_occlusion_frames:
                continue

            other_predicted = predict_track_bbox(other, frame_number, self.trajectory_weight)
            if bbox_iou(candidate_bbox, other.bbox) >= self.overlap_iou_threshold:
                return True
            if bbox_iou(candidate_bbox, other_predicted) >= self.overlap_iou_threshold:
                return True
            if center_distance(candidate_bbox, other_predicted) <= near_distance:
                return True
        return False

    def _identity_reject_state(
        self,
        in_conflict: bool,
        suspicious: bool,
        ambiguous: bool,
        trajectory_guarded: bool,
        direction_guarded: bool,
        bottom_guarded: bool,
        memory_guarded: bool,
        score_parts: AssociationScore,
    ) -> str:
        state = "REJECT"
        if in_conflict:
            state = append_state_token(state, "OCC")
        if suspicious:
            state = append_state_token(state, "SMOOTH")
        if ambiguous:
            state = append_state_token(state, "AMBIG")
        if trajectory_guarded:
            state = append_state_token(state, "PATH")
        if direction_guarded:
            state = append_state_token(state, "DIR")
        if bottom_guarded:
            state = append_state_token(state, "BOTTOM")
        if memory_guarded or score_parts.memory <= self.identity_reject_memory_score:
            state = append_state_token(state, "MEM")
        if score_parts.memory_penalty >= self.identity_reject_competition_penalty:
            state = append_state_token(state, "COMP")
        return append_state_token(state, "COAST")

    def _needs_reid_gate(self, track: TrackState, candidate: AssociationCandidate) -> bool:
        if not track_has_appearance(track):
            return False
        if candidate.source == "lorat" and candidate.source_track_id == track.track_id:
            return False
        return True

    def _should_center_anchor_guard(
        self,
        predicted: BBox,
        proposal: BBox,
        final_bbox: BBox,
        score_parts: AssociationScore,
        in_conflict: bool,
        suspicious: bool,
        ambiguous: bool,
        trajectory_guarded: bool,
        direction_guarded: bool,
        bottom_guarded: bool,
    ) -> bool:
        if self.center_anchor_max_shift_fraction <= 0.0:
            return False

        max_shift = self._center_anchor_max_shift(predicted)
        proposal_shift = center_distance(predicted, proposal)
        final_shift = center_distance(predicted, final_bbox)
        risky_context = (
            in_conflict
            or suspicious
            or ambiguous
            or trajectory_guarded
            or direction_guarded
            or bottom_guarded
            or score_parts.trajectory < self.trajectory_guard_min_score
            or score_parts.motion < 0.45
        )
        return risky_context and proposal_shift > max_shift and final_shift > max_shift

    def _apply_center_anchor_guard(
        self,
        predicted: BBox,
        bbox: BBox,
        frame: Optional[np.ndarray],
    ) -> BBox:
        predicted_center = bbox_center(predicted)
        bbox_center_point = bbox_center(bbox)
        delta = np.array(
            [
                bbox_center_point[0] - predicted_center[0],
                bbox_center_point[1] - predicted_center[1],
            ],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(delta))
        max_shift = self._center_anchor_max_shift(predicted)
        if distance <= max_shift or distance <= 0.0:
            return bbox

        clamped_delta = delta * (max_shift / distance)
        clamped_center = (
            predicted_center[0] + float(clamped_delta[0]),
            predicted_center[1] + float(clamped_delta[1]),
        )
        return clamp_bbox_to_frame_bounds(frame, move_bbox_center(bbox, clamped_center))

    def _center_anchor_max_shift(self, predicted: BBox) -> float:
        return max(
            self.center_anchor_min_shift_px,
            bbox_diagonal(predicted) * self.center_anchor_max_shift_fraction,
        )

    def _should_extent_anchor_guard(
        self,
        previous: BBox,
        predicted: BBox,
        proposal: BBox,
        score_parts: AssociationScore,
        in_conflict: bool,
        suspicious: bool,
        trajectory_guarded: bool,
        direction_guarded: bool,
        bottom_guarded: bool,
        center_guarded: bool,
        memory_guarded: bool,
    ) -> bool:
        if self.extent_anchor_max_upward_fraction <= 0.0:
            return False

        _, previous_bottom = bbox_bottom_center(previous)
        _, predicted_bottom = bbox_bottom_center(predicted)
        _, proposal_bottom = bbox_bottom_center(proposal)
        reference_height = max(1.0, previous[3], predicted[3])
        allowed_upward = max(2.0, reference_height * self.extent_anchor_max_upward_fraction)
        upward_jump = min(previous_bottom, predicted_bottom) - proposal_bottom
        shrinking = proposal[2] < max(previous[2], predicted[2]) * 0.92 or proposal[3] < reference_height * 0.92
        guarded_context = (
            in_conflict
            or suspicious
            or trajectory_guarded
            or direction_guarded
            or bottom_guarded
            or center_guarded
            or memory_guarded
            or score_parts.bottom < self.bottom_guard_min_score
        )
        return guarded_context and shrinking and upward_jump > allowed_upward

    def _apply_extent_anchor_guard(
        self,
        previous: BBox,
        predicted: BBox,
        bbox: BBox,
        frame: Optional[np.ndarray],
    ) -> BBox:
        _, previous_bottom = bbox_bottom_center(previous)
        _, predicted_bottom = bbox_bottom_center(predicted)
        reference_bottom = predicted_bottom
        reference_width = max(previous[2], predicted[2])
        reference_height = max(previous[3], predicted[3])
        allowed_upward = max(2.0, reference_height * self.extent_anchor_max_upward_fraction)
        min_bottom = min(previous_bottom, reference_bottom) - allowed_upward
        min_width = reference_width * self.extent_anchor_min_scale
        min_height = reference_height * self.extent_anchor_min_scale

        x, y, w, h = clamp_bbox_size(bbox)
        center_x, center_y = bbox_center((x, y, w, h))
        bottom = max(y + h, min_bottom)
        w = max(w, min_width)
        h = max(h, min_height)
        y = bottom - h
        x = center_x - (w / 2.0)
        if y + h < min_bottom:
            y = min_bottom - h
        return clamp_bbox_to_frame_bounds(frame, (x, y, w, h))

    def _refresh_pair_order_memories(
        self,
        pending_updates: Sequence[ResolvedTrackUpdate],
        frame_number: int,
    ) -> None:
        if len(pending_updates) < 2:
            return

        active_ids = {update.track.track_id for update in pending_updates}
        for key in list(self.pair_memories):
            if key[0] not in active_ids or key[1] not in active_ids:
                del self.pair_memories[key]
                continue
            if frame_number - self.pair_memories[key].frame_number > self.pair_memory_max_age:
                del self.pair_memories[key]

        for left_index, left_update in enumerate(pending_updates):
            for right_update in pending_updates[left_index + 1 :]:
                previous_iou = bbox_iou(left_update.previous_bbox, right_update.previous_bbox)
                if previous_iou >= self.pair_memory_overlap_iou:
                    continue
                self._store_pair_order_memory(left_update, right_update, frame_number)

    def _store_pair_order_memory(
        self,
        left_update: ResolvedTrackUpdate,
        right_update: ResolvedTrackUpdate,
        frame_number: int,
    ) -> None:
        first_update, second_update = sorted(
            (left_update, right_update),
            key=lambda update: update.track.track_id,
        )
        first_center = bbox_center(first_update.previous_bbox)
        second_center = bbox_center(second_update.previous_bbox)
        _, first_bottom = bbox_bottom_center(first_update.previous_bbox)
        _, second_bottom = bbox_bottom_center(second_update.previous_bbox)
        center_gap_x = second_center[0] - first_center[0]
        center_gap_y = second_center[1] - first_center[1]
        bottom_gap = second_bottom - first_bottom

        if max(abs(center_gap_y), abs(bottom_gap)) < self.pair_memory_min_gap:
            return

        self.pair_memories[(first_update.track.track_id, second_update.track.track_id)] = PairOcclusionMemory(
            frame_number=frame_number,
            center_gap_x=center_gap_x,
            center_gap_y=center_gap_y,
            bottom_gap=bottom_gap,
        )

    def _apply_pair_order_memory(
        self,
        pending_updates: Sequence[ResolvedTrackUpdate],
        frame: Optional[np.ndarray],
        frame_number: int,
    ) -> None:
        if len(pending_updates) < 2 or self.pair_memory_strength <= 0.0:
            return

        updates_by_id = {update.track.track_id: update for update in pending_updates}
        for pair_key, memory in list(self.pair_memories.items()):
            if frame_number - memory.frame_number > self.pair_memory_max_age:
                del self.pair_memories[pair_key]
                continue
            first_update = updates_by_id.get(pair_key[0])
            second_update = updates_by_id.get(pair_key[1])
            if first_update is None or second_update is None:
                continue
            if not self._pair_memory_should_apply(first_update.bbox, second_update.bbox, memory):
                continue

            corrected_first, corrected_second = self._separate_pair_by_order_memory(
                first_update.bbox,
                second_update.bbox,
                memory,
                frame,
            )
            if corrected_first == first_update.bbox and corrected_second == second_update.bbox:
                continue

            first_update.bbox = corrected_first
            second_update.bbox = corrected_second
            first_update.in_conflict = True
            second_update.in_conflict = True
            first_update.overlap_guarded = True
            second_update.overlap_guarded = True
            first_update.state = append_state_token(first_update.state, "ORDER")
            second_update.state = append_state_token(second_update.state, "ORDER")

    def _pair_memory_should_apply(
        self,
        first_bbox: BBox,
        second_bbox: BBox,
        memory: PairOcclusionMemory,
    ) -> bool:
        if bbox_iou(first_bbox, second_bbox) >= self.pair_memory_overlap_iou:
            return True

        first_center = bbox_center(first_bbox)
        second_center = bbox_center(second_bbox)
        _, first_bottom = bbox_bottom_center(first_bbox)
        _, second_bottom = bbox_bottom_center(second_bbox)
        width_scale = max(first_bbox[2], second_bbox[2], 1.0)
        height_scale = max(first_bbox[3], second_bbox[3], 1.0)
        centers_near = (
            abs(second_center[0] - first_center[0]) <= width_scale * 1.75
            and abs(second_center[1] - first_center[1]) <= height_scale * 1.35
        )
        if not centers_near:
            return False

        return (
            self._pair_gap_is_collapsing(second_bottom - first_bottom, memory.bottom_gap)
            or self._pair_gap_is_collapsing(second_center[1] - first_center[1], memory.center_gap_y)
        )

    def _separate_pair_by_order_memory(
        self,
        first_bbox: BBox,
        second_bbox: BBox,
        memory: PairOcclusionMemory,
        frame: Optional[np.ndarray],
    ) -> Tuple[BBox, BBox]:
        first_bbox = clamp_bbox_to_frame_bounds(frame, first_bbox)
        second_bbox = clamp_bbox_to_frame_bounds(frame, second_bbox)
        first_center = bbox_center(first_bbox)
        second_center = bbox_center(second_bbox)
        _, first_bottom = bbox_bottom_center(first_bbox)
        _, second_bottom = bbox_bottom_center(second_bbox)

        first_shift_y = 0.0
        second_shift_y = 0.0
        bottom_shift = self._pair_gap_correction(
            current_gap=second_bottom - first_bottom,
            remembered_gap=memory.bottom_gap,
            scale=(first_bbox[3] + second_bbox[3]) * 0.5,
        )
        if bottom_shift != 0.0:
            first_shift_y -= bottom_shift * 0.5
            second_shift_y += bottom_shift * 0.5
        else:
            center_y_shift = self._pair_gap_correction(
                current_gap=second_center[1] - first_center[1],
                remembered_gap=memory.center_gap_y,
                scale=(first_bbox[3] + second_bbox[3]) * 0.5,
            )
            first_shift_y -= center_y_shift * 0.5
            second_shift_y += center_y_shift * 0.5

        if first_shift_y == 0.0 and second_shift_y == 0.0:
            return first_bbox, second_bbox

        strength = self.pair_memory_strength
        first_bbox = move_bbox_center(first_bbox, (first_center[0], first_center[1] + (first_shift_y * strength)))
        second_bbox = move_bbox_center(second_bbox, (second_center[0], second_center[1] + (second_shift_y * strength)))
        return clamp_bbox_to_frame_bounds(frame, first_bbox), clamp_bbox_to_frame_bounds(frame, second_bbox)

    def _pair_gap_correction(self, current_gap: float, remembered_gap: float, scale: float) -> float:
        if abs(remembered_gap) < self.pair_memory_min_gap:
            return 0.0

        sign = 1.0 if remembered_gap >= 0.0 else -1.0
        target_gap = sign * min(
            max(self.pair_memory_min_gap, abs(remembered_gap) * 0.75),
            max(self.pair_memory_min_gap, scale * 0.60),
        )
        if target_gap >= 0.0 and current_gap >= target_gap:
            return 0.0
        if target_gap < 0.0 and current_gap <= target_gap:
            return 0.0
        return target_gap - current_gap

    def _pair_gap_is_collapsing(self, current_gap: float, remembered_gap: float) -> bool:
        if abs(remembered_gap) < self.pair_memory_min_gap:
            return False
        if current_gap == 0.0 or np.sign(current_gap) != np.sign(remembered_gap):
            return True
        return abs(current_gap) < abs(remembered_gap) * 0.65

    def _apply_track_overlap_guard(
        self,
        pending_updates: Sequence[ResolvedTrackUpdate],
        frame: Optional[np.ndarray],
    ) -> None:
        if self.max_track_overlap_iou >= 1.0 or len(pending_updates) < 2:
            return

        priority_order = sorted(
            range(len(pending_updates)),
            key=lambda index: (
                pending_updates[index].assignment_score,
                pending_updates[index].assignment_margin,
                pending_updates[index].score_parts.appearance,
                pending_updates[index].score_parts.motion,
                pending_updates[index].score_parts.trajectory,
            ),
            reverse=True,
        )
        accepted_indices: List[int] = []

        for update_index in priority_order:
            update = pending_updates[update_index]
            for blocker_index in accepted_indices:
                blocker = pending_updates[blocker_index]
                if bbox_iou(update.bbox, blocker.bbox) < self.max_track_overlap_iou:
                    continue

                update.bbox = separate_bbox_from_overlap(
                    update.bbox,
                    blocker.bbox,
                    update.predicted_bbox,
                    frame,
                    self.max_track_overlap_iou,
                    update.track.track_id,
                )
                update.in_conflict = True
                update.overlap_guarded = True
                update.state = append_state_token(update.state, "SEP")

            accepted_indices.append(update_index)

    def _should_update_appearance(
        self,
        track: TrackState,
        frame: Optional[np.ndarray],
        bbox: BBox,
        in_conflict: bool,
        suspicious: bool,
        ambiguous: bool,
        trajectory_guarded: bool,
        direction_guarded: bool,
        bottom_guarded: bool,
        center_guarded: bool,
        memory_guarded: bool,
        extent_guarded: bool,
        assignment_score: float,
        assignment_margin: float,
        appearance_score: float,
        overlap_guarded: bool,
    ) -> bool:
        if (
            frame is None
            or in_conflict
            or suspicious
            or ambiguous
            or trajectory_guarded
            or direction_guarded
            or bottom_guarded
            or center_guarded
            or memory_guarded
            or extent_guarded
            or overlap_guarded
        ):
            return False
        if assignment_score < self.appearance_commit_min_score:
            return False
        if assignment_margin < self.appearance_commit_min_margin:
            return False
        if track.appearance_updates > 1 and appearance_score < self.appearance_commit_min_similarity:
            return False
        return clip_bbox_to_frame(frame, bbox) is not None

    def _find_candidate_conflicts(self, candidate_boxes: Dict[int, BBox]) -> set[int]:
        conflict_ids: set[int] = set()
        proposal_items = list(candidate_boxes.items())

        for left_index, (left_id, left_bbox) in enumerate(proposal_items):
            for right_id, right_bbox in proposal_items[left_index + 1 :]:
                if bbox_iou(left_bbox, right_bbox) >= self.overlap_iou_threshold:
                    conflict_ids.add(left_id)
                    conflict_ids.add(right_id)

        return conflict_ids

    def _is_suspicious(self, previous: BBox, predicted: BBox, proposal: BBox) -> bool:
        previous_diag = max(1.0, bbox_diagonal(previous))
        center_jump = center_distance(predicted, proposal) / previous_diag

        previous_area = max(1.0, previous[2] * previous[3])
        proposal_area = max(1.0, proposal[2] * proposal[3])
        scale_change = abs(proposal_area - previous_area) / previous_area

        return center_jump > self.max_center_jump or scale_change > self.max_scale_change

    def _update_track_motion(
        self,
        track: TrackState,
        previous: BBox,
        final_bbox: BBox,
        predicted_bbox: BBox,
        raw_bbox: BBox,
        in_conflict: bool,
        state: str,
        frame_number: int,
    ) -> None:
        measured_velocity = bbox_delta(previous, final_bbox)
        track.velocity = blend_bbox(measured_velocity, track.velocity, self.velocity_smoothing)
        track.previous_bbox = previous
        track.predicted_bbox = predicted_bbox
        track.raw_bbox = raw_bbox
        track.occluded_frames = track.occluded_frames + 1 if in_conflict else 0
        track.coordinator_state = state
        record_track_trajectory(track, frame_number, final_bbox, self.trajectory_history_size)

    def _update_appearance(self, track: TrackState, frame: np.ndarray, bbox: BBox) -> None:
        new_hist = extract_reid_histogram(frame, bbox)
        if new_hist is None:
            return
        if track.initial_appearance_hist is None:
            track.initial_appearance_hist = new_hist.copy()
            track.appearance_bank.append(new_hist.copy())
        if track.appearance_hist is None:
            track.appearance_hist = new_hist.copy()
        else:
            track.appearance_hist = (
                (1.0 - self.appearance_update_rate) * track.appearance_hist
                + self.appearance_update_rate * new_hist
            )
            norm = float(np.linalg.norm(track.appearance_hist))
            if norm > 0:
                track.appearance_hist = track.appearance_hist / norm
        self._commit_appearance_memory(track, new_hist)
        track.appearance_updates += 1

    def _commit_appearance_memory(self, track: TrackState, new_hist: np.ndarray) -> None:
        if not track.appearance_bank:
            track.appearance_bank.append(new_hist.copy())
            return

        similarities = [histogram_similarity(memory, new_hist) for memory in track.appearance_bank]
        best_index = int(np.argmax(similarities))
        best_score = similarities[best_index]

        if best_score >= 0.97:
            updated = (0.95 * track.appearance_bank[best_index]) + (0.05 * new_hist)
            norm = float(np.linalg.norm(updated))
            track.appearance_bank[best_index] = updated / norm if norm > 0 else updated
            return

        track.appearance_bank.append(new_hist.copy())
        while len(track.appearance_bank) > self.appearance_bank_size:
            track.appearance_bank.pop(0)


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


class LoRATMultiObjectTracker:
    backend_name = "LoRAT"

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
        confidence_threshold: float,
        disable_amp: bool,
        coordinator_enabled: bool = True,
        overlap_iou_threshold: float = 0.30,
        normal_proposal_weight: float = 0.90,
        occluded_proposal_weight: float = 0.45,
        suspicious_proposal_weight: float = 0.25,
        max_center_jump: float = 0.45,
        max_scale_change: float = 0.65,
        max_occlusion_frames: int = 20,
        detector_method: str = "none",
        detector_interval: int = 5,
        max_detections: int = 12,
        association_min_score: float = 0.15,
        reid_weight: float = 0.45,
        motion_weight: float = 0.30,
        iou_weight: float = 0.15,
        source_weight: float = 0.10,
        appearance_update_rate: float = 0.08,
        appearance_bank_size: int = 8,
        appearance_commit_min_score: float = 0.48,
        appearance_commit_min_margin: float = 0.04,
        appearance_commit_min_similarity: float = 0.35,
        ambiguous_assignment_margin: float = 0.04,
        max_track_overlap_iou: float = 0.70,
        pair_memory_overlap_iou: float = 0.35,
        pair_memory_min_gap: float = 6.0,
        pair_memory_max_age: int = 45,
        pair_memory_strength: float = 0.75,
        source_switch_penalty: float = 0.08,
        reid_gate: float = 0.28,
        reid_competition_weight: float = 0.20,
        reid_competition_margin: float = 0.04,
        trajectory_history_size: int = 12,
        trajectory_weight: float = 0.30,
        trajectory_guard_min_score: float = 0.25,
        trajectory_guard_proposal_weight: float = 0.50,
        trajectory_penalty_weight: float = 0.20,
        recent_memory_frames: int = 10,
        recent_memory_max_shift_fraction: float = 0.25,
        recent_memory_min_shift_px: float = 4.0,
        recent_memory_weight: float = 0.12,
        recent_memory_competition_weight: float = 0.25,
        identity_reject_memory_score: float = 0.12,
        identity_reject_min_score: float = 0.38,
        identity_reject_motion_score: float = 0.55,
        identity_reject_competition_penalty: float = 0.05,
        guarded_max_scale_change: float = 0.25,
        direction_guard_min_score: float = 0.35,
        direction_penalty_weight: float = 0.30,
        bottom_guard_min_score: float = 0.40,
        bottom_penalty_weight: float = 0.25,
        center_anchor_max_shift_fraction: float = 0.25,
        center_anchor_min_shift_px: float = 4.0,
        extent_anchor_max_upward_fraction: float = 0.12,
        extent_anchor_min_scale: float = 0.80,
        resync_guarded_tracks: bool = False,
        guard_resync_min_interval: int = 1,
        memory_recovery_enabled: bool = True,
        memory_recovery_min_score: float = 0.58,
        memory_recovery_search_radius: float = 0.90,
        memory_recovery_scale_step: float = 0.15,
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
        self.confidence_threshold = confidence_threshold
        self.disable_amp = disable_amp
        self.visual_memory_frames = max(1, recent_memory_frames)
        self.resync_guarded_tracks = resync_guarded_tracks
        self.guard_resync_min_interval = max(1, guard_resync_min_interval)
        self.coordinator = MultiObjectCoordinator(
            enabled=coordinator_enabled,
            overlap_iou_threshold=overlap_iou_threshold,
            normal_proposal_weight=normal_proposal_weight,
            occluded_proposal_weight=occluded_proposal_weight,
            suspicious_proposal_weight=suspicious_proposal_weight,
            max_center_jump=max_center_jump,
            max_scale_change=max_scale_change,
            max_occlusion_frames=max_occlusion_frames,
            detector_method=detector_method,
            detector_interval=detector_interval,
            max_detections=max_detections,
            association_min_score=association_min_score,
            reid_weight=reid_weight,
            motion_weight=motion_weight,
            iou_weight=iou_weight,
            source_weight=source_weight,
            appearance_update_rate=appearance_update_rate,
            appearance_bank_size=appearance_bank_size,
            appearance_commit_min_score=appearance_commit_min_score,
            appearance_commit_min_margin=appearance_commit_min_margin,
            appearance_commit_min_similarity=appearance_commit_min_similarity,
            ambiguous_assignment_margin=ambiguous_assignment_margin,
            max_track_overlap_iou=max_track_overlap_iou,
            pair_memory_overlap_iou=pair_memory_overlap_iou,
            pair_memory_min_gap=pair_memory_min_gap,
            pair_memory_max_age=pair_memory_max_age,
            pair_memory_strength=pair_memory_strength,
            source_switch_penalty=source_switch_penalty,
            reid_gate=reid_gate,
            reid_competition_weight=reid_competition_weight,
            reid_competition_margin=reid_competition_margin,
            trajectory_history_size=trajectory_history_size,
            trajectory_weight=trajectory_weight,
            trajectory_guard_min_score=trajectory_guard_min_score,
            trajectory_guard_proposal_weight=trajectory_guard_proposal_weight,
            trajectory_penalty_weight=trajectory_penalty_weight,
            recent_memory_frames=recent_memory_frames,
            recent_memory_max_shift_fraction=recent_memory_max_shift_fraction,
            recent_memory_min_shift_px=recent_memory_min_shift_px,
            recent_memory_weight=recent_memory_weight,
            recent_memory_competition_weight=recent_memory_competition_weight,
            identity_reject_memory_score=identity_reject_memory_score,
            identity_reject_min_score=identity_reject_min_score,
            identity_reject_motion_score=identity_reject_motion_score,
            identity_reject_competition_penalty=identity_reject_competition_penalty,
            guarded_max_scale_change=guarded_max_scale_change,
            direction_guard_min_score=direction_guard_min_score,
            direction_penalty_weight=direction_penalty_weight,
            bottom_guard_min_score=bottom_guard_min_score,
            bottom_penalty_weight=bottom_penalty_weight,
            center_anchor_max_shift_fraction=center_anchor_max_shift_fraction,
            center_anchor_min_shift_px=center_anchor_min_shift_px,
            extent_anchor_max_upward_fraction=extent_anchor_max_upward_fraction,
            extent_anchor_min_scale=extent_anchor_min_scale,
            memory_recovery_enabled=memory_recovery_enabled,
            memory_recovery_min_score=memory_recovery_min_score,
            memory_recovery_search_radius=memory_recovery_search_radius,
            memory_recovery_scale_step=memory_recovery_scale_step,
        )
        self.tracks: List[TrackState] = []
        self.track_by_id: Dict[int, TrackState] = {}
        self.next_track_id = 1
        self.closed = False

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
        self.device = torch.device(self.device_string)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LoRAT was asked to use cuda, but this PyTorch build reports no CUDA/HIP device. "
                "Use --device cpu on this laptop, or install a working CUDA/ROCm PyTorch build."
            )

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
        self.config = config

        self.dtype = self.torch.float32
        transform_config = config["run"]["data"]["eval"]["transform"]
        self.transform = build_data_transform(transform_config, config, self.device, self.dtype)

        model_manager = ModelManager(create_model_build_context(config), rng_fixed_seed=42)
        model_manager.load_state_dict_from_file(str(self.weight_path), strict=False, print_missing=False)
        self.model_manager = model_manager

        inference_config = copy.deepcopy(config["run"]["runner"]["test"]["inference_engine"])
        if self.device.type == "cpu" or self.disable_amp:
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
            max_batch_size=self.track_batch_size,
            num_input_data_streams=1,
            dtype=self.dtype,
            auto_mixed_precision_dtype=self.optimized_model.auto_mixed_precision_dtype,
            model=self.optimized_model.raw_model,
        )
        self.evaluator.start(self.evaluator_context)
        print(
            f"Loaded LoRAT {self.config_name} on {self.device} with weight {self.weight_path.name}. "
            f"Track batch size: {self.track_batch_size}"
        )

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int = 1) -> None:
        self.add_tracks(frame, boxes, frame_number)

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

        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )
        from trackit.data.protocol import SequenceInfo

        added_tracks = []
        tasks = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for bbox in boxes:
            clipped = clip_bbox_to_frame(frame, bbox)
            if clipped is None:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            track = TrackState(
                track_id=track_id,
                bbox=tuple(float(value) for value in clipped),
                color=color_for_track(track_id),
                confidence=1.0,
                previous_bbox=tuple(float(value) for value in clipped),
                predicted_bbox=tuple(float(value) for value in clipped),
                raw_bbox=tuple(float(value) for value in clipped),
            )
            self.tracks.append(track)
            self.track_by_id[track_id] = track
            added_tracks.append(track)
            record_track_trajectory(track, frame_number, track.bbox, self.coordinator.trajectory_history_size)
            remember_visual_template(
                track,
                frame,
                track.bbox,
                frame_number,
                confidence=1.0,
                state="INIT",
                max_recent=self.visual_memory_frames,
                force_initial=True,
            )
            track.active_template_frame = frame_number

            bbox_xyxy = xywh_to_xyxy_np(clipped)
            init_context = SiameseTrackerEvalDataWorker_FrameContext(
                frame_number,
                lambda image=rgb_frame.copy(): image,
                bbox_xyxy,
                None,
            )
            sequence_info = SequenceInfo(
                dataset_name="user",
                data_split=None,
                dataset_full_name=None,
                sequence_name=f"{self.sequence_name}-track-{track_id}",
                length=self.sequence_length,
                fps=self.fps,
            )
            tasks.append(
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track_id,
                    do_task_creation=sequence_info,
                    do_tracker_init=init_context,
                    do_tracker_track=init_context,
                    do_task_finalization=False,
                )
            )

        if added_tracks:
            self.coordinator.initialize_tracks(frame, added_tracks)

        if tasks:
            outputs = self._run_worker_tasks(tasks)
            self._apply_evaluated_frames(outputs, frame, frame_number)
        return added_tracks

    def update(self, frame: np.ndarray, frame_number: int) -> Sequence[TrackState]:
        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )

        active_tracks = [track for track in self.tracks if track.ok]
        if not active_tracks:
            return self.tracks

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tasks = []
        for track in active_tracks:
            track_context = SiameseTrackerEvalDataWorker_FrameContext(
                frame_number,
                lambda image=rgb_frame.copy(): image,
                None,
                None,
            )
            tasks.append(
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track.track_id,
                    do_task_creation=None,
                    do_tracker_init=None,
                    do_tracker_track=track_context,
                    do_task_finalization=False,
                )
            )

        outputs = self._run_worker_tasks(tasks)
        self._apply_evaluated_frames(outputs, frame, frame_number)
        return self.tracks

    def _apply_evaluated_frames(
        self,
        outputs: Optional[dict],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> None:
        evaluated_frames = outputs.get("evaluated_frames", []) if outputs is not None else []
        proposals: Dict[int, Tuple[BBox, Optional[float]]] = {}
        failed_track_ids = set()

        for result in evaluated_frames:
            track = self.track_by_id.get(result.id)
            if track is None:
                continue

            if result.output_box is None:
                failed_track_ids.add(track.track_id)
                continue

            confidence = float(result.output_confidence) if result.output_confidence is not None else None
            if confidence is not None and confidence < self.confidence_threshold:
                track.confidence = confidence
                failed_track_ids.add(track.track_id)
                continue

            proposals[track.track_id] = (xyxy_to_xywh_tuple(result.output_box), confidence)

        for track_id in failed_track_ids - set(proposals):
            track = self.track_by_id.get(track_id)
            if track is None:
                continue
            self._coast_track(track, frame_number, "COAST")

        resolved_updates = self.coordinator.resolve(self.tracks, proposals, frame, frame_number)
        for track_id in set(proposals) - set(resolved_updates):
            track = self.track_by_id.get(track_id)
            if track is not None:
                proposal, confidence = proposals[track_id]
                state = self.coordinator.rejected_track_states.get(track_id, "UNMATCHED+COAST")
                self._coast_track(track, frame_number, state, proposal, confidence)

        for track_id, (bbox, confidence, state) in resolved_updates.items():
            track = self.track_by_id.get(track_id)
            if track is None:
                continue

            track.bbox = bbox
            track.confidence = confidence
            track.ok = True
            track.lost_frames = 0
            track.coordinator_state = state
            if frame is not None and should_commit_visual_template(track):
                remember_visual_template(
                    track,
                    frame,
                    bbox,
                    frame_number,
                    confidence,
                    state,
                    self.visual_memory_frames,
                )

        if frame is not None and self.resync_guarded_tracks:
            guarded_tracks = [
                self.track_by_id[track_id]
                for track_id, (_, _, state) in resolved_updates.items()
                if track_id in self.track_by_id
                and should_resync_state(state)
                and frame_number - self.track_by_id[track_id].last_resync_frame >= self.guard_resync_min_interval
            ]
            if guarded_tracks:
                self._resync_tracks(frame, frame_number, guarded_tracks)

    def _coast_track(
        self,
        track: TrackState,
        frame_number: int,
        state: str,
        raw_bbox: Optional[BBox] = None,
        confidence: Optional[float] = None,
    ) -> None:
        if track.lost_frames >= self.coordinator.max_occlusion_frames:
            mark_track_lost(track, state)
            return

        previous = track.bbox
        coast_prediction = predict_track_bbox(
            track,
            frame_number,
            self.coordinator.trajectory_weight,
            preserve_size=True,
        )
        predicted = clamp_bbox_to_trusted_size_floor(track, coast_prediction, None)
        if predicted != coast_prediction:
            state = append_state_token(state, "SIZE")
        vx, vy, _, _ = track.velocity
        track.velocity = (vx, vy, 0.0, 0.0)
        track.previous_bbox = previous
        track.predicted_bbox = predicted
        track.raw_bbox = raw_bbox
        track.bbox = predicted
        track.confidence = confidence
        track.ok = True
        track.lost_frames += 1
        track.occluded_frames += 1
        track.coordinator_state = state
        track.assigned_source = "coast"
        rejected_score = self.coordinator.rejected_track_scores.get(track.track_id)
        if rejected_score is None:
            track.assignment_score = None
            track.assignment_margin = None
            track.reid_score = None
            track.motion_score = None
            track.trajectory_score = None
            track.memory_score = None
            track.memory_penalty = None
            track.direction_score = None
            track.bottom_score = None
            track.iou_score = None
        else:
            assignment_score, assignment_margin, score_parts, assigned_source = rejected_score
            track.assignment_score = assignment_score
            track.assignment_margin = assignment_margin
            track.reid_score = score_parts.appearance
            track.motion_score = score_parts.motion
            track.trajectory_score = score_parts.trajectory
            track.memory_score = score_parts.memory
            track.memory_penalty = score_parts.memory_penalty
            track.direction_score = score_parts.direction
            track.bottom_score = score_parts.bottom
            track.iou_score = score_parts.iou
            track.assigned_source = f"rejected-{assigned_source}"
        record_track_trajectory(track, frame_number, predicted, self.coordinator.trajectory_history_size)

    def _run_worker_tasks(self, worker_tasks: Sequence[object]):
        from trackit.data.protocol.eval_input import TrackerEvalData

        merged_outputs = {"evaluated_frames": []}
        for task_chunk in chunk_sequence(worker_tasks, self.track_batch_size):
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
        return merged_outputs

    def _resync_tracks(self, frame: np.ndarray, frame_number: int, tracks: Sequence[TrackState]) -> None:
        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tasks = []
        for track in tracks:
            clipped = clip_bbox_to_frame(frame, track.bbox)
            if clipped is None:
                continue
            track.bbox = tuple(float(value) for value in clipped)
            context = SiameseTrackerEvalDataWorker_FrameContext(
                frame_number,
                lambda image=rgb_frame.copy(): image,
                xywh_to_xyxy_np(track.bbox),
                None,
            )
            tasks.append(
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track.track_id,
                    do_task_creation=None,
                    do_tracker_init=context,
                    do_tracker_track=context,
                    do_task_finalization=False,
                )
            )
            track.last_resync_frame = frame_number
            track.resync_count += 1
            track.active_template_frame = frame_number

        if tasks:
            self._run_worker_tasks(tasks)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.tracks:
            from trackit.data.methods.siamese_tracker_eval import SiameseTrackerEvalDataWorker_Task

            tasks = [
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track.track_id,
                    do_task_creation=None,
                    do_tracker_init=None,
                    do_tracker_track=None,
                    do_task_finalization=True,
                )
                for track in self.tracks
            ]
            self._run_worker_tasks(tasks)

        self.evaluator.stop(self.evaluator_context)
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version 3 LoRAT-backed multi-object bounding-box GUI."
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
    parser.add_argument("--device", default="cpu", help="LoRAT device: cpu, cuda:0, etc.")
    parser.add_argument("--lorat-root", type=Path, default=DEFAULT_LORAT_ROOT, help="Local LoRAT checkout.")
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT weight. Defaults from --lorat-config.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Optional track cap. 0 means no cap.")
    parser.add_argument("--track-batch-size", type=int, default=8, help="Internal LoRAT batch size for processing tracks.")
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--disable-amp", action="store_true", help="Disable LoRAT automatic mixed precision.")
    parser.add_argument("--disable-coordinator", action="store_true", help="Disable overlap/motion coordination.")
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.30)
    parser.add_argument("--normal-proposal-weight", type=float, default=0.90)
    parser.add_argument("--occluded-proposal-weight", type=float, default=0.45)
    parser.add_argument("--suspicious-proposal-weight", type=float, default=0.25)
    parser.add_argument("--max-center-jump", type=float, default=0.45)
    parser.add_argument("--max-scale-change", type=float, default=0.65)
    parser.add_argument("--max-occlusion-frames", type=int, default=20)
    parser.add_argument("--detector", choices=("none", "hog"), default="none", help="Optional detector refresh. HOG is person-specific.")
    parser.add_argument("--detector-interval", type=int, default=5, help="Run detector every N frames.")
    parser.add_argument("--max-detections", type=int, default=12, help="Maximum detector boxes considered per frame.")
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
    parser.add_argument(
        "--enable-memory-recovery",
        dest="memory_recovery_enabled",
        action="store_true",
        default=True,
        help="Add object-agnostic memory-matched recovery candidates near the predicted track location.",
    )
    parser.add_argument(
        "--disable-memory-recovery",
        dest="memory_recovery_enabled",
        action="store_false",
        help="Disable memory-matched recovery candidates.",
    )
    parser.add_argument("--memory-recovery-min-score", type=float, default=0.58)
    parser.add_argument("--memory-recovery-search-radius", type=float, default=0.90)
    parser.add_argument("--memory-recovery-scale-step", type=float, default=0.15)
    parser.add_argument(
        "--enable-guard-resync",
        dest="resync_guarded_tracks",
        action="store_true",
        default=False,
        help="Reinitialize LoRAT on coordinator-guarded boxes. Off by default because it can overcorrect.",
    )
    parser.add_argument(
        "--disable-guard-resync",
        dest="resync_guarded_tracks",
        action="store_false",
        help="Keep LoRAT internal state unchanged after coordinator guards.",
    )
    parser.add_argument("--guard-resync-min-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, help="MOTChallenge-format result file.")
    parser.add_argument("--save-video", type=Path, help="Optional annotated MP4 output path.")
    parser.add_argument("--debug-log", type=Path, help="Coordinator debug CSV output path. Defaults to outputs/debug.")
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


def create_backend(args: argparse.Namespace, source: FrameSource):
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
        args.confidence_threshold,
        args.disable_amp,
        not args.disable_coordinator,
        args.overlap_iou_threshold,
        args.normal_proposal_weight,
        args.occluded_proposal_weight,
        args.suspicious_proposal_weight,
        args.max_center_jump,
        args.max_scale_change,
        args.max_occlusion_frames,
        args.detector,
        args.detector_interval,
        args.max_detections,
        args.association_min_score,
        args.reid_weight,
        args.motion_weight,
        args.iou_weight,
        args.source_weight,
        args.appearance_update_rate,
        args.appearance_bank_size,
        args.appearance_commit_min_score,
        args.appearance_commit_min_margin,
        args.appearance_commit_min_similarity,
        args.ambiguous_assignment_margin,
        args.max_track_overlap_iou,
        args.pair_memory_overlap_iou,
        args.pair_memory_min_gap,
        args.pair_memory_max_age,
        args.pair_memory_strength,
        args.source_switch_penalty,
        args.reid_gate,
        args.reid_competition_weight,
        args.reid_competition_margin,
        args.trajectory_history_size,
        args.trajectory_weight,
        args.trajectory_guard_min_score,
        args.trajectory_guard_proposal_weight,
        args.trajectory_penalty_weight,
        args.recent_memory_frames,
        args.recent_memory_max_shift_fraction,
        args.recent_memory_min_shift_px,
        args.recent_memory_weight,
        args.recent_memory_competition_weight,
        args.identity_reject_memory_score,
        args.identity_reject_min_score,
        args.identity_reject_motion_score,
        args.identity_reject_competition_penalty,
        args.guarded_max_scale_change,
        args.direction_guard_min_score,
        args.direction_penalty_weight,
        args.bottom_guard_min_score,
        args.bottom_penalty_weight,
        args.center_anchor_max_shift_fraction,
        args.center_anchor_min_shift_px,
        args.extent_anchor_max_upward_fraction,
        args.extent_anchor_min_scale,
        args.resync_guarded_tracks,
        args.guard_resync_min_interval,
        args.memory_recovery_enabled,
        args.memory_recovery_min_score,
        args.memory_recovery_search_radius,
        args.memory_recovery_scale_step,
    )


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


def make_visual_memory(frame: np.ndarray, bbox: BBox, padding_factor: float = 2.5) -> Optional[Tuple[np.ndarray, BBox]]:
    clipped = clip_bbox_to_frame(frame, bbox)
    if clipped is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), tuple(float(value) for value in clipped)


def remember_visual_template(
    track: TrackState,
    frame: np.ndarray,
    bbox: BBox,
    frame_number: int,
    confidence: Optional[float],
    state: str,
    max_recent: int,
    force_initial: bool = False,
) -> None:
    memory_data = make_visual_memory(frame, bbox)
    if memory_data is None:
        return

    image_rgb, local_bbox = memory_data
    memory_hist = extract_reid_histogram(frame, bbox)
    memory = TrackVisualMemory(
        frame_number=int(frame_number),
        image_rgb=image_rgb,
        bbox=local_bbox,
        confidence=confidence,
        state=state,
        initial=force_initial,
        appearance_hist=memory_hist,
    )
    if force_initial or track.initial_visual_memory is None:
        track.initial_visual_memory = memory
    if force_initial:
        return

    if track.visual_memory and track.visual_memory[-1].frame_number == memory.frame_number:
        track.visual_memory[-1] = memory
    else:
        track.visual_memory.append(memory)

    max_recent = max(1, int(max_recent))
    if len(track.visual_memory) > max_recent:
        del track.visual_memory[: len(track.visual_memory) - max_recent]


def should_commit_visual_template(track: TrackState) -> bool:
    if not track.ok:
        return False
    if track.confidence is not None and track.confidence < 0.45:
        return False
    if track.assignment_score is not None and track.assignment_score < 0.45:
        return False

    tokens = set(track.coordinator_state.split("+")) if track.coordinator_state else set()
    blocked_tokens = {
        "AMBIG",
        "BOTTOM",
        "CENTER",
        "COAST",
        "DIR",
        "EXTENT",
        "LOST",
        "MEM",
        "MEMREC",
        "MEMREC_OCC",
        "OCC",
        "PATH",
        "REID?",
        "REJECT",
        "SEP",
        "SIZE",
        "SMOOTH",
        "UNMATCHED",
    }
    return not bool(tokens & blocked_tokens)


def xywh_to_xyxy_np(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array((x, y, x + w, y + h), dtype=np.float64)


def xyxy_to_xywh_tuple(bbox: np.ndarray) -> BBox:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_iou(left: BBox, right: BBox) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_x2 = lx + lw
    left_y2 = ly + lh
    right_x2 = rx + rw
    right_y2 = ry + rh

    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = (lw * lh) + (rw * rh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def bbox_bottom_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + h


def bbox_diagonal(bbox: BBox) -> float:
    _, _, w, h = bbox
    return float(np.hypot(w, h))


def bbox_area(bbox: BBox) -> float:
    _, _, w, h = bbox
    return max(1.0, float(w) * float(h))


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


def recent_memory_affinity(
    memory_bbox: BBox,
    candidate: BBox,
    max_shift_fraction: float,
    min_shift_px: float,
) -> float:
    tolerance = max(float(min_shift_px), bbox_diagonal(memory_bbox) * max(0.01, float(max_shift_fraction)))
    normalized_distance = center_distance(memory_bbox, candidate) / max(1.0, tolerance)
    return max(0.0, 1.0 - min(1.0, normalized_distance))


def motion_affinity(predicted: BBox, candidate: BBox, reference_diagonal: float) -> float:
    center_score = center_affinity(predicted, candidate, reference_diagonal)
    scale_score = scale_similarity(predicted, candidate)
    aspect_score = aspect_similarity(predicted, candidate)
    return (0.70 * center_score) + (0.20 * scale_score) + (0.10 * aspect_score)


def bottom_anchor_affinity(predicted: BBox, candidate: BBox) -> float:
    _, predicted_bottom = bbox_bottom_center(predicted)
    _, candidate_bottom = bbox_bottom_center(candidate)
    _, _, _, predicted_height = predicted
    tolerance = max(2.0, predicted_height * 0.35)
    normalized_distance = abs(candidate_bottom - predicted_bottom) / tolerance
    return max(0.0, 1.0 - min(1.0, normalized_distance))


def trajectory_center_velocity(track: TrackState) -> Optional[np.ndarray]:
    if not track_has_trajectory(track):
        return None

    history = track.trajectory[-6:]
    frames = np.array([frame_number for frame_number, _ in history], dtype=np.float32)
    centers = np.array([bbox_center(bbox) for _, bbox in history], dtype=np.float32)
    if len(frames) < 3 or float(frames[-1] - frames[0]) <= 0:
        return None

    centered_frames = frames - frames.mean()
    variance = float(np.dot(centered_frames, centered_frames))
    if variance <= 0:
        return None

    return (centered_frames[:, None] * (centers - centers.mean(axis=0))).sum(axis=0) / variance


def direction_consistency_score(track: TrackState, candidate: BBox) -> float:
    expected = trajectory_center_velocity(track)
    if expected is None:
        expected = np.array([track.velocity[0], track.velocity[1]], dtype=np.float32)

    candidate_delta = np.array(
        [
            bbox_center(candidate)[0] - bbox_center(track.bbox)[0],
            bbox_center(candidate)[1] - bbox_center(track.bbox)[1],
        ],
        dtype=np.float32,
    )
    min_motion = max(2.0, bbox_diagonal(track.bbox) * 0.03)
    expected_norm = float(np.linalg.norm(expected))
    candidate_norm = float(np.linalg.norm(candidate_delta))
    if expected_norm < min_motion or candidate_norm < min_motion:
        return 0.50

    cosine = float(np.dot(expected, candidate_delta) / (expected_norm * candidate_norm))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def bbox_delta(previous: BBox, current: BBox) -> BBox:
    return tuple(float(current_value - previous_value) for previous_value, current_value in zip(previous, current))


TRUSTED_SIZE_FLOOR_SCALE = 0.45


def predict_bbox(bbox: BBox, velocity: BBox) -> BBox:
    return clamp_bbox_size(tuple(float(value + delta) for value, delta in zip(bbox, velocity)))


def predict_bbox_preserve_size(bbox: BBox, velocity: BBox) -> BBox:
    x, y, w, h = bbox
    dx, dy, _, _ = velocity
    return clamp_bbox_size((float(x + dx), float(y + dy), w, h))


def clamp_bbox_scale_change(anchor: BBox, bbox: BBox, max_scale_change: float) -> BBox:
    if max_scale_change <= 0:
        return bbox

    x, y, w, h = clamp_bbox_size(bbox)
    anchor_w = max(1.0, float(anchor[2]))
    anchor_h = max(1.0, float(anchor[3]))
    lower = max(0.05, 1.0 - max_scale_change)
    upper = 1.0 + max_scale_change
    clamped_w = max(anchor_w * lower, min(w, anchor_w * upper))
    clamped_h = max(anchor_h * lower, min(h, anchor_h * upper))
    center_x, center_y = bbox_center((x, y, w, h))
    return clamp_bbox_size((center_x - (clamped_w / 2.0), center_y - (clamped_h / 2.0), clamped_w, clamped_h))


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


def track_has_trajectory(track: TrackState, min_points: int = 3) -> bool:
    return len(track.trajectory) >= min_points


def predict_trajectory_bbox(track: TrackState, target_frame_number: int) -> Optional[BBox]:
    if not track_has_trajectory(track):
        return None

    unique_history: List[Tuple[int, BBox]] = []
    for frame_number, bbox in track.trajectory:
        if unique_history and unique_history[-1][0] == frame_number:
            unique_history[-1] = (frame_number, bbox)
        else:
            unique_history.append((frame_number, bbox))

    if len(unique_history) < 3:
        return None

    frames = np.array([frame_number for frame_number, _ in unique_history], dtype=np.float32)
    if float(frames[-1] - frames[0]) <= 0:
        return None

    boxes = np.array([bbox for _, bbox in unique_history], dtype=np.float32)
    centered_frames = frames - frames.mean()
    variance = float(np.dot(centered_frames, centered_frames))
    if variance <= 0:
        return None

    slopes = (centered_frames[:, None] * (boxes - boxes.mean(axis=0))).sum(axis=0) / variance
    frame_delta = max(1.0, float(target_frame_number - frames[-1]))
    predicted = boxes[-1] + (slopes * frame_delta)
    return clamp_bbox_size(tuple(float(value) for value in predicted))


def predict_recent_memory_bbox(
    track: TrackState,
    target_frame_number: int,
    memory_frames: int = 10,
) -> Optional[BBox]:
    if not track_has_trajectory(track):
        return None

    unique_history: List[Tuple[int, BBox]] = []
    for frame_number, bbox in track.trajectory:
        if unique_history and unique_history[-1][0] == frame_number:
            unique_history[-1] = (frame_number, bbox)
        else:
            unique_history.append((frame_number, bbox))

    recent_history = unique_history[-max(3, memory_frames) :]
    if len(recent_history) < 3:
        return None

    frames = np.array([frame_number for frame_number, _ in recent_history], dtype=np.float32)
    boxes = np.array([bbox for _, bbox in recent_history], dtype=np.float32)
    if float(frames[-1] - frames[0]) <= 0:
        return None

    deltas = []
    for index in range(1, len(recent_history)):
        frame_delta = float(frames[index] - frames[index - 1])
        if frame_delta <= 0:
            continue
        deltas.append((boxes[index] - boxes[index - 1]) / frame_delta)

    if not deltas:
        return None

    median_delta = np.median(np.asarray(deltas, dtype=np.float32), axis=0)
    target_delta = max(1.0, float(target_frame_number - frames[-1]))
    predicted = boxes[-1] + (median_delta * target_delta)
    return clamp_bbox_size(tuple(float(value) for value in predicted))


def predict_track_bbox(
    track: TrackState,
    frame_number: int,
    trajectory_weight: float,
    preserve_size: bool = False,
) -> BBox:
    instant_prediction = (
        predict_bbox_preserve_size(track.bbox, track.velocity)
        if preserve_size
        else predict_bbox(track.bbox, track.velocity)
    )
    trajectory_prediction = predict_trajectory_bbox(track, frame_number)
    if trajectory_prediction is None:
        return instant_prediction
    prediction = clamp_bbox_size(blend_bbox(instant_prediction, trajectory_prediction, trajectory_weight))
    return preserve_bbox_size(track.bbox, prediction) if preserve_size else prediction


def clamp_bbox_size(bbox: BBox) -> BBox:
    x, y, w, h = bbox
    return float(x), float(y), max(1.0, float(w)), max(1.0, float(h))


def preserve_bbox_size(anchor: BBox, bbox: BBox) -> BBox:
    _, _, anchor_w, anchor_h = clamp_bbox_size(anchor)
    center_x, center_y = bbox_center(bbox)
    return clamp_bbox_size(
        (
            center_x - (anchor_w / 2.0),
            center_y - (anchor_h / 2.0),
            anchor_w,
            anchor_h,
        )
    )


def trusted_size_floor(track: TrackState) -> Optional[Tuple[float, float]]:
    widths: List[float] = []
    heights: List[float] = []
    if track.initial_visual_memory is not None:
        widths.append(max(1.0, float(track.initial_visual_memory.bbox[2])))
        heights.append(max(1.0, float(track.initial_visual_memory.bbox[3])))
    for memory in track.visual_memory[-10:]:
        widths.append(max(1.0, float(memory.bbox[2])))
        heights.append(max(1.0, float(memory.bbox[3])))

    if not widths or not heights:
        return None

    reference_width = float(np.median(np.asarray(widths, dtype=np.float32)))
    reference_height = float(np.median(np.asarray(heights, dtype=np.float32)))
    return (
        max(1.0, reference_width * TRUSTED_SIZE_FLOOR_SCALE),
        max(1.0, reference_height * TRUSTED_SIZE_FLOOR_SCALE),
    )


def clamp_bbox_to_trusted_size_floor(
    track: TrackState,
    bbox: BBox,
    frame: Optional[np.ndarray],
) -> BBox:
    x, y, w, h = clamp_bbox_size(bbox)
    floor = trusted_size_floor(track)
    if floor is None:
        return clamp_bbox_to_frame_bounds(frame, (x, y, w, h))

    min_w, min_h = floor
    if w >= min_w and h >= min_h:
        return clamp_bbox_to_frame_bounds(frame, (x, y, w, h))

    center_x, center_y = bbox_center((x, y, w, h))
    guarded_w = max(w, min_w)
    guarded_h = max(h, min_h)
    return clamp_bbox_to_frame_bounds(
        frame,
        (
            center_x - (guarded_w / 2.0),
            center_y - (guarded_h / 2.0),
            guarded_w,
            guarded_h,
        ),
    )


def blend_bbox(anchor: BBox, proposal: BBox, proposal_weight: float) -> BBox:
    proposal_weight = max(0.0, min(1.0, proposal_weight))
    anchor_weight = 1.0 - proposal_weight
    return tuple(
        float((anchor_value * anchor_weight) + (proposal_value * proposal_weight))
        for anchor_value, proposal_value in zip(anchor, proposal)
    )


def append_state_token(state: str, token: str) -> str:
    if not state:
        return token
    tokens = state.split("+")
    if token in tokens:
        return state
    return f"{state}+{token}"


def should_resync_state(state: str) -> bool:
    if not state:
        return False
    tokens = set(state.split("+"))
    return bool(tokens & {"PATH", "SEP", "REID?", "SMOOTH", "REJECT"})


def move_bbox_center(bbox: BBox, center: Tuple[float, float]) -> BBox:
    _, _, w, h = bbox
    center_x, center_y = center
    return clamp_bbox_size((center_x - (w / 2.0), center_y - (h / 2.0), w, h))


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


def memory_recovery_search_boxes(
    predicted: BBox,
    frame: np.ndarray,
    radius_fraction: float,
    scale_step: float,
) -> List[BBox]:
    predicted = clamp_bbox_to_frame_bounds(frame, predicted)
    center_x, center_y = bbox_center(predicted)
    _, _, width, height = predicted
    radius = max(4.0, bbox_diagonal(predicted) * max(0.0, radius_fraction))
    offsets = (-radius, -radius * 0.5, 0.0, radius * 0.5, radius)
    scales = (1.0,)
    if scale_step > 0.0:
        scales = (max(0.50, 1.0 - scale_step), 1.0, 1.0 + scale_step)

    boxes: List[BBox] = []
    seen: set[Tuple[int, int, int, int]] = set()
    for scale in scales:
        scaled_w = max(1.0, width * scale)
        scaled_h = max(1.0, height * scale)
        for dx in offsets:
            for dy in offsets:
                candidate = clamp_bbox_to_frame_bounds(
                    frame,
                    (
                        center_x + dx - (scaled_w / 2.0),
                        center_y + dy - (scaled_h / 2.0),
                        scaled_w,
                        scaled_h,
                    ),
                )
                key = tuple(int(round(value)) for value in candidate)
                if key in seen:
                    continue
                seen.add(key)
                boxes.append(candidate)
    return boxes


def separate_bbox_from_overlap(
    moving_bbox: BBox,
    blocker_bbox: BBox,
    preferred_anchor: BBox,
    frame: Optional[np.ndarray],
    max_iou: float,
    seed: int,
) -> BBox:
    moving_bbox = clamp_bbox_to_frame_bounds(frame, moving_bbox)
    blocker_bbox = clamp_bbox_to_frame_bounds(frame, blocker_bbox)
    preferred_anchor = clamp_bbox_to_frame_bounds(frame, preferred_anchor)
    if bbox_iou(moving_bbox, blocker_bbox) < max_iou:
        return moving_bbox
    if bbox_iou(preferred_anchor, blocker_bbox) < max_iou:
        return preferred_anchor

    blocker_center = bbox_center(blocker_bbox)
    anchor_center = bbox_center(preferred_anchor)
    moving_center = bbox_center(moving_bbox)
    direction = np.array(
        [anchor_center[0] - blocker_center[0], anchor_center[1] - blocker_center[1]],
        dtype=np.float32,
    )
    if float(np.linalg.norm(direction)) < 1.0:
        direction = np.array(
            [moving_center[0] - blocker_center[0], moving_center[1] - blocker_center[1]],
            dtype=np.float32,
        )
    if float(np.linalg.norm(direction)) < 1.0:
        angle = ((seed * 137.508) % 360.0) * np.pi / 180.0
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.array([1.0, 0.0], dtype=np.float32)
    step_size = max(2.0, max(bbox_diagonal(moving_bbox), bbox_diagonal(blocker_bbox)) * 0.04)
    best_bbox = moving_bbox
    best_iou = bbox_iou(moving_bbox, blocker_bbox)

    for step in range(1, 41):
        shift = direction * (step_size * step)
        candidate_center = (moving_center[0] + float(shift[0]), moving_center[1] + float(shift[1]))
        candidate = clamp_bbox_to_frame_bounds(frame, move_bbox_center(moving_bbox, candidate_center))
        candidate_iou = bbox_iou(candidate, blocker_bbox)
        if candidate_iou < max_iou:
            return candidate
        if candidate_iou < best_iou:
            best_bbox = candidate
            best_iou = candidate_iou

    return best_bbox


def mark_track_lost(track: TrackState, state: str = "LOST") -> None:
    previous = track.bbox
    predicted = clamp_bbox_to_trusted_size_floor(
        track,
        predict_bbox_preserve_size(track.bbox, track.velocity),
        None,
    )
    vx, vy, _, _ = track.velocity
    track.velocity = (vx, vy, 0.0, 0.0)
    track.previous_bbox = previous
    track.predicted_bbox = predicted
    track.raw_bbox = None
    track.bbox = predicted
    track.ok = False
    track.lost_frames += 1
    track.assignment_score = None
    track.assignment_margin = None
    track.reid_score = None
    track.motion_score = None
    track.trajectory_score = None
    track.memory_score = None
    track.memory_penalty = None
    track.direction_score = None
    track.bottom_score = None
    track.iou_score = None
    track.assigned_source = ""
    track.coordinator_state = state


REGION_FEATURE_LENGTH = (12 * 6 * 4) + (8 * 8 * 4) + 16 + 9 + 1 + 12


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
        [
            crop[:half_h, :half_w],
            crop[:half_h, half_w:],
            crop[half_h:, :half_w],
            crop[half_h:, half_w:],
        ]
    )

    features = []
    for region in regions:
        features.append(region_appearance_features(region))

    hist = np.concatenate(features).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm <= 0:
        return None
    return hist / norm


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
    score = float(np.dot(left, right))
    return max(0.0, min(1.0, score))


def track_has_appearance(track: TrackState) -> bool:
    return (
        track.initial_appearance_hist is not None
        or track.appearance_hist is not None
        or bool(track.appearance_bank)
        or (
            track.initial_visual_memory is not None
            and track.initial_visual_memory.appearance_hist is not None
        )
        or any(memory.appearance_hist is not None for memory in track.visual_memory)
    )


def track_appearance_similarity(track: TrackState, candidate_hist: np.ndarray) -> float:
    scores = []
    if track.initial_appearance_hist is not None:
        scores.append(histogram_similarity(track.initial_appearance_hist, candidate_hist))
    if track.appearance_hist is not None:
        scores.append(histogram_similarity(track.appearance_hist, candidate_hist))
    scores.extend(histogram_similarity(memory, candidate_hist) for memory in track.appearance_bank)
    if track.initial_visual_memory is not None and track.initial_visual_memory.appearance_hist is not None:
        scores.append(histogram_similarity(track.initial_visual_memory.appearance_hist, candidate_hist))
    scores.extend(
        histogram_similarity(memory.appearance_hist, candidate_hist)
        for memory in track.visual_memory
        if memory.appearance_hist is not None
    )
    if not scores:
        return 0.50
    scores.sort(reverse=True)
    if len(scores) == 1:
        return scores[0]
    return (0.75 * scores[0]) + (0.25 * scores[1])


def nms_detections(
    detections: Sequence[Tuple[BBox, Optional[float]]],
    iou_threshold: float,
) -> List[Tuple[BBox, Optional[float]]]:
    kept: List[Tuple[BBox, Optional[float]]] = []
    for bbox, confidence in detections:
        if any(bbox_iou(bbox, kept_bbox) >= iou_threshold for kept_bbox, _ in kept):
            continue
        kept.append((bbox, confidence))
    return kept


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
    if scores.size == 0:
        return []

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
    "frame,track_id,ok,state,confidence,"
    "x,y,w,h,width_px,length_px,area_px,raw_x,raw_y,raw_w,raw_h,pred_x,pred_y,pred_w,pred_h,prev_x,prev_y,prev_w,prev_h,"
    "vel_x,vel_y,vel_w,vel_h,assignment_score,assignment_margin,reid_score,motion_score,"
    "trajectory_score,memory_score,memory_penalty,direction_score,bottom_score,iou_score,assigned_source,lost_frames,occluded_frames,resync_count,"
    "template_frame,visual_memory_count,has_initial_visual_memory\n"
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
            csv_text(track.coordinator_state),
            csv_float(track.confidence),
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
            csv_float(track.trajectory_score),
            csv_float(track.memory_score),
            csv_float(track.memory_penalty),
            csv_float(track.direction_score),
            csv_float(track.bottom_score),
            csv_float(track.iou_score),
            csv_text(track.assigned_source),
            str(track.lost_frames),
            str(track.occluded_frames),
            str(track.resync_count),
            str(track.active_template_frame) if track.active_template_frame is not None else "",
            str(len(track.visual_memory)),
            "1" if track.initial_visual_memory is not None else "0",
        ]
        lines.append(",".join(fields) + "\n")


def write_debug_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEBUG_LOG_HEADER + "".join(lines), encoding="utf-8")


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
) -> np.ndarray:
    output = frame.copy()
    header = f"Frame {frame_number} | {backend_label} | q quit | a add boxes | p pause"
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    for track in tracks:
        x, y, w, h = [int(round(value)) for value in track.bbox]
        color = track.color if track.ok else (0, 0, 255)
        label = f"ID {track.track_id}"
        if track.confidence is not None:
            label += f" {track.confidence:.2f}"
        if track.coordinator_state:
            label += f" {track.coordinator_state}"
        if track.resync_count:
            label += f" R{track.resync_count}"
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


def make_video_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def main() -> int:
    args = parse_args()
    frame_source = open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name, "lorat")
    debug_log_path = args.debug_log or default_debug_log_path(frame_source.name, "lorat")
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

    backend = create_backend(args, frame_source)
    writer = make_video_writer(args.save_video, frame_source.fps, first_frame) if args.save_video else None
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
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        append_mot_results(mot_lines, frame_number, backend.tracks)
        append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)

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

            shown = draw_tracks(frame, backend.tracks, frame_number, backend.backend_name)
            if writer is not None and not paused:
                writer.write(shown)

            cv2.imshow("LoRAT Multi-Object Tracker v3", shown)
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

    if args.save_video:
        print(f"Wrote annotated video to: {args.save_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
