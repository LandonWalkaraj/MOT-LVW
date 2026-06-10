from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bounding_box_v5_lorat_shared as v5


# ---------------------------------------------------------------------------
# V6 LoRAT-gated MOT defaults
# ---------------------------------------------------------------------------

V6_EXECUTION_MODE = "gated-sot-memory"
DEFAULT_V6_PRIMARY_SLOTS_PER_TRACK = 1
DEFAULT_V6_RECOVERY_SLOTS_PER_TRACK = 5
DEFAULT_V6_RECOVERY_INTERVAL = 15
DEFAULT_V6_RECOVERY_MIN_CONFIDENCE = 0.45
DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_SCORE = 0.58
DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_MARGIN = 0.08
DEFAULT_V6_RECOVERY_STALE_SLOT_FRAMES = 30


class LoRATGatedMultiObjectTracker(v5.LoRATMultiObjectTracker):
    """V6 tracker: LoRAT remains the SOT primitive, but memory slots are gated.

    V5 evaluates several LoRAT memory templates for every object on every frame.
    V6 keeps one LoRAT task per object as the normal path, then expands to the
    initial anchor and recent memory templates when cached confidence, identity,
    motion, or occlusion state says the target is uncertain.
    """

    backend_name = "LoRAT-v6-gated"

    def __init__(
        self,
        *args,
        v6_primary_slots_per_track: int = DEFAULT_V6_PRIMARY_SLOTS_PER_TRACK,
        v6_recovery_slots_per_track: int = DEFAULT_V6_RECOVERY_SLOTS_PER_TRACK,
        v6_recovery_interval: int = DEFAULT_V6_RECOVERY_INTERVAL,
        v6_recovery_min_confidence: float = DEFAULT_V6_RECOVERY_MIN_CONFIDENCE,
        v6_recovery_min_assignment_score: float = DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_SCORE,
        v6_recovery_min_assignment_margin: float = DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_MARGIN,
        v6_recovery_stale_slot_frames: int = DEFAULT_V6_RECOVERY_STALE_SLOT_FRAMES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backend_name = "LoRAT-v6-gated"
        self.v6_primary_slots_per_track = max(1, int(v6_primary_slots_per_track))
        self.v6_recovery_slots_per_track = max(1, int(v6_recovery_slots_per_track))
        self.v6_recovery_interval = max(0, int(v6_recovery_interval))
        self.v6_recovery_min_confidence = max(0.0, min(1.0, float(v6_recovery_min_confidence)))
        self.v6_recovery_min_assignment_score = max(0.0, min(1.0, float(v6_recovery_min_assignment_score)))
        self.v6_recovery_min_assignment_margin = max(0.0, float(v6_recovery_min_assignment_margin))
        self.v6_recovery_stale_slot_frames = max(0, int(v6_recovery_stale_slot_frames))
        self.v6_gating_decisions = 0
        self.v6_primary_decisions = 0
        self.v6_recovery_decisions = 0
        self.v6_selected_slot_items = 0
        self.v6_recovery_reason_counts: Counter[str] = Counter()

    def _select_lorat_tracking_slots(self, track: v5.TrackState, frame_number: int) -> List[v5.LoRATMemorySlot]:
        slots = self._get_track_slots(track)
        if not slots:
            self._record_v6_gating([], [])
            return []

        reasons = self._v6_recovery_reasons(track, slots, frame_number)
        if reasons:
            selected = self._select_v6_recovery_slots(track, slots, frame_number)
        else:
            selected = self._select_v6_primary_slots(track, slots)
        self._record_v6_gating(selected, reasons)
        return selected

    def _select_v6_primary_slots(
        self,
        track: v5.TrackState,
        slots: Sequence[v5.LoRATMemorySlot],
    ) -> List[v5.LoRATMemorySlot]:
        hard_limit = self._v6_hard_slot_limit(len(slots))
        limit = min(len(slots), hard_limit, self.v6_primary_slots_per_track)
        if limit <= 0:
            return []

        selected: List[v5.LoRATMemorySlot] = []
        by_label = {slot.label: slot for slot in slots}

        def add(slot: Optional[v5.LoRATMemorySlot]) -> None:
            if slot is None or len(selected) >= limit:
                return
            if any(existing.task_id == slot.task_id for existing in selected):
                return
            selected.append(slot)

        active_slot = by_label.get(track.active_lorat_slot)
        recent_slots = [slot for slot in slots if slot.label != "initial"]
        freshest_recent = max(recent_slots, key=lambda item: (item.last_refresh_frame, item.task_id), default=None)
        primary_slot = active_slot
        if primary_slot is None or primary_slot.label == "initial":
            primary_slot = freshest_recent or primary_slot
        add(primary_slot or by_label.get("initial"))
        add(by_label.get("initial"))
        add(freshest_recent)
        for slot in sorted(recent_slots, key=lambda item: (item.last_refresh_frame, item.task_id), reverse=True):
            add(slot)
        return selected

    def _select_v6_recovery_slots(
        self,
        track: v5.TrackState,
        slots: Sequence[v5.LoRATMemorySlot],
        frame_number: int,
    ) -> List[v5.LoRATMemorySlot]:
        hard_limit = self._v6_hard_slot_limit(len(slots))
        limit = min(len(slots), hard_limit, self.v6_recovery_slots_per_track)
        if limit <= 0:
            return []

        selected: List[v5.LoRATMemorySlot] = []

        def add(slot: Optional[v5.LoRATMemorySlot]) -> None:
            if slot is None or len(selected) >= limit:
                return
            if any(existing.task_id == slot.task_id for existing in selected):
                return
            selected.append(slot)

        by_label = {slot.label: slot for slot in slots}
        add(by_label.get("initial"))
        add(by_label.get(track.active_lorat_slot))

        recent_slots = sorted(
            (slot for slot in slots if slot.label != "initial"),
            key=lambda item: item.label,
        )
        if recent_slots:
            add(max(recent_slots, key=lambda item: (item.last_refresh_frame, item.task_id)))
            start_frame = track.trajectory[0][0] if track.trajectory else frame_number
            rotating_index = max(0, frame_number - start_frame - 1) % len(recent_slots)
            for offset in range(len(recent_slots)):
                add(recent_slots[(rotating_index + offset) % len(recent_slots)])
            for slot in sorted(recent_slots, key=lambda item: (item.last_refresh_frame, item.task_id), reverse=True):
                add(slot)
        return selected

    def _v6_hard_slot_limit(self, slot_count: int) -> int:
        if self.lorat_active_slots_per_track <= 0:
            return slot_count
        return max(0, min(slot_count, self.lorat_active_slots_per_track))

    def _v6_recovery_reasons(
        self,
        track: v5.TrackState,
        slots: Sequence[v5.LoRATMemorySlot],
        frame_number: int,
    ) -> List[str]:
        if len(slots) <= self.v6_primary_slots_per_track:
            return []

        reasons: List[str] = []
        state = str(track.state or "").upper()
        state_tokens = (
            "MISS",
            "LOWCONF",
            "ID_UNCERTAIN",
            "OCCLU",
            "LOST",
            "NOLEARN",
            "SHRINK",
            "REIDRECOVERY",
        )
        for token in state_tokens:
            if token in state:
                reasons.append(f"STATE_{token}")
                break

        if not track.ok:
            reasons.append("NOT_OK")
        if track.lost_frames > 0:
            reasons.append("LOST_FRAMES")
        if track.occluded_frames > 0:
            reasons.append("OCCLUDED_FRAMES")
        if track.learning_block_reason:
            reasons.append("LEARNING_HELD")

        if track.confidence is not None and track.confidence < self.v6_recovery_min_confidence:
            reasons.append("LOW_CONFIDENCE")
        if (
            track.assignment_score is not None
            and track.assignment_score < self.v6_recovery_min_assignment_score
        ):
            reasons.append("LOW_ASSIGNMENT_SCORE")
        if (
            track.assignment_margin is not None
            and track.assignment_margin < self.v6_recovery_min_assignment_margin
        ):
            reasons.append("LOW_ASSIGNMENT_MARGIN")

        active_slot = next((slot for slot in slots if slot.label == track.active_lorat_slot), None)
        if (
            active_slot is not None
            and self.v6_recovery_stale_slot_frames > 0
            and frame_number - active_slot.anchor_frame_number >= self.v6_recovery_stale_slot_frames
        ):
            reasons.append("STALE_ACTIVE_SLOT")

        if self.v6_recovery_interval > 0 and track.trajectory:
            start_frame = track.trajectory[0][0]
            if frame_number > start_frame and (frame_number - start_frame) % self.v6_recovery_interval == 0:
                reasons.append("PERIODIC_ANCHOR_CHECK")

        return reasons

    def _record_v6_gating(self, selected: Sequence[v5.LoRATMemorySlot], reasons: Sequence[str]) -> None:
        self.v6_gating_decisions += 1
        self.v6_selected_slot_items += len(selected)
        if reasons:
            self.v6_recovery_decisions += 1
            self.v6_recovery_reason_counts.update(reasons)
        else:
            self.v6_primary_decisions += 1

    def runtime_status_snapshot(self) -> v5.RuntimeStatus:
        status = super().runtime_status_snapshot()
        status.gating_decisions = self.v6_gating_decisions
        status.gating_primary_decisions = self.v6_primary_decisions
        status.gating_recovery_decisions = self.v6_recovery_decisions
        status.gating_selected_slot_items = self.v6_selected_slot_items
        status.gating_avg_slots_per_decision = (
            self.v6_selected_slot_items / self.v6_gating_decisions
            if self.v6_gating_decisions
            else 0.0
        )
        status.gating_recovery_reasons = ",".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.v6_recovery_reason_counts.items())
        )
        return status

    def status_lines(self) -> List[str]:
        lines = super().status_lines()
        status = self.runtime_status_snapshot()
        for index, line in enumerate(lines):
            if line.startswith("Mode "):
                lines[index] = (
                    f"Mode {V6_EXECUTION_MODE} | Eval calls {status.evaluator_calls} | "
                    f"max batch {status.max_evaluator_batch}"
                )
                break
        lines.append(
            "Gating "
            f"{status.gating_primary_decisions}/{status.gating_decisions} primary | "
            f"{status.gating_recovery_decisions} recovery | "
            f"{status.gating_avg_slots_per_decision:.2f} slots/decision"
        )
        return lines


def add_v6_runtime_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("V6 gated SOT memory")
    group.add_argument("--v6-primary-slots-per-track", type=int, default=DEFAULT_V6_PRIMARY_SLOTS_PER_TRACK)
    group.add_argument("--v6-recovery-slots-per-track", type=int, default=DEFAULT_V6_RECOVERY_SLOTS_PER_TRACK)
    group.add_argument("--v6-recovery-interval", type=int, default=DEFAULT_V6_RECOVERY_INTERVAL)
    group.add_argument("--v6-recovery-min-confidence", type=float, default=DEFAULT_V6_RECOVERY_MIN_CONFIDENCE)
    group.add_argument("--v6-recovery-min-assignment-score", type=float, default=DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_SCORE)
    group.add_argument("--v6-recovery-min-assignment-margin", type=float, default=DEFAULT_V6_RECOVERY_MIN_ASSIGNMENT_MARGIN)
    group.add_argument("--v6-recovery-stale-slot-frames", type=int, default=DEFAULT_V6_RECOVERY_STALE_SLOT_FRAMES)


def create_backend(args: argparse.Namespace, source, expected_tracks: int = 0):
    weight_path = args.weight_path or v5.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return LoRATGatedMultiObjectTracker(
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
        v6_primary_slots_per_track=args.v6_primary_slots_per_track,
        v6_recovery_slots_per_track=args.v6_recovery_slots_per_track,
        v6_recovery_interval=args.v6_recovery_interval,
        v6_recovery_min_confidence=args.v6_recovery_min_confidence,
        v6_recovery_min_assignment_score=args.v6_recovery_min_assignment_score,
        v6_recovery_min_assignment_margin=args.v6_recovery_min_assignment_margin,
        v6_recovery_stale_slot_frames=args.v6_recovery_stale_slot_frames,
    )
