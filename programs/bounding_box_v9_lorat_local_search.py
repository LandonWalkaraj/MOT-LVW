from __future__ import annotations

import sys
import time
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import bounding_box_v9_runtime_base as runtime
import mot_common as mot

BBox = mot.BBox

V9_EXECUTION_MODE = "shared-frame-vit-batched-local-search-roi-heads"
# Compatibility alias for older benchmark/status helpers. New V9 code should use
# V9_EXECUTION_MODE.
V8_EXECUTION_MODE = V9_EXECUTION_MODE
DEFAULT_V9_LOCAL_GRID_SIZE = 17
V9_DIAGNOSTIC_MODES = ("normal", "gt_window", "gt_identity")
DEFAULT_V9_SCALE_GATE_MIN_RATIO = 0.20
DEFAULT_V9_SCALE_GATE_MAX_RATIO = 3.00
DEFAULT_V9_SCALE_GATE_HARD_MIN_RATIO = 0.10
DEFAULT_V9_SCALE_GATE_HARD_MAX_RATIO = 4.50
DEFAULT_V9_SCALE_GATE_SOFT_SWITCH_MARGIN = 0.05
DEFAULT_V9_SCALE_GATE_OVERRIDE_CONFIDENCE = 0.92
DEFAULT_V9_SCALE_GATE_FALLBACK_CONFIDENCE_SCALE = 0.65
DEFAULT_V9_PROTECTIVE_REID = True
DEFAULT_V9_PROTECTIVE_REID_CONFIDENCE_GATE = 0.58
DEFAULT_V9_PROTECTIVE_REID_MARGIN_GATE = 0.018
DEFAULT_V9_PROTECTIVE_REID_OVERLAP_IOU = 0.10
DEFAULT_V9_ACCEPT_MAX_CENTER_RATIO = 2.40
DEFAULT_V9_ACCEPT_MAX_HEALTHY_CENTER_RATIO = 1.45
DEFAULT_V9_LOCAL_HOLD_MIN_CONFIDENCE = 0.28
DEFAULT_V9_LOCAL_HOLD_MAX_CENTER_RATIO = 1.90
DEFAULT_V9_LOCAL_HOLD_MAX_LOST_CENTER_RATIO = 3.00
DEFAULT_V9_STAGE1_LOCAL_MIN_CONFIDENCE = 0.55
DEFAULT_V9_STAGE1_LOCAL_MIN_MARGIN = 0.012
DEFAULT_V9_STAGE1_LOCAL_MIN_MOTION = 0.30
DEFAULT_V9_STAGE1_LOCAL_MIN_PATH = 0.30
DEFAULT_V9_PROBABLY_HEALTHY_MIN_CONFIDENCE = 0.70
DEFAULT_V9_PROBABLY_HEALTHY_MIN_MARGIN = 0.006
DEFAULT_V9_PROBABLY_HEALTHY_MIN_CONTINUITY = 0.50
DEFAULT_V9_NEXT_BEST_MIN_CONFIDENCE = 0.0
DEFAULT_V9_NEXT_BEST_MAX_CANDIDATES = 4
DEFAULT_V9_LOCAL_RESCUE = True
DEFAULT_V9_LOCAL_RESCUE_MIN_ASSIGNMENT_SCORE = 0.22
DEFAULT_V9_LOCAL_RESCUE_MAX_SCALE_ERROR = 8.00
DEFAULT_V9_LOCAL_RESCUE_MIN_ANCHOR = 0.36
DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH = 0.24
DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_WIDTHS = 0.80
DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_HEIGHTS = 0.80
DEFAULT_V9_CONTINUITY_ENABLED = True
DEFAULT_V9_CONTINUITY_TOPK = 5
DEFAULT_V9_CONTINUITY_MIN_SCORE = 0.35
DEFAULT_V9_CONTINUITY_MIN_MARGIN = 0.025
DEFAULT_V9_CONTINUITY_HEALTHY_MIN_SCORE = 0.48
DEFAULT_V9_CONTINUITY_HEAD_WEIGHT = 0.11
DEFAULT_V9_CONTINUITY_MARGIN_WEIGHT = 0.05
DEFAULT_V9_CONTINUITY_MOTION_WEIGHT = 0.27
DEFAULT_V9_CONTINUITY_PATH_WEIGHT = 0.25
DEFAULT_V9_CONTINUITY_ANCHOR_WEIGHT = 0.25
DEFAULT_V9_CONTINUITY_APPEARANCE_WEIGHT = 0.06
DEFAULT_V9_CONTINUITY_SCALE_WEIGHT = 0.04
DEFAULT_V9_CONTINUITY_VISIBILITY_WEIGHT = 0.10
DEFAULT_V9_CONTINUITY_OTHER_ANCHOR_PENALTY = 0.42
DEFAULT_V9_CONTINUITY_NEGATIVE_ANCHOR_PENALTY = 0.32
DEFAULT_V9_CONTINUITY_CENTER_JUMP_PENALTY = 0.20
DEFAULT_V9_CONTINUITY_LOCAL_REJECT_PENALTY = 0.20
DEFAULT_V9_CONTINUITY_DRIFT_RISK_MIN_CONFIDENCE = 0.70
DEFAULT_V9_CONTINUITY_DRIFT_RISK_MAX_SCORE = 0.34
DEFAULT_V9_CONTINUITY_DRIFT_RISK_MIN_IDENTITY_RISK = 0.18
DEFAULT_V9_TRUSTED_ANCHOR_MIN_MARGIN = 0.030
DEFAULT_V9_TRUSTED_ANCHOR_HIGH_CONFIDENCE = 0.85
DEFAULT_V9_TRUSTED_ANCHOR_SUSPECT_FRAMES = 4
DEFAULT_V9_VISIBILITY_LOW = 0.32
DEFAULT_V9_VISIBILITY_MIN_HEALTHY = 0.42
DEFAULT_V9_VISIBILITY_ABSENT_PENALTY = 0.18
DEFAULT_V9_SMALL_TARGET_MODE = runtime.DEFAULT_V9_BASE_SMALL_TARGET_MODE
DEFAULT_V9_SMALL_TARGET_AREA = runtime.DEFAULT_V9_BASE_SMALL_TARGET_AREA
DEFAULT_V9_SMALL_TARGET_MAX_SIDE = runtime.DEFAULT_V9_BASE_SMALL_TARGET_MAX_SIDE
DEFAULT_V9_SMALL_TARGET_MAX_SCALE_CHANGE = runtime.DEFAULT_V9_BASE_SMALL_TARGET_MAX_SCALE_CHANGE
DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_SCORE = runtime.DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_SCORE
DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_MOTION = runtime.DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_MOTION
DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_PATH = runtime.DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_PATH
DEFAULT_V9_SMALL_TARGET_CONFIDENCE_FLOOR = runtime.DEFAULT_V9_BASE_SMALL_TARGET_CONFIDENCE_FLOOR
DEFAULT_V9_WINDOW_PENALTY_RATIO = runtime.DEFAULT_V9_BASE_WINDOW_PENALTY_RATIO


def __getattr__(name: str):
    # Compatibility surface for the benchmark harness. The fallback is now a
    # V9-owned runtime snapshot, not the live V8 file.
    return getattr(runtime, name)


@dataclass(frozen=True)
class V9LocalHeadOutput:
    score_maps: object
    box_delta_maps: object
    visibility_maps: object
    search_windows: Tuple[BBox, ...]
    elapsed_seconds: float
    selected_head_count: int


class V9LocalSearchHead(runtime.BatchedObjectConditionedHead):
    """Object-conditioned head evaluated on per-track local search grids.

    Earlier full-frame heads scored every object over the whole shared feature
    map, then masked candidate extraction to a local ROI. V9 moves the local
    search window before the head: shared frame features are sampled into a
    fixed-size local grid for each track, and the batched head predicts boxes in
    that local coordinate system. This moves us back toward LoRaT's
    template/search geometry while preserving one shared frame ViT pass.
    """

    def _build_module(self):
        torch_module = self.torch
        nn = torch_module.nn
        embed_dim = self.embed_dim
        hidden_dim = self.hidden_dim
        lora_rank = self.lora_rank
        box_delta_scale = self.box_delta_scale

        class V9LocalObjectConditionedLoRAHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_norm = nn.LayerNorm(embed_dim)
                self.template_norm = nn.LayerNorm(embed_dim)
                self.object_norm = nn.LayerNorm(embed_dim)
                self.template_token_type = nn.Parameter(torch_module.empty(2, embed_dim))
                self.template_context = nn.Linear(embed_dim, embed_dim)
                self.template_gate = nn.Linear(embed_dim, embed_dim)
                self.fusion_norm = nn.LayerNorm(embed_dim)
                self.base_projection = nn.Linear(embed_dim, hidden_dim)
                self.lora_down = nn.Linear(embed_dim, lora_rank, bias=False)
                self.lora_up_generator = nn.Linear(embed_dim, lora_rank * hidden_dim)
                self.object_bias = nn.Linear(embed_dim, hidden_dim)
                self.activation = nn.GELU()
                self.score_head = nn.Linear(hidden_dim, 1)
                self.box_head = nn.Linear(hidden_dim, 4)
                self.visibility_head = nn.Linear(hidden_dim, 1)
                self.reid_projection = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                self.lora_scale = 1.0 / float(lora_rank)
                self.box_delta_scale = float(box_delta_scale)
                nn.init.zeros_(self.score_head.bias)
                nn.init.zeros_(self.box_head.weight)
                nn.init.zeros_(self.box_head.bias)
                nn.init.zeros_(self.visibility_head.weight)
                nn.init.zeros_(self.visibility_head.bias)
                nn.init.normal_(self.template_token_type, std=0.02)

            def forward(
                self,
                feature_tokens,
                object_embeddings,
                template_tokens=None,
                template_mask=None,
                template_foreground_mask=None,
            ):
                # feature_tokens: [batch, local_tokens, embed_dim]
                feature_tokens = self.feature_norm(feature_tokens)
                object_embeddings = self.object_norm(object_embeddings)

                if template_tokens is not None and template_mask is not None:
                    template_tokens = self.template_norm(template_tokens)
                    if template_foreground_mask is None:
                        template_foreground_mask = template_mask
                    token_type_ids = template_foreground_mask.to(torch_module.long).clamp(0, 1)
                    template_tokens = template_tokens + self.template_token_type[token_type_ids].to(template_tokens.dtype)
                    token_similarity = torch_module.einsum("bld,btd->blt", feature_tokens, template_tokens) / (embed_dim ** 0.5)
                    token_similarity = token_similarity.masked_fill(~template_mask[:, None, :], -1.0e4)
                    attention = torch_module.softmax(token_similarity, dim=-1)
                    template_context = torch_module.einsum("blt,btd->bld", attention, template_tokens)
                    summary_mask = template_foreground_mask & template_mask
                    empty_summary = summary_mask.sum(dim=1, keepdim=True) == 0
                    summary_mask = torch_module.where(empty_summary, template_mask, summary_mask)
                    template_summary = (template_tokens * summary_mask[:, :, None].to(template_tokens.dtype)).sum(dim=1)
                    template_count = summary_mask.sum(dim=1, keepdim=True).clamp_min(1).to(template_tokens.dtype)
                    template_summary = template_summary / template_count
                    gate = torch_module.sigmoid(self.template_gate(template_summary))[:, None, :]
                    conditioned_features = self.fusion_norm(
                        feature_tokens + gate * self.template_context(template_context)
                    )
                else:
                    conditioned_features = feature_tokens

                base = self.base_projection(conditioned_features)
                down = self.lora_down(conditioned_features)
                up = self.lora_up_generator(object_embeddings).view(
                    object_embeddings.shape[0],
                    lora_rank,
                    hidden_dim,
                )
                lora_delta = torch_module.einsum("blr,brh->blh", down, up) * self.lora_scale
                object_bias = self.object_bias(object_embeddings)[:, None, :]
                hidden = self.activation(base + object_bias + lora_delta)
                score_logits = self.score_head(hidden).squeeze(-1)
                box_deltas = self.box_head(hidden)
                visibility_logits = self.visibility_head(hidden).squeeze(-1)
                return score_logits, box_deltas, visibility_logits

            def project_reid(self, embeddings):
                return torch_module.nn.functional.normalize(self.reid_projection(embeddings), dim=-1)

        return V9LocalObjectConditionedLoRAHead()

    def _reduce_head_scalar_maps(self, per_head_values, head_mask):
        if per_head_values.shape[1] == 1:
            return per_head_values[:, 0, :]
        valid = head_mask[:, :, None]
        if self.score_reduction == "mean":
            weights = valid.to(per_head_values.dtype)
            counts = weights.sum(dim=1).clamp_min(1.0)
            return (per_head_values * weights).sum(dim=1) / counts
        masked_values = self.torch.where(
            valid,
            per_head_values,
            self.torch.full_like(per_head_values, -float("inf")),
        )
        return self.torch.max(masked_values, dim=1).values

    def score_local(self, local_feature_grids, selected_banks: Sequence[Sequence[object]], search_windows: Sequence[BBox]) -> V9LocalHeadOutput:
        started = time.perf_counter()
        context = nullcontext() if self.module.training else self.torch.inference_mode()
        with context:
            object_count = int(local_feature_grids.shape[0])
            grid_height = int(local_feature_grids.shape[1])
            grid_width = int(local_feature_grids.shape[2])
            local_tokens = local_feature_grids.reshape(object_count, grid_height * grid_width, self.embed_dim)
            selected_head_count = self._effective_selected_head_count(selected_banks)
            head_tensor, head_mask, template_tensor, template_mask, template_foreground_mask = self._build_head_tensor(selected_banks)
            max_heads = int(head_tensor.shape[1])

            if not self.weights_loaded and not self.module.training:
                self.last_mode = "v9_local_zero_shot_similarity"
                normalized_features = self.F.normalize(local_tokens.to(self.torch.float32), dim=-1)
                per_head_scores = self.torch.einsum("nhd,nld->nhl", head_tensor, normalized_features) * 10.0
                per_head_visibility = per_head_scores
                per_head_deltas = self.torch.zeros(
                    (object_count, max_heads, grid_height * grid_width, 4),
                    device=self.device,
                    dtype=self.torch.float32,
                )
            else:
                self.last_mode = "v9_local_template_patch_lora_conditioned"
                flat_features = local_tokens[:, None, :, :].expand(-1, max_heads, -1, -1).reshape(
                    object_count * max_heads,
                    grid_height * grid_width,
                    self.embed_dim,
                )
                flat_head_tensor = head_tensor.reshape(-1, self.embed_dim)
                flat_template_tensor = template_tensor.reshape(-1, template_tensor.shape[2], self.embed_dim)
                flat_template_mask = template_mask.reshape(-1, template_mask.shape[2])
                flat_foreground_mask = template_foreground_mask.reshape(-1, template_foreground_mask.shape[2])
                per_head_scores, per_head_deltas, per_head_visibility = self.module(
                    flat_features,
                    flat_head_tensor,
                    flat_template_tensor,
                    flat_template_mask,
                    flat_foreground_mask,
                )
                per_head_scores = per_head_scores.reshape(object_count, max_heads, grid_height * grid_width)
                per_head_deltas = per_head_deltas.reshape(object_count, max_heads, grid_height * grid_width, 4)
                per_head_visibility = per_head_visibility.reshape(object_count, max_heads, grid_height * grid_width)

            if object_count <= 0:
                score_maps = self.torch.zeros((0, grid_height, grid_width), device=self.device, dtype=self.torch.float32)
                box_delta_maps = self.torch.zeros((0, grid_height, grid_width, 4), device=self.device, dtype=self.torch.float32)
                visibility_maps = self.torch.zeros((0, grid_height, grid_width), device=self.device, dtype=self.torch.float32)
            else:
                score_logits, box_deltas = self._reduce_head_scores_and_deltas(per_head_scores, per_head_deltas, head_mask)
                visibility_logits = self._reduce_head_scalar_maps(per_head_visibility, head_mask)
                score_maps = score_logits.reshape(object_count, grid_height, grid_width)
                box_delta_maps = box_deltas.reshape(object_count, grid_height, grid_width, 4)
                visibility_maps = visibility_logits.reshape(object_count, grid_height, grid_width)

        return V9LocalHeadOutput(
            score_maps=score_maps,
            box_delta_maps=box_delta_maps,
            visibility_maps=visibility_maps,
            search_windows=tuple(search_windows),
            elapsed_seconds=time.perf_counter() - started,
            selected_head_count=selected_head_count,
        )


class V9LocalSearchLoRATTracker(runtime.QualityBatchedLoRATTracker):
    """V9 tracker: shared runtime lifecycle plus LoRaT-style local search grids.

    The expensive frame encoder remains shared. The per-object work is a batched
    local-search head over fixed-size feature grids sampled from that shared map.
    """

    backend_name = "LoRAT-v9-local-search-roi"

    def __init__(
        self,
        *args,
        v9_local_grid_size: int = DEFAULT_V9_LOCAL_GRID_SIZE,
        v9_diagnostic_mode: str = "normal",
        v9_scale_gate_min_ratio: float = DEFAULT_V9_SCALE_GATE_MIN_RATIO,
        v9_scale_gate_max_ratio: float = DEFAULT_V9_SCALE_GATE_MAX_RATIO,
        v9_scale_gate_override_confidence: float = DEFAULT_V9_SCALE_GATE_OVERRIDE_CONFIDENCE,
        v9_scale_gate_fallback_confidence_scale: float = DEFAULT_V9_SCALE_GATE_FALLBACK_CONFIDENCE_SCALE,
        v9_protective_reid: bool = DEFAULT_V9_PROTECTIVE_REID,
        v9_protective_reid_confidence_gate: float = DEFAULT_V9_PROTECTIVE_REID_CONFIDENCE_GATE,
        v9_protective_reid_margin_gate: float = DEFAULT_V9_PROTECTIVE_REID_MARGIN_GATE,
        v9_protective_reid_overlap_iou: float = DEFAULT_V9_PROTECTIVE_REID_OVERLAP_IOU,
        v9_accept_max_center_ratio: float = DEFAULT_V9_ACCEPT_MAX_CENTER_RATIO,
        v9_accept_max_healthy_center_ratio: float = DEFAULT_V9_ACCEPT_MAX_HEALTHY_CENTER_RATIO,
        v9_local_hold_min_confidence: float = DEFAULT_V9_LOCAL_HOLD_MIN_CONFIDENCE,
        v9_local_hold_max_center_ratio: float = DEFAULT_V9_LOCAL_HOLD_MAX_CENTER_RATIO,
        v9_local_hold_max_lost_center_ratio: float = DEFAULT_V9_LOCAL_HOLD_MAX_LOST_CENTER_RATIO,
        v9_stage1_local_min_confidence: float = DEFAULT_V9_STAGE1_LOCAL_MIN_CONFIDENCE,
        v9_stage1_local_min_margin: float = DEFAULT_V9_STAGE1_LOCAL_MIN_MARGIN,
        v9_stage1_local_min_motion: float = DEFAULT_V9_STAGE1_LOCAL_MIN_MOTION,
        v9_stage1_local_min_path: float = DEFAULT_V9_STAGE1_LOCAL_MIN_PATH,
        v9_next_best_min_confidence: float = DEFAULT_V9_NEXT_BEST_MIN_CONFIDENCE,
        v9_next_best_max_candidates: int = DEFAULT_V9_NEXT_BEST_MAX_CANDIDATES,
        v9_local_rescue: bool = DEFAULT_V9_LOCAL_RESCUE,
        v9_local_rescue_min_assignment_score: float = DEFAULT_V9_LOCAL_RESCUE_MIN_ASSIGNMENT_SCORE,
        v9_local_rescue_max_scale_error: float = DEFAULT_V9_LOCAL_RESCUE_MAX_SCALE_ERROR,
        v9_continuity_enabled: bool = DEFAULT_V9_CONTINUITY_ENABLED,
        v9_continuity_topk: int = DEFAULT_V9_CONTINUITY_TOPK,
        v9_continuity_min_score: float = DEFAULT_V9_CONTINUITY_MIN_SCORE,
        v9_continuity_min_margin: float = DEFAULT_V9_CONTINUITY_MIN_MARGIN,
        **kwargs,
    ) -> None:
        self.v9_local_grid_size = max(4, int(v9_local_grid_size))
        self.v9_diagnostic_mode = str(v9_diagnostic_mode or "normal").strip().lower()
        if self.v9_diagnostic_mode not in V9_DIAGNOSTIC_MODES:
            self.v9_diagnostic_mode = "normal"
        self.v9_scale_gate_min_ratio = max(0.01, float(v9_scale_gate_min_ratio))
        self.v9_scale_gate_max_ratio = max(self.v9_scale_gate_min_ratio, float(v9_scale_gate_max_ratio))
        self.v9_scale_gate_override_confidence = max(0.0, min(1.0, float(v9_scale_gate_override_confidence)))
        self.v9_scale_gate_fallback_confidence_scale = max(
            0.0,
            min(1.0, float(v9_scale_gate_fallback_confidence_scale)),
        )
        self.v9_protective_reid = bool(v9_protective_reid)
        self.v9_protective_reid_confidence_gate = max(0.0, min(1.0, float(v9_protective_reid_confidence_gate)))
        self.v9_protective_reid_margin_gate = max(0.0, float(v9_protective_reid_margin_gate))
        self.v9_protective_reid_overlap_iou = max(0.0, min(1.0, float(v9_protective_reid_overlap_iou)))
        self.v9_accept_max_center_ratio = max(0.25, float(v9_accept_max_center_ratio))
        self.v9_accept_max_healthy_center_ratio = max(0.25, float(v9_accept_max_healthy_center_ratio))
        self.v9_local_hold_min_confidence = max(0.0, min(1.0, float(v9_local_hold_min_confidence)))
        self.v9_local_hold_max_center_ratio = max(0.25, float(v9_local_hold_max_center_ratio))
        self.v9_local_hold_max_lost_center_ratio = max(
            self.v9_local_hold_max_center_ratio,
            float(v9_local_hold_max_lost_center_ratio),
        )
        self.v9_stage1_local_min_confidence = max(0.0, min(1.0, float(v9_stage1_local_min_confidence)))
        self.v9_stage1_local_min_margin = max(0.0, float(v9_stage1_local_min_margin))
        self.v9_stage1_local_min_motion = max(0.0, min(1.0, float(v9_stage1_local_min_motion)))
        self.v9_stage1_local_min_path = max(0.0, min(1.0, float(v9_stage1_local_min_path)))
        self.v9_next_best_min_confidence = max(0.0, min(1.0, float(v9_next_best_min_confidence)))
        self.v9_next_best_max_candidates = max(0, int(v9_next_best_max_candidates))
        self.v9_local_rescue = bool(v9_local_rescue)
        self.v9_local_rescue_min_assignment_score = max(0.0, float(v9_local_rescue_min_assignment_score))
        self.v9_local_rescue_max_scale_error = max(1.0, float(v9_local_rescue_max_scale_error))
        self.v9_continuity_enabled = bool(v9_continuity_enabled)
        self.v9_continuity_topk = max(1, int(v9_continuity_topk))
        self.v9_continuity_min_score = max(0.0, min(1.0, float(v9_continuity_min_score)))
        self.v9_continuity_min_margin = max(0.0, float(v9_continuity_min_margin))
        self._v9_diagnostic_gt_boxes: Dict[int, BBox] = {}
        self._last_v9_search_windows_by_track_id: Dict[int, BBox] = {}
        super().__init__(*args, **kwargs)
        self.object_conditioned_head = V9LocalSearchHead(
            self.torch,
            self.F,
            self.device,
            self.embed_dim,
            self.head_rank,
            self.score_reduction,
            self.head_hidden_dim,
            self.head_lora_rank,
            weight_path=self.head_weight_path,
        )

    def set_v9_diagnostic_gt_boxes(self, gt_boxes_by_track_id: Dict[int, BBox]) -> None:
        """Benchmark-only hook used by V9 diagnostic modes.

        Production/runtime tracking should leave this empty. The benchmark uses it
        to isolate whether failures come from search-window propagation or from
        the local head/identity logic after the target is inside the window.
        """

        self._v9_diagnostic_gt_boxes = {
            int(track_id): mot.clamp_bbox_size(bbox)
            for track_id, bbox in dict(gt_boxes_by_track_id or {}).items()
            if bbox is not None
        }

    @staticmethod
    def _v9_bbox_center_inside(container: BBox, target: BBox) -> bool:
        center_x, center_y = mot.bbox_center(target)
        x, y, width, height = mot.clamp_bbox_size(container)
        return x <= center_x <= x + width and y <= center_y <= y + height

    @staticmethod
    def _v9_center_ratio(reference: BBox, candidate: BBox) -> float:
        ref_x, ref_y = mot.bbox_center(reference)
        cand_x, cand_y = mot.bbox_center(candidate)
        distance = float(np.hypot(cand_x - ref_x, cand_y - ref_y))
        return distance / max(1.0, mot.bbox_diagonal(reference))

    @staticmethod
    def _v9_track_uncertain(track: mot.TrackState) -> bool:
        state = str(getattr(track, "state", "") or "").upper()
        return (
            not bool(getattr(track, "ok", True))
            or int(getattr(track, "lost_frames", 0) or 0) > 0
            or "LOST" in state
            or "MISS" in state
            or "UNCERTAIN" in state
            or "OCCLU" in state
        )

    @staticmethod
    def _v9_track_last_accepted_bbox(track: mot.TrackState) -> Optional[BBox]:
        bbox = getattr(track, "v9_last_accepted_bbox", None)
        if bbox is None:
            bbox = getattr(track, "last_reliable_bbox", None)
        return None if bbox is None else mot.clamp_bbox_size(bbox)

    @staticmethod
    def _v9_track_trusted_anchor_bbox(track: mot.TrackState) -> Optional[BBox]:
        bbox = getattr(track, "v9_last_trusted_anchor_bbox", None)
        if bbox is None:
            bbox = getattr(track, "v9_last_accepted_bbox", None)
        if bbox is None:
            bbox = getattr(track, "last_reliable_bbox", None)
        return None if bbox is None else mot.clamp_bbox_size(bbox)

    @staticmethod
    def _v9_translate_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
        x, y, width, height = mot.clamp_bbox_size(bbox)
        return mot.clamp_bbox_size((x + float(dx), y + float(dy), width, height))

    @staticmethod
    def _v9_anchor_suspect_frames(track: mot.TrackState) -> int:
        return max(0, int(getattr(track, "v9_anchor_suspect_frames", 0) or 0))

    @staticmethod
    def _v9_is_reference_size_lock_state(scale_gate_state: str) -> bool:
        return str(scale_gate_state or "").startswith("reference_size_lock")

    def _v9_is_scale_modified_state(self, scale_gate_state: str) -> bool:
        state = str(scale_gate_state or "")
        return (
            state == "override_high_confidence"
            or self._v9_is_reference_size_lock_state(state)
            or state == "soft_scale_violation_locked_candidate"
        )

    def _v9_candidate_overlaps_other_tracks(
        self,
        track: mot.TrackState,
        bbox: BBox,
        tracks: Sequence[mot.TrackState],
    ) -> bool:
        for other in tracks:
            if other.track_id == track.track_id:
                continue
            if mot.bbox_iou(bbox, other.bbox) >= self.v9_protective_reid_overlap_iou:
                return True
        return False

    def _v9_candidate_local_reject_state(
        self,
        track: mot.TrackState,
        candidate: BBox,
        predicted: BBox,
        search_window: BBox,
        confidence: float,
        scale_gate_state: str,
        *,
        allow_lost_wide: bool = False,
    ) -> Optional[str]:
        candidate = mot.clamp_bbox_size(candidate)
        if not self._v9_bbox_center_inside(search_window, candidate):
            return "V9_OUTSIDE_SEARCH_WINDOW"
        if scale_gate_state == "override_high_confidence" and confidence < self.v9_scale_gate_override_confidence:
            return "V9_SCALE_OVERRIDE_LOWCONF"
        if self._v9_is_reference_size_lock_state(scale_gate_state) and confidence < max(
            self.min_confidence,
            self.v9_local_hold_min_confidence,
        ):
            return "V9_SCALE_LOCK_LOWCONF"

        reference = predicted if allow_lost_wide or self._v9_track_uncertain(track) else track.bbox
        center_ratio = self._v9_center_ratio(reference, candidate)
        if self._v9_track_uncertain(track) or allow_lost_wide:
            max_ratio = self.v9_accept_max_center_ratio
        else:
            max_ratio = self.v9_accept_max_healthy_center_ratio
        if center_ratio > max_ratio and confidence < self.v9_scale_gate_override_confidence:
            return "V9_CENTER_JUMP"
        return None

    def _v9_should_attach_crop_reid(
        self,
        track: mot.TrackState,
        candidate: BBox,
        confidence: float,
        margin: float,
        scale_gate_state: str,
        candidate_source: str,
        tracks: Sequence[mot.TrackState],
        visibility: Optional[float] = None,
    ) -> bool:
        if not self.v9_protective_reid:
            return True
        if self._v9_track_uncertain(track):
            return True
        if visibility is not None and float(visibility) < DEFAULT_V9_VISIBILITY_LOW:
            return True
        if confidence < self.v9_protective_reid_confidence_gate:
            return True
        if margin < self.v9_protective_reid_margin_gate:
            return True
        if scale_gate_state not in {"pass", "switched_topk"}:
            return True
        if "template" in str(candidate_source) or "reid" in str(candidate_source):
            return True
        return self._v9_candidate_overlaps_other_tracks(track, candidate, tracks)

    def _v9_local_candidate_health(
        self,
        track: mot.TrackState,
        record: Optional[Dict[str, object]],
        tracks: Sequence[mot.TrackState],
    ) -> Tuple[bool, str]:
        """Return whether the owned local candidate is safe enough to avoid ReID steering."""

        details: Dict[str, object] = {
            "v9_local_health_tier": "unhealthy",
            "v9_local_health_confidence_threshold": max(
                self.v9_stage1_local_min_confidence,
                self.min_confidence,
                self.lorat_accept_min_score,
            ),
            "v9_local_health_margin_threshold": self.v9_stage1_local_min_margin,
            "v9_local_health_motion": None,
            "v9_local_health_accepted_anchor_motion": None,
            "v9_local_health_path": None,
            "v9_local_health_continuity_score": 0.0,
            "v9_local_health_identity_risk": 0.0,
            "v9_local_health_visibility": None,
        }
        if record is not None:
            record["_v9_local_health_details"] = details

        if not self.v9_protective_reid:
            return False, "protective_reid_disabled"
        if record is None:
            return False, "no_owned_candidate"
        if self._v9_track_uncertain(track):
            return False, "track_uncertain"

        confidence = float(record.get("confidence") or 0.0)
        margin = float(record.get("margin") or 0.0)
        visibility = record.get("candidate_visibility")
        if visibility is None:
            visibility = record.get("head_visibility")
        visibility_score = 0.5 if visibility is None else max(0.0, min(1.0, float(visibility)))
        details["v9_local_health_visibility"] = visibility_score
        if visibility_score < DEFAULT_V9_VISIBILITY_LOW:
            return False, "local_visibility_low"
        healthy_confidence_threshold = max(
            self.v9_stage1_local_min_confidence,
            self.min_confidence,
            self.lorat_accept_min_score,
        )
        healthy_margin_threshold = self.v9_stage1_local_min_margin
        details["v9_local_health_confidence_threshold"] = healthy_confidence_threshold
        details["v9_local_health_margin_threshold"] = healthy_margin_threshold

        scale_gate_info = dict(record.get("scale_gate_info") or {})
        scale_gate_state = str(scale_gate_info.get("v9_scale_gate_state", "pass"))
        if scale_gate_state not in {"pass", "switched_topk"}:
            continuity_score = float(record.get("v9_continuity_score") or record.get("v9_continuity_best_score") or 0.0)
            continuity_reject = str(record.get("v9_continuity_local_reject") or "")
            details["v9_local_health_continuity_score"] = continuity_score
            if not (
                scale_gate_state
                in {
                    "reference_size_lock_high_confidence",
                    "soft_scale_violation_original_pending",
                    "soft_scale_violation_original_kept",
                    "soft_scale_violation_locked_candidate",
                }
                and continuity_score >= self.v9_continuity_min_score
                and not continuity_reject
            ):
                return False, f"scale_gate_{scale_gate_state}"

        candidate = record.get("candidate")
        predicted = record.get("predicted")
        search_window = record.get("search_window")
        if candidate is None or predicted is None or search_window is None:
            return False, "missing_geometry"
        local_reject = self._v9_candidate_local_reject_state(
            track,
            candidate,  # type: ignore[arg-type]
            predicted,  # type: ignore[arg-type]
            search_window,  # type: ignore[arg-type]
            confidence,
            scale_gate_state,
        )
        if local_reject is not None:
            return False, local_reject

        if self._v9_candidate_overlaps_other_tracks(track, candidate, tracks):  # type: ignore[arg-type]
            return False, "candidate_conflicted"

        last_accepted = self._v9_track_last_accepted_bbox(track)
        reference_bbox = last_accepted if last_accepted is not None else track.bbox
        reference_diagonal = max(1.0, mot.bbox_diagonal(reference_bbox))
        motion = mot.motion_affinity(predicted, candidate, reference_diagonal)  # type: ignore[arg-type]
        accepted_anchor_motion = motion
        if last_accepted is not None:
            accepted_anchor_motion = mot.motion_affinity(last_accepted, candidate, reference_diagonal)  # type: ignore[arg-type]
            motion = max(0.0, min(1.0, (0.65 * motion) + (0.35 * accepted_anchor_motion)))
        path = mot.center_path_affinity(track, candidate)  # type: ignore[arg-type]
        continuity_score = float(record.get("v9_continuity_score") or record.get("v9_continuity_best_score") or 0.0)
        identity_risk = float(record.get("v9_continuity_identity_risk") or 0.0)
        if bool(record.get("v9_continuity_drift_risk")):
            details["v9_local_health_identity_risk"] = identity_risk
            details["v9_local_health_continuity_score"] = continuity_score
            return False, "local_drift_risk"
        details["v9_local_health_motion"] = motion
        details["v9_local_health_accepted_anchor_motion"] = accepted_anchor_motion
        details["v9_local_health_path"] = path
        details["v9_local_health_continuity_score"] = continuity_score
        details["v9_local_health_identity_risk"] = identity_risk
        if identity_risk > 0.12:
            return False, "local_identity_risk"
        path_ready = mot.path_gate_ready(track)

        healthy = (
            confidence >= healthy_confidence_threshold
            and margin >= healthy_margin_threshold
            and motion >= self.v9_stage1_local_min_motion
            and (not path_ready or path >= self.v9_stage1_local_min_path)
            and visibility_score >= DEFAULT_V9_VISIBILITY_MIN_HEALTHY
        )
        continuity_healthy = (
            continuity_score >= DEFAULT_V9_CONTINUITY_HEALTHY_MIN_SCORE
            and margin >= DEFAULT_V9_PROBABLY_HEALTHY_MIN_MARGIN
            and motion >= DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH
            and accepted_anchor_motion >= DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH
            and (not path_ready or path >= DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH)
            and visibility_score >= DEFAULT_V9_VISIBILITY_LOW
        )
        probably_healthy = (
            confidence >= DEFAULT_V9_PROBABLY_HEALTHY_MIN_CONFIDENCE
            and margin >= DEFAULT_V9_PROBABLY_HEALTHY_MIN_MARGIN
            and continuity_score >= DEFAULT_V9_PROBABLY_HEALTHY_MIN_CONTINUITY
            and visibility_score >= DEFAULT_V9_VISIBILITY_MIN_HEALTHY
        )
        if healthy:
            details["v9_local_health_tier"] = "healthy"
            return True, "healthy_local"
        if continuity_healthy:
            details["v9_local_health_tier"] = "continuity_healthy"
            return True, "continuity_healthy_local"
        if probably_healthy:
            details["v9_local_health_tier"] = "probably_healthy"
            return True, "probably_healthy_local"
        if confidence < healthy_confidence_threshold:
            return False, "local_confidence_low"
        if margin < healthy_margin_threshold:
            return False, "local_margin_low"
        if motion < self.v9_stage1_local_min_motion:
            return False, "local_motion_low"
        if path_ready and path < self.v9_stage1_local_min_path:
            return False, "local_path_low"
        if visibility_score < DEFAULT_V9_VISIBILITY_MIN_HEALTHY:
            return False, "local_visibility_borderline"
        return False, "local_continuity_low"

    def _v9_identity_assignment_for_output(
        self,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        margin: float,
    ) -> Optional[mot.IdentityAssignment]:
        if self.identity_arbitrator is None or not self.identity_arbitrator.enabled:
            return None
        score_started = time.perf_counter()
        score = self.identity_arbitrator.score(track, output, tracks)
        self._add_profile_seconds("identity_score", time.perf_counter() - score_started)
        return mot.IdentityAssignment(
            track=track,
            output=output,
            score=score,
            assignment_margin=float(margin),
        )

    def _v9_continuity_identity_score(
        self,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
    ) -> mot.IdentityScore:
        if self.identity_arbitrator is not None and self.identity_arbitrator.enabled:
            score_started = time.perf_counter()
            score = self.identity_arbitrator.score(track, output, tracks)
            self._add_profile_seconds("identity_score", time.perf_counter() - score_started)
            return score

        confidence = 0.5 if output.confidence is None else max(0.0, min(1.0, float(output.confidence)))
        predicted = mot.kalman_prediction_reference(track)
        reference_diagonal = max(1.0, mot.bbox_diagonal(track.bbox))
        motion = mot.motion_affinity(predicted, output.bbox, reference_diagonal)
        path = mot.center_path_affinity(track, output.bbox)
        iou = max(mot.bbox_iou(track.bbox, output.bbox), mot.bbox_iou(predicted, output.bbox))
        return mot.IdentityScore(
            total=max(0.0, min(1.0, (0.45 * motion) + (0.35 * path) + (0.15 * confidence) + (0.05 * iou))),
            appearance=0.5,
            motion=motion,
            path=path,
            source=1.0,
            confidence=confidence,
            iou=iou,
            initial_anchor=0.5,
            other_anchor=0.0,
            other_track_id=None,
            identity_margin=0.5,
            occlusion_track_id=None,
            occlusion_iou=0.0,
            negative_anchor=0.0,
        )

    def _v9_continuity_score_candidate(
        self,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        predicted: BBox,
        search_window: BBox,
        confidence: float,
        margin: float,
        scale_gate_state: str,
        local_reject: Optional[str],
    ) -> Dict[str, object]:
        identity_score = self._v9_continuity_identity_score(tracks, track, output)
        accepted_anchor = self._v9_track_last_accepted_bbox(track)
        reference_bbox = accepted_anchor if accepted_anchor is not None else track.bbox
        reference_diagonal = max(1.0, mot.bbox_diagonal(reference_bbox))
        motion = mot.motion_affinity(predicted, output.bbox, reference_diagonal)
        path = mot.center_path_affinity(track, output.bbox)
        accepted_anchor_motion = motion
        if accepted_anchor is not None:
            accepted_anchor_motion = mot.motion_affinity(accepted_anchor, output.bbox, reference_diagonal)
            motion = max(0.0, min(1.0, (0.65 * motion) + (0.35 * accepted_anchor_motion)))
        scale_stats = self._v9_scale_stats(track, output.bbox)
        scale_score = max(0.0, min(1.0, 1.0 / max(1.0, scale_stats["max_ratio_error"])))
        visibility_score = getattr(output, "v9_visibility_score", None)
        visibility_score = 0.5 if visibility_score is None else max(0.0, min(1.0, float(visibility_score)))
        visibility_absent_penalty = max(
            0.0,
            (DEFAULT_V9_VISIBILITY_LOW - visibility_score) / max(1.0e-6, DEFAULT_V9_VISIBILITY_LOW),
        ) * DEFAULT_V9_VISIBILITY_ABSENT_PENALTY

        reference_for_jump = predicted if self._v9_track_uncertain(track) else track.bbox
        center_ratio = self._v9_center_ratio(reference_for_jump, output.bbox)
        max_center_ratio = (
            self.v9_accept_max_center_ratio
            if self._v9_track_uncertain(track)
            else self.v9_accept_max_healthy_center_ratio
        )
        center_jump_penalty = max(0.0, (center_ratio - max_center_ratio) / max(0.25, max_center_ratio))

        margin_score = max(0.0, min(1.0, float(margin) / max(0.04, self.v9_stage1_local_min_margin)))
        other_anchor_pressure = max(0.0, identity_score.other_anchor - identity_score.initial_anchor + 0.03)
        negative_anchor_pressure = max(0.0, identity_score.negative_anchor - identity_score.initial_anchor + 0.03)
        identity_risk = max(other_anchor_pressure, negative_anchor_pressure)
        local_reject_penalty = 0.0
        if local_reject:
            local_reject_penalty = DEFAULT_V9_CONTINUITY_LOCAL_REJECT_PENALTY
            if local_reject == "V9_OUTSIDE_SEARCH_WINDOW":
                local_reject_penalty *= 2.0

        continuity_score = (
            (DEFAULT_V9_CONTINUITY_HEAD_WEIGHT * max(0.0, min(1.0, float(confidence))))
            + (DEFAULT_V9_CONTINUITY_MARGIN_WEIGHT * margin_score)
            + (DEFAULT_V9_CONTINUITY_MOTION_WEIGHT * motion)
            + (DEFAULT_V9_CONTINUITY_PATH_WEIGHT * path)
            + (DEFAULT_V9_CONTINUITY_ANCHOR_WEIGHT * identity_score.initial_anchor)
            + (DEFAULT_V9_CONTINUITY_APPEARANCE_WEIGHT * identity_score.appearance)
            + (DEFAULT_V9_CONTINUITY_SCALE_WEIGHT * scale_score)
            + (DEFAULT_V9_CONTINUITY_VISIBILITY_WEIGHT * visibility_score)
            - (DEFAULT_V9_CONTINUITY_OTHER_ANCHOR_PENALTY * other_anchor_pressure)
            - (DEFAULT_V9_CONTINUITY_NEGATIVE_ANCHOR_PENALTY * negative_anchor_pressure)
            - (DEFAULT_V9_CONTINUITY_CENTER_JUMP_PENALTY * center_jump_penalty)
            - visibility_absent_penalty
            - local_reject_penalty
        )
        if not self._v9_bbox_center_inside(search_window, output.bbox):
            continuity_score -= 0.30
        return {
            "v9_continuity_score": max(0.0, min(1.0, float(continuity_score))),
            "v9_continuity_head_score": float(confidence),
            "v9_continuity_margin_score": margin_score,
            "v9_continuity_motion_score": motion,
            "v9_continuity_accepted_anchor_motion": accepted_anchor_motion,
            "v9_continuity_path_score": path,
            "v9_continuity_anchor_score": identity_score.initial_anchor,
            "v9_continuity_appearance_score": identity_score.appearance,
            "v9_visibility_score": visibility_score,
            "v9_visibility_absent_penalty": visibility_absent_penalty,
            "v9_continuity_other_anchor_score": identity_score.other_anchor,
            "v9_continuity_negative_anchor_score": identity_score.negative_anchor,
            "v9_continuity_other_anchor_pressure": other_anchor_pressure,
            "v9_continuity_negative_anchor_pressure": negative_anchor_pressure,
            "v9_continuity_identity_risk": identity_risk,
            "v9_continuity_identity_margin": identity_score.identity_margin,
            "v9_continuity_scale_score": scale_score,
            "v9_continuity_center_jump_penalty": center_jump_penalty,
            "v9_continuity_local_reject": local_reject or "",
            "v9_continuity_scale_gate_state": scale_gate_state,
        }

    def _v9_continuity_candidate_output(
        self,
        frame: np.ndarray,
        feature_map,
        slot: mot.LoRATMemorySlot,
        bbox: BBox,
        confidence: float,
        visibility: Optional[float] = None,
    ) -> mot.LoRATSlotOutput:
        output = mot.LoRATSlotOutput(
            source_track_id=slot.track_id,
            slot=slot,
            bbox=mot.clamp_bbox_size(bbox),
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        if visibility is not None:
            setattr(output, "v9_visibility_score", max(0.0, min(1.0, float(visibility))))
        if self.identity_arbitrator is not None and self.identity_arbitrator.enabled:
            output = self._with_feature_appearance(output, feature_map, frame.shape)
        return output

    def _v9_candidate_margin_for_rank(
        self,
        ranked_candidates: Sequence[runtime.HeadCandidate],
        rank_index: int,
        fallback_margin: float,
    ) -> float:
        if rank_index + 1 >= len(ranked_candidates):
            return max(0.0, min(1.0, float(fallback_margin)))
        current_confidence = float(ranked_candidates[rank_index].confidence)
        next_confidence = float(ranked_candidates[rank_index + 1].confidence)
        return max(0.0, min(1.0, current_confidence - next_confidence))

    def _v9_select_continuity_candidate(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        slot: mot.LoRATMemorySlot,
        predicted: BBox,
        search_window: BBox,
        candidate_info: runtime.HeadCandidateInfo,
        scale_gate_info: Dict[str, object],
        candidate: BBox,
        confidence: float,
        margin: float,
        candidate_source: str,
    ) -> Dict[str, object]:
        current_gate_state = str(scale_gate_info.get("v9_scale_gate_state", "pass"))
        candidates: List[Dict[str, object]] = []
        seen_bboxes: List[BBox] = []

        def add_candidate(
            rank: int,
            bbox: BBox,
            candidate_confidence: float,
            candidate_margin: float,
            source: str,
            candidate_scale_gate_info: Dict[str, object],
            candidate_visibility: Optional[float] = None,
        ) -> None:
            bbox = mot.clamp_bbox_size(bbox)
            if any(mot.bbox_iou(bbox, seen) >= 0.985 for seen in seen_bboxes):
                return
            seen_bboxes.append(bbox)
            scale_gate_state = str(candidate_scale_gate_info.get("v9_scale_gate_state", "pass"))
            local_reject = self._v9_candidate_local_reject_state(
                track,
                bbox,
                predicted,
                search_window,
                float(candidate_confidence),
                scale_gate_state,
                allow_lost_wide=self._v9_track_uncertain(track),
            )
            output = self._v9_continuity_candidate_output(
                frame,
                feature_map,
                slot,
                bbox,
                float(candidate_confidence),
                candidate_visibility,
            )
            score = self._v9_continuity_score_candidate(
                tracks,
                track,
                output,
                predicted,
                search_window,
                float(candidate_confidence),
                float(candidate_margin),
                scale_gate_state,
                local_reject,
            )
            candidates.append(
                {
                    "rank": rank,
                    "bbox": bbox,
                    "confidence": max(0.0, min(1.0, float(candidate_confidence))),
                    "margin": max(0.0, min(1.0, float(candidate_margin))),
                    "visibility": None
                    if candidate_visibility is None
                    else max(0.0, min(1.0, float(candidate_visibility))),
                    "source": source,
                    "scale_gate_info": candidate_scale_gate_info,
                    "output": output,
                    **score,
                }
            )

        add_candidate(
            0,
            candidate,
            confidence,
            margin,
            candidate_source,
            scale_gate_info,
            getattr(candidate_info, "visibility", None),
        )
        locked_bbox = scale_gate_info.get("v9_scale_gate_locked_bbox")
        if (
            locked_bbox is not None
            and str(scale_gate_info.get("v9_scale_gate_state", "")) == "soft_scale_violation_original_pending"
        ):
            locked_gate_info = dict(scale_gate_info)
            locked_gate_info.update(
                {
                    "v9_scale_gate_state": "soft_scale_violation_locked_candidate",
                    "v9_scale_gate_reason": "soft_scale_candidate_comparison",
                    "v9_scale_gate_suppressed_original": False,
                    "v9_scale_candidate_selected": "locked_candidate",
                }
            )
            add_candidate(
                -1,
                locked_bbox,  # type: ignore[arg-type]
                confidence,
                margin,
                f"{candidate_source}-v9scale-lock",
                locked_gate_info,
                getattr(candidate_info, "visibility", None),
            )

        ranked_top_candidates = sorted(
            tuple(candidate_info.top_candidates or ()),
            key=lambda item: (int(getattr(item, "rank", 999) or 999), -float(getattr(item, "confidence", 0.0) or 0.0)),
        )
        for rank_index, top_candidate in enumerate(ranked_top_candidates[: self.v9_continuity_topk]):
            top_bbox = getattr(top_candidate, "bbox", None)
            if top_bbox is None:
                continue
            top_confidence = float(getattr(top_candidate, "confidence", 0.0) or 0.0)
            top_visibility = getattr(top_candidate, "visibility", None)
            if top_confidence < self.v9_next_best_min_confidence:
                continue
            top_rank = int(getattr(top_candidate, "rank", rank_index + 1) or rank_index + 1)
            top_info = runtime.HeadCandidateInfo(
                bbox=mot.clamp_bbox_size(top_bbox),
                confidence=top_confidence,
                margin=self._v9_candidate_margin_for_rank(ranked_top_candidates, rank_index, margin),
                roi_tokens=candidate_info.roi_tokens,
                top_candidates=(),
            )
            gated_top_info, top_scale_gate_info = self._apply_v9_scale_gate(track, top_info, frame.shape)
            add_candidate(
                top_rank,
                gated_top_info.bbox,
                gated_top_info.confidence,
                gated_top_info.margin,
                f"top{top_rank}-v9continuity",
                top_scale_gate_info,
                None if top_visibility is None else float(top_visibility),
            )

        if not candidates:
            output = self._v9_continuity_candidate_output(frame, feature_map, slot, candidate, confidence)
            return {
                "applied": False,
                "reason": "no_candidates",
                "bbox": candidate,
                "confidence": confidence,
                "margin": margin,
                "source": candidate_source,
                "scale_gate_info": scale_gate_info,
                "output": output,
                "diagnostic": {},
            }

        current = candidates[0]
        viable_candidates = [
            item
            for item in candidates
            if str(item.get("v9_continuity_local_reject") or "") != "V9_OUTSIDE_SEARCH_WINDOW"
        ] or candidates
        best = max(viable_candidates, key=lambda item: float(item.get("v9_continuity_score") or 0.0))
        current_score = float(current.get("v9_continuity_score") or 0.0)
        best_score = float(best.get("v9_continuity_score") or 0.0)
        score_margin = best_score - current_score
        current_reject = str(current.get("v9_continuity_local_reject") or "")
        current_other_pressure = float(
            current.get("v9_continuity_other_anchor_pressure")
            or max(
                0.0,
                float(current.get("v9_continuity_other_anchor_score") or 0.0)
                - float(current.get("v9_continuity_anchor_score") or 0.0),
            )
        )
        best_other_pressure = float(
            best.get("v9_continuity_other_anchor_pressure")
            or max(
                0.0,
                float(best.get("v9_continuity_other_anchor_score") or 0.0)
                - float(best.get("v9_continuity_anchor_score") or 0.0),
            )
        )
        current_identity_risk = float(current.get("v9_continuity_identity_risk") or current_other_pressure)
        best_identity_risk = float(best.get("v9_continuity_identity_risk") or best_other_pressure)
        current_path = float(current.get("v9_continuity_path_score") or 0.0)
        best_path = float(best.get("v9_continuity_path_score") or 0.0)
        current_motion = float(current.get("v9_continuity_motion_score") or 0.0)
        best_motion = float(best.get("v9_continuity_motion_score") or 0.0)
        current_anchor_motion = float(current.get("v9_continuity_accepted_anchor_motion") or current_motion)
        best_anchor_motion = float(best.get("v9_continuity_accepted_anchor_motion") or best_motion)
        current_visibility_value = current.get("v9_visibility_score")
        if current_visibility_value is None:
            current_visibility_value = current.get("visibility")
        best_visibility_value = best.get("v9_visibility_score")
        if best_visibility_value is None:
            best_visibility_value = best.get("visibility")
        current_visibility = 0.5 if current_visibility_value is None else float(current_visibility_value)
        best_visibility = 0.5 if best_visibility_value is None else float(best_visibility_value)
        current_drift_risk = (
            float(current.get("confidence") or 0.0) >= DEFAULT_V9_CONTINUITY_DRIFT_RISK_MIN_CONFIDENCE
            and current_score <= DEFAULT_V9_CONTINUITY_DRIFT_RISK_MAX_SCORE
            and (
                current_identity_risk >= DEFAULT_V9_CONTINUITY_DRIFT_RISK_MIN_IDENTITY_RISK
                or (current_path < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH and current_anchor_motion < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH)
                or current_visibility < DEFAULT_V9_VISIBILITY_LOW
            )
        )
        best_is_current = int(best.get("rank") or 0) == 0
        applied = False
        reason = "current_best"
        if not self.v9_continuity_enabled:
            best = current
            best_score = current_score
            score_margin = 0.0
            reason = "disabled"
        elif best_score < self.v9_continuity_min_score:
            best = current
            best_score = current_score
            score_margin = 0.0
            reason = "below_min_score"
        elif (
            not best_is_current
            and "v9scale-lock" in str(best.get("source") or "")
            and score_margin >= DEFAULT_V9_SCALE_GATE_SOFT_SWITCH_MARGIN
        ):
            applied = True
            reason = "scale_lock_continuity_score"
        elif not best_is_current and score_margin >= self.v9_continuity_min_margin:
            applied = True
            reason = "topk_continuity_score"
        elif (
            not best_is_current
            and current_drift_risk
            and best_score >= self.v9_continuity_min_score - 0.06
            and best_identity_risk + 0.05 < current_identity_risk
        ):
            applied = True
            reason = "high_conf_drift_risk_rescue"
        elif not best_is_current and current_reject and best_score >= self.v9_continuity_min_score:
            applied = True
            reason = f"replace_rejected_current:{current_reject}"
        elif (
            not best_is_current
            and current_identity_risk > 0.03
            and best_identity_risk + 0.03 < current_identity_risk
            and best_score >= current_score - 0.02
        ):
            applied = True
            reason = "reduce_other_anchor_pressure"
        elif (
            not best_is_current
            and best_path - current_path >= 0.18
            and best_score >= current_score - 0.015
        ):
            applied = True
            reason = "path_continuity_rescue"
        elif (
            not best_is_current
            and best_anchor_motion - current_anchor_motion >= 0.18
            and best_score >= current_score - 0.015
        ):
            applied = True
            reason = "accepted_anchor_motion_rescue"
        else:
            best = current
            best_score = current_score
            score_margin = 0.0
            best_other_pressure = current_other_pressure
            best_identity_risk = current_identity_risk
            best_path = current_path
            best_motion = current_motion
            best_anchor_motion = current_anchor_motion
            if current_drift_risk:
                reason = "current_drift_risk_no_safe_alternative"

        diagnostic = {
            "v9_continuity_enabled": self.v9_continuity_enabled,
            "v9_continuity_candidate_count": len(candidates),
            "v9_continuity_applied": applied,
            "v9_continuity_reason": reason,
            "v9_continuity_current_score": current_score,
            "v9_continuity_best_score": best_score,
            "v9_continuity_score_margin": score_margin,
            "v9_continuity_selected_rank": int(best.get("rank") or 0),
            "v9_continuity_selected_source": str(best.get("source") or ""),
            "v9_continuity_selected_bbox": best.get("bbox"),
            "v9_continuity_current_bbox": current.get("bbox"),
            "v9_continuity_current_source": current.get("source"),
            "v9_continuity_best_local_reject": best.get("v9_continuity_local_reject"),
            "v9_continuity_current_local_reject": current.get("v9_continuity_local_reject"),
            "v9_continuity_current_identity_risk": current_identity_risk,
            "v9_continuity_best_identity_risk": best_identity_risk,
            "v9_continuity_current_path_score": current_path,
            "v9_continuity_best_path_score": best_path,
            "v9_continuity_current_accepted_anchor_motion": current_anchor_motion,
            "v9_continuity_best_accepted_anchor_motion": best_anchor_motion,
            "v9_continuity_current_visibility": current_visibility,
            "v9_continuity_best_visibility": best_visibility,
            "v9_continuity_current_drift_risk": current_drift_risk,
        }
        for key in (
            "v9_continuity_score",
            "v9_continuity_head_score",
            "v9_continuity_margin_score",
            "v9_continuity_motion_score",
            "v9_continuity_accepted_anchor_motion",
            "v9_continuity_path_score",
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
            "v9_continuity_scale_gate_state",
        ):
            diagnostic[key] = best.get(key)

        selected = best if applied else current
        selected_drift_risk = bool(
            selected is current
            and current_drift_risk
            and not applied
        )
        diagnostic["v9_continuity_drift_risk"] = selected_drift_risk
        selected_scale_info = dict(selected["scale_gate_info"])
        selected_scale_info["v9_scale_candidate_selected"] = (
            "locked" if "v9scale-lock" in str(selected.get("source") or "") else "original"
        )
        diagnostic["v9_scale_candidate_selected"] = selected_scale_info["v9_scale_candidate_selected"]
        diagnostic["v9_scale_candidate_original_score"] = selected_scale_info.get("v9_scale_candidate_original_score")
        diagnostic["v9_scale_candidate_locked_score"] = selected_scale_info.get("v9_scale_candidate_locked_score")
        diagnostic["candidate_visibility"] = selected.get("visibility")
        return {
            "applied": applied,
            "reason": reason,
            "bbox": selected["bbox"],
            "confidence": selected["confidence"],
            "margin": selected["margin"],
            "source": f"{selected['source']}-v9driftrisk" if selected_drift_risk else selected["source"],
            "scale_gate_info": selected_scale_info,
            "output": selected["output"],
            "diagnostic": diagnostic,
        }

    def _v9_candidate_source_for_record(
        self,
        record: Dict[str, object],
        suffix: str,
    ) -> str:
        source = str(record.get("candidate_source", "head"))
        gate_state = str(dict(record.get("scale_gate_info") or {}).get("v9_scale_gate_state", "pass"))
        if self._v9_is_scale_modified_state(gate_state):
            source = f"{source}-v9scale"
        return f"{source}-v9local-{suffix}"

    def _v9_accept_record_candidate(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        record: Dict[str, object],
        frame_number: int,
        suffix: str,
    ) -> Tuple[bool, Optional[mot.IdentityAssignment], str]:
        output = record["output"]  # type: ignore[assignment]
        confidence = float(record.get("confidence") or getattr(output, "confidence", 0.0) or 0.0)
        margin = float(record.get("margin") or 0.0)
        predicted = record["predicted"]  # type: ignore[assignment]
        assignment = self._v9_identity_assignment_for_output(tracks, track, output, margin)  # type: ignore[arg-type]
        candidate_source = self._v9_candidate_source_for_record(record, suffix)
        accepted = self._accept_candidate(
            frame,
            feature_map,
            track,
            output.bbox,  # type: ignore[union-attr]
            confidence,
            margin,
            predicted,  # type: ignore[arg-type]
            frame_number,
            assignment,
            candidate_source,
        )
        return bool(accepted), assignment, candidate_source

    def _v9_try_next_best_local_candidate(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        track: mot.TrackState,
        record: Optional[Dict[str, object]],
        frame_number: int,
        reject_state: str,
    ) -> Tuple[bool, Optional[mot.IdentityAssignment], str, str]:
        if record is None or self.v9_next_best_max_candidates <= 0:
            return False, None, "", "no_record"

        top_candidates = tuple(record.get("head_top_candidates") or ())
        if not top_candidates:
            return False, None, "", "no_top_candidates"

        predicted = record.get("predicted")
        search_window = record.get("search_window")
        slot = record.get("slot")
        if predicted is None or search_window is None or slot is None:
            return False, None, "", "missing_geometry"

        scale_gate_state = str(dict(record.get("scale_gate_info") or {}).get("v9_scale_gate_state", "pass"))
        current_bbox = mot.clamp_bbox_size(record["candidate"]) if record.get("candidate") is not None else None
        sorted_candidates = sorted(
            top_candidates,
            key=lambda item: (int(getattr(item, "rank", 999) or 999), -float(getattr(item, "confidence", 0.0) or 0.0)),
        )
        considered = 0
        last_reject = "no_viable_top_candidate"
        for candidate in sorted_candidates:
            if considered >= self.v9_next_best_max_candidates:
                break
            candidate_bbox = getattr(candidate, "bbox", None)
            if candidate_bbox is None:
                continue
            bbox = mot.clamp_bbox_size(candidate_bbox)
            confidence = float(getattr(candidate, "confidence", 0.0) or 0.0)
            visibility = getattr(candidate, "visibility", None)
            visibility_score = 0.5 if visibility is None else max(0.0, min(1.0, float(visibility)))
            rank = int(getattr(candidate, "rank", considered + 1) or considered + 1)
            if current_bbox is not None and mot.bbox_iou(current_bbox, bbox) >= 0.98:
                continue
            considered += 1
            if confidence < self.v9_next_best_min_confidence:
                last_reject = "next_best_confidence_low"
                continue
            if visibility_score < DEFAULT_V9_VISIBILITY_LOW:
                last_reject = "next_best_visibility_low"
                continue
            local_reject = self._v9_candidate_local_reject_state(
                track,
                bbox,
                predicted,  # type: ignore[arg-type]
                search_window,  # type: ignore[arg-type]
                confidence,
                scale_gate_state,
                allow_lost_wide=self._v9_track_uncertain(track),
            )
            if local_reject is not None:
                last_reject = local_reject
                continue

            output = mot.LoRATSlotOutput(
                source_track_id=track.track_id,
                slot=slot,  # type: ignore[arg-type]
                bbox=bbox,
                confidence=confidence,
            )
            setattr(output, "v9_visibility_score", visibility_score)
            if self.identity_arbitrator.enabled:
                output = self._with_feature_appearance(output, feature_map, frame.shape)
            assignment = self._v9_identity_assignment_for_output(tracks, track, output, max(0.0, float(record.get("margin") or 0.0)))
            candidate_source = f"top{rank}-v9local-v9next"
            accepted = self._accept_candidate(
                frame,
                feature_map,
                track,
                bbox,
                confidence,
                max(0.0, float(record.get("margin") or 0.0)),
                predicted,  # type: ignore[arg-type]
                frame_number,
                assignment,
                candidate_source,
            )
            if accepted:
                return True, assignment, candidate_source, f"accepted_after_{reject_state or 'reject'}"
            last_reject = str(getattr(track, "state", "") or "next_best_rejected")
        return False, None, "", last_reject

    @staticmethod
    def _v9_reid_reject_is_identity(reject_state: str) -> bool:
        state = str(reject_state or "").upper()
        return any(
            token in state
            for token in (
                "ID",
                "REID",
                "ANCHOR",
                "OTHER",
                "MOTIONLOW",
                "PATHLOW",
                "NO_ASSIGNMENT",
                "BLOCKED",
            )
        )

    def _v9_should_prefer_local_owner(
        self,
        track: mot.TrackState,
        source_record: Optional[Dict[str, object]],
        own_record: Optional[Dict[str, object]],
        identity_assignment: Optional[mot.IdentityAssignment],
    ) -> bool:
        if (
            not self.v9_protective_reid
            or own_record is None
            or source_record is None
            or identity_assignment is None
            or identity_assignment.output.source_track_id == track.track_id
            or self._v9_track_uncertain(track)
        ):
            return False
        own_confidence = float(own_record.get("confidence") or 0.0)
        assigned_confidence = float(identity_assignment.output.confidence or 0.0)
        own_scale_info = dict(own_record.get("scale_gate_info") or {})
        own_scale_state = str(own_scale_info.get("v9_scale_gate_state", "pass"))
        own_reject = self._v9_candidate_local_reject_state(
            track,
            own_record["candidate"],  # type: ignore[arg-type]
            own_record["predicted"],  # type: ignore[arg-type]
            own_record["search_window"],  # type: ignore[arg-type]
            own_confidence,
            own_scale_state,
        )
        if own_reject is not None:
            return False
        if own_confidence + 0.05 >= assigned_confidence:
            return True
        score = identity_assignment.score
        return (
            score.appearance < max(0.66, self.identity_arbitrator.min_reid if self.identity_arbitrator else 0.0)
            or score.motion < max(0.45, self.identity_arbitrator.min_motion if self.identity_arbitrator else 0.0)
            or score.identity_margin < 0.02
        )

    def _v9_select_hold_bbox(
        self,
        track: mot.TrackState,
        predicted: BBox,
        source_record: Optional[Dict[str, object]],
        identity_assignment: Optional[mot.IdentityAssignment],
        reject_state: str,
    ) -> Tuple[BBox, float, float, str]:
        if source_record is None or identity_assignment is None:
            return predicted, 0.0, 0.0, "kalman"

        candidate = identity_assignment.output.bbox
        confidence = float(identity_assignment.output.confidence or source_record.get("confidence") or 0.0)
        margin = float(identity_assignment.assignment_margin)
        scale_gate_info = dict(source_record.get("scale_gate_info") or {})
        scale_gate_state = str(scale_gate_info.get("v9_scale_gate_state", "pass"))
        search_window = source_record.get("search_window")
        if search_window is None:
            return predicted, confidence, margin, "kalman"
        score = identity_assignment.score
        selected_target_support = (
            identity_assignment.output.source_track_id == track.track_id
            and score.total >= self.v9_local_rescue_min_assignment_score
            and score.identity_margin >= -0.02
            and score.initial_anchor >= max(DEFAULT_V9_LOCAL_RESCUE_MIN_ANCHOR, score.other_anchor - 0.02)
            and (
                score.motion >= DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH
                or score.path >= DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH
            )
            and not self._is_initial_anchor_steal(score)
        )
        if confidence < self.v9_local_hold_min_confidence and not selected_target_support:
            return predicted, confidence, margin, "kalman"
        if scale_gate_state == "override_high_confidence" and confidence < self.v9_scale_gate_override_confidence:
            return predicted, confidence, margin, "kalman"

        local_reject = self._v9_candidate_local_reject_state(
            track,
            candidate,
            predicted,
            search_window,  # type: ignore[arg-type]
            confidence,
            scale_gate_state,
            allow_lost_wide=True,
        )
        soft_selected_reject = local_reject in {"V9_SCALE_LOCK_LOWCONF", "V9_SCALE_OVERRIDE_LOWCONF"}
        if local_reject is not None and local_reject != "V9_CENTER_JUMP" and not (selected_target_support and soft_selected_reject):
            return predicted, confidence, margin, "kalman"
        center_ratio = self._v9_center_ratio(predicted, candidate)
        max_ratio = (
            self.v9_local_hold_max_lost_center_ratio
            if self._v9_track_uncertain(track)
            else self.v9_local_hold_max_center_ratio
        )
        if center_ratio > max_ratio and confidence < self.v9_scale_gate_override_confidence:
            return predicted, confidence, margin, "kalman"
        if str(reject_state or "").startswith("V9_") and "CENTER_JUMP" in str(reject_state):
            return predicted, confidence, margin, "kalman"
        return candidate, confidence, margin, "v9-local-hold"

    def _v9_can_relax_identity_reject(
        self,
        track: mot.TrackState,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
        candidate_source: str,
        reject_state: str,
    ) -> bool:
        if (
            not self.v9_protective_reid
            or "v9local" not in str(candidate_source)
            or identity_assignment is None
            or identity_assignment.output.source_track_id != track.track_id
            or confidence < max(self.min_confidence, self.lorat_accept_min_score)
            or reject_state
            not in {
                "ID_UNCERTAIN",
                "REIDLOW",
                "MOTIONLOW",
                "PATHLOW",
                "ANCHORLOW",
            }
        ):
            return False
        score = identity_assignment.score
        if score.identity_margin < -0.02:
            return False
        if score.initial_anchor < max(0.38, score.other_anchor - 0.02):
            return False
        if score.motion < 0.25 and score.path < 0.25:
            return False
        if self._is_initial_anchor_steal(score):
            return False
        if (
            score.negative_anchor >= runtime.DEFAULT_DISTRACTOR_MIN_SIMILARITY
            and score.negative_anchor >= score.initial_anchor - 0.03
        ):
            return False
        if (
            score.other_track_id is not None
            and score.other_anchor >= max(0.62, score.initial_anchor + 0.06)
            and score.identity_margin < -0.04
        ):
            return False
        return True

    def _v9_can_rescue_local_reject(
        self,
        track: mot.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
        candidate_source: str,
        reject_state: str,
    ) -> bool:
        del confidence
        if not self.v9_local_rescue or "v9local" not in str(candidate_source):
            return False
        if reject_state not in {"LOWCONF", "ID_UNCERTAIN", "REACQUIRE_LOWCONF"}:
            return False
        if self.identity_arbitrator.enabled and identity_assignment is None:
            return False
        if identity_assignment is not None and identity_assignment.output.source_track_id != track.track_id:
            return False

        search_window = self._last_v9_search_windows_by_track_id.get(track.track_id)
        if search_window is not None and not self._v9_bbox_center_inside(search_window, bbox):
            return False

        predicted = mot.kalman_prediction_reference(track)
        reference_diagonal = max(1.0, mot.bbox_diagonal(track.bbox))
        motion = mot.motion_affinity(predicted, bbox, reference_diagonal)
        path = mot.center_path_affinity(track, bbox)
        if motion < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH and path < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH:
            return False

        stats = self._v9_scale_stats(track, bbox)
        if stats["max_ratio_error"] > self.v9_local_rescue_max_scale_error:
            return False

        if identity_assignment is None:
            return True

        score = identity_assignment.score
        if score.total < self.v9_local_rescue_min_assignment_score:
            return False
        if score.identity_margin < -0.02:
            return False
        if score.initial_anchor < max(DEFAULT_V9_LOCAL_RESCUE_MIN_ANCHOR, score.other_anchor - 0.02):
            return False
        if score.motion < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH and score.path < DEFAULT_V9_LOCAL_RESCUE_MIN_MOTION_OR_PATH:
            return False
        if self._is_initial_anchor_steal(score):
            return False
        if (
            score.negative_anchor >= runtime.DEFAULT_DISTRACTOR_MIN_SIMILARITY
            and score.negative_anchor >= score.initial_anchor - 0.03
        ):
            return False
        if (
            score.other_track_id is not None
            and score.other_anchor >= max(0.62, score.initial_anchor + 0.06)
            and score.identity_margin < -0.04
        ):
            return False
        return True

    def _candidate_reject_state(
        self,
        track: mot.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
        candidate_source: str = "head",
    ) -> Optional[str]:
        if "v9driftrisk" in str(candidate_source):
            return "V9_DRIFT_RISK"
        reject_state = super()._candidate_reject_state(
            track,
            bbox,
            confidence,
            identity_assignment,
            candidate_source,
        )
        if reject_state is None:
            return None
        if self._v9_can_relax_identity_reject(
            track,
            confidence,
            identity_assignment,
            candidate_source,
            reject_state,
        ):
            setattr(track, "v9_last_relaxed_identity_reject", reject_state)
            return None
        if self._v9_can_rescue_local_reject(
            track,
            bbox,
            confidence,
            identity_assignment,
            candidate_source,
            reject_state,
        ):
            setattr(track, "v9_last_local_rescue_reject", reject_state)
            return None
        return reject_state

    def _accept_candidate(
        self,
        frame: np.ndarray,
        feature_map,
        track: mot.TrackState,
        candidate: BBox,
        confidence: float,
        margin: float,
        predicted: BBox,
        frame_number: int,
        identity_assignment: Optional[mot.IdentityAssignment] = None,
        candidate_source: str = "head",
    ) -> bool:
        setattr(track, "v9_last_local_rescue_reject", "")
        accepted = super()._accept_candidate(
            frame,
            feature_map,
            track,
            candidate,
            confidence,
            margin,
            predicted,
            frame_number,
            identity_assignment,
            candidate_source,
        )
        rescued_reject = str(getattr(track, "v9_last_local_rescue_reject", "") or "")
        if accepted and rescued_reject:
            track.state = mot.append_state_token(track.state, "V9LOCALRESCUE")
            track.state = mot.append_state_token(track.state, rescued_reject)
            if track.assigned_source:
                track.assigned_source = f"{track.assigned_source}-v9localrescue"
        if accepted:
            setattr(track, "v9_last_accepted_bbox", mot.clamp_bbox_size(track.bbox))
            setattr(track, "v9_last_accepted_frame", int(frame_number))
            setattr(track, "v9_last_accepted_confidence", float(confidence))
            setattr(track, "v9_last_accepted_source", str(candidate_source or "head"))
            source_text = str(candidate_source or "").lower()
            high_conf_low_margin = (
                float(confidence) >= DEFAULT_V9_TRUSTED_ANCHOR_HIGH_CONFIDENCE
                and float(margin) < DEFAULT_V9_TRUSTED_ANCHOR_MIN_MARGIN
            )
            drift_risk_source = "v9driftrisk" in source_text
            scale_modified = "v9scale" in source_text and float(margin) < DEFAULT_V9_TRUSTED_ANCHOR_MIN_MARGIN
            trusted_anchor = not (high_conf_low_margin or drift_risk_source or scale_modified)
            if trusted_anchor:
                setattr(track, "v9_last_trusted_anchor_bbox", mot.clamp_bbox_size(track.bbox))
                setattr(track, "v9_last_trusted_anchor_frame", int(frame_number))
                setattr(track, "v9_last_trusted_anchor_confidence", float(confidence))
                setattr(track, "v9_last_trusted_anchor_source", str(candidate_source or "head"))
                setattr(track, "v9_anchor_suspect_frames", 0)
            else:
                current_suspect = self._v9_anchor_suspect_frames(track)
                setattr(
                    track,
                    "v9_anchor_suspect_frames",
                    max(current_suspect, DEFAULT_V9_TRUSTED_ANCHOR_SUSPECT_FRAMES),
                )
                track.state = mot.append_state_token(track.state, "V9ANCHORSUSPECT")
        return accepted

    def _v9_reference_size(self, track: mot.TrackState) -> Tuple[float, float]:
        reference_w, reference_h = self._template_match_reference_size(track)
        return max(1.0, float(reference_w)), max(1.0, float(reference_h))

    def _v9_scale_stats(self, track: mot.TrackState, bbox: BBox) -> Dict[str, float]:
        _, _, width, height = mot.clamp_bbox_size(bbox)
        reference_w, reference_h = self._v9_reference_size(track)
        width_ratio = width / max(1.0, reference_w)
        height_ratio = height / max(1.0, reference_h)
        return {
            "width_ratio": float(width_ratio),
            "height_ratio": float(height_ratio),
            "max_ratio_error": float(
                max(
                    width_ratio,
                    1.0 / max(width_ratio, 1.0e-6),
                    height_ratio,
                    1.0 / max(height_ratio, 1.0e-6),
                )
            ),
        }

    def _v9_scale_gate_ok(self, track: mot.TrackState, bbox: BBox) -> bool:
        stats = self._v9_scale_stats(track, bbox)
        return (
            self.v9_scale_gate_min_ratio <= stats["width_ratio"] <= self.v9_scale_gate_max_ratio
            and self.v9_scale_gate_min_ratio <= stats["height_ratio"] <= self.v9_scale_gate_max_ratio
        )

    def _v9_scale_gate_hard_ok(self, track: mot.TrackState, bbox: BBox) -> bool:
        stats = self._v9_scale_stats(track, bbox)
        return (
            DEFAULT_V9_SCALE_GATE_HARD_MIN_RATIO <= stats["width_ratio"] <= DEFAULT_V9_SCALE_GATE_HARD_MAX_RATIO
            and DEFAULT_V9_SCALE_GATE_HARD_MIN_RATIO <= stats["height_ratio"] <= DEFAULT_V9_SCALE_GATE_HARD_MAX_RATIO
        )

    @staticmethod
    def _v9_candidate_visibility_value(candidate_info: runtime.HeadCandidateInfo, bbox: Optional[BBox] = None) -> Optional[float]:
        if bbox is not None:
            for candidate in tuple(candidate_info.top_candidates or ()):
                if mot.bbox_iou(mot.clamp_bbox_size(candidate.bbox), mot.clamp_bbox_size(bbox)) >= 0.98:
                    visibility = getattr(candidate, "visibility", None)
                    if visibility is not None:
                        return max(0.0, min(1.0, float(visibility)))
        visibility = getattr(candidate_info, "visibility", None)
        if visibility is None:
            return None
        return max(0.0, min(1.0, float(visibility)))

    @staticmethod
    def _v9_attach_candidate_visibility(candidate_info: runtime.HeadCandidateInfo, visibility: Optional[float]) -> runtime.HeadCandidateInfo:
        if visibility is not None:
            object.__setattr__(candidate_info, "visibility", max(0.0, min(1.0, float(visibility))))
        return candidate_info

    def _rerank_head_candidate(
        self,
        feature_map,
        frame_shape: Tuple[int, ...],
        track: mot.TrackState,
        predicted: BBox,
        candidate_info: runtime.HeadCandidateInfo,
    ) -> runtime.HeadCandidateInfo:
        reranked = super()._rerank_head_candidate(feature_map, frame_shape, track, predicted, candidate_info)
        if reranked is candidate_info:
            return reranked
        visibility = self._v9_candidate_visibility_value(candidate_info, reranked.bbox)
        return self._v9_attach_candidate_visibility(reranked, visibility)

    def _v9_reference_sized_box(self, track: mot.TrackState, bbox: BBox, frame_shape: Optional[Tuple[int, ...]]) -> BBox:
        center_x, center_y = mot.bbox_center(bbox)
        reference_w, reference_h = self._v9_reference_size(track)
        x = center_x - (reference_w * 0.5)
        y = center_y - (reference_h * 0.5)
        if frame_shape is None:
            return mot.clamp_bbox_size((x, y, reference_w, reference_h))
        frame_height, frame_width = frame_shape[:2]
        x = max(0.0, min(float(frame_width) - 1.0, x))
        y = max(0.0, min(float(frame_height) - 1.0, y))
        width = min(reference_w, max(1.0, float(frame_width) - x))
        height = min(reference_h, max(1.0, float(frame_height) - y))
        return mot.clamp_bbox_size((x, y, width, height))

    def _apply_v9_scale_gate(
        self,
        track: mot.TrackState,
        candidate_info: runtime.HeadCandidateInfo,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[runtime.HeadCandidateInfo, Dict[str, object]]:
        original_bbox = mot.clamp_bbox_size(candidate_info.bbox)
        stats = self._v9_scale_stats(track, original_bbox)
        state: Dict[str, object] = {
            "v9_scale_gate_state": "pass",
            "v9_scale_gate_reason": "",
            "v9_scale_gate_width_ratio": stats["width_ratio"],
            "v9_scale_gate_height_ratio": stats["height_ratio"],
            "v9_scale_gate_original_bbox": original_bbox,
            "v9_scale_gate_original_confidence": float(candidate_info.confidence),
        }
        if self._v9_scale_gate_ok(track, original_bbox):
            return candidate_info, state

        for top_candidate in tuple(candidate_info.top_candidates or ()):
            if self._v9_scale_gate_ok(track, top_candidate.bbox):
                state.update(
                    {
                        "v9_scale_gate_state": "switched_topk",
                        "v9_scale_gate_reason": f"top{top_candidate.rank}_scale_ok",
                    }
                )
                switched_info = runtime.HeadCandidateInfo(
                    bbox=top_candidate.bbox,
                    confidence=float(top_candidate.confidence),
                    margin=max(0.0, min(float(candidate_info.margin), float(candidate_info.confidence) - float(top_candidate.confidence))),
                    roi_tokens=candidate_info.roi_tokens,
                    top_candidates=candidate_info.top_candidates,
                )
                return (
                    self._v9_attach_candidate_visibility(switched_info, getattr(top_candidate, "visibility", None)),
                    state,
                )

        candidate_confidence = float(candidate_info.confidence)
        candidate_margin = float(candidate_info.margin)
        locked = self._v9_reference_sized_box(track, original_bbox, frame_shape)
        locked_stats = self._v9_scale_stats(track, locked)
        original_score = max(0.0, min(1.0, 1.0 / max(1.0, stats["max_ratio_error"])))
        locked_score = max(0.0, min(1.0, 1.0 / max(1.0, locked_stats["max_ratio_error"])))
        hard_ok = self._v9_scale_gate_hard_ok(track, original_bbox)
        state.update(
            {
                "v9_scale_gate_locked_bbox": locked,
                "v9_scale_gate_locked_width_ratio": locked_stats["width_ratio"],
                "v9_scale_gate_locked_height_ratio": locked_stats["height_ratio"],
                "v9_scale_candidate_original_score": original_score,
                "v9_scale_candidate_locked_score": locked_score,
                "v9_scale_gate_hard_ok": hard_ok,
            }
        )
        if hard_ok:
            state.update(
                {
                    "v9_scale_gate_state": "soft_scale_violation_original_pending",
                    "v9_scale_gate_reason": "soft_scale_violation_compare_original_vs_lock",
                    "v9_scale_gate_suppressed_original": False,
                    "v9_scale_gate_confidence_preserved": True,
                    "v9_scale_candidate_selected": "original",
                }
            )
            return candidate_info, state

        confidence_scale = 1.0 if candidate_confidence >= self.v9_scale_gate_override_confidence else self.v9_scale_gate_fallback_confidence_scale
        state.update(
            {
                "v9_scale_gate_state": (
                    "reference_size_lock_high_confidence"
                    if candidate_confidence >= self.v9_scale_gate_override_confidence
                    else "reference_size_lock"
                ),
                "v9_scale_gate_reason": (
                    "hard_scale_violation_head_confident_reference_lock"
                    if candidate_confidence >= self.v9_scale_gate_override_confidence
                    else "hard_scale_violation_reference_lock"
                ),
                "v9_scale_gate_suppressed_original": True,
                "v9_scale_gate_confidence_preserved": candidate_confidence >= self.v9_scale_gate_override_confidence,
                "v9_scale_candidate_selected": "locked",
            }
        )
        locked_info = runtime.HeadCandidateInfo(
            bbox=locked,
            confidence=candidate_confidence * confidence_scale,
            margin=candidate_margin * confidence_scale,
            roi_tokens=candidate_info.roi_tokens,
            top_candidates=candidate_info.top_candidates,
        )
        return (
            self._v9_attach_candidate_visibility(locked_info, self._v9_candidate_visibility_value(candidate_info)),
            state,
        )

    def _sample_local_search_grids(
        self,
        feature_map,
        search_windows: Sequence[BBox],
        frame_shape: Tuple[int, ...],
    ):
        if not search_windows:
            return self.torch.zeros((0, self.v9_local_grid_size, self.v9_local_grid_size, self.embed_dim), device=self.device)

        frame_height, frame_width = frame_shape[:2]
        grid_size = self.v9_local_grid_size
        windows = self.torch.tensor(search_windows, device=self.device, dtype=self.torch.float32)
        xs = (self.torch.arange(grid_size, device=self.device, dtype=self.torch.float32) + 0.5) / float(grid_size)
        ys = (self.torch.arange(grid_size, device=self.device, dtype=self.torch.float32) + 0.5) / float(grid_size)
        yy, xx = self.torch.meshgrid(ys, xs, indexing="ij")
        sample_x = windows[:, 0, None, None] + xx[None, :, :] * windows[:, 2, None, None].clamp_min(1.0)
        sample_y = windows[:, 1, None, None] + yy[None, :, :] * windows[:, 3, None, None].clamp_min(1.0)
        norm_x = (sample_x / max(1.0, float(frame_width))) * 2.0 - 1.0
        norm_y = (sample_y / max(1.0, float(frame_height))) * 2.0 - 1.0
        grid = self.torch.stack((norm_x, norm_y), dim=-1)
        source = feature_map.permute(2, 0, 1).unsqueeze(0).expand(len(search_windows), -1, -1, -1)
        sampled = self.F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1).contiguous()
        return self.F.normalize(sampled.to(self.torch.float32), dim=-1)

    def _search_windows_for_tracks(
        self,
        predicted_bboxes: Sequence[BBox],
        tracks: Sequence[mot.TrackState],
        frame_number: Optional[int] = None,
    ) -> List[BBox]:
        windows: List[BBox] = []
        for predicted, track in zip(predicted_bboxes, tracks):
            reference = self._template_match_reference_size(track)
            anchor = mot.clamp_bbox_size(predicted)
            anchor_source = "current"
            if self.v9_diagnostic_mode == "gt_window" and track.track_id in self._v9_diagnostic_gt_boxes:
                gt_bbox = self._v9_diagnostic_gt_boxes[track.track_id]
                _, _, gt_w, gt_h = mot.clamp_bbox_size(gt_bbox)
                reference = (max(float(reference[0]), gt_w), max(float(reference[1]), gt_h))
                anchor = gt_bbox
                anchor_source = "gt_window"
                window = self._expanded_search_bbox(gt_bbox, gt_bbox, reference)
            else:
                last_accepted = self._v9_track_last_accepted_bbox(track)
                trusted_anchor = self._v9_track_trusted_anchor_bbox(track)
                suspect_frames = self._v9_anchor_suspect_frames(track)
                if suspect_frames > 0 and trusted_anchor is not None:
                    last_center = mot.bbox_center(trusted_anchor)
                    predicted_center = mot.bbox_center(predicted)
                    dx = (predicted_center[0] - last_center[0]) * 0.20
                    dy = (predicted_center[1] - last_center[1]) * 0.20
                    _, _, trusted_w, trusted_h = mot.clamp_bbox_size(trusted_anchor)
                    max_dx = max(1.0, trusted_w * DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_WIDTHS)
                    max_dy = max(1.0, trusted_h * DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_HEIGHTS)
                    dx = max(-max_dx, min(max_dx, dx))
                    dy = max(-max_dy, min(max_dy, dy))
                    if abs(dx) + abs(dy) > 0.001:
                        anchor = self._v9_translate_bbox(trusted_anchor, dx, dy)
                        anchor_source = "trusted_velocity"
                    else:
                        anchor = trusted_anchor
                        anchor_source = "trusted"
                    window = self._expanded_search_bbox(anchor, trusted_anchor, reference)
                    setattr(track, "v9_anchor_suspect_frames", max(0, suspect_frames - 1))
                elif self._v9_track_uncertain(track) and trusted_anchor is not None:
                    last_center = mot.bbox_center(trusted_anchor)
                    predicted_center = mot.bbox_center(predicted)
                    dx = (predicted_center[0] - last_center[0]) * 0.35
                    dy = (predicted_center[1] - last_center[1]) * 0.35
                    _, _, accepted_w, accepted_h = mot.clamp_bbox_size(trusted_anchor)
                    max_dx = max(1.0, accepted_w * DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_WIDTHS)
                    max_dy = max(1.0, accepted_h * DEFAULT_V9_UNCERTAIN_VELOCITY_MAX_HEIGHTS)
                    dx = max(-max_dx, min(max_dx, dx))
                    dy = max(-max_dy, min(max_dy, dy))
                    if abs(dx) + abs(dy) > 0.001:
                        anchor = self._v9_translate_bbox(trusted_anchor, dx, dy)
                        anchor_source = "velocity"
                    else:
                        anchor = trusted_anchor
                        anchor_source = "accepted"
                    window = self._expanded_search_bbox(anchor, trusted_anchor, reference)
                elif trusted_anchor is not None and int(getattr(track, "lost_frames", 0) or 0) > 0:
                    anchor = trusted_anchor
                    anchor_source = "accepted"
                    window = self._expanded_search_bbox(anchor, trusted_anchor, reference)
                else:
                    anchor_source = "held" if int(getattr(track, "lost_frames", 0) or 0) > 0 else "current"
                    window = self._expanded_search_bbox(predicted, track.bbox, reference)
            windows.append(mot.clamp_bbox_size(window))
            last_frame = getattr(track, "v9_last_accepted_frame", None)
            anchor_age = None
            if frame_number is not None and last_frame is not None:
                try:
                    anchor_age = max(0, int(frame_number) - int(last_frame))
                except (TypeError, ValueError):
                    anchor_age = None
            setattr(track, "v9_search_anchor_bbox", mot.clamp_bbox_size(anchor))
            setattr(track, "v9_search_anchor_source", anchor_source)
            setattr(track, "v9_search_anchor_age", anchor_age)
        self._last_v9_search_windows_by_track_id = {
            track.track_id: window for track, window in zip(tracks, windows)
        }
        return windows

    def _candidates_from_local_head_output(
        self,
        score_maps,
        box_delta_maps,
        visibility_maps,
        search_windows: Sequence[BBox],
    ) -> List[runtime.HeadCandidateInfo]:
        candidate_started = time.perf_counter()
        object_count = int(score_maps.shape[0])
        if object_count <= 0:
            return []

        grid_height = int(score_maps.shape[1])
        grid_width = int(score_maps.shape[2])
        score_maps_float = score_maps.to(self.torch.float32)
        visibility_maps_float = None if visibility_maps is None else visibility_maps.to(self.torch.float32)
        flat_visibility = (
            None
            if visibility_maps_float is None
            else self.torch.sigmoid(visibility_maps_float.reshape(object_count, -1))
        )
        if self.v8_window_penalty_ratio > 0 and grid_height > 1 and grid_width > 1:
            window = self.torch.outer(
                self.torch.hann_window(grid_height, periodic=False, device=score_maps.device, dtype=self.torch.float32),
                self.torch.hann_window(grid_width, periodic=False, device=score_maps.device, dtype=self.torch.float32),
            )
            selection_scores = (
                self.torch.sigmoid(score_maps_float) * (1.0 - self.v8_window_penalty_ratio)
                + window[None, :, :] * self.v8_window_penalty_ratio
            )
        else:
            selection_scores = score_maps_float

        flat_scores = score_maps_float.reshape(object_count, -1)
        flat_selection_scores = selection_scores.reshape(object_count, -1)
        best_selection_values, best_indices = self.torch.max(flat_selection_scores, dim=1)
        finite_best = self.torch.isfinite(best_selection_values)
        safe_best_indices = self.torch.where(finite_best, best_indices, self.torch.zeros_like(best_indices))
        raw_best_values = self.torch.gather(flat_scores, 1, safe_best_indices[:, None]).squeeze(1)
        safe_best_values = self.torch.where(finite_best, raw_best_values, self.torch.full_like(raw_best_values, -30.0))

        if flat_selection_scores.shape[1] >= 2:
            top2 = self.torch.topk(flat_selection_scores, k=2, dim=1).values
            margins = self.torch.clamp(top2[:, 0] - top2[:, 1], min=0.0)
        else:
            margins = self.torch.zeros_like(safe_best_values)

        topk_count = min(5, int(flat_selection_scores.shape[1]))
        top_packed = []
        if topk_count > 0:
            top_selection_values, top_indices = self.torch.topk(flat_selection_scores, k=topk_count, dim=1)
            finite_top = self.torch.isfinite(top_selection_values)
            safe_top_indices = self.torch.where(finite_top, top_indices, self.torch.zeros_like(top_indices))
            top_grid_y = self.torch.div(safe_top_indices, grid_width, rounding_mode="floor")
            top_grid_x = self.torch.remainder(safe_top_indices, grid_width)
            top_row_indices = self.torch.arange(object_count, device=box_delta_maps.device)[:, None].expand(-1, topk_count)
            top_box_raw = box_delta_maps[
                top_row_indices,
                top_grid_y.to(box_delta_maps.device),
                top_grid_x.to(box_delta_maps.device),
            ].to(self.torch.float32)
            raw_top_values = self.torch.gather(flat_scores, 1, safe_top_indices)
            safe_top_values = self.torch.where(finite_top, raw_top_values, self.torch.full_like(raw_top_values, -30.0))
            top_confidences = self.torch.sigmoid(safe_top_values)
            if flat_visibility is not None:
                top_visibility = self.torch.gather(flat_visibility, 1, safe_top_indices)
                top_visibility = self.torch.where(finite_top, top_visibility, self.torch.full_like(top_visibility, 0.5))
            else:
                top_visibility = self.torch.full_like(top_confidences, 0.5)
            top_decoded = self._decode_local_ltrb(
                top_grid_x.to(self.torch.float32),
                top_grid_y.to(self.torch.float32),
                top_box_raw,
                search_windows,
                grid_width,
                grid_height,
            )
            top_packed_tensor = self.torch.stack(
                (
                    top_confidences,
                    top_grid_x.to(self.torch.float32),
                    top_grid_y.to(self.torch.float32),
                    top_decoded[:, :, 0],
                    top_decoded[:, :, 1],
                    top_decoded[:, :, 2],
                    top_decoded[:, :, 3],
                    top_visibility,
                    finite_top.to(self.torch.float32),
                ),
                dim=2,
            )
        else:
            top_packed_tensor = None

        grid_y = self.torch.div(safe_best_indices, grid_width, rounding_mode="floor")
        grid_x = self.torch.remainder(safe_best_indices, grid_width)
        row_indices = self.torch.arange(object_count, device=box_delta_maps.device)
        box_raw = box_delta_maps[row_indices, grid_y.to(box_delta_maps.device), grid_x.to(box_delta_maps.device)].to(self.torch.float32)
        confidences = self.torch.sigmoid(safe_best_values)
        if flat_visibility is not None:
            visibilities = self.torch.gather(flat_visibility, 1, safe_best_indices[:, None]).squeeze(1)
            visibilities = self.torch.where(finite_best, visibilities, self.torch.full_like(visibilities, 0.5))
        else:
            visibilities = self.torch.full_like(confidences, 0.5)
        decoded = self._decode_local_ltrb(
            grid_x.to(self.torch.float32)[:, None],
            grid_y.to(self.torch.float32)[:, None],
            box_raw[:, None, :],
            search_windows,
            grid_width,
            grid_height,
        )[:, 0, :]
        packed_tensor = self.torch.stack(
            (
                confidences,
                margins,
                grid_x.to(self.torch.float32),
                grid_y.to(self.torch.float32),
                decoded[:, 0],
                decoded[:, 1],
                decoded[:, 2],
                decoded[:, 3],
                visibilities,
            ),
            dim=1,
        )

        transfer_started = time.perf_counter()
        packed = packed_tensor.detach().cpu().tolist()
        if top_packed_tensor is not None:
            top_packed = top_packed_tensor.detach().cpu().tolist()
        else:
            top_packed = []
        transfer_seconds = time.perf_counter() - transfer_started

        candidates: List[runtime.HeadCandidateInfo] = []
        roi_tokens = grid_width * grid_height
        for row_index, row in enumerate(packed):
            confidence, margin, grid_x_value, grid_y_value, bbox_x, bbox_y, bbox_w, bbox_h, visibility = row
            top_candidates: List[runtime.HeadCandidate] = []
            for rank, top_row in enumerate(top_packed[row_index] if top_packed else [], start=1):
                top_confidence, top_grid_x_value, top_grid_y_value, top_x, top_y, top_w, top_h, top_visibility, top_finite = top_row
                if float(top_finite) <= 0.0:
                    continue
                top_candidate = runtime.HeadCandidate(
                    rank=rank,
                    bbox=mot.clamp_bbox_size((top_x, top_y, top_w, top_h)),
                    confidence=float(top_confidence),
                    grid_x=int(np.clip(round(float(top_grid_x_value)), 0, max(0, grid_width - 1))),
                    grid_y=int(np.clip(round(float(top_grid_y_value)), 0, max(0, grid_height - 1))),
                )
                object.__setattr__(top_candidate, "visibility", float(top_visibility))
                top_candidates.append(top_candidate)
            candidate_info = runtime.HeadCandidateInfo(
                bbox=mot.clamp_bbox_size((bbox_x, bbox_y, bbox_w, bbox_h)),
                confidence=float(confidence),
                margin=max(0.0, float(margin)),
                roi_tokens=roi_tokens,
                top_candidates=tuple(top_candidates),
            )
            object.__setattr__(candidate_info, "visibility", float(visibility))
            object.__setattr__(candidate_info, "top_candidate_visibility", tuple(float(getattr(item, "visibility", 0.5)) for item in top_candidates))
            candidates.append(candidate_info)

        self._add_profile_seconds("candidate_transfer", transfer_seconds)
        self._add_profile_seconds("candidate_extract", max(0.0, time.perf_counter() - candidate_started - transfer_seconds))
        return candidates

    def _decode_local_ltrb(self, grid_x, grid_y, box_raw, search_windows: Sequence[BBox], grid_width: int, grid_height: int):
        windows = self.torch.tensor(search_windows, device=box_raw.device, dtype=self.torch.float32)
        ref_x = windows[:, None, 0] + ((grid_x + 0.5) / float(max(1, grid_width))) * windows[:, None, 2].clamp_min(1.0)
        ref_y = windows[:, None, 1] + ((grid_y + 0.5) / float(max(1, grid_height))) * windows[:, None, 3].clamp_min(1.0)
        ltrb = self.torch.sigmoid(self.torch.clamp(box_raw, -30.0, 30.0))
        x1 = ref_x - ltrb[:, :, 0] * windows[:, None, 2].clamp_min(1.0)
        y1 = ref_y - ltrb[:, :, 1] * windows[:, None, 3].clamp_min(1.0)
        x2 = ref_x + ltrb[:, :, 2] * windows[:, None, 2].clamp_min(1.0)
        y2 = ref_y + ltrb[:, :, 3] * windows[:, None, 3].clamp_min(1.0)
        return self.torch.stack(
            (
                x1,
                y1,
                self.torch.clamp(x2 - x1, min=1.0),
                self.torch.clamp(y2 - y1, min=1.0),
            ),
            dim=2,
        )

    def _attach_dinov2_crop_reid_features(
        self,
        frame: np.ndarray,
        outputs: Sequence[mot.LoRATSlotOutput],
    ) -> None:
        if not self.v9_protective_reid:
            return super()._attach_dinov2_crop_reid_features(frame, outputs)
        filtered = [
            output
            for output in outputs
            if bool(getattr(output, "v9_crop_reid_allowed", True))
        ]
        if filtered:
            super()._attach_dinov2_crop_reid_features(frame, filtered)

    def _score_and_update_tracks(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        frame_number: int,
    ) -> None:
        selected_banks = [self._select_track_heads(track, frame_number) for track in tracks]
        predicted_bboxes = [self._predict_track(track) for track in tracks]
        search_windows = self._search_windows_for_tracks(predicted_bboxes, tracks, frame_number)
        local_grids = self._sample_local_search_grids(feature_map, search_windows, frame.shape)
        head_output = self.object_conditioned_head.score_local(local_grids, selected_banks, search_windows)
        self._last_head_seconds = head_output.elapsed_seconds
        self._last_head_mode = self.object_conditioned_head.last_mode
        self._last_selected_head_count = head_output.selected_head_count
        self.runtime_status.object_head_batches += 1
        self.runtime_status.object_head_items += len(tracks)
        self.runtime_status.max_object_head_batch = max(self.runtime_status.max_object_head_batch, len(tracks))
        self.runtime_status.fusion_forward_calls += 1
        self.runtime_status.fusion_forward_items += len(tracks)
        self.runtime_status.max_fusion_forward_batch = max(self.runtime_status.max_fusion_forward_batch, len(tracks))

        candidate_infos = self._candidates_from_local_head_output(
            head_output.score_maps,
            head_output.box_delta_maps,
            head_output.visibility_maps,
            head_output.search_windows,
        )
        total_roi_tokens = 0
        records: Dict[int, Dict[str, object]] = {}
        diagnostics_by_track_id: Dict[int, Dict[str, object]] = {}
        candidate_outputs: List[mot.LoRATSlotOutput] = []
        for index, track in enumerate(tracks):
            predicted = predicted_bboxes[index]
            candidate_info = candidate_infos[index]
            candidate_info = self._rerank_head_candidate(
                feature_map,
                frame.shape,
                track,
                predicted,
                candidate_info,
            )
            scale_gate_info: Dict[str, object]
            candidate_info, scale_gate_info = self._apply_v9_scale_gate(track, candidate_info, frame.shape)
            head_candidate = candidate_info.bbox
            head_confidence = candidate_info.confidence
            head_margin = candidate_info.margin
            head_visibility = getattr(candidate_info, "visibility", None)
            head_visibility = None if head_visibility is None else max(0.0, min(1.0, float(head_visibility)))
            roi_tokens = candidate_info.roi_tokens
            template_started = time.perf_counter()
            template_candidate: Optional[BBox] = None
            template_confidence = 0.0
            template_attempted = self._should_run_template_match(head_confidence, head_margin)
            if template_attempted:
                self.v8_template_match_attempts += 1
                template_candidate, template_confidence = self._feature_template_candidate(feature_map, track, predicted, frame.shape)
            self._add_profile_seconds("template_match", time.perf_counter() - template_started)
            fusion_started = time.perf_counter()
            candidate, confidence, margin, candidate_source = self._fuse_head_and_template_candidate(
                frame,
                track,
                head_candidate,
                head_confidence,
                head_margin,
                template_candidate,
                template_confidence,
            )
            self._add_profile_seconds("candidate_fusion", time.perf_counter() - fusion_started)
            fused_candidate = candidate
            fused_confidence = confidence
            fused_margin = margin
            fused_source = candidate_source
            total_roi_tokens += roi_tokens
            slot = self._synthetic_head_slot(track, frame_number)
            continuity_started = time.perf_counter()
            continuity_result = self._v9_select_continuity_candidate(
                frame,
                feature_map,
                tracks,
                track,
                slot,
                predicted,
                search_windows[index],
                candidate_info,
                scale_gate_info,
                candidate,
                confidence,
                margin,
                candidate_source,
            )
            self._add_profile_seconds("selected_continuity", time.perf_counter() - continuity_started)
            continuity_diagnostic = dict(continuity_result.get("diagnostic") or {})
            candidate = continuity_result["bbox"]  # type: ignore[assignment]
            confidence = float(continuity_result.get("confidence") or confidence)
            margin = float(continuity_result.get("margin") or margin)
            candidate_source = str(continuity_result.get("source") or candidate_source)
            scale_gate_info = dict(continuity_result.get("scale_gate_info") or scale_gate_info)
            output = continuity_result["output"]  # type: ignore[assignment]
            selected_visibility = getattr(output, "v9_visibility_score", head_visibility)
            crop_reid_allowed = self._v9_should_attach_crop_reid(
                track,
                candidate,
                confidence,
                margin,
                str(scale_gate_info.get("v9_scale_gate_state", "pass")),
                candidate_source,
                tracks,
                selected_visibility,
            )
            setattr(output, "v9_crop_reid_allowed", crop_reid_allowed)
            records[track.track_id] = {
                "track": track,
                "predicted": predicted,
                "search_window": search_windows[index],
                "head_candidate": head_candidate,
                "head_confidence": head_confidence,
                "head_margin": head_margin,
                "head_visibility": head_visibility,
                "candidate_visibility": selected_visibility,
                "head_top_candidates": candidate_info.top_candidates,
                "scale_gate_info": scale_gate_info,
                "template_candidate": template_candidate,
                "candidate": candidate,
                "confidence": confidence,
                "margin": margin,
                "candidate_source": candidate_source,
                "template_confidence": template_confidence,
                "slot": slot,
                "output": output,
                "crop_reid_allowed": crop_reid_allowed,
                **continuity_diagnostic,
            }
            diagnostics_by_track_id[track.track_id] = {
                "frame": frame_number,
                "track_id": track.track_id,
                "v9_diagnostic_mode": self.v9_diagnostic_mode,
                "search_window": search_windows[index],
                "v9_search_anchor_bbox": getattr(track, "v9_search_anchor_bbox", None),
                "v9_search_anchor_source": getattr(track, "v9_search_anchor_source", ""),
                "v9_search_anchor_age": getattr(track, "v9_search_anchor_age", ""),
                "v9_last_accepted_bbox": getattr(track, "v9_last_accepted_bbox", None),
                "v9_last_accepted_frame": getattr(track, "v9_last_accepted_frame", ""),
                "v9_last_accepted_source": getattr(track, "v9_last_accepted_source", ""),
                "v9_search_window_contains_final_center": self._v9_bbox_center_inside(search_windows[index], candidate),
                "v9_search_window_contains_head_center": self._v9_bbox_center_inside(search_windows[index], head_candidate),
                "previous_bbox": track.previous_bbox,
                "predicted_bbox": predicted,
                "head_original_bbox": scale_gate_info.get("v9_scale_gate_original_bbox"),
                "head_original_confidence": scale_gate_info.get("v9_scale_gate_original_confidence"),
                "head_bbox": head_candidate,
                "head_confidence": head_confidence,
                "head_margin": head_margin,
                "head_visibility": head_visibility,
                "candidate_visibility": selected_visibility,
                "head_roi_tokens": roi_tokens,
                "head_top_candidates": candidate_info.top_candidates,
                "v9_scale_gate_state": scale_gate_info.get("v9_scale_gate_state"),
                "v9_scale_gate_reason": scale_gate_info.get("v9_scale_gate_reason"),
                "v9_scale_gate_width_ratio": scale_gate_info.get("v9_scale_gate_width_ratio"),
                "v9_scale_gate_height_ratio": scale_gate_info.get("v9_scale_gate_height_ratio"),
                "v9_scale_gate_locked_bbox": scale_gate_info.get("v9_scale_gate_locked_bbox"),
                "v9_scale_gate_locked_width_ratio": scale_gate_info.get("v9_scale_gate_locked_width_ratio"),
                "v9_scale_gate_locked_height_ratio": scale_gate_info.get("v9_scale_gate_locked_height_ratio"),
                "v9_scale_gate_suppressed_original": scale_gate_info.get("v9_scale_gate_suppressed_original"),
                "v9_scale_gate_confidence_preserved": scale_gate_info.get("v9_scale_gate_confidence_preserved"),
                "v9_scale_candidate_original_score": scale_gate_info.get("v9_scale_candidate_original_score"),
                "v9_scale_candidate_locked_score": scale_gate_info.get("v9_scale_candidate_locked_score"),
                "v9_scale_candidate_selected": scale_gate_info.get("v9_scale_candidate_selected"),
                "template_attempted": template_attempted,
                "template_bbox": template_candidate,
                "template_confidence": template_confidence,
                "fused_bbox": fused_candidate,
                "fused_confidence": fused_confidence,
                "fused_margin": fused_margin,
                "fused_source": fused_source,
                "candidate_source": candidate_source,
                **continuity_diagnostic,
                "v9_crop_reid_allowed": crop_reid_allowed,
                "v9_local_health_tier": "",
                "v9_local_health_confidence_threshold": "",
                "v9_local_health_margin_threshold": "",
                "v9_local_health_motion": "",
                "v9_local_health_accepted_anchor_motion": "",
                "v9_local_health_path": "",
                "v9_local_health_continuity_score": "",
                "v9_local_health_identity_risk": "",
                "v9_local_health_visibility": "",
                "v9_local_health_ok": False,
                "v9_local_health_reason": "",
                "v9_association_stage": "",
                "reid_outcome": "",
                "reid_prevented_switch": False,
                "reid_caused_hold": False,
                "reid_recovered_lost": False,
                "reid_wrong_reattach": False,
                "reid_noop_bad_candidate_pool": False,
                "reid_skipped_healthy_local": False,
                "reid_next_best_attempted": False,
                "reid_next_best_accepted": False,
                "reid_next_best_source": "",
                "reid_next_best_reason": "",
                "v9_local_rescue_accept": False,
                "v9_local_rescue_reject_state": "",
            }
            candidate_outputs.append(output)

        local_health_by_track_id: Dict[int, Tuple[bool, str]] = {}
        for track in tracks:
            health_ok, health_reason = self._v9_local_candidate_health(track, records.get(track.track_id), tracks)
            local_health_by_track_id[track.track_id] = (health_ok, health_reason)
            record = records.get(track.track_id)
            diagnostic = diagnostics_by_track_id.get(track.track_id)
            if diagnostic is not None:
                diagnostic["v9_local_health_ok"] = health_ok
                diagnostic["v9_local_health_reason"] = health_reason
                if record is not None:
                    diagnostic.update(dict(record.get("_v9_local_health_details") or {}))
            if health_ok and record is not None:
                output = record.get("output")
                if output is not None:
                    setattr(output, "v9_crop_reid_allowed", False)
                record["crop_reid_allowed"] = False
                if diagnostic is not None:
                    diagnostic["v9_crop_reid_allowed"] = False
                    diagnostic["reid_outcome"] = "reid_skipped_healthy_local"
                    diagnostic["reid_skipped_healthy_local"] = True
            elif record is not None:
                output = record.get("output")
                if output is not None:
                    setattr(output, "v9_crop_reid_allowed", True)
                record["crop_reid_allowed"] = True
                if diagnostic is not None:
                    diagnostic["v9_crop_reid_allowed"] = True

        if self.identity_arbitrator.enabled and self.v8_dinov2_crop_reid:
            self._attach_dinov2_crop_reid_features(frame, candidate_outputs)

        if self.collect_slot_debug:
            debug_started = time.perf_counter()
            self._append_head_debug_rows(frame_number, tracks, candidate_outputs)
            self._add_profile_seconds("debug_output", time.perf_counter() - debug_started)
        resolve_started = time.perf_counter()
        assignments = self.identity_arbitrator.resolve(tracks, candidate_outputs, None)
        self._add_profile_seconds("identity_resolve", time.perf_counter() - resolve_started)
        assignment_by_track_id = {assignment.track.track_id: assignment for assignment in assignments}
        assigned_output_keys = {
            (assignment.output.source_track_id, assignment.output.slot.task_id)
            for assignment in assignments
        }
        assignment_reject_by_track_id: Dict[int, str] = {}
        source_record_by_track_id: Dict[int, Dict[str, object]] = {}

        for track in tracks:
            identity_assignment = assignment_by_track_id.get(track.track_id)
            if identity_assignment is not None:
                source_record = records.get(identity_assignment.output.source_track_id)
            else:
                source_record = records.get(track.track_id)
                if source_record is not None and (
                    track.track_id,
                    source_record["slot"].task_id,  # type: ignore[union-attr]
                ) not in assigned_output_keys:
                    output = source_record["output"]
                    score_started = time.perf_counter()
                    score = self.identity_arbitrator.score_from_cached_matrices(
                        tracks,
                        candidate_outputs,
                        track,
                        output,  # type: ignore[arg-type]
                    )
                    self._add_profile_seconds("identity_score", time.perf_counter() - score_started)
                    assignment_ok, reject_state = self.identity_arbitrator.assignment_gate(
                        track,
                        output,  # type: ignore[arg-type]
                        score,
                    )
                    if assignment_ok:
                        identity_assignment = mot.IdentityAssignment(
                            track=track,
                            output=output,  # type: ignore[arg-type]
                            score=score,
                            assignment_margin=float(source_record["margin"]),
                        )
                    else:
                        if self.identity_arbitrator._should_remember_negative_reject(reject_state):
                            self.identity_arbitrator.remember_negative_candidate(track, output)  # type: ignore[arg-type]
                        assignment_reject_by_track_id[track.track_id] = reject_state
                else:
                    source_record = None
            if source_record is not None and identity_assignment is not None:
                assignment_by_track_id[track.track_id] = identity_assignment
                source_record_by_track_id[track.track_id] = source_record
            elif track.track_id not in assignment_reject_by_track_id:
                assignment_by_track_id.pop(track.track_id, None)
                assignment_reject_by_track_id[track.track_id] = "NO_ASSIGNMENT"

        protected_local_ids = {
            track_id
            for track_id, (health_ok, _) in local_health_by_track_id.items()
            if health_ok
        }
        for assigned_track_id, assignment in list(assignment_by_track_id.items()):
            source_track_id = int(assignment.output.source_track_id)
            if source_track_id not in protected_local_ids or assigned_track_id == source_track_id:
                continue
            assignment_by_track_id.pop(assigned_track_id, None)
            source_record_by_track_id.pop(assigned_track_id, None)
            assignment_reject_by_track_id[assigned_track_id] = "REID_BLOCKED_BY_HEALTHY_LOCAL_OWNER"
            diagnostic = diagnostics_by_track_id.get(assigned_track_id)
            if diagnostic is not None:
                diagnostic["reid_outcome"] = "reid_prevented_switch"
                diagnostic["reid_prevented_switch"] = True
                diagnostic["reject_state"] = "REID_BLOCKED_BY_HEALTHY_LOCAL_OWNER"

        self._resolve_assignment_spatial_conflicts(
            assignment_by_track_id,
            source_record_by_track_id,
            assignment_reject_by_track_id,
            tracks,
            frame,
            feature_map,
            frame.shape,
        )

        for track in tracks:
            identity_assignment = assignment_by_track_id.get(track.track_id)
            source_record = source_record_by_track_id.get(track.track_id)
            own_record = records.get(track.track_id)
            hold_predicted = own_record["predicted"] if own_record is not None else track.predicted_bbox or track.bbox
            hold_confidence = float(own_record["confidence"]) if own_record is not None else 0.0
            hold_margin = float(own_record["margin"]) if own_record is not None else 0.0
            diagnostic = diagnostics_by_track_id.get(track.track_id)
            local_health_ok, local_health_reason = local_health_by_track_id.get(track.track_id, (False, "not_checked"))
            if local_health_ok and own_record is not None:
                stage1_accepted, stage1_assignment, stage1_source = self._v9_accept_record_candidate(
                    frame,
                    feature_map,
                    tracks,
                    track,
                    own_record,
                    frame_number,
                    "v9stage1",
                )
                if diagnostic is not None:
                    diagnostic.update(
                        {
                            "assigned_source_track_id": track.track_id,
                            "assigned_bbox": own_record.get("candidate"),
                            "assigned_confidence": own_record.get("confidence"),
                            "assignment_score": stage1_assignment.score.total if stage1_assignment is not None else own_record.get("confidence"),
                            "assignment_margin": stage1_assignment.assignment_margin if stage1_assignment is not None else own_record.get("margin"),
                            "reject_state": "" if stage1_accepted else str(getattr(track, "state", "")),
                            "accepted": bool(stage1_accepted),
                            "held": False,
                            "v9_accept_guard_state": "",
                            "v9_local_owner_override": True,
                            "v9_association_stage": "stage1_local_continuity",
                            "v9_hold_source": "",
                            "reid_outcome": "reid_skipped_healthy_local" if stage1_accepted else "healthy_local_accept_rejected",
                            "reid_skipped_healthy_local": bool(stage1_accepted),
                            "v9_local_rescue_accept": bool(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                            "v9_local_rescue_reject_state": str(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                            "final_bbox": track.bbox,
                            "final_confidence": track.confidence,
                            "final_state": track.state,
                            "assigned_source": track.assigned_source or stage1_source,
                        }
                    )
                if stage1_accepted:
                    continue

            if source_record is None or identity_assignment is None:
                reject_state = assignment_reject_by_track_id.get(track.track_id, "NO_ASSIGNMENT")
                fallback_source_record = own_record
                fallback_assignment = None
                if fallback_source_record is not None:
                    fallback_output = fallback_source_record["output"]
                    try:
                        fallback_score = self.identity_arbitrator.score_from_cached_matrices(
                            tracks,
                            candidate_outputs,
                            track,
                            fallback_output,  # type: ignore[arg-type]
                        )
                        fallback_assignment = mot.IdentityAssignment(
                            track=track,
                            output=fallback_output,  # type: ignore[arg-type]
                            score=fallback_score,
                            assignment_margin=float(fallback_source_record.get("margin") or 0.0),
                        )
                    except Exception:
                        fallback_assignment = None
                hold_bbox, hold_confidence, hold_margin, hold_source = self._v9_select_hold_bbox(
                    track,
                    hold_predicted,
                    fallback_source_record,
                    fallback_assignment,
                    reject_state,
                )
                fallback_accepted = False
                fallback_accept_guard = ""
                if hold_source == "v9-local-hold" and fallback_source_record is not None and fallback_assignment is not None:
                    fallback_scale_info = dict(fallback_source_record.get("scale_gate_info") or {})
                    fallback_gate_state = str(fallback_scale_info.get("v9_scale_gate_state", "pass"))
                    fallback_accept_guard = self._v9_candidate_local_reject_state(
                        track,
                        fallback_assignment.output.bbox,
                        hold_predicted,
                        fallback_source_record["search_window"],  # type: ignore[arg-type]
                        float(fallback_assignment.output.confidence or 0.0),
                        fallback_gate_state,
                    ) or ""
                    if not fallback_accept_guard:
                        fallback_source = str(fallback_source_record.get("candidate_source", "head"))
                        if self._v9_is_scale_modified_state(fallback_gate_state):
                            fallback_source = f"{fallback_source}-v9scale"
                        fallback_source = f"{fallback_source}-v9local"
                        fallback_accepted = self._accept_candidate(
                            frame,
                            feature_map,
                            track,
                            fallback_assignment.output.bbox,
                            float(fallback_assignment.output.confidence or 0.0),
                            fallback_assignment.assignment_margin,
                            hold_predicted,
                            frame_number,
                            fallback_assignment,
                            fallback_source,
                        )
                if fallback_accepted:
                    if diagnostic is not None:
                        diagnostic.update(
                            {
                                "assigned_source_track_id": fallback_assignment.output.source_track_id if fallback_assignment is not None else "",
                                "assigned_bbox": fallback_assignment.output.bbox if fallback_assignment is not None else None,
                                "assigned_confidence": fallback_assignment.output.confidence if fallback_assignment is not None else None,
                                "assignment_score": fallback_assignment.score.total if fallback_assignment is not None else None,
                                "assignment_margin": fallback_assignment.assignment_margin if fallback_assignment is not None else None,
                                "reject_state": reject_state,
                                "accepted": True,
                                "held": False,
                                "v9_accept_guard_state": "",
                                "v9_local_owner_override": True,
                                "v9_association_stage": "stage2_local_after_assignment_reject",
                                "v9_hold_source": "v9-local-accept-after-assignment-reject",
                                "reid_outcome": "reid_local_recovered_after_reject",
                                "reid_caused_hold": False,
                                "reid_next_best_attempted": False,
                                "reid_next_best_accepted": False,
                                "v9_local_rescue_accept": bool(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "v9_local_rescue_reject_state": str(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "final_bbox": track.bbox,
                                "final_confidence": track.confidence,
                                "final_state": track.state,
                                "assigned_source": track.assigned_source,
                            }
                        )
                    continue
                next_accepted, next_assignment, next_source, next_reason = self._v9_try_next_best_local_candidate(
                    frame,
                    feature_map,
                    tracks,
                    track,
                    own_record,
                    frame_number,
                    reject_state,
                )
                if next_accepted:
                    if diagnostic is not None:
                        diagnostic.update(
                            {
                                "assigned_source_track_id": next_assignment.output.source_track_id if next_assignment is not None else track.track_id,
                                "assigned_bbox": track.bbox,
                                "assigned_confidence": track.confidence,
                                "assignment_score": next_assignment.score.total if next_assignment is not None else track.assignment_score,
                                "assignment_margin": next_assignment.assignment_margin if next_assignment is not None else track.assignment_margin,
                                "reject_state": reject_state,
                                "accepted": True,
                                "held": False,
                                "v9_accept_guard_state": "",
                                "v9_local_owner_override": True,
                                "v9_association_stage": "stage2_next_best_local",
                                "v9_hold_source": "",
                                "reid_outcome": "reid_next_best_local_recovery",
                                "reid_caused_hold": False,
                                "reid_next_best_attempted": True,
                                "reid_next_best_accepted": True,
                                "reid_next_best_source": next_source,
                                "reid_next_best_reason": next_reason,
                                "v9_local_rescue_accept": bool(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "v9_local_rescue_reject_state": str(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "final_bbox": track.bbox,
                                "final_confidence": track.confidence,
                                "final_state": track.state,
                                "assigned_source": track.assigned_source,
                            }
                        )
                    continue
                hold_started = time.perf_counter()
                self._hold_track(track, hold_bbox, hold_confidence, hold_margin, frame_number, frame, reject_state)
                if hold_source == "v9-local-hold":
                    track.assigned_source = "v9-local-hold"
                    track.state = mot.append_state_token(track.state, "V9LOCALHOLD")
                self._add_profile_seconds("hold", time.perf_counter() - hold_started)
                if diagnostic is not None:
                    diagnostic.update(
                        {
                            "assigned_source_track_id": "",
                            "assigned_bbox": None,
                            "assigned_confidence": None,
                            "assignment_score": None,
                            "assignment_margin": None,
                            "reject_state": reject_state,
                            "accepted": False,
                            "held": True,
                            "v9_accept_guard_state": fallback_accept_guard,
                            "v9_association_stage": "hold_after_stage2",
                            "v9_hold_source": hold_source,
                            "reid_outcome": "reid_caused_hold" if self._v9_reid_reject_is_identity(reject_state) else "local_hold",
                            "reid_caused_hold": self._v9_reid_reject_is_identity(reject_state),
                            "reid_noop_bad_candidate_pool": next_reason
                            in {"no_record", "no_top_candidates", "missing_geometry", "no_viable_top_candidate"},
                            "reid_next_best_attempted": bool(own_record is not None),
                            "reid_next_best_accepted": False,
                            "reid_next_best_reason": next_reason,
                            "v9_local_rescue_accept": False,
                            "v9_local_rescue_reject_state": "",
                            "final_bbox": track.bbox,
                            "final_confidence": track.confidence,
                            "final_state": track.state,
                            "assigned_source": track.assigned_source,
                        }
                    )
                continue

            if diagnostic is not None:
                diagnostic.update(
                    {
                        "assigned_source_track_id": identity_assignment.output.source_track_id,
                        "assigned_bbox": identity_assignment.output.bbox,
                        "assigned_confidence": identity_assignment.output.confidence,
                        "assignment_score": identity_assignment.score.total,
                        "assignment_margin": identity_assignment.assignment_margin,
                    }
                )
            accept_started = time.perf_counter()
            local_owner_override = False
            reject_state = ""
            was_uncertain_before_accept = self._v9_track_uncertain(track)
            if self._v9_should_prefer_local_owner(track, source_record, own_record, identity_assignment):
                local_owner_override = True
                source_record = own_record
                output = own_record["output"]  # type: ignore[index]
                score_started = time.perf_counter()
                score = self.identity_arbitrator.score_from_cached_matrices(
                    tracks,
                    candidate_outputs,
                    track,
                    output,  # type: ignore[arg-type]
                )
                self._add_profile_seconds("identity_score", time.perf_counter() - score_started)
                identity_assignment = mot.IdentityAssignment(
                    track=track,
                    output=output,  # type: ignore[arg-type]
                    score=score,
                    assignment_margin=float(own_record.get("margin") or 0.0),  # type: ignore[union-attr]
                )
                if diagnostic is not None:
                    diagnostic["reid_outcome"] = "reid_prevented_switch"
                    diagnostic["reid_prevented_switch"] = True
            scale_gate_info = dict(source_record.get("scale_gate_info") or {})
            v9_gate_state = str(scale_gate_info.get("v9_scale_gate_state", "pass"))
            setattr(track, "v9_last_scale_gate_state", v9_gate_state)
            candidate_source = str(source_record.get("candidate_source", "head"))
            if self._v9_is_scale_modified_state(v9_gate_state):
                candidate_source = f"{candidate_source}-v9scale"
            if identity_assignment.output.source_track_id == track.track_id:
                candidate_source = f"{candidate_source}-v9local"
            v9_accept_reject = self._v9_candidate_local_reject_state(
                track,
                identity_assignment.output.bbox,
                hold_predicted,
                source_record["search_window"],  # type: ignore[arg-type]
                float(identity_assignment.output.confidence or 0.0),
                v9_gate_state,
            )
            if v9_accept_reject is None:
                accepted = self._accept_candidate(
                    frame,
                    feature_map,
                    track,
                    identity_assignment.output.bbox,
                    float(identity_assignment.output.confidence or 0.0),
                    identity_assignment.assignment_margin,
                    hold_predicted,
                    frame_number,
                    identity_assignment,
                    candidate_source,
                )
                if not accepted:
                    reject_state = str(getattr(track, "state", ""))
            else:
                reject_state = v9_accept_reject
                accepted = False
            self._add_profile_seconds("accept", time.perf_counter() - accept_started)
            next_reason = ""
            if not accepted:
                reject_state = reject_state or str(getattr(track, "state", ""))
                next_accepted, next_assignment, next_source, next_reason = self._v9_try_next_best_local_candidate(
                    frame,
                    feature_map,
                    tracks,
                    track,
                    own_record,
                    frame_number,
                    reject_state,
                )
                if next_accepted:
                    if diagnostic is not None:
                        diagnostic.update(
                            {
                                "accepted": True,
                                "held": False,
                                "reject_state": reject_state,
                                "v9_accept_guard_state": "" if v9_accept_reject is None else v9_accept_reject,
                                "v9_local_owner_override": local_owner_override,
                                "v9_association_stage": "stage2_next_best_local",
                                "v9_hold_source": "",
                                "reid_outcome": "reid_next_best_local_recovery",
                                "reid_caused_hold": False,
                                "reid_next_best_attempted": True,
                                "reid_next_best_accepted": True,
                                "reid_next_best_source": next_source,
                                "reid_next_best_reason": next_reason,
                                "v9_local_rescue_accept": bool(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "v9_local_rescue_reject_state": str(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                                "final_bbox": track.bbox,
                                "final_confidence": track.confidence,
                                "final_state": track.state,
                                "assigned_source": track.assigned_source,
                            }
                        )
                    continue
                hold_bbox, hold_confidence_v9, hold_margin_v9, hold_source = self._v9_select_hold_bbox(
                    track,
                    hold_predicted,
                    source_record,
                    identity_assignment,
                    reject_state,
                )
                hold_started = time.perf_counter()
                self._hold_track(track, hold_bbox, hold_confidence_v9, hold_margin_v9, frame_number, frame, reject_state)
                if hold_source == "v9-local-hold":
                    track.assigned_source = "v9-local-hold"
                    track.state = mot.append_state_token(track.state, "V9LOCALHOLD")
                self._add_profile_seconds("hold", time.perf_counter() - hold_started)
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "accepted": bool(accepted),
                        "held": not bool(accepted),
                        "reject_state": "" if accepted else reject_state,
                        "v9_accept_guard_state": "" if v9_accept_reject is None else v9_accept_reject,
                        "v9_local_owner_override": local_owner_override,
                        "v9_association_stage": (
                            "stage2_local_owner_override"
                            if accepted and local_owner_override
                            else (
                                "stage2_owned_identity"
                                if accepted and identity_assignment.output.source_track_id == track.track_id
                                else ("stage2_reid_assignment" if accepted else "hold_after_stage2")
                            )
                        ),
                        "v9_hold_source": "" if accepted else hold_source,
                        "reid_outcome": diagnostic.get("reid_outcome") if diagnostic is not None and diagnostic.get("reid_outcome") else (
                            "reid_recovered_lost"
                            if accepted and was_uncertain_before_accept
                            else (
                                "reid_caused_hold"
                                if (not accepted and self._v9_reid_reject_is_identity(reject_state))
                                else ("reid_accepted" if accepted else "local_hold")
                            )
                        ),
                        "reid_caused_hold": bool((not accepted) and self._v9_reid_reject_is_identity(reject_state)),
                        "reid_recovered_lost": bool(accepted and was_uncertain_before_accept),
                        "reid_noop_bad_candidate_pool": bool(
                            (not accepted)
                            and next_reason in {"no_record", "no_top_candidates", "missing_geometry", "no_viable_top_candidate"}
                        ),
                        "reid_next_best_attempted": bool((not accepted) and own_record is not None),
                        "reid_next_best_accepted": False if not accepted else diagnostic.get("reid_next_best_accepted", False),
                        "reid_next_best_reason": next_reason,
                        "v9_local_rescue_accept": bool(accepted and (getattr(track, "v9_last_local_rescue_reject", "") or "")),
                        "v9_local_rescue_reject_state": str(getattr(track, "v9_last_local_rescue_reject", "") or ""),
                        "final_bbox": track.bbox,
                        "final_confidence": track.confidence,
                        "final_state": track.state,
                        "assigned_source": track.assigned_source,
                    }
                )

        self._last_roi_tokens = total_roi_tokens
        self.runtime_status.object_head_roi_tokens += total_roi_tokens
        for diagnostic in diagnostics_by_track_id.values():
            search_window = diagnostic.get("search_window")
            final_bbox = diagnostic.get("final_bbox")
            head_bbox = diagnostic.get("head_bbox")
            if search_window is not None and final_bbox is not None:
                diagnostic["v9_search_window_contains_final_center"] = self._v9_bbox_center_inside(
                    search_window,  # type: ignore[arg-type]
                    final_bbox,  # type: ignore[arg-type]
                )
            if search_window is not None and head_bbox is not None:
                diagnostic["v9_search_window_contains_head_center"] = self._v9_bbox_center_inside(
                    search_window,  # type: ignore[arg-type]
                    head_bbox,  # type: ignore[arg-type]
                )
        self.last_candidate_diagnostics = list(diagnostics_by_track_id.values())

    def _should_refresh_head_memory(
        self,
        track: mot.TrackState,
        confidence: float,
        frame_number: int,
        candidate_source: str = "head",
    ) -> bool:
        source = str(candidate_source or "")
        if str(getattr(track, "v9_last_local_rescue_reject", "") or ""):
            return False
        if "v9driftrisk" in source:
            return False
        if "v9scale" in source:
            return False
        if "hold" in source.lower():
            return False
        if "v9next" in source:
            return False
        if "reid" in source.lower() and "v9local" not in source:
            return False
        state_upper = str(getattr(track, "state", "") or "").upper()
        if any(token in state_upper for token in ("CONFLICT", "CONTEST", "ID_UNCERTAIN", "V9_DRIFT_RISK")):
            return False
        if self._v9_is_scale_modified_state(str(getattr(track, "v9_last_scale_gate_state", "pass"))):
            return False
        if float(confidence or 0.0) < self.v9_stage1_local_min_confidence:
            return False
        if track.assignment_margin is not None and float(track.assignment_margin) < self.v9_protective_reid_margin_gate:
            return False
        if track.identity_margin is not None and float(track.identity_margin) < self.v8_memory_min_identity_margin:
            return False
        if track.reid_score is not None and self.identity_arbitrator is not None:
            if float(track.reid_score) < max(self.identity_arbitrator.min_reid, self.v8_memory_min_appearance):
                return False
        return super()._should_refresh_head_memory(track, confidence, frame_number, candidate_source)

    def status_lines(self) -> List[str]:
        lines = super().status_lines()
        if lines:
            lines[0] = lines[0].replace(runtime.BASE_EXECUTION_MODE, V9_EXECUTION_MODE)
        lines.append(f"V9 local search grid {self.v9_local_grid_size}x{self.v9_local_grid_size}")
        return lines


def create_backend(args, source: mot.FrameSource, expected_tracks: int = 0):
    weight_path = args.weight_path or mot.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return V9LocalSearchLoRATTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        args.max_tracks,
        source.fps,
        source.length,
        source.name,
        args.disable_amp,
        args.v8_frame_size,
        args.v8_head_rank,
        args.v8_head_hidden_dim,
        args.v8_head_lora_rank,
        args.v8_head_weights,
        args.v8_search_radius_factor,
        args.v8_min_confidence,
        args.v8_template_update_rate,
        args.v8_template_update_min_confidence,
        args.lorat_memory_slots,
        args.lorat_memory_refresh_interval,
        args.lorat_memory_min_score,
        args.lorat_accept_min_score,
        args.lorat_fixed_box_size,
        args.lorat_min_box_area,
        args.lorat_max_area_change_per_frame,
        args.lorat_trusted_size_floor_scale,
        args.shrink_guard_window,
        args.shrink_guard_area_ratio,
        args.shrink_guard_step_ratio,
        args.shrink_guard_min_confidence,
        args.shrink_guard_min_reid,
        args.crop_information_min_score,
        args.crop_information_min_pixels,
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
        args.v8_primary_heads_per_track,
        args.v8_recovery_heads_per_track,
        args.v8_recovery_interval,
        args.v8_recovery_min_confidence,
        args.v8_recovery_min_assignment_score,
        args.v8_recovery_min_assignment_margin,
        args.v8_recovery_stale_head_frames,
        args.v8_score_reduction,
        bool(getattr(args, "slot_debug_log", None)) and not getattr(args, "no_slot_debug_log", False),
        bool(getattr(args, "week2_proof_log", None)) or bool(getattr(args, "collect_week2_proof", False)),
        args.v8_template_match,
        args.v8_template_match_min_score,
        args.v8_template_match_prefer_margin,
        args.v8_template_match_on_uncertain_only,
        args.v8_template_match_head_confidence_gate,
        args.v8_template_match_margin_gate,
        args.v8_head_template_blend,
        args.v8_memory_min_motion,
        args.v8_memory_min_path,
        args.v8_memory_min_appearance,
        args.v8_memory_min_stable_updates,
        args.v8_accept_min_initial_anchor,
        args.v8_accept_min_identity_margin,
        args.v8_memory_min_initial_anchor,
        args.v8_memory_min_identity_margin,
        args.v8_window_penalty_ratio,
        args.v8_dinov2_crop_reid,
        args.v8_dinov2_crop_reid_batch,
        args.v8_dinov2_crop_reid_min_area,
        args.v8_assignment_conflict_iou,
        args.v8_assignment_conflict_hard_iou,
        args.v8_assignment_conflict_score_margin,
        args.v8_assignment_conflict_center_ratio,
        args.v8_assignment_conflict_containment,
        args.v8_assignment_conflict_ownership_margin,
        args.v8_assignment_alt_rescue,
        args.v8_assignment_alt_rescue_max_candidates,
        args.v8_assignment_alt_rescue_min_confidence,
        getattr(args, "v8_small_target_mode", DEFAULT_V9_SMALL_TARGET_MODE),
        getattr(args, "v8_small_target_area", DEFAULT_V9_SMALL_TARGET_AREA),
        getattr(args, "v8_small_target_max_side", DEFAULT_V9_SMALL_TARGET_MAX_SIDE),
        getattr(args, "v8_small_target_max_scale_change", DEFAULT_V9_SMALL_TARGET_MAX_SCALE_CHANGE),
        getattr(args, "v8_small_target_template_min_score", DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_SCORE),
        getattr(args, "v8_small_target_template_min_motion", DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_MOTION),
        getattr(args, "v8_small_target_template_min_path", DEFAULT_V9_SMALL_TARGET_TEMPLATE_MIN_PATH),
        getattr(args, "v8_small_target_confidence_floor", DEFAULT_V9_SMALL_TARGET_CONFIDENCE_FLOOR),
        v9_local_grid_size=getattr(args, "v9_local_grid_size", DEFAULT_V9_LOCAL_GRID_SIZE),
        v9_diagnostic_mode=getattr(args, "v9_diagnostic_mode", "normal"),
        v9_scale_gate_min_ratio=getattr(args, "v9_scale_gate_min_ratio", DEFAULT_V9_SCALE_GATE_MIN_RATIO),
        v9_scale_gate_max_ratio=getattr(args, "v9_scale_gate_max_ratio", DEFAULT_V9_SCALE_GATE_MAX_RATIO),
        v9_scale_gate_override_confidence=getattr(
            args,
            "v9_scale_gate_override_confidence",
            DEFAULT_V9_SCALE_GATE_OVERRIDE_CONFIDENCE,
        ),
        v9_scale_gate_fallback_confidence_scale=getattr(
            args,
            "v9_scale_gate_fallback_confidence_scale",
            DEFAULT_V9_SCALE_GATE_FALLBACK_CONFIDENCE_SCALE,
        ),
        v9_protective_reid=getattr(args, "v9_protective_reid", DEFAULT_V9_PROTECTIVE_REID),
        v9_protective_reid_confidence_gate=getattr(
            args,
            "v9_protective_reid_confidence_gate",
            DEFAULT_V9_PROTECTIVE_REID_CONFIDENCE_GATE,
        ),
        v9_protective_reid_margin_gate=getattr(
            args,
            "v9_protective_reid_margin_gate",
            DEFAULT_V9_PROTECTIVE_REID_MARGIN_GATE,
        ),
        v9_protective_reid_overlap_iou=getattr(
            args,
            "v9_protective_reid_overlap_iou",
            DEFAULT_V9_PROTECTIVE_REID_OVERLAP_IOU,
        ),
        v9_stage1_local_min_confidence=getattr(
            args,
            "v9_stage1_local_min_confidence",
            DEFAULT_V9_STAGE1_LOCAL_MIN_CONFIDENCE,
        ),
        v9_stage1_local_min_margin=getattr(
            args,
            "v9_stage1_local_min_margin",
            DEFAULT_V9_STAGE1_LOCAL_MIN_MARGIN,
        ),
        v9_stage1_local_min_motion=getattr(
            args,
            "v9_stage1_local_min_motion",
            DEFAULT_V9_STAGE1_LOCAL_MIN_MOTION,
        ),
        v9_stage1_local_min_path=getattr(
            args,
            "v9_stage1_local_min_path",
            DEFAULT_V9_STAGE1_LOCAL_MIN_PATH,
        ),
        v9_next_best_min_confidence=getattr(
            args,
            "v9_next_best_min_confidence",
            DEFAULT_V9_NEXT_BEST_MIN_CONFIDENCE,
        ),
        v9_next_best_max_candidates=getattr(
            args,
            "v9_next_best_max_candidates",
            DEFAULT_V9_NEXT_BEST_MAX_CANDIDATES,
        ),
        v9_local_rescue=getattr(args, "v9_local_rescue", DEFAULT_V9_LOCAL_RESCUE),
        v9_local_rescue_min_assignment_score=getattr(
            args,
            "v9_local_rescue_min_assignment_score",
            DEFAULT_V9_LOCAL_RESCUE_MIN_ASSIGNMENT_SCORE,
        ),
        v9_local_rescue_max_scale_error=getattr(
            args,
            "v9_local_rescue_max_scale_error",
            DEFAULT_V9_LOCAL_RESCUE_MAX_SCALE_ERROR,
        ),
        v9_continuity_enabled=getattr(args, "v9_continuity_enabled", DEFAULT_V9_CONTINUITY_ENABLED),
        v9_continuity_topk=getattr(args, "v9_continuity_topk", DEFAULT_V9_CONTINUITY_TOPK),
        v9_continuity_min_score=getattr(
            args,
            "v9_continuity_min_score",
            DEFAULT_V9_CONTINUITY_MIN_SCORE,
        ),
        v9_continuity_min_margin=getattr(
            args,
            "v9_continuity_min_margin",
            DEFAULT_V9_CONTINUITY_MIN_MARGIN,
        ),
        v9_accept_max_center_ratio=getattr(args, "v9_accept_max_center_ratio", DEFAULT_V9_ACCEPT_MAX_CENTER_RATIO),
        v9_accept_max_healthy_center_ratio=getattr(
            args,
            "v9_accept_max_healthy_center_ratio",
            DEFAULT_V9_ACCEPT_MAX_HEALTHY_CENTER_RATIO,
        ),
        v9_local_hold_min_confidence=getattr(
            args,
            "v9_local_hold_min_confidence",
            DEFAULT_V9_LOCAL_HOLD_MIN_CONFIDENCE,
        ),
        v9_local_hold_max_center_ratio=getattr(
            args,
            "v9_local_hold_max_center_ratio",
            DEFAULT_V9_LOCAL_HOLD_MAX_CENTER_RATIO,
        ),
        v9_local_hold_max_lost_center_ratio=getattr(
            args,
            "v9_local_hold_max_lost_center_ratio",
            DEFAULT_V9_LOCAL_HOLD_MAX_LOST_CENTER_RATIO,
        ),
    )


def build_v9_front_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v9-local-grid-size", type=int, default=DEFAULT_V9_LOCAL_GRID_SIZE)
    parser.add_argument("--v9-diagnostic-mode", choices=V9_DIAGNOSTIC_MODES, default="normal")
    parser.add_argument("--v9-scale-gate-min-ratio", type=float, default=DEFAULT_V9_SCALE_GATE_MIN_RATIO)
    parser.add_argument("--v9-scale-gate-max-ratio", type=float, default=DEFAULT_V9_SCALE_GATE_MAX_RATIO)
    parser.add_argument("--v9-scale-gate-override-confidence", type=float, default=DEFAULT_V9_SCALE_GATE_OVERRIDE_CONFIDENCE)
    parser.add_argument(
        "--v9-scale-gate-fallback-confidence-scale",
        type=float,
        default=DEFAULT_V9_SCALE_GATE_FALLBACK_CONFIDENCE_SCALE,
    )
    parser.add_argument(
        "--v9-protective-reid",
        dest="v9_protective_reid",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_V9_PROTECTIVE_REID,
        help="Attach expensive/steering DINOv2 crop ReID only for uncertain, weak, or conflicted V9 tracks.",
    )
    parser.add_argument("--v9-protective-reid-confidence-gate", type=float, default=DEFAULT_V9_PROTECTIVE_REID_CONFIDENCE_GATE)
    parser.add_argument("--v9-protective-reid-margin-gate", type=float, default=DEFAULT_V9_PROTECTIVE_REID_MARGIN_GATE)
    parser.add_argument("--v9-protective-reid-overlap-iou", type=float, default=DEFAULT_V9_PROTECTIVE_REID_OVERLAP_IOU)
    parser.add_argument("--v9-stage1-local-min-confidence", type=float, default=DEFAULT_V9_STAGE1_LOCAL_MIN_CONFIDENCE)
    parser.add_argument("--v9-stage1-local-min-margin", type=float, default=DEFAULT_V9_STAGE1_LOCAL_MIN_MARGIN)
    parser.add_argument("--v9-stage1-local-min-motion", type=float, default=DEFAULT_V9_STAGE1_LOCAL_MIN_MOTION)
    parser.add_argument("--v9-stage1-local-min-path", type=float, default=DEFAULT_V9_STAGE1_LOCAL_MIN_PATH)
    parser.add_argument("--v9-next-best-min-confidence", type=float, default=DEFAULT_V9_NEXT_BEST_MIN_CONFIDENCE)
    parser.add_argument("--v9-next-best-max-candidates", type=int, default=DEFAULT_V9_NEXT_BEST_MAX_CANDIDATES)
    parser.add_argument(
        "--v9-local-rescue",
        dest="v9_local_rescue",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_V9_LOCAL_RESCUE,
        help="Allow sane V9-local owned candidates to override LOWCONF/ID_UNCERTAIN hold when association is acceptable.",
    )
    parser.add_argument("--v9-local-rescue-min-assignment-score", type=float, default=DEFAULT_V9_LOCAL_RESCUE_MIN_ASSIGNMENT_SCORE)
    parser.add_argument("--v9-local-rescue-max-scale-error", type=float, default=DEFAULT_V9_LOCAL_RESCUE_MAX_SCALE_ERROR)
    parser.add_argument(
        "--v9-continuity",
        dest="v9_continuity_enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_V9_CONTINUITY_ENABLED,
        help="Rescore top-k V9 local candidates using selected-target identity, motion path, and scale continuity before ReID association.",
    )
    parser.add_argument("--v9-continuity-topk", type=int, default=DEFAULT_V9_CONTINUITY_TOPK)
    parser.add_argument("--v9-continuity-min-score", type=float, default=DEFAULT_V9_CONTINUITY_MIN_SCORE)
    parser.add_argument("--v9-continuity-min-margin", type=float, default=DEFAULT_V9_CONTINUITY_MIN_MARGIN)
    parser.add_argument("--v9-accept-max-center-ratio", type=float, default=DEFAULT_V9_ACCEPT_MAX_CENTER_RATIO)
    parser.add_argument("--v9-accept-max-healthy-center-ratio", type=float, default=DEFAULT_V9_ACCEPT_MAX_HEALTHY_CENTER_RATIO)
    parser.add_argument("--v9-local-hold-min-confidence", type=float, default=DEFAULT_V9_LOCAL_HOLD_MIN_CONFIDENCE)
    parser.add_argument("--v9-local-hold-max-center-ratio", type=float, default=DEFAULT_V9_LOCAL_HOLD_MAX_CENTER_RATIO)
    parser.add_argument("--v9-local-hold-max-lost-center-ratio", type=float, default=DEFAULT_V9_LOCAL_HOLD_MAX_LOST_CENTER_RATIO)
    return parser


def parse_args():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("V9 tracker options:")
        build_v9_front_parser().print_help()
        print("\nShared/base tracker options:")
    parser = build_v9_front_parser()
    local_args, remaining = parser.parse_known_args()
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *remaining]
        args = runtime.parse_args()
    finally:
        sys.argv = original_argv
    args.v9_local_grid_size = max(4, int(local_args.v9_local_grid_size))
    args.v9_diagnostic_mode = local_args.v9_diagnostic_mode
    args.v9_scale_gate_min_ratio = local_args.v9_scale_gate_min_ratio
    args.v9_scale_gate_max_ratio = local_args.v9_scale_gate_max_ratio
    args.v9_scale_gate_override_confidence = local_args.v9_scale_gate_override_confidence
    args.v9_scale_gate_fallback_confidence_scale = local_args.v9_scale_gate_fallback_confidence_scale
    args.v9_protective_reid = local_args.v9_protective_reid
    args.v9_protective_reid_confidence_gate = local_args.v9_protective_reid_confidence_gate
    args.v9_protective_reid_margin_gate = local_args.v9_protective_reid_margin_gate
    args.v9_protective_reid_overlap_iou = local_args.v9_protective_reid_overlap_iou
    args.v9_stage1_local_min_confidence = local_args.v9_stage1_local_min_confidence
    args.v9_stage1_local_min_margin = local_args.v9_stage1_local_min_margin
    args.v9_stage1_local_min_motion = local_args.v9_stage1_local_min_motion
    args.v9_stage1_local_min_path = local_args.v9_stage1_local_min_path
    args.v9_next_best_min_confidence = local_args.v9_next_best_min_confidence
    args.v9_next_best_max_candidates = local_args.v9_next_best_max_candidates
    args.v9_local_rescue = local_args.v9_local_rescue
    args.v9_local_rescue_min_assignment_score = local_args.v9_local_rescue_min_assignment_score
    args.v9_local_rescue_max_scale_error = local_args.v9_local_rescue_max_scale_error
    args.v9_continuity_enabled = local_args.v9_continuity_enabled
    args.v9_continuity_topk = local_args.v9_continuity_topk
    args.v9_continuity_min_score = local_args.v9_continuity_min_score
    args.v9_continuity_min_margin = local_args.v9_continuity_min_margin
    args.v9_accept_max_center_ratio = local_args.v9_accept_max_center_ratio
    args.v9_accept_max_healthy_center_ratio = local_args.v9_accept_max_healthy_center_ratio
    args.v9_local_hold_min_confidence = local_args.v9_local_hold_min_confidence
    args.v9_local_hold_max_center_ratio = local_args.v9_local_hold_max_center_ratio
    args.v9_local_hold_max_lost_center_ratio = local_args.v9_local_hold_max_lost_center_ratio
    return args


def default_output_path(source_name: str) -> Path:
    return mot.default_output_path(source_name, "lorat_v9_local")


def default_debug_log_path(source_name: str) -> Path:
    return mot.default_debug_log_path(source_name, "lorat_v9_local")


def default_video_path(source_name: str) -> Path:
    return mot.default_video_path(source_name, "lorat_v9_local")


def default_manual_event_log_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return mot.DEFAULT_DEBUG_DIR / f"{safe_name}_lorat_v9_local_manual_events.csv"


def default_week2_proof_log_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return mot.DEFAULT_DEBUG_DIR / f"{safe_name}_lorat_v9_local_week2_proof.csv"


def main() -> int:
    args = parse_args()
    frame_source = mot.open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name)
    debug_log_path = args.debug_log or default_debug_log_path(frame_source.name)
    slot_debug_log_path = None if args.no_slot_debug_log else args.slot_debug_log
    week2_proof_log_path = None if args.no_week2_proof_log else args.week2_proof_log
    manual_event_log_path = None if args.no_manual_event_log else (args.manual_event_log or default_manual_event_log_path(frame_source.name))
    save_video_path = None if args.no_save_video else (args.save_video or default_video_path(frame_source.name))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok, first_frame = frame_source.read()
    if not ok or first_frame is None:
        print("Unable to read the first frame.")
        return 1

    boxes = runtime.parse_initial_boxes(args.initial_boxes)
    if not boxes and args.no_display:
        raise RuntimeError("--no-display requires --initial-boxes.")
    if not boxes:
        boxes = mot.select_boxes(first_frame)
    if not boxes:
        print("No bounding boxes selected. Exiting.")
        frame_source.release()
        cv2.destroyAllWindows()
        return 0

    backend = create_backend(args, frame_source, len(boxes))
    writer = mot.make_video_writer(save_video_path, frame_source.fps, first_frame) if save_video_path is not None else None
    mot_lines: List[str] = []
    debug_lines: List[str] = []
    manual_events: List[mot.ManualReanchorEvent] = []
    frame_number = 1
    paused = False
    outputs_written = False
    last_frame = first_frame

    def flush_outputs() -> None:
        nonlocal outputs_written
        if outputs_written:
            return
        output_path.write_text("".join(mot_lines), encoding="utf-8")
        print(f"Wrote MOTChallenge-format tracks to: {output_path}")
        runtime.write_debug_log(debug_log_path, debug_lines)
        print(f"Wrote debug CSV to: {debug_log_path}")
        if slot_debug_log_path is not None:
            mot.write_slot_debug_log(slot_debug_log_path, backend.slot_debug_lines)
            print(f"Wrote V9 head-bank debug CSV to: {slot_debug_log_path}")
        if week2_proof_log_path is not None:
            runtime.write_week2_proof_log(week2_proof_log_path, backend.week2_proof_lines)
            print(f"Wrote shared-backbone proof CSV to: {week2_proof_log_path}")
        if manual_event_log_path is not None:
            mot.write_manual_event_csv(manual_event_log_path, manual_events)
            print(f"Wrote manual event CSV to: {manual_event_log_path}")
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        mot.append_mot_results(mot_lines, frame_number, backend.tracks)
        runtime.append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
        if writer is not None:
            writer.write(mot.draw_tracks(first_frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines()))

        while True:
            if not paused:
                ok, frame = frame_source.read()
                if not ok or frame is None:
                    break
                last_frame = frame
                frame_number += 1
                backend.update(frame, frame_number)
                mot.append_mot_results(mot_lines, frame_number, backend.tracks)
                runtime.append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
            else:
                frame = last_frame.copy()

            shown = mot.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines())
            if writer is not None and not paused:
                writer.write(shown)

            if not args.no_display:
                cv2.imshow("LoRAT Multi-Object Tracker V9", shown)
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("a"):
                    paused = True
                    new_boxes = mot.select_boxes(frame, "Add Objects")
                    if new_boxes:
                        added_tracks = backend.add_tracks(frame, new_boxes, frame_number)
                        mot.append_mot_results(mot_lines, frame_number, added_tracks)
                        runtime.append_debug_rows(
                            debug_lines,
                            frame_number,
                            backend.tracks,
                            args.debug_frame_start,
                            args.debug_frame_end,
                        )
                    paused = False
                if key == ord("r"):
                    paused = True
                    target_track = mot.choose_manual_reanchor_track(backend.tracks)
                    if target_track is None:
                        print("No track available for manual reanchor.")
                    else:
                        started = time.perf_counter()
                        title = f"Re-anchor Track {target_track.track_id}"
                        reanchor_boxes = mot.select_boxes(frame, title)
                        seconds_spent = time.perf_counter() - started
                        if reanchor_boxes:
                            event = backend.manual_reanchor_track(
                                target_track.track_id,
                                frame,
                                reanchor_boxes[0],
                                frame_number,
                                seconds_spent=seconds_spent,
                                source="keyboard_r",
                            )
                            manual_events.append(event)
                            mot.append_mot_results(mot_lines, frame_number, [target_track])
                            runtime.append_debug_rows(
                                debug_lines,
                                frame_number,
                                backend.tracks,
                                args.debug_frame_start,
                                args.debug_frame_end,
                            )
                            if manual_event_log_path is not None:
                                mot.write_manual_event_csv(manual_event_log_path, manual_events)
                            print(
                                f"Manual reanchor track {target_track.track_id} at frame {frame_number} "
                                f"({seconds_spent:.2f}s)."
                            )
                    paused = False

            if args.max_frames > 0 and frame_number >= args.max_frames:
                break
    finally:
        backend.close()
        frame_source.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        flush_outputs()

    if save_video_path is not None:
        print(f"Wrote annotated video to: {save_video_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
