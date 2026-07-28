from __future__ import annotations

"""V9-owned runtime infrastructure snapshot.

This file is intentionally copied forward from the mature V8 tracker runtime so
V9 can be standalone from older versioned files while retaining the stable
LoRaT loading, batched-head, track-state, ReID, logging, and UI infrastructure
that V9 still builds on. V9-specific tracker decisions live in
``bounding_box_v9_lorat_local_search.py``.
"""

import argparse
import copy
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import mot_common as mot

BBox = mot.BBox

V8_EXECUTION_MODE = "shared-frame-vit-batched-heads-dinov2-crop-reid"
DEFAULT_V8_PRIMARY_HEADS_PER_TRACK = 1
DEFAULT_V8_RECOVERY_HEADS_PER_TRACK = 5
DEFAULT_V8_RECOVERY_INTERVAL = 15
DEFAULT_V8_RECOVERY_MIN_CONFIDENCE = 0.45
DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_SCORE = 0.58
DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_MARGIN = 0.08
DEFAULT_V8_RECOVERY_STALE_HEAD_FRAMES = 30
DEFAULT_V8_TEMPLATE_MATCH_ENABLED = True
DEFAULT_V8_TEMPLATE_MATCH_MIN_SCORE = 0.72
DEFAULT_V8_TEMPLATE_MATCH_PREFER_MARGIN = 0.18
DEFAULT_V8_TEMPLATE_MATCH_ON_UNCERTAIN_ONLY = True
DEFAULT_V8_TEMPLATE_MATCH_HEAD_CONFIDENCE_GATE = 0.62
DEFAULT_V8_TEMPLATE_MATCH_MARGIN_GATE = 0.04
DEFAULT_V8_HEAD_TEMPLATE_BLEND = 0.0
DEFAULT_V8_TEMPLATE_RESCUE_MIN_MOTION = 0.60
DEFAULT_V8_TEMPLATE_RESCUE_MIN_PATH = 0.55
DEFAULT_V8_TEMPLATE_RESCUE_MIN_HEAD_IOU = 0.12
DEFAULT_V8_MEMORY_MIN_MOTION = 0.45
DEFAULT_V8_MEMORY_MIN_PATH = 0.45
DEFAULT_V8_MEMORY_MIN_APPEARANCE = 0.42
DEFAULT_V8_MEMORY_MIN_STABLE_UPDATES = 2
DEFAULT_V8_ACCEPT_MIN_INITIAL_ANCHOR = 0.50
DEFAULT_V8_ACCEPT_MIN_IDENTITY_MARGIN = -0.05
DEFAULT_V8_MEMORY_MIN_INITIAL_ANCHOR = 0.58
DEFAULT_V8_MEMORY_MIN_IDENTITY_MARGIN = 0.02
DEFAULT_V8_WINDOW_PENALTY_RATIO = 0.45
DEFAULT_V8_DINOV2_CROP_REID = True
DEFAULT_V8_DINOV2_CROP_REID_BATCH = 16
DEFAULT_V8_DINOV2_CROP_REID_MIN_AREA = 9.0
DEFAULT_V8_ASSIGNMENT_CONFLICT_IOU = 0.65
DEFAULT_V8_ASSIGNMENT_CONFLICT_HARD_IOU = 0.82
DEFAULT_V8_ASSIGNMENT_CONFLICT_SCORE_MARGIN = 0.04
DEFAULT_V8_ASSIGNMENT_CONFLICT_CENTER_RATIO = 0.38
DEFAULT_V8_ASSIGNMENT_CONFLICT_CONTAINMENT = 0.45
DEFAULT_V8_ASSIGNMENT_CONFLICT_OWNERSHIP_MARGIN = 0.07
DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_ENABLED = True
DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MAX_CANDIDATES = 4
DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MIN_CONFIDENCE = 0.38
DEFAULT_V8_DISTRACTOR_BANK_SIZE = 8
DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY = 0.62
DEFAULT_V8_DISTRACTOR_PENALTY = 0.18
DEFAULT_V8_SMALL_TARGET_MODE = True
DEFAULT_V8_SMALL_TARGET_AREA = 4096.0
DEFAULT_V8_SMALL_TARGET_MAX_SIDE = 96.0
DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE = 1.35
DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE = 0.56
DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION = 0.25
DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH = 0.25
DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR = 0.02
V8_PROFILE_BUCKETS = (
    "candidate_transfer",
    "candidate_extract",
    "template_match",
    "candidate_fusion",
    "reid_appearance",
    "dinov2_crop_reid",
    "identity_resolve",
    "identity_score",
    "debug_output",
    "accept",
    "hold",
    "appearance_refresh",
    "proof_output",
)

V9_BASE_EXECUTION_MODE = V8_EXECUTION_MODE
DEFAULT_V9_BASE_DISTRACTOR_MIN_SIMILARITY = DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY
DEFAULT_V9_BASE_SMALL_TARGET_MODE = DEFAULT_V8_SMALL_TARGET_MODE
DEFAULT_V9_BASE_SMALL_TARGET_AREA = DEFAULT_V8_SMALL_TARGET_AREA
DEFAULT_V9_BASE_SMALL_TARGET_MAX_SIDE = DEFAULT_V8_SMALL_TARGET_MAX_SIDE
DEFAULT_V9_BASE_SMALL_TARGET_MAX_SCALE_CHANGE = DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE
DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_SCORE = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE
DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_MOTION = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION
DEFAULT_V9_BASE_SMALL_TARGET_TEMPLATE_MIN_PATH = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH
DEFAULT_V9_BASE_SMALL_TARGET_CONFIDENCE_FLOOR = DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR
DEFAULT_V9_BASE_WINDOW_PENALTY_RATIO = DEFAULT_V8_WINDOW_PENALTY_RATIO


@dataclass(frozen=True)
class SharedFrameEncoding:
    feature_map: object
    elapsed_seconds: float


@dataclass
class V8TemplateMemorySlot:
    vector: object
    label: str
    frame_number: int
    confidence: Optional[float] = None
    patch_tokens: Optional[object] = None
    patch_foreground_mask: Optional[object] = None


@dataclass(frozen=True)
class BatchedHeadOutput:
    score_maps: object
    box_delta_maps: object
    elapsed_seconds: float
    selected_head_count: int


@dataclass(frozen=True)
class V8HeadCandidate:
    rank: int
    bbox: BBox
    confidence: float
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class V8HeadCandidateInfo:
    bbox: BBox
    confidence: float
    margin: float
    roi_tokens: int
    top_candidates: Tuple[V8HeadCandidate, ...]


WEEK2_PROOF_LOG_HEADER = (
    "frame,phase,mode,head_mode,tracked_objects_this_frame,active_objects,frame_seconds,fps,"
    "shared_backbone_calls_this_frame,object_head_batches_this_frame,object_head_items_this_frame,"
    "selected_head_items_this_frame,cumulative_shared_backbone_calls,cumulative_object_head_batches,"
    "cumulative_object_head_items,cumulative_selected_head_items,max_object_head_batch,last_backbone_ms,"
    "last_head_ms,roi_tokens_this_frame,profile_candidate_transfer_ms,profile_candidate_extract_ms,"
    "profile_template_match_ms,profile_candidate_fusion_ms,profile_reid_appearance_ms,"
    "profile_dinov2_crop_reid_ms,profile_identity_resolve_ms,profile_identity_score_ms,"
    "profile_debug_output_ms,profile_accept_ms,"
    "profile_hold_ms,profile_appearance_refresh_ms,profile_proof_output_ms,"
    "profile_unbucketed_ms,dinov2_crop_reid_calls_this_frame,dinov2_crop_reid_items_this_frame,"
    "cumulative_dinov2_crop_reid_calls,cumulative_dinov2_crop_reid_items,max_dinov2_crop_reid_batch,"
    "gpu_name,gpu_allocated_mb,gpu_reserved_mb,gpu_peak_allocated_mb,"
    "gpu_peak_reserved_mb,week2_shared_backbone_ok,week2_batched_head_ok\n"
)

V8_DEBUG_LOG_HEADER = mot.DEBUG_LOG_HEADER.replace(
    "appearance_bank_size\n",
    "legacy_appearance_bank_size,v8_feature_bank_size,v8_crop_feature_bank_size,track_lifecycle_state\n",
)


class SharedFrameLoRATEncoder:
    """V8-only frame-level encoder that avoids the LoRAT evaluator pipeline."""

    def __init__(
        self,
        torch_module,
        functional_module,
        lorat_model,
        amp_autocast_fn,
        image_normalize_transform,
        device,
        dtype,
        input_width: int,
        input_height: int,
        grid_width: int,
        grid_height: int,
        embed_dim: int,
    ) -> None:
        self.torch = torch_module
        self.F = functional_module
        self.lorat_model = lorat_model
        self.amp_autocast_fn = amp_autocast_fn
        self.image_normalize_transform = image_normalize_transform
        self.device = device
        self.dtype = dtype
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.embed_dim = int(embed_dim)

    def preprocess(self, frame: np.ndarray):
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        tensor = self.torch.from_numpy(rgb)
        tensor = tensor.permute(2, 0, 1).to(self.device).to(self.torch.float32)
        tensor.div_(255.0)
        self.image_normalize_transform(tensor)
        return tensor.to(self.dtype).unsqueeze(0)

    def encode(self, frame: np.ndarray) -> SharedFrameEncoding:
        started = time.perf_counter()
        x = self.preprocess(frame)
        with self.torch.inference_mode(), self.amp_autocast_fn():
            tokens = self.lorat_model._x_feat(x)
            for block in self.lorat_model.blocks:
                tokens = block(tokens)
            tokens = self.lorat_model.norm(tokens)
        feature_map = tokens[0].reshape(self.grid_height, self.grid_width, self.embed_dim)
        feature_map = self.F.normalize(feature_map.to(self.torch.float32), dim=-1)
        return SharedFrameEncoding(feature_map=feature_map, elapsed_seconds=time.perf_counter() - started)


class BatchedObjectConditionedHead:
    """V8-only object-conditioned head over one shared frame feature map."""

    def __init__(
        self,
        torch_module,
        functional_module,
        device,
        embed_dim: int,
        max_head_rank: int,
        score_reduction: str,
        hidden_dim: int = 256,
        lora_rank: int = 16,
        box_delta_scale: float = 0.70,
        weight_path: Optional[Path] = None,
    ) -> None:
        self.torch = torch_module
        self.F = functional_module
        self.device = device
        self.embed_dim = int(embed_dim)
        self.max_head_rank = max(1, int(max_head_rank))
        self.score_reduction = score_reduction
        self.hidden_dim = max(16, int(hidden_dim))
        self.lora_rank = max(1, int(lora_rank))
        self.box_delta_scale = max(0.05, float(box_delta_scale))
        self.module = self._build_module().to(self.device)
        self.module.eval()
        self.weights_loaded = False
        self.last_mode = "zero_shot_similarity"
        if weight_path is not None:
            self.load_weights(weight_path)

    def _build_module(self):
        torch_module = self.torch
        nn = torch_module.nn
        embed_dim = self.embed_dim
        hidden_dim = self.hidden_dim
        lora_rank = self.lora_rank
        box_delta_scale = self.box_delta_scale

        class V8ObjectConditionedLoRAHead(nn.Module):
            """Trainable per-object LoRA-conditioned score and box head.

            The base projection is shared across all objects. Each object's cached
            template embedding generates a low-rank up-projection, so the LoRA delta
            is batched across objects while the frame tokens are shared.
            """

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
                nn.init.normal_(self.template_token_type, std=0.02)

            def forward(self, feature_tokens, object_embeddings, template_tokens=None, template_mask=None, template_foreground_mask=None):
                feature_tokens = self.feature_norm(feature_tokens)
                object_embeddings = self.object_norm(object_embeddings)

                if template_tokens is not None and template_mask is not None:
                    template_tokens = self.template_norm(template_tokens)
                    if template_foreground_mask is None:
                        template_foreground_mask = template_mask
                    token_type_ids = template_foreground_mask.to(torch_module.long).clamp(0, 1)
                    template_tokens = template_tokens + self.template_token_type[token_type_ids].to(template_tokens.dtype)
                    token_similarity = torch_module.einsum("ld,ntd->nlt", feature_tokens, template_tokens) / (embed_dim ** 0.5)
                    token_similarity = token_similarity.masked_fill(~template_mask[:, None, :], -1.0e4)
                    attention = torch_module.softmax(token_similarity, dim=-1)
                    template_context = torch_module.einsum("nlt,ntd->nld", attention, template_tokens)
                    summary_mask = template_foreground_mask & template_mask
                    empty_summary = summary_mask.sum(dim=1, keepdim=True) == 0
                    summary_mask = torch_module.where(empty_summary, template_mask, summary_mask)
                    template_summary = (template_tokens * summary_mask[:, :, None].to(template_tokens.dtype)).sum(dim=1)
                    template_count = summary_mask.sum(dim=1, keepdim=True).clamp_min(1).to(template_tokens.dtype)
                    template_summary = template_summary / template_count
                    gate = torch_module.sigmoid(self.template_gate(template_summary))[:, None, :]
                    conditioned_features = self.fusion_norm(
                        feature_tokens[None, :, :] + gate * self.template_context(template_context)
                    )
                else:
                    conditioned_features = feature_tokens[None, :, :].expand(object_embeddings.shape[0], -1, -1)

                base = self.base_projection(conditioned_features)
                down = self.lora_down(conditioned_features)
                up = self.lora_up_generator(object_embeddings).view(
                    object_embeddings.shape[0],
                    lora_rank,
                    hidden_dim,
                )
                lora_delta = torch_module.einsum("nlr,nrh->nlh", down, up) * self.lora_scale
                object_bias = self.object_bias(object_embeddings)[:, None, :]
                hidden = self.activation(base + object_bias + lora_delta)
                score_logits = self.score_head(hidden).squeeze(-1)
                box_deltas = self.box_head(hidden)
                return score_logits, box_deltas

            def project_reid(self, embeddings):
                return torch_module.nn.functional.normalize(self.reid_projection(embeddings), dim=-1)

        return V8ObjectConditionedLoRAHead()

    def load_weights(self, path: Path) -> None:
        path = path.resolve()
        if not path.exists():
            raise RuntimeError(f"V8 head weight file not found: {path}")
        try:
            state = self.torch.load(str(path), map_location=self.device, weights_only=False)
        except TypeError:
            state = self.torch.load(str(path), map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        try:
            self.module.load_state_dict(state, strict=True)
            loaded_keys = len(state) if isinstance(state, dict) else 0
            print(f"V8 object head loaded: {path} ({loaded_keys} tensors, strict)", flush=True)
        except RuntimeError as error:
            if not isinstance(state, dict):
                raise
            current_state = self.module.state_dict()
            compatible_state = {}
            skipped_keys = []
            for key, value in state.items():
                target_value = current_state.get(key)
                if target_value is None:
                    skipped_keys.append(key)
                    continue
                if tuple(value.shape) != tuple(target_value.shape):
                    skipped_keys.append(key)
                    continue
                compatible_state[key] = value
            if not compatible_state:
                raise RuntimeError(f"V8 head checkpoint has no compatible tensors: {path}") from error
            load_result = self.module.load_state_dict(compatible_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            print(
                "V8 object head checkpoint migrated: "
                f"{path} loaded={len(compatible_state)} "
                f"missing={len(missing_keys)} skipped={len(skipped_keys)} unexpected={len(unexpected_keys)}",
                flush=True,
            )
            if missing_keys:
                print(f"  Initialized new V8 head tensors: {', '.join(missing_keys[:12])}", flush=True)
                if len(missing_keys) > 12:
                    print(f"  ... plus {len(missing_keys) - 12} more", flush=True)
            if skipped_keys:
                print(f"  Skipped incompatible checkpoint tensors: {', '.join(skipped_keys[:12])}", flush=True)
                if len(skipped_keys) > 12:
                    print(f"  ... plus {len(skipped_keys) - 12} more", flush=True)
            if unexpected_keys:
                print(f"  Unexpected checkpoint tensors: {', '.join(unexpected_keys[:12])}", flush=True)
        self.weights_loaded = True
        self.last_mode = "template_patch_lora_conditioned"

    def state_dict(self):
        return self.module.state_dict()

    def parameters(self):
        return self.module.parameters()

    @staticmethod
    def _slot_vector(slot: object):
        return slot.vector if isinstance(slot, V8TemplateMemorySlot) else slot

    def _build_head_tensor(self, selected_banks: Sequence[Sequence[object]]):
        max_heads = max((len(bank) for bank in selected_banks), default=1)
        max_heads = max(1, min(max_heads, self.max_head_rank))
        max_template_tokens = 1
        for head_bank in selected_banks:
            for slot in head_bank[:max_heads]:
                patch_tokens = getattr(slot, "patch_tokens", None)
                if patch_tokens is not None and patch_tokens.numel() > 0:
                    max_template_tokens = max(max_template_tokens, int(patch_tokens.reshape(-1, self.embed_dim).shape[0]))
        head_tensor = self.torch.zeros(
            (len(selected_banks), max_heads, self.embed_dim),
            device=self.device,
            dtype=self.torch.float32,
        )
        head_mask = self.torch.zeros((len(selected_banks), max_heads), device=self.device, dtype=self.torch.bool)
        template_tensor = self.torch.zeros(
            (len(selected_banks), max_heads, max_template_tokens, self.embed_dim),
            device=self.device,
            dtype=self.torch.float32,
        )
        template_mask = self.torch.zeros(
            (len(selected_banks), max_heads, max_template_tokens),
            device=self.device,
            dtype=self.torch.bool,
        )
        template_foreground_mask = self.torch.zeros_like(template_mask)
        for track_index, head_bank in enumerate(selected_banks):
            if not head_bank:
                head_bank = [self.torch.zeros(self.embed_dim, device=self.device, dtype=self.torch.float32)]
            for head_index, slot in enumerate(head_bank[:max_heads]):
                vector = self._slot_vector(slot)
                head_tensor[track_index, head_index] = vector.to(self.device, dtype=self.torch.float32)
                head_mask[track_index, head_index] = True
                patch_tokens = getattr(slot, "patch_tokens", None)
                if patch_tokens is None or patch_tokens.numel() == 0:
                    patch_tokens = vector.reshape(1, self.embed_dim)
                patch_tokens = patch_tokens.to(self.device, dtype=self.torch.float32).reshape(-1, self.embed_dim)
                token_count = min(max_template_tokens, int(patch_tokens.shape[0]))
                template_tensor[track_index, head_index, :token_count] = patch_tokens[:token_count]
                template_mask[track_index, head_index, :token_count] = True
                foreground_mask = getattr(slot, "patch_foreground_mask", None)
                if foreground_mask is None or foreground_mask.numel() == 0:
                    template_foreground_mask[track_index, head_index, :token_count] = True
                else:
                    foreground_mask = foreground_mask.to(self.device, dtype=self.torch.bool).reshape(-1)
                    foreground_count = min(token_count, int(foreground_mask.shape[0]))
                    template_foreground_mask[track_index, head_index, :foreground_count] = foreground_mask[:foreground_count]
                    if not bool(template_foreground_mask[track_index, head_index, :token_count].any().item()):
                        template_foreground_mask[track_index, head_index, :token_count] = True
        return (
            self.F.normalize(head_tensor, dim=-1),
            head_mask,
            self.F.normalize(template_tensor, dim=-1),
            template_mask,
            template_foreground_mask,
        )

    def _reduce_head_scores_and_deltas(self, per_head_scores, per_head_deltas, head_mask):
        if per_head_scores.shape[1] == 1:
            return per_head_scores[:, 0, :], per_head_deltas[:, 0, :, :]

        valid = head_mask[:, :, None]
        if self.score_reduction == "mean":
            weights = valid.to(per_head_scores.dtype)
            counts = weights.sum(dim=1).clamp_min(1.0)
            score_maps = (per_head_scores * weights).sum(dim=1) / counts
            delta_weights = weights[:, :, :, None]
            box_deltas = (per_head_deltas * delta_weights).sum(dim=1) / counts[:, :, None]
            return score_maps, box_deltas

        masked_scores = self.torch.where(
            valid,
            per_head_scores,
            self.torch.full_like(per_head_scores, -float("inf")),
        )
        score_maps, best_head_indices = self.torch.max(masked_scores, dim=1)
        gather_index = best_head_indices[:, None, :, None].expand(-1, 1, -1, per_head_deltas.shape[-1])
        box_deltas = self.torch.gather(per_head_deltas, dim=1, index=gather_index).squeeze(1)
        return score_maps, box_deltas

    def _effective_selected_head_count(self, selected_banks: Sequence[Sequence[object]]) -> int:
        max_heads = max((len(bank) for bank in selected_banks), default=1)
        max_heads = max(1, min(max_heads, self.max_head_rank))
        total = 0
        for bank in selected_banks:
            total += min(len(bank) if bank else 1, max_heads)
        return total

    def score(self, feature_map, selected_banks: Sequence[Sequence[object]]) -> BatchedHeadOutput:
        started = time.perf_counter()
        context = nullcontext() if self.module.training else self.torch.inference_mode()
        with context:
            flat_features = feature_map.reshape(-1, self.embed_dim)
            selected_head_count = self._effective_selected_head_count(selected_banks)
            head_tensor, head_mask, template_tensor, template_mask, template_foreground_mask = self._build_head_tensor(selected_banks)
            if not self.weights_loaded and not self.module.training:
                self.last_mode = "zero_shot_similarity"
                normalized_features = self.F.normalize(flat_features.to(self.torch.float32), dim=-1)
                per_head_scores = self.torch.matmul(head_tensor, normalized_features.transpose(0, 1)) * 10.0
                per_head_deltas = self.torch.zeros(
                    (len(selected_banks), head_tensor.shape[1], flat_features.shape[0], 4),
                    device=self.device,
                    dtype=self.torch.float32,
                )
                score_logits, box_deltas = self._reduce_head_scores_and_deltas(per_head_scores, per_head_deltas, head_mask)
            else:
                self.last_mode = "template_patch_lora_conditioned"
                flat_head_tensor = head_tensor.reshape(-1, self.embed_dim)
                flat_template_tensor = template_tensor.reshape(-1, template_tensor.shape[2], self.embed_dim)
                flat_template_mask = template_mask.reshape(-1, template_mask.shape[2])
                per_head_scores, per_head_deltas = self.module(
                    flat_features,
                    flat_head_tensor,
                    flat_template_tensor,
                    flat_template_mask,
                    template_foreground_mask.reshape(-1, template_foreground_mask.shape[2]),
                )
                per_head_scores = per_head_scores.reshape(len(selected_banks), head_tensor.shape[1], flat_features.shape[0])
                per_head_deltas = per_head_deltas.reshape(len(selected_banks), head_tensor.shape[1], flat_features.shape[0], 4)
                score_logits, box_deltas = self._reduce_head_scores_and_deltas(per_head_scores, per_head_deltas, head_mask)
            if not selected_banks:
                box_deltas = self.torch.zeros(
                    (0, flat_features.shape[0], 4),
                    device=self.device,
                    dtype=self.torch.float32,
                )
            score_maps = score_logits.reshape(len(selected_banks), feature_map.shape[0], feature_map.shape[1])
            box_delta_maps = box_deltas.reshape(len(selected_banks), feature_map.shape[0], feature_map.shape[1], 4)
        return BatchedHeadOutput(
            score_maps=score_maps,
            box_delta_maps=box_delta_maps,
            elapsed_seconds=time.perf_counter() - started,
            selected_head_count=selected_head_count,
        )


class V8FeatureIdentityArbitrator(mot.LightweightIdentityArbitrator):
    """V8 identity arbitration over shared-backbone feature tensors.

    V5's arbitrator extracts color/gradient histograms from image crops when
    needed. V8 already has one shared ViT feature map per frame, so this class
    keeps appearance memory as normalized tensors on the tracker device and
    never falls back to crop histogram extraction.
    """

    def __init__(
        self,
        torch_module,
        functional_module,
        device,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.torch = torch_module
        self.F = functional_module
        self.device = device
        self._last_score_track_ids: Tuple[int, ...] = ()
        self._last_score_output_ids: Tuple[int, ...] = ()
        self._last_score_matrices: Optional[Dict[str, np.ndarray]] = None

    def _clear_score_cache(self) -> None:
        self._last_score_track_ids = ()
        self._last_score_output_ids = ()
        self._last_score_matrices = None

    def _remember_score_matrices(
        self,
        tracks: Sequence[mot.TrackState],
        outputs: Sequence[mot.LoRATSlotOutput],
        matrices: Dict[str, np.ndarray],
    ) -> None:
        self._last_score_track_ids = tuple(id(track) for track in tracks)
        self._last_score_output_ids = tuple(id(output) for output in outputs)
        self._last_score_matrices = matrices

    def score_from_cached_matrices(
        self,
        tracks: Sequence[mot.TrackState],
        outputs: Sequence[mot.LoRATSlotOutput],
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
    ) -> mot.IdentityScore:
        """Return a pair score from the frame-level batched matrix when available."""

        track_ids = tuple(id(candidate_track) for candidate_track in tracks)
        output_ids = tuple(id(candidate_output) for candidate_output in outputs)
        matrices = self._last_score_matrices
        if (
            matrices is None
            or track_ids != self._last_score_track_ids
            or output_ids != self._last_score_output_ids
        ):
            return self.score(track, output, tracks)
        try:
            row = track_ids.index(id(track))
            col = output_ids.index(id(output))
        except ValueError:
            return self.score(track, output, tracks)
        return self._identity_score_from_matrices(matrices, row, col)

    def initialize_track(self, track: mot.TrackState, frame: np.ndarray) -> None:
        return None

    def normalize_feature(self, vector):
        if vector is None:
            return None
        return self.F.normalize(vector.detach().to(self.device, dtype=self.torch.float32).flatten(), dim=0)

    @staticmethod
    def _track_feature_names(prefer_crop: bool) -> Tuple[str, str, str]:
        if prefer_crop:
            return "v8_initial_crop_feature", "v8_appearance_crop_feature", "v8_crop_feature_bank"
        return "v8_initial_feature", "v8_appearance_feature", "v8_feature_bank"

    @staticmethod
    def _track_negative_feature_name(prefer_crop: bool) -> str:
        return "v8_negative_crop_feature_bank" if prefer_crop else "v8_negative_feature_bank"

    @staticmethod
    def _requires_local_owner(track: mot.TrackState) -> bool:
        """Healthy tracks should follow their own local proposal, not a global ReID jump."""

        if not bool(track.ok) or int(track.lost_frames or 0) > 0:
            return False
        state = str(track.state or "").upper()
        recovery_tokens = (
            "MISS",
            "LOWCONF",
            "ID_UNCERTAIN",
            "OCCLU",
            "LOST",
            "REID",
            "NOLEARN",
            "SHRINK",
            "CONFLICT",
            "MANUAL",
        )
        return not any(token in state for token in recovery_tokens)

    @staticmethod
    def _should_remember_negative_reject(reject_state: str) -> bool:
        reject_state = str(reject_state or "").upper()
        if not reject_state:
            return False
        return any(
            token in reject_state
            for token in (
                "ANCHOR",
                "AMBIG",
                "CROSS_SOURCE",
                "CONFLICT",
                "HEALTHY_LOCAL_OWNER",
                "NEGATIVE_MEMORY",
                "OTHERID",
            )
        )

    def output_feature_with_source(self, output: mot.LoRATSlotOutput):
        crop_feature = self.normalize_feature(getattr(output, "v8_crop_feature", None))
        if crop_feature is not None:
            return crop_feature, True
        return self.normalize_feature(getattr(output, "v8_feature", None)), False

    def output_feature(self, output: mot.LoRATSlotOutput):
        feature, _ = self.output_feature_with_source(output)
        return feature

    def track_has_feature_appearance(self, track: mot.TrackState) -> bool:
        return (
            getattr(track, "v8_initial_feature", None) is not None
            or getattr(track, "v8_appearance_feature", None) is not None
            or bool(getattr(track, "v8_feature_bank", []))
            or getattr(track, "v8_initial_crop_feature", None) is not None
            or getattr(track, "v8_appearance_crop_feature", None) is not None
            or bool(getattr(track, "v8_crop_feature_bank", []))
        )

    def remember_negative_candidate(
        self,
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        bank_size: int = DEFAULT_V8_DISTRACTOR_BANK_SIZE,
    ) -> None:
        """Store a rejected lookalike so future frames suppress the same distractor."""

        bank_size = max(1, int(bank_size))
        for feature_name, bank_name in (
            ("v8_feature", self._track_negative_feature_name(False)),
            ("v8_crop_feature", self._track_negative_feature_name(True)),
        ):
            feature = self.normalize_feature(getattr(output, feature_name, None))
            if feature is None:
                continue
            bank = list(getattr(track, bank_name, []))
            if bank and max(self.feature_similarity(memory, feature) for memory in bank) >= 0.985:
                continue
            bank.append(feature.detach().clone())
            if len(bank) > bank_size:
                del bank[: len(bank) - bank_size]
            setattr(track, bank_name, bank)

    @staticmethod
    def _xywh_array(boxes: Sequence[BBox]) -> np.ndarray:
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def _bbox_center_array(boxes: np.ndarray) -> np.ndarray:
        return np.stack((boxes[:, 0] + boxes[:, 2] * 0.5, boxes[:, 1] + boxes[:, 3] * 0.5), axis=1)

    @staticmethod
    def _bbox_area_array(boxes: np.ndarray) -> np.ndarray:
        return np.maximum(1.0, boxes[:, 2] * boxes[:, 3])

    @staticmethod
    def _bbox_aspect_array(boxes: np.ndarray) -> np.ndarray:
        return np.maximum(0.01, boxes[:, 2] / np.maximum(1.0, boxes[:, 3]))

    @staticmethod
    def _pairwise_iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.size == 0 or right.size == 0:
            return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
        left_x2 = left[:, 0] + left[:, 2]
        left_y2 = left[:, 1] + left[:, 3]
        right_x2 = right[:, 0] + right[:, 2]
        right_y2 = right[:, 1] + right[:, 3]
        inter_x1 = np.maximum(left[:, None, 0], right[None, :, 0])
        inter_y1 = np.maximum(left[:, None, 1], right[None, :, 1])
        inter_x2 = np.minimum(left_x2[:, None], right_x2[None, :])
        inter_y2 = np.minimum(left_y2[:, None], right_y2[None, :])
        intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = (
            (left[:, 2] * left[:, 3])[:, None]
            + (right[:, 2] * right[:, 3])[None, :]
            - intersection
        )
        return np.where(union > 0.0, intersection / np.maximum(union, 1e-6), 0.0).astype(np.float32)

    def _output_feature_stack(self, outputs: Sequence[mot.LoRATSlotOutput]):
        normalized = []
        valid: List[bool] = []
        use_crop: List[bool] = []
        feature_dim: Optional[int] = None
        for output in outputs:
            vector, is_crop = self.output_feature_with_source(output)
            if vector is not None:
                feature_dim = int(vector.numel())
                normalized.append(vector)
                valid.append(True)
                use_crop.append(is_crop)
            else:
                normalized.append(None)
                valid.append(False)
                use_crop.append(False)
        if feature_dim is None:
            return None, np.zeros(len(outputs), dtype=bool), False
        zero = self.torch.zeros(feature_dim, device=self.device, dtype=self.torch.float32)
        stacked = self.torch.stack([vector if vector is not None else zero for vector in normalized], dim=0)
        return stacked, np.asarray(valid, dtype=bool), bool(any(use_crop))

    def _track_memory_similarity_rows(
        self,
        tracks: Sequence[mot.TrackState],
        candidate_features,
        candidate_valid: np.ndarray,
        prefer_crop: bool = False,
    ):
        track_count = len(tracks)
        output_count = int(candidate_features.shape[0]) if candidate_features is not None else int(candidate_valid.shape[0])
        appearance_rows = []
        initial_rows = []
        initial_feature_rows = []
        initial_valid: List[bool] = []
        if candidate_features is None or output_count <= 0:
            return (
                np.full((track_count, output_count), 0.5, dtype=np.float32),
                np.full((track_count, output_count), 0.5, dtype=np.float32),
                np.zeros((track_count, output_count), dtype=np.float32),
                np.full((track_count, output_count), -1, dtype=np.int32),
            )

        for track in tracks:
            memory_vectors = []
            initial_name, current_name, bank_name = self._track_feature_names(prefer_crop)
            fallback_initial_name, fallback_current_name, fallback_bank_name = self._track_feature_names(not prefer_crop)
            initial = self.normalize_feature(getattr(track, initial_name, None))
            if initial is None:
                initial = self.normalize_feature(getattr(track, fallback_initial_name, None))
            if initial is not None:
                initial_feature_rows.append(initial)
                initial_valid.append(True)
                memory_vectors.append(initial)
            else:
                initial_feature_rows.append(None)
                initial_valid.append(False)
            current = self.normalize_feature(getattr(track, current_name, None))
            if current is None:
                current = self.normalize_feature(getattr(track, fallback_current_name, None))
            if current is not None:
                memory_vectors.append(current)
            memories = list(getattr(track, bank_name, []))
            if not memories:
                memories = list(getattr(track, fallback_bank_name, []))
            for memory in memories:
                memory_vector = self.normalize_feature(memory)
                if memory_vector is not None:
                    memory_vectors.append(memory_vector)

            if not memory_vectors:
                appearance = self.torch.full((output_count,), 0.5, device=self.device, dtype=self.torch.float32)
                initial_anchor = appearance.clone()
            else:
                memory_stack = self.torch.stack(memory_vectors, dim=0)
                similarities = self.torch.clamp(((memory_stack @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
                sorted_scores = self.torch.sort(similarities, dim=0, descending=True).values
                if similarities.shape[0] == 1:
                    best_pair = sorted_scores[0]
                else:
                    best_pair = (0.72 * sorted_scores[0]) + (0.28 * sorted_scores[1])
                if initial is None:
                    appearance = best_pair
                    initial_anchor = best_pair.clone()
                else:
                    initial_anchor = self.torch.clamp(((initial @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
                    top_count = min(4, int(sorted_scores.shape[0]))
                    top_average = sorted_scores[:top_count].mean(dim=0)
                    appearance = (0.44 * best_pair) + (0.42 * initial_anchor) + (0.14 * top_average)
            appearance_rows.append(appearance)
            initial_rows.append(initial_anchor)

        appearance_tensor = self.torch.stack(appearance_rows, dim=0)
        initial_tensor = self.torch.stack(initial_rows, dim=0)
        if not bool(candidate_valid.all()):
            valid_tensor = self.torch.as_tensor(candidate_valid, device=self.device, dtype=self.torch.bool)
            appearance_tensor = self.torch.where(valid_tensor[None, :], appearance_tensor, self.torch.full_like(appearance_tensor, 0.5))
            initial_tensor = self.torch.where(valid_tensor[None, :], initial_tensor, self.torch.full_like(initial_tensor, 0.5))

        appearance = appearance_tensor.detach().to(device="cpu", dtype=self.torch.float32).numpy()
        initial_anchor = initial_tensor.detach().to(device="cpu", dtype=self.torch.float32).numpy()

        other_anchor = np.zeros((track_count, output_count), dtype=np.float32)
        other_track_ids = np.full((track_count, output_count), -1, dtype=np.int32)
        valid_initial_indices = [index for index, is_valid in enumerate(initial_valid) if is_valid]
        if valid_initial_indices:
            initial_stack = self.torch.stack([initial_feature_rows[index] for index in valid_initial_indices], dim=0)
            all_initial_similarity = self.torch.clamp(((initial_stack @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
            all_initial = np.zeros((track_count, output_count), dtype=np.float32)
            all_initial[np.asarray(valid_initial_indices, dtype=np.int32), :] = (
                all_initial_similarity.detach().to(device="cpu", dtype=self.torch.float32).numpy()
            )
            for row_index, track in enumerate(tracks):
                eligible = np.asarray(initial_valid, dtype=bool)
                eligible[row_index] = False
                if not bool(eligible.any()):
                    continue
                scores = np.where(eligible[:, None], all_initial, -np.inf)
                best_indices = np.argmax(scores, axis=0)
                best_values = scores[best_indices, np.arange(output_count)]
                finite = np.isfinite(best_values) & candidate_valid
                other_anchor[row_index, finite] = best_values[finite].astype(np.float32)
                for col_index in np.where(finite)[0].tolist():
                    other_track_ids[row_index, col_index] = int(tracks[int(best_indices[col_index])].track_id)
        return appearance, initial_anchor, other_anchor, other_track_ids

    def _track_negative_similarity_rows(
        self,
        tracks: Sequence[mot.TrackState],
        candidate_features,
        candidate_valid: np.ndarray,
        prefer_crop: bool = False,
    ) -> np.ndarray:
        track_count = len(tracks)
        output_count = int(candidate_features.shape[0]) if candidate_features is not None else int(candidate_valid.shape[0])
        if candidate_features is None or output_count <= 0:
            return np.zeros((track_count, output_count), dtype=np.float32)

        rows = []
        for track in tracks:
            bank_name = self._track_negative_feature_name(prefer_crop)
            fallback_bank_name = self._track_negative_feature_name(not prefer_crop)
            negative_vectors = []
            for memory in list(getattr(track, bank_name, [])) or list(getattr(track, fallback_bank_name, [])):
                memory_vector = self.normalize_feature(memory)
                if memory_vector is not None:
                    negative_vectors.append(memory_vector)
            if not negative_vectors:
                rows.append(self.torch.zeros((output_count,), device=self.device, dtype=self.torch.float32))
                continue
            memory_stack = self.torch.stack(negative_vectors, dim=0)
            similarities = self.torch.clamp(((memory_stack @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
            rows.append(self.torch.max(similarities, dim=0).values)

        negative_tensor = self.torch.stack(rows, dim=0)
        if not bool(candidate_valid.all()):
            valid_tensor = self.torch.as_tensor(candidate_valid, device=self.device, dtype=self.torch.bool)
            negative_tensor = self.torch.where(valid_tensor[None, :], negative_tensor, self.torch.zeros_like(negative_tensor))
        return negative_tensor.detach().to(device="cpu", dtype=self.torch.float32).numpy().astype(np.float32)

    def track_feature_similarity_many(self, track: mot.TrackState, candidate_features) -> np.ndarray:
        if candidate_features is None:
            return np.zeros((0,), dtype=np.float32)
        if candidate_features.ndim == 1:
            candidate_features = candidate_features.reshape(1, -1)
        candidate_features = self.F.normalize(candidate_features.detach().to(self.device, dtype=self.torch.float32), dim=1)
        output_count = int(candidate_features.shape[0])
        memory_vectors = []
        initial = self.normalize_feature(getattr(track, "v8_initial_feature", None))
        initial_score = None
        if initial is not None:
            memory_vectors.append(initial)
        current = self.normalize_feature(getattr(track, "v8_appearance_feature", None))
        if current is not None:
            memory_vectors.append(current)
        for memory in getattr(track, "v8_feature_bank", []):
            memory_vector = self.normalize_feature(memory)
            if memory_vector is not None:
                memory_vectors.append(memory_vector)
        if not memory_vectors:
            return np.full((output_count,), 0.5, dtype=np.float32)
        memory_stack = self.torch.stack(memory_vectors, dim=0)
        similarities = self.torch.clamp(((memory_stack @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
        sorted_scores = self.torch.sort(similarities, dim=0, descending=True).values
        best_pair = sorted_scores[0] if similarities.shape[0] == 1 else (0.72 * sorted_scores[0]) + (0.28 * sorted_scores[1])
        if initial is not None:
            initial_score = self.torch.clamp(((initial @ candidate_features.transpose(0, 1)) + 1.0) * 0.5, 0.0, 1.0)
        if initial_score is None:
            appearance = best_pair
        else:
            top_count = min(4, int(sorted_scores.shape[0]))
            appearance = (0.44 * best_pair) + (0.42 * initial_score) + (0.14 * sorted_scores[:top_count].mean(dim=0))
        return appearance.detach().to(device="cpu", dtype=self.torch.float32).numpy().astype(np.float32)

    def _motion_matrix(self, predicted: np.ndarray, candidates: np.ndarray, reference_diagonal: np.ndarray) -> np.ndarray:
        predicted_centers = self._bbox_center_array(predicted)
        candidate_centers = self._bbox_center_array(candidates)
        distances = np.linalg.norm(predicted_centers[:, None, :] - candidate_centers[None, :, :], axis=2)
        center_score = np.maximum(0.0, 1.0 - np.minimum(1.0, distances / np.maximum(1.0, reference_diagonal[:, None])))
        scale_change = np.abs(np.log(self._bbox_area_array(candidates)[None, :] / self._bbox_area_array(predicted)[:, None]))
        scale_score = np.maximum(0.0, 1.0 - np.minimum(1.0, scale_change))
        aspect_change = np.abs(
            np.log(self._bbox_aspect_array(candidates)[None, :] / self._bbox_aspect_array(predicted)[:, None])
        )
        aspect_score = np.maximum(0.0, 1.0 - np.minimum(1.0, aspect_change))
        return ((0.68 * center_score) + (0.22 * scale_score) + (0.10 * aspect_score)).astype(np.float32)

    def _path_matrix(self, tracks: Sequence[mot.TrackState], candidates: np.ndarray) -> np.ndarray:
        output_count = candidates.shape[0]
        candidate_centers = self._bbox_center_array(candidates)
        path = np.full((len(tracks), output_count), 0.5, dtype=np.float32)
        for row_index, track in enumerate(tracks):
            if len(track.reliable_trajectory) < 2:
                continue
            samples = track.reliable_trajectory[-6:]
            first_frame, first_bbox = samples[0]
            last_frame, last_bbox = samples[-1]
            dt = max(1, int(last_frame) - int(first_frame))
            first_center = np.asarray(mot.bbox_center(first_bbox), dtype=np.float32)
            last_center = np.asarray(mot.bbox_center(last_bbox), dtype=np.float32)
            velocity = (last_center - first_center) / float(dt)
            speed = float(np.linalg.norm(velocity))
            reference_diagonal = max(1.0, mot.bbox_diagonal(last_bbox))
            recent_centers = [np.asarray(mot.bbox_center(bbox), dtype=np.float32) for _, bbox in samples]
            recent_steps = [
                float(np.linalg.norm(right - left))
                for left, right in zip(recent_centers, recent_centers[1:])
            ]
            median_step = float(np.median(np.asarray(recent_steps, dtype=np.float32))) if recent_steps else 0.0
            is_directional = speed >= mot.DEFAULT_CENTER_PATH_DIRECTION_MIN_SPEED and median_step >= 1.0
            candidate_vectors = candidate_centers - last_center[None, :]
            candidate_distances = np.linalg.norm(candidate_vectors, axis=1)
            if not is_directional:
                local_radius = max(
                    8.0,
                    min(
                        mot.DEFAULT_CENTER_PATH_STATIONARY_RADIUS,
                        (median_step * mot.DEFAULT_CENTER_PATH_STATIONARY_STEP_FACTOR)
                        + (reference_diagonal * mot.DEFAULT_CENTER_PATH_STATIONARY_BOX_FACTOR),
                    ),
                )
                row = np.maximum(0.0, 1.0 - np.minimum(1.0, candidate_distances / local_radius))
                previous_vector = recent_centers[-1] - recent_centers[-2]
                previous_distance = float(np.linalg.norm(previous_vector))
                reversal_distance = max(6.0, median_step * mot.DEFAULT_CENTER_PATH_REVERSAL_STEP_FACTOR)
                reversal_mask = (previous_distance >= 2.0) & (candidate_distances >= reversal_distance)
                if bool(np.any(reversal_mask)):
                    cosine = np.sum(previous_vector[None, :] * candidate_vectors, axis=1) / np.maximum(
                        previous_distance * candidate_distances,
                        1e-6,
                    )
                    row = np.where(
                        reversal_mask & (cosine <= mot.DEFAULT_CENTER_PATH_REVERSAL_MIN_COSINE),
                        row * mot.DEFAULT_CENTER_PATH_REVERSAL_PENALTY,
                        row,
                    )
                path[row_index, :] = np.clip(row, 0.0, 1.0).astype(np.float32)
                continue

            frames_since_reliable = max(1, track.lost_frames + 1)
            expected_center = last_center + (velocity * float(frames_since_reliable))
            distance_score = np.maximum(
                0.0,
                1.0 - np.minimum(1.0, np.linalg.norm(candidate_centers - expected_center[None, :], axis=1) / reference_diagonal),
            )
            unit_velocity = velocity / max(speed, 1e-6)
            direction_score = np.ones(output_count, dtype=np.float32)
            lateral_score = np.ones(output_count, dtype=np.float32)
            moving = candidate_distances >= 1.0
            if bool(np.any(moving)):
                unit_candidate = candidate_vectors[moving] / np.maximum(candidate_distances[moving, None], 1e-6)
                direction_score[moving] = np.maximum(0.0, unit_candidate @ unit_velocity)
                lateral_distance = np.abs(
                    (unit_velocity[0] * candidate_vectors[moving, 1])
                    - (unit_velocity[1] * candidate_vectors[moving, 0])
                )
                lateral_score[moving] = np.maximum(0.0, 1.0 - np.minimum(1.0, lateral_distance / reference_diagonal))
            row = (0.55 * distance_score) + (0.25 * direction_score) + (0.20 * lateral_score)
            path[row_index, :] = np.clip(row, 0.0, 1.0).astype(np.float32)
        return path

    def _occlusion_matrices(self, tracks: Sequence[mot.TrackState], candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        track_count = len(tracks)
        output_count = candidates.shape[0]
        occlusion_iou = np.zeros((track_count, output_count), dtype=np.float32)
        occlusion_track_ids = np.full((track_count, output_count), -1, dtype=np.int32)
        references = self._xywh_array([track.predicted_bbox or track.bbox for track in tracks])
        ok_mask = np.asarray([bool(track.ok) for track in tracks], dtype=bool)
        if not bool(ok_mask.any()):
            return occlusion_track_ids, occlusion_iou
        candidate_to_track_iou = self._pairwise_iou_matrix(candidates, references)
        for row_index, track in enumerate(tracks):
            eligible = ok_mask.copy()
            eligible[row_index] = False
            if not bool(eligible.any()):
                continue
            scores = np.where(eligible[None, :], candidate_to_track_iou, -np.inf)
            best_indices = np.argmax(scores, axis=1)
            best_values = scores[np.arange(output_count), best_indices]
            finite = np.isfinite(best_values)
            occlusion_iou[row_index, finite] = best_values[finite].astype(np.float32)
            for col_index in np.where(finite)[0].tolist():
                occlusion_track_ids[row_index, col_index] = int(tracks[int(best_indices[col_index])].track_id)
        return occlusion_track_ids, occlusion_iou

    def _identity_score_matrices(
        self,
        tracks: Sequence[mot.TrackState],
        outputs: Sequence[mot.LoRATSlotOutput],
    ) -> Dict[str, np.ndarray]:
        track_count = len(tracks)
        output_count = len(outputs)
        candidate_features, candidate_valid, prefer_crop = self._output_feature_stack(outputs)
        appearance, initial_anchor, other_anchor, other_track_ids = self._track_memory_similarity_rows(
            tracks,
            candidate_features,
            candidate_valid,
            prefer_crop=prefer_crop,
        )
        negative_anchor = self._track_negative_similarity_rows(
            tracks,
            candidate_features,
            candidate_valid,
            prefer_crop=prefer_crop,
        )
        track_boxes = self._xywh_array([track.bbox for track in tracks])
        predicted_boxes = self._xywh_array([mot.kalman_prediction_reference(track) for track in tracks])
        candidate_boxes = self._xywh_array([output.bbox for output in outputs])
        reference_diagonal = np.maximum(1.0, np.hypot(track_boxes[:, 2], track_boxes[:, 3]))
        motion = self._motion_matrix(predicted_boxes, candidate_boxes, reference_diagonal)
        path = self._path_matrix(tracks, candidate_boxes)
        iou = np.maximum(
            self._pairwise_iou_matrix(track_boxes, candidate_boxes),
            self._pairwise_iou_matrix(predicted_boxes, candidate_boxes),
        ).astype(np.float32)
        occlusion_track_ids, occlusion_iou = self._occlusion_matrices(tracks, candidate_boxes)
        track_ids = np.asarray([int(track.track_id) for track in tracks], dtype=np.int32)
        source_ids = np.asarray([int(output.source_track_id) for output in outputs], dtype=np.int32)
        source = np.where(track_ids[:, None] == source_ids[None, :], 1.0, 0.24).astype(np.float32)
        cross_source = track_ids[:, None] != source_ids[None, :]
        local_owner_only = np.asarray([self._requires_local_owner(track) for track in tracks], dtype=bool)[:, None]
        confidence = np.asarray(
            [0.5 if output.confidence is None else max(0.0, min(1.0, float(output.confidence))) for output in outputs],
            dtype=np.float32,
        )[None, :].repeat(track_count, axis=0)
        identity_margin = initial_anchor - other_anchor
        negative_margin = initial_anchor - negative_anchor

        anchor_conflict = (
            (other_track_ids >= 0)
            & (occlusion_iou >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU)
            & (other_anchor >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER)
            & (identity_margin <= -mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN)
        )
        source = np.where(anchor_conflict, source * mot.DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY, source)
        motion = np.where(anchor_conflict, motion * mot.DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY, motion)

        negative_conflict = (
            (negative_anchor >= DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY)
            & (negative_margin <= 0.03)
            & (confidence < 0.92)
        )
        source = np.where(negative_conflict, source * 0.25, source)
        motion = np.where(negative_conflict & (motion < 0.70), motion * 0.65, motion)
        path = np.where(negative_conflict & (path < 0.70), path * 0.70, path)
        source = np.where(cross_source & (appearance < max(0.30, self.min_reid)), source * 0.35, source)
        has_feature = np.asarray([self.track_has_feature_appearance(track) for track in tracks], dtype=bool)[:, None]
        motion = np.where(has_feature & (appearance < self.min_reid), motion * 0.45, motion)
        weak_initial_anchor = has_feature & (initial_anchor < max(0.45, self.min_reid - 0.02)) & (appearance < max(0.52, self.min_reid))
        source = np.where(weak_initial_anchor, source * 0.60, source)
        motion = np.where(weak_initial_anchor, motion * 0.70, motion)
        ambiguous_initial_anchor = has_feature & (other_track_ids >= 0) & (other_anchor >= 0.60) & (identity_margin < -0.03)
        source = np.where(ambiguous_initial_anchor, source * 0.45, source)
        motion = np.where(ambiguous_initial_anchor, motion * 0.70, motion)
        source = np.where(local_owner_only & cross_source, source * 0.10, source)
        motion = np.where(local_owner_only & cross_source, motion * 0.50, motion)
        path = np.where(local_owner_only & cross_source, path * 0.50, path)
        source = np.where((motion < self.min_motion) & (confidence < 0.70), source * 0.65, source)
        source = np.where((path < self.min_path) & (confidence < 0.75), source * 0.55, source)

        total = np.clip(
            (0.38 * appearance)
            + (0.22 * motion)
            + (0.18 * path)
            + (0.08 * source)
            + (0.10 * confidence)
            + (0.04 * iou),
            0.0,
            1.0,
        ).astype(np.float32)
        total = np.clip(total - (DEFAULT_V8_DISTRACTOR_PENALTY * negative_conflict.astype(np.float32)), 0.0, 1.0)
        total = np.where(local_owner_only & cross_source, total * 0.20, total).astype(np.float32)
        return {
            "total": total,
            "appearance": appearance.astype(np.float32),
            "motion": motion.astype(np.float32),
            "path": path.astype(np.float32),
            "source": source.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "iou": iou.astype(np.float32),
            "initial_anchor": initial_anchor.astype(np.float32),
            "other_anchor": other_anchor.astype(np.float32),
            "other_track_ids": other_track_ids.astype(np.int32),
            "identity_margin": identity_margin.astype(np.float32),
            "occlusion_track_ids": occlusion_track_ids.astype(np.int32),
            "occlusion_iou": occlusion_iou.astype(np.float32),
            "negative_anchor": negative_anchor.astype(np.float32),
        }

    @staticmethod
    def _identity_score_from_matrices(matrices: Dict[str, np.ndarray], row: int, col: int) -> mot.IdentityScore:
        other_track_id_value = int(matrices["other_track_ids"][row, col])
        occlusion_track_id_value = int(matrices["occlusion_track_ids"][row, col])
        return mot.IdentityScore(
            total=float(matrices["total"][row, col]),
            appearance=float(matrices["appearance"][row, col]),
            motion=float(matrices["motion"][row, col]),
            path=float(matrices["path"][row, col]),
            source=float(matrices["source"][row, col]),
            confidence=float(matrices["confidence"][row, col]),
            iou=float(matrices["iou"][row, col]),
            initial_anchor=float(matrices["initial_anchor"][row, col]),
            other_anchor=float(matrices["other_anchor"][row, col]),
            other_track_id=None if other_track_id_value < 0 else other_track_id_value,
            identity_margin=float(matrices["identity_margin"][row, col]),
            occlusion_track_id=None if occlusion_track_id_value < 0 else occlusion_track_id_value,
            occlusion_iou=float(matrices["occlusion_iou"][row, col]),
            negative_anchor=float(matrices.get("negative_anchor", np.zeros_like(matrices["total"]))[row, col]),
        )

    @staticmethod
    def _assignment_margin_from_matrix(score_matrix: np.ndarray, row: int, assigned_col: int) -> float:
        if score_matrix.shape[1] <= 1:
            return 1.0
        assigned_score = float(score_matrix[row, assigned_col])
        alternatives = np.delete(score_matrix[row], assigned_col)
        if alternatives.size <= 0:
            return 1.0
        return assigned_score - float(np.max(alternatives))

    @staticmethod
    def _solve_assignment_matrix(score_matrix: np.ndarray, min_score: float) -> List[Tuple[int, int, float]]:
        if score_matrix.size == 0 or score_matrix.shape[0] == 0 or score_matrix.shape[1] == 0:
            return []
        row_indices, col_indices = mot.linear_sum_assignment(-score_matrix)
        assignments: List[Tuple[int, int, float]] = []
        for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
            score = float(score_matrix[row_index, col_index])
            if score >= min_score:
                assignments.append((int(row_index), int(col_index), score))
        assignments.sort(key=lambda item: item[0])
        return assignments

    def assignment_gate(
        self,
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        score: mot.IdentityScore,
    ) -> Tuple[bool, str]:
        """Apply one identity policy to both Hungarian and fallback assignments."""

        is_view_change = self.is_view_change_candidate(track, output, score)
        if score.total < self.min_score and not is_view_change:
            return False, "LOW_IDENTITY_SCORE"
        if self.track_has_feature_appearance(track) and score.appearance < self.min_reid and not is_view_change:
            return False, "LOW_REID_SIMILARITY"
        anchor_conflict = (
            score.other_track_id is not None
            and score.occlusion_iou >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and score.other_anchor >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and score.identity_margin <= -mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )
        if anchor_conflict and not is_view_change:
            return False, "ANCHOR_CONFLICT"
        if (
            score.negative_anchor >= DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY
            and score.negative_anchor >= score.initial_anchor - 0.03
            and score.confidence < 0.92
            and not is_view_change
        ):
            return False, "NEGATIVE_MEMORY"
        ambiguous_anchor = (
            score.other_track_id is not None
            and score.other_anchor >= max(0.60, score.initial_anchor + 0.04)
            and score.identity_margin < -0.03
            and (score.occlusion_iou >= 0.05 or output.source_track_id != track.track_id)
        )
        if ambiguous_anchor and not is_view_change:
            return False, "AMBIGUOUS_IDENTITY"
        if self._requires_local_owner(track) and output.source_track_id != track.track_id:
            return False, "HEALTHY_LOCAL_OWNER"
        if output.source_track_id != track.track_id and not is_view_change:
            if score.appearance < max(self.min_reid, 0.50):
                return False, "CROSS_SOURCE_LOW_REID"
            if score.motion < max(self.min_motion, 0.30) and score.path < max(self.min_path, 0.45):
                return False, "CROSS_SOURCE_GEOMETRY"
            if score.identity_margin < -0.02:
                return False, "CROSS_SOURCE_ANCHOR_MARGIN"
        return True, ""

    def feature_similarity(self, left, right) -> float:
        left = self.normalize_feature(left)
        right = self.normalize_feature(right)
        if left is None or right is None or left.shape != right.shape:
            return 0.0
        cosine = float(self.torch.dot(left, right).detach().to(device="cpu", dtype=self.torch.float32).item())
        return max(0.0, min(1.0, (cosine + 1.0) * 0.5))

    def track_feature_similarity(self, track: mot.TrackState, candidate, prefer_crop: bool = False) -> float:
        candidate = self.normalize_feature(candidate)
        if candidate is None:
            return 0.50
        scores: List[float] = []
        initial_score: Optional[float] = None
        initial_name, current_name, bank_name = self._track_feature_names(prefer_crop)
        fallback_initial_name, fallback_current_name, fallback_bank_name = self._track_feature_names(not prefer_crop)
        initial = getattr(track, initial_name, None)
        if initial is None:
            initial = getattr(track, fallback_initial_name, None)
        if initial is not None:
            initial_score = self.feature_similarity(initial, candidate)
            scores.append(initial_score)
        current = getattr(track, current_name, None)
        if current is None:
            current = getattr(track, fallback_current_name, None)
        if current is not None:
            scores.append(self.feature_similarity(current, candidate))
        memories = list(getattr(track, bank_name, []))
        if not memories:
            memories = list(getattr(track, fallback_bank_name, []))
        for memory in memories:
            scores.append(self.feature_similarity(memory, candidate))
        if not scores:
            return 0.50
        scores.sort(reverse=True)
        best_pair = scores[0] if len(scores) == 1 else (0.72 * scores[0]) + (0.28 * scores[1])
        if initial_score is None:
            return best_pair
        top_count = min(4, len(scores))
        top_average = float(sum(scores[:top_count]) / top_count)
        return (0.44 * best_pair) + (0.42 * initial_score) + (0.14 * top_average)

    def resolve(
        self,
        tracks: Sequence[mot.TrackState],
        outputs: Sequence[mot.LoRATSlotOutput],
        frame: Optional[np.ndarray],
    ) -> List[mot.IdentityAssignment]:
        if not self.enabled:
            self._clear_score_cache()
            return self._owned_only_assignments(tracks, outputs)
        if not tracks or not outputs:
            self._clear_score_cache()
            return []

        score_matrices = self._identity_score_matrices(tracks, outputs)
        self._remember_score_matrices(tracks, outputs, score_matrices)
        score_matrix = score_matrices["total"]
        gated_score_matrix = np.full_like(score_matrix, -1_000_000.0, dtype=np.float32)
        score_parts_by_pair: Dict[Tuple[int, int], mot.IdentityScore] = {}
        for row, track in enumerate(tracks):
            for col, output in enumerate(outputs):
                score_parts = self._identity_score_from_matrices(score_matrices, row, col)
                score_parts_by_pair[(row, col)] = score_parts
                accepted, reject_state = self.assignment_gate(track, output, score_parts)
                if accepted:
                    gated_score_matrix[row, col] = score_matrix[row, col]
                elif self._should_remember_negative_reject(reject_state):
                    self.remember_negative_candidate(track, output)
        assignments = []
        assignment_floor = min(self.min_score, self.view_change_min_score)
        for row, col, score in self._solve_assignment_matrix(gated_score_matrix, assignment_floor):
            score_parts = score_parts_by_pair[(row, col)]
            track = tracks[row]
            output = outputs[col]
            assignments.append(
                mot.IdentityAssignment(
                    track=track,
                    output=output,
                    score=score_parts,
                    assignment_margin=self._assignment_margin_from_matrix(score_matrix, row, col),
                )
            )
        return assignments

    def score(
        self,
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        all_tracks: Sequence[mot.TrackState] = (),
    ) -> mot.IdentityScore:
        confidence = 0.5 if output.confidence is None else max(0.0, min(1.0, float(output.confidence)))
        candidate, prefer_crop = self.output_feature_with_source(output)
        appearance = 0.5
        initial_anchor = 0.5
        other_anchor = 0.0
        other_track_id: Optional[int] = None
        if candidate is not None:
            appearance = self.track_feature_similarity(track, candidate, prefer_crop=prefer_crop)
            initial_name, _, _ = self._track_feature_names(prefer_crop)
            fallback_initial_name, _, _ = self._track_feature_names(not prefer_crop)
            initial = getattr(track, initial_name, None)
            if initial is None:
                initial = getattr(track, fallback_initial_name, None)
            if initial is not None:
                initial_anchor = self.feature_similarity(initial, candidate)
            else:
                initial_anchor = appearance
            for other in all_tracks:
                other_initial = getattr(other, initial_name, None)
                if other_initial is None:
                    other_initial = getattr(other, fallback_initial_name, None)
                if other.track_id == track.track_id or other_initial is None:
                    continue
                other_score = self.feature_similarity(other_initial, candidate)
                if other_score > other_anchor:
                    other_anchor = other_score
                    other_track_id = other.track_id

        predicted = mot.kalman_prediction_reference(track)
        reference_diagonal = max(1.0, mot.bbox_diagonal(track.bbox))
        motion = mot.motion_affinity(predicted, output.bbox, reference_diagonal)
        path = mot.center_path_affinity(track, output.bbox)
        iou = max(mot.bbox_iou(track.bbox, output.bbox), mot.bbox_iou(predicted, output.bbox))
        occlusion_track_id, occlusion_iou = mot.strongest_track_overlap(track, output.bbox, all_tracks)
        source = 1.0 if output.source_track_id == track.track_id else 0.24

        identity_margin = initial_anchor - other_anchor
        anchor_conflict = (
            other_track_id is not None
            and occlusion_iou >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and other_anchor >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and identity_margin <= -mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )
        if anchor_conflict:
            source *= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY
            motion *= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_PENALTY

        if output.source_track_id != track.track_id and appearance < max(0.30, self.min_reid):
            source *= 0.35
        if self.track_has_feature_appearance(track) and appearance < self.min_reid:
            motion *= 0.45
        if self.track_has_feature_appearance(track) and initial_anchor < max(0.45, self.min_reid - 0.02) and appearance < max(0.52, self.min_reid):
            source *= 0.60
            motion *= 0.70
        if other_track_id is not None and other_anchor >= 0.60 and identity_margin < -0.03:
            source *= 0.45
            motion *= 0.70
        negative_anchor = 0.0
        if candidate is not None:
            negative_values = self._track_negative_similarity_rows(
                [track],
                candidate.reshape(1, -1),
                np.asarray([True], dtype=bool),
                prefer_crop=prefer_crop,
            )
            negative_anchor = float(negative_values[0, 0]) if negative_values.size else 0.0
        if negative_anchor >= DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY and negative_anchor >= initial_anchor - 0.03:
            source *= 0.25
            if motion < 0.70:
                motion *= 0.65
            if path < 0.70:
                path *= 0.70
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
        if negative_anchor >= DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY and negative_anchor >= initial_anchor - 0.03:
            total -= DEFAULT_V8_DISTRACTOR_PENALTY
        if self._requires_local_owner(track) and output.source_track_id != track.track_id:
            total *= 0.20
        return mot.IdentityScore(
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
            negative_anchor=negative_anchor,
        )

    def commit_track_memory(
        self,
        track: mot.TrackState,
        output: mot.LoRATSlotOutput,
        assignment: mot.IdentityAssignment,
        frame: Optional[np.ndarray],
    ) -> None:
        feature, prefer_crop = self.output_feature_with_source(output)
        if feature is None:
            return
        is_view_change = self.is_view_change_candidate(track, output, assignment.score)
        if assignment.score.confidence < self.memory_min_confidence:
            return
        if self.track_has_feature_appearance(track) and assignment.score.appearance < max(self.min_reid, 0.30) and not is_view_change:
            return
        initial_name, current_name, bank_name = self._track_feature_names(prefer_crop)
        if getattr(track, initial_name, None) is not None and not is_view_change:
            if assignment.score.initial_anchor < max(self.min_reid, 0.50):
                return
            if (
                assignment.score.negative_anchor >= DEFAULT_V8_DISTRACTOR_MIN_SIMILARITY
                and assignment.score.negative_anchor >= assignment.score.initial_anchor - 0.03
            ):
                return
            if (
                assignment.score.other_track_id is not None
                and assignment.score.other_anchor >= 0.55
                and assignment.score.identity_margin < 0.0
            ):
                return
        if assignment.score.motion < self.min_motion and not is_view_change:
            return
        if assignment.score.total < max(self.min_score, 0.50) and not is_view_change:
            return

        initial = getattr(track, initial_name, None)
        if initial is None:
            setattr(track, initial_name, feature.detach().clone())
        current = getattr(track, current_name, None)
        if current is None:
            setattr(track, current_name, feature.detach().clone())
        else:
            update_rate = self.appearance_update_rate * (0.5 if is_view_change else 1.0)
            updated = self.F.normalize(((1.0 - update_rate) * current) + (update_rate * feature), dim=0)
            setattr(track, current_name, updated.detach().clone())

        bank = list(getattr(track, bank_name, []))
        if not bank or self.feature_similarity(bank[-1], feature) < 0.985:
            bank.append(feature.detach().clone())
            if len(bank) > self.appearance_bank_size:
                del bank[: len(bank) - self.appearance_bank_size]
        setattr(track, bank_name, bank)
        track.appearance_updates += 1


class V8QualityBatchedLoRATTracker:
    """Standalone V8 tracker with one shared LoRAT ViT frame pass per video frame.

    Upstream LoRAT fuses template and search tokens inside the ViT blocks, so the exact
    original SOT head cannot be reused without per-object transformer work. This branch
    keeps LoRAT's LoRA-adapted DINOv2 blocks as the shared frame backbone and moves the
    object-specific work into a small batched low-rank head bank.

    V8 intentionally does not subclass the previous tracker versions and does not call the
    upstream per-object LoRAT evaluator in its frame update path. Shared dataclasses,
    geometry helpers, debug writers, and UI/output helpers now live in mot_common.
    Compared with v7, it adds guarded shared-feature template recovery so the
    untrained shared head can recover some of the v6 quality behavior without losing
    the Week 2 property: one shared backbone pass plus one batched per-object head pass.
    """

    backend_name = "LoRAT-v8-quality-batched"

    def __init__(
        self,
        lorat_root: Path,
        config_name: str,
        weight_path: Path,
        device: str,
        max_tracks: int,
        fps: Optional[float],
        sequence_length: Optional[int],
        sequence_name: str,
        disable_amp: bool,
        frame_size: int = 0,
        head_rank: int = mot.DEFAULT_LORAT_MEMORY_SLOTS,
        head_hidden_dim: int = 256,
        head_lora_rank: int = 16,
        head_weight_path: Optional[Path] = None,
        search_radius_factor: float = 2.25,
        min_confidence: float = 0.48,
        template_update_rate: float = 0.08,
        template_update_min_confidence: float = 0.58,
        lorat_memory_slots: int = mot.DEFAULT_LORAT_MEMORY_SLOTS,
        lorat_memory_refresh_interval: int = mot.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL,
        lorat_memory_min_score: float = 0.55,
        lorat_accept_min_score: float = mot.DEFAULT_LORAT_ACCEPT_MIN_SCORE,
        lorat_fixed_box_size: bool = mot.DEFAULT_LORAT_FIXED_BOX_SIZE,
        lorat_min_box_area: float = mot.DEFAULT_LORAT_MIN_BOX_AREA,
        lorat_max_area_change_per_frame: float = mot.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME,
        lorat_trusted_size_floor_scale: float = mot.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE,
        shrink_guard_window: int = mot.DEFAULT_SHRINK_GUARD_WINDOW,
        shrink_guard_area_ratio: float = mot.DEFAULT_SHRINK_GUARD_AREA_RATIO,
        shrink_guard_step_ratio: float = mot.DEFAULT_SHRINK_GUARD_STEP_RATIO,
        shrink_guard_min_confidence: float = mot.DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE,
        shrink_guard_min_reid: float = mot.DEFAULT_SHRINK_GUARD_MIN_REID,
        crop_information_min_score: float = mot.DEFAULT_CROP_INFORMATION_MIN_SCORE,
        crop_information_min_pixels: int = mot.DEFAULT_CROP_INFORMATION_MIN_PIXELS,
        identity_arbitration: bool = True,
        identity_min_score: float = mot.DEFAULT_IDENTITY_MIN_SCORE,
        identity_min_reid: float = mot.DEFAULT_IDENTITY_MIN_REID,
        identity_min_motion: float = mot.DEFAULT_IDENTITY_MIN_MOTION,
        identity_min_path: float = mot.DEFAULT_IDENTITY_MIN_PATH,
        identity_bank_size: int = 12,
        identity_memory_min_confidence: float = mot.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE,
        occlusion_max_frames: int = mot.DEFAULT_OCCLUSION_MAX_FRAMES,
        occlusion_iou_threshold: float = mot.DEFAULT_OCCLUSION_IOU_THRESHOLD,
        occlusion_velocity_damping: float = mot.DEFAULT_OCCLUSION_VELOCITY_DAMPING,
        reid_recovery_min_score: float = mot.DEFAULT_REID_RECOVERY_MIN_SCORE,
        reid_recovery_min_reid: float = mot.DEFAULT_REID_RECOVERY_MIN_REID,
        reid_recovery_min_motion: float = mot.DEFAULT_REID_RECOVERY_MIN_MOTION,
        reid_recovery_min_confidence: float = mot.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE,
        view_change_min_score: float = mot.DEFAULT_VIEW_CHANGE_MIN_SCORE,
        view_change_min_motion: float = mot.DEFAULT_VIEW_CHANGE_MIN_MOTION,
        view_change_min_confidence: float = mot.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE,
        view_change_max_lost_frames: int = mot.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES,
        v8_primary_heads_per_track: int = DEFAULT_V8_PRIMARY_HEADS_PER_TRACK,
        v8_recovery_heads_per_track: int = DEFAULT_V8_RECOVERY_HEADS_PER_TRACK,
        v8_recovery_interval: int = DEFAULT_V8_RECOVERY_INTERVAL,
        v8_recovery_min_confidence: float = DEFAULT_V8_RECOVERY_MIN_CONFIDENCE,
        v8_recovery_min_assignment_score: float = DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_SCORE,
        v8_recovery_min_assignment_margin: float = DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_MARGIN,
        v8_recovery_stale_head_frames: int = DEFAULT_V8_RECOVERY_STALE_HEAD_FRAMES,
        score_reduction: str = "max",
        collect_slot_debug: bool = False,
        collect_week2_proof: bool = False,
        v8_template_match: bool = DEFAULT_V8_TEMPLATE_MATCH_ENABLED,
        v8_template_match_min_score: float = DEFAULT_V8_TEMPLATE_MATCH_MIN_SCORE,
        v8_template_match_prefer_margin: float = DEFAULT_V8_TEMPLATE_MATCH_PREFER_MARGIN,
        v8_template_match_on_uncertain_only: bool = DEFAULT_V8_TEMPLATE_MATCH_ON_UNCERTAIN_ONLY,
        v8_template_match_head_confidence_gate: float = DEFAULT_V8_TEMPLATE_MATCH_HEAD_CONFIDENCE_GATE,
        v8_template_match_margin_gate: float = DEFAULT_V8_TEMPLATE_MATCH_MARGIN_GATE,
        v8_head_template_blend: float = DEFAULT_V8_HEAD_TEMPLATE_BLEND,
        v8_memory_min_motion: float = DEFAULT_V8_MEMORY_MIN_MOTION,
        v8_memory_min_path: float = DEFAULT_V8_MEMORY_MIN_PATH,
        v8_memory_min_appearance: float = DEFAULT_V8_MEMORY_MIN_APPEARANCE,
        v8_memory_min_stable_updates: int = DEFAULT_V8_MEMORY_MIN_STABLE_UPDATES,
        v8_accept_min_initial_anchor: float = DEFAULT_V8_ACCEPT_MIN_INITIAL_ANCHOR,
        v8_accept_min_identity_margin: float = DEFAULT_V8_ACCEPT_MIN_IDENTITY_MARGIN,
        v8_memory_min_initial_anchor: float = DEFAULT_V8_MEMORY_MIN_INITIAL_ANCHOR,
        v8_memory_min_identity_margin: float = DEFAULT_V8_MEMORY_MIN_IDENTITY_MARGIN,
        v8_window_penalty_ratio: float = DEFAULT_V8_WINDOW_PENALTY_RATIO,
        v8_dinov2_crop_reid: bool = DEFAULT_V8_DINOV2_CROP_REID,
        v8_dinov2_crop_reid_batch: int = DEFAULT_V8_DINOV2_CROP_REID_BATCH,
        v8_dinov2_crop_reid_min_area: float = DEFAULT_V8_DINOV2_CROP_REID_MIN_AREA,
        v8_assignment_conflict_iou: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_IOU,
        v8_assignment_conflict_hard_iou: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_HARD_IOU,
        v8_assignment_conflict_score_margin: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_SCORE_MARGIN,
        v8_assignment_conflict_center_ratio: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_CENTER_RATIO,
        v8_assignment_conflict_containment: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_CONTAINMENT,
        v8_assignment_conflict_ownership_margin: float = DEFAULT_V8_ASSIGNMENT_CONFLICT_OWNERSHIP_MARGIN,
        v8_assignment_alt_rescue: bool = DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_ENABLED,
        v8_assignment_alt_rescue_max_candidates: int = DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MAX_CANDIDATES,
        v8_assignment_alt_rescue_min_confidence: float = DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MIN_CONFIDENCE,
        v8_small_target_mode: bool = DEFAULT_V8_SMALL_TARGET_MODE,
        v8_small_target_area: float = DEFAULT_V8_SMALL_TARGET_AREA,
        v8_small_target_max_side: float = DEFAULT_V8_SMALL_TARGET_MAX_SIDE,
        v8_small_target_max_scale_change: float = DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE,
        v8_small_target_template_min_score: float = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE,
        v8_small_target_template_min_motion: float = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION,
        v8_small_target_template_min_path: float = DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH,
        v8_small_target_confidence_floor: float = DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR,
    ) -> None:
        self.lorat_root = lorat_root.resolve()
        self.config_name = config_name
        self.weight_path = weight_path.resolve()
        self.device_string = device
        self.max_tracks = max(0, int(max_tracks))
        self.fps = fps
        self.sequence_length = sequence_length
        self.sequence_name = sequence_name
        self.disable_amp = bool(disable_amp)
        self.frame_size_override = max(0, int(frame_size))
        self.lorat_memory_slots = max(1, min(mot.MAX_LORAT_MEMORY_SLOTS, int(lorat_memory_slots)))
        self.head_rank = max(1, min(mot.MAX_LORAT_MEMORY_SLOTS, int(head_rank or self.lorat_memory_slots)))
        self.head_hidden_dim = max(16, int(head_hidden_dim))
        self.head_lora_rank = max(1, int(head_lora_rank))
        self.head_weight_path = head_weight_path.resolve() if head_weight_path is not None else None
        self.search_radius_factor = max(0.25, float(search_radius_factor))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.template_update_rate = max(0.0, min(1.0, float(template_update_rate)))
        self.template_update_min_confidence = max(0.0, min(1.0, float(template_update_min_confidence)))
        self.lorat_memory_refresh_interval = max(1, int(lorat_memory_refresh_interval))
        self.lorat_memory_min_score = max(0.0, float(lorat_memory_min_score))
        self.lorat_accept_min_score = max(0.0, min(1.0, float(lorat_accept_min_score)))
        self.lorat_fixed_box_size = bool(lorat_fixed_box_size)
        self.lorat_min_box_area = max(0.0, float(lorat_min_box_area))
        self.lorat_max_area_change_per_frame = max(0.0, float(lorat_max_area_change_per_frame))
        self.lorat_trusted_size_floor_scale = max(0.0, min(1.0, float(lorat_trusted_size_floor_scale)))
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
        self._identity_arbitrator_kwargs = {
            "enabled": identity_arbitration,
            "min_score": identity_min_score,
            "min_reid": identity_min_reid,
            "min_motion": identity_min_motion,
            "min_path": identity_min_path,
            "appearance_bank_size": identity_bank_size,
            "memory_min_confidence": identity_memory_min_confidence,
            "view_change_min_score": view_change_min_score,
            "view_change_min_motion": view_change_min_motion,
            "view_change_min_confidence": view_change_min_confidence,
            "view_change_max_lost_frames": view_change_max_lost_frames,
        }
        self.identity_arbitrator: Optional[V8FeatureIdentityArbitrator] = None
        self.v8_primary_heads_per_track = max(1, int(v8_primary_heads_per_track))
        self.v8_recovery_heads_per_track = max(1, int(v8_recovery_heads_per_track))
        self.v8_recovery_interval = max(0, int(v8_recovery_interval))
        self.v8_recovery_min_confidence = max(0.0, min(1.0, float(v8_recovery_min_confidence)))
        self.v8_recovery_min_assignment_score = max(0.0, min(1.0, float(v8_recovery_min_assignment_score)))
        self.v8_recovery_min_assignment_margin = max(0.0, float(v8_recovery_min_assignment_margin))
        self.v8_recovery_stale_head_frames = max(0, int(v8_recovery_stale_head_frames))
        self.v8_gating_decisions = 0
        self.v8_primary_decisions = 0
        self.v8_recovery_decisions = 0
        self.v8_selected_head_items = 0
        self.v8_recovery_reason_counts: Counter[str] = Counter()
        self.v8_assignment_conflict_reason_counts: Counter[str] = Counter()
        self.v8_assignment_alt_rescue_attempts = 0
        self.v8_assignment_alt_rescue_hits = 0
        self.v8_assignment_alt_rescue_reject_counts: Counter[str] = Counter()
        self.score_reduction = score_reduction
        if self.score_reduction not in {"max", "mean"}:
            raise ValueError("--v8-score-reduction must be 'max' or 'mean'.")
        self.collect_slot_debug = bool(collect_slot_debug)
        self.collect_week2_proof = bool(collect_week2_proof)
        self.v8_template_match = bool(v8_template_match)
        self.v8_template_match_min_score = max(0.0, min(1.0, float(v8_template_match_min_score)))
        self.v8_template_match_prefer_margin = max(0.0, float(v8_template_match_prefer_margin))
        self.v8_template_match_on_uncertain_only = bool(v8_template_match_on_uncertain_only)
        self.v8_template_match_head_confidence_gate = max(0.0, min(1.0, float(v8_template_match_head_confidence_gate)))
        self.v8_template_match_margin_gate = max(0.0, float(v8_template_match_margin_gate))
        self.v8_head_template_blend = max(0.0, min(1.0, float(v8_head_template_blend)))
        self.v8_memory_min_motion = max(0.0, min(1.0, float(v8_memory_min_motion)))
        self.v8_memory_min_path = max(0.0, min(1.0, float(v8_memory_min_path)))
        self.v8_memory_min_appearance = max(0.0, min(1.0, float(v8_memory_min_appearance)))
        self.v8_memory_min_stable_updates = max(1, int(v8_memory_min_stable_updates))
        self.v8_accept_min_initial_anchor = max(0.0, min(1.0, float(v8_accept_min_initial_anchor)))
        self.v8_accept_min_identity_margin = float(v8_accept_min_identity_margin)
        self.v8_memory_min_initial_anchor = max(0.0, min(1.0, float(v8_memory_min_initial_anchor)))
        self.v8_memory_min_identity_margin = float(v8_memory_min_identity_margin)
        self.v8_window_penalty_ratio = max(0.0, min(1.0, float(v8_window_penalty_ratio)))
        self.v8_dinov2_crop_reid = bool(v8_dinov2_crop_reid)
        self.v8_dinov2_crop_reid_batch = max(1, int(v8_dinov2_crop_reid_batch))
        self.v8_dinov2_crop_reid_min_area = max(1.0, float(v8_dinov2_crop_reid_min_area))
        self.v8_assignment_conflict_iou = max(0.0, min(1.0, float(v8_assignment_conflict_iou)))
        self.v8_assignment_conflict_hard_iou = max(
            self.v8_assignment_conflict_iou,
            min(1.0, float(v8_assignment_conflict_hard_iou)),
        )
        self.v8_assignment_conflict_score_margin = max(0.0, float(v8_assignment_conflict_score_margin))
        self.v8_assignment_conflict_center_ratio = max(0.0, float(v8_assignment_conflict_center_ratio))
        self.v8_assignment_conflict_containment = max(0.0, min(1.0, float(v8_assignment_conflict_containment)))
        self.v8_assignment_conflict_ownership_margin = float(v8_assignment_conflict_ownership_margin)
        self.v8_assignment_alt_rescue = bool(v8_assignment_alt_rescue)
        self.v8_assignment_alt_rescue_max_candidates = max(0, int(v8_assignment_alt_rescue_max_candidates))
        self.v8_assignment_alt_rescue_min_confidence = max(
            0.0,
            min(1.0, float(v8_assignment_alt_rescue_min_confidence)),
        )
        self.v8_small_target_mode = bool(v8_small_target_mode)
        self.v8_small_target_area = max(1.0, float(v8_small_target_area))
        self.v8_small_target_max_side = max(1.0, float(v8_small_target_max_side))
        self.v8_small_target_max_scale_change = max(1.0, float(v8_small_target_max_scale_change))
        self.v8_small_target_template_min_score = max(0.0, min(1.0, float(v8_small_target_template_min_score)))
        self.v8_small_target_template_min_motion = max(0.0, min(1.0, float(v8_small_target_template_min_motion)))
        self.v8_small_target_template_min_path = max(0.0, min(1.0, float(v8_small_target_template_min_path)))
        self.v8_small_target_confidence_floor = max(0.0, min(1.0, float(v8_small_target_confidence_floor)))
        self.v8_template_match_attempts = 0
        self.v8_template_match_hits = 0
        self.v8_template_fused_candidates = 0
        self.v8_template_preferred_candidates = 0

        self.tracks: List[mot.TrackState] = []
        self.track_by_id: Dict[int, mot.TrackState] = {}
        self.next_track_id = 1
        self.closed = False
        self.using_directml = False
        self.device_label = self.device_string
        self.gpu_name = ""
        self.runtime_status = mot.RuntimeStatus()
        self.slot_debug_lines: List[str] = []
        self.week2_proof_lines: List[str] = []
        self.last_candidate_diagnostics: List[Dict[str, object]] = []
        self.trajectory_history_size = 12
        self._fps_smoothing = 0.15
        self._last_backbone_seconds = 0.0
        self._last_head_seconds = 0.0
        self._last_head_mode = "none"
        self._last_roi_tokens = 0
        self._last_selected_head_count = 0
        self._last_frame_backbone_delta = 0
        self._last_frame_object_head_batch_delta = 0
        self._last_frame_object_head_items_delta = 0
        self._last_frame_selected_head_items_delta = 0
        self._last_frame_week2_shared_ok = False
        self._last_frame_week2_head_ok = False
        self._last_profile_seconds: Dict[str, float] = {bucket: 0.0 for bucket in V8_PROFILE_BUCKETS}
        self._profile_total_seconds: Dict[str, float] = {bucket: 0.0 for bucket in V8_PROFILE_BUCKETS}

        self._load_lorat_shared_backbone()

    def _load_lorat_shared_backbone(self) -> None:
        if not self.lorat_root.exists():
            raise RuntimeError(f"LoRAT checkout not found: {self.lorat_root}")
        if not self.weight_path.exists():
            raise RuntimeError(f"LoRAT weight not found: {self.weight_path}")

        lorat_root_str = str(self.lorat_root)
        if lorat_root_str not in sys.path:
            sys.path.insert(0, lorat_root_str)

        try:
            import torch
            import torch.nn.functional as F
            from trackit.core.boot.funcs.main.load_config import load_config
            from trackit.core.runtime.global_constant import get_global_constant
            from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
            from trackit.models import ModelManager
            from trackit.models.compiling.plain.builder import build_plain_inference_engine
            from trackit.models.methods.builder import create_model_build_context
        except ModuleNotFoundError as exc:
            package_name = exc.name or "unknown"
            raise RuntimeError(
                f"LoRAT dependency '{package_name}' is missing from this Python interpreter: "
                f"{sys.executable}. Run scripts/setup-lorat-env.ps1 or select the project venv."
            ) from exc

        self.torch = torch
        self.F = F

        requested_device = self.device_string.lower()
        if requested_device in {"dml", "directml"} or requested_device.startswith(("dml:", "directml:")):
            try:
                import torch_directml
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "LoRAT V8 was asked to use DirectML, but torch-directml is not installed."
                ) from exc
            device_index = 0
            if ":" in requested_device:
                device_index = int(requested_device.rsplit(":", 1)[1])
            self.device = torch_directml.device(device_index)
            self.using_directml = True
            self.device_label = f"DirectML {device_index} [{self.device}]"
        else:
            self.device = torch.device(self.device_string)
            self.device_label = str(self.device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LoRAT V8 was asked to use cuda, but this PyTorch build reports no CUDA/HIP device."
            )
        if self.device.type == "cuda":
            self.gpu_name = torch.cuda.get_device_name(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.using_directml:
            self.gpu_name = self.device_label
        self.runtime_status.gpu_name = self.gpu_name

        if get_global_constant("TIMM_USE_OLD_CACHE", default=True):
            os.environ["TIMM_USE_OLD_CACHE"] = "1"

        runtime_vars = SimpleNamespace(
            root_path=str(self.lorat_root),
            config_path=str(self.lorat_root / "config"),
            method_name="LoRAT",
            config_name=self.config_name,
            mixin_config=None,
        )
        self.config = load_config(runtime_vars)
        self.dtype = torch.float32

        model_manager = ModelManager(create_model_build_context(self.config), rng_fixed_seed=42)
        model_manager.load_state_dict_from_file(str(self.weight_path), strict=False, print_missing=False)
        self.model_manager = model_manager

        inference_config = copy.deepcopy(self.config["run"]["runner"]["test"]["inference_engine"])
        if self.device.type != "cuda" or self.disable_amp:
            inference_config["auto_mixed_precision"]["enabled"] = False
        inference_config["torch_compile"]["enabled"] = False
        inference_engine = build_plain_inference_engine(inference_config, self.device)
        self.optimized_model = inference_engine(model_manager, self.device, self.dtype, 1, 1)
        self.amp_autocast_fn = getattr(self.optimized_model.model, "amp_autocast_fn", nullcontext)

        self.lorat_model = self._unwrap_lorat_model(self.optimized_model.raw_model)
        required = ("_x_feat", "blocks", "norm", "x_size")
        if not all(hasattr(self.lorat_model, name) for name in required):
            raise RuntimeError(
                "LoRAT V8 expected a LoRAT_DINOv2-style model with _x_feat, blocks, norm, and x_size. "
                "The loaded model does not expose those internals."
            )
        self.lorat_model.eval()

        common = self.config["common"]
        search_region_size = common.get("search_region_size", (224, 224))
        self.input_height = int(search_region_size[1] if len(search_region_size) > 1 else search_region_size[0])
        self.input_width = int(search_region_size[0])
        if self.frame_size_override > 0:
            if self.frame_size_override != self.input_width or self.frame_size_override != self.input_height:
                raise ValueError(
                    "--v8-frame-size must match the configured LoRAT search_region_size. "
                    "The current shared-frame encoder reuses LoRAT's fixed positional embedding."
                )
            self.input_width = self.frame_size_override
            self.input_height = self.frame_size_override
        self.grid_width = int(self.lorat_model.x_size[0])
        self.grid_height = int(self.lorat_model.x_size[1])
        self.embed_dim = int(getattr(self.lorat_model, "embed_dim"))
        self.image_normalize_transform = get_dataset_norm_stats_transform(common["normalization"], inplace=True)
        self.shared_frame_encoder = SharedFrameLoRATEncoder(
            self.torch,
            self.F,
            self.lorat_model,
            self.amp_autocast_fn,
            self.image_normalize_transform,
            self.device,
            self.dtype,
            self.input_width,
            self.input_height,
            self.grid_width,
            self.grid_height,
            self.embed_dim,
        )
        self.object_conditioned_head = BatchedObjectConditionedHead(
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
        self.identity_arbitrator = V8FeatureIdentityArbitrator(
            self.torch,
            self.F,
            self.device,
            **self._identity_arbitrator_kwargs,
        )

        print(
            f"Loaded LoRAT V8 shared-frame backend {self.config_name} on {self.device_label} "
            f"with weight {self.weight_path.name}. "
            f"Mode: {V8_EXECUTION_MODE}; frame tensor: {self.input_width}x{self.input_height}; "
            f"feature grid: {self.grid_width}x{self.grid_height}; head rank: {self.head_rank}; "
            f"LoRA head dim/rank: {self.head_hidden_dim}/{self.head_lora_rank}; "
            f"memory heads: {self.lorat_memory_slots}; primary/recovery heads: "
            f"{self.v8_primary_heads_per_track}/{self.v8_recovery_heads_per_track}; "
            f"score reduction: {self.score_reduction}; min confidence: {self.min_confidence:.2f}; "
            f"identity arbitration: {self.identity_arbitrator.enabled}; "
            f"DINOv2 crop ReID: {self.v8_dinov2_crop_reid} "
            f"(batch {self.v8_dinov2_crop_reid_batch}); "
            f"assignment conflict IoU/hard/margin: "
            f"{self.v8_assignment_conflict_iou:.2f}/{self.v8_assignment_conflict_hard_iou:.2f}/"
            f"{self.v8_assignment_conflict_score_margin:.2f}; "
            f"center/contain/own: {self.v8_assignment_conflict_center_ratio:.2f}/"
            f"{self.v8_assignment_conflict_containment:.2f}/{self.v8_assignment_conflict_ownership_margin:.2f}; "
            f"alt rescue: {self.v8_assignment_alt_rescue} "
            f"(max {self.v8_assignment_alt_rescue_max_candidates}, "
            f"min conf {self.v8_assignment_alt_rescue_min_confidence:.2f}); "
            f"feature template recovery: {self.v8_template_match}"
            f" (uncertain-only after trained head: {self.v8_template_match_on_uncertain_only})."
        )
        if self.head_weight_path is None:
            print(
                "V8 object head weights: none supplied. Runtime will use the zero-shot shared-feature "
                "similarity head; load --v8-head-weights to use trained LoRA-conditioned heads."
            )
        else:
            print(f"V8 object head weights: {self.head_weight_path}")
        print(
            "Note: V8 uses the LoRA-adapted ViT as one shared frame backbone pass, then batches "
            "per-object low-rank heads. It intentionally does not call upstream LoRAT once per object."
        )

    @staticmethod
    def _unwrap_lorat_model(model):
        current = model
        for _ in range(4):
            if hasattr(current, "_x_feat"):
                return current
            if hasattr(current, "module"):
                current = current.module
                continue
            if hasattr(current, "model"):
                current = current.model
                continue
            break
        return current

    def _encode_frame(self, frame: np.ndarray):
        encoding = self.shared_frame_encoder.encode(frame)
        feature_map = encoding.feature_map
        self._last_backbone_seconds = encoding.elapsed_seconds
        self.runtime_status.shared_frame_backbone_calls += 1
        self.runtime_status.shared_frame_backbone_items += 1
        self.runtime_status.model_forward_calls += 1
        self.runtime_status.model_forward_items += 1
        self.runtime_status.max_model_forward_batch = max(self.runtime_status.max_model_forward_batch, 1)
        return feature_map

    def _reset_last_profile(self) -> None:
        for bucket in V8_PROFILE_BUCKETS:
            self._last_profile_seconds[bucket] = 0.0

    def _add_profile_seconds(self, bucket: str, elapsed_seconds: float) -> None:
        if bucket not in self._last_profile_seconds:
            return
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        self._last_profile_seconds[bucket] += elapsed_seconds
        self._profile_total_seconds[bucket] += elapsed_seconds

    def _profile_ms(self, bucket: str) -> float:
        return self._last_profile_seconds.get(bucket, 0.0) * 1000.0

    def _profile_total_ms_per_update(self, bucket: str) -> float:
        updates = max(1, self.runtime_status.object_head_batches)
        return (self._profile_total_seconds.get(bucket, 0.0) * 1000.0) / float(updates)

    def _week2_counter_snapshot(self) -> Tuple[int, int, int, int, int, int]:
        return (
            self.runtime_status.shared_frame_backbone_calls,
            self.runtime_status.object_head_batches,
            self.runtime_status.object_head_items,
            self.v8_selected_head_items,
            self.runtime_status.crop_reid_forward_calls,
            self.runtime_status.crop_reid_forward_items,
        )

    def _record_frame_status(self, elapsed_seconds: float) -> None:
        self.runtime_status.last_frame_seconds = elapsed_seconds
        instant_fps = 1.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0
        if self.runtime_status.fps <= 0:
            self.runtime_status.fps = instant_fps
        else:
            alpha = self._fps_smoothing
            self.runtime_status.fps = (alpha * instant_fps) + ((1.0 - alpha) * self.runtime_status.fps)
        self.runtime_status.active_objects = len([track for track in self.tracks if track.ok])
        self._update_gpu_status()

    def _append_week2_proof_row(
        self,
        frame_number: int,
        phase: str,
        tracked_objects_this_frame: int,
        before: Tuple[int, int, int, int, int, int],
    ) -> None:
        proof_started = time.perf_counter()
        status = self.runtime_status_snapshot()
        (
            backbone_before,
            head_batches_before,
            head_items_before,
            selected_heads_before,
            crop_reid_calls_before,
            crop_reid_items_before,
        ) = before
        backbone_delta = status.shared_frame_backbone_calls - backbone_before
        head_batch_delta = status.object_head_batches - head_batches_before
        head_item_delta = status.object_head_items - head_items_before
        selected_head_delta = status.gating_selected_slot_items - selected_heads_before
        crop_reid_call_delta = status.crop_reid_forward_calls - crop_reid_calls_before
        crop_reid_item_delta = status.crop_reid_forward_items - crop_reid_items_before

        self._last_frame_backbone_delta = backbone_delta
        self._last_frame_object_head_batch_delta = head_batch_delta
        self._last_frame_object_head_items_delta = head_item_delta
        self._last_frame_selected_head_items_delta = selected_head_delta
        self._last_frame_week2_shared_ok = backbone_delta == 1
        self._last_frame_week2_head_ok = (
            (tracked_objects_this_frame > 0 and head_batch_delta == 1 and head_item_delta == tracked_objects_this_frame)
            or (tracked_objects_this_frame == 0 and head_batch_delta == 0 and head_item_delta == 0)
        )

        profiled_frame_seconds = sum(
            self._last_profile_seconds.get(bucket, 0.0)
            for bucket in V8_PROFILE_BUCKETS
            if bucket != "proof_output"
        )
        unbucketed_seconds = max(
            0.0,
            status.last_frame_seconds
            - self._last_backbone_seconds
            - self._last_head_seconds
            - profiled_frame_seconds,
        )
        proof_elapsed_ms = (time.perf_counter() - proof_started) * 1000.0
        fields = [
            str(frame_number),
            mot.csv_text(phase),
            mot.csv_text(V8_EXECUTION_MODE),
            mot.csv_text(self._last_head_mode),
            str(tracked_objects_this_frame),
            str(status.active_objects),
            mot.csv_float(status.last_frame_seconds),
            mot.csv_float(status.fps),
            str(backbone_delta),
            str(head_batch_delta),
            str(head_item_delta),
            str(selected_head_delta),
            str(status.shared_frame_backbone_calls),
            str(status.object_head_batches),
            str(status.object_head_items),
            str(status.gating_selected_slot_items),
            str(status.max_object_head_batch),
            mot.csv_float(self._last_backbone_seconds * 1000.0),
            mot.csv_float(self._last_head_seconds * 1000.0),
            str(self._last_roi_tokens),
            mot.csv_float(self._profile_ms("candidate_transfer")),
            mot.csv_float(self._profile_ms("candidate_extract")),
            mot.csv_float(self._profile_ms("template_match")),
            mot.csv_float(self._profile_ms("candidate_fusion")),
            mot.csv_float(self._profile_ms("reid_appearance")),
            mot.csv_float(self._profile_ms("dinov2_crop_reid")),
            mot.csv_float(self._profile_ms("identity_resolve")),
            mot.csv_float(self._profile_ms("identity_score")),
            mot.csv_float(self._profile_ms("debug_output")),
            mot.csv_float(self._profile_ms("accept")),
            mot.csv_float(self._profile_ms("hold")),
            mot.csv_float(self._profile_ms("appearance_refresh")),
            mot.csv_float(proof_elapsed_ms),
            mot.csv_float(unbucketed_seconds * 1000.0),
            str(crop_reid_call_delta),
            str(crop_reid_item_delta),
            str(status.crop_reid_forward_calls),
            str(status.crop_reid_forward_items),
            str(status.max_crop_reid_batch),
            mot.csv_text(status.gpu_name),
            mot.csv_float(status.gpu_allocated_mb),
            mot.csv_float(status.gpu_reserved_mb),
            mot.csv_float(status.gpu_peak_allocated_mb),
            mot.csv_float(status.gpu_peak_reserved_mb),
            "1" if self._last_frame_week2_shared_ok else "0",
            "1" if self._last_frame_week2_head_ok else "0",
        ]
        self.week2_proof_lines.append(",".join(fields) + "\n")
        self._add_profile_seconds("proof_output", time.perf_counter() - proof_started)

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int = 1) -> None:
        if self.max_tracks > 0:
            boxes = boxes[: self.max_tracks]
        started = time.perf_counter()
        before = self._week2_counter_snapshot()
        self._reset_last_profile()
        self._last_head_seconds = 0.0
        self._last_head_mode = "none"
        self._last_roi_tokens = 0
        self._last_selected_head_count = 0
        feature_map = self._encode_frame(frame)
        for bbox in boxes:
            self._create_track(frame, feature_map, bbox, frame_number)
        self._record_frame_status(time.perf_counter() - started)
        if self.collect_week2_proof:
            self._append_week2_proof_row(frame_number, "initialize", 0, before)

    def add_tracks(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int) -> List[mot.TrackState]:
        if self.max_tracks > 0:
            remaining = max(0, self.max_tracks - len(self.tracks))
            boxes = boxes[:remaining]
        if not boxes:
            return []
        started = time.perf_counter()
        before = self._week2_counter_snapshot()
        self._reset_last_profile()
        self._last_head_seconds = 0.0
        self._last_head_mode = "none"
        self._last_roi_tokens = 0
        self._last_selected_head_count = 0
        feature_map = self._encode_frame(frame)
        added = [self._create_track(frame, feature_map, bbox, frame_number) for bbox in boxes]
        self._record_frame_status(time.perf_counter() - started)
        if self.collect_week2_proof:
            self._append_week2_proof_row(frame_number, "add_tracks", 0, before)
        return added

    def _create_track(
        self,
        frame: np.ndarray,
        feature_map,
        bbox: BBox,
        frame_number: int,
    ) -> mot.TrackState:
        clipped = mot.clamp_bbox_to_frame_bounds(frame, bbox)
        initial_slot = self._template_slot_for_bbox(
            feature_map,
            clipped,
            frame.shape,
            "initial",
            frame_number,
            1.0,
            self._siamfc_context_bbox(clipped, 2.0),
        )
        head_vector = initial_slot.vector
        track = mot.TrackState(
            track_id=self.next_track_id,
            bbox=clipped,
            previous_bbox=clipped,
            predicted_bbox=clipped,
            raw_bbox=clipped,
            color=mot.color_for_track(self.next_track_id),
            confidence=1.0,
            raw_confidence=1.0,
            confidence_baseline=1.0,
            state="initialized",
            active_template_frame=frame_number,
            assigned_source="v8-initial-selection",
            active_lorat_slot="initial",
            lorat_memory_slot_count=1,
            initial_bbox=clipped,
            trusted_size_bank=[mot.clamp_bbox_size(clipped)],
            appearance_updates=1,
            kalman=mot.BBoxKalmanFilter(clipped),
            last_reliable_bbox=clipped,
            last_reliable_frame=frame_number,
        )
        initial_feature = self.F.normalize(head_vector.detach().clone(), dim=0)
        track.v8_initial_feature = initial_feature
        track.v8_appearance_feature = initial_feature.detach().clone()
        track.v8_feature_bank = [initial_feature.detach().clone()]
        crop_feature = self._dinov2_crop_embeddings_for_bboxes(frame, [clipped])[0] if self.v8_dinov2_crop_reid else None
        if crop_feature is not None:
            track.v8_initial_crop_feature = crop_feature.detach().clone()
            track.v8_appearance_crop_feature = crop_feature.detach().clone()
            track.v8_crop_feature_bank = [crop_feature.detach().clone()]
        setattr(track, "v8_stable_update_streak", 0)
        mot.set_track_lifecycle(track, mot.TrackLifecycle.HEALTHY)
        track.size_history = [(frame_number, mot.clamp_bbox_size(clipped))]
        self._set_track_head_bank(track, [initial_slot])
        mot.record_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        mot.record_reliable_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        self.tracks.append(track)
        self.track_by_id[track.track_id] = track
        self.next_track_id += 1
        return track

    def update(self, frame: np.ndarray, frame_number: int) -> Sequence[mot.TrackState]:
        frame_started = time.perf_counter()
        before = self._week2_counter_snapshot()
        self.last_candidate_diagnostics = []
        self._reset_last_profile()
        self._last_head_seconds = 0.0
        self._last_head_mode = "none"
        self._last_roi_tokens = 0
        self._last_selected_head_count = 0
        feature_map = self._encode_frame(frame)
        evaluated_tracks = self._tracks_for_frame_update()
        tracked_objects_this_frame = len(evaluated_tracks)
        if evaluated_tracks:
            self._score_and_update_tracks(frame, feature_map, evaluated_tracks, frame_number)

        self._record_frame_status(time.perf_counter() - frame_started)
        if self.collect_week2_proof:
            self._append_week2_proof_row(frame_number, "track", tracked_objects_this_frame, before)
        return self.tracks

    def _tracks_for_frame_update(self) -> List[mot.TrackState]:
        """Return visible and lost-but-recoverable tracks for the frame-level V8 pass."""
        tracks: List[mot.TrackState] = []
        for track in self.tracks:
            if track.ok:
                tracks.append(track)
                continue
            if track.lost_frames > 0 and self._get_track_head_bank(track):
                tracks.append(track)
        return tracks

    @staticmethod
    def _bbox_overlap_fraction(left: BBox, right: BBox) -> float:
        """Intersection over the smaller box area, useful for near-containment conflicts."""

        lx, ly, lw, lh = mot.clamp_bbox_size(left)
        rx, ry, rw, rh = mot.clamp_bbox_size(right)
        inter_x1 = max(lx, rx)
        inter_y1 = max(ly, ry)
        inter_x2 = min(lx + lw, rx + rw)
        inter_y2 = min(ly + lh, ry + rh)
        intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        smaller_area = max(1.0, min(lw * lh, rw * rh))
        return float(intersection / smaller_area)

    @staticmethod
    def _bbox_center_ratio(left: BBox, right: BBox) -> float:
        """Center distance normalized by the smaller box diagonal."""

        left_center = np.asarray(mot.bbox_center(left), dtype=np.float32)
        right_center = np.asarray(mot.bbox_center(right), dtype=np.float32)
        distance = float(np.linalg.norm(left_center - right_center))
        reference = max(1.0, min(mot.bbox_diagonal(left), mot.bbox_diagonal(right)))
        return float(distance / reference)

    def _assignment_strength(self, track_id: int, assignment: mot.IdentityAssignment) -> Tuple[float, float, float, float, float, float]:
        confidence = 0.5 if assignment.output.confidence is None else max(0.0, min(1.0, float(assignment.output.confidence)))
        own_source_bonus = 0.04 if int(assignment.output.source_track_id) == int(track_id) else 0.0
        return (
            float(assignment.score.total),
            float(assignment.score.appearance),
            float(assignment.score.initial_anchor),
            float(assignment.score.identity_margin),
            float(assignment.assignment_margin) + own_source_bonus,
            confidence,
        )

    def _assignment_has_strong_ownership(self, track_id: int, assignment: mot.IdentityAssignment) -> bool:
        if int(assignment.output.source_track_id) != int(track_id):
            return False
        return (
            assignment.score.appearance >= max(self.identity_arbitrator.min_reid + 0.08, 0.52)
            and assignment.score.initial_anchor >= max(self.v8_accept_min_initial_anchor + 0.04, 0.54)
            and assignment.score.identity_margin >= self.v8_assignment_conflict_ownership_margin
            and assignment.assignment_margin >= self.v8_assignment_conflict_score_margin
            and assignment.score.motion >= max(self.identity_arbitrator.min_motion, 0.24)
        )

    def _assignment_conflict_reason(
        self,
        track_id: int,
        assignment: mot.IdentityAssignment,
        kept_track_id: int,
        kept_assignment: mot.IdentityAssignment,
    ) -> Optional[str]:
        overlap_iou = mot.bbox_iou(assignment.output.bbox, kept_assignment.output.bbox)
        containment = self._bbox_overlap_fraction(assignment.output.bbox, kept_assignment.output.bbox)
        center_ratio = self._bbox_center_ratio(assignment.output.bbox, kept_assignment.output.bbox)
        same_output_source = assignment.output.source_track_id == kept_assignment.output.source_track_id
        soft_overlap = self.v8_assignment_conflict_iou > 0.0 and overlap_iou >= self.v8_assignment_conflict_iou
        hard_overlap = self.v8_assignment_conflict_hard_iou > 0.0 and overlap_iou >= self.v8_assignment_conflict_hard_iou
        hard_containment = (
            self.v8_assignment_conflict_containment > 0.0
            and containment >= self.v8_assignment_conflict_containment
        )
        near_same_center = (
            self.v8_assignment_conflict_center_ratio > 0.0
            and center_ratio <= self.v8_assignment_conflict_center_ratio
            and (overlap_iou >= 0.08 or containment >= 0.22)
        )
        anchor_owned_by_kept = (
            assignment.score.other_track_id == kept_track_id
            and assignment.score.identity_margin <= -self.v8_assignment_conflict_score_margin
        )
        if not (
            same_output_source
            or soft_overlap
            or hard_overlap
            or hard_containment
            or near_same_center
            or anchor_owned_by_kept
        ):
            return None

        if (
            self._assignment_has_strong_ownership(track_id, assignment)
            and not same_output_source
            and not hard_overlap
            and not anchor_owned_by_kept
        ):
            return None

        kept_strength = self._assignment_strength(kept_track_id, kept_assignment)
        candidate_strength = self._assignment_strength(track_id, assignment)
        total_gap = float(kept_assignment.score.total) - float(assignment.score.total)
        appearance_gap = float(kept_assignment.score.appearance) - float(assignment.score.appearance)
        margin_gap = float(kept_assignment.assignment_margin) - float(assignment.assignment_margin)
        candidate_weak = (
            int(assignment.output.source_track_id) != int(track_id)
            or assignment.score.identity_margin < self.v8_assignment_conflict_ownership_margin
            or assignment.assignment_margin < self.v8_assignment_conflict_score_margin
            or assignment.score.initial_anchor < self.v8_accept_min_initial_anchor
            or assignment.score.appearance < max(self.identity_arbitrator.min_reid, 0.42)
        )
        kept_not_weaker = kept_strength >= candidate_strength
        if same_output_source:
            reason = "SAME_SOURCE"
        elif hard_overlap:
            reason = "HARD_IOU"
        elif soft_overlap:
            reason = "SOFT_IOU"
        elif hard_containment:
            reason = "CONTAINMENT"
        elif anchor_owned_by_kept:
            reason = "ANCHOR_OWNER"
        else:
            reason = "NEAR_CENTER"
        if (
            same_output_source
            or hard_overlap
            or anchor_owned_by_kept
            or (soft_overlap and kept_not_weaker and candidate_weak)
            or (kept_not_weaker and candidate_weak)
            or total_gap >= -self.v8_assignment_conflict_score_margin
            or appearance_gap >= self.v8_assignment_conflict_score_margin
            or margin_gap >= self.v8_assignment_conflict_score_margin
        ):
            return (
                f"ASSIGNMENT_CONFLICT_{reason}_T{kept_track_id}"
                f"_IOU{overlap_iou:.2f}_CTR{center_ratio:.2f}_OWN{assignment.score.identity_margin:.2f}"
        )
        return None

    @staticmethod
    def _alternate_candidate_margin(
        candidate: V8HeadCandidate,
        candidates: Sequence[V8HeadCandidate],
    ) -> float:
        competing = [
            float(other.confidence)
            for other in candidates
            if int(other.rank) != int(candidate.rank)
        ]
        return max(0.0, float(candidate.confidence) - (max(competing) if competing else 0.0))

    def _try_rescue_assignment_with_alternate_candidate(
        self,
        track_id: int,
        kept_assignments: Sequence[Tuple[int, mot.IdentityAssignment]],
        source_record: Dict[str, object],
        tracks: Sequence[mot.TrackState],
        frame: np.ndarray,
        feature_map,
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[mot.IdentityAssignment, Dict[str, object]]]:
        if (
            not self.v8_assignment_alt_rescue
            or self.v8_assignment_alt_rescue_max_candidates <= 0
            or not self.identity_arbitrator.enabled
        ):
            return None

        track = self.track_by_id.get(track_id)
        if track is None:
            return None

        top_candidates = tuple(source_record.get("head_top_candidates") or ())
        if len(top_candidates) <= 1:
            self.v8_assignment_alt_rescue_reject_counts["NO_ALT"] += 1
            return None

        already_used_bbox = source_record.get("candidate")
        tried = 0
        for candidate in top_candidates:
            if tried >= self.v8_assignment_alt_rescue_max_candidates:
                break
            bbox = mot.clamp_bbox_size(candidate.bbox)
            if already_used_bbox is not None and mot.bbox_iou(bbox, already_used_bbox) >= 0.92:
                continue
            confidence = float(candidate.confidence)
            if confidence < max(self.v8_assignment_alt_rescue_min_confidence, self.lorat_accept_min_score * 0.75):
                self.v8_assignment_alt_rescue_reject_counts["ALT_LOWCONF"] += 1
                continue
            tried += 1

            frame_number = int(getattr(source_record.get("slot"), "frame_number", 0) or 0)
            slot = self._synthetic_head_slot(track, frame_number)
            output = mot.LoRATSlotOutput(
                source_track_id=track_id,
                slot=slot,
                bbox=bbox,
                confidence=confidence,
            )
            output = self._with_feature_appearance(output, feature_map, frame_shape)
            if self.v8_dinov2_crop_reid:
                self._attach_dinov2_crop_reid_features(frame, [output])

            score = self.identity_arbitrator.score(track, output, tracks)
            assignment_margin = self._alternate_candidate_margin(candidate, top_candidates)
            assignment = mot.IdentityAssignment(
                track=track,
                output=output,
                score=score,
                assignment_margin=assignment_margin,
            )
            assignment_ok, reject_state = self.identity_arbitrator.assignment_gate(track, output, score)
            if not assignment_ok:
                if self.identity_arbitrator._should_remember_negative_reject(reject_state):
                    self.identity_arbitrator.remember_negative_candidate(track, output)
                self.v8_assignment_alt_rescue_reject_counts[f"GATE_{reject_state}"] += 1
                continue
            candidate_reject = self._candidate_reject_state(
                track,
                bbox,
                confidence,
                assignment,
                f"head_alt_r{candidate.rank}",
            )
            if candidate_reject is not None:
                self.v8_assignment_alt_rescue_reject_counts[f"ACCEPT_{candidate_reject}"] += 1
                continue

            conflict_reason = None
            for kept_track_id, kept_assignment in kept_assignments:
                conflict_reason = self._assignment_conflict_reason(track_id, assignment, kept_track_id, kept_assignment)
                if conflict_reason is not None:
                    break
            if conflict_reason is not None:
                self.v8_assignment_alt_rescue_reject_counts["ALT_CONFLICT"] += 1
                continue

            rescued_record = dict(source_record)
            rescued_record.update(
                {
                    "head_candidate": bbox,
                    "candidate": bbox,
                    "confidence": confidence,
                    "margin": assignment_margin,
                    "candidate_source": f"head_alt_r{candidate.rank}",
                    "output": output,
                }
            )
            return assignment, rescued_record

        if tried <= 0:
            self.v8_assignment_alt_rescue_reject_counts["NO_USABLE_ALT"] += 1
        return None

    def _resolve_assignment_spatial_conflicts(
        self,
        assignment_by_track_id: Dict[int, mot.IdentityAssignment],
        source_record_by_track_id: Dict[int, Dict[str, object]],
        assignment_reject_by_track_id: Dict[int, str],
        tracks: Sequence[mot.TrackState],
        frame: np.ndarray,
        feature_map,
        frame_shape: Tuple[int, ...],
    ) -> None:
        if (
            not self.identity_arbitrator.enabled
            or len(assignment_by_track_id) <= 1
        ):
            return
        if (
            self.v8_assignment_conflict_iou <= 0.0
            and self.v8_assignment_conflict_hard_iou <= 0.0
            and self.v8_assignment_conflict_containment <= 0.0
            and self.v8_assignment_conflict_center_ratio <= 0.0
        ):
            return

        kept_assignments: List[Tuple[int, mot.IdentityAssignment]] = []
        ordered_assignments = sorted(
            assignment_by_track_id.items(),
            key=lambda item: self._assignment_strength(item[0], item[1]),
            reverse=True,
        )
        for track_id, assignment in ordered_assignments:
            conflict_reason: Optional[str] = None
            for kept_track_id, kept_assignment in kept_assignments:
                conflict_reason = self._assignment_conflict_reason(track_id, assignment, kept_track_id, kept_assignment)
                if conflict_reason is not None:
                    break
            if conflict_reason is None:
                kept_assignments.append((track_id, assignment))
                continue
            source_record = source_record_by_track_id.get(track_id)
            rescued = None
            if (
                source_record is not None
                and self.v8_assignment_alt_rescue
                and self.v8_assignment_alt_rescue_max_candidates > 0
            ):
                self.v8_assignment_alt_rescue_attempts += 1
                rescued = self._try_rescue_assignment_with_alternate_candidate(
                    track_id,
                    kept_assignments,
                    source_record,
                    tracks,
                    frame,
                    feature_map,
                    frame_shape,
                )
            if rescued is not None:
                rescued_assignment, rescued_record = rescued
                assignment_by_track_id[track_id] = rescued_assignment
                source_record_by_track_id[track_id] = rescued_record
                kept_assignments.append((track_id, rescued_assignment))
                self.v8_assignment_alt_rescue_hits += 1
                continue
            track = self.track_by_id.get(track_id)
            if track is not None:
                self.identity_arbitrator.remember_negative_candidate(track, assignment.output)
            assignment_by_track_id.pop(track_id, None)
            source_record_by_track_id.pop(track_id, None)
            assignment_reject_by_track_id[track_id] = conflict_reason
            self.v8_assignment_conflict_reason_counts[conflict_reason.split("_T", 1)[0]] += 1

    def _score_and_update_tracks(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        frame_number: int,
    ) -> None:
        selected_banks = [self._select_track_heads(track, frame_number) for track in tracks]
        head_output = self.object_conditioned_head.score(feature_map, selected_banks)
        score_maps = head_output.score_maps
        box_delta_maps = head_output.box_delta_maps
        self._last_head_seconds = head_output.elapsed_seconds
        self._last_head_mode = self.object_conditioned_head.last_mode
        self._last_selected_head_count = head_output.selected_head_count
        self.runtime_status.object_head_batches += 1
        self.runtime_status.object_head_items += len(tracks)
        self.runtime_status.max_object_head_batch = max(self.runtime_status.max_object_head_batch, len(tracks))
        self.runtime_status.fusion_forward_calls += 1
        self.runtime_status.fusion_forward_items += len(tracks)
        self.runtime_status.max_fusion_forward_batch = max(self.runtime_status.max_fusion_forward_batch, len(tracks))

        predicted_bboxes = [self._predict_track(track) for track in tracks]
        candidate_infos = self._candidates_from_head_output(
            score_maps,
            box_delta_maps,
            predicted_bboxes,
            tracks,
            frame.shape,
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
            head_candidate = candidate_info.bbox
            head_confidence = candidate_info.confidence
            head_margin = candidate_info.margin
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
            total_roi_tokens += roi_tokens
            slot = self._synthetic_head_slot(track, frame_number)
            output = mot.LoRATSlotOutput(
                source_track_id=track.track_id,
                slot=slot,
                bbox=candidate,
                confidence=confidence,
            )
            if self.identity_arbitrator.enabled:
                appearance_started = time.perf_counter()
                output = self._with_feature_appearance(output, feature_map, frame.shape)
                self._add_profile_seconds("reid_appearance", time.perf_counter() - appearance_started)
            records[track.track_id] = {
                "track": track,
                "predicted": predicted,
                "head_candidate": head_candidate,
                "head_confidence": head_confidence,
                "head_margin": head_margin,
                "head_top_candidates": candidate_info.top_candidates,
                "template_candidate": template_candidate,
                "candidate": candidate,
                "confidence": confidence,
                "margin": margin,
                "candidate_source": candidate_source,
                "template_confidence": template_confidence,
                "slot": slot,
                "output": output,
            }
            diagnostics_by_track_id[track.track_id] = {
                "frame": frame_number,
                "track_id": track.track_id,
                "previous_bbox": track.previous_bbox,
                "predicted_bbox": predicted,
                "head_bbox": head_candidate,
                "head_confidence": head_confidence,
                "head_margin": head_margin,
                "head_roi_tokens": roi_tokens,
                "head_top_candidates": candidate_info.top_candidates,
                "template_attempted": template_attempted,
                "template_bbox": template_candidate,
                "template_confidence": template_confidence,
                "fused_bbox": candidate,
                "fused_confidence": confidence,
                "fused_margin": margin,
                "candidate_source": candidate_source,
            }
            candidate_outputs.append(output)

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
            if source_record is None or identity_assignment is None:
                reject_state = assignment_reject_by_track_id.get(track.track_id, "NO_ASSIGNMENT")
                hold_started = time.perf_counter()
                self._hold_track(track, hold_predicted, hold_confidence, hold_margin, frame_number, frame, reject_state)
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
                str(source_record.get("candidate_source", "head")),
            )
            self._add_profile_seconds("accept", time.perf_counter() - accept_started)
            if not accepted:
                reject_state = str(getattr(track, "state", ""))
                hold_started = time.perf_counter()
                self._hold_track(track, hold_predicted, hold_confidence, hold_margin, frame_number, frame, reject_state)
                self._add_profile_seconds("hold", time.perf_counter() - hold_started)
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "accepted": bool(accepted),
                        "held": not bool(accepted),
                        "reject_state": "" if accepted else reject_state,
                        "final_bbox": track.bbox,
                        "final_confidence": track.confidence,
                        "final_state": track.state,
                        "assigned_source": track.assigned_source,
                    }
                )

        self._last_roi_tokens = total_roi_tokens
        self.runtime_status.object_head_roi_tokens += total_roi_tokens
        self.last_candidate_diagnostics = list(diagnostics_by_track_id.values())

    def _select_track_heads(self, track: mot.TrackState, frame_number: int) -> List[object]:
        head_bank = self._get_track_head_bank(track)
        if not head_bank:
            self._record_v8_gating([], [])
            return []

        reasons = self._v8_recovery_reasons(track, head_bank, frame_number)
        if reasons:
            limit = min(len(head_bank), self.head_rank, self.v8_recovery_heads_per_track)
            selected = self._select_recovery_heads(track, head_bank, frame_number, limit)
        else:
            limit = min(len(head_bank), self.head_rank, self.v8_primary_heads_per_track)
            selected = self._select_primary_heads(track, head_bank, limit)
        self._record_v8_gating(selected, reasons)
        return selected

    def _select_primary_heads(self, track: mot.TrackState, head_bank: Sequence[object], limit: int) -> List[object]:
        if limit <= 0:
            return []
        selected: List[object] = []

        def add(vector: Optional[object]) -> None:
            if vector is None or len(selected) >= limit:
                return
            if any(existing is vector for existing in selected):
                return
            selected.append(vector)

        add(head_bank[0])
        add(head_bank[-1])
        for vector in reversed(head_bank[1:]):
            add(vector)
        return selected

    def _select_recovery_heads(
        self,
        track: mot.TrackState,
        head_bank: Sequence[object],
        frame_number: int,
        limit: int,
    ) -> List[object]:
        if limit <= 0:
            return []
        selected: List[object] = []

        def add(vector: Optional[object]) -> None:
            if vector is None or len(selected) >= limit:
                return
            if any(existing is vector for existing in selected):
                return
            selected.append(vector)

        add(head_bank[0])
        add(head_bank[-1])
        recents = list(head_bank[1:])
        if recents:
            start_frame = track.trajectory[0][0] if track.trajectory else frame_number
            rotating_index = max(0, frame_number - start_frame - 1) % len(recents)
            for offset in range(len(recents)):
                add(recents[(rotating_index + offset) % len(recents)])
            for vector in reversed(recents):
                add(vector)
        return selected

    def _v8_recovery_reasons(
        self,
        track: mot.TrackState,
        head_bank: Sequence[object],
        frame_number: int,
    ) -> List[str]:
        if len(head_bank) <= self.v8_primary_heads_per_track:
            return []

        reasons: List[str] = []
        state = str(track.state or "").upper()
        for token in ("MISS", "LOWCONF", "ID_UNCERTAIN", "OCCLU", "LOST", "NOLEARN", "SHRINK", "REIDRECOVERY"):
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
        if track.confidence is not None and track.confidence < self.v8_recovery_min_confidence:
            reasons.append("LOW_CONFIDENCE")
        if track.assignment_score is not None and track.assignment_score < self.v8_recovery_min_assignment_score:
            reasons.append("LOW_ASSIGNMENT_SCORE")
        if track.assignment_margin is not None and track.assignment_margin < self.v8_recovery_min_assignment_margin:
            reasons.append("LOW_ASSIGNMENT_MARGIN")
        last_update = int(getattr(track, "v8_last_head_update_frame", track.active_template_frame or frame_number))
        if self.v8_recovery_stale_head_frames > 0 and frame_number - last_update >= self.v8_recovery_stale_head_frames:
            reasons.append("STALE_ACTIVE_HEAD")
        if self.v8_recovery_interval > 0 and track.trajectory:
            start_frame = track.trajectory[0][0]
            if frame_number > start_frame and (frame_number - start_frame) % self.v8_recovery_interval == 0:
                reasons.append("PERIODIC_ANCHOR_CHECK")
        return reasons

    def _record_v8_gating(self, selected: Sequence[object], reasons: Sequence[str]) -> None:
        self.v8_gating_decisions += 1
        self.v8_selected_head_items += len(selected)
        if reasons:
            self.v8_recovery_decisions += 1
            self.v8_recovery_reason_counts.update(reasons)
        else:
            self.v8_primary_decisions += 1

    def _synthetic_head_slot(self, track: mot.TrackState, frame_number: int) -> mot.LoRATMemorySlot:
        return mot.LoRATMemorySlot(
            task_id=(track.track_id * 1_000_000) + frame_number,
            track_id=track.track_id,
            label=str(track.active_lorat_slot or "V8-head"),
            frame_number=frame_number,
            bbox=track.bbox,
            confidence=track.confidence,
            raw_confidence=track.raw_confidence,
            confidence_baseline=track.confidence_baseline,
            last_refresh_frame=int(getattr(track, "v8_last_head_update_frame", track.active_template_frame or frame_number)),
            active=True,
            anchor_frame_number=int(track.active_template_frame or frame_number),
            anchor_bbox=track.initial_bbox,
        )

    def _append_head_debug_rows(
        self,
        frame_number: int,
        evaluated_tracks: Sequence[mot.TrackState],
        candidate_outputs: Sequence[mot.LoRATSlotOutput],
    ) -> None:
        if not evaluated_tracks or not candidate_outputs:
            return
        score_matrices = self.identity_arbitrator._identity_score_matrices(evaluated_tracks, candidate_outputs)
        track_row_by_id = {track.track_id: index for index, track in enumerate(evaluated_tracks)}
        for output_index, output in enumerate(candidate_outputs):
            track = self.track_by_id.get(output.source_track_id)
            if track is None:
                continue
            track_row = track_row_by_id.get(track.track_id)
            if track_row is None:
                continue
            score = self.identity_arbitrator._identity_score_from_matrices(score_matrices, track_row, output_index)
            fields = [
                str(frame_number),
                str(track.track_id),
                mot.csv_text(output.slot.label),
                str(output.slot.task_id),
                str(output.slot.frame_number),
                str(output.slot.anchor_frame_number),
                str(output.slot.last_refresh_frame),
                "1" if output.slot.active else "0",
                mot.csv_text(track.active_lorat_slot),
                mot.csv_float(output.confidence),
                mot.csv_float(output.confidence),
                mot.csv_float(output.slot.confidence_baseline),
                mot.csv_float(track.confidence_baseline),
                *mot.csv_bbox(output.bbox),
                *mot.csv_bbox(track.bbox),
                *mot.csv_bbox(track.predicted_bbox),
                mot.csv_float(score.total),
                mot.csv_float(score.appearance),
                mot.csv_float(score.motion),
                mot.csv_float(score.path),
                mot.csv_float(score.source),
                mot.csv_float(score.confidence),
                mot.csv_float(score.iou),
                mot.csv_float(score.initial_anchor),
                mot.csv_float(score.other_anchor),
                str(score.other_track_id) if score.other_track_id is not None else "",
                mot.csv_float(score.identity_margin),
                str(score.occlusion_track_id) if score.occlusion_track_id is not None else "",
                mot.csv_float(score.occlusion_iou),
            ]
            self.slot_debug_lines.append(",".join(fields) + "\n")

    def _predict_track(self, track: mot.TrackState) -> BBox:
        track.previous_bbox = track.bbox
        if track.kalman is not None:
            predicted = track.kalman.predict()
        else:
            predicted = mot.predict_bbox(track.bbox, track.velocity)
        track.predicted_bbox = predicted
        return predicted

    @staticmethod
    def _head_decode_reference_size(track: mot.TrackState) -> Tuple[float, float]:
        samples: List[BBox] = []
        if track.trusted_size_bank:
            samples.extend(track.trusted_size_bank[-mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:])
        if track.last_reliable_bbox is not None:
            samples.append(track.last_reliable_bbox)
        if track.initial_bbox is not None:
            samples.append(track.initial_bbox)
        if not samples:
            samples.append(track.bbox)
        widths = np.asarray([max(1.0, float(sample[2])) for sample in samples], dtype=np.float32)
        heights = np.asarray([max(1.0, float(sample[3])) for sample in samples], dtype=np.float32)
        return max(1.0, float(np.median(widths))), max(1.0, float(np.median(heights)))

    def _is_small_target_track(self, track: mot.TrackState) -> bool:
        if not self.v8_small_target_mode:
            return False
        reference = track.initial_bbox or track.last_reliable_bbox or track.bbox
        _, _, width, height = mot.clamp_bbox_size(reference)
        return (width * height) <= self.v8_small_target_area or max(width, height) <= self.v8_small_target_max_side

    def _small_target_reference_size(self, track: mot.TrackState) -> Tuple[float, float]:
        if track.initial_bbox is not None:
            _, _, width, height = mot.clamp_bbox_size(track.initial_bbox)
            return max(1.0, width), max(1.0, height)
        return self._head_decode_reference_size(track)

    def _template_match_reference_size(self, track: mot.TrackState) -> Tuple[float, float]:
        if self._is_small_target_track(track):
            return self._small_target_reference_size(track)
        return self._head_decode_reference_size(track)

    def _small_target_scale_error(self, track: mot.TrackState, bbox: BBox) -> float:
        if not self._is_small_target_track(track):
            return 1.0
        _, _, width, height = mot.clamp_bbox_size(bbox)
        reference_w, reference_h = self._small_target_reference_size(track)
        return max(
            width / max(1.0, reference_w),
            reference_w / max(1.0, width),
            height / max(1.0, reference_h),
            reference_h / max(1.0, height),
        )

    def _apply_small_target_scale_lock(self, track: mot.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, bool]:
        if not self._is_small_target_track(track):
            return mot.clamp_bbox_to_frame_bounds(frame, bbox), False
        x, y, width, height = mot.clamp_bbox_size(bbox)
        reference_w, reference_h = self._small_target_reference_size(track)
        center_x, center_y = mot.bbox_center((x, y, width, height))
        locked = mot.clamp_bbox_to_frame_bounds(
            frame,
            (
                center_x - (reference_w / 2.0),
                center_y - (reference_h / 2.0),
                reference_w,
                reference_h,
            ),
        )
        changed = abs(locked[2] - width) > 0.01 or abs(locked[3] - height) > 0.01
        return locked, changed

    def _candidates_from_head_output(
        self,
        score_maps,
        box_delta_maps,
        predicted_bboxes: Sequence[BBox],
        tracks: Sequence[mot.TrackState],
        frame_shape: Tuple[int, ...],
    ) -> List[V8HeadCandidateInfo]:
        candidate_started = time.perf_counter()
        transfer_seconds = 0.0
        object_count = int(score_maps.shape[0])
        if object_count <= 0:
            return []

        grid_height = int(score_maps.shape[1])
        grid_width = int(score_maps.shape[2])
        score_maps_float = score_maps.to(self.torch.float32)
        masked_scores = self.torch.full_like(score_maps_float, -float("inf"))
        selection_scores = None
        if self.v8_window_penalty_ratio > 0:
            selection_scores = self.torch.full_like(score_maps_float, -float("inf"))
        roi_token_counts: List[int] = []
        for index, (track, predicted) in enumerate(zip(tracks, predicted_bboxes)):
            roi = self._expanded_search_bbox(predicted, track.bbox, self._head_decode_reference_size(track))
            y_slice, x_slice = self._bbox_to_grid_slices(roi, frame_shape)
            roi_token_count = max(0, int(y_slice.stop - y_slice.start) * int(x_slice.stop - x_slice.start))
            roi_token_counts.append(roi_token_count)
            if roi_token_count > 0:
                masked_scores[index, y_slice, x_slice] = score_maps_float[index, y_slice, x_slice]
                if selection_scores is not None:
                    roi_h = int(y_slice.stop - y_slice.start)
                    roi_w = int(x_slice.stop - x_slice.start)
                    if roi_h <= 1 or roi_w <= 1:
                        window = self.torch.ones((roi_h, roi_w), device=score_maps.device, dtype=self.torch.float32)
                    else:
                        window = self.torch.outer(
                            self.torch.hann_window(roi_h, periodic=False, device=score_maps.device, dtype=self.torch.float32),
                            self.torch.hann_window(roi_w, periodic=False, device=score_maps.device, dtype=self.torch.float32),
                        )
                    roi_probs = self.torch.sigmoid(score_maps_float[index, y_slice, x_slice])
                    selection_scores[index, y_slice, x_slice] = (
                        roi_probs * (1.0 - self.v8_window_penalty_ratio)
                        + window * self.v8_window_penalty_ratio
                    )

        flat_scores = masked_scores.reshape(object_count, -1)
        flat_selection_scores = selection_scores.reshape(object_count, -1) if selection_scores is not None else flat_scores
        best_selection_values, best_indices = self.torch.max(flat_selection_scores, dim=1)
        finite_best = self.torch.isfinite(best_selection_values)
        safe_best_indices = self.torch.where(
            finite_best,
            best_indices,
            self.torch.zeros_like(best_indices),
        )
        raw_best_values = self.torch.gather(flat_scores, 1, safe_best_indices[:, None]).squeeze(1)
        safe_best_values = self.torch.where(
            finite_best,
            raw_best_values,
            self.torch.full_like(best_selection_values, -30.0),
        )

        if flat_selection_scores.shape[1] >= 2:
            top2 = self.torch.topk(flat_selection_scores, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
        else:
            margins = self.torch.zeros_like(safe_best_values)
        roi_counts_tensor = self.torch.tensor(
            roi_token_counts,
            device=score_maps.device,
            dtype=self.torch.long,
        )
        margins = self.torch.where(
            (roi_counts_tensor >= 2) & self.torch.isfinite(margins),
            self.torch.clamp(margins, min=0.0),
            self.torch.zeros_like(margins),
        )

        frame_height, frame_width = frame_shape[:2]
        frame_width_float = float(frame_width)
        frame_height_float = float(frame_height)
        topk_count = min(5, int(flat_selection_scores.shape[1]))
        top_packed = []
        if topk_count > 0:
            top_selection_values, top_indices = self.torch.topk(flat_selection_scores, k=topk_count, dim=1)
            finite_top = self.torch.isfinite(top_selection_values)
            safe_top_indices = self.torch.where(
                finite_top,
                top_indices,
                self.torch.zeros_like(top_indices),
            )
            top_grid_y = self.torch.div(safe_top_indices, grid_width, rounding_mode="floor")
            top_grid_x = self.torch.remainder(safe_top_indices, grid_width)
            top_row_indices = self.torch.arange(object_count, device=box_delta_maps.device)[:, None].expand(-1, topk_count)
            top_box_raw = (
                box_delta_maps[
                    top_row_indices,
                    top_grid_y.to(box_delta_maps.device),
                    top_grid_x.to(box_delta_maps.device),
                ]
            ).to(self.torch.float32)
            raw_top_values = self.torch.gather(flat_scores, 1, safe_top_indices)
            safe_top_values = self.torch.where(
                finite_top,
                raw_top_values,
                self.torch.full_like(top_selection_values, -30.0),
            )
            top_confidences = self.torch.sigmoid(safe_top_values)
            top_ref_x = (top_grid_x.to(self.torch.float32) + 0.5) / float(max(1, grid_width))
            top_ref_y = (top_grid_y.to(self.torch.float32) + 0.5) / float(max(1, grid_height))
            top_ltrb = self.torch.sigmoid(self.torch.clamp(top_box_raw, -30.0, 30.0))
            top_x1 = (top_ref_x - top_ltrb[:, :, 0]) * frame_width_float
            top_y1 = (top_ref_y - top_ltrb[:, :, 1]) * frame_height_float
            top_x2 = (top_ref_x + top_ltrb[:, :, 2]) * frame_width_float
            top_y2 = (top_ref_y + top_ltrb[:, :, 3]) * frame_height_float
            top_decoded = self.torch.stack(
                (
                    top_x1,
                    top_y1,
                    self.torch.clamp(top_x2 - top_x1, min=1.0),
                    self.torch.clamp(top_y2 - top_y1, min=1.0),
                ),
                dim=2,
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
                    finite_top.to(self.torch.float32),
                ),
                dim=2,
            )
        else:
            top_packed_tensor = None

        grid_y = self.torch.div(safe_best_indices, grid_width, rounding_mode="floor")
        grid_x = self.torch.remainder(safe_best_indices, grid_width)
        row_indices = self.torch.arange(object_count, device=box_delta_maps.device)
        box_raw = box_delta_maps[row_indices, grid_y.to(box_delta_maps.device), grid_x.to(box_delta_maps.device)].to(
            self.torch.float32
        )
        confidences = self.torch.sigmoid(safe_best_values)
        ref_x = (grid_x.to(self.torch.float32) + 0.5) / float(max(1, grid_width))
        ref_y = (grid_y.to(self.torch.float32) + 0.5) / float(max(1, grid_height))
        ltrb = self.torch.sigmoid(self.torch.clamp(box_raw, -30.0, 30.0))
        x1 = (ref_x - ltrb[:, 0]) * frame_width_float
        y1 = (ref_y - ltrb[:, 1]) * frame_height_float
        x2 = (ref_x + ltrb[:, 2]) * frame_width_float
        y2 = (ref_y + ltrb[:, 3]) * frame_height_float
        decoded = self.torch.stack(
            (
                x1,
                y1,
                self.torch.clamp(x2 - x1, min=1.0),
                self.torch.clamp(y2 - y1, min=1.0),
            ),
            dim=1,
        )
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
            ),
            dim=1,
        )
        transfer_started = time.perf_counter()
        packed = packed_tensor.detach().cpu().tolist()
        if top_packed_tensor is not None:
            top_packed = top_packed_tensor.detach().cpu().tolist()
        transfer_seconds = time.perf_counter() - transfer_started

        candidates: List[V8HeadCandidateInfo] = []

        for row_index, (row, track, predicted, roi_tokens) in enumerate(zip(packed, tracks, predicted_bboxes, roi_token_counts)):
            if roi_tokens <= 0:
                candidates.append(
                    V8HeadCandidateInfo(
                        bbox=mot.clamp_bbox_size(predicted),
                        confidence=0.0,
                        margin=0.0,
                        roi_tokens=0,
                        top_candidates=(),
                    )
                )
                continue
            confidence, margin, grid_x_value, grid_y_value, bbox_x, bbox_y, bbox_w, bbox_h = row
            top_candidates: List[V8HeadCandidate] = []
            for rank, top_row in enumerate(top_packed[row_index] if top_packed else [], start=1):
                (
                    top_confidence,
                    top_grid_x_value,
                    top_grid_y_value,
                    top_x,
                    top_y,
                    top_w,
                    top_h,
                    top_finite,
                ) = top_row
                if float(top_finite) <= 0.0:
                    continue
                top_bbox = mot.clamp_bbox_size((top_x, top_y, top_w, top_h))
                top_candidates.append(
                    V8HeadCandidate(
                        rank=rank,
                        bbox=top_bbox,
                        confidence=float(top_confidence),
                        grid_x=int(np.clip(round(float(top_grid_x_value)), 0, max(0, grid_width - 1))),
                        grid_y=int(np.clip(round(float(top_grid_y_value)), 0, max(0, grid_height - 1))),
                    )
                )
            candidate = mot.clamp_bbox_size((bbox_x, bbox_y, bbox_w, bbox_h))
            candidates.append(
                V8HeadCandidateInfo(
                    bbox=candidate,
                    confidence=float(confidence),
                    margin=max(0.0, float(margin)),
                    roi_tokens=roi_tokens,
                    top_candidates=tuple(top_candidates),
                )
            )
        self._add_profile_seconds("candidate_transfer", transfer_seconds)
        self._add_profile_seconds("candidate_extract", max(0.0, time.perf_counter() - candidate_started - transfer_seconds))
        return candidates

    def _rerank_head_candidate(
        self,
        feature_map,
        frame_shape: Tuple[int, ...],
        track: mot.TrackState,
        predicted: BBox,
        candidate_info: V8HeadCandidateInfo,
    ) -> V8HeadCandidateInfo:
        top_candidates = tuple(candidate_info.top_candidates or ())
        if len(top_candidates) <= 1:
            return candidate_info

        reference_diagonal = max(1.0, mot.bbox_diagonal(track.bbox))
        best_confidence = max(float(candidate.confidence) for candidate in top_candidates)
        kept_candidates: List[V8HeadCandidate] = []
        feature_vectors = []
        for candidate in top_candidates:
            confidence = float(candidate.confidence)
            if confidence < best_confidence - 0.12:
                continue
            kept_candidates.append(candidate)
            if self.identity_arbitrator is not None and self.identity_arbitrator.enabled:
                feature_vectors.append(self._feature_mean_for_bbox(feature_map, candidate.bbox, frame_shape))
        appearance_scores = np.full((len(kept_candidates),), 0.50, dtype=np.float32)
        if feature_vectors and self.identity_arbitrator is not None and self.identity_arbitrator.enabled:
            appearance_scores = self.identity_arbitrator.track_feature_similarity_many(
                track,
                self.torch.stack(feature_vectors, dim=0),
            )

        scored: List[Tuple[float, V8HeadCandidate, float, float, float, float]] = []
        for candidate, appearance in zip(kept_candidates, appearance_scores.tolist()):
            confidence = float(candidate.confidence)
            bbox = candidate.bbox
            motion = mot.motion_affinity(predicted, bbox, reference_diagonal)
            path = mot.center_path_affinity(track, bbox)
            continuity = max(mot.bbox_iou(track.bbox, bbox), mot.bbox_iou(predicted, bbox))
            score = (
                (0.42 * confidence)
                + (0.22 * motion)
                + (0.16 * path)
                + (0.12 * appearance)
                + (0.08 * continuity)
                - (0.006 * max(0, int(candidate.rank) - 1))
            )
            scored.append((score, candidate, motion, path, appearance, continuity))

        if not scored:
            return candidate_info
        scored.sort(key=lambda row: row[0], reverse=True)
        selected = scored[0][1]
        if selected.bbox == candidate_info.bbox:
            return candidate_info
        rerank_margin = candidate_info.margin
        if len(scored) > 1:
            rerank_margin = max(0.0, min(candidate_info.margin, scored[0][0] - scored[1][0]))
        return V8HeadCandidateInfo(
            bbox=selected.bbox,
            confidence=float(selected.confidence),
            margin=float(rerank_margin),
            roi_tokens=candidate_info.roi_tokens,
            top_candidates=top_candidates,
        )

    def _apply_identity_scores(
        self,
        track: mot.TrackState,
        identity_assignment: Optional[mot.IdentityAssignment],
        fallback_confidence: float,
        fallback_margin: float,
    ) -> None:
        if identity_assignment is None:
            track.assignment_score = fallback_confidence
            track.assignment_margin = fallback_margin
            track.reid_score = fallback_confidence
            track.source_score = fallback_confidence
            return
        score = identity_assignment.score
        track.assignment_score = score.total
        track.assignment_margin = identity_assignment.assignment_margin
        track.reid_score = score.appearance
        track.motion_score = score.motion
        track.path_score = score.path
        track.source_score = score.source
        track.initial_anchor_score = score.initial_anchor
        track.other_anchor_score = score.other_anchor
        track.other_anchor_track_id = score.other_track_id
        track.identity_margin = score.identity_margin
        track.negative_anchor_score = score.negative_anchor
        track.occlusion_track_id = score.occlusion_track_id
        track.occlusion_iou = score.occlusion_iou

    def _candidate_reject_state(
        self,
        track: mot.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
        candidate_source: str = "head",
    ) -> Optional[str]:
        if candidate_source == "template" and track.lost_frames <= 0:
            return "TEMPLATE_HOLD"
        confidence_floor = max(self.min_confidence, self.lorat_accept_min_score)
        if confidence < confidence_floor:
            if (
                self._is_small_target_track(track)
                and "template" in str(candidate_source)
                and confidence >= self.v8_small_target_confidence_floor
                and identity_assignment is not None
            ):
                score = identity_assignment.score
                if (
                    score.initial_anchor >= max(0.70, self.v8_accept_min_initial_anchor)
                    and score.appearance >= max(0.55, self.identity_arbitrator.min_reid if self.identity_arbitrator else 0.0)
                    and score.motion >= self.v8_small_target_template_min_motion
                ):
                    return None
            if self._is_reid_recovery(track, confidence, identity_assignment):
                return None
            return "LOWCONF"
        if identity_assignment is None:
            return "ID_UNCERTAIN" if self.identity_arbitrator.enabled else None

        score = identity_assignment.score
        is_view_change = self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            score,
        )
        if self._is_initial_anchor_steal(score):
            return "OTHERID"
        anchor_reject = self._anchor_identity_reject_state(track, identity_assignment, memory_gate=False)
        if anchor_reject is not None:
            return anchor_reject
        if (
            mot.path_gate_ready(track)
            and score.path < self.identity_arbitrator.min_path
            and not is_view_change
            and not self._is_path_recovery(track, confidence, identity_assignment)
        ):
            return "PATHLOW"
        if score.total < self.identity_arbitrator.min_score and not is_view_change:
            return "ID_UNCERTAIN"
        if self.identity_arbitrator.track_has_feature_appearance(track) and score.appearance < self.identity_arbitrator.min_reid and not is_view_change:
            return "REIDLOW"
        if score.motion < self.identity_arbitrator.min_motion and not is_view_change:
            return "MOTIONLOW"
        if track.lost_frames > 0:
            reacquire_confidence = min(0.95, max(self.lorat_accept_min_score + 0.10, 0.40))
            if confidence < reacquire_confidence and score.appearance < max(0.55, self.identity_arbitrator.min_reid):
                return "REACQUIRE_LOWCONF"
        return None

    @staticmethod
    def _is_initial_anchor_steal(score: mot.IdentityScore) -> bool:
        return (
            score.other_track_id is not None
            and score.occlusion_iou >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and score.other_anchor >= mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and score.identity_margin <= -mot.DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )

    def _anchor_identity_reject_state(
        self,
        track: mot.TrackState,
        identity_assignment: Optional[mot.IdentityAssignment],
        *,
        memory_gate: bool,
    ) -> Optional[str]:
        if identity_assignment is None or getattr(track, "v8_initial_feature", None) is None:
            return None
        score = identity_assignment.score
        is_view_change = self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            score,
        )
        if is_view_change:
            return None

        if memory_gate:
            if score.initial_anchor < self.v8_memory_min_initial_anchor:
                return "MEMANCHORLOW"
            if (
                score.other_track_id is not None
                and score.other_anchor >= 0.55
                and score.identity_margin < self.v8_memory_min_identity_margin
            ):
                return "MEMANCHORAMBIG"
            return None

        if score.initial_anchor < self.v8_accept_min_initial_anchor and score.appearance < max(
            self.identity_arbitrator.min_reid,
            self.v8_accept_min_initial_anchor,
        ):
            return "ANCHORLOW"
        if (
            score.other_track_id is not None
            and score.other_anchor >= max(0.60, score.initial_anchor + 0.04)
            and score.identity_margin < self.v8_accept_min_identity_margin
        ):
            return "ANCHORAMBIG"
        return None

    def _is_path_recovery(
        self,
        track: mot.TrackState,
        confidence_value: float,
        identity_assignment: Optional[mot.IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames < mot.DEFAULT_PATH_RECOVERY_AFTER_FRAMES:
            return False
        score = identity_assignment.score
        return (
            identity_assignment.output.source_track_id == track.track_id
            and confidence_value >= mot.DEFAULT_PATH_RECOVERY_MIN_CONFIDENCE
            and score.appearance >= mot.DEFAULT_PATH_RECOVERY_MIN_REID
            and score.motion >= mot.DEFAULT_PATH_RECOVERY_MIN_MOTION
            and score.total >= self.identity_arbitrator.min_score
        )

    def _is_reid_recovery(
        self,
        track: mot.TrackState,
        confidence_value: float,
        identity_assignment: Optional[mot.IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames <= 0:
            return False
        score = identity_assignment.score
        anchor_reject = self._anchor_identity_reject_state(track, identity_assignment, memory_gate=False)
        return (
            confidence_value >= self.reid_recovery_min_confidence
            and score.total >= self.reid_recovery_min_score
            and score.appearance >= self.reid_recovery_min_reid
            and anchor_reject is None
            and score.motion >= self.reid_recovery_min_motion
            and (score.path >= self.identity_arbitrator.min_path or self._is_path_recovery(track, confidence_value, identity_assignment))
        )

    def _learning_evidence_is_strong(
        self,
        track: mot.TrackState,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
    ) -> bool:
        if confidence < self.shrink_guard_min_confidence:
            return False
        if identity_assignment is None or not self.identity_arbitrator.track_has_feature_appearance(track):
            return True
        return identity_assignment.score.appearance >= self.shrink_guard_min_reid

    def _assess_learning_hold(
        self,
        track: mot.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[mot.IdentityAssignment],
        frame: Optional[np.ndarray],
    ) -> Tuple[bool, List[str], mot.CropInformation, float, float, int]:
        crop_info = mot.measure_crop_information(frame, bbox, self.crop_information_min_pixels)
        previous_area = mot.bbox_area(track.bbox)
        current_area = mot.bbox_area(bbox)
        step_ratio = current_area / max(1.0, previous_area)
        recent_history = track.size_history[-self.shrink_guard_window :] if self.shrink_guard_window > 0 else []
        reference_area = max([previous_area] + [mot.bbox_area(sample_bbox) for _, sample_bbox in recent_history])
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

    def _trusted_size_floor(self, track: mot.TrackState) -> Optional[Tuple[float, float]]:
        if self.lorat_trusted_size_floor_scale <= 0:
            return None
        initial_floor: Optional[Tuple[float, float]] = None
        if track.initial_bbox is not None:
            _, _, initial_w, initial_h = mot.clamp_bbox_size(track.initial_bbox)
            initial_floor = (
                max(1.0, initial_w * self.lorat_trusted_size_floor_scale),
                max(1.0, initial_h * self.lorat_trusted_size_floor_scale),
            )
        samples = list(track.trusted_size_bank[-mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:])
        if not samples:
            return initial_floor
        widths = np.asarray([max(1.0, float(sample[2])) for sample in samples], dtype=np.float32)
        heights = np.asarray([max(1.0, float(sample[3])) for sample in samples], dtype=np.float32)
        memory_floor = (
            max(1.0, float(np.median(widths)) * self.lorat_trusted_size_floor_scale),
            max(1.0, float(np.median(heights)) * self.lorat_trusted_size_floor_scale),
        )
        if initial_floor is None:
            return memory_floor
        return max(initial_floor[0], memory_floor[0]), max(initial_floor[1], memory_floor[1])

    def _apply_trusted_size_floor(self, track: mot.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, bool]:
        x, y, w, h = mot.clamp_bbox_size(bbox)
        floor = self._trusted_size_floor(track)
        if floor is None:
            return mot.clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False
        min_w, min_h = floor
        if w >= min_w and h >= min_h:
            return mot.clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False
        center_x, center_y = mot.bbox_center((x, y, w, h))
        guarded_w = max(w, min_w)
        guarded_h = max(h, min_h)
        return (
            mot.clamp_bbox_to_frame_bounds(
                frame,
                (center_x - (guarded_w / 2.0), center_y - (guarded_h / 2.0), guarded_w, guarded_h),
            ),
            True,
        )

    def _apply_fixed_box_size(self, track: mot.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, bool]:
        if not self.lorat_fixed_box_size or track.initial_bbox is None:
            return mot.clamp_bbox_to_frame_bounds(frame, bbox), False
        _, _, fixed_w, fixed_h = mot.clamp_bbox_size(track.initial_bbox)
        x, y, w, h = mot.clamp_bbox_size(bbox)
        center_x, center_y = mot.bbox_center((x, y, w, h))
        fixed = mot.clamp_bbox_to_frame_bounds(
            frame,
            (center_x - (fixed_w / 2.0), center_y - (fixed_h / 2.0), fixed_w, fixed_h),
        )
        changed = abs(fixed[2] - w) > 0.01 or abs(fixed[3] - h) > 0.01
        return fixed, changed

    @staticmethod
    def _scale_bbox_to_area(bbox: BBox, target_area: float, frame: Optional[np.ndarray]) -> BBox:
        x, y, w, h = mot.clamp_bbox_size(bbox)
        scale = float(np.sqrt(max(1.0, target_area) / max(1.0, w * h)))
        center_x, center_y = mot.bbox_center((x, y, w, h))
        scaled_w = max(1.0, w * scale)
        scaled_h = max(1.0, h * scale)
        return mot.clamp_bbox_to_frame_bounds(
            frame,
            (center_x - (scaled_w / 2.0), center_y - (scaled_h / 2.0), scaled_w, scaled_h),
        )

    def _apply_scale_limits(self, track: mot.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, List[str]]:
        limited = mot.clamp_bbox_to_frame_bounds(frame, bbox)
        tokens: List[str] = []
        if self.lorat_min_box_area > 0 and mot.bbox_area(limited) < self.lorat_min_box_area:
            limited = self._scale_bbox_to_area(limited, self.lorat_min_box_area, frame)
            tokens.append("MINAREA")
        if self.lorat_max_area_change_per_frame > 1.0 and track.bbox is not None:
            previous_area = mot.bbox_area(track.bbox)
            current_area = mot.bbox_area(limited)
            min_area = max(self.lorat_min_box_area, previous_area / self.lorat_max_area_change_per_frame)
            max_area = max(min_area, previous_area * self.lorat_max_area_change_per_frame)
            target_area = min(max(current_area, min_area), max_area)
            if abs(target_area - current_area) > 0.5:
                limited = self._scale_bbox_to_area(limited, target_area, frame)
                tokens.append("SCALELIMIT")
        limited, size_floor_applied = self._apply_trusted_size_floor(track, limited, frame)
        if size_floor_applied:
            tokens.append("SIZEFLOOR")
        limited, small_lock_applied = self._apply_small_target_scale_lock(track, limited, frame)
        if small_lock_applied:
            tokens.append("SMALLLOCK")
        return limited, tokens

    def _candidate_occlusion_info(self, track: mot.TrackState, bbox: BBox) -> Tuple[Optional[int], float]:
        if self.occlusion_iou_threshold <= 0:
            return None, 0.0
        other_track_id, overlap = mot.strongest_track_overlap(track, bbox, self.tracks)
        if other_track_id is None or overlap < self.occlusion_iou_threshold:
            return None, overlap
        return other_track_id, overlap

    def _is_strong_memory_update(
        self,
        track: mot.TrackState,
        confidence: float,
        candidate_source: str,
        identity_assignment: Optional[mot.IdentityAssignment],
        motion_score: float,
        path_score: float,
    ) -> bool:
        if candidate_source != "head" and not self._is_reid_recovery(track, confidence, identity_assignment):
            return False
        if confidence < max(self.template_update_min_confidence, self.lorat_memory_min_score):
            return False
        if motion_score < self.v8_memory_min_motion:
            return False
        if mot.path_gate_ready(track) and path_score < self.v8_memory_min_path:
            return False
        if identity_assignment is None:
            return not self.identity_arbitrator.enabled

        score = identity_assignment.score
        is_view_change = self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            score,
        )
        if identity_assignment.output.source_track_id != track.track_id and not self._is_reid_recovery(track, confidence, identity_assignment):
            return False
        if self._is_initial_anchor_steal(score):
            return False
        if self._anchor_identity_reject_state(track, identity_assignment, memory_gate=True) is not None:
            return False
        if score.total < max(self.identity_arbitrator.min_score, self.v8_recovery_min_assignment_score) and not is_view_change:
            return False
        if score.appearance < self.v8_memory_min_appearance and not is_view_change:
            return False
        if score.motion < self.v8_memory_min_motion and not is_view_change:
            return False
        if mot.path_gate_ready(track) and score.path < self.v8_memory_min_path and not is_view_change:
            return False
        return True

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
        clipped = mot.clip_bbox_to_frame(frame, candidate)
        if clipped is None:
            return False
        raw_bbox = tuple(float(value) for value in clipped)
        if self.lorat_fixed_box_size:
            accepted, fixed_size_applied = self._apply_fixed_box_size(track, raw_bbox, frame)
            scale_tokens: List[str] = []
        else:
            accepted, scale_tokens = self._apply_scale_limits(track, raw_bbox, frame)
            fixed_size_applied = False

        reject_state = self._candidate_reject_state(track, accepted, confidence, identity_assignment, candidate_source)
        if reject_state is not None:
            if (
                identity_assignment is not None
                and self.identity_arbitrator.enabled
                and self.identity_arbitrator._should_remember_negative_reject(reject_state)
            ):
                self.identity_arbitrator.remember_negative_candidate(track, identity_assignment.output)
            track.raw_bbox = raw_bbox
            track.raw_confidence = confidence
            track.confidence = confidence
            track.state = reject_state
            self._apply_identity_scores(track, identity_assignment, fallback_confidence=confidence, fallback_margin=margin)
            return False

        (
            learning_held,
            learning_hold_reasons,
            crop_info,
            area_ratio,
            window_area_ratio,
            projected_shrink_frames,
        ) = self._assess_learning_hold(track, accepted, confidence, identity_assignment, frame)
        occlusion_track_id, occlusion_iou = self._candidate_occlusion_info(track, accepted)
        candidate_occluded = occlusion_track_id is not None
        previous = track.bbox
        motion_score = mot.motion_affinity(predicted, accepted, mot.bbox_diagonal(previous))
        path_score = mot.center_path_affinity(track, accepted)
        strong_memory_update = self._is_strong_memory_update(
            track,
            confidence,
            candidate_source,
            identity_assignment,
            motion_score,
            path_score,
        )
        if strong_memory_update:
            stable_update_streak = int(getattr(track, "v8_stable_update_streak", 0) or 0) + 1
        else:
            stable_update_streak = 0
        setattr(track, "v8_stable_update_streak", stable_update_streak)
        track.bbox = accepted
        track.raw_bbox = raw_bbox
        track.previous_bbox = previous
        track.predicted_bbox = predicted
        track.velocity = mot.bbox_delta(previous, accepted)
        track.ok = True
        track.confidence = confidence
        track.raw_confidence = confidence
        self._apply_identity_scores(track, identity_assignment, fallback_confidence=confidence, fallback_margin=margin)
        if identity_assignment is None:
            track.assignment_score = confidence
            track.assignment_margin = margin
            track.source_score = confidence
            track.initial_anchor_score = None
            track.other_anchor_score = None
            track.other_anchor_track_id = None
            track.identity_margin = None
        track.reid_score = track.reid_score if track.reid_score is not None else confidence
        track.motion_score = motion_score
        track.path_score = path_score
        if identity_assignment is not None and identity_assignment.output.source_track_id != track.track_id:
            track.assigned_source = f"v8-reid-{candidate_source}-track-{identity_assignment.output.source_track_id}"
        else:
            track.assigned_source = f"v8-{candidate_source}"
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

        state = "V8HEAD" if candidate_source == "head" else f"V8HEAD-{candidate_source.upper()}"
        if identity_assignment is not None and identity_assignment.output.source_track_id != track.track_id:
            state = mot.append_state_token(f"REID-{state}", f"SRC{identity_assignment.output.source_track_id}")
        if fixed_size_applied:
            state = mot.append_state_token(state, "FIXEDSIZE")
        for token in scale_tokens:
            state = mot.append_state_token(state, token)
        if identity_assignment is not None and self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            identity_assignment.score,
        ):
            state = mot.append_state_token(state, "VIEWCHANGE")
        if self._is_reid_recovery(track, confidence, identity_assignment):
            state = mot.append_state_token(state, "REIDRECOVERY")
        if candidate_occluded:
            state = mot.append_state_token(state, "OCCLUSION")
        if learning_held:
            state = mot.append_state_token(state, "NOLEARN")
            for reason in learning_hold_reasons:
                state = mot.append_state_token(state, reason)
        if not strong_memory_update:
            state = mot.append_state_token(state, "MEMHELD")
        track.state = state
        mot.set_track_lifecycle(track)

        if track.kalman is None:
            track.kalman = mot.BBoxKalmanFilter(accepted)
        track.kalman.update(accepted, confidence)

        can_refresh_memory = (
            strong_memory_update
            and stable_update_streak >= self.v8_memory_min_stable_updates
            and not candidate_occluded
            and not learning_held
            and self._should_refresh_head_memory(track, confidence, frame_number, candidate_source)
        )
        if can_refresh_memory:
            new_head = self._template_slot_for_bbox(
                feature_map,
                accepted,
                frame.shape,
                "recent",
                frame_number,
                confidence,
                self._siamfc_context_bbox(accepted, 2.0),
            )
            self._refresh_track_head_bank(track, new_head, frame_number)
            self._commit_trusted_size(track, accepted)
            track.last_reliable_bbox = accepted
            track.last_reliable_frame = frame_number
            mot.record_reliable_track_trajectory(track, frame_number, accepted, self.trajectory_history_size)
            if identity_assignment is not None:
                refresh_started = time.perf_counter()
                self.identity_arbitrator.commit_track_memory(
                    track,
                    identity_assignment.output,
                    identity_assignment,
                    None,
                )
                self._add_profile_seconds("appearance_refresh", time.perf_counter() - refresh_started)
            else:
                self._refresh_feature_appearance(track, new_head.vector)
        self._record_size_history(track, frame_number, accepted)
        mot.record_track_trajectory(track, frame_number, accepted, self.trajectory_history_size)
        return True

    def _hold_track(
        self,
        track: mot.TrackState,
        predicted: BBox,
        confidence: float,
        margin: float,
        frame_number: int,
        frame: Optional[np.ndarray] = None,
        hold_reason: str = "",
    ) -> None:
        previous = track.bbox
        held_bbox = mot.clamp_bbox_to_frame_bounds(frame, predicted)
        track.previous_bbox = previous
        track.predicted_bbox = held_bbox
        track.raw_bbox = held_bbox
        track.bbox = mot.clamp_bbox_size(held_bbox)
        track.velocity = mot.bbox_delta(previous, track.bbox)
        track.confidence = max(0.0, min(1.0, confidence))
        track.raw_confidence = confidence
        track.assignment_score = confidence
        track.assignment_margin = margin
        track.reid_score = confidence
        track.motion_score = mot.motion_affinity(predicted, track.bbox, mot.bbox_diagonal(previous))
        track.path_score = mot.center_path_affinity(track, track.bbox)
        track.source_score = confidence
        track.assigned_source = "v8-kalman-hold"
        track.lost_frames += 1
        track.occluded_frames += 1
        track.learning_block_reason = ""
        track.occlusion_track_id = None
        track.occlusion_iou = None
        if track.kalman is not None:
            track.kalman.state[4:] *= self.occlusion_velocity_damping
        conflict_suppressed = str(hold_reason or "").startswith("ASSIGNMENT_CONFLICT")
        if conflict_suppressed:
            track.ok = False
            track.assigned_source = "v8-conflict-suppressed-hold"
        else:
            track.ok = self.occlusion_max_frames > 0 and track.lost_frames <= self.occlusion_max_frames
        state = mot.append_state_token("V8HEAD_MISS", "OCCLUDED") if track.ok else mot.append_state_token("V8HEAD_MISS", "LOST")
        if conflict_suppressed:
            state = mot.append_state_token(state, "CONFLICT_SUPPRESSED")
        if hold_reason:
            state = mot.append_state_token(state, hold_reason)
        track.state = state
        mot.set_track_lifecycle(track)
        mot.record_track_trajectory(track, frame_number, track.bbox, self.trajectory_history_size)

    def manual_reanchor_track(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: BBox,
        frame_number: int,
        seconds_spent: Optional[float] = None,
        source: str = "manual",
    ) -> mot.ManualReanchorEvent:
        track = self.track_by_id.get(int(track_id))
        if track is None:
            raise KeyError(f"Unknown track id for manual reanchor: {track_id}")

        previous_bbox = track.bbox
        previous_state = str(track.state or "")
        previous_lifecycle = mot.classify_track_lifecycle(track)
        clipped = mot.clamp_bbox_to_frame_bounds(frame, bbox)
        feature_map = self._encode_frame(frame)
        new_head = self._template_slot_for_bbox(
            feature_map,
            clipped,
            frame.shape,
            "manual",
            frame_number,
            1.0,
            self._siamfc_context_bbox(clipped, 2.0),
        )

        track.previous_bbox = previous_bbox
        track.predicted_bbox = clipped
        track.raw_bbox = clipped
        track.bbox = clipped
        track.velocity = mot.bbox_delta(previous_bbox, clipped)
        track.ok = True
        track.confidence = 1.0
        track.raw_confidence = 1.0
        track.assignment_score = 1.0
        track.assignment_margin = 1.0
        track.reid_score = 1.0
        track.motion_score = 1.0
        track.path_score = 1.0
        track.source_score = 1.0
        track.identity_margin = None
        track.occlusion_track_id = None
        track.occlusion_iou = None
        track.assigned_source = "manual_reanchor"
        track.lost_frames = 0
        track.occluded_frames = 0
        track.learning_block_reason = ""
        track.learning_held_frames = 0
        track.shrink_risk_frames = 0
        track.state = "MANUAL_REANCHOR"
        track.active_template_frame = frame_number
        track.last_reliable_bbox = clipped
        track.last_reliable_frame = frame_number
        track.kalman = mot.BBoxKalmanFilter(clipped)
        setattr(track, "v8_stable_update_streak", self.v8_memory_min_stable_updates)

        bank = self._get_track_head_bank(track)
        if bank:
            updated_bank = [bank[0], new_head, *bank[1:]]
        else:
            updated_bank = [new_head]
        self._set_track_head_bank(track, updated_bank[: self.lorat_memory_slots])
        self._refresh_feature_appearance(track, new_head.vector)
        crop_feature = self._dinov2_crop_embeddings_for_bboxes(frame, [clipped])[0] if self.v8_dinov2_crop_reid else None
        if crop_feature is not None:
            track.v8_initial_crop_feature = crop_feature.detach().clone()
            track.v8_appearance_crop_feature = crop_feature.detach().clone()
            track.v8_crop_feature_bank = [crop_feature.detach().clone()]
        track.v8_negative_feature_bank = []
        track.v8_negative_crop_feature_bank = []
        self._commit_trusted_size(track, clipped)
        self._record_size_history(track, frame_number, clipped)
        mot.record_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        mot.record_reliable_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        mot.set_track_lifecycle(track, mot.TrackLifecycle.MANUAL_REANCHOR)

        return mot.make_manual_reanchor_event(
            frame_number,
            track.track_id,
            previous_bbox,
            clipped,
            previous_state,
            previous_lifecycle,
            seconds_spent=seconds_spent,
            source=source,
        )

    def _expanded_search_bbox(
        self,
        predicted: BBox,
        current: BBox,
        reference_size: Optional[Tuple[float, float]] = None,
    ) -> BBox:
        center_x, center_y = mot.bbox_center(predicted)
        _, _, current_w, current_h = current
        reference_w, reference_h = reference_size if reference_size is not None else (0.0, 0.0)
        search_w = max(current_w, predicted[2], reference_w) * self.search_radius_factor
        search_h = max(current_h, predicted[3], reference_h) * self.search_radius_factor
        return center_x - (search_w / 2.0), center_y - (search_h / 2.0), search_w, search_h

    def _bbox_to_grid_slices(self, bbox: BBox, frame_shape: Tuple[int, ...]) -> Tuple[slice, slice]:
        frame_height, frame_width = frame_shape[:2]
        x, y, w, h = mot.clamp_bbox_size(bbox)
        left = int(np.floor((x / max(1.0, float(frame_width))) * self.grid_width))
        right = int(np.ceil(((x + w) / max(1.0, float(frame_width))) * self.grid_width))
        top = int(np.floor((y / max(1.0, float(frame_height))) * self.grid_height))
        bottom = int(np.ceil(((y + h) / max(1.0, float(frame_height))) * self.grid_height))
        left = max(0, min(self.grid_width - 1, left))
        top = max(0, min(self.grid_height - 1, top))
        right = max(left + 1, min(self.grid_width, right))
        bottom = max(top + 1, min(self.grid_height, bottom))
        return slice(top, bottom), slice(left, right)

    def _feature_mean_for_bbox(self, feature_map, bbox: BBox, frame_shape: Tuple[int, ...]):
        y_slice, x_slice = self._bbox_to_grid_slices(bbox, frame_shape)
        roi = feature_map[y_slice, x_slice]
        if roi.numel() == 0:
            frame_height, frame_width = frame_shape[:2]
            center_x, center_y = mot.bbox_center(bbox)
            grid_x = int(np.clip((center_x / max(1.0, frame_width)) * self.grid_width, 0, self.grid_width - 1))
            grid_y = int(np.clip((center_y / max(1.0, frame_height)) * self.grid_height, 0, self.grid_height - 1))
            vector = feature_map[grid_y, grid_x]
        else:
            vector = roi.reshape(-1, self.embed_dim).mean(dim=0)
        return self.F.normalize(vector.to(self.torch.float32), dim=0).detach()

    def _feature_patch_for_bbox(self, feature_map, bbox: BBox, frame_shape: Tuple[int, ...]):
        y_slice, x_slice = self._bbox_to_grid_slices(bbox, frame_shape)
        roi = feature_map[y_slice, x_slice]
        if roi.numel() == 0:
            return self._feature_mean_for_bbox(feature_map, bbox, frame_shape).reshape(1, self.embed_dim)
        tokens = roi.reshape(-1, self.embed_dim).to(self.torch.float32)
        return self.F.normalize(tokens, dim=-1).detach()

    def _crop_tensor_for_bbox(self, frame: np.ndarray, bbox: BBox):
        clipped = mot.clip_bbox_to_frame(frame, bbox)
        if clipped is None or mot.bbox_area(clipped) < self.v8_dinov2_crop_reid_min_area:
            return None
        x, y, w, h = clipped
        left = int(max(0, np.floor(x)))
        top = int(max(0, np.floor(y)))
        right = int(min(frame.shape[1], np.ceil(x + w)))
        bottom = int(min(frame.shape[0], np.ceil(y + h)))
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).to(self.torch.float32)
        tensor.div_(255.0)
        self.image_normalize_transform(tensor)
        return tensor.to(self.dtype)

    def _dinov2_crop_embeddings_for_bboxes(self, frame: np.ndarray, bboxes: Sequence[BBox]):
        embeddings: List[Optional[object]] = [None for _ in bboxes]
        if not self.v8_dinov2_crop_reid or not bboxes:
            return embeddings

        started = time.perf_counter()
        try:
            valid: List[Tuple[int, object]] = []
            for index, bbox in enumerate(bboxes):
                tensor = self._crop_tensor_for_bbox(frame, bbox)
                if tensor is not None:
                    valid.append((index, tensor))
            for chunk in mot.chunk_sequence(valid, self.v8_dinov2_crop_reid_batch):
                if not chunk:
                    continue
                indices = [index for index, _ in chunk]
                batch = self.torch.stack([tensor for _, tensor in chunk], dim=0)
                with self.torch.inference_mode(), self.amp_autocast_fn():
                    tokens = self.lorat_model._x_feat(batch)
                    for block in self.lorat_model.blocks:
                        tokens = block(tokens)
                    tokens = self.lorat_model.norm(tokens)
                    pooled = tokens.reshape(tokens.shape[0], -1, self.embed_dim).mean(dim=1)
                    pooled = self.F.normalize(pooled.to(self.torch.float32), dim=-1).detach()
                self.runtime_status.crop_reid_forward_calls += 1
                self.runtime_status.crop_reid_forward_items += int(pooled.shape[0])
                self.runtime_status.max_crop_reid_batch = max(
                    self.runtime_status.max_crop_reid_batch,
                    int(pooled.shape[0]),
                )
                for output_index, embedding in zip(indices, pooled):
                    embeddings[output_index] = embedding
        finally:
            self._add_profile_seconds("dinov2_crop_reid", time.perf_counter() - started)
        return embeddings

    def _attach_dinov2_crop_reid_features(
        self,
        frame: np.ndarray,
        outputs: Sequence[mot.LoRATSlotOutput],
    ) -> None:
        if not self.v8_dinov2_crop_reid or not outputs:
            return
        embeddings = self._dinov2_crop_embeddings_for_bboxes(frame, [output.bbox for output in outputs])
        for output, embedding in zip(outputs, embeddings):
            if embedding is not None:
                output.v8_crop_feature = embedding

    @staticmethod
    def _siamfc_context_bbox(bbox: BBox, area_factor: float) -> BBox:
        x, y, w, h = mot.clamp_bbox_size(bbox)
        context_w = w + (float(area_factor) - 1.0) * ((w + h) * 0.5)
        context_h = h + (float(area_factor) - 1.0) * ((w + h) * 0.5)
        center_x, center_y = mot.bbox_center((x, y, w, h))
        return center_x - (context_w / 2.0), center_y - (context_h / 2.0), max(1.0, context_w), max(1.0, context_h)

    def _feature_patch_and_foreground_mask_for_bbox(
        self,
        feature_map,
        bbox: BBox,
        frame_shape: Tuple[int, ...],
        context_bbox: Optional[BBox] = None,
    ):
        context = context_bbox if context_bbox is not None else bbox
        y_slice, x_slice = self._bbox_to_grid_slices(context, frame_shape)
        roi = feature_map[y_slice, x_slice]
        if roi.numel() == 0:
            token = self._feature_mean_for_bbox(feature_map, bbox, frame_shape).reshape(1, self.embed_dim)
            return token, self.torch.ones((1,), device=feature_map.device, dtype=self.torch.bool)

        tokens = self.F.normalize(roi.reshape(-1, self.embed_dim).to(self.torch.float32), dim=-1).detach()
        foreground = self.torch.zeros((int(y_slice.stop - y_slice.start), int(x_slice.stop - x_slice.start)), device=feature_map.device, dtype=self.torch.bool)
        bbox_y, bbox_x = self._bbox_to_grid_slices(bbox, frame_shape)
        local_top = max(0, int(bbox_y.start - y_slice.start))
        local_bottom = min(foreground.shape[0], int(bbox_y.stop - y_slice.start))
        local_left = max(0, int(bbox_x.start - x_slice.start))
        local_right = min(foreground.shape[1], int(bbox_x.stop - x_slice.start))
        if local_bottom > local_top and local_right > local_left:
            foreground[local_top:local_bottom, local_left:local_right] = True
        if not bool(foreground.any().item()):
            foreground[:] = True
        return tokens, foreground.reshape(-1).detach()

    def _template_slot_for_bbox(
        self,
        feature_map,
        bbox: BBox,
        frame_shape: Tuple[int, ...],
        label: str,
        frame_number: int,
        confidence: Optional[float],
        context_bbox: Optional[BBox] = None,
    ) -> V8TemplateMemorySlot:
        patch_tokens, foreground_mask = self._feature_patch_and_foreground_mask_for_bbox(feature_map, bbox, frame_shape, context_bbox)
        foreground_tokens = patch_tokens[foreground_mask]
        if foreground_tokens.numel() == 0:
            foreground_tokens = patch_tokens
        vector = self.F.normalize(foreground_tokens.mean(dim=0), dim=0).detach()
        return V8TemplateMemorySlot(
            vector=vector,
            label=label,
            frame_number=int(frame_number),
            confidence=confidence,
            patch_tokens=patch_tokens.detach(),
            patch_foreground_mask=foreground_mask.detach(),
        )

    def _with_feature_appearance(
        self,
        output: mot.LoRATSlotOutput,
        feature_map,
        frame_shape: Tuple[int, ...],
    ) -> mot.LoRATSlotOutput:
        if getattr(output, "v8_feature", None) is not None:
            return output
        enriched = mot.LoRATSlotOutput(
            source_track_id=output.source_track_id,
            slot=output.slot,
            bbox=output.bbox,
            confidence=output.confidence,
            appearance_hist=output.appearance_hist,
            v8_crop_feature=output.v8_crop_feature,
        )
        enriched.v8_feature = self._feature_mean_for_bbox(feature_map, output.bbox, frame_shape)
        return enriched

    def _should_run_template_match(self, head_confidence: float, head_margin: float) -> bool:
        if not self.v8_template_match:
            return False
        if not self.object_conditioned_head.weights_loaded:
            return True
        if not self.v8_template_match_on_uncertain_only:
            return True
        return (
            head_confidence < self.v8_template_match_head_confidence_gate
            or head_margin < self.v8_template_match_margin_gate
        )

    def _feature_template_candidate(
        self,
        feature_map,
        track: mot.TrackState,
        predicted: BBox,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[Optional[BBox], float]:
        if not self.v8_template_match:
            return None, 0.0
        bank = self._get_track_head_bank(track)
        if not bank:
            return None, 0.0

        reference_w, reference_h = self._template_match_reference_size(track)
        roi = self._expanded_search_bbox(predicted, track.bbox, (reference_w, reference_h))
        y_slice, x_slice = self._bbox_to_grid_slices(roi, frame_shape)
        roi_features = feature_map[y_slice, x_slice]
        if roi_features.numel() == 0:
            return None, 0.0

        template_token_sets = []
        for slot in bank[: self.head_rank]:
            patch_tokens = getattr(slot, "patch_tokens", None)
            if patch_tokens is None or patch_tokens.numel() == 0:
                patch_tokens = slot.vector.reshape(1, self.embed_dim)
            patch_tokens = patch_tokens.to(feature_map.device, dtype=self.torch.float32).reshape(-1, self.embed_dim)
            foreground_mask = getattr(slot, "patch_foreground_mask", None)
            if foreground_mask is not None and foreground_mask.numel() > 0:
                foreground_mask = foreground_mask.to(feature_map.device, dtype=self.torch.bool).reshape(-1)
                foreground_count = min(int(foreground_mask.shape[0]), int(patch_tokens.shape[0]))
                selected_tokens = patch_tokens[:foreground_count][foreground_mask[:foreground_count]]
                if selected_tokens.numel() > 0:
                    patch_tokens = selected_tokens
            template_token_sets.append(patch_tokens)
        if not template_token_sets:
            return None, 0.0
        templates = self.F.normalize(self.torch.cat(template_token_sets, dim=0), dim=-1)
        flat_features = self.F.normalize(roi_features.reshape(-1, self.embed_dim).to(self.torch.float32), dim=-1)
        token_similarity = self.torch.matmul(flat_features, templates.transpose(0, 1))
        top_template_count = min(3, int(token_similarity.shape[1]))
        if top_template_count <= 1:
            token_scores = token_similarity.max(dim=1).values
        else:
            token_scores = self.torch.topk(token_similarity, k=top_template_count, dim=1).values.mean(dim=1)
        if token_scores.numel() == 0:
            return None, 0.0

        best_value, best_index = self.torch.max(token_scores, dim=0)
        if token_scores.numel() >= 2:
            top2 = self.torch.topk(token_scores, k=2).values
            margin = float((top2[0] - top2[1]).detach().to(device="cpu", dtype=self.torch.float32).item())
        else:
            margin = 0.0

        confidence = max(0.0, min(1.0, (float(best_value.detach().to(device="cpu", dtype=self.torch.float32).item()) + 1.0) * 0.5))
        min_template_score = (
            self.v8_small_target_template_min_score
            if self._is_small_target_track(track)
            else self.v8_template_match_min_score
        )
        if confidence < min_template_score:
            return None, confidence

        local_index = int(best_index.detach().to(device="cpu").item())
        roi_width = max(1, int(x_slice.stop - x_slice.start))
        local_y = local_index // roi_width
        local_x = local_index % roi_width
        grid_y = y_slice.start + local_y
        grid_x = x_slice.start + local_x
        frame_height, frame_width = frame_shape[:2]
        cell_width = float(frame_width) / float(max(1, self.grid_width))
        cell_height = float(frame_height) / float(max(1, self.grid_height))
        center_x = (float(grid_x) + 0.5) * cell_width
        center_y = (float(grid_y) + 0.5) * cell_height
        candidate = (
            center_x - (reference_w / 2.0),
            center_y - (reference_h / 2.0),
            reference_w,
            reference_h,
        )
        return mot.clamp_bbox_to_frame_bounds(None, candidate), max(confidence, min(1.0, confidence + max(0.0, margin) * 0.05))

    def _blend_bboxes(self, primary: BBox, secondary: BBox, secondary_weight: float, frame: np.ndarray) -> BBox:
        secondary_weight = max(0.0, min(1.0, float(secondary_weight)))
        primary_weight = 1.0 - secondary_weight
        p_center_x, p_center_y = mot.bbox_center(primary)
        s_center_x, s_center_y = mot.bbox_center(secondary)
        blended_w = (primary[2] * primary_weight) + (secondary[2] * secondary_weight)
        blended_h = (primary[3] * primary_weight) + (secondary[3] * secondary_weight)
        blended_center_x = (p_center_x * primary_weight) + (s_center_x * secondary_weight)
        blended_center_y = (p_center_y * primary_weight) + (s_center_y * secondary_weight)
        return mot.clamp_bbox_to_frame_bounds(
            frame,
            (
                blended_center_x - (blended_w / 2.0),
                blended_center_y - (blended_h / 2.0),
                blended_w,
                blended_h,
            ),
        )

    def _fuse_head_and_template_candidate(
        self,
        frame: np.ndarray,
        track: mot.TrackState,
        head_candidate: BBox,
        head_confidence: float,
        head_margin: float,
        template_candidate: Optional[BBox],
        template_confidence: float,
    ) -> Tuple[BBox, float, float, str]:
        if template_candidate is None:
            if self._is_small_target_track(track):
                locked, changed = self._apply_small_target_scale_lock(track, head_candidate, frame)
                if changed:
                    return locked, head_confidence, head_margin, "head-smallscale"
            return head_candidate, head_confidence, head_margin, "head"
        self.v8_template_match_hits += 1

        predicted = track.predicted_bbox or track.bbox
        reference_diagonal = max(1.0, mot.bbox_diagonal(track.bbox))
        head_motion = mot.motion_affinity(predicted, head_candidate, reference_diagonal)
        template_motion = mot.motion_affinity(predicted, template_candidate, reference_diagonal)
        head_path = mot.center_path_affinity(track, head_candidate)
        template_path = mot.center_path_affinity(track, template_candidate)
        head_template_iou = mot.bbox_iou(head_candidate, template_candidate)
        head_is_uncertain = head_confidence < self.min_confidence or head_margin < self.v8_template_match_margin_gate
        small_target = self._is_small_target_track(track)
        head_scale_bad = small_target and self._small_target_scale_error(track, head_candidate) > self.v8_small_target_max_scale_change
        if small_target:
            small_template_is_strong = (
                template_confidence >= self.v8_small_target_template_min_score
                and template_motion >= self.v8_small_target_template_min_motion
                and template_path >= self.v8_small_target_template_min_path
            )
            if small_template_is_strong and (head_is_uncertain or head_scale_bad or track.lost_frames > 0):
                self.v8_template_preferred_candidates += 1
                locked_template, _ = self._apply_small_target_scale_lock(track, template_candidate, frame)
                return (
                    locked_template,
                    max(head_confidence, template_confidence),
                    max(head_margin, template_confidence - head_confidence),
                    "small-template",
                )
        template_is_strong = (
            template_confidence >= max(self.v8_template_match_min_score, head_confidence + self.v8_template_match_prefer_margin)
            and template_motion >= DEFAULT_V8_TEMPLATE_RESCUE_MIN_MOTION
            and template_path >= DEFAULT_V8_TEMPLATE_RESCUE_MIN_PATH
        )
        template_agrees_with_head = head_template_iou >= DEFAULT_V8_TEMPLATE_RESCUE_MIN_HEAD_IOU
        prefer_template = (
            template_is_strong
            and head_is_uncertain
            and (template_agrees_with_head or track.lost_frames > 0 or not self.object_conditioned_head.weights_loaded)
        )

        if prefer_template:
            self.v8_template_preferred_candidates += 1
            return template_candidate, max(head_confidence, template_confidence), max(head_margin, template_confidence - head_confidence), "template"

        can_blend_template = (
            self.v8_head_template_blend > 0
            and template_confidence >= self.v8_template_match_min_score
            and template_agrees_with_head
            and template_motion >= max(0.45, head_motion - 0.05)
            and template_path >= max(0.45, head_path - 0.05)
        )
        if not can_blend_template:
            if small_target:
                locked, changed = self._apply_small_target_scale_lock(track, head_candidate, frame)
                if changed:
                    return locked, head_confidence, head_margin, "head-smallscale"
            return head_candidate, head_confidence, head_margin, "head"
        self.v8_template_fused_candidates += 1
        fused = self._blend_bboxes(head_candidate, template_candidate, self.v8_head_template_blend, frame)
        fused_confidence = max(head_confidence, (head_confidence * (1.0 - self.v8_head_template_blend)) + (template_confidence * self.v8_head_template_blend))
        fused_margin = max(head_margin, abs(template_confidence - head_confidence))
        return fused, fused_confidence, fused_margin, "fused-template"

    def _set_track_head_bank(self, track: mot.TrackState, bank) -> None:
        slots: List[V8TemplateMemorySlot] = []
        for index, item in enumerate(bank[: self.lorat_memory_slots]):
            if isinstance(item, V8TemplateMemorySlot):
                vector = item.vector
                patch_tokens = item.patch_tokens
                patch_foreground_mask = item.patch_foreground_mask
                frame_number = item.frame_number
                confidence = item.confidence
            else:
                vector = item
                patch_tokens = None
                patch_foreground_mask = None
                frame_number = int(track.active_template_frame or 0)
                confidence = track.confidence
            slots.append(
                V8TemplateMemorySlot(
                    vector=self.F.normalize(vector.detach().clone(), dim=0),
                    label="initial" if index == 0 else f"recent-{index}",
                    frame_number=int(frame_number),
                    confidence=confidence,
                    patch_tokens=patch_tokens.detach().clone() if patch_tokens is not None else None,
                    patch_foreground_mask=patch_foreground_mask.detach().clone() if patch_foreground_mask is not None else None,
                )
            )
        setattr(track, "v8_head_bank", slots)
        setattr(track, "v8_last_head_update_frame", track.active_template_frame)
        track.lorat_memory_slot_count = len(self._get_track_head_bank(track))

    @staticmethod
    def _get_track_head_bank(track: mot.TrackState):
        return list(getattr(track, "v8_head_bank", []))

    def _renumber_head_bank(self, bank: Sequence[V8TemplateMemorySlot]) -> List[V8TemplateMemorySlot]:
        renumbered: List[V8TemplateMemorySlot] = []
        for index, slot in enumerate(bank[: self.lorat_memory_slots]):
            renumbered.append(
                V8TemplateMemorySlot(
                    vector=slot.vector,
                    label="initial" if index == 0 else f"recent-{index}",
                    frame_number=slot.frame_number,
                    confidence=slot.confidence,
                    patch_tokens=slot.patch_tokens,
                    patch_foreground_mask=slot.patch_foreground_mask,
                )
            )
        return renumbered

    def _refresh_track_head_bank(self, track: mot.TrackState, slot_or_vector, frame_number: int) -> None:
        if isinstance(slot_or_vector, V8TemplateMemorySlot):
            vector = self.F.normalize(slot_or_vector.vector.detach().clone(), dim=0)
            patch_tokens = slot_or_vector.patch_tokens.detach().clone() if slot_or_vector.patch_tokens is not None else None
            patch_foreground_mask = slot_or_vector.patch_foreground_mask.detach().clone() if slot_or_vector.patch_foreground_mask is not None else None
        else:
            vector = self.F.normalize(slot_or_vector.detach().clone(), dim=0)
            patch_tokens = None
            patch_foreground_mask = None
        bank = self._get_track_head_bank(track)
        new_slot = V8TemplateMemorySlot(
            vector=vector,
            label=f"recent-{max(1, len(bank))}",
            frame_number=frame_number,
            confidence=track.confidence,
            patch_tokens=patch_tokens,
            patch_foreground_mask=patch_foreground_mask,
        )
        if not bank:
            bank = [
                V8TemplateMemorySlot(
                    vector=vector,
                    label="initial",
                    frame_number=frame_number,
                    confidence=track.confidence,
                    patch_tokens=patch_tokens,
                    patch_foreground_mask=patch_foreground_mask,
                )
            ]
        elif len(bank) < self.lorat_memory_slots:
            bank.append(new_slot)
        else:
            last_slot = bank[-1]
            updated_vector = self.F.normalize(
                ((1.0 - self.template_update_rate) * last_slot.vector) + (self.template_update_rate * vector),
                dim=0,
            )
            bank[-1] = V8TemplateMemorySlot(
                vector=updated_vector,
                label=last_slot.label,
                frame_number=frame_number,
                confidence=track.confidence,
                patch_tokens=patch_tokens if patch_tokens is not None else last_slot.patch_tokens,
                patch_foreground_mask=patch_foreground_mask if patch_foreground_mask is not None else last_slot.patch_foreground_mask,
            )
            if frame_number - int(getattr(track, "v8_last_head_update_frame", 0) or 0) >= self.lorat_memory_refresh_interval:
                bank = [bank[0], *bank[2:], new_slot]
        bank = self._renumber_head_bank(bank)
        setattr(track, "v8_head_bank", bank)
        setattr(track, "v8_last_head_update_frame", frame_number)
        track.active_template_frame = frame_number
        track.lorat_memory_slot_count = len(bank)
        track.active_lorat_slot = f"shared-head-r{track.lorat_memory_slot_count}"

    def _should_refresh_head_memory(
        self,
        track: mot.TrackState,
        confidence: float,
        frame_number: int,
        candidate_source: str = "head",
    ) -> bool:
        if candidate_source != "head":
            return False
        if confidence < max(self.template_update_min_confidence, self.lorat_memory_min_score):
            return False
        if track.assignment_score is not None and track.assignment_score < self.identity_arbitrator.min_score:
            return False
        if track.reid_score is not None and track.reid_score < self.identity_arbitrator.min_reid:
            if track.motion_score is None or track.motion_score < self.identity_arbitrator.view_change_min_motion:
                return False
        if track.path_score is not None and track.path_score < self.identity_arbitrator.min_path:
            return False
        if track.initial_anchor_score is not None and track.initial_anchor_score < self.v8_memory_min_initial_anchor:
            return False
        if (
            track.other_anchor_track_id is not None
            and track.other_anchor_score is not None
            and track.other_anchor_score >= 0.55
            and track.identity_margin is not None
            and track.identity_margin < self.v8_memory_min_identity_margin
        ):
            return False
        bank = self._get_track_head_bank(track)
        if len(bank) < self.lorat_memory_slots:
            return True
        last_update = int(getattr(track, "v8_last_head_update_frame", track.active_template_frame or frame_number))
        return frame_number - last_update >= self.lorat_memory_refresh_interval

    @staticmethod
    def _commit_trusted_size(track: mot.TrackState, bbox: BBox) -> None:
        track.trusted_size_bank.append(mot.clamp_bbox_size(bbox))
        if len(track.trusted_size_bank) > mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:
            del track.trusted_size_bank[: len(track.trusted_size_bank) - mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE]

    def _refresh_feature_appearance(self, track: mot.TrackState, vector) -> None:
        refresh_started = time.perf_counter()
        feature = self.F.normalize(vector.detach().to(self.device, dtype=self.torch.float32).clone(), dim=0)
        try:
            if getattr(track, "v8_initial_feature", None) is None:
                setattr(track, "v8_initial_feature", feature.detach().clone())
            current = getattr(track, "v8_appearance_feature", None)
            if current is None:
                setattr(track, "v8_appearance_feature", feature.detach().clone())
            else:
                update_rate = 0.06
                updated = self.F.normalize(((1.0 - update_rate) * current) + (update_rate * feature), dim=0)
                setattr(track, "v8_appearance_feature", updated.detach().clone())
            bank = list(getattr(track, "v8_feature_bank", []))
            if not bank or self.identity_arbitrator.feature_similarity(bank[-1], feature) < 0.985:
                bank.append(feature.detach().clone())
                if len(bank) > self.identity_arbitrator.appearance_bank_size:
                    del bank[: len(bank) - self.identity_arbitrator.appearance_bank_size]
            setattr(track, "v8_feature_bank", bank)
            track.appearance_updates += 1
        finally:
            self._add_profile_seconds("appearance_refresh", time.perf_counter() - refresh_started)

    @staticmethod
    def _record_size_history(track: mot.TrackState, frame_number: int, bbox: BBox) -> None:
        track.size_history.append((frame_number, mot.clamp_bbox_size(bbox)))
        if len(track.size_history) > mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:
            del track.size_history[: len(track.size_history) - mot.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE]

    def _update_gpu_status(self) -> None:
        if self.device.type != "cuda":
            return
        allocated = self.torch.cuda.memory_allocated(self.device)
        reserved = self.torch.cuda.memory_reserved(self.device)
        peak_allocated = self.torch.cuda.max_memory_allocated(self.device)
        peak_reserved = self.torch.cuda.max_memory_reserved(self.device)
        self.runtime_status.gpu_allocated_mb = mot.bytes_to_mb(allocated)
        self.runtime_status.gpu_reserved_mb = mot.bytes_to_mb(reserved)
        self.runtime_status.gpu_peak_allocated_mb = mot.bytes_to_mb(peak_allocated)
        self.runtime_status.gpu_peak_reserved_mb = mot.bytes_to_mb(peak_reserved)

    def status_lines(self) -> List[str]:
        status = self.runtime_status_snapshot()
        lifecycle_counts = mot.track_lifecycle_counts(self.tracks)
        negative_memories = sum(
            len(getattr(track, "v8_negative_feature_bank", []))
            + len(getattr(track, "v8_negative_crop_feature_bank", []))
            for track in self.tracks
        )
        lines = [
            f"FPS {status.fps:.2f} | objects {status.active_objects} | mode {V8_EXECUTION_MODE}",
            (
                "Track states "
                f"healthy {lifecycle_counts.get(mot.TrackLifecycle.HEALTHY, 0)} | "
                f"uncertain {lifecycle_counts.get(mot.TrackLifecycle.UNCERTAIN, 0)} | "
                f"lost {lifecycle_counts.get(mot.TrackLifecycle.LOST, 0)} | "
                f"reacquired {lifecycle_counts.get(mot.TrackLifecycle.REACQUIRED, 0)}"
            ),
            f"distractor memories {negative_memories}",
            (
                "shared ViT/frame "
                f"{status.shared_frame_backbone_calls} | head batches {status.object_head_batches} | "
                f"max head N {status.max_object_head_batch}"
            ),
            (
                "Week2 frame proof "
                f"backbone +{self._last_frame_backbone_delta} | "
                f"head batches +{self._last_frame_object_head_batch_delta} | "
                f"head objects +{self._last_frame_object_head_items_delta} | "
                f"selected heads +{self._last_frame_selected_head_items_delta}"
            ),
            (
                f"last timing backbone {self._last_backbone_seconds * 1000.0:.1f} ms | "
                f"heads {self._last_head_seconds * 1000.0:.1f} ms | {self._last_head_mode}"
            ),
            (
                "profile cand "
                f"{self._profile_ms('candidate_extract'):.1f} ms | "
                f"feat tmpl {self._profile_ms('template_match'):.1f} ms | "
                f"id {self._profile_ms('identity_resolve') + self._profile_ms('identity_score'):.1f} ms | "
                f"accept {self._profile_ms('accept'):.1f} ms | "
                f"debug {self._profile_ms('debug_output') + self._profile_ms('proof_output'):.1f} ms"
            ),
            (
                f"head memory {self.lorat_memory_slots} | primary/recovery "
                f"{status.gating_primary_decisions}/{status.gating_recovery_decisions} | "
                f"{status.gating_avg_slots_per_decision:.2f} heads/decision"
            ),
            f"head rank {self.head_rank} | selected heads last {self._last_selected_head_count} | roi tokens last {self._last_roi_tokens}",
            (
                "feature-template quality "
                f"{self.v8_template_match_hits}/{self.v8_template_match_attempts} hits | "
                f"{self.v8_template_preferred_candidates} preferred | "
                f"{self.v8_template_fused_candidates} fused"
            ),
            (
                "DINOv2 crop ReID "
                f"{status.crop_reid_forward_calls} calls | "
                f"{status.crop_reid_forward_items} crops | "
                f"max batch {status.max_crop_reid_batch}"
            ),
            (
                "assignment conflicts "
                + (
                    ", ".join(
                        f"{reason}:{count}"
                        for reason, count in sorted(self.v8_assignment_conflict_reason_counts.items())
                    )
                    if self.v8_assignment_conflict_reason_counts
                    else "0"
                )
            ),
            (
                "assignment alt rescue "
                f"{self.v8_assignment_alt_rescue_hits}/{self.v8_assignment_alt_rescue_attempts} hits"
                + (
                    " | rejects "
                    + ", ".join(
                        f"{reason}:{count}"
                        for reason, count in sorted(self.v8_assignment_alt_rescue_reject_counts.items())
                    )
                    if self.v8_assignment_alt_rescue_reject_counts
                    else ""
                )
            ),
        ]
        if status.gpu_name:
            gpu = f"GPU {status.gpu_name}"
            if status.gpu_allocated_mb is not None:
                gpu += (
                    f" | mem {status.gpu_allocated_mb:.0f}/{status.gpu_reserved_mb:.0f} MB "
                    f"peak {status.gpu_peak_allocated_mb:.0f} MB"
                )
            lines.append(gpu)
        return lines

    def runtime_status_snapshot(self) -> mot.RuntimeStatus:
        status = copy.copy(self.runtime_status)
        status.gating_decisions = self.v8_gating_decisions
        status.gating_primary_decisions = self.v8_primary_decisions
        status.gating_recovery_decisions = self.v8_recovery_decisions
        status.gating_selected_slot_items = self.v8_selected_head_items
        status.gating_avg_slots_per_decision = (
            self.v8_selected_head_items / self.v8_gating_decisions
            if self.v8_gating_decisions
            else 0.0
        )
        status.gating_recovery_reasons = ",".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.v8_recovery_reason_counts.items())
        )
        status.v8_assignment_conflict_rejections = sum(self.v8_assignment_conflict_reason_counts.values())
        status.v8_assignment_conflict_reasons = ",".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.v8_assignment_conflict_reason_counts.items())
        )
        status.v8_assignment_alt_rescue_attempts = self.v8_assignment_alt_rescue_attempts
        status.v8_assignment_alt_rescue_hits = self.v8_assignment_alt_rescue_hits
        status.v8_assignment_alt_rescue_rejects = ",".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.v8_assignment_alt_rescue_reject_counts.items())
        )
        status.v8_template_match_attempts = self.v8_template_match_attempts
        status.v8_template_match_hits = self.v8_template_match_hits
        status.v8_template_fused_candidates = self.v8_template_fused_candidates
        status.v8_template_preferred_candidates = self.v8_template_preferred_candidates
        status.crop_reid_forward_calls = self.runtime_status.crop_reid_forward_calls
        status.crop_reid_forward_items = self.runtime_status.crop_reid_forward_items
        status.max_crop_reid_batch = self.runtime_status.max_crop_reid_batch
        for bucket in V8_PROFILE_BUCKETS:
            setattr(status, f"v8_profile_{bucket}_seconds", self._profile_total_seconds.get(bucket, 0.0))
            setattr(status, f"v8_profile_{bucket}_ms_per_update", self._profile_total_ms_per_update(bucket))
            setattr(status, f"v8_last_{bucket}_seconds", self._last_profile_seconds.get(bucket, 0.0))
        return status

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "torch") and getattr(self, "device", None) is not None and self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def parse_initial_boxes(value: Optional[str]) -> List[BBox]:
    if not value:
        return []
    boxes: List[BBox] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError("--initial-boxes must use x,y,w,h entries separated by semicolons.")
        boxes.append(tuple(float(part) for part in parts))  # type: ignore[arg-type]
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Version 8 LoRAT-backed multi-object tracker with one shared frame ViT pass "
            "and batched per-object low-rank heads."
        )
    )
    parser.add_argument("--video", help="Path to a video file or camera index. Use 0 for webcam.")
    parser.add_argument("--sequence", type=Path, help="Path to a DanceTrack/MOT17-style sequence folder.")
    parser.add_argument("--dataset-root", type=Path, help="Root containing DanceTrack or MOT17 sequences.")
    parser.add_argument("--dataset", choices=("dancetrack", "mot17"), help="Dataset layout for --dataset-root.")
    parser.add_argument("--sequence-name", help="Sequence folder name to run from --dataset-root.")
    parser.add_argument("--list-sequences", action="store_true", help="List resolved dataset sequences and exit.")
    parser.add_argument("--sequence-fps", type=float, default=30.0, help="Playback FPS for image sequence inputs.")
    parser.add_argument("--initial-boxes", help="Semicolon-separated x,y,w,h boxes for headless/non-interactive runs.")
    parser.add_argument("--device", default="cpu", help="LoRAT device: cpu, dml/directml, cuda:0.")
    parser.add_argument("--lorat-root", type=Path, default=mot.DEFAULT_LORAT_ROOT, help="Local LoRAT checkout.")
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT weight. Defaults from --lorat-config.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Optional track cap. 0 means no cap.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable LoRAT automatic mixed precision.")
    parser.add_argument(
        "--v8-frame-size",
        type=int,
        default=0,
        help="Optional square shared-frame tensor size override; must match the selected LoRAT config size.",
    )
    parser.add_argument(
        "--v8-head-rank",
        type=int,
        default=mot.DEFAULT_LORAT_MEMORY_SLOTS,
        help="Maximum per-object low-rank heads scored from the V8 head bank.",
    )
    parser.add_argument("--v8-head-hidden-dim", type=int, default=256, help="Hidden dimension for the trainable V8 LoRA object head.")
    parser.add_argument("--v8-head-lora-rank", type=int, default=16, help="Low-rank adapter dimension for the trainable V8 object head.")
    parser.add_argument("--v8-head-weights", type=Path, help="Optional trained V8 object-head checkpoint.")
    parser.add_argument(
        "--v8-search-radius-factor",
        type=float,
        default=2.25,
        help="Search window size as a multiple of the current box size on the shared feature grid.",
    )
    parser.add_argument("--v8-min-confidence", type=float, default=0.48, help="Minimum head score to accept an update.")
    parser.add_argument(
        "--v8-template-update-rate",
        type=float,
        default=0.08,
        help="EMA rate for refreshing the newest per-object head vector.",
    )
    parser.add_argument(
        "--v8-template-update-min-confidence",
        type=float,
        default=0.58,
        help="Minimum confidence before a new ROI feature can refresh an object head.",
    )
    parser.add_argument(
        "--v8-score-reduction",
        choices=("max", "mean"),
        default="max",
        help="How to reduce a per-object head bank to one response map.",
    )
    parser.add_argument("--lorat-memory-slots", type=int, default=mot.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--lorat-memory-refresh-interval", type=int, default=mot.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL)
    parser.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    parser.add_argument("--lorat-accept-min-score", type=float, default=mot.DEFAULT_LORAT_ACCEPT_MIN_SCORE)
    parser.add_argument(
        "--fixed-lorat-box-size",
        dest="lorat_fixed_box_size",
        action="store_true",
        default=mot.DEFAULT_LORAT_FIXED_BOX_SIZE,
    )
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
    parser.add_argument("--v8-primary-heads-per-track", type=int, default=DEFAULT_V8_PRIMARY_HEADS_PER_TRACK)
    parser.add_argument("--v8-recovery-heads-per-track", type=int, default=DEFAULT_V8_RECOVERY_HEADS_PER_TRACK)
    parser.add_argument("--v8-recovery-interval", type=int, default=DEFAULT_V8_RECOVERY_INTERVAL)
    parser.add_argument("--v8-recovery-min-confidence", type=float, default=DEFAULT_V8_RECOVERY_MIN_CONFIDENCE)
    parser.add_argument("--v8-recovery-min-assignment-score", type=float, default=DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_SCORE)
    parser.add_argument("--v8-recovery-min-assignment-margin", type=float, default=DEFAULT_V8_RECOVERY_MIN_ASSIGNMENT_MARGIN)
    parser.add_argument("--v8-recovery-stale-head-frames", type=int, default=DEFAULT_V8_RECOVERY_STALE_HEAD_FRAMES)
    parser.add_argument(
        "--disable-v8-template-match",
        dest="v8_template_match",
        action="store_false",
        default=DEFAULT_V8_TEMPLATE_MATCH_ENABLED,
        help="Disable V8 shared-feature template recovery after the batched head.",
    )
    parser.add_argument("--v8-template-match-min-score", type=float, default=DEFAULT_V8_TEMPLATE_MATCH_MIN_SCORE)
    parser.add_argument("--v8-template-match-prefer-margin", type=float, default=DEFAULT_V8_TEMPLATE_MATCH_PREFER_MARGIN)
    parser.add_argument(
        "--v8-template-match-every-frame",
        dest="v8_template_match_on_uncertain_only",
        action="store_false",
        default=DEFAULT_V8_TEMPLATE_MATCH_ON_UNCERTAIN_ONLY,
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
        default=DEFAULT_V8_TEMPLATE_MATCH_HEAD_CONFIDENCE_GATE,
        help="With trained V8 heads, only run shared-feature template recovery below this head confidence.",
    )
    parser.add_argument(
        "--v8-template-match-margin-gate",
        type=float,
        default=DEFAULT_V8_TEMPLATE_MATCH_MARGIN_GATE,
        help="With trained V8 heads, only run shared-feature template recovery below this head margin.",
    )
    parser.add_argument(
        "--v8-head-template-blend",
        type=float,
        default=DEFAULT_V8_HEAD_TEMPLATE_BLEND,
        help="Blend weight for template candidate when the batched head remains the primary candidate.",
    )
    parser.add_argument("--v8-memory-min-motion", type=float, default=DEFAULT_V8_MEMORY_MIN_MOTION)
    parser.add_argument("--v8-memory-min-path", type=float, default=DEFAULT_V8_MEMORY_MIN_PATH)
    parser.add_argument("--v8-memory-min-appearance", type=float, default=DEFAULT_V8_MEMORY_MIN_APPEARANCE)
    parser.add_argument("--v8-memory-min-stable-updates", type=int, default=DEFAULT_V8_MEMORY_MIN_STABLE_UPDATES)
    parser.add_argument(
        "--v8-accept-min-initial-anchor",
        type=float,
        default=DEFAULT_V8_ACCEPT_MIN_INITIAL_ANCHOR,
        help="Minimum first-template appearance anchor for accepting low-appearance V8 candidates.",
    )
    parser.add_argument(
        "--v8-accept-min-identity-margin",
        type=float,
        default=DEFAULT_V8_ACCEPT_MIN_IDENTITY_MARGIN,
        help="Minimum initial-anchor margin over other tracks before accepting ambiguous V8 candidates.",
    )
    parser.add_argument(
        "--v8-memory-min-initial-anchor",
        type=float,
        default=DEFAULT_V8_MEMORY_MIN_INITIAL_ANCHOR,
        help="Minimum first-template appearance anchor before a V8 candidate can refresh learned memory.",
    )
    parser.add_argument(
        "--v8-memory-min-identity-margin",
        type=float,
        default=DEFAULT_V8_MEMORY_MIN_IDENTITY_MARGIN,
        help="Minimum first-template margin over other tracks before refreshing learned memory.",
    )
    parser.add_argument("--v8-window-penalty-ratio", type=float, default=DEFAULT_V8_WINDOW_PENALTY_RATIO)
    parser.add_argument(
        "--disable-v8-dinov2-crop-reid",
        dest="v8_dinov2_crop_reid",
        action="store_false",
        default=DEFAULT_V8_DINOV2_CROP_REID,
        help="Disable literal DINOv2 crop embeddings for Week 3 ReID.",
    )
    parser.add_argument("--v8-dinov2-crop-reid-batch", type=int, default=DEFAULT_V8_DINOV2_CROP_REID_BATCH)
    parser.add_argument("--v8-dinov2-crop-reid-min-area", type=float, default=DEFAULT_V8_DINOV2_CROP_REID_MIN_AREA)
    parser.add_argument("--v8-assignment-conflict-iou", type=float, default=DEFAULT_V8_ASSIGNMENT_CONFLICT_IOU)
    parser.add_argument("--v8-assignment-conflict-hard-iou", type=float, default=DEFAULT_V8_ASSIGNMENT_CONFLICT_HARD_IOU)
    parser.add_argument("--v8-assignment-conflict-score-margin", type=float, default=DEFAULT_V8_ASSIGNMENT_CONFLICT_SCORE_MARGIN)
    parser.add_argument(
        "--v8-assignment-conflict-center-ratio",
        type=float,
        default=DEFAULT_V8_ASSIGNMENT_CONFLICT_CENTER_RATIO,
        help="Reject weaker assignments whose boxes have centers this close, measured as center distance / smaller box diagonal.",
    )
    parser.add_argument(
        "--v8-assignment-conflict-containment",
        type=float,
        default=DEFAULT_V8_ASSIGNMENT_CONFLICT_CONTAINMENT,
        help="Reject weaker assignments when candidate boxes overlap by at least this fraction of the smaller box area.",
    )
    parser.add_argument(
        "--v8-assignment-conflict-ownership-margin",
        type=float,
        default=DEFAULT_V8_ASSIGNMENT_CONFLICT_OWNERSHIP_MARGIN,
        help="Minimum initial-anchor margin needed to protect a same-source track from spatial conflict suppression.",
    )
    parser.add_argument(
        "--disable-v8-assignment-alt-rescue",
        dest="v8_assignment_alt_rescue",
        action="store_false",
        default=DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_ENABLED,
        help="Disable trying a conflicted track's next-best V8 head candidates before holding it.",
    )
    parser.add_argument(
        "--v8-assignment-alt-rescue-max-candidates",
        type=int,
        default=DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MAX_CANDIDATES,
        help="Maximum top-k head alternatives to test when a track loses spatial assignment arbitration.",
    )
    parser.add_argument(
        "--v8-assignment-alt-rescue-min-confidence",
        type=float,
        default=DEFAULT_V8_ASSIGNMENT_ALT_RESCUE_MIN_CONFIDENCE,
        help="Minimum head confidence for alternate candidates used to rescue spatial assignment conflicts.",
    )
    parser.add_argument(
        "--disable-v8-small-target-mode",
        dest="v8_small_target_mode",
        action="store_false",
        default=DEFAULT_V8_SMALL_TARGET_MODE,
        help="Disable selected-small-target scale lock and template rescue.",
    )
    parser.add_argument("--v8-small-target-area", type=float, default=DEFAULT_V8_SMALL_TARGET_AREA)
    parser.add_argument("--v8-small-target-max-side", type=float, default=DEFAULT_V8_SMALL_TARGET_MAX_SIDE)
    parser.add_argument("--v8-small-target-max-scale-change", type=float, default=DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE)
    parser.add_argument("--v8-small-target-template-min-score", type=float, default=DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE)
    parser.add_argument("--v8-small-target-template-min-motion", type=float, default=DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION)
    parser.add_argument("--v8-small-target-template-min-path", type=float, default=DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH)
    parser.add_argument("--v8-small-target-confidence-floor", type=float, default=DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR)
    parser.add_argument("--output", type=Path, help="MOTChallenge-format result file.")
    parser.add_argument("--save-video", type=Path, help="Annotated MP4 output path.")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated MP4 writing.")
    parser.add_argument("--debug-log", type=Path, help="Tracking debug CSV output path.")
    parser.add_argument("--slot-debug-log", type=Path, help="Enable and write V8 head-bank debug CSV output to this path.")
    parser.add_argument("--no-slot-debug-log", action="store_true", help="Compatibility no-op; slot debug is disabled unless --slot-debug-log is set.")
    parser.add_argument("--week2-proof-log", type=Path, help="Enable and write Week 2 shared-backbone proof CSV output to this path.")
    parser.add_argument("--no-week2-proof-log", action="store_true", help="Compatibility no-op; proof logging is disabled unless --week2-proof-log is set or a benchmark enables it.")
    parser.add_argument("--manual-event-log", type=Path, help="Human-cost event CSV for manual reanchors.")
    parser.add_argument("--no-manual-event-log", action="store_true", help="Disable manual reanchor event CSV writing.")
    parser.add_argument("--debug-frame-start", type=int, default=0, help="First frame to include in --debug-log; 0 means all.")
    parser.add_argument("--debug-frame-end", type=int, default=0, help="Last frame to include in --debug-log; 0 means all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for smoke tests.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow; requires --initial-boxes.")
    return parser.parse_args()


def create_backend(args: argparse.Namespace, source: mot.FrameSource, expected_tracks: int = 0):
    weight_path = args.weight_path or mot.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return V8QualityBatchedLoRATTracker(
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
        args.v8_small_target_mode,
        args.v8_small_target_area,
        args.v8_small_target_max_side,
        args.v8_small_target_max_scale_change,
        args.v8_small_target_template_min_score,
        args.v8_small_target_template_min_motion,
        args.v8_small_target_template_min_path,
        args.v8_small_target_confidence_floor,
    )


def default_output_path(source_name: str) -> Path:
    return mot.default_output_path(source_name, "lorat_v8")


def default_debug_log_path(source_name: str) -> Path:
    return mot.default_debug_log_path(source_name, "lorat_v8")


def default_week2_proof_log_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return mot.DEFAULT_DEBUG_DIR / f"{safe_name}_lorat_v8_week2_proof.csv"


def default_manual_event_log_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return mot.DEFAULT_DEBUG_DIR / f"{safe_name}_lorat_v8_manual_events.csv"


def write_week2_proof_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WEEK2_PROOF_LOG_HEADER + "".join(lines), encoding="utf-8")


def append_v8_debug_rows(
    lines: List[str],
    frame_number: int,
    tracks: Sequence[mot.TrackState],
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
            mot.csv_text(track.state),
            mot.csv_float(track.confidence),
            mot.csv_float(track.raw_confidence),
            mot.csv_float(track.confidence_baseline),
            *mot.csv_bbox(track.bbox),
            *mot.csv_bbox_measurements(track.bbox),
            *mot.csv_bbox(track.raw_bbox),
            *mot.csv_bbox(track.predicted_bbox),
            *mot.csv_bbox(track.previous_bbox),
            *mot.csv_bbox(track.velocity),
            mot.csv_float(track.assignment_score),
            mot.csv_float(track.assignment_margin),
            mot.csv_float(track.reid_score),
            mot.csv_float(track.motion_score),
            mot.csv_float(track.path_score),
            mot.csv_float(track.source_score),
            mot.csv_float(track.initial_anchor_score),
            mot.csv_float(track.other_anchor_score),
            str(track.other_anchor_track_id) if track.other_anchor_track_id is not None else "",
            mot.csv_float(track.identity_margin),
            str(track.occlusion_track_id) if track.occlusion_track_id is not None else "",
            mot.csv_float(track.occlusion_iou),
            mot.csv_text(track.assigned_source),
            str(track.lost_frames),
            str(track.occluded_frames),
            str(track.last_reliable_frame) if track.last_reliable_frame else "",
            str(track.active_template_frame) if track.active_template_frame is not None else "",
            mot.csv_text(track.active_lorat_slot),
            str(track.lorat_memory_slot_count),
            str(len(track.appearance_bank)),
            str(len(getattr(track, "v8_feature_bank", []))),
            str(len(getattr(track, "v8_crop_feature_bank", []))),
            mot.csv_text(mot.set_track_lifecycle(track)),
        ]
        lines.append(",".join(fields) + "\n")


def write_v8_debug_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(V8_DEBUG_LOG_HEADER + "".join(lines), encoding="utf-8")


def default_video_path(source_name: str) -> Path:
    return mot.default_video_path(source_name, "lorat_v8")


BASE_EXECUTION_MODE = V9_BASE_EXECUTION_MODE
DEFAULT_DISTRACTOR_MIN_SIMILARITY = DEFAULT_V9_BASE_DISTRACTOR_MIN_SIMILARITY
HeadCandidate = V8HeadCandidate
HeadCandidateInfo = V8HeadCandidateInfo
QualityBatchedLoRATTracker = V8QualityBatchedLoRATTracker
append_debug_rows = append_v8_debug_rows
write_debug_log = write_v8_debug_log


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

    boxes = parse_initial_boxes(args.initial_boxes)
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
        write_v8_debug_log(debug_log_path, debug_lines)
        print(f"Wrote debug CSV to: {debug_log_path}")
        if slot_debug_log_path is not None:
            mot.write_slot_debug_log(slot_debug_log_path, backend.slot_debug_lines)
            print(f"Wrote V8 head-bank debug CSV to: {slot_debug_log_path}")
        if week2_proof_log_path is not None:
            write_week2_proof_log(week2_proof_log_path, backend.week2_proof_lines)
            print(f"Wrote Week 2 shared-backbone proof CSV to: {week2_proof_log_path}")
        if manual_event_log_path is not None:
            mot.write_manual_event_csv(manual_event_log_path, manual_events)
            print(f"Wrote manual event CSV to: {manual_event_log_path}")
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        mot.append_mot_results(mot_lines, frame_number, backend.tracks)
        append_v8_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
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
                append_v8_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
            else:
                frame = last_frame.copy()

            shown = mot.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines())
            if writer is not None and not paused:
                writer.write(shown)

            if not args.no_display:
                cv2.imshow("LoRAT Multi-Object Tracker V8", shown)
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
                        append_v8_debug_rows(
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
                            append_v8_debug_rows(
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
