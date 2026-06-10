from __future__ import annotations

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

import bounding_box_v5_lorat_shared as v5

BBox = v5.BBox

V7_EXECUTION_MODE = "shared-frame-vit-batched-heads"
DEFAULT_V7_PRIMARY_HEADS_PER_TRACK = 1
DEFAULT_V7_RECOVERY_HEADS_PER_TRACK = 5
DEFAULT_V7_RECOVERY_INTERVAL = 15
DEFAULT_V7_RECOVERY_MIN_CONFIDENCE = 0.45
DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_SCORE = 0.58
DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_MARGIN = 0.08
DEFAULT_V7_RECOVERY_STALE_HEAD_FRAMES = 30
V7_PROFILE_BUCKETS = (
    "candidate_transfer",
    "candidate_extract",
    "reid_appearance",
    "identity_resolve",
    "identity_score",
    "debug_output",
    "accept",
    "hold",
    "appearance_refresh",
    "proof_output",
)


@dataclass(frozen=True)
class SharedFrameEncoding:
    feature_map: object
    elapsed_seconds: float


@dataclass
class V7TemplateMemorySlot:
    vector: object
    label: str
    frame_number: int
    confidence: Optional[float] = None


@dataclass(frozen=True)
class BatchedHeadOutput:
    score_maps: object
    box_delta_maps: object
    elapsed_seconds: float
    selected_head_count: int


WEEK2_PROOF_LOG_HEADER = (
    "frame,phase,mode,head_mode,tracked_objects_this_frame,active_objects,frame_seconds,fps,"
    "shared_backbone_calls_this_frame,object_head_batches_this_frame,object_head_items_this_frame,"
    "selected_head_items_this_frame,cumulative_shared_backbone_calls,cumulative_object_head_batches,"
    "cumulative_object_head_items,cumulative_selected_head_items,max_object_head_batch,last_backbone_ms,"
    "last_head_ms,roi_tokens_this_frame,profile_candidate_transfer_ms,profile_candidate_extract_ms,"
    "profile_reid_appearance_ms,profile_identity_resolve_ms,profile_identity_score_ms,profile_debug_output_ms,"
    "profile_accept_ms,profile_hold_ms,profile_appearance_refresh_ms,profile_proof_output_ms,"
    "profile_unbucketed_ms,gpu_name,gpu_allocated_mb,gpu_reserved_mb,gpu_peak_allocated_mb,"
    "gpu_peak_reserved_mb,week2_shared_backbone_ok,week2_batched_head_ok\n"
)


class SharedFrameLoRATEncoder:
    """V7-only frame-level encoder that avoids the LoRAT evaluator pipeline."""

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
    """V7-only object-conditioned head over one shared frame feature map."""

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

        class V7ObjectConditionedLoRAHead(nn.Module):
            """Trainable per-object LoRA-conditioned score and box head.

            The base projection is shared across all objects. Each object's cached
            template embedding generates a low-rank up-projection, so the LoRA delta
            is batched across objects while the frame tokens are shared.
            """

            def __init__(self) -> None:
                super().__init__()
                self.feature_norm = nn.LayerNorm(embed_dim)
                self.object_norm = nn.LayerNorm(embed_dim)
                self.base_projection = nn.Linear(embed_dim, hidden_dim)
                self.lora_down = nn.Linear(embed_dim, lora_rank, bias=False)
                self.lora_up_generator = nn.Linear(embed_dim, lora_rank * hidden_dim)
                self.object_bias = nn.Linear(embed_dim, hidden_dim)
                self.activation = nn.GELU()
                self.score_head = nn.Linear(hidden_dim, 1)
                self.box_head = nn.Linear(hidden_dim, 4)
                self.lora_scale = 1.0 / float(lora_rank)
                self.box_delta_scale = float(box_delta_scale)
                nn.init.zeros_(self.score_head.bias)
                nn.init.zeros_(self.box_head.weight)
                nn.init.zeros_(self.box_head.bias)

            def forward(self, feature_tokens, object_embeddings):
                feature_tokens = self.feature_norm(feature_tokens)
                object_embeddings = self.object_norm(object_embeddings)
                base = self.base_projection(feature_tokens)
                down = self.lora_down(feature_tokens)
                up = self.lora_up_generator(object_embeddings).view(
                    object_embeddings.shape[0],
                    lora_rank,
                    hidden_dim,
                )
                lora_delta = torch_module.einsum("lr,nrh->nlh", down, up) * self.lora_scale
                object_bias = self.object_bias(object_embeddings)[:, None, :]
                hidden = self.activation(base[None, :, :] + object_bias + lora_delta)
                score_logits = self.score_head(hidden).squeeze(-1)
                box_deltas = self.box_head(hidden) * self.box_delta_scale
                return score_logits, box_deltas

        return V7ObjectConditionedLoRAHead()

    def load_weights(self, path: Path) -> None:
        path = path.resolve()
        if not path.exists():
            raise RuntimeError(f"V7 head weight file not found: {path}")
        state = self.torch.load(str(path), map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.module.load_state_dict(state, strict=True)
        self.weights_loaded = True
        self.last_mode = "lora_conditioned"

    def state_dict(self):
        return self.module.state_dict()

    def parameters(self):
        return self.module.parameters()

    @staticmethod
    def _slot_vector(slot: object):
        return slot.vector if isinstance(slot, V7TemplateMemorySlot) else slot

    def _build_head_tensor(self, selected_banks: Sequence[Sequence[object]]):
        max_heads = max((len(bank) for bank in selected_banks), default=1)
        max_heads = max(1, min(max_heads, self.max_head_rank))
        head_tensor = self.torch.zeros(
            (len(selected_banks), max_heads, self.embed_dim),
            device=self.device,
            dtype=self.torch.float32,
        )
        head_mask = self.torch.zeros((len(selected_banks), max_heads), device=self.device, dtype=self.torch.bool)
        for track_index, head_bank in enumerate(selected_banks):
            if not head_bank:
                head_bank = [self.torch.zeros(self.embed_dim, device=self.device, dtype=self.torch.float32)]
            for head_index, slot in enumerate(head_bank[:max_heads]):
                vector = self._slot_vector(slot)
                head_tensor[track_index, head_index] = vector.to(self.device, dtype=self.torch.float32)
                head_mask[track_index, head_index] = True
        return self.F.normalize(head_tensor, dim=-1), head_mask

    def _object_embeddings(self, selected_banks: Sequence[Sequence[object]]):
        head_tensor, head_mask = self._build_head_tensor(selected_banks)
        weights = head_mask.to(head_tensor.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        embeddings = (head_tensor * weights[:, :, None]).sum(dim=1)
        return self.F.normalize(embeddings, dim=-1)

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
            object_embeddings = self._object_embeddings(selected_banks)
            if not self.weights_loaded and not self.module.training:
                self.last_mode = "zero_shot_similarity"
                normalized_features = self.F.normalize(flat_features.to(self.torch.float32), dim=-1)
                score_logits = self.torch.matmul(object_embeddings, normalized_features.transpose(0, 1)) * 10.0
                box_deltas = self.torch.zeros(
                    (len(selected_banks), flat_features.shape[0], 4),
                    device=self.device,
                    dtype=self.torch.float32,
                )
            else:
                self.last_mode = "lora_conditioned"
                score_logits, box_deltas = self.module(flat_features, object_embeddings)
            score_maps = score_logits.reshape(len(selected_banks), feature_map.shape[0], feature_map.shape[1])
            box_delta_maps = box_deltas.reshape(len(selected_banks), feature_map.shape[0], feature_map.shape[1], 4)
        return BatchedHeadOutput(
            score_maps=score_maps,
            box_delta_maps=box_delta_maps,
            elapsed_seconds=time.perf_counter() - started,
            selected_head_count=selected_head_count,
        )


class SharedFrameLoRATMultiObjectTracker:
    """Standalone v7 tracker with one shared LoRAT ViT frame pass per video frame.

    Upstream LoRAT fuses template and search tokens inside the ViT blocks, so the exact
    original SOT head cannot be reused without per-object transformer work. This branch
    keeps LoRAT's LoRA-adapted DINOv2 blocks as the shared frame backbone and moves the
    object-specific work into a small batched low-rank head bank.

    V7 intentionally does not subclass the v5/v6 trackers and does not call the
    upstream per-object LoRAT evaluator in its frame update path. It imports v5 only
    for shared dataclasses, geometry helpers, debug writers, and UI/output helpers.
    """

    backend_name = "LoRAT-v7-frame-shared"

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
        head_rank: int = v5.DEFAULT_LORAT_MEMORY_SLOTS,
        head_hidden_dim: int = 256,
        head_lora_rank: int = 16,
        head_weight_path: Optional[Path] = None,
        search_radius_factor: float = 2.25,
        min_confidence: float = 0.48,
        template_update_rate: float = 0.08,
        template_update_min_confidence: float = 0.58,
        lorat_memory_slots: int = v5.DEFAULT_LORAT_MEMORY_SLOTS,
        lorat_memory_refresh_interval: int = v5.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL,
        lorat_memory_min_score: float = 0.55,
        lorat_accept_min_score: float = v5.DEFAULT_LORAT_ACCEPT_MIN_SCORE,
        lorat_fixed_box_size: bool = v5.DEFAULT_LORAT_FIXED_BOX_SIZE,
        lorat_min_box_area: float = v5.DEFAULT_LORAT_MIN_BOX_AREA,
        lorat_max_area_change_per_frame: float = v5.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME,
        lorat_trusted_size_floor_scale: float = v5.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE,
        shrink_guard_window: int = v5.DEFAULT_SHRINK_GUARD_WINDOW,
        shrink_guard_area_ratio: float = v5.DEFAULT_SHRINK_GUARD_AREA_RATIO,
        shrink_guard_step_ratio: float = v5.DEFAULT_SHRINK_GUARD_STEP_RATIO,
        shrink_guard_min_confidence: float = v5.DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE,
        shrink_guard_min_reid: float = v5.DEFAULT_SHRINK_GUARD_MIN_REID,
        crop_information_min_score: float = v5.DEFAULT_CROP_INFORMATION_MIN_SCORE,
        crop_information_min_pixels: int = v5.DEFAULT_CROP_INFORMATION_MIN_PIXELS,
        identity_arbitration: bool = True,
        identity_min_score: float = v5.DEFAULT_IDENTITY_MIN_SCORE,
        identity_min_reid: float = v5.DEFAULT_IDENTITY_MIN_REID,
        identity_min_motion: float = v5.DEFAULT_IDENTITY_MIN_MOTION,
        identity_min_path: float = v5.DEFAULT_IDENTITY_MIN_PATH,
        identity_bank_size: int = 12,
        identity_memory_min_confidence: float = v5.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE,
        occlusion_max_frames: int = v5.DEFAULT_OCCLUSION_MAX_FRAMES,
        occlusion_iou_threshold: float = v5.DEFAULT_OCCLUSION_IOU_THRESHOLD,
        occlusion_velocity_damping: float = v5.DEFAULT_OCCLUSION_VELOCITY_DAMPING,
        reid_recovery_min_score: float = v5.DEFAULT_REID_RECOVERY_MIN_SCORE,
        reid_recovery_min_reid: float = v5.DEFAULT_REID_RECOVERY_MIN_REID,
        reid_recovery_min_motion: float = v5.DEFAULT_REID_RECOVERY_MIN_MOTION,
        reid_recovery_min_confidence: float = v5.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE,
        view_change_min_score: float = v5.DEFAULT_VIEW_CHANGE_MIN_SCORE,
        view_change_min_motion: float = v5.DEFAULT_VIEW_CHANGE_MIN_MOTION,
        view_change_min_confidence: float = v5.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE,
        view_change_max_lost_frames: int = v5.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES,
        v7_primary_heads_per_track: int = DEFAULT_V7_PRIMARY_HEADS_PER_TRACK,
        v7_recovery_heads_per_track: int = DEFAULT_V7_RECOVERY_HEADS_PER_TRACK,
        v7_recovery_interval: int = DEFAULT_V7_RECOVERY_INTERVAL,
        v7_recovery_min_confidence: float = DEFAULT_V7_RECOVERY_MIN_CONFIDENCE,
        v7_recovery_min_assignment_score: float = DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_SCORE,
        v7_recovery_min_assignment_margin: float = DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_MARGIN,
        v7_recovery_stale_head_frames: int = DEFAULT_V7_RECOVERY_STALE_HEAD_FRAMES,
        score_reduction: str = "max",
        collect_slot_debug: bool = True,
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
        self.lorat_memory_slots = max(1, min(v5.MAX_LORAT_MEMORY_SLOTS, int(lorat_memory_slots)))
        self.head_rank = max(1, min(v5.MAX_LORAT_MEMORY_SLOTS, int(head_rank or self.lorat_memory_slots)))
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
        self.identity_arbitrator = v5.LightweightIdentityArbitrator(
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
        self.v7_primary_heads_per_track = max(1, int(v7_primary_heads_per_track))
        self.v7_recovery_heads_per_track = max(1, int(v7_recovery_heads_per_track))
        self.v7_recovery_interval = max(0, int(v7_recovery_interval))
        self.v7_recovery_min_confidence = max(0.0, min(1.0, float(v7_recovery_min_confidence)))
        self.v7_recovery_min_assignment_score = max(0.0, min(1.0, float(v7_recovery_min_assignment_score)))
        self.v7_recovery_min_assignment_margin = max(0.0, float(v7_recovery_min_assignment_margin))
        self.v7_recovery_stale_head_frames = max(0, int(v7_recovery_stale_head_frames))
        self.v7_gating_decisions = 0
        self.v7_primary_decisions = 0
        self.v7_recovery_decisions = 0
        self.v7_selected_head_items = 0
        self.v7_recovery_reason_counts: Counter[str] = Counter()
        self.score_reduction = score_reduction
        if self.score_reduction not in {"max", "mean"}:
            raise ValueError("--v7-score-reduction must be 'max' or 'mean'.")
        self.collect_slot_debug = bool(collect_slot_debug)

        self.tracks: List[v5.TrackState] = []
        self.track_by_id: Dict[int, v5.TrackState] = {}
        self.next_track_id = 1
        self.closed = False
        self.using_directml = False
        self.device_label = self.device_string
        self.gpu_name = ""
        self.runtime_status = v5.RuntimeStatus()
        self.slot_debug_lines: List[str] = []
        self.week2_proof_lines: List[str] = []
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
        self._last_profile_seconds: Dict[str, float] = {bucket: 0.0 for bucket in V7_PROFILE_BUCKETS}
        self._profile_total_seconds: Dict[str, float] = {bucket: 0.0 for bucket in V7_PROFILE_BUCKETS}

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
                    "LoRAT v7 was asked to use DirectML, but torch-directml is not installed."
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
                "LoRAT v7 was asked to use cuda, but this PyTorch build reports no CUDA/HIP device."
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
                "LoRAT v7 expected a LoRAT_DINOv2-style model with _x_feat, blocks, norm, and x_size. "
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
                    "--v7-frame-size must match the configured LoRAT search_region_size. "
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

        print(
            f"Loaded LoRAT v7 shared-frame backend {self.config_name} on {self.device_label} "
            f"with weight {self.weight_path.name}. "
            f"Mode: {V7_EXECUTION_MODE}; frame tensor: {self.input_width}x{self.input_height}; "
            f"feature grid: {self.grid_width}x{self.grid_height}; head rank: {self.head_rank}; "
            f"LoRA head dim/rank: {self.head_hidden_dim}/{self.head_lora_rank}; "
            f"memory heads: {self.lorat_memory_slots}; primary/recovery heads: "
            f"{self.v7_primary_heads_per_track}/{self.v7_recovery_heads_per_track}; "
            f"score reduction: {self.score_reduction}; min confidence: {self.min_confidence:.2f}; "
            f"identity arbitration: {self.identity_arbitrator.enabled}."
        )
        if self.head_weight_path is None:
            print(
                "V7 object head weights: none supplied. Runtime will use the zero-shot shared-feature "
                "similarity head; load --v7-head-weights to use trained LoRA-conditioned heads."
            )
        else:
            print(f"V7 object head weights: {self.head_weight_path}")
        print(
            "Note: v7 uses the LoRA-adapted ViT as one shared frame backbone pass, then batches "
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
        for bucket in V7_PROFILE_BUCKETS:
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

    def _week2_counter_snapshot(self) -> Tuple[int, int, int, int]:
        return (
            self.runtime_status.shared_frame_backbone_calls,
            self.runtime_status.object_head_batches,
            self.runtime_status.object_head_items,
            self.v7_selected_head_items,
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
        before: Tuple[int, int, int, int],
    ) -> None:
        proof_started = time.perf_counter()
        status = self.runtime_status_snapshot()
        backbone_before, head_batches_before, head_items_before, selected_heads_before = before
        backbone_delta = status.shared_frame_backbone_calls - backbone_before
        head_batch_delta = status.object_head_batches - head_batches_before
        head_item_delta = status.object_head_items - head_items_before
        selected_head_delta = status.gating_selected_slot_items - selected_heads_before

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
            for bucket in V7_PROFILE_BUCKETS
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
            v5.csv_text(phase),
            v5.csv_text(V7_EXECUTION_MODE),
            v5.csv_text(self._last_head_mode),
            str(tracked_objects_this_frame),
            str(status.active_objects),
            v5.csv_float(status.last_frame_seconds),
            v5.csv_float(status.fps),
            str(backbone_delta),
            str(head_batch_delta),
            str(head_item_delta),
            str(selected_head_delta),
            str(status.shared_frame_backbone_calls),
            str(status.object_head_batches),
            str(status.object_head_items),
            str(status.gating_selected_slot_items),
            str(status.max_object_head_batch),
            v5.csv_float(self._last_backbone_seconds * 1000.0),
            v5.csv_float(self._last_head_seconds * 1000.0),
            str(self._last_roi_tokens),
            v5.csv_float(self._profile_ms("candidate_transfer")),
            v5.csv_float(self._profile_ms("candidate_extract")),
            v5.csv_float(self._profile_ms("reid_appearance")),
            v5.csv_float(self._profile_ms("identity_resolve")),
            v5.csv_float(self._profile_ms("identity_score")),
            v5.csv_float(self._profile_ms("debug_output")),
            v5.csv_float(self._profile_ms("accept")),
            v5.csv_float(self._profile_ms("hold")),
            v5.csv_float(self._profile_ms("appearance_refresh")),
            v5.csv_float(proof_elapsed_ms),
            v5.csv_float(unbucketed_seconds * 1000.0),
            v5.csv_text(status.gpu_name),
            v5.csv_float(status.gpu_allocated_mb),
            v5.csv_float(status.gpu_reserved_mb),
            v5.csv_float(status.gpu_peak_allocated_mb),
            v5.csv_float(status.gpu_peak_reserved_mb),
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
        self._append_week2_proof_row(frame_number, "initialize", 0, before)

    def add_tracks(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int) -> List[v5.TrackState]:
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
        self._append_week2_proof_row(frame_number, "add_tracks", 0, before)
        return added

    def _create_track(
        self,
        frame: np.ndarray,
        feature_map,
        bbox: BBox,
        frame_number: int,
    ) -> v5.TrackState:
        clipped = v5.clamp_bbox_to_frame_bounds(frame, bbox)
        head_vector = self._feature_mean_for_bbox(feature_map, clipped, frame.shape)
        track = v5.TrackState(
            track_id=self.next_track_id,
            bbox=clipped,
            previous_bbox=clipped,
            predicted_bbox=clipped,
            raw_bbox=clipped,
            color=v5.color_for_track(self.next_track_id),
            confidence=1.0,
            raw_confidence=1.0,
            confidence_baseline=1.0,
            state="initialized",
            active_template_frame=frame_number,
            assigned_source="v7-initial-selection",
            active_lorat_slot="initial",
            lorat_memory_slot_count=1,
            initial_bbox=clipped,
            trusted_size_bank=[v5.clamp_bbox_size(clipped)],
            appearance_hist=v5.extract_reid_histogram(frame, clipped),
            initial_appearance_hist=v5.extract_reid_histogram(frame, clipped),
            appearance_updates=1,
            kalman=v5.BBoxKalmanFilter(clipped),
            last_reliable_bbox=clipped,
            last_reliable_frame=frame_number,
        )
        if track.appearance_hist is not None:
            track.appearance_bank = [track.appearance_hist.copy()]
        self.identity_arbitrator.initialize_track(track, frame)
        track.size_history = [(frame_number, v5.clamp_bbox_size(clipped))]
        self._set_track_head_bank(track, [head_vector])
        v5.record_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        v5.record_reliable_track_trajectory(track, frame_number, clipped, self.trajectory_history_size)
        self.tracks.append(track)
        self.track_by_id[track.track_id] = track
        self.next_track_id += 1
        return track

    def update(self, frame: np.ndarray, frame_number: int) -> Sequence[v5.TrackState]:
        frame_started = time.perf_counter()
        before = self._week2_counter_snapshot()
        self._reset_last_profile()
        self._last_head_seconds = 0.0
        self._last_head_mode = "none"
        self._last_roi_tokens = 0
        self._last_selected_head_count = 0
        feature_map = self._encode_frame(frame)
        active_tracks = [track for track in self.tracks if track.ok]
        tracked_objects_this_frame = len(active_tracks)
        if active_tracks:
            self._score_and_update_tracks(frame, feature_map, active_tracks, frame_number)

        self._record_frame_status(time.perf_counter() - frame_started)
        self._append_week2_proof_row(frame_number, "track", tracked_objects_this_frame, before)
        return self.tracks

    def _score_and_update_tracks(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[v5.TrackState],
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

        transfer_started = time.perf_counter()
        score_maps_for_candidates = score_maps.detach().cpu()
        box_delta_maps_for_candidates = box_delta_maps.detach().cpu()
        self._add_profile_seconds("candidate_transfer", time.perf_counter() - transfer_started)

        total_roi_tokens = 0
        records: Dict[int, Dict[str, object]] = {}
        candidate_outputs: List[v5.LoRATSlotOutput] = []
        for index, track in enumerate(tracks):
            predicted = self._predict_track(track)
            candidate_started = time.perf_counter()
            candidate, confidence, margin, roi_tokens = self._candidate_from_score_map(
                score_maps_for_candidates[index],
                box_delta_maps_for_candidates[index],
                predicted,
                track,
                frame.shape,
            )
            self._add_profile_seconds("candidate_extract", time.perf_counter() - candidate_started)
            total_roi_tokens += roi_tokens
            slot = self._synthetic_head_slot(track, frame_number)
            output = v5.LoRATSlotOutput(
                source_track_id=track.track_id,
                slot=slot,
                bbox=candidate,
                confidence=confidence,
            )
            if self.identity_arbitrator.enabled:
                appearance_started = time.perf_counter()
                output = self.identity_arbitrator._with_appearance(output, frame)
                self._add_profile_seconds("reid_appearance", time.perf_counter() - appearance_started)
            records[track.track_id] = {
                "track": track,
                "predicted": predicted,
                "candidate": candidate,
                "confidence": confidence,
                "margin": margin,
                "slot": slot,
                "output": output,
            }
            candidate_outputs.append(output)

        if self.collect_slot_debug:
            debug_started = time.perf_counter()
            self._append_head_debug_rows(frame_number, tracks, candidate_outputs)
            self._add_profile_seconds("debug_output", time.perf_counter() - debug_started)
        resolve_started = time.perf_counter()
        assignments = self.identity_arbitrator.resolve(tracks, candidate_outputs, frame)
        self._add_profile_seconds("identity_resolve", time.perf_counter() - resolve_started)
        assignment_by_track_id = {assignment.track.track_id: assignment for assignment in assignments}
        assigned_output_keys = {
            (assignment.output.source_track_id, assignment.output.slot.task_id)
            for assignment in assignments
        }

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
                    score = self.identity_arbitrator.score(track, output, tracks)  # type: ignore[arg-type]
                    self._add_profile_seconds("identity_score", time.perf_counter() - score_started)
                    identity_assignment = v5.IdentityAssignment(
                        track=track,
                        output=output,  # type: ignore[arg-type]
                        score=score,
                        assignment_margin=float(source_record["margin"]),
                    )

            own_record = records.get(track.track_id)
            hold_predicted = own_record["predicted"] if own_record is not None else track.predicted_bbox or track.bbox
            hold_confidence = float(own_record["confidence"]) if own_record is not None else 0.0
            hold_margin = float(own_record["margin"]) if own_record is not None else 0.0
            if source_record is None or identity_assignment is None:
                hold_started = time.perf_counter()
                self._hold_track(track, hold_predicted, hold_confidence, hold_margin, frame_number)
                self._add_profile_seconds("hold", time.perf_counter() - hold_started)
                continue

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
            )
            self._add_profile_seconds("accept", time.perf_counter() - accept_started)
            if not accepted:
                hold_started = time.perf_counter()
                self._hold_track(track, hold_predicted, hold_confidence, hold_margin, frame_number)
                self._add_profile_seconds("hold", time.perf_counter() - hold_started)

        self._last_roi_tokens = total_roi_tokens
        self.runtime_status.object_head_roi_tokens += total_roi_tokens

    def _select_track_heads(self, track: v5.TrackState, frame_number: int) -> List[object]:
        head_bank = self._get_track_head_bank(track)
        if not head_bank:
            self._record_v7_gating([], [])
            return []

        reasons = self._v7_recovery_reasons(track, head_bank, frame_number)
        if reasons:
            limit = min(len(head_bank), self.head_rank, self.v7_recovery_heads_per_track)
            selected = self._select_recovery_heads(track, head_bank, frame_number, limit)
        else:
            limit = min(len(head_bank), self.head_rank, self.v7_primary_heads_per_track)
            selected = self._select_primary_heads(track, head_bank, limit)
        self._record_v7_gating(selected, reasons)
        return selected

    def _select_primary_heads(self, track: v5.TrackState, head_bank: Sequence[object], limit: int) -> List[object]:
        if limit <= 0:
            return []
        selected: List[object] = []

        def add(vector: Optional[object]) -> None:
            if vector is None or len(selected) >= limit:
                return
            if any(existing is vector for existing in selected):
                return
            selected.append(vector)

        add(head_bank[-1])
        add(head_bank[0])
        for vector in reversed(head_bank[1:]):
            add(vector)
        return selected

    def _select_recovery_heads(
        self,
        track: v5.TrackState,
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

    def _v7_recovery_reasons(
        self,
        track: v5.TrackState,
        head_bank: Sequence[object],
        frame_number: int,
    ) -> List[str]:
        if len(head_bank) <= self.v7_primary_heads_per_track:
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
        if track.confidence is not None and track.confidence < self.v7_recovery_min_confidence:
            reasons.append("LOW_CONFIDENCE")
        if track.assignment_score is not None and track.assignment_score < self.v7_recovery_min_assignment_score:
            reasons.append("LOW_ASSIGNMENT_SCORE")
        if track.assignment_margin is not None and track.assignment_margin < self.v7_recovery_min_assignment_margin:
            reasons.append("LOW_ASSIGNMENT_MARGIN")
        last_update = int(getattr(track, "v7_last_head_update_frame", track.active_template_frame or frame_number))
        if self.v7_recovery_stale_head_frames > 0 and frame_number - last_update >= self.v7_recovery_stale_head_frames:
            reasons.append("STALE_ACTIVE_HEAD")
        if self.v7_recovery_interval > 0 and track.trajectory:
            start_frame = track.trajectory[0][0]
            if frame_number > start_frame and (frame_number - start_frame) % self.v7_recovery_interval == 0:
                reasons.append("PERIODIC_ANCHOR_CHECK")
        return reasons

    def _record_v7_gating(self, selected: Sequence[object], reasons: Sequence[str]) -> None:
        self.v7_gating_decisions += 1
        self.v7_selected_head_items += len(selected)
        if reasons:
            self.v7_recovery_decisions += 1
            self.v7_recovery_reason_counts.update(reasons)
        else:
            self.v7_primary_decisions += 1

    def _synthetic_head_slot(self, track: v5.TrackState, frame_number: int) -> v5.LoRATMemorySlot:
        return v5.LoRATMemorySlot(
            task_id=(track.track_id * 1_000_000) + frame_number,
            track_id=track.track_id,
            label=str(track.active_lorat_slot or "v7-head"),
            frame_number=frame_number,
            bbox=track.bbox,
            confidence=track.confidence,
            raw_confidence=track.raw_confidence,
            confidence_baseline=track.confidence_baseline,
            last_refresh_frame=int(getattr(track, "v7_last_head_update_frame", track.active_template_frame or frame_number)),
            active=True,
            anchor_frame_number=int(track.active_template_frame or frame_number),
            anchor_bbox=track.initial_bbox,
        )

    def _append_head_debug_rows(
        self,
        frame_number: int,
        evaluated_tracks: Sequence[v5.TrackState],
        candidate_outputs: Sequence[v5.LoRATSlotOutput],
    ) -> None:
        for output in candidate_outputs:
            track = self.track_by_id.get(output.source_track_id)
            if track is None:
                continue
            score = self.identity_arbitrator.score(track, output, evaluated_tracks)
            fields = [
                str(frame_number),
                str(track.track_id),
                v5.csv_text(output.slot.label),
                str(output.slot.task_id),
                str(output.slot.frame_number),
                str(output.slot.anchor_frame_number),
                str(output.slot.last_refresh_frame),
                "1" if output.slot.active else "0",
                v5.csv_text(track.active_lorat_slot),
                v5.csv_float(output.confidence),
                v5.csv_float(output.confidence),
                v5.csv_float(output.slot.confidence_baseline),
                v5.csv_float(track.confidence_baseline),
                *v5.csv_bbox(output.bbox),
                *v5.csv_bbox(track.bbox),
                *v5.csv_bbox(track.predicted_bbox),
                v5.csv_float(score.total),
                v5.csv_float(score.appearance),
                v5.csv_float(score.motion),
                v5.csv_float(score.path),
                v5.csv_float(score.source),
                v5.csv_float(score.confidence),
                v5.csv_float(score.iou),
                v5.csv_float(score.initial_anchor),
                v5.csv_float(score.other_anchor),
                str(score.other_track_id) if score.other_track_id is not None else "",
                v5.csv_float(score.identity_margin),
                str(score.occlusion_track_id) if score.occlusion_track_id is not None else "",
                v5.csv_float(score.occlusion_iou),
            ]
            self.slot_debug_lines.append(",".join(fields) + "\n")

    def _predict_track(self, track: v5.TrackState) -> BBox:
        track.previous_bbox = track.bbox
        if track.kalman is not None:
            predicted = track.kalman.predict()
        else:
            predicted = v5.predict_bbox(track.bbox, track.velocity)
        track.predicted_bbox = predicted
        return predicted

    def _candidate_from_score_map(
        self,
        score_map,
        box_delta_map,
        predicted: BBox,
        track: v5.TrackState,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[BBox, float, float, int]:
        roi = self._expanded_search_bbox(predicted, track.bbox)
        y_slice, x_slice = self._bbox_to_grid_slices(roi, frame_shape)
        roi_scores = score_map[y_slice, x_slice]
        roi_tokens = int(roi_scores.numel())
        if roi_tokens <= 0:
            return v5.clamp_bbox_size(predicted), 0.0, 0.0, 0

        flat = roi_scores.flatten()
        best_index = int(self.torch.argmax(flat).item())
        best_value = float(flat[best_index].item())
        if flat.numel() >= 2:
            top2 = self.torch.topk(flat, k=2).values
            margin = float((top2[0] - top2[1]).item())
        else:
            margin = 0.0

        local_y = best_index // max(1, roi_scores.shape[1])
        local_x = best_index % max(1, roi_scores.shape[1])
        grid_y = y_slice.start + local_y
        grid_x = x_slice.start + local_x
        frame_height, frame_width = frame_shape[:2]
        cell_width = float(frame_width) / float(self.grid_width)
        cell_height = float(frame_height) / float(self.grid_height)
        delta = self.torch.tanh(box_delta_map[grid_y, grid_x]).detach()
        delta_x, delta_y, delta_log_w, delta_log_h = [float(value.item()) for value in delta]
        center_x = (float(grid_x) + 0.5 + (0.5 * delta_x)) * cell_width
        center_y = (float(grid_y) + 0.5 + (0.5 * delta_y)) * cell_height
        _, _, previous_w, previous_h = track.bbox
        predicted_w = max(1.0, previous_w * float(np.exp(np.clip(delta_log_w, -1.5, 1.5))))
        predicted_h = max(1.0, previous_h * float(np.exp(np.clip(delta_log_h, -1.5, 1.5))))
        candidate = (
            center_x - (predicted_w / 2.0),
            center_y - (predicted_h / 2.0),
            predicted_w,
            predicted_h,
        )
        confidence = float(self.torch.sigmoid(flat[best_index]).item())
        return v5.clamp_bbox_size(candidate), confidence, max(0.0, margin), roi_tokens

    def _apply_identity_scores(
        self,
        track: v5.TrackState,
        identity_assignment: Optional[v5.IdentityAssignment],
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
        track.occlusion_track_id = score.occlusion_track_id
        track.occlusion_iou = score.occlusion_iou

    def _candidate_reject_state(
        self,
        track: v5.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[v5.IdentityAssignment],
    ) -> Optional[str]:
        confidence_floor = max(self.min_confidence, self.lorat_accept_min_score)
        if confidence < confidence_floor:
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
        if (
            v5.path_gate_ready(track)
            and score.path < self.identity_arbitrator.min_path
            and confidence < 0.85
            and not is_view_change
            and not self._is_path_recovery(track, confidence, identity_assignment)
        ):
            return "PATHLOW"
        if score.total < self.identity_arbitrator.min_score and not is_view_change:
            return "ID_UNCERTAIN"
        if v5.track_has_appearance(track) and score.appearance < self.identity_arbitrator.min_reid and not is_view_change:
            return "REIDLOW"
        if score.motion < self.identity_arbitrator.min_motion and confidence < 0.70:
            return "MOTIONLOW"
        if track.lost_frames > 0:
            reacquire_confidence = min(0.95, max(self.lorat_accept_min_score + 0.10, 0.40))
            if confidence < reacquire_confidence and score.appearance < max(0.55, self.identity_arbitrator.min_reid):
                return "REACQUIRE_LOWCONF"
        return None

    @staticmethod
    def _is_initial_anchor_steal(score: v5.IdentityScore) -> bool:
        return (
            score.other_track_id is not None
            and score.occlusion_iou >= v5.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_IOU
            and score.other_anchor >= v5.DEFAULT_IDENTITY_ANCHOR_STEAL_MIN_OTHER
            and score.identity_margin <= -v5.DEFAULT_IDENTITY_ANCHOR_STEAL_MARGIN
        )

    def _is_path_recovery(
        self,
        track: v5.TrackState,
        confidence_value: float,
        identity_assignment: Optional[v5.IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames < v5.DEFAULT_PATH_RECOVERY_AFTER_FRAMES:
            return False
        score = identity_assignment.score
        return (
            identity_assignment.output.source_track_id == track.track_id
            and confidence_value >= v5.DEFAULT_PATH_RECOVERY_MIN_CONFIDENCE
            and score.appearance >= v5.DEFAULT_PATH_RECOVERY_MIN_REID
            and score.motion >= v5.DEFAULT_PATH_RECOVERY_MIN_MOTION
            and score.total >= self.identity_arbitrator.min_score
        )

    def _is_reid_recovery(
        self,
        track: v5.TrackState,
        confidence_value: float,
        identity_assignment: Optional[v5.IdentityAssignment],
    ) -> bool:
        if identity_assignment is None or track.lost_frames <= 0:
            return False
        score = identity_assignment.score
        return (
            confidence_value >= self.reid_recovery_min_confidence
            and score.total >= self.reid_recovery_min_score
            and score.appearance >= self.reid_recovery_min_reid
            and score.motion >= self.reid_recovery_min_motion
            and (score.path >= self.identity_arbitrator.min_path or self._is_path_recovery(track, confidence_value, identity_assignment))
        )

    def _learning_evidence_is_strong(
        self,
        track: v5.TrackState,
        confidence: float,
        identity_assignment: Optional[v5.IdentityAssignment],
    ) -> bool:
        if confidence < self.shrink_guard_min_confidence:
            return False
        if identity_assignment is None or not v5.track_has_appearance(track):
            return True
        return identity_assignment.score.appearance >= self.shrink_guard_min_reid

    def _assess_learning_hold(
        self,
        track: v5.TrackState,
        bbox: BBox,
        confidence: float,
        identity_assignment: Optional[v5.IdentityAssignment],
        frame: Optional[np.ndarray],
    ) -> Tuple[bool, List[str], v5.CropInformation, float, float, int]:
        crop_info = v5.measure_crop_information(frame, bbox, self.crop_information_min_pixels)
        previous_area = v5.bbox_area(track.bbox)
        current_area = v5.bbox_area(bbox)
        step_ratio = current_area / max(1.0, previous_area)
        recent_history = track.size_history[-self.shrink_guard_window :] if self.shrink_guard_window > 0 else []
        reference_area = max([previous_area] + [v5.bbox_area(sample_bbox) for _, sample_bbox in recent_history])
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

    def _trusted_size_floor(self, track: v5.TrackState) -> Optional[Tuple[float, float]]:
        if self.lorat_trusted_size_floor_scale <= 0:
            return None
        initial_floor: Optional[Tuple[float, float]] = None
        if track.initial_bbox is not None:
            _, _, initial_w, initial_h = v5.clamp_bbox_size(track.initial_bbox)
            initial_floor = (
                max(1.0, initial_w * self.lorat_trusted_size_floor_scale),
                max(1.0, initial_h * self.lorat_trusted_size_floor_scale),
            )
        samples = list(track.trusted_size_bank[-v5.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:])
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

    def _apply_trusted_size_floor(self, track: v5.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, bool]:
        x, y, w, h = v5.clamp_bbox_size(bbox)
        floor = self._trusted_size_floor(track)
        if floor is None:
            return v5.clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False
        min_w, min_h = floor
        if w >= min_w and h >= min_h:
            return v5.clamp_bbox_to_frame_bounds(frame, (x, y, w, h)), False
        center_x, center_y = v5.bbox_center((x, y, w, h))
        guarded_w = max(w, min_w)
        guarded_h = max(h, min_h)
        return (
            v5.clamp_bbox_to_frame_bounds(
                frame,
                (center_x - (guarded_w / 2.0), center_y - (guarded_h / 2.0), guarded_w, guarded_h),
            ),
            True,
        )

    def _apply_fixed_box_size(self, track: v5.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, bool]:
        if not self.lorat_fixed_box_size or track.initial_bbox is None:
            return v5.clamp_bbox_to_frame_bounds(frame, bbox), False
        _, _, fixed_w, fixed_h = v5.clamp_bbox_size(track.initial_bbox)
        x, y, w, h = v5.clamp_bbox_size(bbox)
        center_x, center_y = v5.bbox_center((x, y, w, h))
        fixed = v5.clamp_bbox_to_frame_bounds(
            frame,
            (center_x - (fixed_w / 2.0), center_y - (fixed_h / 2.0), fixed_w, fixed_h),
        )
        changed = abs(fixed[2] - w) > 0.01 or abs(fixed[3] - h) > 0.01
        return fixed, changed

    @staticmethod
    def _scale_bbox_to_area(bbox: BBox, target_area: float, frame: Optional[np.ndarray]) -> BBox:
        x, y, w, h = v5.clamp_bbox_size(bbox)
        scale = float(np.sqrt(max(1.0, target_area) / max(1.0, w * h)))
        center_x, center_y = v5.bbox_center((x, y, w, h))
        scaled_w = max(1.0, w * scale)
        scaled_h = max(1.0, h * scale)
        return v5.clamp_bbox_to_frame_bounds(
            frame,
            (center_x - (scaled_w / 2.0), center_y - (scaled_h / 2.0), scaled_w, scaled_h),
        )

    def _apply_scale_limits(self, track: v5.TrackState, bbox: BBox, frame: Optional[np.ndarray]) -> Tuple[BBox, List[str]]:
        limited = v5.clamp_bbox_to_frame_bounds(frame, bbox)
        tokens: List[str] = []
        if self.lorat_min_box_area > 0 and v5.bbox_area(limited) < self.lorat_min_box_area:
            limited = self._scale_bbox_to_area(limited, self.lorat_min_box_area, frame)
            tokens.append("MINAREA")
        if self.lorat_max_area_change_per_frame > 1.0 and track.bbox is not None:
            previous_area = v5.bbox_area(track.bbox)
            current_area = v5.bbox_area(limited)
            min_area = max(self.lorat_min_box_area, previous_area / self.lorat_max_area_change_per_frame)
            max_area = max(min_area, previous_area * self.lorat_max_area_change_per_frame)
            target_area = min(max(current_area, min_area), max_area)
            if abs(target_area - current_area) > 0.5:
                limited = self._scale_bbox_to_area(limited, target_area, frame)
                tokens.append("SCALELIMIT")
        limited, size_floor_applied = self._apply_trusted_size_floor(track, limited, frame)
        if size_floor_applied:
            tokens.append("SIZEFLOOR")
        return limited, tokens

    def _candidate_occlusion_info(self, track: v5.TrackState, bbox: BBox) -> Tuple[Optional[int], float]:
        if self.occlusion_iou_threshold <= 0:
            return None, 0.0
        other_track_id, overlap = v5.strongest_track_overlap(track, bbox, self.tracks)
        if other_track_id is None or overlap < self.occlusion_iou_threshold:
            return None, overlap
        return other_track_id, overlap

    def _accept_candidate(
        self,
        frame: np.ndarray,
        feature_map,
        track: v5.TrackState,
        candidate: BBox,
        confidence: float,
        margin: float,
        predicted: BBox,
        frame_number: int,
        identity_assignment: Optional[v5.IdentityAssignment] = None,
    ) -> bool:
        clipped = v5.clip_bbox_to_frame(frame, candidate)
        if clipped is None:
            return False
        raw_bbox = tuple(float(value) for value in clipped)
        if self.lorat_fixed_box_size:
            accepted, fixed_size_applied = self._apply_fixed_box_size(track, raw_bbox, frame)
            scale_tokens: List[str] = []
        else:
            accepted, scale_tokens = self._apply_scale_limits(track, raw_bbox, frame)
            fixed_size_applied = False

        reject_state = self._candidate_reject_state(track, accepted, confidence, identity_assignment)
        if reject_state is not None:
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
        motion_score = v5.motion_affinity(predicted, accepted, v5.bbox_diagonal(previous))
        path_score = v5.center_path_affinity(track, accepted)
        track.bbox = accepted
        track.raw_bbox = raw_bbox
        track.previous_bbox = previous
        track.predicted_bbox = predicted
        track.velocity = v5.bbox_delta(previous, accepted)
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
            track.assigned_source = f"v7-reid-head-track-{identity_assignment.output.source_track_id}"
        else:
            track.assigned_source = "v7-shared-frame-head"
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

        state = "V7HEAD"
        if identity_assignment is not None and identity_assignment.output.source_track_id != track.track_id:
            state = v5.append_state_token("REID-V7HEAD", f"SRC{identity_assignment.output.source_track_id}")
        if fixed_size_applied:
            state = v5.append_state_token(state, "FIXEDSIZE")
        for token in scale_tokens:
            state = v5.append_state_token(state, token)
        if identity_assignment is not None and self.identity_arbitrator.is_view_change_candidate(
            track,
            identity_assignment.output,
            identity_assignment.score,
        ):
            state = v5.append_state_token(state, "VIEWCHANGE")
        if self._is_reid_recovery(track, confidence, identity_assignment):
            state = v5.append_state_token(state, "REIDRECOVERY")
        if candidate_occluded:
            state = v5.append_state_token(state, "OCCLUSION")
        if learning_held:
            state = v5.append_state_token(state, "NOLEARN")
            for reason in learning_hold_reasons:
                state = v5.append_state_token(state, reason)
        track.state = state

        if track.kalman is None:
            track.kalman = v5.BBoxKalmanFilter(accepted)
        track.kalman.update(accepted, confidence)

        new_head = self._feature_mean_for_bbox(feature_map, accepted, frame.shape)
        if not candidate_occluded and not learning_held and self._should_refresh_head_memory(track, confidence, frame_number):
            self._refresh_track_head_bank(track, new_head, frame_number)
            self._commit_trusted_size(track, accepted)
            track.last_reliable_bbox = accepted
            track.last_reliable_frame = frame_number
            v5.record_reliable_track_trajectory(track, frame_number, accepted, self.trajectory_history_size)
            if identity_assignment is not None:
                refresh_started = time.perf_counter()
                self.identity_arbitrator.commit_track_memory(
                    track,
                    identity_assignment.output,
                    identity_assignment,
                    frame,
                )
                self._add_profile_seconds("appearance_refresh", time.perf_counter() - refresh_started)
            else:
                self._refresh_appearance(track, frame, accepted)
        elif not candidate_occluded and not learning_held:
            self._refresh_appearance(track, frame, accepted)
        self._record_size_history(track, frame_number, accepted)
        v5.record_track_trajectory(track, frame_number, accepted, self.trajectory_history_size)
        return True

    def _hold_track(
        self,
        track: v5.TrackState,
        predicted: BBox,
        confidence: float,
        margin: float,
        frame_number: int,
    ) -> None:
        previous = track.bbox
        track.previous_bbox = previous
        track.predicted_bbox = predicted
        track.raw_bbox = predicted
        track.bbox = v5.clamp_bbox_size(predicted)
        track.velocity = v5.bbox_delta(previous, track.bbox)
        track.confidence = max(0.0, min(1.0, confidence))
        track.raw_confidence = confidence
        track.assignment_score = confidence
        track.assignment_margin = margin
        track.reid_score = confidence
        track.motion_score = v5.motion_affinity(predicted, track.bbox, v5.bbox_diagonal(previous))
        track.path_score = v5.center_path_affinity(track, track.bbox)
        track.source_score = confidence
        track.assigned_source = "v7-kalman-hold"
        track.lost_frames += 1
        track.occluded_frames += 1
        track.learning_block_reason = ""
        track.occlusion_track_id = None
        track.occlusion_iou = None
        if track.kalman is not None:
            track.kalman.state[4:] *= self.occlusion_velocity_damping
        track.ok = self.occlusion_max_frames > 0 and track.lost_frames <= self.occlusion_max_frames
        track.state = v5.append_state_token("V7HEAD_MISS", "OCCLUDED") if track.ok else v5.append_state_token("V7HEAD_MISS", "LOST")
        v5.record_track_trajectory(track, frame_number, track.bbox, self.trajectory_history_size)

    def _expanded_search_bbox(self, predicted: BBox, current: BBox) -> BBox:
        center_x, center_y = v5.bbox_center(predicted)
        _, _, current_w, current_h = current
        search_w = max(current_w, predicted[2]) * self.search_radius_factor
        search_h = max(current_h, predicted[3]) * self.search_radius_factor
        return center_x - (search_w / 2.0), center_y - (search_h / 2.0), search_w, search_h

    def _bbox_to_grid_slices(self, bbox: BBox, frame_shape: Tuple[int, ...]) -> Tuple[slice, slice]:
        frame_height, frame_width = frame_shape[:2]
        x, y, w, h = v5.clamp_bbox_size(bbox)
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
            center_x, center_y = v5.bbox_center(bbox)
            grid_x = int(np.clip((center_x / max(1.0, frame_width)) * self.grid_width, 0, self.grid_width - 1))
            grid_y = int(np.clip((center_y / max(1.0, frame_height)) * self.grid_height, 0, self.grid_height - 1))
            vector = feature_map[grid_y, grid_x]
        else:
            vector = roi.reshape(-1, self.embed_dim).mean(dim=0)
        return self.F.normalize(vector.to(self.torch.float32), dim=0).detach()

    def _set_track_head_bank(self, track: v5.TrackState, bank) -> None:
        slots: List[V7TemplateMemorySlot] = []
        for index, vector in enumerate(bank[: self.lorat_memory_slots]):
            slots.append(
                V7TemplateMemorySlot(
                    vector=self.F.normalize(vector.detach().clone(), dim=0),
                    label="initial" if index == 0 else f"recent-{index}",
                    frame_number=int(track.active_template_frame or 0),
                    confidence=track.confidence,
                )
            )
        setattr(track, "v7_head_bank", slots)
        setattr(track, "v7_last_head_update_frame", track.active_template_frame)
        track.lorat_memory_slot_count = len(self._get_track_head_bank(track))

    @staticmethod
    def _get_track_head_bank(track: v5.TrackState):
        return list(getattr(track, "v7_head_bank", []))

    def _renumber_head_bank(self, bank: Sequence[V7TemplateMemorySlot]) -> List[V7TemplateMemorySlot]:
        renumbered: List[V7TemplateMemorySlot] = []
        for index, slot in enumerate(bank[: self.lorat_memory_slots]):
            renumbered.append(
                V7TemplateMemorySlot(
                    vector=slot.vector,
                    label="initial" if index == 0 else f"recent-{index}",
                    frame_number=slot.frame_number,
                    confidence=slot.confidence,
                )
            )
        return renumbered

    def _refresh_track_head_bank(self, track: v5.TrackState, vector, frame_number: int) -> None:
        vector = self.F.normalize(vector.detach().clone(), dim=0)
        bank = self._get_track_head_bank(track)
        new_slot = V7TemplateMemorySlot(
            vector=vector,
            label=f"recent-{max(1, len(bank))}",
            frame_number=frame_number,
            confidence=track.confidence,
        )
        if not bank:
            bank = [
                V7TemplateMemorySlot(
                    vector=vector,
                    label="initial",
                    frame_number=frame_number,
                    confidence=track.confidence,
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
            bank[-1] = V7TemplateMemorySlot(
                vector=updated_vector,
                label=last_slot.label,
                frame_number=frame_number,
                confidence=track.confidence,
            )
            if frame_number - int(getattr(track, "v7_last_head_update_frame", 0) or 0) >= self.lorat_memory_refresh_interval:
                bank = [bank[0], *bank[2:], new_slot]
        bank = self._renumber_head_bank(bank)
        setattr(track, "v7_head_bank", bank)
        setattr(track, "v7_last_head_update_frame", frame_number)
        track.active_template_frame = frame_number
        track.lorat_memory_slot_count = len(bank)
        track.active_lorat_slot = f"shared-head-r{track.lorat_memory_slot_count}"

    def _should_refresh_head_memory(self, track: v5.TrackState, confidence: float, frame_number: int) -> bool:
        if confidence < max(self.template_update_min_confidence, self.lorat_memory_min_score):
            return False
        if track.assignment_score is not None and track.assignment_score < self.identity_arbitrator.min_score:
            return False
        if track.reid_score is not None and track.reid_score < self.identity_arbitrator.min_reid:
            if track.motion_score is None or track.motion_score < self.identity_arbitrator.view_change_min_motion:
                return False
        if track.path_score is not None and track.path_score < self.identity_arbitrator.min_path:
            return False
        bank = self._get_track_head_bank(track)
        if len(bank) < self.lorat_memory_slots:
            return True
        last_update = int(getattr(track, "v7_last_head_update_frame", track.active_template_frame or frame_number))
        return frame_number - last_update >= self.lorat_memory_refresh_interval

    @staticmethod
    def _commit_trusted_size(track: v5.TrackState, bbox: BBox) -> None:
        track.trusted_size_bank.append(v5.clamp_bbox_size(bbox))
        if len(track.trusted_size_bank) > v5.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:
            del track.trusted_size_bank[: len(track.trusted_size_bank) - v5.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE]

    def _refresh_appearance(self, track: v5.TrackState, frame: np.ndarray, bbox: BBox) -> None:
        refresh_started = time.perf_counter()
        hist = v5.extract_reid_histogram(frame, bbox)
        try:
            if hist is None:
                return
            if track.initial_appearance_hist is None:
                track.initial_appearance_hist = hist.copy()
            if track.appearance_hist is None:
                track.appearance_hist = hist.copy()
            else:
                update_rate = 0.06
                track.appearance_hist = ((1.0 - update_rate) * track.appearance_hist) + (update_rate * hist)
                norm = float(np.linalg.norm(track.appearance_hist))
                if norm > 0:
                    track.appearance_hist /= norm
            if not track.appearance_bank or v5.histogram_similarity(track.appearance_bank[-1], hist) < 0.985:
                track.appearance_bank.append(hist.copy())
                if len(track.appearance_bank) > 12:
                    del track.appearance_bank[: len(track.appearance_bank) - 12]
            track.appearance_updates += 1
        finally:
            self._add_profile_seconds("appearance_refresh", time.perf_counter() - refresh_started)

    @staticmethod
    def _record_size_history(track: v5.TrackState, frame_number: int, bbox: BBox) -> None:
        track.size_history.append((frame_number, v5.clamp_bbox_size(bbox)))
        if len(track.size_history) > v5.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE:
            del track.size_history[: len(track.size_history) - v5.DEFAULT_LORAT_SIZE_MEMORY_BANK_SIZE]

    def _update_gpu_status(self) -> None:
        if self.device.type != "cuda":
            return
        allocated = self.torch.cuda.memory_allocated(self.device)
        reserved = self.torch.cuda.memory_reserved(self.device)
        peak_allocated = self.torch.cuda.max_memory_allocated(self.device)
        peak_reserved = self.torch.cuda.max_memory_reserved(self.device)
        self.runtime_status.gpu_allocated_mb = v5.bytes_to_mb(allocated)
        self.runtime_status.gpu_reserved_mb = v5.bytes_to_mb(reserved)
        self.runtime_status.gpu_peak_allocated_mb = v5.bytes_to_mb(peak_allocated)
        self.runtime_status.gpu_peak_reserved_mb = v5.bytes_to_mb(peak_reserved)

    def status_lines(self) -> List[str]:
        status = self.runtime_status_snapshot()
        lines = [
            f"FPS {status.fps:.2f} | objects {status.active_objects} | mode {V7_EXECUTION_MODE}",
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

    def runtime_status_snapshot(self) -> v5.RuntimeStatus:
        status = copy.copy(self.runtime_status)
        status.gating_decisions = self.v7_gating_decisions
        status.gating_primary_decisions = self.v7_primary_decisions
        status.gating_recovery_decisions = self.v7_recovery_decisions
        status.gating_selected_slot_items = self.v7_selected_head_items
        status.gating_avg_slots_per_decision = (
            self.v7_selected_head_items / self.v7_gating_decisions
            if self.v7_gating_decisions
            else 0.0
        )
        status.gating_recovery_reasons = ",".join(
            f"{reason}:{count}"
            for reason, count in sorted(self.v7_recovery_reason_counts.items())
        )
        for bucket in V7_PROFILE_BUCKETS:
            setattr(status, f"v7_profile_{bucket}_seconds", self._profile_total_seconds.get(bucket, 0.0))
            setattr(status, f"v7_profile_{bucket}_ms_per_update", self._profile_total_ms_per_update(bucket))
            setattr(status, f"v7_last_{bucket}_seconds", self._last_profile_seconds.get(bucket, 0.0))
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
            "Version 7 LoRAT-backed multi-object tracker with one shared frame ViT pass "
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
    parser.add_argument("--lorat-root", type=Path, default=v5.DEFAULT_LORAT_ROOT, help="Local LoRAT checkout.")
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v5.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT weight. Defaults from --lorat-config.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Optional track cap. 0 means no cap.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable LoRAT automatic mixed precision.")
    parser.add_argument(
        "--v7-frame-size",
        type=int,
        default=0,
        help="Optional square shared-frame tensor size override; must match the selected LoRAT config size.",
    )
    parser.add_argument(
        "--v7-head-rank",
        type=int,
        default=v5.DEFAULT_LORAT_MEMORY_SLOTS,
        help="Maximum per-object low-rank heads scored from the v7 head bank.",
    )
    parser.add_argument("--v7-head-hidden-dim", type=int, default=256, help="Hidden dimension for the trainable v7 LoRA object head.")
    parser.add_argument("--v7-head-lora-rank", type=int, default=16, help="Low-rank adapter dimension for the trainable v7 object head.")
    parser.add_argument("--v7-head-weights", type=Path, help="Optional trained v7 object-head checkpoint.")
    parser.add_argument(
        "--v7-search-radius-factor",
        type=float,
        default=2.25,
        help="Search window size as a multiple of the current box size on the shared feature grid.",
    )
    parser.add_argument("--v7-min-confidence", type=float, default=0.48, help="Minimum head score to accept an update.")
    parser.add_argument(
        "--v7-template-update-rate",
        type=float,
        default=0.08,
        help="EMA rate for refreshing the newest per-object head vector.",
    )
    parser.add_argument(
        "--v7-template-update-min-confidence",
        type=float,
        default=0.58,
        help="Minimum confidence before a new ROI feature can refresh an object head.",
    )
    parser.add_argument(
        "--v7-score-reduction",
        choices=("max", "mean"),
        default="max",
        help="How to reduce a per-object head bank to one response map.",
    )
    parser.add_argument("--lorat-memory-slots", type=int, default=v5.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--lorat-memory-refresh-interval", type=int, default=v5.DEFAULT_LORAT_MEMORY_REFRESH_INTERVAL)
    parser.add_argument("--lorat-memory-min-score", type=float, default=0.55)
    parser.add_argument("--lorat-accept-min-score", type=float, default=v5.DEFAULT_LORAT_ACCEPT_MIN_SCORE)
    parser.add_argument(
        "--fixed-lorat-box-size",
        dest="lorat_fixed_box_size",
        action="store_true",
        default=v5.DEFAULT_LORAT_FIXED_BOX_SIZE,
    )
    parser.add_argument("--allow-lorat-size-change", dest="lorat_fixed_box_size", action="store_false")
    parser.add_argument("--lorat-min-box-area", type=float, default=v5.DEFAULT_LORAT_MIN_BOX_AREA)
    parser.add_argument("--lorat-max-area-change-per-frame", type=float, default=v5.DEFAULT_LORAT_MAX_AREA_CHANGE_PER_FRAME)
    parser.add_argument("--lorat-trusted-size-floor-scale", type=float, default=v5.DEFAULT_LORAT_TRUSTED_SIZE_FLOOR_SCALE)
    parser.add_argument("--shrink-guard-window", type=int, default=v5.DEFAULT_SHRINK_GUARD_WINDOW)
    parser.add_argument("--shrink-guard-area-ratio", type=float, default=v5.DEFAULT_SHRINK_GUARD_AREA_RATIO)
    parser.add_argument("--shrink-guard-step-ratio", type=float, default=v5.DEFAULT_SHRINK_GUARD_STEP_RATIO)
    parser.add_argument("--shrink-guard-min-confidence", type=float, default=v5.DEFAULT_SHRINK_GUARD_MIN_CONFIDENCE)
    parser.add_argument("--shrink-guard-min-reid", type=float, default=v5.DEFAULT_SHRINK_GUARD_MIN_REID)
    parser.add_argument("--crop-information-min-score", type=float, default=v5.DEFAULT_CROP_INFORMATION_MIN_SCORE)
    parser.add_argument("--crop-information-min-pixels", type=int, default=v5.DEFAULT_CROP_INFORMATION_MIN_PIXELS)
    parser.add_argument("--disable-identity-arbitration", action="store_true")
    parser.add_argument("--identity-min-score", type=float, default=v5.DEFAULT_IDENTITY_MIN_SCORE)
    parser.add_argument("--identity-min-reid", type=float, default=v5.DEFAULT_IDENTITY_MIN_REID)
    parser.add_argument("--identity-min-motion", type=float, default=v5.DEFAULT_IDENTITY_MIN_MOTION)
    parser.add_argument("--identity-min-path", type=float, default=v5.DEFAULT_IDENTITY_MIN_PATH)
    parser.add_argument("--identity-bank-size", type=int, default=12)
    parser.add_argument("--identity-memory-min-confidence", type=float, default=v5.DEFAULT_IDENTITY_MEMORY_MIN_CONFIDENCE)
    parser.add_argument("--occlusion-max-frames", type=int, default=v5.DEFAULT_OCCLUSION_MAX_FRAMES)
    parser.add_argument("--occlusion-iou-threshold", type=float, default=v5.DEFAULT_OCCLUSION_IOU_THRESHOLD)
    parser.add_argument("--occlusion-velocity-damping", type=float, default=v5.DEFAULT_OCCLUSION_VELOCITY_DAMPING)
    parser.add_argument("--reid-recovery-min-score", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_SCORE)
    parser.add_argument("--reid-recovery-min-reid", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_REID)
    parser.add_argument("--reid-recovery-min-motion", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_MOTION)
    parser.add_argument("--reid-recovery-min-confidence", type=float, default=v5.DEFAULT_REID_RECOVERY_MIN_CONFIDENCE)
    parser.add_argument("--view-change-min-score", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_SCORE)
    parser.add_argument("--view-change-min-motion", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_MOTION)
    parser.add_argument("--view-change-min-confidence", type=float, default=v5.DEFAULT_VIEW_CHANGE_MIN_CONFIDENCE)
    parser.add_argument("--view-change-max-lost-frames", type=int, default=v5.DEFAULT_VIEW_CHANGE_MAX_LOST_FRAMES)
    parser.add_argument("--v7-primary-heads-per-track", type=int, default=DEFAULT_V7_PRIMARY_HEADS_PER_TRACK)
    parser.add_argument("--v7-recovery-heads-per-track", type=int, default=DEFAULT_V7_RECOVERY_HEADS_PER_TRACK)
    parser.add_argument("--v7-recovery-interval", type=int, default=DEFAULT_V7_RECOVERY_INTERVAL)
    parser.add_argument("--v7-recovery-min-confidence", type=float, default=DEFAULT_V7_RECOVERY_MIN_CONFIDENCE)
    parser.add_argument("--v7-recovery-min-assignment-score", type=float, default=DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_SCORE)
    parser.add_argument("--v7-recovery-min-assignment-margin", type=float, default=DEFAULT_V7_RECOVERY_MIN_ASSIGNMENT_MARGIN)
    parser.add_argument("--v7-recovery-stale-head-frames", type=int, default=DEFAULT_V7_RECOVERY_STALE_HEAD_FRAMES)
    parser.add_argument("--output", type=Path, help="MOTChallenge-format result file.")
    parser.add_argument("--save-video", type=Path, help="Annotated MP4 output path.")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated MP4 writing.")
    parser.add_argument("--debug-log", type=Path, help="Tracking debug CSV output path.")
    parser.add_argument("--slot-debug-log", type=Path, help="V7 head-bank debug CSV output path.")
    parser.add_argument("--no-slot-debug-log", action="store_true", help="Disable V7 head-bank debug CSV writing.")
    parser.add_argument("--week2-proof-log", type=Path, help="Week 2 shared-backbone proof CSV output path.")
    parser.add_argument("--no-week2-proof-log", action="store_true", help="Disable Week 2 proof CSV writing.")
    parser.add_argument("--debug-frame-start", type=int, default=0, help="First frame to include in --debug-log; 0 means all.")
    parser.add_argument("--debug-frame-end", type=int, default=0, help="Last frame to include in --debug-log; 0 means all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for smoke tests.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow; requires --initial-boxes.")
    return parser.parse_args()


def create_backend(args: argparse.Namespace, source: v5.FrameSource, expected_tracks: int = 0):
    weight_path = args.weight_path or v5.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return SharedFrameLoRATMultiObjectTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        args.max_tracks,
        source.fps,
        source.length,
        source.name,
        args.disable_amp,
        args.v7_frame_size,
        args.v7_head_rank,
        args.v7_head_hidden_dim,
        args.v7_head_lora_rank,
        args.v7_head_weights,
        args.v7_search_radius_factor,
        args.v7_min_confidence,
        args.v7_template_update_rate,
        args.v7_template_update_min_confidence,
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
        args.v7_primary_heads_per_track,
        args.v7_recovery_heads_per_track,
        args.v7_recovery_interval,
        args.v7_recovery_min_confidence,
        args.v7_recovery_min_assignment_score,
        args.v7_recovery_min_assignment_margin,
        args.v7_recovery_stale_head_frames,
        args.v7_score_reduction,
        not getattr(args, "no_slot_debug_log", False),
    )


def default_output_path(source_name: str) -> Path:
    return v5.default_output_path(source_name, "lorat_v7")


def default_debug_log_path(source_name: str) -> Path:
    return v5.default_debug_log_path(source_name, "lorat_v7")


def default_week2_proof_log_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return v5.DEFAULT_DEBUG_DIR / f"{safe_name}_lorat_v7_week2_proof.csv"


def write_week2_proof_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WEEK2_PROOF_LOG_HEADER + "".join(lines), encoding="utf-8")


def default_video_path(source_name: str) -> Path:
    return v5.default_video_path(source_name, "lorat_v7")


def main() -> int:
    args = parse_args()
    frame_source = v5.open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name)
    debug_log_path = args.debug_log or default_debug_log_path(frame_source.name)
    slot_debug_log_path = (
        None
        if args.no_slot_debug_log
        else (args.slot_debug_log or v5.default_slot_debug_log_path(frame_source.name, "lorat_v7"))
    )
    week2_proof_log_path = (
        None
        if args.no_week2_proof_log
        else (args.week2_proof_log or default_week2_proof_log_path(frame_source.name))
    )
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
        boxes = v5.select_boxes(first_frame)
    if not boxes:
        print("No bounding boxes selected. Exiting.")
        frame_source.release()
        cv2.destroyAllWindows()
        return 0

    backend = create_backend(args, frame_source, len(boxes))
    writer = v5.make_video_writer(save_video_path, frame_source.fps, first_frame) if save_video_path is not None else None
    mot_lines: List[str] = []
    debug_lines: List[str] = []
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
        v5.write_debug_log(debug_log_path, debug_lines)
        print(f"Wrote debug CSV to: {debug_log_path}")
        if slot_debug_log_path is not None:
            v5.write_slot_debug_log(slot_debug_log_path, backend.slot_debug_lines)
            print(f"Wrote V7 head-bank debug CSV to: {slot_debug_log_path}")
        if week2_proof_log_path is not None:
            write_week2_proof_log(week2_proof_log_path, backend.week2_proof_lines)
            print(f"Wrote Week 2 shared-backbone proof CSV to: {week2_proof_log_path}")
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        v5.append_mot_results(mot_lines, frame_number, backend.tracks)
        v5.append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
        if writer is not None:
            writer.write(v5.draw_tracks(first_frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines()))

        while True:
            if not paused:
                ok, frame = frame_source.read()
                if not ok or frame is None:
                    break
                last_frame = frame
                frame_number += 1
                backend.update(frame, frame_number)
                v5.append_mot_results(mot_lines, frame_number, backend.tracks)
                v5.append_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
            else:
                frame = last_frame.copy()

            shown = v5.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name, backend.status_lines())
            if writer is not None and not paused:
                writer.write(shown)

            if not args.no_display:
                cv2.imshow("LoRAT Multi-Object Tracker v7", shown)
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("a"):
                    paused = True
                    new_boxes = v5.select_boxes(frame, "Add Objects")
                    if new_boxes:
                        added_tracks = backend.add_tracks(frame, new_boxes, frame_number)
                        v5.append_mot_results(mot_lines, frame_number, added_tracks)
                        v5.append_debug_rows(
                            debug_lines,
                            frame_number,
                            backend.tracks,
                            args.debug_frame_start,
                            args.debug_frame_end,
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
