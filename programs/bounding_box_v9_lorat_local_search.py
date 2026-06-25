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

import bounding_box_v8_lorat_quality_batched as v8
import mot_common as mot

BBox = mot.BBox

V9_EXECUTION_MODE = "shared-frame-vit-batched-local-search-roi-heads"
V8_EXECUTION_MODE = V9_EXECUTION_MODE
DEFAULT_V9_LOCAL_GRID_SIZE = 16


def __getattr__(name: str):
    return getattr(v8, name)


@dataclass(frozen=True)
class V9LocalHeadOutput:
    score_maps: object
    box_delta_maps: object
    search_windows: Tuple[BBox, ...]
    elapsed_seconds: float
    selected_head_count: int


class V9LocalSearchHead(v8.BatchedObjectConditionedHead):
    """Object-conditioned head evaluated on per-track local search grids.

    V8 scores every object over the whole shared frame feature map, then masks
    candidate extraction to a local ROI. V9 moves the local search window before
    the head: the shared frame features are sampled into a fixed-size local grid
    for each track, and the batched head predicts boxes in that local coordinate
    system. This is the first step back toward LoRaT's template/search geometry
    while preserving one shared frame ViT pass.
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
                return score_logits, box_deltas

            def project_reid(self, embeddings):
                return torch_module.nn.functional.normalize(self.reid_projection(embeddings), dim=-1)

        return V9LocalObjectConditionedLoRAHead()

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
                per_head_scores, per_head_deltas = self.module(
                    flat_features,
                    flat_head_tensor,
                    flat_template_tensor,
                    flat_template_mask,
                    flat_foreground_mask,
                )
                per_head_scores = per_head_scores.reshape(object_count, max_heads, grid_height * grid_width)
                per_head_deltas = per_head_deltas.reshape(object_count, max_heads, grid_height * grid_width, 4)

            if object_count <= 0:
                score_maps = self.torch.zeros((0, grid_height, grid_width), device=self.device, dtype=self.torch.float32)
                box_delta_maps = self.torch.zeros((0, grid_height, grid_width, 4), device=self.device, dtype=self.torch.float32)
            else:
                score_logits, box_deltas = self._reduce_head_scores_and_deltas(per_head_scores, per_head_deltas, head_mask)
                score_maps = score_logits.reshape(object_count, grid_height, grid_width)
                box_delta_maps = box_deltas.reshape(object_count, grid_height, grid_width, 4)

        return V9LocalHeadOutput(
            score_maps=score_maps,
            box_delta_maps=box_delta_maps,
            search_windows=tuple(search_windows),
            elapsed_seconds=time.perf_counter() - started,
            selected_head_count=selected_head_count,
        )


class V9LocalSearchLoRATTracker(v8.V8QualityBatchedLoRATTracker):
    """V9 tracker: V8 lifecycle plus LoRaT-style local search grids.

    The expensive frame encoder remains shared. The per-object work is a batched
    local-search head over fixed-size feature grids sampled from that shared map.
    """

    backend_name = "LoRAT-v9-local-search-roi"

    def __init__(self, *args, v9_local_grid_size: int = DEFAULT_V9_LOCAL_GRID_SIZE, **kwargs) -> None:
        self.v9_local_grid_size = max(4, int(v9_local_grid_size))
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
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 3, 1).contiguous()
        return self.F.normalize(sampled.to(self.torch.float32), dim=-1)

    def _search_windows_for_tracks(
        self,
        predicted_bboxes: Sequence[BBox],
        tracks: Sequence[mot.TrackState],
    ) -> List[BBox]:
        windows: List[BBox] = []
        for predicted, track in zip(predicted_bboxes, tracks):
            reference = self._template_match_reference_size(track)
            windows.append(mot.clamp_bbox_size(self._expanded_search_bbox(predicted, track.bbox, reference)))
        return windows

    def _candidates_from_local_head_output(
        self,
        score_maps,
        box_delta_maps,
        search_windows: Sequence[BBox],
    ) -> List[v8.V8HeadCandidateInfo]:
        candidate_started = time.perf_counter()
        object_count = int(score_maps.shape[0])
        if object_count <= 0:
            return []

        grid_height = int(score_maps.shape[1])
        grid_width = int(score_maps.shape[2])
        score_maps_float = score_maps.to(self.torch.float32)
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

        candidates: List[v8.V8HeadCandidateInfo] = []
        roi_tokens = grid_width * grid_height
        for row_index, row in enumerate(packed):
            confidence, margin, grid_x_value, grid_y_value, bbox_x, bbox_y, bbox_w, bbox_h = row
            top_candidates: List[v8.V8HeadCandidate] = []
            for rank, top_row in enumerate(top_packed[row_index] if top_packed else [], start=1):
                top_confidence, top_grid_x_value, top_grid_y_value, top_x, top_y, top_w, top_h, top_finite = top_row
                if float(top_finite) <= 0.0:
                    continue
                top_candidates.append(
                    v8.V8HeadCandidate(
                        rank=rank,
                        bbox=mot.clamp_bbox_size((top_x, top_y, top_w, top_h)),
                        confidence=float(top_confidence),
                        grid_x=int(np.clip(round(float(top_grid_x_value)), 0, max(0, grid_width - 1))),
                        grid_y=int(np.clip(round(float(top_grid_y_value)), 0, max(0, grid_height - 1))),
                    )
                )
            candidates.append(
                v8.V8HeadCandidateInfo(
                    bbox=mot.clamp_bbox_size((bbox_x, bbox_y, bbox_w, bbox_h)),
                    confidence=float(confidence),
                    margin=max(0.0, float(margin)),
                    roi_tokens=roi_tokens,
                    top_candidates=tuple(top_candidates),
                )
            )

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

    def _score_and_update_tracks(
        self,
        frame: np.ndarray,
        feature_map,
        tracks: Sequence[mot.TrackState],
        frame_number: int,
    ) -> None:
        selected_banks = [self._select_track_heads(track, frame_number) for track in tracks]
        predicted_bboxes = [self._predict_track(track) for track in tracks]
        search_windows = self._search_windows_for_tracks(predicted_bboxes, tracks)
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

    def status_lines(self) -> List[str]:
        lines = super().status_lines()
        if lines:
            lines[0] = lines[0].replace(v8.V8_EXECUTION_MODE, V9_EXECUTION_MODE)
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
        getattr(args, "v8_small_target_mode", v8.DEFAULT_V8_SMALL_TARGET_MODE),
        getattr(args, "v8_small_target_area", v8.DEFAULT_V8_SMALL_TARGET_AREA),
        getattr(args, "v8_small_target_max_side", v8.DEFAULT_V8_SMALL_TARGET_MAX_SIDE),
        getattr(args, "v8_small_target_max_scale_change", v8.DEFAULT_V8_SMALL_TARGET_MAX_SCALE_CHANGE),
        getattr(args, "v8_small_target_template_min_score", v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_SCORE),
        getattr(args, "v8_small_target_template_min_motion", v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_MOTION),
        getattr(args, "v8_small_target_template_min_path", v8.DEFAULT_V8_SMALL_TARGET_TEMPLATE_MIN_PATH),
        getattr(args, "v8_small_target_confidence_floor", v8.DEFAULT_V8_SMALL_TARGET_CONFIDENCE_FLOOR),
        v9_local_grid_size=getattr(args, "v9_local_grid_size", DEFAULT_V9_LOCAL_GRID_SIZE),
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v9-local-grid-size", type=int, default=DEFAULT_V9_LOCAL_GRID_SIZE)
    local_args, remaining = parser.parse_known_args()
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *remaining]
        args = v8.parse_args()
    finally:
        sys.argv = original_argv
    args.v9_local_grid_size = max(4, int(local_args.v9_local_grid_size))
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

    boxes = v8.parse_initial_boxes(args.initial_boxes)
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
        v8.write_v8_debug_log(debug_log_path, debug_lines)
        print(f"Wrote debug CSV to: {debug_log_path}")
        if slot_debug_log_path is not None:
            mot.write_slot_debug_log(slot_debug_log_path, backend.slot_debug_lines)
            print(f"Wrote V9 head-bank debug CSV to: {slot_debug_log_path}")
        if week2_proof_log_path is not None:
            v8.write_week2_proof_log(week2_proof_log_path, backend.week2_proof_lines)
            print(f"Wrote shared-backbone proof CSV to: {week2_proof_log_path}")
        if manual_event_log_path is not None:
            mot.write_manual_event_csv(manual_event_log_path, manual_events)
            print(f"Wrote manual event CSV to: {manual_event_log_path}")
        outputs_written = True

    try:
        backend.initialize(first_frame, boxes, frame_number)
        mot.append_mot_results(mot_lines, frame_number, backend.tracks)
        v8.append_v8_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
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
                v8.append_v8_debug_rows(debug_lines, frame_number, backend.tracks, args.debug_frame_start, args.debug_frame_end)
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
                        v8.append_v8_debug_rows(
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
                            v8.append_v8_debug_rows(
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
