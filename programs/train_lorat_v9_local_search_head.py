"""Train the V9 LoRaT-MOT local-search head.

V9 keeps the Week 2 shared-frame property, but moves the supervised head back
toward LoRaT's template/search geometry:

* one frozen LoRaT/DINOv2 frame encoder pass per frame;
* one fixed local search grid sampled from that shared feature map per object;
* one batched object-conditioned LoRA head call across all objects;
* local l/t/r/b box targets normalized by each object's search window.

The dataset construction intentionally reuses the shared training adapters for
DanceTrack/MOT-style data and TAO/TAO-OW JSON annotations. The part that changes
is the head path and target coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import bounding_box_v9_lorat_local_search as v9
import exercise_lorat_mot as exercise
import mot_common as mot
import train_lorat_v9_data_common as train_common

BBox = mot.BBox

SOURCE_WEIGHT_ALIASES = {
    "dance": "MOTFrameHeadDataset",
    "dancetrack": "MOTFrameHeadDataset",
    "mot": "MOTFrameHeadDataset",
    "mot17": "MOTFrameHeadDataset",
    "tao": "TAOFrameHeadDataset",
    "tao-ow": "TAOFrameHeadDataset",
    "tao_ow": "TAOFrameHeadDataset",
    "lasot": "LaSOTFrameHeadDataset",
}


@dataclass(frozen=True)
class V9LocalSearchTargets:
    score_labels: torch.Tensor
    visibility_labels: torch.Tensor
    positive_mask: torch.Tensor
    hard_negative_mask: torch.Tensor
    loss_mask: torch.Tensor
    visibility_loss_mask: torch.Tensor
    visibility_weights: torch.Tensor
    positive_weights: torch.Tensor
    ltrb_targets: torch.Tensor
    search_windows: torch.Tensor
    target_boxes_xywh: torch.Tensor
    target_boxes_xyxy: torch.Tensor
    present_mask: torch.Tensor
    positive_cells: int
    hard_negative_cells: int
    loss_cells: int
    missing_targets: int


def tensor_from_boxes(boxes: Sequence[BBox], device: torch.device) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 4), device=device, dtype=torch.float32)
    return torch.tensor(boxes, device=device, dtype=torch.float32)


def xywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2].clamp_min(1.0)
    y2 = boxes[:, 1] + boxes[:, 3].clamp_min(1.0)
    return torch.stack((x1, y1, x2, y2), dim=1)


def local_search_window_for_box(
    previous_box: BBox,
    target_box: Optional[BBox] = None,
    *,
    radius_factor: float = 2.25,
    min_size: float = 4.0,
) -> BBox:
    """Build a LoRaT-style local search window around a selected target."""

    px, py, pw, ph = mot.clamp_bbox_size(previous_box)
    center_x, center_y = mot.bbox_center((px, py, pw, ph))
    ref_w = max(float(min_size), pw)
    ref_h = max(float(min_size), ph)
    if target_box is not None:
        tx, ty, tw, th = mot.clamp_bbox_size(target_box)
        target_center_x, target_center_y = mot.bbox_center((tx, ty, tw, th))
        ref_w = max(ref_w, tw, abs(target_center_x - center_x) * 2.0 + tw)
        ref_h = max(ref_h, th, abs(target_center_y - center_y) * 2.0 + th)
    search_w = max(float(min_size), ref_w * max(1.0, float(radius_factor)))
    search_h = max(float(min_size), ref_h * max(1.0, float(radius_factor)))
    return center_x - (search_w / 2.0), center_y - (search_h / 2.0), search_w, search_h


def training_search_window(item: train_common.TrainingObject, radius_factor: float) -> BBox:
    """Use the shared adapter's LoRaT-style search box when available.

    The shared data adapter already samples search windows from first/previous/mixed
    templates, applies jitter, and repairs windows to cover the selected target.
    For V9 we preserve that sampling instead of recreating it later.
    """

    if item.search_bbox is not None:
        return mot.clamp_bbox_size(item.search_bbox)
    target = item.current_bbox if item.is_present else None
    return local_search_window_for_box(item.previous_bbox, target, radius_factor=radius_factor)


def _bbox_center_outside(outer: BBox, inner: BBox) -> bool:
    x, y, width, height = mot.clamp_bbox_size(outer)
    center_x, center_y = mot.bbox_center(inner)
    return not (x <= center_x <= x + width and y <= center_y <= y + height)


def _shifted_drift_candidate(
    target_bbox: BBox,
    frame_shape: Tuple[int, ...],
    direction: Tuple[float, float],
    shift_factor: float,
) -> BBox:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = mot.clamp_bbox_size(target_bbox)
    center_x, center_y = mot.bbox_center((x, y, width, height))
    diagonal = max(8.0, float(np.hypot(width, height)))
    dx = float(direction[0]) * max(width * float(shift_factor), diagonal)
    dy = float(direction[1]) * max(height * float(shift_factor), diagonal)
    shifted = (center_x + dx - width * 0.5, center_y + dy - height * 0.5, width, height)
    return train_common.clamp_bbox_to_frame_shape(shifted, frame_shape)


def _farthest_same_size_candidate(target_bbox: BBox, frame_shape: Tuple[int, ...]) -> BBox:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = mot.clamp_bbox_size(target_bbox)
    target_center = mot.bbox_center((x, y, width, height))
    corners = (
        (0.0, 0.0, width, height),
        (max(0.0, float(frame_width) - width), 0.0, width, height),
        (0.0, max(0.0, float(frame_height) - height), width, height),
        (
            max(0.0, float(frame_width) - width),
            max(0.0, float(frame_height) - height),
            width,
            height,
        ),
    )
    return max(
        corners,
        key=lambda candidate: float(np.hypot(
            mot.bbox_center(candidate)[0] - target_center[0],
            mot.bbox_center(candidate)[1] - target_center[1],
        )),
    )


def add_drift_negative_training_objects(
    objects: Sequence[train_common.TrainingObject],
    frame_shape: Tuple[int, ...],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[train_common.TrainingObject], int]:
    """Add selected-template/search-miss examples that mirror V9 drift failures.

    These are not generic hard negatives. They preserve the selected object's
    immutable template, but put the local search crop on a distractor or a
    shifted-off-target region and mark the selected target absent. That teaches
    the visibility/objectness heads that a plausible local object is not enough
    when the selected target has left the search window.
    """

    probability = max(0.0, min(1.0, float(getattr(args, "drift_negative_probability", 0.0))))
    max_added = max(0, int(getattr(args, "drift_negative_max_per_frame", 0)))
    if probability <= 0.0 or max_added <= 0:
        return list(objects), 0

    added: List[train_common.TrainingObject] = []
    directions: Tuple[Tuple[float, float], ...] = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    )
    shift_factor = max(1.0, float(getattr(args, "drift_negative_shift_factor", 3.0)))
    scale_jitter = min(float(args.search_scale_jitter), 0.10)
    translation_jitter = min(float(args.search_translation_jitter), 0.50)
    jitter_amount = min(max(0.0, float(args.previous_box_jitter)), 0.08)

    for item in objects:
        if len(added) >= max_added:
            break
        if not item.is_present or float(rng.random()) > probability:
            continue

        candidate_bboxes: List[BBox] = [mot.clamp_bbox_size(box) for box in item.distractor_bboxes]
        shuffled_directions = [directions[int(index)] for index in rng.permutation(len(directions))]
        candidate_bboxes.extend(
            _shifted_drift_candidate(item.current_bbox, frame_shape, direction, shift_factor)
            for direction in shuffled_directions
        )
        candidate_bboxes.append(_farthest_same_size_candidate(item.current_bbox, frame_shape))

        chosen_anchor: Optional[BBox] = None
        chosen_search: Optional[BBox] = None
        chosen_distractor: Optional[BBox] = None
        for candidate_bbox in candidate_bboxes:
            candidate_bbox = train_common.clamp_bbox_to_frame_shape(candidate_bbox, frame_shape)
            for _ in range(4):
                wrong_anchor = train_common.jitter_reference_bbox(candidate_bbox, rng, jitter_amount)
                wrong_anchor = train_common.clamp_bbox_to_frame_shape(wrong_anchor, frame_shape)
                search_bbox = train_common.siamfc_search_bbox(
                    wrong_anchor,
                    (224, 224),
                    args.search_area_factor,
                    scale_jitter,
                    translation_jitter,
                    args.search_min_object_size,
                    rng,
                    retry_count=2,
                )
                if _bbox_center_outside(search_bbox, item.current_bbox):
                    chosen_anchor = wrong_anchor
                    chosen_search = search_bbox
                    chosen_distractor = candidate_bbox
                    break
            if chosen_anchor is not None:
                break

        if chosen_anchor is None or chosen_search is None or chosen_distractor is None:
            continue

        added.append(
            replace(
                item,
                previous_bbox=chosen_anchor,
                previous_context_bbox=train_common.siamfc_context_bbox(chosen_anchor, args.template_area_factor),
                search_bbox=chosen_search,
                target_kind=f"{item.target_kind}:drift_absent",
                is_present=False,
                distractor_bboxes=(chosen_distractor,) + tuple(item.distractor_bboxes),
            )
        )

    if not added:
        return list(objects), 0
    return list(objects) + added, len(added)


def make_v9_local_search_targets(
    target_boxes: Sequence[BBox],
    search_windows: Sequence[BBox],
    *,
    present: Optional[Sequence[bool]] = None,
    distractor_bboxes: Optional[Sequence[Sequence[BBox]]] = None,
    target_kinds: Optional[Sequence[str]] = None,
    grid_size: int = v9.DEFAULT_V9_LOCAL_GRID_SIZE,
    device: Optional[torch.device] = None,
    positive_radius_cells: float = 1.5,
    center_positive_weight: float = 0.5,
    small_target_loss_weight: float = 2.5,
    small_target_area_threshold: float = 128.0,
    small_target_max_side: float = 18.0,
) -> V9LocalSearchTargets:
    """Create score/ltrb targets in V9 local-search coordinates."""

    device = device or torch.device("cpu")
    grid_size = max(2, int(grid_size))
    boxes = tensor_from_boxes(target_boxes, device)
    windows = tensor_from_boxes(search_windows, device)
    if boxes.shape[0] != windows.shape[0]:
        raise ValueError("target_boxes and search_windows must have the same length.")
    object_count = int(boxes.shape[0])
    score_labels = torch.zeros((object_count, grid_size, grid_size), device=device, dtype=torch.float32)
    visibility_labels = torch.zeros_like(score_labels)
    positive_mask = torch.zeros_like(score_labels, dtype=torch.bool)
    hard_negative_mask = torch.zeros_like(positive_mask)
    loss_mask = torch.ones_like(positive_mask)
    visibility_loss_mask = torch.ones_like(positive_mask)
    visibility_weights = torch.ones_like(score_labels, dtype=torch.float32)
    positive_weights = torch.ones_like(score_labels, dtype=torch.float32)
    ltrb_targets = torch.zeros((object_count, grid_size, grid_size, 4), device=device, dtype=torch.float32)
    present_mask = torch.tensor(
        [True] * object_count if present is None else [bool(value) for value in present],
        device=device,
        dtype=torch.bool,
    )
    target_boxes_xyxy = xywh_to_xyxy_tensor(boxes)
    if object_count == 0:
        return V9LocalSearchTargets(
            score_labels,
            visibility_labels,
            positive_mask,
            hard_negative_mask,
            loss_mask,
            visibility_loss_mask,
            visibility_weights,
            positive_weights,
            ltrb_targets,
            windows,
            boxes,
            target_boxes_xyxy,
            present_mask,
            0,
            0,
            0,
            0,
        )

    xs = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    ys = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    window_w = windows[:, None, None, 2].clamp_min(1.0)
    window_h = windows[:, None, None, 3].clamp_min(1.0)
    center_x = windows[:, None, None, 0] + xx[None, :, :] * window_w
    center_y = windows[:, None, None, 1] + yy[None, :, :] * window_h

    target_x1 = boxes[:, None, None, 0]
    target_y1 = boxes[:, None, None, 1]
    target_x2 = target_x1 + boxes[:, None, None, 2].clamp_min(1.0)
    target_y2 = target_y1 + boxes[:, None, None, 3].clamp_min(1.0)
    inside = (center_x >= target_x1) & (center_x <= target_x2) & (center_y >= target_y1) & (center_y <= target_y2)
    inside &= present_mask[:, None, None]

    target_center_x = (target_x1 + target_x2) * 0.5
    target_center_y = (target_y1 + target_y2) * 0.5
    cell_w = window_w / float(grid_size)
    cell_h = window_h / float(grid_size)
    center_distance = torch.sqrt(
        ((center_x - target_center_x) / cell_w.clamp_min(1.0e-6)) ** 2
        + ((center_y - target_center_y) / cell_h.clamp_min(1.0e-6)) ** 2
    )
    positive_mask = inside | ((center_distance <= float(positive_radius_cells)) & present_mask[:, None, None])

    flat_positive_count = positive_mask.reshape(object_count, -1).sum(dim=1)
    missing_positive = (flat_positive_count == 0) & present_mask
    if bool(missing_positive.any().item()):
        nearest = torch.argmin(center_distance.reshape(object_count, -1), dim=1)
        flat_mask = positive_mask.reshape(object_count, -1)
        rows = torch.arange(object_count, device=device)[missing_positive]
        flat_mask[rows, nearest[missing_positive]] = True
        positive_mask = flat_mask.reshape(object_count, grid_size, grid_size)

    score_labels = positive_mask.to(torch.float32)
    visibility_labels = score_labels.clone()
    gaussian = torch.exp(-center_distance.pow(2) / (2.0 * max(0.1, float(positive_radius_cells)) ** 2))
    gaussian = gaussian / gaussian.reshape(object_count, -1).amax(dim=1).clamp_min(1e-6)[:, None, None]
    blend = max(0.0, min(1.0, float(center_positive_weight)))
    positive_weights = (1.0 - blend) + (blend * gaussian)

    target_kinds = list(target_kinds or ["full"] * object_count)
    for index, (bbox, kind) in enumerate(zip(target_boxes, target_kinds)):
        if not bool(present_mask[index].item()):
            continue
        probe = train_common.TrainingObject(
            track_id=index,
            current_bbox=bbox,
            previous_bbox=bbox,
            previous_frame=0,
            previous_context_bbox=bbox,
            template_frame=0,
            template_bbox=bbox,
            template_context_bbox=bbox,
            search_bbox=search_windows[index],
            target_kind=kind,
        )
        if train_common.is_small_target_object(probe, small_target_area_threshold, small_target_max_side):
            positive_weights[index] *= max(1.0, float(small_target_loss_weight))

    ltrb_targets[..., 0] = (center_x - target_x1) / window_w
    ltrb_targets[..., 1] = (center_y - target_y1) / window_h
    ltrb_targets[..., 2] = (target_x2 - center_x) / window_w
    ltrb_targets[..., 3] = (target_y2 - center_y) / window_h
    ltrb_targets = torch.clamp(ltrb_targets, min=0.0, max=1.0)

    if distractor_bboxes is not None:
        for index, distractors in enumerate(distractor_bboxes):
            if index >= object_count:
                break
            for distractor in distractors:
                dx, dy, dw, dh = mot.clamp_bbox_size(distractor)
                dx2 = dx + dw
                dy2 = dy + dh
                inside_distractor = (center_x[index] >= dx) & (center_x[index] <= dx2) & (center_y[index] >= dy) & (center_y[index] <= dy2)
                hard_negative_mask[index] |= inside_distractor
    hard_negative_mask &= ~positive_mask
    visibility_weights[positive_mask] = positive_weights[positive_mask].clamp_min(1.0)
    visibility_weights[hard_negative_mask] = torch.maximum(
        visibility_weights[hard_negative_mask],
        torch.full_like(visibility_weights[hard_negative_mask], 1.0),
    )
    missing_mask = ~present_mask
    if bool(missing_mask.any().item()):
        visibility_weights[missing_mask] = torch.maximum(
            visibility_weights[missing_mask],
            torch.full_like(visibility_weights[missing_mask], 1.0),
        )

    return V9LocalSearchTargets(
        score_labels=score_labels,
        visibility_labels=visibility_labels,
        positive_mask=positive_mask,
        hard_negative_mask=hard_negative_mask,
        loss_mask=loss_mask,
        visibility_loss_mask=visibility_loss_mask,
        visibility_weights=visibility_weights,
        positive_weights=positive_weights,
        ltrb_targets=ltrb_targets,
        search_windows=windows,
        target_boxes_xywh=boxes,
        target_boxes_xyxy=target_boxes_xyxy,
        present_mask=present_mask,
        positive_cells=int(positive_mask.sum().detach().item()),
        hard_negative_cells=int(hard_negative_mask.sum().detach().item()),
        loss_cells=int(loss_mask.sum().detach().item()),
        missing_targets=int((~present_mask).sum().detach().item()),
    )


def decode_v9_box_maps_xyxy(trainer: v9.V9LocalSearchLoRATTracker, head_output: v9.V9LocalHeadOutput) -> torch.Tensor:
    torch_module = trainer.torch
    box_delta_maps = head_output.box_delta_maps
    object_count, grid_height, grid_width = box_delta_maps.shape[:3]
    windows = torch_module.tensor(head_output.search_windows, device=trainer.device, dtype=torch_module.float32)
    if object_count <= 0:
        return torch_module.zeros((0, grid_height, grid_width, 4), device=trainer.device, dtype=torch_module.float32)
    ys = torch_module.arange(grid_height, device=trainer.device, dtype=torch_module.float32)
    xs = torch_module.arange(grid_width, device=trainer.device, dtype=torch_module.float32)
    yy, xx = torch_module.meshgrid(ys, xs, indexing="ij")
    ref_x = windows[:, None, None, 0] + ((xx[None, :, :] + 0.5) / float(grid_width)) * windows[:, None, None, 2].clamp_min(1.0)
    ref_y = windows[:, None, None, 1] + ((yy[None, :, :] + 0.5) / float(grid_height)) * windows[:, None, None, 3].clamp_min(1.0)
    ltrb = torch_module.sigmoid(torch_module.clamp(box_delta_maps.to(torch_module.float32), -30.0, 30.0))
    x1 = ref_x - ltrb[..., 0] * windows[:, None, None, 2].clamp_min(1.0)
    y1 = ref_y - ltrb[..., 1] * windows[:, None, None, 3].clamp_min(1.0)
    x2 = ref_x + ltrb[..., 2] * windows[:, None, None, 2].clamp_min(1.0)
    y2 = ref_y + ltrb[..., 3] * windows[:, None, None, 3].clamp_min(1.0)
    return torch_module.stack((x1, y1, x2, y2), dim=-1)


def decode_v9_predictions(trainer: v9.V9LocalSearchLoRATTracker, head_output: v9.V9LocalHeadOutput) -> List[BBox]:
    torch_module = trainer.torch
    score_maps = head_output.score_maps.detach().to(torch_module.float32)
    if score_maps.numel() == 0:
        return []
    object_count, grid_height, grid_width = score_maps.shape
    selection_scores = score_maps
    if getattr(trainer, "v8_window_penalty_ratio", 0.0) > 0.0 and grid_height > 1 and grid_width > 1:
        window = torch_module.outer(
            torch_module.hann_window(grid_height, periodic=False, device=trainer.device, dtype=torch_module.float32),
            torch_module.hann_window(grid_width, periodic=False, device=trainer.device, dtype=torch_module.float32),
        )
        ratio = float(trainer.v8_window_penalty_ratio)
        selection_scores = torch_module.sigmoid(score_maps) * (1.0 - ratio) + window[None, :, :] * ratio
    flat_indices = torch_module.argmax(selection_scores.reshape(object_count, -1), dim=1)
    flat_boxes = decode_v9_box_maps_xyxy(trainer, head_output).reshape(object_count, -1, 4)
    selected = flat_boxes[torch_module.arange(object_count, device=trainer.device), flat_indices].detach().cpu().tolist()
    return [
        (
            float(x1),
            float(y1),
            max(1.0, float(x2 - x1)),
            max(1.0, float(y2 - y1)),
        )
        for x1, y1, x2, y2 in selected
    ]


def v9_training_head_output(
    trainer: v9.V9LocalSearchLoRATTracker,
    frame_features,
    selected_banks: Sequence[Sequence[object]],
    objects: Sequence[train_common.TrainingObject],
    frame_shape: Tuple[int, ...],
) -> v9.V9LocalHeadOutput:
    windows = [training_search_window(item, trainer.search_radius_factor) for item in objects]
    local_grids = trainer._sample_local_search_grids(frame_features, windows, frame_shape)
    return trainer.object_conditioned_head.score_local(local_grids, selected_banks, windows)


def local_ranking_losses(
    trainer: v9.V9LocalSearchLoRATTracker,
    head_output: v9.V9LocalHeadOutput,
    targets: V9LocalSearchTargets,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score_maps = head_output.score_maps.to(torch.float32)
    if score_maps.numel() == 0 or not bool(targets.positive_mask.any().detach().item()):
        zero = score_maps.sum() * 0.0
        return zero, zero
    dcfst_loss = score_maps.sum() * 0.0
    assignment_loss = score_maps.sum() * 0.0
    dcfst_terms = 0
    assignment_terms = 0
    for index in range(score_maps.shape[0]):
        if not bool(targets.present_mask[index].detach().item()):
            continue
        pos_logits = score_maps[index][targets.positive_mask[index]]
        if pos_logits.numel() == 0:
            continue
        positive_logit = torch.logsumexp(pos_logits, dim=0) - torch.log(
            torch.as_tensor(float(pos_logits.numel()), device=trainer.device, dtype=torch.float32)
        )
        neg_logits = score_maps[index][targets.hard_negative_mask[index]]
        if neg_logits.numel() > 0:
            dcfst_loss = dcfst_loss + F.softplus(neg_logits - positive_logit + float(margin)).mean()
            dcfst_terms += 1
        other_centers = []
        for other in range(score_maps.shape[0]):
            if other == index or not bool(targets.present_mask[other].detach().item()):
                continue
            other_box = targets.target_boxes_xywh[other]
            window = targets.search_windows[index]
            cx = other_box[0] + other_box[2] * 0.5
            cy = other_box[1] + other_box[3] * 0.5
            if cx < window[0] or cy < window[1] or cx > window[0] + window[2] or cy > window[1] + window[3]:
                continue
            gx = int(torch.clamp(((cx - window[0]) / window[2].clamp_min(1.0) * score_maps.shape[2]).floor(), 0, score_maps.shape[2] - 1).detach().item())
            gy = int(torch.clamp(((cy - window[1]) / window[3].clamp_min(1.0) * score_maps.shape[1]).floor(), 0, score_maps.shape[1] - 1).detach().item())
            other_centers.append(score_maps[index, gy, gx])
        if other_centers:
            other_logits = torch.stack(other_centers)
            assignment_loss = assignment_loss + F.softplus(other_logits - positive_logit + float(margin)).mean()
            assignment_terms += 1
    if dcfst_terms:
        dcfst_loss = dcfst_loss / float(dcfst_terms)
    if assignment_terms:
        assignment_loss = assignment_loss / float(assignment_terms)
    return dcfst_loss, assignment_loss


def box_iou_map_xyxy(boxes_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    """IoU between every decoded candidate box and one target box."""

    boxes = boxes_xyxy.reshape(-1, 4)
    target = target_xyxy.reshape(1, 4)
    inter_x1 = torch.maximum(boxes[:, 0], target[:, 0])
    inter_y1 = torch.maximum(boxes[:, 1], target[:, 1])
    inter_x2 = torch.minimum(boxes[:, 2], target[:, 2])
    inter_y2 = torch.minimum(boxes[:, 3], target[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
    box_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    target_area = (target[:, 2] - target[:, 0]).clamp_min(0.0) * (target[:, 3] - target[:, 1]).clamp_min(0.0)
    union = (box_area + target_area - inter_area).clamp_min(1.0e-6)
    return (inter_area / union).reshape(boxes_xyxy.shape[:-1])


def bbox_xywh_to_xyxy_tensor(bbox: BBox, device: torch.device) -> torch.Tensor:
    x, y, width, height = mot.clamp_bbox_size(bbox)
    return torch.tensor((x, y, x + width, y + height), device=device, dtype=torch.float32)


def selected_target_candidate_ranking_loss(
    trainer: v9.V9LocalSearchLoRATTracker,
    head_output: v9.V9LocalHeadOutput,
    decoded_boxes_xyxy: torch.Tensor,
    targets: V9LocalSearchTargets,
    objects: Sequence[train_common.TrainingObject],
    *,
    margin: float,
    topk: int,
    positive_iou_threshold: float,
    other_iou_threshold: float,
) -> Tuple[torch.Tensor, int, int]:
    """Rank the selected target above distractor candidates in the local head.

    The objectness loss teaches "some target-like cell should be hot." This
    extra ranking term teaches "the selected target's cells should be hotter
    than candidate boxes that land on another visible object." That matches the
    V9 runtime failure mode we observed in the benchmark videos.
    """

    score_maps = head_output.score_maps.to(torch.float32)
    if score_maps.numel() == 0 or not bool(targets.present_mask.any().detach().item()):
        zero = score_maps.sum() * 0.0
        return zero, 0, 0

    topk = max(1, int(topk))
    ranking_loss = score_maps.sum() * 0.0
    object_terms = 0
    switch_candidate_cells = 0
    object_count = min(score_maps.shape[0], len(objects))
    for index in range(object_count):
        if not bool(targets.present_mask[index].detach().item()):
            continue
        own_iou = box_iou_map_xyxy(decoded_boxes_xyxy[index], targets.target_boxes_xyxy[index])
        positive_mask = targets.positive_mask[index] | (own_iou >= float(positive_iou_threshold))
        if not bool(positive_mask.any().detach().item()):
            continue

        other_iou = torch.zeros_like(own_iou)
        item = objects[index]
        for other_index, other in enumerate(objects[:object_count]):
            if other_index == index or other.track_id == item.track_id or not other.is_present:
                continue
            other_box = bbox_xywh_to_xyxy_tensor(other.current_bbox, trainer.device)
            other_iou = torch.maximum(other_iou, box_iou_map_xyxy(decoded_boxes_xyxy[index], other_box))
        for distractor_bbox in item.distractor_bboxes:
            distractor_box = bbox_xywh_to_xyxy_tensor(distractor_bbox, trainer.device)
            other_iou = torch.maximum(other_iou, box_iou_map_xyxy(decoded_boxes_xyxy[index], distractor_box))

        switch_mask = (other_iou >= float(other_iou_threshold)) & (other_iou > own_iou + 0.05)
        switch_candidate_cells += int(switch_mask.sum().detach().item())
        negative_mask = (targets.loss_mask[index] & ~positive_mask) | targets.hard_negative_mask[index] | switch_mask
        if not bool(negative_mask.any().detach().item()):
            continue

        positive_logits = score_maps[index][positive_mask]
        positive_logit = torch.logsumexp(positive_logits, dim=0) - torch.log(
            torch.as_tensor(float(positive_logits.numel()), device=trainer.device, dtype=torch.float32)
        )
        negative_logits = score_maps[index][negative_mask]
        if negative_logits.numel() > topk:
            negative_logits = torch.topk(negative_logits, k=topk).values
        ranking_loss = ranking_loss + F.softplus(negative_logits - positive_logit + float(margin)).mean()
        object_terms += 1

    if object_terms:
        ranking_loss = ranking_loss / float(object_terms)
    return ranking_loss, object_terms, switch_candidate_cells


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_v9_checkpoint(
    path: Path,
    trainer: v9.V9LocalSearchLoRATTracker,
    args: argparse.Namespace,
    epoch: int,
    steps: int,
    train_mean_iou: Optional[float],
    val_mean_iou: Optional[float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    best_val_iou: Optional[float] = None,
    best_checkpoint_score: Optional[float] = None,
    checkpoint_selection_score: Optional[float] = None,
    val_rollout_correct_rate: Optional[float] = None,
    val_rollout_identity_switch_rate: Optional[float] = None,
    val_rollout_track_loss_rate: Optional[float] = None,
    val_rollout_mean_frames_until_loss: Optional[float] = None,
    samples_seen_total: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": trainer.object_conditioned_head.state_dict(),
        "epoch": int(epoch),
        "steps": int(steps),
        "samples_seen_total": int(samples_seen_total if samples_seen_total is not None else steps),
        "effective_batch_or_sample_count": (
            float(samples_seen_total) / float(max(1, steps)) if samples_seen_total is not None else 1
        ),
        "lorat_config": args.lorat_config,
        "head_architecture": "v9_local_search_template_patch_lora_conditioned",
        "box_parameterization": "local_search_ltrb",
        "shared_backbone": "frozen_lorat_vit_frame_encoder",
        "v9_local_grid_size": int(args.v9_local_grid_size),
        "head_hidden_dim": int(args.v8_head_hidden_dim),
        "head_lora_rank": int(args.v8_head_lora_rank),
        "search_radius_factor": float(args.v8_search_radius_factor),
        "dataset_root": str(args.dataset_root),
        "mot17_root": "" if args.mot17_root is None else str(args.mot17_root),
        "tao_root": "" if args.tao_root is None else str(args.tao_root),
        "lasot_root": "" if args.lasot_root is None else str(args.lasot_root),
        "target_region_mode": args.target_region_mode,
        "target_regions_per_object": int(args.target_regions_per_object),
        "training_memory_slots": int(args.training_memory_slots),
        "visibility_loss_weight": float(getattr(args, "visibility_loss_weight", 0.0)),
        "visibility_positive_weight": float(getattr(args, "visibility_positive_weight", 1.0)),
        "visibility_hard_negative_weight": float(getattr(args, "visibility_hard_negative_weight", 1.0)),
        "visibility_missing_weight": float(getattr(args, "visibility_missing_weight", 1.0)),
        "drift_negative_probability": float(getattr(args, "drift_negative_probability", 0.0)),
        "drift_negative_start_epoch": int(getattr(args, "drift_negative_start_epoch", 0)),
        "drift_negative_max_per_frame": int(getattr(args, "drift_negative_max_per_frame", 0)),
        "drift_negative_shift_factor": float(getattr(args, "drift_negative_shift_factor", 0.0)),
        "rollout_validation_mode": str(args.rollout_validation_mode),
        "train_mean_iou": train_mean_iou,
        "val_mean_iou": val_mean_iou,
        "best_val_iou": best_val_iou,
        "best_checkpoint_score": best_checkpoint_score,
        "checkpoint_selection_score": checkpoint_selection_score,
        "val_rollout_correct_rate_iou30": val_rollout_correct_rate,
        "val_rollout_identity_switch_rate": val_rollout_identity_switch_rate,
        "val_rollout_track_loss_rate": val_rollout_track_loss_rate,
        "val_rollout_mean_frames_until_loss": val_rollout_mean_frames_until_loss,
        "command": " ".join(sys.argv),
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, str(path))


class LaSOTFrameHeadDataset:
    """LaSOT-style SOT samples using img/ plus groundtruth/occlusion files.

    LaSOT is single-target tracking data rather than MOT data. For V9 that is a
    feature, not a limitation: it provides real template/search supervision for
    "track the selected target" behavior that the older full-frame path struggled
    to preserve.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        max_objects: int,
        frame_stride: int,
        max_sequences: int,
        max_samples: int,
        target_region_mode: str,
        target_regions_per_object: int,
        template_area_factor: float,
        search_area_factor: float,
        search_scale_jitter: float,
        search_translation_jitter: float,
        search_min_object_size: float,
        template_sampling: str,
        previous_box_jitter: float,
        sequence_window_length: int,
        search_anchor_mode: str = "union",
        repair_search_to_target: bool = True,
        search_target_padding_fraction: float = 0.05,
        sequence_name_filter: Optional[str] = None,
        val_fraction: float = 0.20,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = str(split or "train").strip().lower()
        self.max_objects = max(1, int(max_objects))
        self.frame_stride = max(1, int(frame_stride))
        self.region_specs = train_common.region_specs_for_mode(target_region_mode)
        self.target_regions_per_object = max(1, int(target_regions_per_object))
        self.template_area_factor = max(1.0, float(template_area_factor))
        self.search_area_factor = max(1.0, float(search_area_factor))
        self.search_scale_jitter = max(0.0, float(search_scale_jitter))
        self.search_translation_jitter = max(0.0, float(search_translation_jitter))
        self.search_min_object_size = max(0.0, float(search_min_object_size))
        self.template_sampling = str(template_sampling or "mixed").strip().lower()
        if self.template_sampling not in {"first", "previous", "mixed", "window"}:
            raise ValueError(f"Unknown template sampling mode: {template_sampling!r}")
        self.previous_box_jitter = max(0.0, float(previous_box_jitter))
        self.sequence_window_length = max(1, int(sequence_window_length))
        self.search_anchor_mode = str(search_anchor_mode or "union").strip().lower()
        if self.search_anchor_mode not in {"current", "previous", "union"}:
            raise ValueError(f"Unknown search_anchor_mode: {search_anchor_mode!r}")
        self.repair_search_to_target = bool(repair_search_to_target)
        self.search_target_padding_fraction = max(0.0, float(search_target_padding_fraction))
        self.sequence_name_filter = str(sequence_name_filter).strip() if sequence_name_filter else None
        self.val_fraction = max(0.0, min(0.8, float(val_fraction)))
        self.max_sequences = max(0, int(max_sequences))
        self.samples: List[train_common.TrainingSample] = []
        self._build(max_sequences=max_sequences, max_samples=max_samples)

    @staticmethod
    def _read_bbox_rows(path: Path) -> List[BBox]:
        rows: List[BBox] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = [field.strip() for field in line.strip().replace("\t", ",").split(",") if field.strip()]
                if len(fields) < 4:
                    continue
                try:
                    x, y, w, h = (float(fields[index]) for index in range(4))
                except ValueError:
                    continue
                rows.append((x, y, max(1.0, w), max(1.0, h)))
        return rows

    @staticmethod
    def _read_binary_flags(path: Path, expected_length: int) -> List[int]:
        if not path.exists():
            return [0] * expected_length
        text = path.read_text(encoding="utf-8", errors="ignore")
        values: List[int] = []
        for token in text.replace("\n", ",").replace("\t", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(1 if int(float(token)) != 0 else 0)
            except ValueError:
                values.append(0)
        if len(values) < expected_length:
            values.extend([0] * (expected_length - len(values)))
        return values[:expected_length]

    def _sequence_dirs(self) -> List[Path]:
        sequences = [
            path
            for path in self.dataset_root.iterdir()
            if path.is_dir()
            and (path / "img").is_dir()
            and (path / "groundtruth.txt").exists()
            and (not path.name.startswith("_"))
        ]
        sequences = sorted(sequences, key=lambda path: path.name.lower())
        if self.sequence_name_filter:
            sequences = [path for path in sequences if path.name == self.sequence_name_filter]
        if self.split in {"val", "valid", "validation"} and len(sequences) > 1:
            val_count = max(1, int(round(len(sequences) * self.val_fraction)))
            sequences = sequences[-val_count:]
        elif self.split in {"train", "training"} and len(sequences) > 1 and self.val_fraction > 0.0:
            val_count = max(1, int(round(len(sequences) * self.val_fraction)))
            sequences = sequences[:-val_count] or sequences
        if self.max_sequences > 0:
            sequences = sequences[: self.max_sequences]
        return sequences

    def _choose_template_frame(
        self,
        first_frame: int,
        first_bbox: BBox,
        previous_frame: int,
        previous_bbox: BBox,
        recent_history: Sequence[Tuple[int, BBox]],
        rng: np.random.Generator,
    ) -> Tuple[int, BBox]:
        if self.template_sampling == "previous":
            return previous_frame, previous_bbox
        if self.template_sampling == "window" and recent_history:
            return recent_history[int(rng.integers(0, len(recent_history)))]
        if self.template_sampling == "mixed":
            draw = float(rng.random())
            if draw < 0.50:
                return first_frame, first_bbox
            if draw < 0.75:
                return previous_frame, previous_bbox
            if recent_history:
                return recent_history[int(rng.integers(0, len(recent_history)))]
        return first_frame, first_bbox

    def _build(self, max_sequences: int, max_samples: int) -> None:
        for sequence_path in self._sequence_dirs():
            image_paths = exercise.get_image_paths(sequence_path / "img")
            boxes = self._read_bbox_rows(sequence_path / "groundtruth.txt")
            if not image_paths or not boxes:
                continue
            usable_length = min(len(image_paths), len(boxes))
            full_occlusion = self._read_binary_flags(sequence_path / "full_occlusion.txt", usable_length)
            out_of_view = self._read_binary_flags(sequence_path / "out_of_view.txt", usable_length)
            visible_frames: List[Tuple[int, BBox]] = []
            for index in range(usable_length):
                bbox = boxes[index]
                if bbox[2] <= 1 or bbox[3] <= 1:
                    continue
                if full_occlusion[index] or out_of_view[index]:
                    continue
                visible_frames.append((index + 1, bbox))
            if not visible_frames:
                continue

            first_frame, first_bbox = visible_frames[0]
            recent_history: List[Tuple[int, BBox]] = []
            visible_by_frame = {frame: bbox for frame, bbox in visible_frames}
            for frame_number, current_bbox_full in visible_frames:
                if frame_number % self.frame_stride != 0:
                    recent_history.append((frame_number, current_bbox_full))
                    if len(recent_history) > self.sequence_window_length:
                        del recent_history[:-self.sequence_window_length]
                    continue
                image_index = exercise.frame_to_image_index(frame_number)
                if image_index < 0 or image_index >= len(image_paths):
                    continue
                previous_frame, previous_bbox_full = recent_history[-1] if recent_history else (first_frame, first_bbox)
                objects: List[train_common.TrainingObject] = []
                for spec in train_common.stable_region_order(self.region_specs, 1, frame_number)[: self.target_regions_per_object]:
                    rng = train_common.deterministic_rng(sequence_path, frame_number, 1, spec.name)
                    template_frame, template_bbox_full = self._choose_template_frame(
                        first_frame,
                        first_bbox,
                        previous_frame,
                        previous_bbox_full,
                        recent_history,
                        rng,
                    )
                    if template_frame not in visible_by_frame:
                        template_frame, template_bbox_full = first_frame, first_bbox
                    current_bbox = train_common.selected_region_bbox(current_bbox_full, spec)
                    previous_bbox = train_common.selected_region_bbox(previous_bbox_full, spec)
                    template_bbox = train_common.selected_region_bbox(template_bbox_full, spec)
                    noisy_previous_bbox = train_common.jitter_reference_bbox(previous_bbox, rng, self.previous_box_jitter)
                    if self.search_anchor_mode == "previous":
                        search_anchor_bbox = noisy_previous_bbox
                    elif self.search_anchor_mode == "union":
                        search_anchor_bbox = train_common.union_bbox_xywh(noisy_previous_bbox, current_bbox)
                    else:
                        search_anchor_bbox = current_bbox
                    search_bbox = train_common.siamfc_search_bbox(
                        search_anchor_bbox,
                        (224, 224),
                        self.search_area_factor,
                        self.search_scale_jitter,
                        self.search_translation_jitter,
                        self.search_min_object_size,
                        rng,
                    )
                    if self.repair_search_to_target:
                        search_bbox = train_common.repair_search_bbox_to_cover_target(
                            search_bbox,
                            current_bbox,
                            self.search_target_padding_fraction,
                        )
                    objects.append(
                        train_common.TrainingObject(
                            track_id=1,
                            current_bbox=current_bbox,
                            previous_bbox=noisy_previous_bbox,
                            previous_frame=previous_frame,
                            previous_context_bbox=train_common.siamfc_context_bbox(noisy_previous_bbox, self.template_area_factor),
                            template_frame=template_frame,
                            template_bbox=template_bbox,
                            template_context_bbox=train_common.siamfc_context_bbox(template_bbox, self.template_area_factor),
                            search_bbox=search_bbox,
                            target_kind=spec.name,
                            is_present=True,
                        )
                    )
                    if len(objects) >= self.max_objects:
                        break
                if objects:
                    self.samples.append(
                        train_common.TrainingSample(
                            sequence_path=sequence_path,
                            image_paths=image_paths,
                            frame_number=frame_number,
                            objects=objects,
                        )
                    )
                if max_samples > 0 and len(self.samples) >= max_samples:
                    return
                recent_history.append((frame_number, current_bbox_full))
                if len(recent_history) > self.sequence_window_length:
                    del recent_history[:-self.sequence_window_length]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> train_common.TrainingSample:
        return self.samples[index]


def build_mixed_dataset(args: argparse.Namespace, *, train: bool) -> train_common.MixedFrameHeadDataset:
    class_ids = train_common.parse_class_ids(args.class_id)
    datasets: List[object] = []
    dataset_specs: List[Tuple[str, Path, str, int, int, int]] = [
        (
            "DanceTrack",
            args.dataset_root,
            args.split if train else args.val_split,
            args.frame_stride if train else args.val_frame_stride,
            args.max_sequences if train else args.max_val_sequences,
            args.max_samples if train else args.max_val_samples,
        )
    ]
    if args.mot17_root is not None and args.mot17_root.exists() and not args.disable_mot17:
        dataset_specs.append(
            (
                "MOT17",
                args.mot17_root,
                args.mot17_split if train else args.mot17_val_split,
                args.mot17_frame_stride if train else args.mot17_val_frame_stride,
                args.mot17_max_sequences if train else args.mot17_max_val_sequences,
                args.mot17_max_samples if train else args.mot17_max_val_samples,
            )
        )
    for label, root, split, stride, max_sequences, max_samples in dataset_specs:
        if root is None or not root.exists():
            continue
        dataset = train_common.MOTFrameHeadDataset(
            root,
            split,
            class_ids,
            args.min_visibility,
            args.max_objects,
            stride,
            max_sequences,
            max_samples,
            args.target_region_mode,
            args.target_regions_per_object,
            args.template_area_factor,
            args.search_area_factor,
            args.search_scale_jitter,
            args.search_translation_jitter,
            args.search_min_object_size,
            args.template_sampling,
            args.previous_box_jitter,
            args.sequence_window_length,
            args.lost_target_probability if train else 0.0,
            args.max_lost_targets_per_frame if train else 0,
            args.max_missing_gap_frames,
            args.search_anchor_mode,
            args.repair_search_to_target,
            args.search_target_padding_fraction,
            args.debug_sequence_name,
            args.debug_track_id,
        )
        print(f"{'train' if train else 'val'} dataset {label}: samples={len(dataset)} root={root} split={split}", flush=True)
        datasets.append(dataset)
    if args.tao_root is not None and args.tao_root.exists() and not args.disable_tao:
        tao_dataset = train_common.TAOFrameHeadDataset(
            args.tao_root,
            args.tao_split if train else args.tao_val_split,
            class_ids,
            args.min_visibility,
            args.max_objects,
            args.tao_frame_stride if train else args.tao_val_frame_stride,
            args.tao_max_sequences if train else args.tao_max_val_sequences,
            args.tao_max_samples if train else args.tao_max_val_samples,
            args.target_region_mode,
            args.target_regions_per_object,
            args.template_area_factor,
            args.search_area_factor,
            args.search_scale_jitter,
            args.search_translation_jitter,
            args.search_min_object_size,
            args.template_sampling,
            args.previous_box_jitter,
            args.sequence_window_length,
            args.lost_target_probability if train else 0.0,
            args.max_lost_targets_per_frame if train else 0,
            args.max_missing_gap_frames,
            args.search_anchor_mode,
            args.repair_search_to_target,
            args.search_target_padding_fraction,
            args.debug_sequence_name,
            args.debug_track_id,
            args.tao_use_freeform,
        )
        print(f"{'train' if train else 'val'} dataset TAO: samples={len(tao_dataset)} root={args.tao_root}", flush=True)
        datasets.append(tao_dataset)
    if args.lasot_root is not None and args.lasot_root.exists() and not args.disable_lasot:
        lasot_dataset = LaSOTFrameHeadDataset(
            args.lasot_root,
            args.lasot_split if train else args.lasot_val_split,
            args.max_objects,
            args.lasot_frame_stride if train else args.lasot_val_frame_stride,
            args.lasot_max_sequences if train else args.lasot_max_val_sequences,
            args.lasot_max_samples if train else args.lasot_max_val_samples,
            args.target_region_mode,
            args.target_regions_per_object,
            args.template_area_factor,
            args.search_area_factor,
            args.search_scale_jitter,
            args.search_translation_jitter,
            args.search_min_object_size,
            args.template_sampling,
            args.previous_box_jitter,
            args.sequence_window_length,
            args.search_anchor_mode,
            args.repair_search_to_target,
            args.search_target_padding_fraction,
            args.debug_sequence_name,
            args.lasot_val_fraction,
        )
        print(
            f"{'train' if train else 'val'} dataset LaSOT: samples={len(lasot_dataset)} root={args.lasot_root}",
            flush=True,
        )
        datasets.append(lasot_dataset)
    return train_common.MixedFrameHeadDataset(datasets)


def canonical_source_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    return SOURCE_WEIGHT_ALIASES.get(cleaned.lower(), cleaned)


def parse_source_sampling_weights(text: Optional[str]) -> Dict[str, float]:
    if text is None or not text.strip():
        return {}
    weights: Dict[str, float] = {}
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"source sampling item must be name=value, got {item!r}")
        name, value = item.split("=", 1)
        source = canonical_source_name(name)
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"source sampling weight for {source!r} must be numeric, got {value!r}") from exc
        if weight < 0.0:
            raise ValueError(f"source sampling weight for {source!r} must be non-negative")
        weights[source] = weight
    return weights


def source_spans(dataset: train_common.MixedFrameHeadDataset) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    start = 0
    for source_dataset in getattr(dataset, "datasets", []):
        end = start + len(source_dataset)
        spans.append((source_dataset.__class__.__name__, start, end))
        start = end
    if not spans:
        spans.append((dataset.__class__.__name__, 0, len(dataset)))
    return spans


def source_count_summary(order: Sequence[int], spans: Sequence[Tuple[str, int, int]]) -> Dict[str, int]:
    counts = {name: 0 for name, _, _ in spans}
    if not order:
        return counts
    span_index = 0
    sorted_order = sorted(order)
    for sample_index in sorted_order:
        while span_index + 1 < len(spans) and sample_index >= spans[span_index][2]:
            span_index += 1
        name, start, end = spans[span_index]
        if start <= sample_index < end:
            counts[name] = counts.get(name, 0) + 1
    return counts


def format_source_counts(counts: Dict[str, int]) -> str:
    return ";".join(f"{name}={count}" for name, count in sorted(counts.items()))


def build_epoch_sample_order(
    dataset: train_common.MixedFrameHeadDataset,
    args: argparse.Namespace,
    epoch: int,
) -> Tuple[List[int], Dict[str, int], Dict[str, float]]:
    sample_count = len(dataset)
    if sample_count == 0:
        return [], {}, {}
    target_count = sample_count
    if args.max_train_samples_per_epoch > 0:
        target_count = min(sample_count, int(args.max_train_samples_per_epoch))
    rng = random.Random(int(args.seed) + (epoch * 1009))
    spans = source_spans(dataset)
    weights = parse_source_sampling_weights(args.source_sampling_weights)
    if not weights:
        order = list(range(sample_count))
        rng.shuffle(order)
        order = order[:target_count]
        return order, source_count_summary(order, spans), weights

    groups: List[Tuple[str, List[int], float, float]] = []
    total_weighted_size = 0.0
    for name, start, end in spans:
        indices = list(range(start, end))
        weight = weights.get(name, 1.0)
        weighted_size = float(len(indices)) * max(0.0, weight)
        if indices and weighted_size > 0.0:
            groups.append((name, indices, weight, weighted_size))
            total_weighted_size += weighted_size
    if total_weighted_size <= 0.0:
        order = list(range(sample_count))
        rng.shuffle(order)
        order = order[:target_count]
        return order, source_count_summary(order, spans), weights

    selected: List[int] = []
    quotas: Dict[str, int] = {}
    for name, indices, _weight, weighted_size in groups:
        quotas[name] = int(round((weighted_size / total_weighted_size) * target_count))
    diff = target_count - sum(quotas.values())
    if diff != 0:
        groups_by_size = sorted(groups, key=lambda item: item[3], reverse=(diff > 0))
        for idx in range(abs(diff)):
            name = groups_by_size[idx % len(groups_by_size)][0]
            quotas[name] = max(0, quotas.get(name, 0) + (1 if diff > 0 else -1))

    for name, indices, _weight, _weighted_size in groups:
        quota = min(target_count, max(0, quotas.get(name, 0)))
        rng.shuffle(indices)
        if quota <= len(indices):
            selected.extend(indices[:quota])
        else:
            selected.extend(indices)
            selected.extend(rng.choice(indices) for _ in range(quota - len(indices)))
    if len(selected) < target_count:
        fallback = list(range(sample_count))
        rng.shuffle(fallback)
        selected.extend(fallback[: target_count - len(selected)])
    rng.shuffle(selected)
    selected = selected[:target_count]
    return selected, source_count_summary(selected, spans), weights


def load_resume_training_metadata(path: Optional[Path]) -> Tuple[int, int, Optional[float], Optional[float]]:
    if path is None or not path.exists():
        return 0, 0, None, None
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: could not read resume metadata from {path}: {exc}", flush=True)
        return 0, 0, None, None
    if not isinstance(checkpoint, dict):
        return 0, 0, None, None
    epoch = int(checkpoint.get("epoch") or 0)
    steps = int(checkpoint.get("steps") or 0)
    val_iou = checkpoint.get("best_val_iou", checkpoint.get("val_mean_iou"))
    checkpoint_score = checkpoint.get("best_checkpoint_score", checkpoint.get("checkpoint_selection_score"))
    try:
        best_val_iou = None if val_iou is None else float(val_iou)
    except (TypeError, ValueError):
        best_val_iou = None
    try:
        best_checkpoint_score = None if checkpoint_score is None else float(checkpoint_score)
    except (TypeError, ValueError):
        best_checkpoint_score = None
    return max(0, epoch), max(0, steps), best_val_iou, best_checkpoint_score


def resolve_auto_resume_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> Optional[Path]:
    explicit = args.resume_head
    if explicit is not None:
        return explicit
    if not bool(getattr(args, "auto_resume_latest", False)):
        return None
    config_key = args.lorat_config.replace("-", "_")
    candidates = [
        checkpoint_dir / f"{args.output.stem}_best_by_rollout_identity.pt",
        checkpoint_dir / f"{args.output.stem}_best_by_val_iou.pt",
        checkpoint_dir / f"v9_local_head_{config_key}_best_by_rollout_identity.pt",
        checkpoint_dir / f"v9_local_head_{config_key}_best_by_val_iou.pt",
        checkpoint_dir / f"{args.output.stem}_latest.pt",
        checkpoint_dir / f"v9_local_head_{config_key}_latest.pt",
        args.output,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def warn_if_resuming_latest_despite_best(resume_path: Optional[Path], args: argparse.Namespace, checkpoint_dir: Path) -> None:
    if resume_path is None:
        return
    path_text = str(resume_path)
    if "latest" not in resume_path.name:
        return
    config_key = args.lorat_config.replace("-", "_")
    best_candidates = [
        checkpoint_dir / f"{args.output.stem}_best_by_rollout_identity.pt",
        checkpoint_dir / f"{args.output.stem}_best_by_val_iou.pt",
        checkpoint_dir / f"v9_local_head_{config_key}_best_by_rollout_identity.pt",
        checkpoint_dir / f"v9_local_head_{config_key}_best_by_val_iou.pt",
    ]
    existing = [candidate for candidate in best_candidates if candidate.exists()]
    if existing:
        print(
            "Warning: resuming from latest while preferred best checkpoint(s) exist: "
            f"resume={path_text} best_candidates={[str(path) for path in existing]}",
            flush=True,
        )


def seed_checkpoint_aliases_from_resume(resume_path: Optional[Path], args: argparse.Namespace, checkpoint_dir: Path) -> None:
    """Copy inherited best/latest checkpoints into the current result folder.

    Long V9 runs often resume from a previous result directory. If the resumed
    checkpoint remains the best score, the new folder would otherwise export
    only fresh latest checkpoints and silently omit the preferred
    best_by_rollout_identity/best_by_val_iou aliases used by benchmarking.
    """
    if resume_path is None or not resume_path.exists():
        return

    resume_name = resume_path.name
    suffixes: List[str] = []
    if "best_by_rollout_identity" in resume_name:
        suffixes.append("best_by_rollout_identity")
    if "best_by_val_iou" in resume_name:
        suffixes.append("best_by_val_iou")
    if "latest" in resume_name:
        suffixes.append("latest")
    if not suffixes:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_key = args.lorat_config.replace("-", "_")
    destinations: List[Path] = []
    for suffix in suffixes:
        destinations.append(checkpoint_dir / f"{args.output.stem}_{suffix}.pt")
        destinations.append(checkpoint_dir / f"v9_local_head_{config_key}_{suffix}.pt")

    copied: List[str] = []
    for destination in dict.fromkeys(destinations):
        if destination.exists():
            continue
        try:
            if resume_path.resolve() == destination.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(resume_path, destination)
        copied.append(str(destination))

    if copied:
        print(
            "Seeded current V9 checkpoint alias(es) from resume checkpoint: "
            f"source={resume_path} destinations={copied}",
            flush=True,
        )


def restore_optimizer_state(path: Optional[Path], optimizer: torch.optim.Optimizer, device: object) -> bool:
    if path is None or not path.exists():
        return False
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except Exception as exc:
        print(f"Warning: could not read optimizer state from {path}: {exc}", flush=True)
        return False
    if not isinstance(checkpoint, dict) or "optimizer" not in checkpoint:
        return False
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as exc:
        print(f"Warning: could not restore optimizer state from {path}: {exc}", flush=True)
        return False
    return True


def evaluate_v9_head(
    trainer: v9.V9LocalSearchLoRATTracker,
    dataset: train_common.MixedFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    memory_slots: int,
) -> Tuple[Optional[float], Optional[float]]:
    if len(dataset) == 0 or max_samples == 0:
        return None, None
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()
    sample_indices = train_common.spread_sample_indices(len(dataset), max_samples)
    iou_sum = 0.0
    hit50 = 0
    count = 0
    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            frame = train_common.try_load_frame(sample.image_paths[frame_index], f"v9-eval {sample.sequence_path.name} frame={sample.frame_number}")
            if frame is None:
                continue
            frame_features = trainer.shared_frame_encoder.encode(frame).feature_map.detach()
            objects = list(sample.objects)
            selected_banks = train_common.build_selected_banks(
                trainer,
                sample,
                objects,
                template_feature_cache,
                memory_slots=memory_slots,
            )
            if selected_banks is None:
                continue
            head_output = v9_training_head_output(trainer, frame_features, selected_banks, objects, frame.shape)
            predictions = decode_v9_predictions(trainer, head_output)
            for prediction, item in zip(predictions, objects):
                if not item.is_present:
                    continue
                iou = exercise.bbox_iou(prediction, item.current_bbox)
                iou_sum += iou
                hit50 += 1 if iou >= 0.5 else 0
                count += 1
    if was_training:
        head.module.train()
    if count == 0:
        return None, None
    return iou_sum / float(count), hit50 / float(count)


def evaluate_v9_closed_loop_probe(
    trainer: v9.V9LocalSearchLoRATTracker,
    dataset: train_common.MixedFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    memory_slots: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Two-pass validation using the first prediction as the second search anchor.

    This is deliberately small enough to run during training. It is not a full
    video rollout, but it flags whether local search/geometry collapses when the
    model must condition on its own previous box instead of a repaired GT window.
    """

    if len(dataset) == 0 or max_samples == 0:
        return None, None, None, None, None, None
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()
    sample_indices = train_common.spread_sample_indices(len(dataset), max_samples)
    iou_sum = 0.0
    correct_count = 0
    identity_switch_count = 0
    window_miss_count = 0
    loss_events = 0
    frames_until_loss: List[int] = []
    count = 0
    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            frame = train_common.try_load_frame(sample.image_paths[frame_index], f"v9-rollout {sample.sequence_path.name} frame={sample.frame_number}")
            if frame is None:
                continue
            frame_features = trainer.shared_frame_encoder.encode(frame).feature_map.detach()
            objects = list(sample.objects)
            selected_banks = train_common.build_selected_banks(
                trainer,
                sample,
                objects,
                template_feature_cache,
                memory_slots=memory_slots,
            )
            if selected_banks is None:
                continue
            first_output = v9_training_head_output(trainer, frame_features, selected_banks, objects, frame.shape)
            first_predictions = decode_v9_predictions(trainer, first_output)
            rolled_objects: List[train_common.TrainingObject] = []
            for item, prediction in zip(objects, first_predictions):
                if not item.is_present:
                    rolled_objects.append(item)
                    continue
                predicted_previous = train_common.clamp_bbox_to_frame_shape(prediction, frame.shape)
                search_bbox = train_common.siamfc_search_bbox(
                    predicted_previous,
                    (224, 224),
                    4.0,
                    0.0,
                    0.0,
                    10.0,
                    train_common.deterministic_rng(sample.sequence_path, sample.frame_number, sample_index, "v9-rollout-probe"),
                )
                rolled_objects.append(replace(item, previous_bbox=predicted_previous, search_bbox=search_bbox))
            second_output = v9_training_head_output(trainer, frame_features, selected_banks, rolled_objects, frame.shape)
            second_predictions = decode_v9_predictions(trainer, second_output)
            for prediction, item in zip(second_predictions, rolled_objects):
                if not item.is_present:
                    continue
                best_other_iou = 0.0
                for other in rolled_objects:
                    if other.track_id == item.track_id or not other.is_present:
                        continue
                    best_other_iou = max(best_other_iou, exercise.bbox_iou(prediction, other.current_bbox))
                for distractor_bbox in item.distractor_bboxes:
                    best_other_iou = max(best_other_iou, exercise.bbox_iou(prediction, distractor_bbox))
                search_window = training_search_window(item, trainer.search_radius_factor)
                target_center_x, target_center_y = mot.bbox_center(item.current_bbox)
                wx, wy, ww, wh = mot.clamp_bbox_size(search_window)
                if not (wx <= target_center_x <= wx + ww and wy <= target_center_y <= wy + wh):
                    window_miss_count += 1
                iou = exercise.bbox_iou(prediction, item.current_bbox)
                iou_sum += iou
                correct = iou >= 0.30 and iou + 0.05 >= best_other_iou
                switched = best_other_iou >= 0.30 and best_other_iou > iou + 0.05
                correct_count += int(correct)
                identity_switch_count += int(switched)
                lost = not correct or switched
                loss_events += int(lost)
                if lost:
                    frames_until_loss.append(2)
                count += 1
    if was_training:
        head.module.train()
    if count == 0:
        return None, None, None, None, None, None
    return (
        iou_sum / float(count),
        correct_count / float(count),
        window_miss_count / float(count),
        identity_switch_count / float(count),
        loss_events / float(count),
        statistics.fmean(frames_until_loss) if frames_until_loss else None,
    )


def dense_rollout_validation_samples(
    dataset: train_common.MixedFrameHeadDataset,
    max_samples: int,
    clip_frames: int,
) -> List[train_common.TrainingSample]:
    if len(dataset) == 0 or max_samples <= 0:
        return []
    grouped: Dict[str, List[train_common.TrainingSample]] = {}
    scan_count = min(len(dataset), max(max_samples * 8, max_samples))
    for index in train_common.spread_sample_indices(len(dataset), scan_count):
        sample = dataset[index]
        grouped.setdefault(str(sample.sequence_path), []).append(sample)
    selected: List[train_common.TrainingSample] = []
    clip_limit = max(1, int(clip_frames))
    for samples in grouped.values():
        ordered = sorted(samples, key=lambda item: int(item.frame_number))
        if not ordered:
            continue
        if len(selected) >= max_samples:
            break
        window = ordered[:clip_limit]
        selected.extend(window[: max(0, max_samples - len(selected))])
    selected.sort(key=lambda item: (str(item.sequence_path), int(item.frame_number)))
    return selected


def evaluate_v9_sequence_rollout_probe(
    trainer: v9.V9LocalSearchLoRATTracker,
    dataset: train_common.MixedFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    memory_slots: int,
    clip_frames: int = 96,
    max_frame_gap: int = 30,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Validation over real consecutive frames with predicted boxes as anchors.

    This is still lightweight enough for training-time diagnostics, but it is
    much closer to the benchmark failure mode than the older two-pass probe.
    It keeps a per-sequence, per-track predicted previous box and feeds that
    prediction into later search-window construction.
    """

    if len(dataset) == 0 or max_samples == 0:
        return None, None, None, None, None, None
    torch = trainer.torch
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()

    samples = dense_rollout_validation_samples(dataset, max_samples, clip_frames)
    samples.sort(key=lambda item: (str(item.sequence_path), int(item.frame_number)))

    predicted_previous_by_key: Dict[Tuple[str, int, str], BBox] = {}
    previous_frame_by_key: Dict[Tuple[str, int, str], int] = {}
    alive_frames_by_key: Dict[Tuple[str, int, str], int] = {}
    lost_seen_by_key: Dict[Tuple[str, int, str], bool] = {}
    iou_sum = 0.0
    correct_count = 0
    identity_switch_count = 0
    window_miss_count = 0
    loss_events = 0
    frames_until_loss: List[int] = []
    count = 0

    with torch.no_grad():
        for sample in samples:
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            frame = train_common.try_load_frame(
                sample.image_paths[frame_index],
                f"v9-sequence-rollout {sample.sequence_path.name} frame={sample.frame_number}",
            )
            if frame is None:
                continue

            objects: List[train_common.TrainingObject] = []
            for item in sample.objects:
                key = (str(sample.sequence_path), int(item.track_id), str(item.target_kind))
                previous_prediction = predicted_previous_by_key.get(key)
                previous_frame = previous_frame_by_key.get(key)
                if (
                    previous_prediction is None
                    or previous_frame is None
                    or int(sample.frame_number) - int(previous_frame) > max(1, int(max_frame_gap))
                ):
                    predicted_previous_by_key.pop(key, None)
                    previous_frame_by_key.pop(key, None)
                    alive_frames_by_key[key] = 0
                    lost_seen_by_key[key] = False
                    objects.append(item)
                    continue
                search_bbox = train_common.siamfc_search_bbox(
                    previous_prediction,
                    (224, 224),
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    train_common.deterministic_rng(
                        sample.sequence_path,
                        sample.frame_number,
                        item.track_id,
                        f"v9-sequence-rollout-{item.target_kind}",
                    ),
                )
                objects.append(
                    replace(
                        item,
                        previous_bbox=previous_prediction,
                        previous_frame=max(1, int(sample.frame_number) - 1),
                        previous_context_bbox=train_common.siamfc_context_bbox(previous_prediction, 2.0),
                        search_bbox=search_bbox,
                    )
                )

            frame_features = trainer.shared_frame_encoder.encode(frame).feature_map.detach()
            selected_banks = train_common.build_selected_banks(
                trainer,
                sample,
                objects,
                template_feature_cache,
                memory_slots=memory_slots,
            )
            if selected_banks is None:
                continue
            head_output = v9_training_head_output(trainer, frame_features, selected_banks, objects, frame.shape)
            predictions = decode_v9_predictions(trainer, head_output)

            for prediction, item in zip(predictions, objects):
                key = (str(sample.sequence_path), int(item.track_id), str(item.target_kind))
                if not item.is_present:
                    predicted_previous_by_key.pop(key, None)
                    previous_frame_by_key.pop(key, None)
                    alive_frames_by_key[key] = 0
                    lost_seen_by_key[key] = False
                    continue
                prediction = train_common.clamp_bbox_to_frame_shape(prediction, frame.shape)
                predicted_previous_by_key[key] = prediction
                previous_frame_by_key[key] = int(sample.frame_number)

                best_other_iou = 0.0
                for other in objects:
                    if other.track_id == item.track_id or not other.is_present:
                        continue
                    best_other_iou = max(best_other_iou, exercise.bbox_iou(prediction, other.current_bbox))
                for distractor_bbox in item.distractor_bboxes:
                    best_other_iou = max(best_other_iou, exercise.bbox_iou(prediction, distractor_bbox))

                search_window = training_search_window(item, trainer.search_radius_factor)
                target_center_x, target_center_y = mot.bbox_center(item.current_bbox)
                wx, wy, ww, wh = mot.clamp_bbox_size(search_window)
                if not (wx <= target_center_x <= wx + ww and wy <= target_center_y <= wy + wh):
                    window_miss_count += 1

                iou = exercise.bbox_iou(prediction, item.current_bbox)
                iou_sum += iou
                correct = iou >= 0.30 and iou + 0.05 >= best_other_iou
                switched = best_other_iou >= 0.30 and best_other_iou > iou + 0.05
                correct_count += int(correct)
                identity_switch_count += int(switched)
                alive_frames_by_key[key] = int(alive_frames_by_key.get(key, 0)) + 1
                lost = not correct or switched
                loss_events += int(lost)
                if lost and not bool(lost_seen_by_key.get(key, False)):
                    frames_until_loss.append(alive_frames_by_key[key])
                    lost_seen_by_key[key] = True
                count += 1

    if was_training:
        head.module.train()
    if count == 0:
        return None, None, None, None, None, None
    return (
        iou_sum / float(count),
        correct_count / float(count),
        window_miss_count / float(count),
        identity_switch_count / float(count),
        loss_events / float(count),
        statistics.fmean(frames_until_loss) if frames_until_loss else None,
    )


def checkpoint_selection_score(
    val_iou: Optional[float],
    val_rollout_correct: Optional[float],
    val_rollout_identity_switch: Optional[float],
    val_rollout_window_miss: Optional[float],
    val_rollout_track_loss: Optional[float],
    *,
    identity_switch_penalty: float,
    window_miss_penalty: float,
    track_loss_penalty: float,
) -> Optional[float]:
    if val_rollout_correct is not None:
        score = float(val_rollout_correct)
        if val_rollout_identity_switch is not None:
            score -= max(0.0, float(identity_switch_penalty)) * float(val_rollout_identity_switch)
        if val_rollout_window_miss is not None:
            score -= max(0.0, float(window_miss_penalty)) * float(val_rollout_window_miss)
        if val_rollout_track_loss is not None:
            score -= max(0.0, float(track_loss_penalty)) * float(val_rollout_track_loss)
        return score
    return None if val_iou is None else float(val_iou)


def parse_args() -> argparse.Namespace:
    project_root = mot.PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Train V9 local-search LoRaT-MOT head on DanceTrack/MOT17/TAO data.")
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--mot17-root", type=Path, default=project_root / "data" / "raw" / "MOTChallenge" / "MOT17")
    parser.add_argument("--mot17-split", default="train")
    parser.add_argument("--mot17-val-split", default="train")
    parser.add_argument("--disable-mot17", action="store_true")
    parser.add_argument("--tao-root", type=Path, default=project_root / "data" / "raw" / "TAO_OW_SUBSET")
    parser.add_argument("--tao-split", default="train")
    parser.add_argument("--tao-val-split", default="validation")
    parser.add_argument("--tao-use-freeform", action="store_true")
    parser.add_argument("--disable-tao", action="store_true")
    parser.add_argument("--lasot-root", type=Path, default=project_root / "data" / "raw" / "LaSOT_subset")
    parser.add_argument("--lasot-split", default="train")
    parser.add_argument("--lasot-val-split", default="val")
    parser.add_argument("--lasot-val-fraction", type=float, default=0.20)
    parser.add_argument("--disable-lasot", action="store_true")
    parser.add_argument("--class-id", type=int, action="append")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--target-region-mode", default="mixed", choices=("full", "parts", "mixed"))
    parser.add_argument("--target-regions-per-object", type=int, default=4)
    parser.add_argument("--template-area-factor", type=float, default=2.0)
    parser.add_argument("--search-area-factor", type=float, default=4.0)
    parser.add_argument("--search-scale-jitter", type=float, default=0.25)
    parser.add_argument("--search-translation-jitter", type=float, default=3.0)
    parser.add_argument("--search-min-object-size", type=float, default=10.0)
    parser.add_argument("--search-anchor-mode", default="union", choices=("current", "previous", "union"))
    parser.add_argument("--disable-search-target-repair", dest="repair_search_to_target", action="store_false", default=True)
    parser.add_argument("--search-target-padding-fraction", type=float, default=0.05)
    parser.add_argument("--template-sampling", default="mixed", choices=("first", "previous", "mixed", "window"))
    parser.add_argument("--previous-box-jitter", type=float, default=0.10)
    parser.add_argument("--sequence-window-length", type=int, default=5)
    parser.add_argument("--lost-target-probability", type=float, default=0.10)
    parser.add_argument("--max-lost-targets-per-frame", type=int, default=2)
    parser.add_argument("--max-missing-gap-frames", type=int, default=30)
    parser.add_argument(
        "--drift-negative-probability",
        type=float,
        default=0.35,
        help=(
            "Probability of adding selected-target drift negatives after warmup. "
            "These keep the original template but place the local search window on a distractor/off-target region."
        ),
    )
    parser.add_argument("--drift-negative-start-epoch", type=int, default=3)
    parser.add_argument("--drift-negative-max-per-frame", type=int, default=2)
    parser.add_argument("--drift-negative-shift-factor", type=float, default=3.0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--val-frame-stride", type=int, default=15)
    parser.add_argument("--mot17-frame-stride", type=int, default=5)
    parser.add_argument("--mot17-val-frame-stride", type=int, default=15)
    parser.add_argument("--tao-frame-stride", type=int, default=2)
    parser.add_argument("--tao-val-frame-stride", type=int, default=5)
    parser.add_argument("--lasot-frame-stride", type=int, default=5)
    parser.add_argument("--lasot-val-frame-stride", type=int, default=15)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=2)
    parser.add_argument("--mot17-max-sequences", type=int, default=0)
    parser.add_argument("--mot17-max-val-sequences", type=int, default=2)
    parser.add_argument("--tao-max-sequences", type=int, default=0)
    parser.add_argument("--tao-max-val-sequences", type=int, default=2)
    parser.add_argument("--lasot-max-sequences", type=int, default=0)
    parser.add_argument("--lasot-max-val-sequences", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--mot17-max-samples", type=int, default=0)
    parser.add_argument("--mot17-max-val-samples", type=int, default=128)
    parser.add_argument("--tao-max-samples", type=int, default=0)
    parser.add_argument("--tao-max-val-samples", type=int, default=128)
    parser.add_argument("--lasot-max-samples", type=int, default=0)
    parser.add_argument("--lasot-max-val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--max-train-samples-per-epoch", type=int, default=2048)
    parser.add_argument(
        "--source-sampling-weights",
        default="",
        help=(
            "Optional comma-separated source weights for each epoch subset, for example "
            "'MOTFrameHeadDataset=1.0,TAOFrameHeadDataset=4.0,LaSOTFrameHeadDataset=1.5'. "
            "Aliases such as dancetrack, tao, and lasot are also accepted."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-wall-hours", type=float, default=0.0)
    parser.add_argument("--eval-interval-epochs", type=int, default=5)
    parser.add_argument("--train-diagnostic-samples", type=int, default=64)
    parser.add_argument("--rollout-diagnostic-samples", type=int, default=32)
    parser.add_argument(
        "--rollout-validation-mode",
        choices=("sequence", "two_pass"),
        default="sequence",
        help="Use real sequential-frame rollout validation by default; two_pass keeps the older same-frame geometry probe.",
    )
    parser.add_argument("--rollout-clip-frames", type=int, default=96)
    parser.add_argument("--rollout-max-frame-gap", type=int, default=30)
    parser.add_argument("--reid-diagnostic-samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--ltrb-loss-weight", type=float, default=1.0)
    parser.add_argument("--reid-loss-weight", type=float, default=0.35)
    parser.add_argument("--center-positive-weight", type=float, default=0.5)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=3.0)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.35)
    parser.add_argument("--visibility-positive-weight", type=float, default=1.25)
    parser.add_argument("--visibility-hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--visibility-missing-weight", type=float, default=3.0)
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--small-target-loss-weight", type=float, default=3.0)
    parser.add_argument("--small-target-area-threshold", type=float, default=v9.DEFAULT_V9_SMALL_TARGET_AREA)
    parser.add_argument("--small-target-max-side", type=float, default=v9.DEFAULT_V9_SMALL_TARGET_MAX_SIDE)
    parser.add_argument("--dcfst-discrimination-weight", type=float, default=0.50)
    parser.add_argument("--assignment-discrimination-weight", type=float, default=0.50)
    parser.add_argument("--assignment-margin", type=float, default=0.25)
    parser.add_argument("--candidate-ranking-loss-weight", type=float, default=0.60)
    parser.add_argument("--candidate-ranking-start-epoch", type=int, default=3)
    parser.add_argument("--candidate-ranking-margin", type=float, default=0.25)
    parser.add_argument("--candidate-ranking-topk", type=int, default=16)
    parser.add_argument("--candidate-ranking-positive-iou", type=float, default=0.35)
    parser.add_argument("--candidate-ranking-other-iou", type=float, default=0.30)
    parser.add_argument("--rollout-identity-switch-penalty", type=float, default=0.75)
    parser.add_argument("--rollout-window-miss-penalty", type=float, default=0.25)
    parser.add_argument("--rollout-track-loss-penalty", type=float, default=0.50)
    parser.add_argument("--geometry-only-epochs", type=int, default=2)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=3)
    parser.add_argument("--reid-start-epoch", type=int, default=5)
    parser.add_argument("--dcfst-start-epoch", type=int, default=3)
    parser.add_argument("--assignment-start-epoch", type=int, default=3)
    parser.add_argument("--closed-loop-start-epoch", type=int, default=5)
    parser.add_argument("--closed-loop-probability", type=float, default=0.25)
    parser.add_argument("--disable-lorat-augmentation", dest="lorat_augmentation", action="store_false", default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lorat-root", type=Path, default=mot.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--v8-head-hidden-dim", type=int, default=256)
    parser.add_argument("--v8-head-lora-rank", type=int, default=16)
    parser.add_argument("--v8-head-rank", type=int, default=mot.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--v8-search-radius-factor", type=float, default=2.25)
    parser.add_argument("--v8-window-penalty-ratio", type=float, default=v9.DEFAULT_V9_WINDOW_PENALTY_RATIO)
    parser.add_argument("--v9-local-grid-size", type=int, default=v9.DEFAULT_V9_LOCAL_GRID_SIZE)
    parser.add_argument("--training-memory-slots", type=int, default=2)
    parser.add_argument("--resume-head", type=Path)
    parser.add_argument(
        "--auto-resume-latest",
        action="store_true",
        help="When --resume-head is omitted, resume from the latest checkpoint in --checkpoint-dir/--output if present.",
    )
    parser.add_argument("--output", type=Path, default=mot.PROJECT_ROOT / "models" / "lorat" / "v9_local_head.pt")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("all", "latest", "none"),
        default="latest",
        help="How to retain step checkpoints. Epoch/latest/best checkpoints are always preserved.",
    )
    parser.add_argument("--diagnostic-csv", type=Path)
    parser.add_argument("--debug-sequence-name")
    parser.add_argument("--debug-track-id", type=int)
    parser.add_argument("--overfit-smoke", action="store_true")
    parser.add_argument("--overfit-smoke-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-targets", action="store_true")
    return parser.parse_args()


def smoke_test_targets() -> None:
    previous = [(100.0, 80.0, 40.0, 60.0), (30.0, 40.0, 14.0, 16.0)]
    current = [(112.0, 92.0, 40.0, 60.0), (35.0, 46.0, 14.0, 16.0)]
    windows = [local_search_window_for_box(prev, target) for prev, target in zip(previous, current)]
    targets = make_v9_local_search_targets(current, windows, grid_size=v9.DEFAULT_V9_LOCAL_GRID_SIZE)
    positive_counts = targets.positive_mask.reshape(len(current), -1).sum(dim=1).tolist()
    roundtrip_errors: List[float] = []
    coverage_ok: List[bool] = []
    grid_size = v9.DEFAULT_V9_LOCAL_GRID_SIZE
    for index, (target_box, window) in enumerate(zip(current, windows)):
        x, y, width, height = mot.clamp_bbox_size(target_box)
        wx, wy, ww, wh = mot.clamp_bbox_size(window)
        target_center_x, target_center_y = mot.bbox_center(target_box)
        coverage_ok.append(wx <= target_center_x <= wx + ww and wy <= target_center_y <= wy + wh)
        positive_cells = torch.nonzero(targets.positive_mask[index], as_tuple=False)
        if positive_cells.numel() == 0:
            roundtrip_errors.append(float("inf"))
            continue
        y_index, x_index = positive_cells[0].tolist()
        cell_center_x = wx + ((float(x_index) + 0.5) / float(grid_size)) * max(1.0, ww)
        cell_center_y = wy + ((float(y_index) + 0.5) / float(grid_size)) * max(1.0, wh)
        ltrb = targets.ltrb_targets[index, y_index, x_index].detach().cpu().tolist()
        decoded = (
            cell_center_x - (float(ltrb[0]) * max(1.0, ww)),
            cell_center_y - (float(ltrb[1]) * max(1.0, wh)),
            (float(ltrb[0]) + float(ltrb[2])) * max(1.0, ww),
            (float(ltrb[1]) + float(ltrb[3])) * max(1.0, wh),
        )
        roundtrip_errors.append(max(abs(a - b) for a, b in zip(decoded, (x, y, width, height))))
    print("V9 local target smoke test")
    print(f"windows={windows}")
    print(f"positive_counts={positive_counts}")
    print(f"search_window_coverage={coverage_ok}")
    print(f"encode_decode_roundtrip_max_px={max(roundtrip_errors) if roundtrip_errors else None}")
    print(
        f"score_shape={tuple(targets.score_labels.shape)} "
        f"visibility_shape={tuple(targets.visibility_labels.shape)} "
        f"visibility_positive_cells={int(targets.visibility_labels.sum().item())} "
        f"ltrb_shape={tuple(targets.ltrb_targets.shape)}"
    )
    if not all(coverage_ok):
        raise AssertionError("V9 smoke failed: search window did not contain target center.")
    if roundtrip_errors and max(roundtrip_errors) > 1.0:
        raise AssertionError(f"V9 smoke failed: encode/decode roundtrip error {max(roundtrip_errors):.3f}px.")


def main() -> int:
    args = parse_args()
    if args.smoke_targets:
        smoke_test_targets()
        return 0
    if args.overfit_smoke:
        args.max_sequences = 1 if args.max_sequences <= 0 else min(args.max_sequences, 1)
        args.max_val_sequences = 1 if args.max_val_sequences <= 0 else min(args.max_val_sequences, 1)
        args.disable_mot17 = True
        args.disable_tao = True
        args.disable_lasot = True
        args.max_samples = max(1, int(args.overfit_smoke_samples))
        args.max_val_samples = max(1, int(args.overfit_smoke_samples))
        args.max_train_samples_per_epoch = max(1, int(args.overfit_smoke_samples))
        args.eval_interval_epochs = 1
        args.source_sampling_weights = ""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = build_mixed_dataset(args, train=True)
    val_dataset = build_mixed_dataset(args, train=False)
    if len(train_dataset) == 0:
        raise RuntimeError("No V9 training samples were found. Check dataset roots/splits.")
    print(
        f"V9 training samples total={len(train_dataset)} sources={getattr(train_dataset, 'source_counts', {})}",
        flush=True,
    )
    print(
        f"V9 validation samples total={len(val_dataset)} sources={getattr(val_dataset, 'source_counts', {})}",
        flush=True,
    )

    weight_path = args.weight_path or mot.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    checkpoint_dir = args.checkpoint_dir or (args.output.parent / f"{args.output.stem}_checkpoints")
    diagnostic_csv = args.diagnostic_csv or (checkpoint_dir / f"{args.output.stem}_training_diagnostics.csv")
    resolved_resume_head = resolve_auto_resume_checkpoint(args, checkpoint_dir)
    if resolved_resume_head is not None:
        args.resume_head = resolved_resume_head
    warn_if_resuming_latest_despite_best(args.resume_head, args, checkpoint_dir)
    seed_checkpoint_aliases_from_resume(args.resume_head, args, checkpoint_dir)

    trainer = v9.V9LocalSearchLoRATTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        max_tracks=args.max_objects,
        fps=None,
        sequence_length=None,
        sequence_name="v9-train",
        disable_amp=args.disable_amp,
        frame_size=0,
        head_rank=args.v8_head_rank,
        head_hidden_dim=args.v8_head_hidden_dim,
        head_lora_rank=args.v8_head_lora_rank,
        head_weight_path=args.resume_head,
        search_radius_factor=args.v8_search_radius_factor,
        collect_slot_debug=False,
        collect_week2_proof=False,
        v8_window_penalty_ratio=args.v8_window_penalty_ratio,
        v9_local_grid_size=args.v9_local_grid_size,
    )
    head = trainer.object_conditioned_head
    head.weights_loaded = True
    head.module.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_restored = restore_optimizer_state(args.resume_head, optimizer, trainer.device)
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]] = {}
    resume_epoch, resume_steps, resume_best_val_iou, resume_best_checkpoint_score = load_resume_training_metadata(args.resume_head)
    best_val_iou: Optional[float] = resume_best_val_iou
    best_checkpoint_score: Optional[float] = resume_best_checkpoint_score
    steps = resume_steps
    samples_seen_total = int(resume_steps)
    start_epoch = min(max(1, resume_epoch + 1), max(1, int(args.epochs)))
    if args.resume_head is not None:
        print(
            f"V9 resume head={args.resume_head} metadata_epoch={resume_epoch} "
            f"start_epoch={start_epoch} metadata_steps={resume_steps} "
            f"metadata_best_val_iou={resume_best_val_iou} "
            f"metadata_best_checkpoint_score={resume_best_checkpoint_score} "
            f"optimizer_restored={optimizer_restored}",
            flush=True,
        )
    wall_start = time.perf_counter()
    max_wall_seconds = max(0.0, float(args.max_wall_hours)) * 3600.0

    for epoch in range(start_epoch, max(1, int(args.epochs)) + 1):
        phase = train_common.training_phase_settings(args, epoch)
        order, epoch_source_counts, source_weights = build_epoch_sample_order(train_dataset, args, epoch)
        print(
            f"v9 epoch_sources epoch={epoch} samples={len(order)} "
            f"counts={format_source_counts(epoch_source_counts)} weights={source_weights}",
            flush=True,
        )
        epoch_start = time.perf_counter()
        epoch_steps = 0
        running = {
            "loss": 0.0,
            "objectness": 0.0,
            "box": 0.0,
            "ltrb": 0.0,
            "visibility": 0.0,
            "reid": 0.0,
            "dcfst": 0.0,
            "assignment": 0.0,
            "candidate_rank": 0.0,
            "positive_cells": 0.0,
            "hard_negative_cells": 0.0,
            "switch_candidate_cells": 0.0,
            "drift_negative_objects": 0.0,
            "missing_targets": 0.0,
            "objects": 0.0,
        }
        epoch_drift_negative_objects = 0
        epoch_missing_targets = 0
        epoch_object_instances = 0
        stop_for_wall = False
        for sample_index in order:
            if max_wall_seconds > 0.0 and (time.perf_counter() - wall_start) >= max_wall_seconds:
                stop_for_wall = True
                break
            sample = train_dataset[sample_index]
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            current_frame = train_common.try_load_frame(
                sample.image_paths[frame_index],
                f"v9-train sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            augmentation_rng = train_common.deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"v9-epoch-{epoch}")
            augmentation_spec = train_common.make_lorat_augmentation_spec(augmentation_rng, bool(args.lorat_augmentation))
            train_objects = train_common.apply_search_bbox_augmentation(sample.objects, int(current_frame.shape[1]), augmentation_spec)
            if augmentation_spec.enabled:
                current_frame = train_common.apply_lorat_image_augmentation(current_frame, augmentation_spec, "search")
            drift_negative_count = 0
            if epoch >= max(1, int(args.drift_negative_start_epoch)):
                drift_rng = train_common.deterministic_rng(
                    sample.sequence_path,
                    sample.frame_number,
                    steps,
                    f"v9-drift-negative-{epoch}",
                )
                train_objects, drift_negative_count = add_drift_negative_training_objects(
                    train_objects,
                    current_frame.shape,
                    args,
                    drift_rng,
                )
            with torch.no_grad():
                frame_features = trainer.shared_frame_encoder.encode(current_frame).feature_map.detach()
            selected_banks = train_common.build_selected_banks(
                trainer,
                sample,
                train_objects,
                template_feature_cache,
                augmentation_spec,
                memory_slots=args.training_memory_slots,
            )
            if selected_banks is None:
                continue
            head_output = v9_training_head_output(trainer, frame_features, selected_banks, train_objects, current_frame.shape)
            if phase.closed_loop_probability > 0.0:
                rollout_rng = train_common.deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"v9-closed-loop-{epoch}")
                predictions = decode_v9_predictions(trainer, head_output)
                updated_objects: List[train_common.TrainingObject] = []
                for item, prediction in zip(train_objects, predictions):
                    if item.is_present and float(rollout_rng.random()) <= phase.closed_loop_probability:
                        predicted_previous = train_common.clamp_bbox_to_frame_shape(prediction, current_frame.shape)
                        search_anchor = train_common.union_bbox_xywh(predicted_previous, item.current_bbox)
                        search_bbox = train_common.siamfc_search_bbox(
                            search_anchor,
                            (224, 224),
                            args.search_area_factor,
                            args.search_scale_jitter,
                            args.search_translation_jitter,
                            args.search_min_object_size,
                            rollout_rng,
                        )
                        if args.repair_search_to_target:
                            search_bbox = train_common.repair_search_bbox_to_cover_target(
                                search_bbox,
                                item.current_bbox,
                                args.search_target_padding_fraction,
                            )
                        updated_objects.append(replace(item, previous_bbox=predicted_previous, search_bbox=search_bbox))
                    else:
                        updated_objects.append(item)
                train_objects = updated_objects
                head_output = v9_training_head_output(trainer, frame_features, selected_banks, train_objects, current_frame.shape)

            target_boxes = [item.current_bbox for item in train_objects]
            search_windows = [training_search_window(item, trainer.search_radius_factor) for item in train_objects]
            targets = make_v9_local_search_targets(
                target_boxes,
                search_windows,
                present=[item.is_present for item in train_objects],
                distractor_bboxes=[item.distractor_bboxes for item in train_objects],
                target_kinds=[item.target_kind for item in train_objects],
                grid_size=args.v9_local_grid_size,
                device=trainer.device,
                center_positive_weight=args.center_positive_weight,
                small_target_loss_weight=args.small_target_loss_weight,
                small_target_area_threshold=args.small_target_area_threshold,
                small_target_max_side=args.small_target_max_side,
            )
            objectness_logits = head_output.score_maps.to(torch.float32)
            objectness_map = F.binary_cross_entropy_with_logits(objectness_logits, targets.score_labels, reduction="none")
            if args.focal_loss_gamma > 0.0:
                with torch.no_grad():
                    probability = torch.sigmoid(objectness_logits)
                    focal_weight = torch.abs(targets.score_labels - probability).clamp_min(1e-4).pow(float(args.focal_loss_gamma))
                objectness_map = objectness_map * focal_weight
            objectness_weights = torch.ones_like(objectness_map)
            objectness_weights[targets.loss_mask & ~targets.positive_mask] = max(0.0, float(args.negative_loss_weight))
            if phase.hard_negative_loss_weight != 1.0 and targets.hard_negative_mask.any():
                objectness_weights[targets.hard_negative_mask] = max(0.0, float(phase.hard_negative_loss_weight))
            objectness_weights[targets.positive_mask] = targets.positive_weights[targets.positive_mask].clamp_min(1.0)
            positive_count = targets.positive_mask.sum().clamp_min(1).to(torch.float32)
            objectness_loss = (objectness_map * objectness_weights)[targets.loss_mask].sum() / positive_count

            visibility_logits = head_output.visibility_maps.to(torch.float32)
            visibility_map = F.binary_cross_entropy_with_logits(visibility_logits, targets.visibility_labels, reduction="none")
            visibility_weights = targets.visibility_weights.clone()
            if targets.positive_mask.any():
                visibility_weights[targets.positive_mask] *= max(0.0, float(args.visibility_positive_weight))
            if targets.hard_negative_mask.any():
                visibility_weights[targets.hard_negative_mask] *= max(0.0, float(args.visibility_hard_negative_weight))
            missing_visibility_mask = (~targets.present_mask)[:, None, None].expand_as(visibility_weights)
            if missing_visibility_mask.any():
                visibility_weights[missing_visibility_mask] *= max(0.0, float(args.visibility_missing_weight))
            visibility_weight_sum = visibility_weights[targets.visibility_loss_mask].sum().clamp_min(1.0)
            visibility_loss = (visibility_map * visibility_weights)[targets.visibility_loss_mask].sum() / visibility_weight_sum

            decoded_boxes_xyxy = decode_v9_box_maps_xyxy(trainer, head_output)
            if targets.positive_mask.any():
                positive_weights = targets.positive_weights[targets.positive_mask].clamp_min(1e-4)
                weight_sum = positive_weights.sum().clamp_min(1.0)
                giou = train_common.generalized_iou_aligned(
                    torch,
                    decoded_boxes_xyxy[targets.positive_mask],
                    targets.target_boxes_xyxy[:, None, None, :].expand_as(decoded_boxes_xyxy)[targets.positive_mask],
                )
                box_loss = ((1.0 - giou) * positive_weights).sum() / weight_sum
                ltrb_loss_map = F.smooth_l1_loss(
                    torch.sigmoid(head_output.box_delta_maps.to(torch.float32)),
                    targets.ltrb_targets,
                    reduction="none",
                ).sum(dim=-1)
                ltrb_loss = (ltrb_loss_map[targets.positive_mask] * positive_weights).sum() / weight_sum
            else:
                box_loss = head_output.box_delta_maps.sum() * 0.0
                ltrb_loss = head_output.box_delta_maps.sum() * 0.0

            if phase.reid_loss_weight > 0.0:
                reid_loss = train_common.contrastive_reid_loss(torch, trainer, head, selected_banks, frame_features, train_objects, current_frame.shape)
            else:
                reid_loss = head_output.score_maps.sum() * 0.0
            dcfst_loss, assignment_loss = local_ranking_losses(trainer, head_output, targets, args.assignment_margin)
            if epoch >= max(1, int(args.candidate_ranking_start_epoch)) and args.candidate_ranking_loss_weight > 0.0:
                candidate_ranking_loss, candidate_ranking_terms, switch_candidate_cells = selected_target_candidate_ranking_loss(
                    trainer,
                    head_output,
                    decoded_boxes_xyxy,
                    targets,
                    train_objects,
                    margin=args.candidate_ranking_margin,
                    topk=args.candidate_ranking_topk,
                    positive_iou_threshold=args.candidate_ranking_positive_iou,
                    other_iou_threshold=args.candidate_ranking_other_iou,
                )
            else:
                candidate_ranking_loss = head_output.score_maps.sum() * 0.0
                candidate_ranking_terms = 0
                switch_candidate_cells = 0
            loss = (
                objectness_loss
                + (args.box_loss_weight * box_loss)
                + (args.ltrb_loss_weight * ltrb_loss)
                + (max(0.0, float(args.visibility_loss_weight)) * visibility_loss)
                + (phase.reid_loss_weight * reid_loss)
                + (phase.dcfst_discrimination_weight * dcfst_loss)
                + (phase.assignment_discrimination_weight * assignment_loss)
                + (max(0.0, float(args.candidate_ranking_loss_weight)) * candidate_ranking_loss)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
            optimizer.step()

            steps += 1
            epoch_steps += 1
            samples_seen_total += len(train_objects)
            running["loss"] += float(loss.detach().item())
            running["objectness"] += float(objectness_loss.detach().item())
            running["box"] += float(box_loss.detach().item())
            running["ltrb"] += float(ltrb_loss.detach().item())
            running["visibility"] += float(visibility_loss.detach().item())
            running["reid"] += float(reid_loss.detach().item())
            running["dcfst"] += float(dcfst_loss.detach().item())
            running["assignment"] += float(assignment_loss.detach().item())
            running["candidate_rank"] += float(candidate_ranking_loss.detach().item())
            running["positive_cells"] += float(targets.positive_cells)
            running["hard_negative_cells"] += float(targets.hard_negative_cells)
            running["switch_candidate_cells"] += float(switch_candidate_cells)
            running["drift_negative_objects"] += float(drift_negative_count)
            running["missing_targets"] += float(targets.missing_targets)
            running["objects"] += float(len(train_objects))
            epoch_drift_negative_objects += int(drift_negative_count)
            epoch_missing_targets += int(targets.missing_targets)
            epoch_object_instances += int(len(train_objects))

            if steps % 25 == 0:
                denom = 25.0
                print(
                    f"v9 epoch={epoch} step={steps} loss={running['loss'] / denom:.4f} "
                    f"obj={running['objectness'] / denom:.4f} box={running['box'] / denom:.4f} "
                    f"ltrb={running['ltrb'] / denom:.4f} vis={running['visibility'] / denom:.4f} "
                    f"reid={running['reid'] / denom:.4f} "
                    f"dcfst={running['dcfst'] / denom:.4f} assign={running['assignment'] / denom:.4f} "
                    f"cand_rank={running['candidate_rank'] / denom:.4f} "
                    f"pos_cells={running['positive_cells'] / denom:.1f} hard_neg={running['hard_negative_cells'] / denom:.1f} "
                    f"switch_cells={running['switch_candidate_cells'] / denom:.1f} "
                    f"drift_neg={running['drift_negative_objects'] / denom:.1f} "
                    f"missing={running['missing_targets'] / denom:.1f} objects={running['objects'] / denom:.1f} "
                    f"samples_seen={samples_seen_total} "
                    f"rank_terms={candidate_ranking_terms} "
                    f"phase={phase.name}",
                    flush=True,
                )
                for key in running:
                    running[key] = 0.0
            if (
                args.checkpoint_interval > 0
                and args.checkpoint_retention != "none"
                and steps % args.checkpoint_interval == 0
            ):
                if args.checkpoint_retention == "all":
                    save_v9_checkpoint(
                        checkpoint_dir / f"{args.output.stem}_epoch{epoch:03d}_step{steps:07d}.pt",
                        trainer,
                        args,
                        epoch,
                        steps,
                        None,
                        None,
                        optimizer,
                        best_val_iou,
                        best_checkpoint_score,
                        samples_seen_total=samples_seen_total,
                    )
                save_v9_checkpoint(
                    checkpoint_dir / f"{args.output.stem}_latest.pt",
                    trainer,
                    args,
                    epoch,
                    steps,
                    None,
                    None,
                    optimizer,
                    best_val_iou,
                    best_checkpoint_score,
                    samples_seen_total=samples_seen_total,
                )
            if args.max_steps > 0 and steps >= args.max_steps:
                stop_for_wall = True
                break

        elapsed = time.perf_counter() - epoch_start
        should_eval = epoch == 1 or stop_for_wall or epoch % max(1, int(args.eval_interval_epochs)) == 0 or epoch == args.epochs
        train_iou = train_iou50 = val_iou = val_iou50 = None
        val_rollout_iou = val_rollout_correct = val_rollout_window_miss = val_rollout_identity_switch = None
        val_rollout_track_loss = val_rollout_frames_until_loss = None
        train_reid_same = train_reid_diff = train_reid_top1 = None
        val_reid_same = val_reid_diff = val_reid_top1 = None
        if should_eval:
            train_iou, train_iou50 = evaluate_v9_head(
                trainer,
                train_dataset,
                args.train_diagnostic_samples,
                template_feature_cache,
                args.training_memory_slots,
            )
            rollout_eval_fn = (
                evaluate_v9_closed_loop_probe
                if args.rollout_validation_mode == "two_pass"
                else evaluate_v9_sequence_rollout_probe
            )
            if args.rollout_validation_mode == "sequence":
                val_rollout_iou, val_rollout_correct, val_rollout_window_miss, val_rollout_identity_switch, val_rollout_track_loss, val_rollout_frames_until_loss = rollout_eval_fn(
                    trainer,
                    val_dataset,
                    args.rollout_diagnostic_samples,
                    template_feature_cache,
                    args.training_memory_slots,
                    args.rollout_clip_frames,
                    args.rollout_max_frame_gap,
                )
            else:
                val_rollout_iou, val_rollout_correct, val_rollout_window_miss, val_rollout_identity_switch, val_rollout_track_loss, val_rollout_frames_until_loss = rollout_eval_fn(
                    trainer,
                    val_dataset,
                    args.rollout_diagnostic_samples,
                    template_feature_cache,
                    args.training_memory_slots,
                )
            train_reid_same, train_reid_diff, train_reid_top1 = train_common.reid_similarity_probe(
                trainer,
                train_dataset,
                args.reid_diagnostic_samples,
                template_feature_cache,
                memory_slots=args.training_memory_slots,
            )
            val_reid_same, val_reid_diff, val_reid_top1 = train_common.reid_similarity_probe(
                trainer,
                val_dataset,
                args.reid_diagnostic_samples,
                template_feature_cache,
                memory_slots=args.training_memory_slots,
            )
            val_iou, val_iou50 = evaluate_v9_head(
                trainer,
                val_dataset,
                args.max_val_samples,
                template_feature_cache,
                args.training_memory_slots,
            )
        selection_score = checkpoint_selection_score(
            val_iou,
            val_rollout_correct,
            val_rollout_identity_switch,
            val_rollout_window_miss,
            val_rollout_track_loss,
            identity_switch_penalty=args.rollout_identity_switch_penalty,
            window_miss_penalty=args.rollout_window_miss_penalty,
            track_loss_penalty=args.rollout_track_loss_penalty,
        )
        save_v9_checkpoint(
            args.output,
            trainer,
            args,
            epoch,
            steps,
            train_iou,
            val_iou,
            optimizer,
            best_val_iou,
            best_checkpoint_score,
            selection_score,
            val_rollout_correct,
            val_rollout_identity_switch,
            val_rollout_track_loss,
            val_rollout_frames_until_loss,
            samples_seen_total=samples_seen_total,
        )
        save_v9_checkpoint(
            checkpoint_dir / f"{args.output.stem}_latest.pt",
            trainer,
            args,
            epoch,
            steps,
            train_iou,
            val_iou,
            optimizer,
            best_val_iou,
            best_checkpoint_score,
            selection_score,
            val_rollout_correct,
            val_rollout_identity_switch,
            val_rollout_track_loss,
            val_rollout_frames_until_loss,
            samples_seen_total=samples_seen_total,
        )
        if val_iou is not None and (best_val_iou is None or val_iou > best_val_iou):
            best_val_iou = float(val_iou)
            save_v9_checkpoint(
                checkpoint_dir / f"{args.output.stem}_best_by_val_iou.pt",
                trainer,
                args,
                epoch,
                steps,
                train_iou,
                val_iou,
                optimizer,
                best_val_iou,
                best_checkpoint_score,
                selection_score,
                val_rollout_correct,
                val_rollout_identity_switch,
                val_rollout_track_loss,
                val_rollout_frames_until_loss,
                samples_seen_total=samples_seen_total,
            )
        if selection_score is not None and (
            best_checkpoint_score is None or selection_score > best_checkpoint_score
        ):
            best_checkpoint_score = float(selection_score)
            save_v9_checkpoint(
                checkpoint_dir / f"{args.output.stem}_best_by_rollout_identity.pt",
                trainer,
                args,
                epoch,
                steps,
                train_iou,
                val_iou,
                optimizer,
                best_val_iou,
                best_checkpoint_score,
                selection_score,
                val_rollout_correct,
                val_rollout_identity_switch,
                val_rollout_track_loss,
                val_rollout_frames_until_loss,
                samples_seen_total=samples_seen_total,
            )
        append_csv_row(
            diagnostic_csv,
            {
                "epoch": epoch,
                "steps": steps,
                "samples_seen_total": samples_seen_total,
                "effective_batch_or_sample_count": samples_seen_total / float(max(1, steps)),
                "epoch_steps": epoch_steps,
                "epoch_object_instances": epoch_object_instances,
                "epoch_drift_negative_objects": epoch_drift_negative_objects,
                "epoch_missing_targets": epoch_missing_targets,
                "epoch_seconds": elapsed,
                "seconds_per_step": elapsed / float(max(1, epoch_steps)),
                "train_mean_iou": train_iou,
                "train_iou50": train_iou50,
                "val_mean_iou": val_iou,
                "val_iou50": val_iou50,
                "val_rollout_mean_iou": val_rollout_iou,
                "val_rollout_correct_rate_iou30": val_rollout_correct,
                "val_rollout_window_miss_rate": val_rollout_window_miss,
                "val_rollout_identity_switch_rate": val_rollout_identity_switch,
                "val_rollout_track_loss_rate": val_rollout_track_loss,
                "val_rollout_mean_frames_until_loss": val_rollout_frames_until_loss,
                "rollout_validation_mode": args.rollout_validation_mode,
                "checkpoint_selection_score": selection_score,
                "best_checkpoint_score": best_checkpoint_score,
                "train_reid_same_similarity": train_reid_same,
                "train_reid_diff_similarity": train_reid_diff,
                "train_reid_top1": train_reid_top1,
                "val_reid_same_similarity": val_reid_same,
                "val_reid_diff_similarity": val_reid_diff,
                "val_reid_top1": val_reid_top1,
                "best_val_iou": best_val_iou,
                "phase": phase.name,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "v9_local_grid_size": args.v9_local_grid_size,
                "max_train_samples_per_epoch": args.max_train_samples_per_epoch,
                "candidate_ranking_loss_weight": args.candidate_ranking_loss_weight,
                "candidate_ranking_start_epoch": args.candidate_ranking_start_epoch,
                "candidate_ranking_margin": args.candidate_ranking_margin,
                "candidate_ranking_topk": args.candidate_ranking_topk,
                "candidate_ranking_positive_iou": args.candidate_ranking_positive_iou,
                "candidate_ranking_other_iou": args.candidate_ranking_other_iou,
                "visibility_loss_weight": args.visibility_loss_weight,
                "visibility_positive_weight": args.visibility_positive_weight,
                "visibility_hard_negative_weight": args.visibility_hard_negative_weight,
                "visibility_missing_weight": args.visibility_missing_weight,
                "drift_negative_probability": args.drift_negative_probability,
                "drift_negative_start_epoch": args.drift_negative_start_epoch,
                "drift_negative_max_per_frame": args.drift_negative_max_per_frame,
                "drift_negative_shift_factor": args.drift_negative_shift_factor,
                "rollout_identity_switch_penalty": args.rollout_identity_switch_penalty,
                "rollout_window_miss_penalty": args.rollout_window_miss_penalty,
                "rollout_track_loss_penalty": args.rollout_track_loss_penalty,
                "rollout_clip_frames": args.rollout_clip_frames,
                "rollout_max_frame_gap": args.rollout_max_frame_gap,
                "epoch_source_counts": format_source_counts(epoch_source_counts),
                "source_sampling_weights": args.source_sampling_weights,
                "elapsed_hours": (time.perf_counter() - wall_start) / 3600.0,
            },
        )
        print(
            f"v9 epoch_timing epoch={epoch} seconds={elapsed:.2f} steps={epoch_steps} "
            f"train_iou={train_iou} val_iou={val_iou} rollout_iou={val_rollout_iou} "
            f"rollout_mode={args.rollout_validation_mode} "
            f"rollout_correct={val_rollout_correct} rollout_window_miss={val_rollout_window_miss} "
            f"rollout_switch={val_rollout_identity_switch} rollout_track_loss={val_rollout_track_loss} "
            f"rollout_frames_until_loss={val_rollout_frames_until_loss} selection_score={selection_score} "
            f"best_checkpoint_score={best_checkpoint_score} val_reid_top1={val_reid_top1} "
            f"best_val_iou={best_val_iou}",
            flush=True,
        )
        if stop_for_wall:
            print("V9 training stopped cleanly for wall clock/step limit after checkpoint save.", flush=True)
            break

    print(f"Saved V9 head checkpoint to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
