"""Train the V9 LoRaT-MOT local-search head.

V9 keeps the Week 2 shared-frame property, but moves the supervised head back
toward LoRaT's template/search geometry:

* one frozen LoRaT/DINOv2 frame encoder pass per frame;
* one fixed local search grid sampled from that shared feature map per object;
* one batched object-conditioned LoRA head call across all objects;
* local l/t/r/b box targets normalized by each object's search window.

The dataset construction intentionally reuses the V8 training adapters for
DanceTrack/MOT-style data and TAO/TAO-OW JSON annotations. The part that changes
is the head path and target coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import random
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
import train_lorat_v8_head as v8train

BBox = mot.BBox


@dataclass(frozen=True)
class V9LocalSearchTargets:
    score_labels: torch.Tensor
    positive_mask: torch.Tensor
    hard_negative_mask: torch.Tensor
    loss_mask: torch.Tensor
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


def training_search_window(item: v8train.V8TrainingObject, radius_factor: float) -> BBox:
    """Use the V8 dataset's LoRaT-style search box when available.

    The V8 data adapter already samples search windows from first/previous/mixed
    templates, applies jitter, and repairs windows to cover the selected target.
    For V9 we preserve that sampling instead of recreating it later.
    """

    if item.search_bbox is not None:
        return mot.clamp_bbox_size(item.search_bbox)
    target = item.current_bbox if item.is_present else None
    return local_search_window_for_box(item.previous_bbox, target, radius_factor=radius_factor)


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
    positive_mask = torch.zeros_like(score_labels, dtype=torch.bool)
    hard_negative_mask = torch.zeros_like(positive_mask)
    loss_mask = torch.ones_like(positive_mask)
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
            positive_mask,
            hard_negative_mask,
            loss_mask,
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
    gaussian = torch.exp(-center_distance.pow(2) / (2.0 * max(0.1, float(positive_radius_cells)) ** 2))
    gaussian = gaussian / gaussian.reshape(object_count, -1).amax(dim=1).clamp_min(1e-6)[:, None, None]
    blend = max(0.0, min(1.0, float(center_positive_weight)))
    positive_weights = (1.0 - blend) + (blend * gaussian)

    target_kinds = list(target_kinds or ["full"] * object_count)
    for index, (bbox, kind) in enumerate(zip(target_boxes, target_kinds)):
        if not bool(present_mask[index].item()):
            continue
        probe = v8train.V8TrainingObject(
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
        if v8train.is_small_target_object(probe, small_target_area_threshold, small_target_max_side):
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

    return V9LocalSearchTargets(
        score_labels=score_labels,
        positive_mask=positive_mask,
        hard_negative_mask=hard_negative_mask,
        loss_mask=loss_mask,
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
    objects: Sequence[v8train.V8TrainingObject],
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": trainer.object_conditioned_head.state_dict(),
            "epoch": int(epoch),
            "steps": int(steps),
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
            "train_mean_iou": train_mean_iou,
            "val_mean_iou": val_mean_iou,
            "command": " ".join(sys.argv),
            "torch_version": getattr(torch, "__version__", ""),
            "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        str(path),
    )


class LaSOTFrameHeadDataset:
    """LaSOT-style SOT samples using img/ plus groundtruth/occlusion files.

    LaSOT is single-target tracking data rather than MOT data. For V9 that is a
    feature, not a limitation: it provides real template/search supervision for
    "track the selected target" behavior that the full-frame V8 path struggled
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
        self.region_specs = v8train.region_specs_for_mode(target_region_mode)
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
        self.samples: List[v8train.V8TrainingSample] = []
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
                objects: List[v8train.V8TrainingObject] = []
                for spec in v8train.stable_region_order(self.region_specs, 1, frame_number)[: self.target_regions_per_object]:
                    rng = v8train.deterministic_rng(sequence_path, frame_number, 1, spec.name)
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
                    current_bbox = v8train.selected_region_bbox(current_bbox_full, spec)
                    previous_bbox = v8train.selected_region_bbox(previous_bbox_full, spec)
                    template_bbox = v8train.selected_region_bbox(template_bbox_full, spec)
                    noisy_previous_bbox = v8train.jitter_reference_bbox(previous_bbox, rng, self.previous_box_jitter)
                    if self.search_anchor_mode == "previous":
                        search_anchor_bbox = noisy_previous_bbox
                    elif self.search_anchor_mode == "union":
                        search_anchor_bbox = v8train.union_bbox_xywh(noisy_previous_bbox, current_bbox)
                    else:
                        search_anchor_bbox = current_bbox
                    search_bbox = v8train.siamfc_search_bbox(
                        search_anchor_bbox,
                        (224, 224),
                        self.search_area_factor,
                        self.search_scale_jitter,
                        self.search_translation_jitter,
                        self.search_min_object_size,
                        rng,
                    )
                    if self.repair_search_to_target:
                        search_bbox = v8train.repair_search_bbox_to_cover_target(
                            search_bbox,
                            current_bbox,
                            self.search_target_padding_fraction,
                        )
                    objects.append(
                        v8train.V8TrainingObject(
                            track_id=1,
                            current_bbox=current_bbox,
                            previous_bbox=noisy_previous_bbox,
                            previous_frame=previous_frame,
                            previous_context_bbox=v8train.siamfc_context_bbox(noisy_previous_bbox, self.template_area_factor),
                            template_frame=template_frame,
                            template_bbox=template_bbox,
                            template_context_bbox=v8train.siamfc_context_bbox(template_bbox, self.template_area_factor),
                            search_bbox=search_bbox,
                            target_kind=spec.name,
                            is_present=True,
                        )
                    )
                    if len(objects) >= self.max_objects:
                        break
                if objects:
                    self.samples.append(
                        v8train.V8TrainingSample(
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

    def __getitem__(self, index: int) -> v8train.V8TrainingSample:
        return self.samples[index]


def build_mixed_dataset(args: argparse.Namespace, *, train: bool) -> v8train.CombinedFrameHeadDataset:
    class_ids = v8train.parse_class_ids(args.class_id)
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
        dataset = v8train.MOTFrameHeadDataset(
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
        tao_dataset = v8train.TAOFrameHeadDataset(
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
    return v8train.CombinedFrameHeadDataset(datasets)


def evaluate_v9_head(
    trainer: v9.V9LocalSearchLoRATTracker,
    dataset: v8train.CombinedFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    memory_slots: int,
) -> Tuple[Optional[float], Optional[float]]:
    if len(dataset) == 0 or max_samples == 0:
        return None, None
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()
    sample_indices = v8train.spread_sample_indices(len(dataset), max_samples)
    iou_sum = 0.0
    hit50 = 0
    count = 0
    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            frame = v8train.try_load_frame(sample.image_paths[frame_index], f"v9-eval {sample.sequence_path.name} frame={sample.frame_number}")
            if frame is None:
                continue
            frame_features = trainer.shared_frame_encoder.encode(frame).feature_map.detach()
            objects = list(sample.objects)
            selected_banks = v8train.build_selected_banks(
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
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-wall-hours", type=float, default=0.0)
    parser.add_argument("--eval-interval-epochs", type=int, default=5)
    parser.add_argument("--train-diagnostic-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--ltrb-loss-weight", type=float, default=1.0)
    parser.add_argument("--reid-loss-weight", type=float, default=0.35)
    parser.add_argument("--center-positive-weight", type=float, default=0.5)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=3.0)
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--small-target-loss-weight", type=float, default=3.0)
    parser.add_argument("--small-target-area-threshold", type=float, default=v9.DEFAULT_V8_SMALL_TARGET_AREA)
    parser.add_argument("--small-target-max-side", type=float, default=v9.DEFAULT_V8_SMALL_TARGET_MAX_SIDE)
    parser.add_argument("--dcfst-discrimination-weight", type=float, default=0.50)
    parser.add_argument("--assignment-discrimination-weight", type=float, default=0.50)
    parser.add_argument("--assignment-margin", type=float, default=0.25)
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
    parser.add_argument("--v8-window-penalty-ratio", type=float, default=v9.DEFAULT_V8_WINDOW_PENALTY_RATIO)
    parser.add_argument("--v9-local-grid-size", type=int, default=v9.DEFAULT_V9_LOCAL_GRID_SIZE)
    parser.add_argument("--training-memory-slots", type=int, default=2)
    parser.add_argument("--resume-head", type=Path)
    parser.add_argument("--output", type=Path, default=mot.PROJECT_ROOT / "models" / "lorat" / "v9_local_head.pt")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
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
    targets = make_v9_local_search_targets(current, windows, grid_size=16)
    positive_counts = targets.positive_mask.reshape(len(current), -1).sum(dim=1).tolist()
    print("V9 local target smoke test")
    print(f"windows={windows}")
    print(f"positive_counts={positive_counts}")
    print(f"score_shape={tuple(targets.score_labels.shape)} ltrb_shape={tuple(targets.ltrb_targets.shape)}")


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

    checkpoint_dir = args.checkpoint_dir or (args.output.parent / f"{args.output.stem}_checkpoints")
    diagnostic_csv = args.diagnostic_csv or (checkpoint_dir / f"{args.output.stem}_training_diagnostics.csv")
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]] = {}
    best_val_iou: Optional[float] = None
    steps = 0
    wall_start = time.perf_counter()
    max_wall_seconds = max(0.0, float(args.max_wall_hours)) * 3600.0

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        phase = v8train.training_phase_settings(args, epoch)
        order = list(range(len(train_dataset)))
        random.shuffle(order)
        if args.max_train_samples_per_epoch > 0:
            order = order[: min(len(order), int(args.max_train_samples_per_epoch))]
        epoch_start = time.perf_counter()
        epoch_steps = 0
        running = {
            "loss": 0.0,
            "objectness": 0.0,
            "box": 0.0,
            "ltrb": 0.0,
            "reid": 0.0,
            "dcfst": 0.0,
            "assignment": 0.0,
            "positive_cells": 0.0,
            "hard_negative_cells": 0.0,
        }
        stop_for_wall = False
        for sample_index in order:
            if max_wall_seconds > 0.0 and (time.perf_counter() - wall_start) >= max_wall_seconds:
                stop_for_wall = True
                break
            sample = train_dataset[sample_index]
            frame_index = exercise.frame_to_image_index(sample.frame_number)
            if frame_index < 0 or frame_index >= len(sample.image_paths):
                continue
            current_frame = v8train.try_load_frame(
                sample.image_paths[frame_index],
                f"v9-train sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            augmentation_rng = v8train.deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"v9-epoch-{epoch}")
            augmentation_spec = v8train.make_lorat_augmentation_spec(augmentation_rng, bool(args.lorat_augmentation))
            train_objects = v8train.apply_search_bbox_augmentation(sample.objects, int(current_frame.shape[1]), augmentation_spec)
            if augmentation_spec.enabled:
                current_frame = v8train.apply_lorat_image_augmentation(current_frame, augmentation_spec, "search")
            with torch.no_grad():
                frame_features = trainer.shared_frame_encoder.encode(current_frame).feature_map.detach()
            selected_banks = v8train.build_selected_banks(
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
                rollout_rng = v8train.deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"v9-closed-loop-{epoch}")
                predictions = decode_v9_predictions(trainer, head_output)
                updated_objects: List[v8train.V8TrainingObject] = []
                for item, prediction in zip(train_objects, predictions):
                    if item.is_present and float(rollout_rng.random()) <= phase.closed_loop_probability:
                        predicted_previous = v8train.clamp_bbox_to_frame_shape(prediction, current_frame.shape)
                        search_anchor = v8train.union_bbox_xywh(predicted_previous, item.current_bbox)
                        search_bbox = v8train.siamfc_search_bbox(
                            search_anchor,
                            (224, 224),
                            args.search_area_factor,
                            args.search_scale_jitter,
                            args.search_translation_jitter,
                            args.search_min_object_size,
                            rollout_rng,
                        )
                        if args.repair_search_to_target:
                            search_bbox = v8train.repair_search_bbox_to_cover_target(
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

            decoded_boxes_xyxy = decode_v9_box_maps_xyxy(trainer, head_output)
            if targets.positive_mask.any():
                positive_weights = targets.positive_weights[targets.positive_mask].clamp_min(1e-4)
                weight_sum = positive_weights.sum().clamp_min(1.0)
                giou = v8train.generalized_iou_aligned(
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
                reid_loss = v8train.contrastive_reid_loss(torch, trainer, head, selected_banks, frame_features, train_objects, current_frame.shape)
            else:
                reid_loss = head_output.score_maps.sum() * 0.0
            dcfst_loss, assignment_loss = local_ranking_losses(trainer, head_output, targets, args.assignment_margin)
            loss = (
                objectness_loss
                + (args.box_loss_weight * box_loss)
                + (args.ltrb_loss_weight * ltrb_loss)
                + (phase.reid_loss_weight * reid_loss)
                + (phase.dcfst_discrimination_weight * dcfst_loss)
                + (phase.assignment_discrimination_weight * assignment_loss)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
            optimizer.step()

            steps += 1
            epoch_steps += 1
            running["loss"] += float(loss.detach().item())
            running["objectness"] += float(objectness_loss.detach().item())
            running["box"] += float(box_loss.detach().item())
            running["ltrb"] += float(ltrb_loss.detach().item())
            running["reid"] += float(reid_loss.detach().item())
            running["dcfst"] += float(dcfst_loss.detach().item())
            running["assignment"] += float(assignment_loss.detach().item())
            running["positive_cells"] += float(targets.positive_cells)
            running["hard_negative_cells"] += float(targets.hard_negative_cells)

            if steps % 25 == 0:
                denom = 25.0
                print(
                    f"v9 epoch={epoch} step={steps} loss={running['loss'] / denom:.4f} "
                    f"obj={running['objectness'] / denom:.4f} box={running['box'] / denom:.4f} "
                    f"ltrb={running['ltrb'] / denom:.4f} reid={running['reid'] / denom:.4f} "
                    f"dcfst={running['dcfst'] / denom:.4f} assign={running['assignment'] / denom:.4f} "
                    f"pos_cells={running['positive_cells'] / denom:.1f} hard_neg={running['hard_negative_cells'] / denom:.1f} "
                    f"phase={phase.name}",
                    flush=True,
                )
                for key in running:
                    running[key] = 0.0
            if args.checkpoint_interval > 0 and steps % args.checkpoint_interval == 0:
                save_v9_checkpoint(checkpoint_dir / f"{args.output.stem}_epoch{epoch:03d}_step{steps:07d}.pt", trainer, args, epoch, steps, None, None)
                save_v9_checkpoint(checkpoint_dir / f"{args.output.stem}_latest.pt", trainer, args, epoch, steps, None, None)
            if args.max_steps > 0 and steps >= args.max_steps:
                stop_for_wall = True
                break

        elapsed = time.perf_counter() - epoch_start
        should_eval = epoch == 1 or stop_for_wall or epoch % max(1, int(args.eval_interval_epochs)) == 0 or epoch == args.epochs
        train_iou = train_iou50 = val_iou = val_iou50 = None
        if should_eval:
            train_iou, train_iou50 = evaluate_v9_head(
                trainer,
                train_dataset,
                args.train_diagnostic_samples,
                template_feature_cache,
                args.training_memory_slots,
            )
            val_iou, val_iou50 = evaluate_v9_head(
                trainer,
                val_dataset,
                args.max_val_samples,
                template_feature_cache,
                args.training_memory_slots,
            )
        save_v9_checkpoint(args.output, trainer, args, epoch, steps, train_iou, val_iou)
        save_v9_checkpoint(checkpoint_dir / f"{args.output.stem}_latest.pt", trainer, args, epoch, steps, train_iou, val_iou)
        if val_iou is not None and (best_val_iou is None or val_iou > best_val_iou):
            best_val_iou = float(val_iou)
            save_v9_checkpoint(checkpoint_dir / f"{args.output.stem}_best_by_val_iou.pt", trainer, args, epoch, steps, train_iou, val_iou)
        append_csv_row(
            diagnostic_csv,
            {
                "epoch": epoch,
                "steps": steps,
                "epoch_steps": epoch_steps,
                "epoch_seconds": elapsed,
                "seconds_per_step": elapsed / float(max(1, epoch_steps)),
                "train_mean_iou": train_iou,
                "train_iou50": train_iou50,
                "val_mean_iou": val_iou,
                "val_iou50": val_iou50,
                "best_val_iou": best_val_iou,
                "phase": phase.name,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "v9_local_grid_size": args.v9_local_grid_size,
                "max_train_samples_per_epoch": args.max_train_samples_per_epoch,
                "elapsed_hours": (time.perf_counter() - wall_start) / 3600.0,
            },
        )
        print(
            f"v9 epoch_timing epoch={epoch} seconds={elapsed:.2f} steps={epoch_steps} "
            f"train_iou={train_iou} val_iou={val_iou} best_val_iou={best_val_iou}",
            flush=True,
        )
        if stop_for_wall:
            print("V9 training stopped cleanly for wall clock/step limit after checkpoint save.", flush=True)
            break

    print(f"Saved V9 head checkpoint to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
