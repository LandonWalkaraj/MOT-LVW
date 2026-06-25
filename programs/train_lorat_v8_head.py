"""Train the V8 shared-frame LoRAT head.

The trainer is intentionally staged:

1. Preserve LoRAT's target-conditioned SOT behavior with template/search-style
   samples and LoRAT-style score-map plus box losses.
2. Teach the V8 head to run on one frozen shared LoRAT/DINOv2 frame feature map.
3. Add selected-target-vs-distractor ranking so selected parts, such as a head
   or face crop, do not drift toward the whole annotated object.
4. Keep ReID/missing-target probes in the same training file so Week 3 recovery
   behavior is trained and measured against the same head.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import bounding_box_v8_lorat_quality_batched as v8
import exercise_lorat_mot as exercise
import mot_common as mot

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class V8SelectedRegionSpec:
    name: str
    rel_x: float
    rel_y: float
    rel_w: float
    rel_h: float


@dataclass(frozen=True)
class V8TrainingObject:
    track_id: int
    current_bbox: BBox
    previous_bbox: BBox
    previous_frame: int
    previous_context_bbox: BBox
    template_frame: int
    template_bbox: BBox
    template_context_bbox: BBox
    search_bbox: BBox
    target_kind: str = "full"
    is_present: bool = True
    distractor_bboxes: Tuple[BBox, ...] = ()


@dataclass(frozen=True)
class V8TrainingSample:
    sequence_path: Path
    image_paths: Sequence[Path]
    frame_number: int
    objects: Sequence[V8TrainingObject]


@dataclass(frozen=True)
class V8TrainingTargetStats:
    positive_cells: int
    loss_cells: int
    positive_cells_outside_search: int
    hard_negative_cells: int
    missing_targets: int
    present_targets: int = 0
    target_center_outside_search: int = 0
    mean_target_search_coverage: float = 1.0
    min_target_search_coverage: float = 1.0
    mean_search_target_area_ratio: float = 0.0


@dataclass(frozen=True)
class V8CandidateDiscriminationStats:
    objects: int = 0
    positive_candidates: int = 0
    negative_candidates: int = 0


@dataclass(frozen=True)
class V8AssignmentDiscriminationStats:
    objects: int = 0
    positive_candidates: int = 0
    negative_candidates: int = 0


@dataclass(frozen=True)
class V8TrainingPhaseSettings:
    name: str
    hard_negative_loss_weight: float
    reid_loss_weight: float
    dcfst_discrimination_weight: float
    assignment_discrimination_weight: float
    closed_loop_probability: float


@dataclass(frozen=True)
class LoRATTrainingAugmentationSpec:
    enabled: bool
    template_flip: bool = False
    search_flip: bool = False
    template_brightness: float = 1.0
    template_contrast: float = 1.0
    template_saturation: float = 1.0
    search_brightness: float = 1.0
    search_contrast: float = 1.0
    search_saturation: float = 1.0
    joint_deit_op: str = "none"
    joint_blur_kernel: int = 3


FULL_REGION_SPEC = V8SelectedRegionSpec("full", 0.0, 0.0, 1.0, 1.0)
SELECTED_REGION_SPECS: Tuple[V8SelectedRegionSpec, ...] = (
    FULL_REGION_SPEC,
    V8SelectedRegionSpec("upper", 0.15, 0.00, 0.70, 0.52),
    V8SelectedRegionSpec("head_like_top", 0.28, 0.00, 0.44, 0.28),
    V8SelectedRegionSpec("face_like_top", 0.34, 0.00, 0.32, 0.20),
    V8SelectedRegionSpec("tiny_head", 0.38, 0.00, 0.24, 0.16),
    V8SelectedRegionSpec("center", 0.25, 0.25, 0.50, 0.50),
    V8SelectedRegionSpec("small_center", 0.35, 0.20, 0.30, 0.34),
    V8SelectedRegionSpec("small_upper_center", 0.32, 0.08, 0.36, 0.30),
    V8SelectedRegionSpec("left_half", 0.00, 0.12, 0.52, 0.76),
    V8SelectedRegionSpec("right_half", 0.48, 0.12, 0.52, 0.76),
)
SMALL_TARGET_REGION_NAMES = {
    "head_like_top",
    "face_like_top",
    "tiny_head",
    "small_center",
    "small_upper_center",
}


def normalized_target_kind(target_kind: str) -> str:
    text = str(target_kind or "full")
    if text.startswith("tao_"):
        return text[4:]
    return text


def is_small_target_object(item: V8TrainingObject, area_threshold: float, max_side: float) -> bool:
    base_kind = normalized_target_kind(item.target_kind)
    _, _, width, height = item.current_bbox
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    return (
        base_kind in SMALL_TARGET_REGION_NAMES
        or (width * height) <= max(1.0, float(area_threshold))
        or max(width, height) <= max(1.0, float(max_side))
    )


def region_specs_for_mode(mode: str) -> Tuple[V8SelectedRegionSpec, ...]:
    mode = str(mode or "mixed").strip().lower()
    if mode == "full":
        return (FULL_REGION_SPEC,)
    if mode == "parts":
        return tuple(spec for spec in SELECTED_REGION_SPECS if spec.name != "full")
    if mode == "mixed":
        return SELECTED_REGION_SPECS
    raise ValueError(f"Unknown target region mode: {mode!r}")


def selected_region_bbox(full_bbox: BBox, spec: V8SelectedRegionSpec) -> BBox:
    x, y, w, h = full_bbox
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    sub_w = max(1.0, w * max(0.02, min(1.0, float(spec.rel_w))))
    sub_h = max(1.0, h * max(0.02, min(1.0, float(spec.rel_h))))
    sub_x = x + (w * max(0.0, min(1.0, float(spec.rel_x))))
    sub_y = y + (h * max(0.0, min(1.0, float(spec.rel_y))))
    if sub_x + sub_w > x + w:
        sub_x = (x + w) - sub_w
    if sub_y + sub_h > y + h:
        sub_y = (y + h) - sub_h
    return float(sub_x), float(sub_y), float(sub_w), float(sub_h)


def siamfc_context_bbox(bbox: BBox, area_factor: float) -> BBox:
    x, y, w, h = bbox
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    context_w = w + (float(area_factor) - 1.0) * ((w + h) * 0.5)
    context_h = h + (float(area_factor) - 1.0) * ((w + h) * 0.5)
    center_x, center_y = bbox_center(bbox)
    return center_x - (context_w / 2.0), center_y - (context_h / 2.0), max(1.0, context_w), max(1.0, context_h)


def deterministic_rng(sequence_path: Path, frame_number: int, track_id: int, target_kind: str) -> np.random.Generator:
    seed_text = f"{sequence_path.as_posix()}|{int(frame_number)}|{int(track_id)}|{target_kind}"
    seed = 2166136261
    for byte in seed_text.encode("utf-8"):
        seed ^= byte
        seed = (seed * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def siamfc_search_bbox(
    bbox: BBox,
    output_size: Tuple[int, int],
    area_factor: float,
    scale_jitter: float,
    translation_jitter: float,
    min_object_size: float,
    rng: np.random.Generator,
    retry_count: int = 10,
) -> BBox:
    x, y, w, h = bbox
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    output_w, output_h = float(output_size[0]), float(output_size[1])
    base_context_w = w + (float(area_factor) - 1.0) * ((w + h) * 0.5)
    base_context_h = h + (float(area_factor) - 1.0) * ((w + h) * 0.5)
    base_scale = np.sqrt((output_w * output_h) / max(1.0, base_context_w * base_context_h))
    center_x, center_y = bbox_center(bbox)
    target_center = np.asarray((output_w / 2.0, output_h / 2.0), dtype=np.float64)
    source_center = np.asarray((center_x, center_y), dtype=np.float64)
    wh = np.asarray((w, h), dtype=np.float64)

    best_crop = siamfc_context_bbox(bbox, area_factor)
    for attempt in range(max(1, retry_count + 1)):
        scale = np.asarray((base_scale, base_scale), dtype=np.float64)
        if scale_jitter > 0:
            scale = scale / np.exp(rng.standard_normal(2) * float(scale_jitter))
        translation = target_center - (source_center * scale)
        if translation_jitter > 0:
            max_translate = float((wh * scale).sum() * 0.5 * float(translation_jitter))
            translation = translation + (rng.uniform(low=-1.0, high=1.0, size=2) * max_translate)

        crop_left_top = -translation / scale
        crop_size = np.asarray((output_w, output_h), dtype=np.float64) / scale
        crop = (
            float(crop_left_top[0]),
            float(crop_left_top[1]),
            max(1.0, float(crop_size[0])),
            max(1.0, float(crop_size[1])),
        )
        best_crop = crop

        output_bbox_w = w * scale[0]
        output_bbox_h = h * scale[1]
        if output_bbox_w >= min_object_size and output_bbox_h >= min_object_size:
            break
        if attempt >= retry_count:
            break
    return best_crop


def jitter_reference_bbox(bbox: BBox, rng: np.random.Generator, amount: float) -> BBox:
    amount = max(0.0, float(amount))
    if amount <= 0:
        return bbox
    x, y, w, h = bbox
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    center_x, center_y = bbox_center((x, y, w, h))
    center_x += float(rng.uniform(-amount, amount) * w)
    center_y += float(rng.uniform(-amount, amount) * h)
    scale_w = float(np.exp(rng.normal(0.0, amount * 0.35)))
    scale_h = float(np.exp(rng.normal(0.0, amount * 0.35)))
    new_w = max(1.0, w * scale_w)
    new_h = max(1.0, h * scale_h)
    return center_x - (new_w / 2.0), center_y - (new_h / 2.0), new_w, new_h


def stable_region_order(specs: Sequence[V8SelectedRegionSpec], track_id: int, frame_number: int) -> List[V8SelectedRegionSpec]:
    if not specs:
        return [FULL_REGION_SPEC]
    specs = list(specs)
    if len(specs) == 1:
        return specs
    offset = abs((int(track_id) * 1315423911) ^ int(frame_number)) % len(specs)
    rotated = specs[offset:] + specs[:offset]
    priority_names = ("full", "head_like_top", "face_like_top", "tiny_head", "small_center")
    prioritized: List[V8SelectedRegionSpec] = []
    for name in priority_names:
        for spec in specs:
            if spec.name == name and spec not in prioritized:
                prioritized.append(spec)
                break
    prioritized_names = {spec.name for spec in prioritized}
    return [*prioritized, *[spec for spec in rotated if spec.name not in prioritized_names]]


def make_lorat_augmentation_spec(rng: np.random.Generator, enabled: bool) -> LoRATTrainingAugmentationSpec:
    if not enabled:
        return LoRATTrainingAugmentationSpec(enabled=False)

    def jitter_factor(amount: float) -> float:
        return float(rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount))

    joint_op = str(rng.choice(("grayscale", "solarize", "blur")))
    blur_kernel = int(rng.choice((3, 5)))
    return LoRATTrainingAugmentationSpec(
        enabled=True,
        template_flip=bool(rng.random() < 0.5),
        search_flip=bool(rng.random() < 0.5),
        template_brightness=jitter_factor(0.4),
        template_contrast=jitter_factor(0.4),
        template_saturation=jitter_factor(0.4),
        search_brightness=jitter_factor(0.4),
        search_contrast=jitter_factor(0.4),
        search_saturation=jitter_factor(0.4),
        joint_deit_op=joint_op,
        joint_blur_kernel=blur_kernel,
    )


def flip_bbox_horizontal(bbox: BBox, frame_width: int) -> BBox:
    x, y, w, h = bbox
    return float(frame_width - (x + w)), float(y), float(w), float(h)


def apply_color_jitter_bgr(frame: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    image = frame.astype(np.float32)
    image *= float(brightness)
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * float(contrast) + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    if abs(float(saturation) - 1.0) > 1e-3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image


def apply_joint_deit_aug_bgr(frame: np.ndarray, spec: LoRATTrainingAugmentationSpec) -> np.ndarray:
    if not spec.enabled:
        return frame
    if spec.joint_deit_op == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if spec.joint_deit_op == "solarize":
        return np.where(frame < 128, frame, 255 - frame).astype(np.uint8)
    if spec.joint_deit_op == "blur":
        kernel = max(3, int(spec.joint_blur_kernel) | 1)
        return cv2.GaussianBlur(frame, (kernel, kernel), 0)
    return frame


def apply_lorat_image_augmentation(frame: np.ndarray, spec: LoRATTrainingAugmentationSpec, role: str) -> np.ndarray:
    if not spec.enabled:
        return frame
    image = np.ascontiguousarray(frame)
    if role == "template":
        if spec.template_flip:
            image = cv2.flip(image, 1)
        image = apply_color_jitter_bgr(image, spec.template_brightness, spec.template_contrast, spec.template_saturation)
    else:
        if spec.search_flip:
            image = cv2.flip(image, 1)
        image = apply_color_jitter_bgr(image, spec.search_brightness, spec.search_contrast, spec.search_saturation)
    return apply_joint_deit_aug_bgr(image, spec)


def apply_search_bbox_augmentation(objects: Sequence[V8TrainingObject], frame_width: int, spec: LoRATTrainingAugmentationSpec) -> List[V8TrainingObject]:
    if not spec.enabled or not spec.search_flip:
        return list(objects)
    return [
        replace(
            item,
            current_bbox=flip_bbox_horizontal(item.current_bbox, frame_width),
            previous_bbox=flip_bbox_horizontal(item.previous_bbox, frame_width),
            search_bbox=flip_bbox_horizontal(item.search_bbox, frame_width),
            distractor_bboxes=tuple(flip_bbox_horizontal(bbox, frame_width) for bbox in item.distractor_bboxes),
        )
        for item in objects
    ]


def attach_distractor_bboxes(objects: Sequence[V8TrainingObject]) -> List[V8TrainingObject]:
    present = [item for item in objects if item.is_present]
    updated: List[V8TrainingObject] = []
    for item in objects:
        distractors = tuple(
            other.current_bbox
            for other in present
            if other.track_id != item.track_id
        )
        updated.append(replace(item, distractor_bboxes=distractors))
    return updated


def translate_bbox(bbox: BBox, origin_x: float, origin_y: float) -> BBox:
    x, y, w, h = bbox
    return float(x - origin_x), float(y - origin_y), float(w), float(h)


def crop_frame_with_padding(frame: np.ndarray, crop_bbox: BBox) -> Tuple[np.ndarray, float, float]:
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = crop_bbox
    left = int(np.floor(float(x)))
    top = int(np.floor(float(y)))
    right = int(np.ceil(float(x + max(1.0, w))))
    bottom = int(np.ceil(float(y + max(1.0, h))))
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    crop = np.zeros((crop_height, crop_width, frame.shape[2]), dtype=frame.dtype)

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(frame_width, right)
    src_bottom = min(frame_height, bottom)
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - left
        dst_top = src_top - top
        crop[dst_top:dst_top + (src_bottom - src_top), dst_left:dst_left + (src_right - src_left)] = frame[src_top:src_bottom, src_left:src_right]
    return np.ascontiguousarray(crop), float(left), float(top)


def crop_training_batches(frame: np.ndarray, objects: Sequence[V8TrainingObject], mode: str) -> List[Tuple[np.ndarray, List[V8TrainingObject], str]]:
    mode = str(mode or "full").strip().lower()
    if mode == "full":
        return [(frame, list(objects), "full")]
    if mode != "search_crop":
        raise ValueError(f"Unknown training frame mode: {mode!r}")

    batches: List[Tuple[np.ndarray, List[V8TrainingObject], str]] = []
    for item in objects:
        crop, origin_x, origin_y = crop_frame_with_padding(frame, expanded_training_search_bbox(item, 1.0))
        crop_h, crop_w = crop.shape[:2]
        transformed = replace(
            item,
            current_bbox=translate_bbox(item.current_bbox, origin_x, origin_y),
            previous_bbox=translate_bbox(item.previous_bbox, origin_x, origin_y),
            search_bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
            distractor_bboxes=tuple(translate_bbox(bbox, origin_x, origin_y) for bbox in item.distractor_bboxes),
        )
        batches.append((crop, [transformed], "search_crop"))
    return batches


def effective_training_frame_mode(mode: str, epoch: int, crop_stage_epochs: int) -> str:
    mode = str(mode or "full").strip().lower()
    if mode == "staged":
        return "search_crop" if int(epoch) <= max(0, int(crop_stage_epochs)) else "full"
    if mode in {"full", "search_crop"}:
        return mode
    raise ValueError(f"Unknown training frame mode: {mode!r}")


def update_recent_track_history(
    recent_rows_by_track: Dict[int, List[exercise.GroundTruthRow]],
    rows: Sequence[exercise.GroundTruthRow],
    max_length: int,
) -> None:
    max_length = max(1, int(max_length))
    for row in rows:
        history = recent_rows_by_track.setdefault(row.track_id, [])
        history.append(row)
        if len(history) > max_length:
            del history[:-max_length]


def template_bboxes_for_augmentation(item: V8TrainingObject, template_width: int, spec: LoRATTrainingAugmentationSpec) -> Tuple[BBox, BBox]:
    if spec.enabled and spec.template_flip:
        return (
            flip_bbox_horizontal(item.template_bbox, template_width),
            flip_bbox_horizontal(item.template_context_bbox, template_width),
        )
    return item.template_bbox, item.template_context_bbox


class MOTFrameHeadDataset:
    """Frame-level multi-object samples for the V8 shared-frame LoRA head."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        class_ids: Optional[Sequence[int]],
        min_visibility: float,
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
        lost_target_probability: float,
        max_lost_targets_per_frame: int,
        max_missing_gap_frames: int,
        search_anchor_mode: str = "union",
        repair_search_to_target: bool = True,
        search_target_padding_fraction: float = 0.05,
        sequence_name_filter: Optional[str] = None,
        track_id_filter: Optional[int] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.class_ids = set(class_ids) if class_ids else None
        self.min_visibility = max(0.0, float(min_visibility))
        self.max_objects = max(1, int(max_objects))
        self.frame_stride = max(1, int(frame_stride))
        self.region_specs = region_specs_for_mode(target_region_mode)
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
        self.lost_target_probability = max(0.0, min(1.0, float(lost_target_probability)))
        self.max_lost_targets_per_frame = max(0, int(max_lost_targets_per_frame))
        self.max_missing_gap_frames = max(1, int(max_missing_gap_frames))
        self.search_anchor_mode = str(search_anchor_mode or "union").strip().lower()
        if self.search_anchor_mode not in {"current", "previous", "union"}:
            raise ValueError(f"Unknown search_anchor_mode: {search_anchor_mode!r}")
        self.repair_search_to_target = bool(repair_search_to_target)
        self.search_target_padding_fraction = max(0.0, float(search_target_padding_fraction))
        self.sequence_name_filter = str(sequence_name_filter).strip() if sequence_name_filter else None
        self.track_id_filter = int(track_id_filter) if track_id_filter is not None else None
        self.samples: List[V8TrainingSample] = []
        self._build(max_sequences=max_sequences, max_samples=max_samples)

    def _usable_rows(self, rows: Sequence[exercise.GroundTruthRow]) -> List[exercise.GroundTruthRow]:
        selected = [
            row
            for row in rows
            if row.confidence != 0
            and row.visibility >= self.min_visibility
            and (self.track_id_filter is None or row.track_id == self.track_id_filter)
            and (self.class_ids is None or row.class_id in self.class_ids)
            and row.bbox[2] > 1
            and row.bbox[3] > 1
        ]
        return sorted(selected, key=lambda row: row.bbox[2] * row.bbox[3], reverse=True)

    def _build(self, max_sequences: int, max_samples: int) -> None:
        sequences = exercise.find_sequences(self.dataset_root, self.split)
        if max_sequences > 0:
            sequences = sequences[:max_sequences]
        for sequence_path in sequences:
            if self.sequence_name_filter and sequence_path.name != self.sequence_name_filter:
                continue
            image_paths = exercise.get_image_paths(sequence_path)
            gt_by_frame = exercise.read_gt(sequence_path)
            if not image_paths or not gt_by_frame:
                continue

            first_row_by_track: Dict[int, exercise.GroundTruthRow] = {}
            for frame in sorted(gt_by_frame):
                for row in self._usable_rows(gt_by_frame[frame]):
                    first_row_by_track.setdefault(row.track_id, row)

            recent_rows_by_track: Dict[int, List[exercise.GroundTruthRow]] = {}
            for frame_number in sorted(gt_by_frame):
                usable_rows = self._usable_rows(gt_by_frame[frame_number])
                if frame_number % self.frame_stride != 0:
                    update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)
                    continue
                image_index = exercise.frame_to_image_index(frame_number)
                if image_index >= len(image_paths):
                    update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)
                    continue

                objects: List[V8TrainingObject] = []
                for row in usable_rows:
                    first_template_row = first_row_by_track.get(row.track_id, row)
                    recent_history = recent_rows_by_track.get(row.track_id, [])
                    previous_row = recent_history[-1] if recent_history else first_template_row
                    for spec in stable_region_order(self.region_specs, row.track_id, frame_number)[: self.target_regions_per_object]:
                        rng = deterministic_rng(sequence_path, frame_number, row.track_id, spec.name)
                        template_row = first_template_row
                        if self.template_sampling == "previous":
                            template_row = previous_row
                        elif self.template_sampling == "window" and recent_history:
                            template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                        elif self.template_sampling == "mixed":
                            draw = float(rng.random())
                            if draw < 0.50:
                                template_row = first_template_row
                            elif draw < 0.75:
                                template_row = previous_row
                            elif recent_history:
                                template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                        template_index = exercise.frame_to_image_index(template_row.frame)
                        if template_index >= len(image_paths):
                            template_row = row
                        current_bbox = selected_region_bbox(row.bbox, spec)
                        previous_bbox = selected_region_bbox(previous_row.bbox, spec)
                        template_bbox = selected_region_bbox(template_row.bbox, spec)
                        noisy_previous_bbox = jitter_reference_bbox(previous_bbox, rng, self.previous_box_jitter)
                        if self.search_anchor_mode == "previous":
                            search_anchor_bbox = noisy_previous_bbox
                        elif self.search_anchor_mode == "union":
                            search_anchor_bbox = union_bbox_xywh(noisy_previous_bbox, current_bbox)
                        else:
                            search_anchor_bbox = current_bbox
                        search_bbox = siamfc_search_bbox(
                            search_anchor_bbox,
                            (224, 224),
                            self.search_area_factor,
                            self.search_scale_jitter,
                            self.search_translation_jitter,
                            self.search_min_object_size,
                            rng,
                        )
                        if self.repair_search_to_target:
                            search_bbox = repair_search_bbox_to_cover_target(
                                search_bbox,
                                current_bbox,
                                self.search_target_padding_fraction,
                            )
                        objects.append(
                            V8TrainingObject(
                                track_id=row.track_id,
                                current_bbox=current_bbox,
                                previous_bbox=noisy_previous_bbox,
                                previous_frame=previous_row.frame,
                                previous_context_bbox=siamfc_context_bbox(noisy_previous_bbox, self.template_area_factor),
                                template_frame=template_row.frame,
                                template_bbox=template_bbox,
                                template_context_bbox=siamfc_context_bbox(template_bbox, self.template_area_factor),
                                search_bbox=search_bbox,
                                target_kind=spec.name,
                                is_present=True,
                            )
                        )
                        if len(objects) >= self.max_objects:
                            break
                    if len(objects) >= self.max_objects:
                        break

                if len(objects) < self.max_objects and self.lost_target_probability > 0.0 and self.max_lost_targets_per_frame > 0:
                    visible_track_ids = {row.track_id for row in usable_rows}
                    missing_candidates: List[exercise.GroundTruthRow] = []
                    for track_id, history in recent_rows_by_track.items():
                        if track_id in visible_track_ids or not history:
                            continue
                        last_row = history[-1]
                        if int(frame_number) - int(last_row.frame) <= self.max_missing_gap_frames:
                            missing_candidates.append(last_row)
                    missing_candidates.sort(key=lambda row: (frame_number - row.frame, row.track_id))
                    added_missing = 0
                    for previous_row in missing_candidates:
                        if len(objects) >= self.max_objects or added_missing >= self.max_lost_targets_per_frame:
                            break
                        first_template_row = first_row_by_track.get(previous_row.track_id, previous_row)
                        for spec in stable_region_order(self.region_specs, previous_row.track_id, frame_number)[:1]:
                            rng = deterministic_rng(sequence_path, frame_number, previous_row.track_id, f"missing-{spec.name}")
                            if float(rng.random()) > self.lost_target_probability:
                                continue
                            recent_history = recent_rows_by_track.get(previous_row.track_id, [])
                            template_row = previous_row
                            if self.template_sampling == "first":
                                template_row = first_template_row
                            elif self.template_sampling == "mixed":
                                draw = float(rng.random())
                                if draw < 0.50:
                                    template_row = first_template_row
                                elif recent_history:
                                    template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                            elif self.template_sampling == "window" and recent_history:
                                template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                            previous_bbox = selected_region_bbox(previous_row.bbox, spec)
                            template_bbox = selected_region_bbox(template_row.bbox, spec)
                            noisy_previous_bbox = jitter_reference_bbox(previous_bbox, rng, self.previous_box_jitter)
                            objects.append(
                                V8TrainingObject(
                                    track_id=previous_row.track_id,
                                    current_bbox=previous_bbox,
                                    previous_bbox=noisy_previous_bbox,
                                    previous_frame=previous_row.frame,
                                    previous_context_bbox=siamfc_context_bbox(noisy_previous_bbox, self.template_area_factor),
                                    template_frame=template_row.frame,
                                    template_bbox=template_bbox,
                                    template_context_bbox=siamfc_context_bbox(template_bbox, self.template_area_factor),
                                    search_bbox=siamfc_search_bbox(
                                        noisy_previous_bbox,
                                        (224, 224),
                                        self.search_area_factor,
                                        self.search_scale_jitter,
                                        self.search_translation_jitter,
                                        self.search_min_object_size,
                                        rng,
                                    ),
                                    target_kind=spec.name,
                                    is_present=False,
                                )
                            )
                            added_missing += 1
                            break

                if objects:
                    objects = attach_distractor_bboxes(objects)
                    self.samples.append(
                        V8TrainingSample(
                            sequence_path=sequence_path,
                            image_paths=image_paths,
                            frame_number=frame_number,
                            objects=objects,
                        )
                    )
                if max_samples > 0 and len(self.samples) >= max_samples:
                    return
                update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> V8TrainingSample:
        return self.samples[index]


class CombinedFrameHeadDataset:
    def __init__(self, datasets: Sequence[object]) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]  # type: ignore[arg-type]
        self.samples: List[V8TrainingSample] = []
        self.source_counts: Dict[str, int] = {}
        for dataset in self.datasets:
            label = dataset.__class__.__name__
            count = len(dataset)  # type: ignore[arg-type]
            self.source_counts[label] = self.source_counts.get(label, 0) + count
            self.samples.extend(dataset[index] for index in range(count))  # type: ignore[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> V8TrainingSample:
        return self.samples[index]


class TAOFrameHeadDataset(MOTFrameHeadDataset):
    """TAO/TAO-OW frame-level samples using TAO JSON annotations and extracted frames."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        class_ids: Optional[Sequence[int]],
        min_visibility: float,
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
        lost_target_probability: float,
        max_lost_targets_per_frame: int,
        max_missing_gap_frames: int,
        search_anchor_mode: str = "union",
        repair_search_to_target: bool = True,
        search_target_padding_fraction: float = 0.05,
        sequence_name_filter: Optional[str] = None,
        track_id_filter: Optional[int] = None,
        use_freeform_annotations: bool = False,
    ) -> None:
        self.use_freeform_annotations = bool(use_freeform_annotations)
        super().__init__(
            dataset_root,
            split,
            class_ids,
            min_visibility,
            max_objects,
            frame_stride,
            max_sequences,
            max_samples,
            target_region_mode,
            target_regions_per_object,
            template_area_factor,
            search_area_factor,
            search_scale_jitter,
            search_translation_jitter,
            search_min_object_size,
            template_sampling,
            previous_box_jitter,
            sequence_window_length,
            lost_target_probability,
            max_lost_targets_per_frame,
            max_missing_gap_frames,
            search_anchor_mode,
            repair_search_to_target,
            search_target_padding_fraction,
            sequence_name_filter,
            track_id_filter,
        )

    def _annotation_path(self) -> Path:
        split_name = str(self.split or "train").strip().lower()
        if split_name == "val":
            split_name = "validation"
        suffix = "_with_freeform" if self.use_freeform_annotations and split_name in {"train", "validation"} else ""
        candidates = [
            self.dataset_root / "annotations" / f"{split_name}{suffix}.json",
            self.dataset_root / "annotations_public" / f"{split_name}{suffix}.json",
            self.dataset_root / f"{split_name}{suffix}.json",
        ]
        if suffix:
            candidates.extend(
                [
                    self.dataset_root / "annotations" / f"{split_name}.json",
                    self.dataset_root / "annotations_public" / f"{split_name}.json",
                    self.dataset_root / f"{split_name}.json",
                ]
            )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find TAO annotation JSON for split={self.split!r} under {self.dataset_root}")

    def _build(self, max_sequences: int, max_samples: int) -> None:
        annotation_path = self._annotation_path()
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        videos_by_id = {int(video["id"]): video for video in data.get("videos", []) if "id" in video}
        annotations_by_image: Dict[int, List[dict]] = {}
        for annotation in data.get("annotations", []):
            if "image_id" not in annotation or "bbox" not in annotation:
                continue
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        image_records_by_video: Dict[str, List[Tuple[int, Path, int]]] = {}
        for image in data.get("images", []):
            file_name = str(image.get("file_name", ""))
            if not file_name:
                continue
            path = self.dataset_root / "frames" / file_name
            if not path.exists():
                continue
            video_id = int(image.get("video_id", -1))
            video = videos_by_id.get(video_id, {})
            video_name = str(image.get("video") or video.get("name") or Path(file_name).parent.as_posix())
            frame_index = int(image.get("frame_index", image.get("frame_id", len(image_records_by_video.get(video_name, [])))))
            image_records_by_video.setdefault(video_name, []).append((frame_index, path, int(image["id"])))

        sequence_items = sorted(image_records_by_video.items(), key=lambda item: item[0].lower())
        if max_sequences > 0:
            sequence_items = sequence_items[:max_sequences]

        for video_name, records in sequence_items:
            sequence_path = self.dataset_root / "frames" / video_name
            if self.sequence_name_filter and self.sequence_name_filter not in video_name and sequence_path.name != self.sequence_name_filter:
                continue
            records = sorted(records, key=lambda item: (item[0], str(item[1])))
            image_paths = [path for _, path, _ in records]
            image_id_to_frame = {image_id: index + 1 for index, (_, _, image_id) in enumerate(records)}
            gt_by_frame: Dict[int, List[exercise.GroundTruthRow]] = {}
            for _, _, image_id in records:
                frame_number = image_id_to_frame[image_id]
                rows: List[exercise.GroundTruthRow] = []
                for annotation in annotations_by_image.get(image_id, []):
                    bbox_values = annotation.get("bbox", [])
                    if len(bbox_values) != 4:
                        continue
                    bbox = tuple(float(value) for value in bbox_values)  # type: ignore[assignment]
                    rows.append(
                        exercise.GroundTruthRow(
                            frame=frame_number,
                            track_id=int(annotation.get("track_id", annotation.get("id", len(rows) + 1))),
                            bbox=bbox,
                            confidence=1.0,
                            class_id=int(annotation.get("category_id", 0)),
                            visibility=float(annotation.get("visibility", 1.0)),
                        )
                    )
                if rows:
                    gt_by_frame[frame_number] = rows
            if not image_paths or not gt_by_frame:
                continue

            first_row_by_track: Dict[int, exercise.GroundTruthRow] = {}
            for frame in sorted(gt_by_frame):
                for row in self._usable_rows(gt_by_frame[frame]):
                    first_row_by_track.setdefault(row.track_id, row)

            recent_rows_by_track: Dict[int, List[exercise.GroundTruthRow]] = {}
            for frame_number in sorted(gt_by_frame):
                usable_rows = self._usable_rows(gt_by_frame[frame_number])
                if frame_number % self.frame_stride != 0:
                    update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)
                    continue
                image_index = exercise.frame_to_image_index(frame_number)
                if image_index >= len(image_paths):
                    update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)
                    continue

                objects: List[V8TrainingObject] = []
                for row in usable_rows:
                    first_template_row = first_row_by_track.get(row.track_id, row)
                    recent_history = recent_rows_by_track.get(row.track_id, [])
                    previous_row = recent_history[-1] if recent_history else first_template_row
                    for spec in stable_region_order(self.region_specs, row.track_id, frame_number)[: self.target_regions_per_object]:
                        rng = deterministic_rng(sequence_path, frame_number, row.track_id, spec.name)
                        template_row = first_template_row
                        if self.template_sampling == "previous":
                            template_row = previous_row
                        elif self.template_sampling == "window" and recent_history:
                            template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                        elif self.template_sampling == "mixed":
                            draw = float(rng.random())
                            if draw < 0.50:
                                template_row = first_template_row
                            elif draw < 0.75:
                                template_row = previous_row
                            elif recent_history:
                                template_row = recent_history[int(rng.integers(0, len(recent_history)))]
                        template_index = exercise.frame_to_image_index(template_row.frame)
                        if template_index >= len(image_paths):
                            template_row = row
                        current_bbox = selected_region_bbox(row.bbox, spec)
                        previous_bbox = selected_region_bbox(previous_row.bbox, spec)
                        template_bbox = selected_region_bbox(template_row.bbox, spec)
                        noisy_previous_bbox = jitter_reference_bbox(previous_bbox, rng, self.previous_box_jitter)
                        if self.search_anchor_mode == "previous":
                            search_anchor_bbox = noisy_previous_bbox
                        elif self.search_anchor_mode == "union":
                            search_anchor_bbox = union_bbox_xywh(noisy_previous_bbox, current_bbox)
                        else:
                            search_anchor_bbox = current_bbox
                        search_bbox = siamfc_search_bbox(
                            search_anchor_bbox,
                            (224, 224),
                            self.search_area_factor,
                            self.search_scale_jitter,
                            self.search_translation_jitter,
                            self.search_min_object_size,
                            rng,
                        )
                        if self.repair_search_to_target:
                            search_bbox = repair_search_bbox_to_cover_target(
                                search_bbox,
                                current_bbox,
                                self.search_target_padding_fraction,
                            )
                        objects.append(
                            V8TrainingObject(
                                track_id=row.track_id,
                                current_bbox=current_bbox,
                                previous_bbox=noisy_previous_bbox,
                                previous_frame=previous_row.frame,
                                previous_context_bbox=siamfc_context_bbox(noisy_previous_bbox, self.template_area_factor),
                                template_frame=template_row.frame,
                                template_bbox=template_bbox,
                                template_context_bbox=siamfc_context_bbox(template_bbox, self.template_area_factor),
                                search_bbox=search_bbox,
                                target_kind=f"tao_{spec.name}",
                                is_present=True,
                            )
                        )
                        if len(objects) >= self.max_objects:
                            break
                    if len(objects) >= self.max_objects:
                        break

                if objects:
                    objects = attach_distractor_bboxes(objects)
                    self.samples.append(
                        V8TrainingSample(
                            sequence_path=sequence_path,
                            image_paths=image_paths,
                            frame_number=frame_number,
                            objects=objects,
                        )
                    )
                if max_samples > 0 and len(self.samples) >= max_samples:
                    return
                update_recent_track_history(recent_rows_by_track, usable_rows, self.sequence_window_length)


def parse_class_ids(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    if not values:
        return None
    return list(dict.fromkeys(int(value) for value in values))


_FRAME_READ_WARNING_LIMIT = 100
_frame_read_warning_count = 0


def load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None and path.exists():
        try:
            raw_bytes = np.fromfile(str(path), dtype=np.uint8)
            if raw_bytes.size:
                frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        except OSError:
            frame = None
    if frame is None:
        raise RuntimeError(f"Unable to read frame: {path}")
    return frame


def try_load_frame(path: Path, context: str) -> Optional[np.ndarray]:
    global _frame_read_warning_count
    try:
        return load_frame(path)
    except RuntimeError as error:
        _frame_read_warning_count += 1
        if _frame_read_warning_count <= _FRAME_READ_WARNING_LIMIT:
            print(f"WARNING: skipping unreadable frame during {context}: {error}", file=sys.stderr, flush=True)
        elif _frame_read_warning_count == _FRAME_READ_WARNING_LIMIT + 1:
            print(
                "WARNING: additional unreadable-frame messages suppressed; training will continue skipping bad samples.",
                file=sys.stderr,
                flush=True,
            )
        return None


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def gaussian_heatmap(torch_module, grid_height: int, grid_width: int, center_y: int, center_x: int, sigma: float, device):
    y = torch_module.arange(grid_height, device=device, dtype=torch_module.float32)[:, None]
    x = torch_module.arange(grid_width, device=device, dtype=torch_module.float32)[None, :]
    dist2 = ((x - float(center_x)) ** 2) + ((y - float(center_y)) ** 2)
    return torch_module.exp(-dist2 / (2.0 * max(0.01, float(sigma)) ** 2))


def xywh_to_xyxy_tuple(bbox: BBox) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_area_xywh(bbox: BBox) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def bbox_center_distance_xywh(left: BBox, right: BBox) -> float:
    left_x, left_y = bbox_center(left)
    right_x, right_y = bbox_center(right)
    return float(np.hypot(float(left_x) - float(right_x), float(left_y) - float(right_y)))


def bbox_motion_ratio_xywh(previous: BBox, current: BBox) -> float:
    _, _, width, height = current
    target_diagonal = max(1.0, float(np.hypot(max(1.0, float(width)), max(1.0, float(height)))))
    return bbox_center_distance_xywh(previous, current) / target_diagonal


def bbox_intersection_area_xywh(left: BBox, right: BBox) -> float:
    lx1, ly1, lx2, ly2 = xywh_to_xyxy_tuple(left)
    rx1, ry1, rx2, ry2 = xywh_to_xyxy_tuple(right)
    ix1 = max(float(lx1), float(rx1))
    iy1 = max(float(ly1), float(ry1))
    ix2 = min(float(lx2), float(rx2))
    iy2 = min(float(ly2), float(ry2))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def bbox_coverage_xywh(inner: BBox, outer: BBox) -> float:
    return bbox_intersection_area_xywh(inner, outer) / max(1.0, bbox_area_xywh(inner))


def bbox_contains_center_xywh(outer: BBox, inner: BBox) -> bool:
    x, y, w, h = outer
    cx, cy = bbox_center(inner)
    return float(x) <= cx <= float(x + w) and float(y) <= cy <= float(y + h)


def bbox_center_inside_frame(bbox: BBox, frame_shape: Tuple[int, ...]) -> bool:
    frame_height, frame_width = frame_shape[:2]
    center_x, center_y = bbox_center(bbox)
    return 0.0 <= float(center_x) < float(frame_width) and 0.0 <= float(center_y) < float(frame_height)


def clamp_bbox_to_frame_shape(bbox: BBox, frame_shape: Tuple[int, ...]) -> BBox:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = bbox
    width = min(max(1.0, float(width)), max(1.0, float(frame_width)))
    height = min(max(1.0, float(height)), max(1.0, float(frame_height)))
    x = max(0.0, min(float(x), max(0.0, float(frame_width) - width)))
    y = max(0.0, min(float(y), max(0.0, float(frame_height) - height)))
    return x, y, width, height


def union_bbox_xywh(left: BBox, right: BBox, padding_fraction: float = 0.0) -> BBox:
    lx1, ly1, lx2, ly2 = xywh_to_xyxy_tuple(left)
    rx1, ry1, rx2, ry2 = xywh_to_xyxy_tuple(right)
    x1 = min(float(lx1), float(rx1))
    y1 = min(float(ly1), float(ry1))
    x2 = max(float(lx2), float(rx2))
    y2 = max(float(ly2), float(ry2))
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad = max(0.0, float(padding_fraction))
    if pad > 0.0:
        pad_x = width * pad
        pad_y = height * pad
        x1 -= pad_x
        y1 -= pad_y
        width += pad_x * 2.0
        height += pad_y * 2.0
    return float(x1), float(y1), max(1.0, float(width)), max(1.0, float(height))


def repair_search_bbox_to_cover_target(search_bbox: BBox, target_bbox: BBox, padding_fraction: float) -> BBox:
    if bbox_coverage_xywh(target_bbox, search_bbox) >= 0.999 and bbox_contains_center_xywh(search_bbox, target_bbox):
        return search_bbox
    return union_bbox_xywh(search_bbox, target_bbox, padding_fraction)


def bbox_to_grid_slice(
    bbox: BBox,
    frame_shape: Tuple[int, ...],
    grid_height: int,
    grid_width: int,
) -> Tuple[slice, slice]:
    frame_height, frame_width = frame_shape[:2]
    x, y, w, h = bbox
    left = int(np.floor((x / max(1.0, float(frame_width))) * grid_width))
    right = int(np.ceil(((x + max(1.0, w)) / max(1.0, float(frame_width))) * grid_width))
    top = int(np.floor((y / max(1.0, float(frame_height))) * grid_height))
    bottom = int(np.ceil(((y + max(1.0, h)) / max(1.0, float(frame_height))) * grid_height))
    left = max(0, min(grid_width - 1, left))
    top = max(0, min(grid_height - 1, top))
    right = max(left + 1, min(grid_width, right))
    bottom = max(top + 1, min(grid_height, bottom))
    return slice(top, bottom), slice(left, right)


def expanded_training_search_bbox(item: V8TrainingObject, search_radius_factor: float) -> BBox:
    if item.search_bbox is not None:
        return item.search_bbox
    center_x, center_y = bbox_center(item.previous_bbox)
    _, _, previous_w, previous_h = item.previous_bbox
    _, _, template_w, template_h = item.template_bbox
    search_w = max(1.0, previous_w, template_w) * max(0.25, float(search_radius_factor))
    search_h = max(1.0, previous_h, template_h) * max(0.25, float(search_radius_factor))
    return center_x - (search_w / 2.0), center_y - (search_h / 2.0), search_w, search_h


def bbox_area_xyxy(torch_module, boxes):
    width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
    return width * height


def bbox_iou_aligned(torch_module, predicted, target, eps: float = 1e-7):
    left_top = torch_module.maximum(predicted[..., :2], target[..., :2])
    right_bottom = torch_module.minimum(predicted[..., 2:], target[..., 2:])
    intersection_wh = (right_bottom - left_top).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = bbox_area_xyxy(torch_module, predicted) + bbox_area_xyxy(torch_module, target) - intersection
    return intersection / union.clamp_min(eps)


def generalized_iou_aligned(torch_module, predicted, target, eps: float = 1e-7):
    iou = bbox_iou_aligned(torch_module, predicted, target, eps)
    enclosing_left_top = torch_module.minimum(predicted[..., :2], target[..., :2])
    enclosing_right_bottom = torch_module.maximum(predicted[..., 2:], target[..., 2:])
    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp_min(0.0)
    enclosing_area = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp_min(eps)

    left_top = torch_module.maximum(predicted[..., :2], target[..., :2])
    right_bottom = torch_module.minimum(predicted[..., 2:], target[..., 2:])
    intersection_wh = (right_bottom - left_top).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = bbox_area_xyxy(torch_module, predicted) + bbox_area_xyxy(torch_module, target) - intersection
    return iou - ((enclosing_area - union) / enclosing_area)


def decode_box_maps_xyxy(
    torch_module,
    box_delta_maps,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    device,
):
    frame_height, frame_width = frame_shape[:2]
    object_count, grid_height, grid_width = box_delta_maps.shape[:3]
    y = torch_module.arange(grid_height, device=device, dtype=torch_module.float32)[None, :, None]
    x = torch_module.arange(grid_width, device=device, dtype=torch_module.float32)[None, None, :]
    ref_x = (x + 0.5) / float(grid_width)
    ref_y = (y + 0.5) / float(grid_height)
    offsets = torch_module.sigmoid(box_delta_maps.to(torch_module.float32))
    left = (ref_x - offsets[..., 0]) * float(frame_width)
    top = (ref_y - offsets[..., 1]) * float(frame_height)
    right = (ref_x + offsets[..., 2]) * float(frame_width)
    bottom = (ref_y + offsets[..., 3]) * float(frame_height)
    return torch_module.stack(
        (
            left,
            top,
            right,
            bottom,
        ),
        dim=-1,
    )


def make_lorat_style_targets(
    torch_module,
    score_maps,
    decoded_boxes_xyxy,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    search_radius_factor: float,
    iou_aware_classification: bool,
    center_positive_weight: float,
    small_target_loss_weight: float,
    small_target_area_threshold: float,
    small_target_max_side: float,
    device,
) -> Tuple[object, object, object, object, object, object, object, V8TrainingTargetStats]:
    object_count, grid_height, grid_width = score_maps.shape
    target_scores = torch_module.zeros_like(score_maps, dtype=torch_module.float32)
    positive_mask = torch_module.zeros((object_count, grid_height, grid_width), device=device, dtype=torch_module.bool)
    hard_negative_mask = torch_module.zeros_like(positive_mask)
    loss_mask = torch_module.zeros_like(positive_mask)
    positive_weights = torch_module.zeros_like(score_maps, dtype=torch_module.float32)
    target_boxes_xyxy = torch_module.zeros_like(decoded_boxes_xyxy, dtype=torch_module.float32)
    target_ltrb_offsets = torch_module.zeros_like(decoded_boxes_xyxy, dtype=torch_module.float32)
    ref_y = (torch_module.arange(grid_height, device=device, dtype=torch_module.float32) + 0.5) / float(grid_height)
    ref_x = (torch_module.arange(grid_width, device=device, dtype=torch_module.float32) + 0.5) / float(grid_width)
    ref_y = ref_y[None, :, None].expand(object_count, grid_height, grid_width)
    ref_x = ref_x[None, None, :].expand(object_count, grid_height, grid_width)
    frame_height, frame_width = frame_shape[:2]
    present_targets = 0
    target_center_outside_search = 0
    coverage_sum = 0.0
    min_coverage = 1.0
    area_ratio_sum = 0.0

    for index, item in enumerate(objects):
        if not item.is_present:
            search_y, search_x = bbox_to_grid_slice(
                expanded_training_search_bbox(item, search_radius_factor),
                frame_shape,
                grid_height,
                grid_width,
            )
            loss_mask[index, search_y, search_x] = True
            continue
        present_targets += 1
        pos_y, pos_x = bbox_to_grid_slice(item.current_bbox, frame_shape, grid_height, grid_width)
        positive_mask[index, pos_y, pos_x] = True
        item_search_bbox = expanded_training_search_bbox(item, search_radius_factor)
        coverage = bbox_coverage_xywh(item.current_bbox, item_search_bbox)
        coverage_sum += float(coverage)
        min_coverage = min(min_coverage, float(coverage))
        area_ratio_sum += bbox_area_xywh(item_search_bbox) / max(1.0, bbox_area_xywh(item.current_bbox))
        if not bbox_contains_center_xywh(item_search_bbox, item.current_bbox):
            target_center_outside_search += 1
        if pos_y.stop > pos_y.start and pos_x.stop > pos_x.start:
            ys = torch_module.arange(pos_y.start, pos_y.stop, device=device, dtype=torch_module.float32)[:, None]
            xs = torch_module.arange(pos_x.start, pos_x.stop, device=device, dtype=torch_module.float32)[None, :]
            center_x, center_y = bbox_center(item.current_bbox)
            center_grid_x = (center_x / max(1.0, float(frame_shape[1]))) * float(grid_width)
            center_grid_y = (center_y / max(1.0, float(frame_shape[0]))) * float(grid_height)
            sigma = max(0.75, 0.5 * max(float(pos_x.stop - pos_x.start), float(pos_y.stop - pos_y.start)))
            gaussian = torch_module.exp(-(((xs + 0.5 - center_grid_x) ** 2) + ((ys + 0.5 - center_grid_y) ** 2)) / (2.0 * sigma * sigma))
            gaussian = gaussian / gaussian.max().clamp_min(1e-6)
            strength = max(0.0, min(1.0, float(center_positive_weight)))
            positive_weights[index, pos_y, pos_x] = (1.0 - strength) + (strength * gaussian)
            if is_small_target_object(item, small_target_area_threshold, small_target_max_side):
                positive_weights[index, pos_y, pos_x] *= max(1.0, float(small_target_loss_weight))
        search_y, search_x = bbox_to_grid_slice(
            expanded_training_search_bbox(item, search_radius_factor),
            frame_shape,
            grid_height,
            grid_width,
        )
        loss_mask[index, search_y, search_x] = True
        target_box = torch_module.tensor(
            xywh_to_xyxy_tuple(item.current_bbox),
            device=device,
            dtype=torch_module.float32,
        )
        target_boxes_xyxy[index, :, :, :] = target_box
        x1, y1, x2, y2 = xywh_to_xyxy_tuple(item.current_bbox)
        target_ltrb_offsets[index, :, :, 0] = ref_x[index] - (float(x1) / max(1.0, float(frame_width)))
        target_ltrb_offsets[index, :, :, 1] = ref_y[index] - (float(y1) / max(1.0, float(frame_height)))
        target_ltrb_offsets[index, :, :, 2] = (float(x2) / max(1.0, float(frame_width))) - ref_x[index]
        target_ltrb_offsets[index, :, :, 3] = (float(y2) / max(1.0, float(frame_height))) - ref_y[index]
        target_ltrb_offsets[index].clamp_(0.0, 1.0)

    for index, item in enumerate(objects):
        search_y, search_x = bbox_to_grid_slice(
            expanded_training_search_bbox(item, search_radius_factor),
            frame_shape,
            grid_height,
            grid_width,
        )
        for other in objects:
            if other.track_id == item.track_id or not other.is_present:
                continue
            neg_y, neg_x = bbox_to_grid_slice(other.current_bbox, frame_shape, grid_height, grid_width)
            hard_negative_mask[index, neg_y, neg_x] = True
        hard_negative_mask[index] &= loss_mask[index]

    hard_negative_mask &= ~positive_mask

    positive_outside_search = int((positive_mask & ~loss_mask).sum().detach().item())
    loss_mask = loss_mask | positive_mask

    if positive_mask.any():
        if iou_aware_classification:
            with torch_module.no_grad():
                positive_iou = bbox_iou_aligned(
                    torch_module,
                    decoded_boxes_xyxy[positive_mask].detach(),
                    target_boxes_xyxy[positive_mask],
                ).clamp(0.0, 1.0)
            target_scores[positive_mask] = positive_iou
        else:
            target_scores[positive_mask] = 1.0

    stats = V8TrainingTargetStats(
        positive_cells=int(positive_mask.sum().detach().item()),
        loss_cells=int(loss_mask.sum().detach().item()),
        positive_cells_outside_search=positive_outside_search,
        hard_negative_cells=int(hard_negative_mask.sum().detach().item()),
        missing_targets=sum(1 for item in objects if not item.is_present),
        present_targets=present_targets,
        target_center_outside_search=target_center_outside_search,
        mean_target_search_coverage=(coverage_sum / float(present_targets)) if present_targets else 1.0,
        min_target_search_coverage=min_coverage if present_targets else 1.0,
        mean_search_target_area_ratio=(area_ratio_sum / float(present_targets)) if present_targets else 0.0,
    )
    return target_scores, positive_mask, hard_negative_mask, loss_mask, positive_weights, target_boxes_xyxy, target_ltrb_offsets, stats


def decode_predictions(trainer: v8.V8QualityBatchedLoRATTracker, head_output, objects: Sequence[V8TrainingObject], frame_shape: Tuple[int, ...]) -> List[BBox]:
    torch = trainer.torch
    score_maps = head_output.score_maps.detach().to(torch.float32)
    decoded_boxes = decode_box_maps_xyxy(
        torch,
        head_output.box_delta_maps.detach(),
        objects,
        frame_shape,
        trainer.device,
    )
    object_count, grid_height, grid_width = score_maps.shape
    search_mask = torch.zeros((object_count, grid_height, grid_width), device=trainer.device, dtype=torch.bool)
    for index, item in enumerate(objects):
        search_y, search_x = bbox_to_grid_slice(
            expanded_training_search_bbox(item, trainer.search_radius_factor),
            frame_shape,
            grid_height,
            grid_width,
        )
        search_mask[index, search_y, search_x] = True
    masked_scores = score_maps.masked_fill(~search_mask, -float("inf"))
    selection_scores = masked_scores
    if getattr(trainer, "v8_window_penalty_ratio", 0.0) > 0:
        selection_scores = torch.full_like(masked_scores, -float("inf"))
        for index, item in enumerate(objects):
            search_y, search_x = bbox_to_grid_slice(
                expanded_training_search_bbox(item, trainer.search_radius_factor),
                frame_shape,
                grid_height,
                grid_width,
            )
            roi_h = int(search_y.stop - search_y.start)
            roi_w = int(search_x.stop - search_x.start)
            if roi_h <= 0 or roi_w <= 0:
                continue
            if roi_h <= 1 or roi_w <= 1:
                window = torch.ones((roi_h, roi_w), device=trainer.device, dtype=torch.float32)
            else:
                window = torch.outer(
                    torch.hann_window(roi_h, periodic=False, device=trainer.device, dtype=torch.float32),
                    torch.hann_window(roi_w, periodic=False, device=trainer.device, dtype=torch.float32),
                )
            ratio = float(trainer.v8_window_penalty_ratio)
            selection_scores[index, search_y, search_x] = (
                torch.sigmoid(score_maps[index, search_y, search_x]) * (1.0 - ratio)
                + window * ratio
            )
    flat_indices = torch.argmax(selection_scores.reshape(object_count, -1), dim=1)
    flat_boxes = decoded_boxes.reshape(object_count, -1, 4)
    selected_boxes = flat_boxes[torch.arange(object_count, device=trainer.device), flat_indices].detach().cpu().tolist()

    predictions: List[BBox] = []
    for x1, y1, x2, y2 in selected_boxes:
        predictions.append((float(x1), float(y1), max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))))
    return predictions


def diagnostic_bbox_text(bbox: BBox) -> str:
    return ",".join(f"{float(value):.2f}" for value in bbox)


def prediction_diagnostic_rows(
    trainer: v8.V8QualityBatchedLoRATTracker,
    sample: V8TrainingSample,
    head_output,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    epoch: int,
    step: int,
) -> List[Dict[str, object]]:
    torch = trainer.torch
    score_maps = head_output.score_maps.detach().to(torch.float32)
    decoded_boxes = decode_box_maps_xyxy(
        torch,
        head_output.box_delta_maps.detach(),
        objects,
        frame_shape,
        trainer.device,
    )
    object_count, grid_height, grid_width = score_maps.shape
    rows: List[Dict[str, object]] = []
    for index, item in enumerate(objects):
        if not item.is_present:
            continue
        search_y, search_x = bbox_to_grid_slice(
            expanded_training_search_bbox(item, trainer.search_radius_factor),
            frame_shape,
            grid_height,
            grid_width,
        )
        pos_y, pos_x = bbox_to_grid_slice(item.current_bbox, frame_shape, grid_height, grid_width)
        search_scores = score_maps[index, search_y, search_x]
        if search_scores.numel() == 0:
            continue
        local_flat = int(torch.argmax(search_scores.reshape(-1)).detach().item())
        local_w = int(search_x.stop - search_x.start)
        best_y = int(search_y.start + (local_flat // max(1, local_w)))
        best_x = int(search_x.start + (local_flat % max(1, local_w)))
        best_box_xyxy = decoded_boxes[index, best_y, best_x].detach().cpu().tolist()
        prediction = (
            float(best_box_xyxy[0]),
            float(best_box_xyxy[1]),
            max(1.0, float(best_box_xyxy[2] - best_box_xyxy[0])),
            max(1.0, float(best_box_xyxy[3] - best_box_xyxy[1])),
        )
        gt_center_x, gt_center_y = bbox_center(item.current_bbox)
        gt_grid_x = max(0, min(grid_width - 1, int((gt_center_x / max(1.0, float(frame_shape[1]))) * grid_width)))
        gt_grid_y = max(0, min(grid_height - 1, int((gt_center_y / max(1.0, float(frame_shape[0]))) * grid_height)))
        gt_cell_box_xyxy = decoded_boxes[index, gt_grid_y, gt_grid_x].detach().cpu().tolist()
        gt_cell_prediction = (
            float(gt_cell_box_xyxy[0]),
            float(gt_cell_box_xyxy[1]),
            max(1.0, float(gt_cell_box_xyxy[2] - gt_cell_box_xyxy[0])),
            max(1.0, float(gt_cell_box_xyxy[3] - gt_cell_box_xyxy[1])),
        )
        positive_scores = score_maps[index, pos_y, pos_x]
        positive_inside_y_start = max(pos_y.start, search_y.start)
        positive_inside_y_stop = min(pos_y.stop, search_y.stop)
        positive_inside_x_start = max(pos_x.start, search_x.start)
        positive_inside_x_stop = min(pos_x.stop, search_x.stop)
        positive_inside_count = max(0, positive_inside_y_stop - positive_inside_y_start) * max(
            0,
            positive_inside_x_stop - positive_inside_x_start,
        )
        if positive_inside_count > 0:
            positive_inside_scores = score_maps[
                index,
                positive_inside_y_start:positive_inside_y_stop,
                positive_inside_x_start:positive_inside_x_stop,
            ]
            best_positive_logit = float(positive_inside_scores.max().detach().item())
            better_than_positive = int((search_scores > best_positive_logit).sum().detach().item())
        elif positive_scores.numel() > 0:
            best_positive_logit = float(positive_scores.max().detach().item())
            better_than_positive = int((search_scores > best_positive_logit).sum().detach().item())
        else:
            best_positive_logit = float("nan")
            better_than_positive = -1
        row = {
            "epoch": epoch,
            "step": step,
            "sequence": sample.sequence_path.name,
            "frame": int(sample.frame_number),
            "object_index": index,
            "track_id": int(item.track_id),
            "target_kind": item.target_kind,
            "small_target": int(is_small_target_object(item, v8.DEFAULT_V8_SMALL_TARGET_AREA, v8.DEFAULT_V8_SMALL_TARGET_MAX_SIDE)),
            "iou": exercise.bbox_iou(prediction, item.current_bbox),
            "gt_center_cell_iou": exercise.bbox_iou(gt_cell_prediction, item.current_bbox),
            "pred_bbox": diagnostic_bbox_text(prediction),
            "gt_bbox": diagnostic_bbox_text(item.current_bbox),
            "previous_bbox": diagnostic_bbox_text(item.previous_bbox),
            "search_bbox": diagnostic_bbox_text(expanded_training_search_bbox(item, trainer.search_radius_factor)),
            "previous_current_center_px": bbox_center_distance_xywh(item.previous_bbox, item.current_bbox),
            "previous_current_center_norm": bbox_motion_ratio_xywh(item.previous_bbox, item.current_bbox),
            "target_search_coverage": bbox_coverage_xywh(item.current_bbox, expanded_training_search_bbox(item, trainer.search_radius_factor)),
            "target_center_in_search": int(bbox_contains_center_xywh(expanded_training_search_bbox(item, trainer.search_radius_factor), item.current_bbox)),
            "search_target_area_ratio": bbox_area_xywh(expanded_training_search_bbox(item, trainer.search_radius_factor)) / max(1.0, bbox_area_xywh(item.current_bbox)),
            "best_grid_x": best_x,
            "best_grid_y": best_y,
            "gt_center_grid_x": gt_grid_x,
            "gt_center_grid_y": gt_grid_y,
            "best_score": float(torch.sigmoid(score_maps[index, best_y, best_x]).detach().item()),
            "gt_center_score": float(torch.sigmoid(score_maps[index, gt_grid_y, gt_grid_x]).detach().item()),
            "best_positive_score": float(torch.sigmoid(torch.tensor(best_positive_logit)).item()) if np.isfinite(best_positive_logit) else "",
            "search_cells": int(search_scores.numel()),
            "positive_cells": int((pos_y.stop - pos_y.start) * (pos_x.stop - pos_x.start)),
            "positive_inside_search_cells": int(positive_inside_count),
            "search_scores_better_than_best_positive": int(better_than_positive),
        }
        rows.append(row)
    return rows


def append_diagnostic_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def draw_debug_bbox(image: np.ndarray, bbox: BBox, color: Tuple[int, int, int], label: str) -> None:
    x, y, w, h = bbox
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_prediction_debug_visuals(
    trainer: v8.V8QualityBatchedLoRATTracker,
    output_dir: Path,
    sample: V8TrainingSample,
    frame: np.ndarray,
    head_output,
    objects: Sequence[V8TrainingObject],
    rows: Sequence[Dict[str, object]],
    epoch: int,
    step: int,
    max_visuals_remaining: int,
) -> int:
    if max_visuals_remaining <= 0 or not rows:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    torch = trainer.torch
    score_maps = head_output.score_maps.detach().to(torch.float32)
    saved = 0
    frame_height, frame_width = frame.shape[:2]
    for row in rows:
        if saved >= max_visuals_remaining:
            break
        index = int(row["object_index"])
        item = objects[index]
        image = frame.copy()
        draw_debug_bbox(image, expanded_training_search_bbox(item, trainer.search_radius_factor), (0, 255, 255), "search")
        draw_debug_bbox(image, item.previous_bbox, (255, 128, 0), "prev")
        draw_debug_bbox(image, item.current_bbox, (0, 255, 0), "gt")
        pred_values = tuple(float(value) for value in str(row["pred_bbox"]).split(","))
        draw_debug_bbox(image, pred_values, (0, 0, 255), "pred")
        score_map = torch.sigmoid(score_maps[index]).detach().cpu().numpy().astype(np.float32)
        score_map = score_map - float(score_map.min())
        score_map = score_map / max(1e-6, float(score_map.max()))
        heat = cv2.resize(score_map, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(np.uint8(np.clip(heat * 255.0, 0, 255)), cv2.COLORMAP_JET)
        composite = np.hstack((image, cv2.addWeighted(frame, 0.45, heat, 0.55, 0.0)))
        text = (
            f"IoU {float(row['iou']):.3f} | gt-cell IoU {float(row['gt_center_cell_iou']):.3f} | "
            f"coverage {float(row['target_search_coverage']):.3f} | better-pos {row['search_scores_better_than_best_positive']}"
        )
        cv2.putText(composite, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(composite, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        name = (
            f"epoch{epoch:03d}_step{step:07d}_{sample.sequence_path.name}_"
            f"f{int(sample.frame_number):06d}_trk{int(item.track_id)}_{item.target_kind}.jpg"
        )
        cv2.imwrite(str(output_dir / name), composite)
        saved += 1
    return saved


def contrastive_reid_loss(
    torch_module,
    trainer: v8.V8QualityBatchedLoRATTracker,
    head,
    selected_banks: Sequence[Sequence[object]],
    frame_features,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    temperature: float = 0.07,
):
    if len(objects) == 0:
        return frame_features.sum() * 0.0
    template_vectors = []
    current_vectors = []
    labels = []
    for bank, item in zip(selected_banks, objects):
        if not item.is_present:
            continue
        if bank:
            slot = bank[0]
            template_vectors.append((slot.vector if isinstance(slot, v8.V8TemplateMemorySlot) else slot).to(trainer.device, dtype=torch_module.float32))
        else:
            template_vectors.append(torch_module.zeros((trainer.embed_dim,), device=trainer.device, dtype=torch_module.float32))
        current_vectors.append(trainer._feature_mean_for_bbox(frame_features, item.current_bbox, frame_shape).to(trainer.device, dtype=torch_module.float32))
        labels.append((int(item.track_id), str(item.target_kind)))
    if not template_vectors:
        return frame_features.sum() * 0.0

    template_vectors = torch_module.stack(template_vectors, dim=0)
    current_vectors = torch_module.stack(current_vectors, dim=0)
    template_embeddings = head.module.project_reid(template_vectors)
    current_embeddings = head.module.project_reid(current_vectors)
    logits = torch_module.matmul(template_embeddings, current_embeddings.transpose(0, 1)) / max(1e-4, float(temperature))
    positive_mask = torch_module.zeros_like(logits, dtype=torch_module.bool)
    for row, left in enumerate(labels):
        for col, right in enumerate(labels):
            if left == right:
                positive_mask[row, col] = True
    if not bool(positive_mask.any().detach().cpu().item()):
        return logits.sum() * 0.0

    row_log_den = torch_module.logsumexp(logits, dim=1)
    row_log_pos = torch_module.logsumexp(logits.masked_fill(~positive_mask, -1.0e4), dim=1)
    col_log_den = torch_module.logsumexp(logits, dim=0)
    col_log_pos = torch_module.logsumexp(logits.masked_fill(~positive_mask, -1.0e4), dim=0)
    return 0.5 * ((row_log_den - row_log_pos).mean() + (col_log_den - col_log_pos).mean())


def grid_cell_for_bbox_center(bbox: BBox, frame_shape: Tuple[int, ...], grid_height: int, grid_width: int) -> Tuple[int, int]:
    center_x, center_y = bbox_center(bbox)
    frame_height, frame_width = frame_shape[:2]
    grid_x = int((center_x / max(1.0, float(frame_width))) * float(grid_width))
    grid_y = int((center_y / max(1.0, float(frame_height))) * float(grid_height))
    grid_x = max(0, min(grid_width - 1, grid_x))
    grid_y = max(0, min(grid_height - 1, grid_y))
    return grid_y, grid_x


def dcfst_candidate_discrimination_loss(
    torch_module,
    trainer: v8.V8QualityBatchedLoRATTracker,
    head_output,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    negative_candidate_count: int,
    margin: float,
) -> Tuple[object, V8CandidateDiscriminationStats]:
    score_maps = head_output.score_maps.to(torch_module.float32)
    if score_maps.numel() == 0 or len(objects) == 0:
        return score_maps.sum() * 0.0, V8CandidateDiscriminationStats()

    _, grid_height, grid_width = score_maps.shape
    total_loss = score_maps.sum() * 0.0
    object_terms = 0
    positive_candidates = 0
    negative_candidates = 0

    for index, item in enumerate(objects):
        if not item.is_present:
            continue
        pos_y, pos_x = bbox_to_grid_slice(item.current_bbox, frame_shape, grid_height, grid_width)
        positive_logits = score_maps[index, pos_y, pos_x].reshape(-1)
        if positive_logits.numel() == 0:
            continue
        positive_logit = torch_module.logsumexp(positive_logits, dim=0) - torch_module.log(
            torch_module.as_tensor(float(positive_logits.numel()), device=trainer.device, dtype=torch_module.float32)
        )
        positive_candidates += int(positive_logits.numel())

        search_y, search_x = bbox_to_grid_slice(
            expanded_training_search_bbox(item, trainer.search_radius_factor),
            frame_shape,
            grid_height,
            grid_width,
        )
        candidate_cells: List[Tuple[int, int]] = []
        seen = set()

        def add_cell(grid_y: int, grid_x: int) -> None:
            if grid_y < search_y.start or grid_y >= search_y.stop or grid_x < search_x.start or grid_x >= search_x.stop:
                return
            if pos_y.start <= grid_y < pos_y.stop and pos_x.start <= grid_x < pos_x.stop:
                return
            key = (int(grid_y), int(grid_x))
            if key in seen:
                return
            seen.add(key)
            candidate_cells.append(key)

        for distractor_bbox in item.distractor_bboxes:
            if not bbox_contains_center_xywh(expanded_training_search_bbox(item, trainer.search_radius_factor), distractor_bbox):
                continue
            if not bbox_center_inside_frame(distractor_bbox, frame_shape):
                continue
            grid_y, grid_x = grid_cell_for_bbox_center(distractor_bbox, frame_shape, grid_height, grid_width)
            add_cell(grid_y, grid_x)

        max_background = max(0, int(negative_candidate_count) - len(candidate_cells))
        if max_background > 0:
            ys = np.linspace(search_y.start, max(search_y.start, search_y.stop - 1), num=max(1, int(np.sqrt(max_background)) + 1), dtype=np.int64)
            xs = np.linspace(search_x.start, max(search_x.start, search_x.stop - 1), num=max(1, int(np.sqrt(max_background)) + 1), dtype=np.int64)
            for grid_y in ys.tolist():
                for grid_x in xs.tolist():
                    add_cell(int(grid_y), int(grid_x))
                    if len(candidate_cells) >= int(negative_candidate_count):
                        break
                if len(candidate_cells) >= int(negative_candidate_count):
                    break

        if not candidate_cells:
            continue
        candidate_cells = candidate_cells[: max(1, int(negative_candidate_count))]
        neg_y = torch_module.as_tensor([cell[0] for cell in candidate_cells], device=trainer.device, dtype=torch_module.long)
        neg_x = torch_module.as_tensor([cell[1] for cell in candidate_cells], device=trainer.device, dtype=torch_module.long)
        negative_logits = score_maps[index, neg_y, neg_x]
        if negative_logits.numel() == 0:
            continue

        ranking_logits = torch_module.cat((positive_logit.reshape(1), negative_logits.reshape(-1)), dim=0)
        ranking_target = torch_module.zeros((1,), device=trainer.device, dtype=torch_module.long)
        cross_entropy = torch_module.nn.functional.cross_entropy(ranking_logits.reshape(1, -1), ranking_target)
        hard_negative_margin = torch_module.nn.functional.softplus(negative_logits - positive_logit + float(margin)).mean()
        total_loss = total_loss + cross_entropy + hard_negative_margin
        object_terms += 1
        negative_candidates += int(negative_logits.numel())

    if object_terms <= 0:
        return score_maps.sum() * 0.0, V8CandidateDiscriminationStats()
    return total_loss / float(object_terms), V8CandidateDiscriminationStats(
        objects=object_terms,
        positive_candidates=positive_candidates,
        negative_candidates=negative_candidates,
    )


def object_assignment_discrimination_loss(
    torch_module,
    trainer: v8.V8QualityBatchedLoRATTracker,
    head_output,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    margin: float,
) -> Tuple[object, V8AssignmentDiscriminationStats]:
    score_maps = head_output.score_maps.to(torch_module.float32)
    if score_maps.numel() == 0 or len(objects) < 2:
        return score_maps.sum() * 0.0, V8AssignmentDiscriminationStats()

    _, grid_height, grid_width = score_maps.shape
    total_loss = score_maps.sum() * 0.0
    object_terms = 0
    positive_candidates = 0
    negative_candidates = 0
    present = [(index, item) for index, item in enumerate(objects) if item.is_present]
    if len(present) < 2:
        return score_maps.sum() * 0.0, V8AssignmentDiscriminationStats()

    for index, item in present:
        pos_y, pos_x = bbox_to_grid_slice(item.current_bbox, frame_shape, grid_height, grid_width)
        positive_logits = score_maps[index, pos_y, pos_x].reshape(-1)
        if positive_logits.numel() == 0:
            continue
        positive_logit = torch_module.logsumexp(positive_logits, dim=0) - torch_module.log(
            torch_module.as_tensor(float(positive_logits.numel()), device=trainer.device, dtype=torch_module.float32)
        )
        positive_candidates += int(positive_logits.numel())

        other_logits = []
        for other_index, other in present:
            if other_index == index or other.track_id == item.track_id:
                continue
            neg_y, neg_x = bbox_to_grid_slice(other.current_bbox, frame_shape, grid_height, grid_width)
            logits = score_maps[index, neg_y, neg_x].reshape(-1)
            if logits.numel() > 0:
                other_logits.append(logits.max())
                negative_candidates += int(logits.numel())
        if not other_logits:
            continue
        negative_logits = torch_module.stack(other_logits)
        total_loss = total_loss + torch_module.nn.functional.softplus(negative_logits - positive_logit + float(margin)).mean()
        object_terms += 1

    if object_terms <= 0:
        return score_maps.sum() * 0.0, V8AssignmentDiscriminationStats()
    return total_loss / float(object_terms), V8AssignmentDiscriminationStats(
        objects=object_terms,
        positive_candidates=positive_candidates,
        negative_candidates=negative_candidates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V8 shared-frame LoRAT object-conditioned LoRA head.")
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--tao-root", type=Path, help="Optional TAO/TAO-OW subset root to mix into V8 head training.")
    parser.add_argument("--tao-split", default="train", help="TAO annotation split for mixed training.")
    parser.add_argument("--tao-val-split", default="validation", help="TAO annotation split for mixed validation.")
    parser.add_argument("--tao-use-freeform", action="store_true", help="Use TAO *_with_freeform annotations when available.")
    parser.add_argument("--tao-frame-stride", type=int, default=1)
    parser.add_argument("--tao-val-frame-stride", type=int, default=1)
    parser.add_argument("--tao-max-sequences", type=int, default=0)
    parser.add_argument("--tao-max-val-sequences", type=int, default=0)
    parser.add_argument("--tao-max-samples", type=int, default=0)
    parser.add_argument("--tao-max-val-samples", type=int, default=256)
    parser.add_argument("--class-id", type=int, action="append", help="Optional GT class IDs to train. Defaults to all valid annotated tracks.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument(
        "--target-region-mode",
        default="mixed",
        choices=("full", "parts", "mixed"),
        help="Selected target regions used for object-agnostic training. 'mixed' trains full boxes plus deterministic sub-boxes.",
    )
    parser.add_argument(
        "--target-regions-per-object",
        type=int,
        default=4,
        help="How many selected-region variants to emit per annotated object before max-objects truncation.",
    )
    parser.add_argument("--template-area-factor", type=float, default=2.0, help="LoRaT/SiamFC template crop area factor.")
    parser.add_argument("--search-area-factor", type=float, default=4.0, help="LoRaT/SiamFC search crop area factor.")
    parser.add_argument("--search-scale-jitter", type=float, default=0.25, help="LoRaT training search scale jitter.")
    parser.add_argument("--search-translation-jitter", type=float, default=3.0, help="LoRaT training search translation jitter.")
    parser.add_argument("--search-min-object-size", type=float, default=10.0, help="Minimum object size in the LoRaT search crop.")
    parser.add_argument(
        "--search-anchor-mode",
        default="union",
        choices=("current", "previous", "union"),
        help="Which box anchors supervised search regions. 'union' matches runtime recovery by spanning previous/noisy and current target boxes.",
    )
    parser.add_argument(
        "--disable-search-target-repair",
        dest="repair_search_to_target",
        action="store_false",
        default=True,
        help="Do not expand jittered training search regions that fail to cover the positive target.",
    )
    parser.add_argument(
        "--search-target-padding-fraction",
        type=float,
        default=0.05,
        help="Extra padding added when repairing a training search region to include the target.",
    )
    parser.add_argument("--template-sampling", default="mixed", choices=("first", "previous", "mixed", "window"), help="Use initial, recent previous, mixed, or random short-window template frames for training.")
    parser.add_argument("--previous-box-jitter", type=float, default=0.10, help="Relative jitter applied to previous boxes before search-region construction.")
    parser.add_argument("--sequence-window-length", type=int, default=5, help="Number of recent same-track annotations available for short-window template sampling.")
    parser.add_argument("--lost-target-probability", type=float, default=0.10, help="Probability of adding a recently visible but currently absent track as a no-confident-target training query.")
    parser.add_argument("--max-lost-targets-per-frame", type=int, default=2, help="Maximum missing/lost target queries added to one training frame.")
    parser.add_argument("--max-missing-gap-frames", type=int, default=30, help="Only sample missing-target queries within this many frames of the last visible annotation.")
    parser.add_argument(
        "--training-frame-mode",
        default="staged",
        choices=("full", "search_crop", "staged"),
        help="Train on full shared frames, LoRaT-style search crops, or search crops for the first --crop-stage-epochs then full frames.",
    )
    parser.add_argument(
        "--crop-stage-epochs",
        type=int,
        default=2,
        help="When --training-frame-mode=staged, train on LoRaT-style search crops for this many initial epochs.",
    )
    parser.add_argument(
        "--disable-lorat-augmentation",
        dest="lorat_augmentation",
        action="store_false",
        default=True,
        help="Disable LoRaT-parity flip/color/DeiT training augmentation.",
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--val-frame-stride", type=int, default=15)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-val-sequences", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Maximum training epochs to attempt. Long Slurm jobs should pair this with --max-wall-hours.",
    )
    parser.add_argument(
        "--max-train-samples-per-epoch",
        type=int,
        default=0,
        help="If positive, train on this many shuffled samples per epoch. This keeps epoch cadence meaningful on large mixed datasets.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        default=0.0,
        help="Stop after this many wall-clock hours, then run epoch probes and save cleanly. 0 disables.",
    )
    parser.add_argument(
        "--eval-interval-epochs",
        type=int,
        default=5,
        help="Run train/validation IoU probes every N epochs, plus epoch 1 and the final wall-clock epoch.",
    )
    parser.add_argument(
        "--diagnostic-interval-epochs",
        type=int,
        default=10,
        help="Run heavier ReID and IoU failure diagnostics every N epochs that are evaluated. 0 disables heavy epoch diagnostics.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--ltrb-loss-weight", type=float, default=1.0, help="SmoothL1 loss weight on LoRaT-style normalized l/t/r/b offsets.")
    parser.add_argument("--reid-loss-weight", type=float, default=0.50, help="Contrastive same-track/different-track ReID loss weight.")
    parser.add_argument("--center-positive-weight", type=float, default=0.5, help="0 keeps LoRaT uniform positives; higher values emphasize positive cells near the target center for box losses.")
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=3.0, help="Extra BCE weight for cells covered by other visible track boxes.")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0, help="Focal-style objectness modulation gamma. 0 keeps plain LoRaT-style BCE.")
    parser.add_argument("--small-target-loss-weight", type=float, default=2.5, help="Positive-cell loss multiplier for small/manual selected targets such as heads or face crops.")
    parser.add_argument("--small-target-area-threshold", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_AREA, help="Area threshold used to identify small selected targets during training.")
    parser.add_argument("--small-target-max-side", type=float, default=v8.DEFAULT_V8_SMALL_TARGET_MAX_SIDE, help="Max side threshold used to identify small selected targets during training.")
    parser.add_argument("--dcfst-discrimination-weight", type=float, default=0.70, help="Weight for selected-target-vs-distractor candidate ranking loss.")
    parser.add_argument("--dcfst-negative-candidates", type=int, default=48, help="Maximum distractor/background candidate cells per object for DCFST-style ranking.")
    parser.add_argument("--dcfst-margin", type=float, default=0.35, help="Margin used by DCFST-style hard-negative ranking.")
    parser.add_argument(
        "--disable-iou-aware-classification",
        dest="iou_aware_classification",
        action="store_false",
        default=True,
        help="Use binary 1/0 positive labels instead of LoRAT-style IoU-aware classification labels.",
    )
    parser.add_argument(
        "--iou-aware-warmup-steps",
        type=int,
        default=250,
        help="Train positive objectness as 1.0 for the first N steps before switching to IoU-aware labels.",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=1.25, help="Legacy option retained for compatibility; not used by LoRAT-style targets.")
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lorat-root", type=Path, default=mot.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(mot.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT backbone weight. Defaults from --lorat-config.")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--v8-head-hidden-dim", type=int, default=256)
    parser.add_argument("--v8-head-lora-rank", type=int, default=16)
    parser.add_argument("--v8-head-rank", type=int, default=mot.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--v8-search-radius-factor", type=float, default=2.25)
    parser.add_argument("--training-memory-slots", type=int, default=2, help="Template/memory slots per object used by the V8 head during full-frame training.")
    parser.add_argument("--geometry-only-epochs", type=int, default=2, help="Initial epochs that train localization only, before ReID/assignment/recovery objectives.")
    parser.add_argument("--hard-negative-start-epoch", type=int, default=3, help="First epoch where extra hard-negative weighting is enabled.")
    parser.add_argument("--reid-start-epoch", type=int, default=5, help="First epoch where ReID contrastive loss is enabled.")
    parser.add_argument("--dcfst-start-epoch", type=int, default=3, help="First epoch where DCFST-style candidate ranking is enabled.")
    parser.add_argument("--assignment-start-epoch", type=int, default=3, help="First epoch where explicit cross-object assignment ranking is enabled.")
    parser.add_argument("--assignment-discrimination-weight", type=float, default=0.50, help="Weight for penalizing object queries that score other objects higher than their own target.")
    parser.add_argument("--assignment-margin", type=float, default=0.25, help="Margin used by explicit cross-object assignment ranking.")
    parser.add_argument("--closed-loop-start-epoch", type=int, default=5, help="First epoch where model-predicted boxes can be used as simulated previous tracker state.")
    parser.add_argument("--closed-loop-probability", type=float, default=0.35, help="Per-object probability of replacing the previous/search state with the model's own prediction.")
    parser.add_argument("--resume-head", type=Path)
    parser.add_argument("--output", type=Path, default=mot.PROJECT_ROOT / "models" / "lorat" / "v8_head.pt")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=250,
        help="Save an intermediate checkpoint every N optimizer steps. 0 disables step checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for intermediate checkpoints. Defaults beside --output.",
    )
    parser.add_argument(
        "--train-diagnostic-samples",
        type=int,
        default=64,
        help="Validate on N spread-out training samples each epoch to prove the head can overfit/follow known objects. 0 disables.",
    )
    parser.add_argument("--diagnostic-csv", type=Path, help="Optional CSV path for per-epoch training diagnostics.")
    parser.add_argument(
        "--overfit-smoke",
        action="store_true",
        help="Force a tiny one-sequence training run and fail the process if the train probe does not pass the quality gate.",
    )
    parser.add_argument("--overfit-smoke-samples", type=int, default=16)
    parser.add_argument("--overfit-single-object", action="store_true", help="Limit samples to one object each for target-geometry/overfit debugging.")
    parser.add_argument("--debug-sequence-name", help="Optional exact sequence folder name to train/debug on.")
    parser.add_argument("--debug-track-id", type=int, help="Optional exact track id to train/debug on.")
    parser.add_argument("--geometry-diagnostic-csv", type=Path, help="Optional CSV path for per-sample geometry diagnostics.")
    parser.add_argument("--iou-diagnostic-csv", type=Path, help="Optional CSV path for per-prediction IoU failure diagnostics.")
    parser.add_argument("--debug-visual-dir", type=Path, help="Optional directory for visual geometry/prediction debug images.")
    parser.add_argument("--debug-visual-samples", type=int, default=24)
    parser.add_argument("--debug-visual-every-epoch", action="store_true")
    parser.add_argument("--quality-gate-mean-iou", type=float, default=0.0)
    parser.add_argument("--quality-gate-iou50", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_checkpoint(
    torch_module,
    path: Path,
    head,
    args: argparse.Namespace,
    epoch: int,
    steps: int,
    val_mean_iou: Optional[float] = None,
    val_iou50: Optional[float] = None,
    full_val_mean_iou: Optional[float] = None,
    full_val_iou50: Optional[float] = None,
    selection_val_mean_iou: Optional[float] = None,
    selection_val_iou50: Optional[float] = None,
    selection_metric_frame_mode: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model": head.state_dict(),
            "epoch": epoch,
            "steps": steps,
            "lorat_config": args.lorat_config,
            "head_hidden_dim": args.v8_head_hidden_dim,
            "head_lora_rank": args.v8_head_lora_rank,
            "gaussian_sigma": args.gaussian_sigma,
            "head_architecture": "template_patch_lora_conditioned",
            "box_parameterization": "lorat_anchor_free_ltrb",
            "loss_style": "lorat_iou_aware_focal_bce_giou_ltrb_reid_dcfst_candidate",
            "iou_aware_classification": args.iou_aware_classification,
            "iou_aware_warmup_steps": args.iou_aware_warmup_steps,
            "negative_loss_weight": args.negative_loss_weight,
            "hard_negative_loss_weight": args.hard_negative_loss_weight,
            "focal_loss_gamma": args.focal_loss_gamma,
            "small_target_loss_weight": args.small_target_loss_weight,
            "small_target_area_threshold": args.small_target_area_threshold,
            "small_target_max_side": args.small_target_max_side,
            "dcfst_discrimination_weight": args.dcfst_discrimination_weight,
            "dcfst_negative_candidates": args.dcfst_negative_candidates,
            "dcfst_margin": args.dcfst_margin,
            "ltrb_loss_weight": args.ltrb_loss_weight,
            "reid_loss_weight": args.reid_loss_weight,
            "center_positive_weight": args.center_positive_weight,
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "val_split": args.val_split,
            "tao_root": "" if args.tao_root is None else str(args.tao_root),
            "tao_split": args.tao_split,
            "tao_val_split": args.tao_val_split,
            "tao_use_freeform": bool(args.tao_use_freeform),
            "tao_frame_stride": args.tao_frame_stride,
            "tao_val_frame_stride": args.tao_val_frame_stride,
            "tao_max_sequences": args.tao_max_sequences,
            "tao_max_val_sequences": args.tao_max_val_sequences,
            "tao_max_samples": args.tao_max_samples,
            "tao_max_val_samples": args.tao_max_val_samples,
            "max_train_samples_per_epoch": args.max_train_samples_per_epoch,
            "eval_interval_epochs": args.eval_interval_epochs,
            "diagnostic_interval_epochs": args.diagnostic_interval_epochs,
            "search_radius_factor": args.v8_search_radius_factor,
            "target_region_mode": args.target_region_mode,
            "target_regions_per_object": args.target_regions_per_object,
            "template_area_factor": args.template_area_factor,
            "search_area_factor": args.search_area_factor,
            "search_scale_jitter": args.search_scale_jitter,
            "search_translation_jitter": args.search_translation_jitter,
            "search_min_object_size": args.search_min_object_size,
            "search_anchor_mode": args.search_anchor_mode,
            "repair_search_to_target": args.repair_search_to_target,
            "search_target_padding_fraction": args.search_target_padding_fraction,
            "template_sampling": args.template_sampling,
            "previous_box_jitter": args.previous_box_jitter,
            "sequence_window_length": args.sequence_window_length,
            "lost_target_probability": args.lost_target_probability,
            "max_lost_targets_per_frame": args.max_lost_targets_per_frame,
            "max_missing_gap_frames": args.max_missing_gap_frames,
            "training_frame_mode": args.training_frame_mode,
            "crop_stage_epochs": args.crop_stage_epochs,
            "training_memory_slots": args.training_memory_slots,
            "geometry_only_epochs": args.geometry_only_epochs,
            "hard_negative_start_epoch": args.hard_negative_start_epoch,
            "reid_start_epoch": args.reid_start_epoch,
            "dcfst_start_epoch": args.dcfst_start_epoch,
            "assignment_start_epoch": args.assignment_start_epoch,
            "assignment_discrimination_weight": args.assignment_discrimination_weight,
            "assignment_margin": args.assignment_margin,
            "closed_loop_start_epoch": args.closed_loop_start_epoch,
            "closed_loop_probability": args.closed_loop_probability,
            "lorat_augmentation": args.lorat_augmentation,
            "command": " ".join(sys.argv),
            "torch_version": getattr(torch_module, "__version__", ""),
            "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", ""),
            "cuda_available": bool(torch_module.cuda.is_available()) if hasattr(torch_module, "cuda") else False,
            "cuda_device_name": torch_module.cuda.get_device_name(0) if hasattr(torch_module, "cuda") and torch_module.cuda.is_available() else "",
            "val_mean_iou": val_mean_iou,
            "val_iou50": val_iou50,
            "full_val_mean_iou": full_val_mean_iou,
            "full_val_iou50": full_val_iou50,
            "selection_val_mean_iou": selection_val_mean_iou,
            "selection_val_iou50": selection_val_iou50,
            "selection_metric_frame_mode": selection_metric_frame_mode,
        },
        str(path),
    )


def build_selected_banks(
    trainer: v8.V8QualityBatchedLoRATTracker,
    sample: V8TrainingSample,
    objects: Sequence[V8TrainingObject],
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    augmentation_spec: Optional[LoRATTrainingAugmentationSpec] = None,
    memory_slots: int = 1,
):
    augmentation_spec = augmentation_spec or LoRATTrainingAugmentationSpec(enabled=False)
    selected_banks = []
    use_cache = not augmentation_spec.enabled

    def load_template_features(frame_number: int, context: str):
        template_key = (sample.sequence_path, int(frame_number))
        cached_template = template_feature_cache.get(template_key) if use_cache else None
        if cached_template is None:
            template_index = exercise.frame_to_image_index(frame_number)
            if template_index < 0 or template_index >= len(sample.image_paths):
                return None
            template_frame = try_load_frame(
                sample.image_paths[template_index],
                f"{context} sequence={sample.sequence_path.name} frame={frame_number}",
            )
            if template_frame is None:
                return None
            if augmentation_spec.enabled:
                template_frame = apply_lorat_image_augmentation(template_frame, augmentation_spec, "template")
            with trainer.torch.no_grad():
                template_features = trainer.shared_frame_encoder.encode(template_frame).feature_map.detach()
            template_shape = template_frame.shape
            if use_cache:
                template_feature_cache[template_key] = (template_features, template_shape)
            return template_features, template_shape
        return cached_template

    def maybe_flip_template_bbox_pair(bbox: BBox, context_bbox: BBox, template_width: int) -> Tuple[BBox, BBox]:
        if augmentation_spec.enabled and augmentation_spec.template_flip:
            return flip_bbox_horizontal(bbox, template_width), flip_bbox_horizontal(context_bbox, template_width)
        return bbox, context_bbox

    for item in objects:
        slot_specs: List[Tuple[int, BBox, BBox, str]] = [
            (item.template_frame, item.template_bbox, item.template_context_bbox, item.target_kind)
        ]
        if int(memory_slots) > 1 and item.is_present:
            slot_specs.append((item.previous_frame, item.previous_bbox, item.previous_context_bbox, f"{item.target_kind}:prev"))

        bank = []
        seen_slots = set()
        for frame_number, bbox, context_bbox, label in slot_specs[: max(1, int(memory_slots))]:
            key = (int(frame_number), tuple(round(float(value), 3) for value in bbox), label)
            if key in seen_slots:
                continue
            seen_slots.add(key)
            loaded = load_template_features(int(frame_number), "template-memory")
            if loaded is None:
                return None
            template_features, template_shape = loaded
            aug_bbox, aug_context_bbox = maybe_flip_template_bbox_pair(bbox, context_bbox, int(template_shape[1]))
            bank.append(
                trainer._template_slot_for_bbox(
                    template_features,
                    aug_bbox,
                    template_shape,
                    label,
                    int(frame_number),
                    1.0,
                    aug_context_bbox,
                )
            )
        selected_banks.append(bank)
    return selected_banks


def validate_head(
    trainer: v8.V8QualityBatchedLoRATTracker,
    dataset: MOTFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    training_frame_mode: str = "full",
    memory_slots: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    if len(dataset) == 0 or max_samples == 0:
        return None, None
    torch = trainer.torch
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()

    sample_indices = spread_sample_indices(len(dataset), max_samples)

    iou_sum = 0.0
    hit50 = 0
    count = 0
    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            current_index = exercise.frame_to_image_index(sample.frame_number)
            current_frame = try_load_frame(
                sample.image_paths[current_index],
                f"validation sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            for eval_frame, eval_objects, _ in crop_training_batches(current_frame, sample.objects, training_frame_mode):
                frame_features = trainer.shared_frame_encoder.encode(eval_frame).feature_map.detach()
                selected_banks = build_selected_banks(
                    trainer,
                    sample,
                    eval_objects,
                    template_feature_cache,
                    memory_slots=memory_slots if training_frame_mode == "full" else 1,
                )
                if selected_banks is None:
                    continue
                head_output = head.score(frame_features, selected_banks)
                predictions = decode_predictions(trainer, head_output, eval_objects, eval_frame.shape)
                for prediction, item in zip(predictions, eval_objects):
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


def collect_iou_failure_diagnostics(
    trainer: v8.V8QualityBatchedLoRATTracker,
    dataset: MOTFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    epoch: int,
    step: int,
    phase: str,
    output_csv: Path,
    visual_dir: Optional[Path] = None,
    max_visuals: int = 0,
    training_frame_mode: str = "full",
    memory_slots: int = 1,
) -> int:
    if len(dataset) == 0 or max_samples == 0:
        return 0
    torch = trainer.torch
    head = trainer.object_conditioned_head
    was_training = head.module.training
    head.module.eval()
    saved_visuals = 0

    sample_indices = spread_sample_indices(len(dataset), max_samples)

    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            current_index = exercise.frame_to_image_index(sample.frame_number)
            current_frame = try_load_frame(
                sample.image_paths[current_index],
                f"validation sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            for diag_frame, diag_objects, diag_mode in crop_training_batches(current_frame, sample.objects, training_frame_mode):
                frame_features = trainer.shared_frame_encoder.encode(diag_frame).feature_map.detach()
                selected_banks = build_selected_banks(
                    trainer,
                    sample,
                    diag_objects,
                    template_feature_cache,
                    memory_slots=memory_slots if training_frame_mode == "full" else 1,
                )
                if selected_banks is None:
                    continue
                head_output = head.score(frame_features, selected_banks)
                rows = prediction_diagnostic_rows(trainer, sample, head_output, diag_objects, diag_frame.shape, epoch, step)
                for row in rows:
                    row["phase"] = phase
                    row["sample_index"] = sample_index
                    row["training_frame_mode"] = diag_mode
                append_diagnostic_rows(output_csv, rows)
                if visual_dir is not None and saved_visuals < max_visuals:
                    saved_visuals += save_prediction_debug_visuals(
                        trainer,
                        visual_dir,
                        sample,
                        diag_frame,
                        head_output,
                        diag_objects,
                        rows,
                        epoch,
                        step,
                        max_visuals - saved_visuals,
                    )
    if was_training:
        head.module.train()
    return saved_visuals


def reid_similarity_probe(
    trainer: v8.V8QualityBatchedLoRATTracker,
    dataset: MOTFrameHeadDataset,
    max_samples: int,
    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]],
    training_frame_mode: str = "full",
    memory_slots: int = 1,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(dataset) == 0 or max_samples == 0:
        return None, None, None
    torch = trainer.torch
    head = trainer.object_conditioned_head
    if not hasattr(head.module, "project_reid"):
        return None, None, None
    was_training = head.module.training
    head.module.eval()

    sample_indices = spread_sample_indices(len(dataset), max_samples)

    same_sum = 0.0
    same_count = 0
    diff_sum = 0.0
    diff_count = 0
    top1_count = 0
    top1_total = 0
    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            present_objects = [item for item in sample.objects if item.is_present]
            if len(present_objects) < 1:
                continue
            current_index = exercise.frame_to_image_index(sample.frame_number)
            current_frame = try_load_frame(
                sample.image_paths[current_index],
                f"diagnostic sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            for probe_frame, probe_objects, _ in crop_training_batches(current_frame, present_objects, training_frame_mode):
                frame_features = trainer.shared_frame_encoder.encode(probe_frame).feature_map.detach()
                selected_banks = build_selected_banks(
                    trainer,
                    sample,
                    probe_objects,
                    template_feature_cache,
                    memory_slots=memory_slots if training_frame_mode == "full" else 1,
                )
                if selected_banks is None:
                    continue
                template_vectors = []
                current_vectors = []
                labels = []
                for bank, item in zip(selected_banks, probe_objects):
                    if not bank:
                        continue
                    slot = bank[0]
                    template_vectors.append((slot.vector if isinstance(slot, v8.V8TemplateMemorySlot) else slot).to(trainer.device, dtype=torch.float32))
                    current_vectors.append(trainer._feature_mean_for_bbox(frame_features, item.current_bbox, probe_frame.shape).to(trainer.device, dtype=torch.float32))
                    labels.append((int(item.track_id), str(item.target_kind)))
                if len(template_vectors) < 1:
                    continue
                template_embeddings = head.module.project_reid(torch.stack(template_vectors, dim=0))
                current_embeddings = head.module.project_reid(torch.stack(current_vectors, dim=0))
                similarity = torch.matmul(template_embeddings, current_embeddings.transpose(0, 1))
                for row, left in enumerate(labels):
                    row_positive_cols = [col for col, right in enumerate(labels) if right == left]
                    row_negative_cols = [col for col, right in enumerate(labels) if right != left]
                    if row_positive_cols:
                        same_values = similarity[row, row_positive_cols]
                        same_sum += float(same_values.sum().detach().item())
                        same_count += int(same_values.numel())
                        best_col = int(torch.argmax(similarity[row]).detach().item())
                        top1_count += 1 if labels[best_col] == left else 0
                        top1_total += 1
                    if row_negative_cols:
                        diff_values = similarity[row, row_negative_cols]
                        diff_sum += float(diff_values.sum().detach().item())
                        diff_count += int(diff_values.numel())

    if was_training:
        head.module.train()
    same_mean = same_sum / float(same_count) if same_count else None
    diff_mean = diff_sum / float(diff_count) if diff_count else None
    top1 = top1_count / float(top1_total) if top1_total else None
    return same_mean, diff_mean, top1


def append_diagnostic_row(path: Path, row: Dict[str, object]) -> None:
    append_diagnostic_rows(path, [row])


def spread_sample_indices(total: int, max_samples: int) -> List[int]:
    total = max(0, int(total))
    max_samples = int(max_samples)
    if total == 0:
        return []
    if max_samples <= 0 or max_samples >= total:
        return list(range(total))
    return sorted({int(index) for index in np.linspace(0, total - 1, num=max_samples)})


def training_phase_settings(args: argparse.Namespace, epoch: int) -> V8TrainingPhaseSettings:
    epoch = int(epoch)
    if epoch <= max(0, int(args.geometry_only_epochs)):
        return V8TrainingPhaseSettings(
            name="geometry_warmup",
            hard_negative_loss_weight=1.0,
            reid_loss_weight=0.0,
            dcfst_discrimination_weight=0.0,
            assignment_discrimination_weight=0.0,
            closed_loop_probability=0.0,
        )

    hard_negative_weight = float(args.hard_negative_loss_weight) if epoch >= int(args.hard_negative_start_epoch) else 1.0
    reid_weight = float(args.reid_loss_weight) if epoch >= int(args.reid_start_epoch) else 0.0
    dcfst_weight = float(args.dcfst_discrimination_weight) if epoch >= int(args.dcfst_start_epoch) else 0.0
    assignment_weight = float(args.assignment_discrimination_weight) if epoch >= int(args.assignment_start_epoch) else 0.0
    closed_loop_probability = float(args.closed_loop_probability) if epoch >= int(args.closed_loop_start_epoch) else 0.0
    phase_name = "association_recovery" if closed_loop_probability > 0.0 or reid_weight > 0.0 else "full_frame_discrimination"
    return V8TrainingPhaseSettings(
        name=phase_name,
        hard_negative_loss_weight=max(0.0, hard_negative_weight),
        reid_loss_weight=max(0.0, reid_weight),
        dcfst_discrimination_weight=max(0.0, dcfst_weight),
        assignment_discrimination_weight=max(0.0, assignment_weight),
        closed_loop_probability=max(0.0, min(1.0, closed_loop_probability)),
    )


def apply_closed_loop_rollout(
    trainer: v8.V8QualityBatchedLoRATTracker,
    head_output,
    objects: Sequence[V8TrainingObject],
    frame_shape: Tuple[int, ...],
    probability: float,
    rng: np.random.Generator,
    search_area_factor: float,
    search_scale_jitter: float,
    search_translation_jitter: float,
    search_min_object_size: float,
    repair_search_to_target: bool,
    search_target_padding_fraction: float,
) -> Tuple[List[V8TrainingObject], int]:
    probability = max(0.0, min(1.0, float(probability)))
    if probability <= 0.0 or not objects:
        return list(objects), 0
    predictions = decode_predictions(trainer, head_output, objects, frame_shape)
    updated: List[V8TrainingObject] = []
    applied = 0
    for item, prediction in zip(objects, predictions):
        if not item.is_present or float(rng.random()) > probability:
            updated.append(item)
            continue
        predicted_previous = clamp_bbox_to_frame_shape(tuple(float(value) for value in prediction), frame_shape)
        search_anchor_bbox = union_bbox_xywh(predicted_previous, item.current_bbox)
        search_bbox = siamfc_search_bbox(
            search_anchor_bbox,
            (224, 224),
            search_area_factor,
            search_scale_jitter,
            search_translation_jitter,
            search_min_object_size,
            rng,
        )
        if repair_search_to_target:
            search_bbox = repair_search_bbox_to_cover_target(
                search_bbox,
                item.current_bbox,
                search_target_padding_fraction,
            )
        updated.append(
            replace(
                item,
                previous_bbox=predicted_previous,
                search_bbox=search_bbox,
            )
        )
        applied += 1
    return updated, applied


def main() -> int:
    args = parse_args()
    if args.overfit_smoke:
        smoke_samples = max(1, int(args.overfit_smoke_samples))
        args.max_sequences = 1 if args.max_sequences <= 0 else min(args.max_sequences, 1)
        args.max_val_sequences = 1 if args.max_val_sequences <= 0 else min(args.max_val_sequences, 1)
        args.max_samples = smoke_samples if args.max_samples <= 0 else min(args.max_samples, smoke_samples)
        args.max_val_samples = smoke_samples if args.max_val_samples <= 0 else min(args.max_val_samples, smoke_samples)
        args.train_diagnostic_samples = smoke_samples
        args.eval_interval_epochs = 1
        args.diagnostic_interval_epochs = 1
        args.quality_gate_mean_iou = max(float(args.quality_gate_mean_iou), 0.5)
        args.quality_gate_iou50 = max(float(args.quality_gate_iou50), 0.5)
    if args.overfit_single_object:
        args.max_objects = 1
        args.target_regions_per_object = 1
        args.lost_target_probability = 0.0
        args.max_lost_targets_per_frame = 0
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    import torch.nn.functional as F

    class_ids = parse_class_ids(args.class_id)

    train_datasets: List[object] = []
    train_dataset_base = MOTFrameHeadDataset(
        args.dataset_root,
        args.split,
        class_ids,
        args.min_visibility,
        args.max_objects,
        args.frame_stride,
        args.max_sequences,
        args.max_samples,
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
        args.lost_target_probability,
        args.max_lost_targets_per_frame,
        args.max_missing_gap_frames,
        args.search_anchor_mode,
        args.repair_search_to_target,
        args.search_target_padding_fraction,
        args.debug_sequence_name,
        args.debug_track_id,
    )
    train_datasets.append(train_dataset_base)
    if args.tao_root is not None:
        train_datasets.append(
            TAOFrameHeadDataset(
                args.tao_root,
                args.tao_split,
                class_ids,
                args.min_visibility,
                args.max_objects,
                args.tao_frame_stride,
                args.tao_max_sequences,
                args.tao_max_samples,
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
                args.lost_target_probability,
                args.max_lost_targets_per_frame,
                args.max_missing_gap_frames,
                args.search_anchor_mode,
                args.repair_search_to_target,
                args.search_target_padding_fraction,
                args.debug_sequence_name,
                args.debug_track_id,
                args.tao_use_freeform,
            )
        )
    train_dataset = CombinedFrameHeadDataset(train_datasets)
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found under {args.dataset_root} split={args.split}.")

    val_datasets: List[object] = []
    val_dataset_base = MOTFrameHeadDataset(
        args.dataset_root,
        args.val_split,
        class_ids,
        args.min_visibility,
        args.max_objects,
        args.val_frame_stride,
        args.max_val_sequences,
        args.max_val_samples,
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
        0.0,
        0,
        args.max_missing_gap_frames,
        args.search_anchor_mode,
        args.repair_search_to_target,
        args.search_target_padding_fraction,
        args.debug_sequence_name,
        args.debug_track_id,
    )
    val_datasets.append(val_dataset_base)
    if args.tao_root is not None:
        val_datasets.append(
            TAOFrameHeadDataset(
                args.tao_root,
                args.tao_val_split,
                class_ids,
                args.min_visibility,
                args.max_objects,
                args.tao_val_frame_stride,
                args.tao_max_val_sequences,
                args.tao_max_val_samples,
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
                0.0,
                0,
                args.max_missing_gap_frames,
                args.search_anchor_mode,
                args.repair_search_to_target,
                args.search_target_padding_fraction,
                args.debug_sequence_name,
                args.debug_track_id,
                args.tao_use_freeform,
            )
        )
    val_dataset = CombinedFrameHeadDataset(val_datasets)

    print(
        f"Training samples: total={len(train_dataset)} sources={getattr(train_dataset, 'source_counts', {})}",
        flush=True,
    )
    print(
        f"Validation samples: total={len(val_dataset)} sources={getattr(val_dataset, 'source_counts', {})}",
        flush=True,
    )
    print(
        f"Training schedule: epochs={args.epochs} max_train_samples_per_epoch={args.max_train_samples_per_epoch} "
        f"eval_interval_epochs={args.eval_interval_epochs} diagnostic_interval_epochs={args.diagnostic_interval_epochs} "
        f"max_wall_hours={args.max_wall_hours}",
        flush=True,
    )
    effective_samples_per_epoch = len(train_dataset)
    if int(args.max_train_samples_per_epoch) > 0:
        effective_samples_per_epoch = min(len(train_dataset), int(args.max_train_samples_per_epoch))
    effective_epoch_fraction = (
        float(effective_samples_per_epoch) / float(max(1, len(train_dataset)))
    )
    print(
        f"Effective sampled epoch: samples={effective_samples_per_epoch}/{len(train_dataset)} "
        f"fraction={effective_epoch_fraction:.4f}",
        flush=True,
    )

    weight_path = args.weight_path or mot.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    trainer = v8.V8QualityBatchedLoRATTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        max_tracks=args.max_objects,
        fps=None,
        sequence_length=None,
        sequence_name="v8-train",
        disable_amp=args.disable_amp,
        head_rank=args.v8_head_rank,
        head_hidden_dim=args.v8_head_hidden_dim,
        head_lora_rank=args.v8_head_lora_rank,
        head_weight_path=args.resume_head,
        search_radius_factor=args.v8_search_radius_factor,
        collect_slot_debug=False,
    )
    head = trainer.object_conditioned_head
    # The training run starts from freshly initialized head weights, so no file
    # has been "loaded" yet. Mark it usable here so validation/probe passes use
    # the learned module instead of the runtime zero-shot fallback.
    head.weights_loaded = True
    head.module.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]] = {}
    checkpoint_dir = args.checkpoint_dir or (args.output.parent / f"{args.output.stem}_checkpoints")
    diagnostic_csv = args.diagnostic_csv or (checkpoint_dir / f"{args.output.stem}_training_diagnostics.csv")
    geometry_diagnostic_csv = args.geometry_diagnostic_csv or (checkpoint_dir / f"{args.output.stem}_geometry_diagnostics.csv")
    iou_diagnostic_csv = args.iou_diagnostic_csv or (checkpoint_dir / f"{args.output.stem}_iou_failure_diagnostics.csv")
    debug_visual_dir = args.debug_visual_dir or (checkpoint_dir / f"{args.output.stem}_debug_visuals")
    steps = 0
    training_wall_start = time.perf_counter()
    max_wall_seconds = max(0.0, float(args.max_wall_hours)) * 3600.0
    best_selection_val_mean_iou: Optional[float] = None
    full_train_mean_iou: Optional[float] = None
    full_train_iou50: Optional[float] = None
    full_val_mean_iou: Optional[float] = None
    full_val_iou50: Optional[float] = None
    for epoch in range(1, max(1, args.epochs) + 1):
        epoch_wall_start = time.perf_counter()
        epoch_train_start = epoch_wall_start
        epoch_start_steps = steps
        epoch_samples_attempted = 0
        epoch_samples_trained = 0
        phase_settings = training_phase_settings(args, epoch)
        debug_visuals_saved = 0
        order = list(range(len(train_dataset)))
        random.shuffle(order)
        if int(args.max_train_samples_per_epoch) > 0:
            order = order[: min(len(order), int(args.max_train_samples_per_epoch))]
        running_loss = 0.0
        running_objectness = 0.0
        running_box = 0.0
        running_ltrb = 0.0
        running_reid = 0.0
        running_dcfst = 0.0
        running_assignment = 0.0
        running_positive_cells = 0
        running_loss_cells = 0
        running_hard_negative_cells = 0
        running_dcfst_objects = 0
        running_dcfst_positive_candidates = 0
        running_dcfst_negative_candidates = 0
        running_assignment_objects = 0
        running_assignment_positive_candidates = 0
        running_assignment_negative_candidates = 0
        running_closed_loop_objects = 0
        running_missing_targets = 0
        running_positive_outside_search = 0
        running_present_targets = 0
        running_center_outside_search = 0
        running_coverage_sum = 0.0
        running_min_coverage = 1.0
        running_area_ratio_sum = 0.0
        stop_for_wall_clock = False
        for sample_index in order:
            if max_wall_seconds > 0.0 and (time.perf_counter() - training_wall_start) >= max_wall_seconds:
                stop_for_wall_clock = True
                print(
                    f"wall_clock_stop_requested epoch={epoch} step={steps} "
                    f"elapsed_hours={(time.perf_counter() - training_wall_start) / 3600.0:.3f} "
                    f"max_wall_hours={float(args.max_wall_hours):.3f}",
                    flush=True,
                )
                break
            epoch_samples_attempted += 1
            sample = train_dataset[sample_index]
            current_index = exercise.frame_to_image_index(sample.frame_number)
            current_frame = try_load_frame(
                sample.image_paths[current_index],
                f"train sequence={sample.sequence_path.name} frame={sample.frame_number}",
            )
            if current_frame is None:
                continue
            augmentation_rng = deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"epoch-{epoch}")
            augmentation_spec = make_lorat_augmentation_spec(augmentation_rng, bool(args.lorat_augmentation))
            train_objects = apply_search_bbox_augmentation(sample.objects, int(current_frame.shape[1]), augmentation_spec)
            if augmentation_spec.enabled:
                current_frame = apply_lorat_image_augmentation(current_frame, augmentation_spec, "search")
            training_frame_mode = effective_training_frame_mode(args.training_frame_mode, epoch, args.crop_stage_epochs)
            batch_mode = training_frame_mode
            if training_frame_mode == "search_crop":
                crop_batches = crop_training_batches(current_frame, train_objects, "search_crop")
                if not crop_batches:
                    continue
                current_frame, train_objects, batch_mode = crop_batches[steps % len(crop_batches)]
            with torch.no_grad():
                frame_features = trainer.shared_frame_encoder.encode(current_frame).feature_map.detach()

            selected_banks = build_selected_banks(
                trainer,
                sample,
                train_objects,
                template_feature_cache,
                augmentation_spec,
                memory_slots=args.training_memory_slots if batch_mode == "full" else 1,
            )
            if selected_banks is None:
                continue
            head_output = head.score(frame_features, selected_banks)
            closed_loop_rng = deterministic_rng(sample.sequence_path, sample.frame_number, steps, f"closed-loop-{epoch}")
            train_objects, closed_loop_applied = apply_closed_loop_rollout(
                trainer,
                head_output,
                train_objects,
                current_frame.shape,
                phase_settings.closed_loop_probability,
                closed_loop_rng,
                args.search_area_factor,
                args.search_scale_jitter,
                args.search_translation_jitter,
                args.search_min_object_size,
                args.repair_search_to_target,
                args.search_target_padding_fraction,
            )
            decoded_boxes_xyxy = decode_box_maps_xyxy(
                torch,
                head_output.box_delta_maps,
                train_objects,
                current_frame.shape,
                trainer.device,
            )
            use_iou_aware = bool(args.iou_aware_classification and steps >= max(0, args.iou_aware_warmup_steps))
            target_scores, positive_mask, hard_negative_mask, loss_mask, positive_weights, target_boxes_xyxy, target_ltrb_offsets, target_stats = make_lorat_style_targets(
                torch,
                head_output.score_maps,
                decoded_boxes_xyxy,
                train_objects,
                current_frame.shape,
                trainer.search_radius_factor,
                use_iou_aware,
                args.center_positive_weight,
                args.small_target_loss_weight,
                args.small_target_area_threshold,
                args.small_target_max_side,
                trainer.device,
            )
            geometry_rows = []
            for diag_index, diag_item in enumerate(train_objects):
                if not diag_item.is_present:
                    continue
                search_bbox = expanded_training_search_bbox(diag_item, trainer.search_radius_factor)
                geometry_rows.append(
                    {
                        "epoch": epoch,
                        "step": steps,
                        "sample_index": sample_index,
                        "sequence": sample.sequence_path.name,
                        "frame": int(sample.frame_number),
                        "object_index": diag_index,
                        "track_id": int(diag_item.track_id),
                        "target_kind": diag_item.target_kind,
                        "small_target": int(is_small_target_object(diag_item, args.small_target_area_threshold, args.small_target_max_side)),
                        "small_target_loss_weight": args.small_target_loss_weight,
                        "current_bbox": diagnostic_bbox_text(diag_item.current_bbox),
                        "previous_bbox": diagnostic_bbox_text(diag_item.previous_bbox),
                        "template_bbox": diagnostic_bbox_text(diag_item.template_bbox),
                        "search_bbox": diagnostic_bbox_text(search_bbox),
                        "previous_current_center_px": bbox_center_distance_xywh(diag_item.previous_bbox, diag_item.current_bbox),
                        "previous_current_center_norm": bbox_motion_ratio_xywh(diag_item.previous_bbox, diag_item.current_bbox),
                        "target_search_coverage": bbox_coverage_xywh(diag_item.current_bbox, search_bbox),
                        "target_center_in_search": int(bbox_contains_center_xywh(search_bbox, diag_item.current_bbox)),
                        "search_target_area_ratio": bbox_area_xywh(search_bbox) / max(1.0, bbox_area_xywh(diag_item.current_bbox)),
                        "positive_cells": target_stats.positive_cells,
                        "loss_cells": target_stats.loss_cells,
                        "positive_cells_outside_search": target_stats.positive_cells_outside_search,
                        "missing_targets": target_stats.missing_targets,
                        "training_frame_mode": batch_mode,
                        "training_phase": phase_settings.name,
                        "closed_loop_applied": int(closed_loop_applied > 0),
                        "distractor_bboxes": len(diag_item.distractor_bboxes),
                        "search_anchor_mode": args.search_anchor_mode,
                        "repair_search_to_target": int(bool(args.repair_search_to_target)),
                    }
                )
            if geometry_rows and (steps < max(100, int(args.debug_visual_samples)) or target_stats.positive_cells_outside_search > 0):
                append_diagnostic_rows(geometry_diagnostic_csv, geometry_rows)
            objectness_logits = head_output.score_maps.to(torch.float32)
            objectness_map = F.binary_cross_entropy_with_logits(
                objectness_logits,
                target_scores,
                reduction="none",
            )
            focal_gamma = max(0.0, float(args.focal_loss_gamma))
            if focal_gamma > 0.0:
                with torch.no_grad():
                    objectness_probability = torch.sigmoid(objectness_logits)
                    focal_weight = torch.abs(target_scores - objectness_probability).clamp_min(1e-4).pow(focal_gamma)
                objectness_map = objectness_map * focal_weight
            if args.negative_loss_weight != 1.0:
                objectness_weights = torch.ones_like(objectness_map)
                objectness_weights[loss_mask & ~positive_mask] = max(0.0, float(args.negative_loss_weight))
                objectness_map = objectness_map * objectness_weights
            if phase_settings.hard_negative_loss_weight != 1.0 and hard_negative_mask.any():
                hard_negative_weights = torch.ones_like(objectness_map)
                hard_negative_weights[hard_negative_mask] = phase_settings.hard_negative_loss_weight
                objectness_map = objectness_map * hard_negative_weights
            if positive_mask.any():
                positive_objectness_weights = torch.ones_like(objectness_map)
                positive_objectness_weights[positive_mask] = positive_weights[positive_mask].clamp_min(1.0)
                objectness_map = objectness_map * positive_objectness_weights
            positive_count = positive_mask.sum().clamp_min(1).to(torch.float32)
            objectness_loss = objectness_map[loss_mask].sum() / positive_count

            if positive_mask.any():
                positive_box_weights = positive_weights[positive_mask].clamp_min(1e-4)
                positive_weight_sum = positive_box_weights.sum().clamp_min(1.0)
                giou = generalized_iou_aligned(
                    torch,
                    decoded_boxes_xyxy[positive_mask],
                    target_boxes_xyxy[positive_mask],
                )
                box_loss = ((1.0 - giou) * positive_box_weights).sum() / positive_weight_sum
                predicted_ltrb_offsets = torch.sigmoid(head_output.box_delta_maps.to(torch.float32))
                ltrb_loss_map = F.smooth_l1_loss(predicted_ltrb_offsets, target_ltrb_offsets, reduction="none").sum(dim=-1)
                ltrb_loss = (ltrb_loss_map[positive_mask] * positive_box_weights).sum() / positive_weight_sum
            else:
                box_loss = head_output.box_delta_maps.sum() * 0.0
                ltrb_loss = head_output.box_delta_maps.sum() * 0.0
            if phase_settings.reid_loss_weight > 0.0:
                reid_loss = contrastive_reid_loss(
                    torch,
                    trainer,
                    head,
                    selected_banks,
                    frame_features,
                    train_objects,
                    current_frame.shape,
                )
            else:
                reid_loss = head_output.score_maps.sum() * 0.0
            if phase_settings.dcfst_discrimination_weight > 0.0:
                dcfst_loss, dcfst_stats = dcfst_candidate_discrimination_loss(
                    torch,
                    trainer,
                    head_output,
                    train_objects,
                    current_frame.shape,
                    args.dcfst_negative_candidates,
                    args.dcfst_margin,
                )
            else:
                dcfst_loss = head_output.score_maps.sum() * 0.0
                dcfst_stats = V8CandidateDiscriminationStats()
            if phase_settings.assignment_discrimination_weight > 0.0:
                assignment_loss, assignment_stats = object_assignment_discrimination_loss(
                    torch,
                    trainer,
                    head_output,
                    train_objects,
                    current_frame.shape,
                    args.assignment_margin,
                )
            else:
                assignment_loss = head_output.score_maps.sum() * 0.0
                assignment_stats = V8AssignmentDiscriminationStats()
            loss = (
                objectness_loss
                + (args.box_loss_weight * box_loss)
                + (args.ltrb_loss_weight * ltrb_loss)
                + (phase_settings.reid_loss_weight * reid_loss)
                + (phase_settings.dcfst_discrimination_weight * dcfst_loss)
                + (phase_settings.assignment_discrimination_weight * assignment_loss)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(head.parameters()), float(args.clip_max_norm))
            optimizer.step()

            running_loss += float(loss.detach().item())
            running_objectness += float(objectness_loss.detach().item())
            running_box += float(box_loss.detach().item())
            running_ltrb += float(ltrb_loss.detach().item())
            running_reid += float(reid_loss.detach().item())
            running_dcfst += float(dcfst_loss.detach().item())
            running_assignment += float(assignment_loss.detach().item())
            running_positive_cells += target_stats.positive_cells
            running_loss_cells += target_stats.loss_cells
            running_hard_negative_cells += target_stats.hard_negative_cells
            running_dcfst_objects += int(dcfst_stats.objects)
            running_dcfst_positive_candidates += int(dcfst_stats.positive_candidates)
            running_dcfst_negative_candidates += int(dcfst_stats.negative_candidates)
            running_assignment_objects += int(assignment_stats.objects)
            running_assignment_positive_candidates += int(assignment_stats.positive_candidates)
            running_assignment_negative_candidates += int(assignment_stats.negative_candidates)
            running_closed_loop_objects += int(closed_loop_applied)
            running_missing_targets += target_stats.missing_targets
            running_positive_outside_search += target_stats.positive_cells_outside_search
            running_present_targets += target_stats.present_targets
            running_center_outside_search += target_stats.target_center_outside_search
            running_coverage_sum += target_stats.mean_target_search_coverage * max(1, target_stats.present_targets)
            running_min_coverage = min(running_min_coverage, target_stats.min_target_search_coverage)
            running_area_ratio_sum += target_stats.mean_search_target_area_ratio * max(1, target_stats.present_targets)
            steps += 1
            epoch_samples_trained += 1
            if (args.debug_visual_every_epoch or epoch == 1) and debug_visuals_saved < max(0, int(args.debug_visual_samples)):
                rows = prediction_diagnostic_rows(trainer, sample, head_output, train_objects, current_frame.shape, epoch, steps)
                if rows:
                    for row in rows:
                        row["phase"] = "train_step"
                        row["sample_index"] = sample_index
                        row["training_frame_mode"] = batch_mode
                    append_diagnostic_rows(iou_diagnostic_csv, rows)
                    debug_visuals_saved += save_prediction_debug_visuals(
                        trainer,
                        debug_visual_dir,
                        sample,
                        current_frame,
                        head_output,
                        train_objects,
                        rows,
                        epoch,
                        steps,
                        max(0, int(args.debug_visual_samples) - debug_visuals_saved),
                    )
            if steps % 25 == 0:
                present_denominator = max(1, running_present_targets)
                print(
                    f"epoch={epoch} step={steps} loss={running_loss / 25.0:.4f} "
                    f"objectness={running_objectness / 25.0:.4f} giou_box={running_box / 25.0:.4f} "
                    f"ltrb={running_ltrb / 25.0:.4f} reid={running_reid / 25.0:.4f} "
                    f"dcfst={running_dcfst / 25.0:.4f} assignment={running_assignment / 25.0:.4f} "
                    f"phase={phase_settings.name} mode={batch_mode} "
                    f"pos_cells={running_positive_cells / 25.0:.1f} loss_cells={running_loss_cells / 25.0:.1f} "
                    f"hard_neg_cells={running_hard_negative_cells / 25.0:.1f} "
                    f"dcfst_objs={running_dcfst_objects / 25.0:.1f} "
                    f"dcfst_pos={running_dcfst_positive_candidates / 25.0:.1f} "
                    f"dcfst_neg={running_dcfst_negative_candidates / 25.0:.1f} "
                    f"assign_objs={running_assignment_objects / 25.0:.1f} "
                    f"assign_pos={running_assignment_positive_candidates / 25.0:.1f} "
                    f"assign_neg={running_assignment_negative_candidates / 25.0:.1f} "
                    f"closed_loop_objs={running_closed_loop_objects / 25.0:.1f} "
                    f"memory_slots={args.training_memory_slots if batch_mode == 'full' else 1} "
                    f"missing_targets={running_missing_targets / 25.0:.1f} "
                    f"pos_outside_search={running_positive_outside_search / 25.0:.1f} "
                    f"target_center_outside={running_center_outside_search / 25.0:.1f} "
                    f"target_search_coverage_mean={running_coverage_sum / float(present_denominator):.3f} "
                    f"target_search_coverage_min={running_min_coverage:.3f} "
                    f"search_target_area_ratio_mean={running_area_ratio_sum / float(present_denominator):.1f} "
                    f"iou_aware={'yes' if use_iou_aware else 'warmup'}",
                    flush=True,
                )
                running_loss = 0.0
                running_objectness = 0.0
                running_box = 0.0
                running_ltrb = 0.0
                running_reid = 0.0
                running_dcfst = 0.0
                running_assignment = 0.0
                running_positive_cells = 0
                running_loss_cells = 0
                running_hard_negative_cells = 0
                running_dcfst_objects = 0
                running_dcfst_positive_candidates = 0
                running_dcfst_negative_candidates = 0
                running_assignment_objects = 0
                running_assignment_positive_candidates = 0
                running_assignment_negative_candidates = 0
                running_closed_loop_objects = 0
                running_missing_targets = 0
                running_positive_outside_search = 0
                running_present_targets = 0
                running_center_outside_search = 0
                running_coverage_sum = 0.0
                running_min_coverage = 1.0
                running_area_ratio_sum = 0.0
            if args.checkpoint_interval > 0 and steps % args.checkpoint_interval == 0:
                step_path = checkpoint_dir / f"{args.output.stem}_epoch{epoch:03d}_step{steps:07d}.pt"
                latest_path = checkpoint_dir / f"{args.output.stem}_latest.pt"
                save_checkpoint(torch, step_path, head, args, epoch, steps)
                save_checkpoint(torch, latest_path, head, args, epoch, steps)
                print(f"Saved step checkpoint to {step_path}", flush=True)
            if args.max_steps > 0 and steps >= args.max_steps:
                break

        epoch_train_wall_seconds = time.perf_counter() - epoch_train_start
        epoch_train_steps = steps - epoch_start_steps
        epoch_seconds_per_step = epoch_train_wall_seconds / float(max(1, epoch_train_steps))
        print(
            f"epoch_timing phase=train epoch={epoch} "
            f"train_seconds={epoch_train_wall_seconds:.2f} "
            f"train_steps={epoch_train_steps} "
            f"seconds_per_step={epoch_seconds_per_step:.4f} "
            f"samples_attempted={epoch_samples_attempted} "
            f"samples_trained={epoch_samples_trained} "
            f"total_steps={steps}",
            flush=True,
        )
        epoch_eval_start = time.perf_counter()
        eval_interval_epochs = max(1, int(args.eval_interval_epochs))
        diagnostic_interval_epochs = int(args.diagnostic_interval_epochs)
        reached_step_limit = bool(args.max_steps > 0 and steps >= args.max_steps)
        is_last_requested_epoch = epoch >= max(1, int(args.epochs))
        should_eval = (
            epoch == 1
            or stop_for_wall_clock
            or reached_step_limit
            or is_last_requested_epoch
            or (epoch % eval_interval_epochs == 0)
        )
        should_run_heavy_diagnostics = (
            should_eval
            and diagnostic_interval_epochs > 0
            and (
                epoch == 1
                or stop_for_wall_clock
                or reached_step_limit
                or is_last_requested_epoch
                or (epoch % diagnostic_interval_epochs == 0)
            )
        )
        eval_frame_mode = effective_training_frame_mode(args.training_frame_mode, epoch, args.crop_stage_epochs)
        train_mean_iou: Optional[float] = None
        train_iou50: Optional[float] = None
        val_mean_iou: Optional[float] = None
        val_iou50: Optional[float] = None
        selection_metric_frame_mode = "skipped"
        selection_val_mean_iou: Optional[float] = None
        selection_val_iou50: Optional[float] = None
        train_reid_same: Optional[float] = None
        train_reid_diff: Optional[float] = None
        train_reid_top1: Optional[float] = None
        val_reid_same: Optional[float] = None
        val_reid_diff: Optional[float] = None
        val_reid_top1: Optional[float] = None
        if should_eval:
            train_mean_iou, train_iou50 = validate_head(
                trainer,
                train_dataset,
                args.train_diagnostic_samples,
                template_feature_cache,
                eval_frame_mode,
                args.training_memory_slots,
            )
            val_mean_iou, val_iou50 = validate_head(
                trainer,
                val_dataset,
                args.max_val_samples,
                template_feature_cache,
                eval_frame_mode,
                args.training_memory_slots,
            )
            full_train_mean_iou = train_mean_iou
            full_train_iou50 = train_iou50
            full_val_mean_iou = val_mean_iou
            full_val_iou50 = val_iou50
            if eval_frame_mode != "full":
                full_train_mean_iou, full_train_iou50 = validate_head(
                    trainer,
                    train_dataset,
                    args.train_diagnostic_samples,
                    template_feature_cache,
                    "full",
                    args.training_memory_slots,
                )
                full_val_mean_iou, full_val_iou50 = validate_head(
                    trainer,
                    val_dataset,
                    args.max_val_samples,
                    template_feature_cache,
                    "full",
                    args.training_memory_slots,
                )
            selection_metric_frame_mode = "full" if full_val_mean_iou is not None else eval_frame_mode
            selection_val_mean_iou = full_val_mean_iou if full_val_mean_iou is not None else val_mean_iou
            selection_val_iou50 = full_val_iou50 if full_val_iou50 is not None else val_iou50
            if should_run_heavy_diagnostics:
                train_reid_same, train_reid_diff, train_reid_top1 = reid_similarity_probe(
                    trainer,
                    train_dataset,
                    args.train_diagnostic_samples,
                    template_feature_cache,
                    "full",
                    args.training_memory_slots,
                )
                val_reid_same, val_reid_diff, val_reid_top1 = reid_similarity_probe(
                    trainer,
                    val_dataset,
                    args.max_val_samples,
                    template_feature_cache,
                    "full",
                    args.training_memory_slots,
                )
                debug_visuals_saved += collect_iou_failure_diagnostics(
                    trainer,
                    train_dataset,
                    min(max(0, int(args.train_diagnostic_samples)), 32),
                    template_feature_cache,
                    epoch,
                    steps,
                    "train_epoch_probe",
                    iou_diagnostic_csv,
                    debug_visual_dir if args.debug_visual_every_epoch else None,
                    max(0, int(args.debug_visual_samples) - debug_visuals_saved) if args.debug_visual_every_epoch else 0,
                    eval_frame_mode,
                    args.training_memory_slots,
                )
                debug_visuals_saved += collect_iou_failure_diagnostics(
                    trainer,
                    val_dataset,
                    min(max(0, int(args.max_val_samples)), 32),
                    template_feature_cache,
                    epoch,
                    steps,
                    "val_epoch_probe",
                    iou_diagnostic_csv,
                    debug_visual_dir if args.debug_visual_every_epoch else None,
                    max(0, int(args.debug_visual_samples) - debug_visuals_saved) if args.debug_visual_every_epoch else 0,
                    eval_frame_mode,
                    args.training_memory_slots,
                )
        epoch_eval_wall_seconds = time.perf_counter() - epoch_eval_start
        save_checkpoint(
            torch,
            args.output,
            head,
            args,
            epoch,
            steps,
            val_mean_iou,
            val_iou50,
            full_val_mean_iou if should_eval else None,
            full_val_iou50 if should_eval else None,
            selection_val_mean_iou,
            selection_val_iou50,
            selection_metric_frame_mode,
        )
        save_checkpoint(
            torch,
            checkpoint_dir / f"{args.output.stem}_latest.pt",
            head,
            args,
            epoch,
            steps,
            val_mean_iou,
            val_iou50,
            full_val_mean_iou if should_eval else None,
            full_val_iou50 if should_eval else None,
            selection_val_mean_iou,
            selection_val_iou50,
            selection_metric_frame_mode,
        )
        best_updated = False
        if selection_val_mean_iou is not None and (
            best_selection_val_mean_iou is None or selection_val_mean_iou > best_selection_val_mean_iou
        ):
            best_selection_val_mean_iou = float(selection_val_mean_iou)
            save_checkpoint(
                torch,
                checkpoint_dir / f"{args.output.stem}_best_by_val_iou.pt",
                head,
                args,
                epoch,
                steps,
                val_mean_iou,
                val_iou50,
                full_val_mean_iou,
                full_val_iou50,
                selection_val_mean_iou,
                selection_val_iou50,
                selection_metric_frame_mode,
            )
            best_updated = True
        val_text = "validation skipped"
        if val_mean_iou is not None and val_iou50 is not None:
            val_text = f"val_mean_iou={val_mean_iou:.4f} val_iou50={val_iou50:.4f}"
            if eval_frame_mode != "full" and full_val_mean_iou is not None and full_val_iou50 is not None:
                val_text += f" full_val_mean_iou={full_val_mean_iou:.4f} full_val_iou50={full_val_iou50:.4f}"
        train_text = "train probe skipped"
        if train_mean_iou is not None and train_iou50 is not None:
            train_text = f"train_probe_mean_iou={train_mean_iou:.4f} train_probe_iou50={train_iou50:.4f}"
            if eval_frame_mode != "full" and full_train_mean_iou is not None and full_train_iou50 is not None:
                train_text += f" full_train_probe_mean_iou={full_train_mean_iou:.4f} full_train_probe_iou50={full_train_iou50:.4f}"
        reid_text = "reid probe skipped"
        if val_reid_top1 is not None:
            reid_text = f"val_reid_top1={val_reid_top1:.4f}"
        epoch_total_wall_seconds = time.perf_counter() - epoch_wall_start
        append_diagnostic_row(
            diagnostic_csv,
            {
                "epoch": epoch,
                "steps": steps,
                "epoch_train_steps": epoch_train_steps,
                "epoch_samples_attempted": epoch_samples_attempted,
                "epoch_samples_trained": epoch_samples_trained,
                "effective_epoch_samples": effective_samples_per_epoch,
                "effective_epoch_fraction": effective_epoch_fraction,
                "epoch_train_wall_seconds": epoch_train_wall_seconds,
                "epoch_eval_wall_seconds": epoch_eval_wall_seconds,
                "epoch_total_wall_seconds": epoch_total_wall_seconds,
                "epoch_seconds_per_train_step": epoch_seconds_per_step,
                "training_elapsed_hours": (time.perf_counter() - training_wall_start) / 3600.0,
                "max_wall_hours": float(args.max_wall_hours),
                "stopped_for_wall_clock": int(stop_for_wall_clock),
                "max_train_samples_per_epoch": int(args.max_train_samples_per_epoch),
                "eval_interval_epochs": int(args.eval_interval_epochs),
                "diagnostic_interval_epochs": int(args.diagnostic_interval_epochs),
                "eval_ran": int(should_eval),
                "heavy_diagnostics_ran": int(should_run_heavy_diagnostics),
                "template_feature_cache_entries": len(template_feature_cache),
                "lorat_config": args.lorat_config,
                "head_architecture": "template_patch_lora_conditioned",
                "box_parameterization": "lorat_anchor_free_ltrb",
                "loss_style": "lorat_iou_aware_focal_bce_giou_ltrb_staged_reid_dcfst_assignment_closed_loop",
                "training_phase": phase_settings.name,
                "training_memory_slots": int(args.training_memory_slots),
                "geometry_only_epochs": int(args.geometry_only_epochs),
                "hard_negative_start_epoch": int(args.hard_negative_start_epoch),
                "reid_start_epoch": int(args.reid_start_epoch),
                "dcfst_start_epoch": int(args.dcfst_start_epoch),
                "assignment_start_epoch": int(args.assignment_start_epoch),
                "closed_loop_start_epoch": int(args.closed_loop_start_epoch),
                "phase_hard_negative_loss_weight": float(phase_settings.hard_negative_loss_weight),
                "phase_reid_loss_weight": float(phase_settings.reid_loss_weight),
                "phase_dcfst_discrimination_weight": float(phase_settings.dcfst_discrimination_weight),
                "phase_assignment_discrimination_weight": float(phase_settings.assignment_discrimination_weight),
                "phase_closed_loop_probability": float(phase_settings.closed_loop_probability),
                "iou_aware_classification": int(bool(args.iou_aware_classification)),
                "iou_aware_warmup_steps": int(args.iou_aware_warmup_steps),
                "box_loss_weight": float(args.box_loss_weight),
                "ltrb_loss_weight": float(args.ltrb_loss_weight),
                "reid_loss_weight": float(args.reid_loss_weight),
                "center_positive_weight": float(args.center_positive_weight),
                "negative_loss_weight": float(args.negative_loss_weight),
                "hard_negative_loss_weight": float(args.hard_negative_loss_weight),
                "focal_loss_gamma": float(args.focal_loss_gamma),
                "dcfst_discrimination_weight": float(args.dcfst_discrimination_weight),
                "dcfst_negative_candidates": int(args.dcfst_negative_candidates),
                "dcfst_margin": float(args.dcfst_margin),
                "assignment_discrimination_weight": float(args.assignment_discrimination_weight),
                "assignment_margin": float(args.assignment_margin),
                "closed_loop_probability": float(args.closed_loop_probability),
                "search_radius_factor": float(trainer.search_radius_factor),
                "target_region_mode": args.target_region_mode,
                "target_regions_per_object": int(args.target_regions_per_object),
                "template_area_factor": float(args.template_area_factor),
                "search_area_factor": float(args.search_area_factor),
                "search_scale_jitter": float(args.search_scale_jitter),
                "search_translation_jitter": float(args.search_translation_jitter),
                "search_min_object_size": float(args.search_min_object_size),
                "search_anchor_mode": args.search_anchor_mode,
                "repair_search_to_target": int(bool(args.repair_search_to_target)),
                "search_target_padding_fraction": float(args.search_target_padding_fraction),
                "template_sampling": args.template_sampling,
                "previous_box_jitter": float(args.previous_box_jitter),
                "sequence_window_length": int(args.sequence_window_length),
                "lost_target_probability": float(args.lost_target_probability),
                "max_lost_targets_per_frame": int(args.max_lost_targets_per_frame),
                "max_missing_gap_frames": int(args.max_missing_gap_frames),
                "training_frame_mode": args.training_frame_mode,
                "effective_training_frame_mode": eval_frame_mode,
                "checkpoint_selection_frame_mode": selection_metric_frame_mode,
                "crop_stage_epochs": int(args.crop_stage_epochs),
                "debug_sequence_name": args.debug_sequence_name or "",
                "debug_track_id": "" if args.debug_track_id is None else int(args.debug_track_id),
                "overfit_single_object": int(bool(args.overfit_single_object)),
                "lorat_augmentation": int(bool(args.lorat_augmentation)),
                "train_probe_mean_iou": train_mean_iou,
                "train_probe_iou50": train_iou50,
                "val_mean_iou": val_mean_iou,
                "val_iou50": val_iou50,
                "full_train_probe_mean_iou": full_train_mean_iou if should_eval else None,
                "full_train_probe_iou50": full_train_iou50 if should_eval else None,
                "full_val_mean_iou": full_val_mean_iou if should_eval else None,
                "full_val_iou50": full_val_iou50 if should_eval else None,
                "selection_val_mean_iou": selection_val_mean_iou,
                "selection_val_iou50": selection_val_iou50,
                "reid_probe_frame_mode": "full",
                "train_reid_same_cosine": train_reid_same,
                "train_reid_diff_cosine": train_reid_diff,
                "train_reid_top1": train_reid_top1,
                "val_reid_same_cosine": val_reid_same,
                "val_reid_diff_cosine": val_reid_diff,
                "val_reid_top1": val_reid_top1,
                "best_selection_val_mean_iou": best_selection_val_mean_iou,
                "best_updated": int(best_updated),
            },
        )
        best_text = "best updated" if best_updated else f"best_selection_val_mean_iou={best_selection_val_mean_iou:.4f}" if best_selection_val_mean_iou is not None else "best pending"
        print(
            f"epoch_timing phase=total epoch={epoch} "
            f"total_seconds={epoch_total_wall_seconds:.2f} "
            f"train_seconds={epoch_train_wall_seconds:.2f} "
            f"eval_seconds={epoch_eval_wall_seconds:.2f} "
            f"train_steps={epoch_train_steps} "
            f"seconds_per_step={epoch_seconds_per_step:.4f} "
            f"eval_ran={int(should_eval)} "
            f"heavy_diagnostics_ran={int(should_run_heavy_diagnostics)}",
            flush=True,
        )
        print(f"Saved V8 head checkpoint to {args.output} ({train_text}; {val_text}; {reid_text}; {best_text})", flush=True)
        if stop_for_wall_clock:
            print(
                f"Stopping cleanly for max wall clock after epoch={epoch} step={steps}; checkpoint saved.",
                flush=True,
            )
            break
        if args.max_steps > 0 and steps >= args.max_steps:
            break

    trainer.close()
    gate_mean = float(args.quality_gate_mean_iou)
    gate_iou50 = float(args.quality_gate_iou50)
    if gate_mean > 0.0 or gate_iou50 > 0.0:
        probe_mean = full_train_mean_iou if args.overfit_smoke else full_val_mean_iou
        probe_iou50 = full_train_iou50 if args.overfit_smoke else full_val_iou50
        passed = (
            probe_mean is not None
            and probe_iou50 is not None
            and float(probe_mean) >= gate_mean
            and float(probe_iou50) >= gate_iou50
        )
        print(
            f"quality_gate={'PASS' if passed else 'FAIL'} "
            f"mean_iou={probe_mean} required_mean_iou={gate_mean} "
            f"iou50={probe_iou50} required_iou50={gate_iou50}",
            flush=True,
        )
        if not passed:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
