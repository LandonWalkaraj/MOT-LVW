from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import bounding_box_v7_lorat_frame_shared as v7
import exercise_lorat_mot as exercise

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class V7TrainingObject:
    track_id: int
    current_bbox: BBox
    template_frame: int
    template_bbox: BBox


@dataclass(frozen=True)
class V7TrainingSample:
    sequence_path: Path
    image_paths: Sequence[Path]
    frame_number: int
    objects: Sequence[V7TrainingObject]


class DanceTrackMOTFrameDataset:
    """Frame-level multi-object samples for the v7 shared-frame head."""

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
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.class_ids = set(class_ids or [1])
        self.min_visibility = max(0.0, float(min_visibility))
        self.max_objects = max(1, int(max_objects))
        self.frame_stride = max(1, int(frame_stride))
        self.samples: List[V7TrainingSample] = []
        self._build(max_sequences=max_sequences, max_samples=max_samples)

    def _usable_rows(self, rows: Sequence[exercise.GroundTruthRow]) -> List[exercise.GroundTruthRow]:
        selected = [
            row
            for row in rows
            if row.confidence != 0
            and row.visibility >= self.min_visibility
            and row.class_id in self.class_ids
            and row.bbox[2] > 1
            and row.bbox[3] > 1
        ]
        return sorted(selected, key=lambda row: row.bbox[2] * row.bbox[3], reverse=True)

    def _build(self, max_sequences: int, max_samples: int) -> None:
        sequences = exercise.find_sequences(self.dataset_root, self.split)
        if max_sequences > 0:
            sequences = sequences[:max_sequences]
        for sequence_path in sequences:
            image_paths = exercise.get_image_paths(sequence_path)
            gt_by_frame = exercise.read_gt(sequence_path)
            if not image_paths or not gt_by_frame:
                continue

            first_row_by_track: Dict[int, exercise.GroundTruthRow] = {}
            for frame in sorted(gt_by_frame):
                for row in self._usable_rows(gt_by_frame[frame]):
                    first_row_by_track.setdefault(row.track_id, row)

            for frame_number in sorted(gt_by_frame):
                if frame_number % self.frame_stride != 0:
                    continue
                image_index = exercise.frame_to_image_index(frame_number)
                if image_index >= len(image_paths):
                    continue
                rows = self._usable_rows(gt_by_frame[frame_number])[: self.max_objects]
                objects: List[V7TrainingObject] = []
                for row in rows:
                    template_row = first_row_by_track.get(row.track_id, row)
                    template_index = exercise.frame_to_image_index(template_row.frame)
                    if template_index >= len(image_paths):
                        template_row = row
                    objects.append(
                        V7TrainingObject(
                            track_id=row.track_id,
                            current_bbox=row.bbox,
                            template_frame=template_row.frame,
                            template_bbox=template_row.bbox,
                        )
                    )
                if objects:
                    self.samples.append(
                        V7TrainingSample(
                            sequence_path=sequence_path,
                            image_paths=image_paths,
                            frame_number=frame_number,
                            objects=objects,
                        )
                    )
                if max_samples > 0 and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> V7TrainingSample:
        return self.samples[index]


def parse_class_ids(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    if not values:
        return None
    return list(dict.fromkeys(int(value) for value in values))


def load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Unable to read frame: {path}")
    return frame


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def make_targets(
    torch_module,
    score_maps,
    box_delta_maps,
    objects: Sequence[V7TrainingObject],
    frame_shape: Tuple[int, ...],
    box_delta_scale: float,
    device,
):
    frame_height, frame_width = frame_shape[:2]
    object_count, grid_height, grid_width = score_maps.shape
    target_scores = torch_module.zeros_like(score_maps)
    target_deltas = torch_module.zeros_like(box_delta_maps)
    delta_mask = torch_module.zeros((object_count, grid_height, grid_width), device=device, dtype=torch_module.bool)
    cell_width = float(frame_width) / float(grid_width)
    cell_height = float(frame_height) / float(grid_height)

    for index, item in enumerate(objects):
        current_x, current_y, current_w, current_h = item.current_bbox
        center_x, center_y = bbox_center(item.current_bbox)
        grid_x = int(np.clip(center_x / max(1.0, cell_width), 0, grid_width - 1))
        grid_y = int(np.clip(center_y / max(1.0, cell_height), 0, grid_height - 1))
        target_scores[index, grid_y, grid_x] = 1.0
        cell_center_x = (float(grid_x) + 0.5) * cell_width
        cell_center_y = (float(grid_y) + 0.5) * cell_height
        template_w = max(1.0, float(item.template_bbox[2]))
        template_h = max(1.0, float(item.template_bbox[3]))
        target_deltas[index, grid_y, grid_x, 0] = float(np.clip(2.0 * (center_x - cell_center_x) / max(1.0, cell_width), -1.0, 1.0))
        target_deltas[index, grid_y, grid_x, 1] = float(np.clip(2.0 * (center_y - cell_center_y) / max(1.0, cell_height), -1.0, 1.0))
        target_deltas[index, grid_y, grid_x, 2] = float(np.clip(np.log(max(1.0, current_w) / template_w), -1.0, 1.0))
        target_deltas[index, grid_y, grid_x, 3] = float(np.clip(np.log(max(1.0, current_h) / template_h), -1.0, 1.0))
        delta_mask[index, grid_y, grid_x] = True
    return target_scores, target_deltas, delta_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v7 shared-frame LoRAT object-conditioned head.")
    parser.add_argument("--dataset-root", type=Path, default=exercise.DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to train. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lorat-root", type=Path, default=v7.v5.DEFAULT_LORAT_ROOT)
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(v7.v5.LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT backbone weight. Defaults from --lorat-config.")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--v7-head-hidden-dim", type=int, default=256)
    parser.add_argument("--v7-head-lora-rank", type=int, default=16)
    parser.add_argument("--v7-head-rank", type=int, default=v7.v5.DEFAULT_LORAT_MEMORY_SLOTS)
    parser.add_argument("--resume-head", type=Path)
    parser.add_argument("--output", type=Path, default=v7.v5.PROJECT_ROOT / "models" / "lorat" / "v7_head.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    import torch.nn.functional as F

    dataset = DanceTrackMOTFrameDataset(
        args.dataset_root,
        args.split,
        parse_class_ids(args.class_id),
        args.min_visibility,
        args.max_objects,
        args.frame_stride,
        args.max_sequences,
        args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found under {args.dataset_root} split={args.split}.")

    weight_path = args.weight_path or v7.v5.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    trainer = v7.SharedFrameLoRATMultiObjectTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        max_tracks=args.max_objects,
        fps=None,
        sequence_length=None,
        sequence_name="v7-train",
        disable_amp=args.disable_amp,
        head_rank=args.v7_head_rank,
        head_hidden_dim=args.v7_head_hidden_dim,
        head_lora_rank=args.v7_head_lora_rank,
        head_weight_path=args.resume_head,
    )
    head = trainer.object_conditioned_head
    head.module.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.tensor(float(trainer.grid_width * trainer.grid_height), device=trainer.device)

    template_feature_cache: Dict[Tuple[Path, int], Tuple[object, Tuple[int, ...]]] = {}
    steps = 0
    for epoch in range(1, max(1, args.epochs) + 1):
        order = list(range(len(dataset)))
        random.shuffle(order)
        running_loss = 0.0
        for sample_index in order:
            sample = dataset[sample_index]
            current_index = exercise.frame_to_image_index(sample.frame_number)
            current_frame = load_frame(sample.image_paths[current_index])
            with torch.no_grad():
                frame_features = trainer.shared_frame_encoder.encode(current_frame).feature_map.detach()

            selected_banks = []
            for item in sample.objects:
                template_key = (sample.sequence_path, item.template_frame)
                cached_template = template_feature_cache.get(template_key)
                if cached_template is None:
                    template_index = exercise.frame_to_image_index(item.template_frame)
                    template_frame = load_frame(sample.image_paths[template_index])
                    with torch.no_grad():
                        template_features = trainer.shared_frame_encoder.encode(template_frame).feature_map.detach()
                    template_shape = template_frame.shape
                    template_feature_cache[template_key] = (template_features, template_shape)
                else:
                    template_features, template_shape = cached_template
                slot_vector = trainer._feature_mean_for_bbox(template_features, item.template_bbox, template_shape)
                selected_banks.append([
                    v7.V7TemplateMemorySlot(
                        vector=slot_vector,
                        label="initial",
                        frame_number=item.template_frame,
                        confidence=1.0,
                    )
                ])

            head_output = head.score(frame_features, selected_banks)
            target_scores, target_deltas, delta_mask = make_targets(
                torch,
                head_output.score_maps,
                head_output.box_delta_maps,
                sample.objects,
                current_frame.shape,
                head.box_delta_scale,
                trainer.device,
            )
            objectness_loss = F.binary_cross_entropy_with_logits(
                head_output.score_maps,
                target_scores,
                pos_weight=pos_weight,
            )
            if delta_mask.any():
                predicted_deltas = torch.tanh(head_output.box_delta_maps[delta_mask])
                box_loss = F.smooth_l1_loss(predicted_deltas, target_deltas[delta_mask])
            else:
                box_loss = head_output.box_delta_maps.sum() * 0.0
            loss = objectness_loss + (args.box_loss_weight * box_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().item())
            steps += 1
            if steps % 25 == 0:
                print(
                    f"epoch={epoch} step={steps} loss={running_loss / 25.0:.4f} "
                    f"objectness={float(objectness_loss.detach().item()):.4f} box={float(box_loss.detach().item()):.4f}",
                    flush=True,
                )
                running_loss = 0.0
            if args.max_steps > 0 and steps >= args.max_steps:
                break

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": head.state_dict(),
                "epoch": epoch,
                "steps": steps,
                "lorat_config": args.lorat_config,
                "head_hidden_dim": args.v7_head_hidden_dim,
                "head_lora_rank": args.v7_head_lora_rank,
            },
            str(args.output),
        )
        print(f"Saved v7 head checkpoint to {args.output}", flush=True)
        if args.max_steps > 0 and steps >= args.max_steps:
            break

    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
