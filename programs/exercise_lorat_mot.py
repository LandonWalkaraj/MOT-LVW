from __future__ import annotations

import argparse
import configparser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "DanceTrack"
LORAT_CONFIG_CHOICES = ("B-224", "B-378", "L-224", "L-378", "g-224", "g-378")


@dataclass(frozen=True)
class GroundTruthRow:
    frame: int
    track_id: int
    bbox: BBox
    confidence: float = 1.0
    class_id: int = 1
    visibility: float = 1.0


def normalized_compare_configs(configs: Sequence[str]) -> List[str]:
    if any(config == "all" for config in configs):
        return list(LORAT_CONFIG_CHOICES)
    result: List[str] = []
    for config in configs:
        if config not in LORAT_CONFIG_CHOICES:
            raise RuntimeError(f"Unsupported LoRAT config: {config}")
        if config not in result:
            result.append(config)
    return result


def extract_zips(dataset_root: Path) -> None:
    for archive in dataset_root.rglob("*.zip"):
        target_dir = archive.with_suffix("")
        if target_dir.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(target_dir)


def find_sequences(dataset_root: Path, split: str = "val") -> List[Path]:
    if not dataset_root.exists():
        return []
    split_roots = [
        candidate
        for candidate in (dataset_root / split, dataset_root / split / split)
        if candidate.is_dir()
    ]
    roots = split_roots or [dataset_root]
    sequences = {
        path.parent
        for root in roots
        for path in root.rglob("img1")
        if path.is_dir() and any(path.glob("*.jpg"))
    }
    return sorted(sequences, key=lambda path: str(path).lower())


def get_image_paths(sequence_path: Path) -> List[Path]:
    image_dir = sequence_path / "img1" if (sequence_path / "img1").is_dir() else sequence_path
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: List[Path] = []
    for pattern in extensions:
        paths.extend(image_dir.glob(pattern))
    return sorted(paths, key=lambda path: path.name)


def read_gt(sequence_path: Path) -> Dict[int, List[GroundTruthRow]]:
    gt_path = sequence_path / "gt" / "gt.txt"
    if not gt_path.exists():
        return {}
    rows_by_frame: Dict[int, List[GroundTruthRow]] = {}
    with gt_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, w, h = (float(fields[index]) for index in range(2, 6))
            confidence = float(fields[6]) if len(fields) > 6 and fields[6] else 1.0
            class_id = int(float(fields[7])) if len(fields) > 7 and fields[7] else 1
            visibility = float(fields[8]) if len(fields) > 8 and fields[8] else 1.0
            rows_by_frame.setdefault(frame, []).append(
                GroundTruthRow(
                    frame=frame,
                    track_id=track_id,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    class_id=class_id,
                    visibility=visibility,
                )
            )
    return rows_by_frame


def read_sequence_info(sequence_path: Path) -> Tuple[float, Optional[int]]:
    seqinfo_path = sequence_path / "seqinfo.ini"
    if not seqinfo_path.exists():
        return 30.0, len(get_image_paths(sequence_path)) or None
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    section = parser["Sequence"] if parser.has_section("Sequence") else {}
    fps = float(section.get("frameRate", 30.0))
    length = int(section["seqLength"]) if "seqLength" in section else len(get_image_paths(sequence_path)) or None
    return fps, length


def pick_initial_rows(
    gt_by_frame: Dict[int, List[GroundTruthRow]],
    init_frame: str,
    class_ids: Optional[Sequence[int]],
    min_visibility: float,
    target_tracks: int,
    min_init_tracks: int,
    selection_mode: str = "largest",
    min_area: float = 0.0,
    max_area: float = 0.0,
    track_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, List[GroundTruthRow]]:
    wanted_classes = set(class_ids or [1])
    wanted_track_ids = set(track_ids or [])
    min_area = max(0.0, float(min_area or 0.0))
    max_area = max(0.0, float(max_area or 0.0))
    selection_mode = (selection_mode or "largest").strip().lower()

    def usable(rows: Iterable[GroundTruthRow]) -> List[GroundTruthRow]:
        selected = [
            row
            for row in rows
            if row.confidence != 0
            and row.visibility >= min_visibility
            and row.class_id in wanted_classes
            and row.bbox[2] > 0
            and row.bbox[3] > 0
            and (not wanted_track_ids or row.track_id in wanted_track_ids)
            and (min_area <= 0 or (row.bbox[2] * row.bbox[3]) >= min_area)
            and (max_area <= 0 or (row.bbox[2] * row.bbox[3]) <= max_area)
        ]
        if wanted_track_ids:
            return sorted(
                selected,
                key=lambda row: (
                    list(track_ids or []).index(row.track_id) if row.track_id in wanted_track_ids else 10**9,
                    row.bbox[2] * row.bbox[3],
                ),
            )
        if selection_mode in {"smallest", "area-window", "area_window"}:
            return sorted(selected, key=lambda row: row.bbox[2] * row.bbox[3])
        if selection_mode == "middle":
            by_area = sorted(selected, key=lambda row: row.bbox[2] * row.bbox[3])
            midpoint = len(by_area) // 2
            return by_area[midpoint:] + by_area[:midpoint]
        return sorted(selected, key=lambda row: row.bbox[2] * row.bbox[3], reverse=True)

    if init_frame != "auto":
        frame = int(init_frame)
        rows = usable(gt_by_frame.get(frame, []))
        return frame, rows[:target_tracks]

    required = max(1, min_init_tracks)
    for frame in sorted(gt_by_frame):
        rows = usable(gt_by_frame[frame])
        if len(rows) >= required:
            return frame, rows[:target_tracks]
    first_frame = min(gt_by_frame, default=1)
    return first_frame, usable(gt_by_frame.get(first_frame, []))[:target_tracks]


def frame_to_image_index(frame_number: int) -> int:
    return max(0, int(frame_number) - 1)


def lorat_config_metadata(lorat_config: str) -> Tuple[str, int]:
    family, size_text = lorat_config.split("-", 1)
    backbone_by_family = {
        "B": "ViT-B/14",
        "L": "ViT-L/14",
        "g": "ViT-g/14",
    }
    return backbone_by_family.get(family, f"ViT-{family}/14"), int(size_text)


def bbox_iou(left: BBox, right: BBox) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(lx + lw, rx + rw)
    inter_y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = (lw * lh) + (rw * rh) - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def update_overlap_metrics(
    metrics: Dict[str, float],
    tracks,
    gt_by_frame: Dict[int, List[GroundTruthRow]],
    frame_number: int,
    tracker_to_gt_id: Dict[int, int],
    min_visibility: float,
) -> None:
    gt_rows = {
        row.track_id: row
        for row in gt_by_frame.get(frame_number, [])
        if row.confidence != 0 and row.visibility >= min_visibility
    }
    for track in tracks:
        gt_track_id = tracker_to_gt_id.get(track.track_id)
        gt_row = gt_rows.get(gt_track_id) if gt_track_id is not None else None
        if gt_row is None:
            continue
        iou = bbox_iou(track.bbox, gt_row.bbox)
        metrics["count"] = metrics.get("count", 0.0) + 1.0
        metrics["iou_sum"] = metrics.get("iou_sum", 0.0) + iou
        metrics["hit50"] = metrics.get("hit50", 0.0) + (1.0 if iou >= 0.5 else 0.0)


def build_backend(gui_v3, run_args: argparse.Namespace, sequence_name: str, fps: float, length: Optional[int]):
    source = SimpleNamespace(name=sequence_name, fps=fps, length=length)
    return gui_v3.create_backend(run_args, source)


def main() -> int:
    parser = argparse.ArgumentParser(description="List available DanceTrack/MOT-style sequences.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-sequences", action="store_true")
    args = parser.parse_args()
    for sequence in find_sequences(args.dataset_root, args.split):
        print(sequence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
