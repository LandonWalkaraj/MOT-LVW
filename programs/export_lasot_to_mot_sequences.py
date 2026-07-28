"""Export a few LaSOT videos into MOTChallenge-style benchmark folders.

LaSOT is single-object tracking data, but the V8/V9 benchmark harness reads
MOT-style `img1/` and `gt/gt.txt` folders. This adapter keeps LaSOT examples in
the same evaluator path as DanceTrack/MOT and TAO exports.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

import mot_common as mot

BBox = Tuple[float, float, float, float]


def slugify_sequence_name(name: str) -> str:
    result = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name.strip())
    result = result.strip("_")
    return result or "lasot_sequence"


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def read_bbox_rows(path: Path) -> List[BBox]:
    rows: List[BBox] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = [field.strip() for field in line.strip().replace("\t", ",").split(",") if field.strip()]
            if len(fields) < 4:
                continue
            try:
                x, y, width, height = (float(fields[index]) for index in range(4))
            except ValueError:
                continue
            rows.append((x, y, max(1.0, width), max(1.0, height)))
    return rows


def read_binary_flags(path: Path, expected_length: int) -> List[int]:
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


def image_paths_for_sequence(sequence_dir: Path) -> List[Path]:
    image_dir = sequence_dir / "img"
    paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(image_dir.glob(pattern))
    return sorted(paths, key=lambda path: path.name)


def sequence_dirs(dataset_root: Path, split: str, val_fraction: float) -> List[Path]:
    sequences = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
        and (path / "img").is_dir()
        and (path / "groundtruth.txt").exists()
        and not path.name.startswith("_")
    ]
    sequences = sorted(sequences, key=lambda path: path.name.lower())
    if len(sequences) > 1:
        val_count = max(1, int(round(len(sequences) * max(0.0, min(0.8, float(val_fraction))))))
        split_key = str(split or "val").strip().lower()
        if split_key in {"val", "valid", "validation"}:
            sequences = sequences[-val_count:]
        elif split_key in {"train", "training"}:
            sequences = sequences[:-val_count] or sequences
    return sequences


def export_sequences(
    lasot_root: Path,
    output_root: Path,
    split: str,
    max_sequences: int,
    min_visible_frames: int,
    max_frames: int,
    copy_mode: str,
    val_fraction: float,
) -> List[str]:
    output_split = "val" if split in {"val", "valid", "validation"} else split
    target_root = output_root / output_split
    target_root.mkdir(parents=True, exist_ok=True)
    exported: List[str] = []
    for source_sequence in sequence_dirs(lasot_root, split, val_fraction):
        image_paths = image_paths_for_sequence(source_sequence)
        boxes = read_bbox_rows(source_sequence / "groundtruth.txt")
        if not image_paths or not boxes:
            continue
        usable_length = min(len(image_paths), len(boxes))
        full_occlusion = read_binary_flags(source_sequence / "full_occlusion.txt", usable_length)
        out_of_view = read_binary_flags(source_sequence / "out_of_view.txt", usable_length)
        visible_count = sum(
            1
            for index in range(usable_length)
            if boxes[index][2] > 1 and boxes[index][3] > 1 and not full_occlusion[index] and not out_of_view[index]
        )
        if visible_count < max(1, int(min_visible_frames)):
            continue

        safe_name = slugify_sequence_name(f"lasot_{source_sequence.name}")
        if safe_name in exported:
            safe_name = f"{safe_name}_{len(exported) + 1}"
        sequence_dir = target_root / safe_name
        image_dir = sequence_dir / "img1"
        gt_dir = sequence_dir / "gt"
        image_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        selected_length = min(usable_length, max_frames) if max_frames > 0 else usable_length
        gt_lines: List[str] = []
        width = 0
        height = 0
        for frame_number, src in enumerate(image_paths[:selected_length], start=1):
            dst = image_dir / f"{frame_number:06d}{src.suffix.lower() or '.jpg'}"
            copy_or_link(src, dst, copy_mode)
            if width <= 0 or height <= 0:
                probe = cv2.imread(str(src))
                if probe is not None:
                    height, width = probe.shape[:2]
            index = frame_number - 1
            x, y, box_width, box_height = boxes[index]
            if box_width <= 1 or box_height <= 1:
                continue
            if full_occlusion[index] or out_of_view[index]:
                continue
            gt_lines.append(f"{frame_number},1,{x:.2f},{y:.2f},{box_width:.2f},{box_height:.2f},1,1,1.000\n")

        if not gt_lines:
            shutil.rmtree(sequence_dir, ignore_errors=True)
            continue
        (gt_dir / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")
        (sequence_dir / "seqinfo.ini").write_text(
            "\n".join(
                [
                    "[Sequence]",
                    f"name={safe_name}",
                    "imDir=img1",
                    "frameRate=30",
                    f"seqLength={selected_length}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    "imExt=.jpg",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        exported.append(safe_name)
        print(f"Exported LaSOT example {safe_name}: frames={selected_length} visible_gt_rows={len(gt_lines)}")
        if max_sequences > 0 and len(exported) >= max_sequences:
            break
    if not exported:
        raise RuntimeError("No LaSOT videos met the export criteria.")
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaSOT subset videos to MOT-style benchmark folders.")
    parser.add_argument("--lasot-root", type=Path, default=mot.PROJECT_ROOT / "data" / "raw" / "LaSOT_subset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "LaSOT_MOT_EXAMPLES")
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--min-visible-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exported = export_sequences(
        args.lasot_root,
        args.output_root,
        args.split,
        args.max_sequences,
        args.min_visible_frames,
        args.max_frames,
        args.copy_mode,
        args.val_fraction,
    )
    print("Exported sequences:")
    for name in exported:
        print(f"  {name}")
    print(f"Dataset root for benchmark: {args.output_root}")
    print(f"Benchmark split: {'val' if args.split in {'val', 'valid', 'validation'} else args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
