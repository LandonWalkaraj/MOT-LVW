from __future__ import annotations

import argparse
import configparser
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


def read_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def write_gt(rows: Iterable[dict], output_root: Path) -> int:
    rows_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_sequence[str(row["sequence"])].append(row)

    count = 0
    for sequence, sequence_rows in rows_by_sequence.items():
        gt_dir = output_root / "train" / sequence / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_path = gt_dir / "gt.txt"
        with gt_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in sorted(sequence_rows, key=lambda item: (item["frame"], item["track_id"])):
                fields = (
                    row["frame"],
                    row["track_id"],
                    row["bbox_left"],
                    row["bbox_top"],
                    row["bbox_width"],
                    row["bbox_height"],
                    row["conf"],
                    row["class_id"],
                    row["visibility"],
                )
                file.write(",".join(str(value) for value in fields) + "\n")
                count += 1
    return count


def write_seqinfo(rows: Iterable[dict], output_root: Path) -> int:
    count = 0
    for row in rows:
        if row["split"] != "train":
            continue
        sequence_dir = output_root / "train" / str(row["sequence"])
        sequence_dir.mkdir(parents=True, exist_ok=True)
        config = configparser.ConfigParser()
        config["Sequence"] = {
            "name": str(row["sequence"]),
            "imDir": "img1",
            "frameRate": str(row["fps"]),
            "seqLength": str(row["seq_length"]),
            "imWidth": str(row["width"]),
            "imHeight": str(row["height"]),
            "imExt": ".jpg",
        }
        with (sequence_dir / "seqinfo.ini").open("w", encoding="utf-8") as file:
            config.write(file)
        count += 1
    return count


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_images(rows: Iterable[dict], source_root: Path, output_root: Path, mode: str) -> int:
    count = 0
    for row in rows:
        relative = Path(row["image_path"])
        src = source_root / relative
        dst = output_root / "train" / str(row["sequence"]) / "img1" / relative.name
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        link_or_copy(src, dst, mode)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOT17 parquet mirror into MOTChallenge train layout.")
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/MOTChallenge/mot17-parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/MOTChallenge/MOT17"))
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="Use hardlinks by default to avoid duplicating the downloaded images on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    frames = read_rows(source_root / "data" / "frames" / "train-00000-of-00001.parquet")
    gt_rows = read_rows(source_root / "data" / "gt" / "train-00000-of-00001.parquet")
    seqinfo_rows = read_rows(source_root / "data" / "seqinfo" / "seqinfo.parquet")

    image_count = export_images(frames, source_root, output_root, args.image_mode)
    gt_count = write_gt(gt_rows, output_root)
    seqinfo_count = write_seqinfo(seqinfo_rows, output_root)
    sequence_count = len({row["sequence"] for row in frames})

    print(f"Exported MOT17 train layout to {output_root}")
    print(f"Sequences: {sequence_count}")
    print(f"Images: {image_count}")
    print(f"GT rows: {gt_count}")
    print(f"seqinfo.ini files: {seqinfo_count}")


if __name__ == "__main__":
    main()
