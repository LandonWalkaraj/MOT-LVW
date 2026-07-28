"""Export BDD100K MOT videos into MOTChallenge-style benchmark folders.

The V9 benchmark reads MOT-style `img1/` and `gt/gt.txt` folders. BDD100K
tracking annotations are Scalabel-style JSON files, one JSON file per video, so
this adapter converts a small benchmark subset without adding another evaluator
path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2

import mot_common as mot


DEFAULT_BDD100K_CATEGORIES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
DEFAULT_BDD100K_DISTRACTORS = ("other person", "trailer", "other vehicle")


def slugify_sequence_name(name: str) -> str:
    result = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name.strip())
    result = result.strip("_")
    return result or "bdd100k_sequence"


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


def first_existing_dir(candidates: Sequence[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find BDD100K {description}. Checked: {checked}")


def image_split_root(root: Path, split: str) -> Path:
    return first_existing_dir(
        [
            root / "images" / "track" / split,
            root / "bdd100k" / "images" / "track" / split,
            root / "images" / "track_20" / split,
            root / "bdd100k" / "images" / "track_20" / split,
        ],
        f"image split root for split={split!r}",
    )


def label_split_root(root: Path, split: str) -> Path:
    return first_existing_dir(
        [
            root / "labels-20" / "box-track" / split,
            root / "bdd100k" / "labels-20" / "box-track" / split,
            root / "labels-20" / "box_track" / split,
            root / "bdd100k" / "labels-20" / "box_track" / split,
            root / "labels" / "box_track_20" / split,
            root / "bdd100k" / "labels" / "box_track_20" / split,
            root / "labels" / "box-track" / split,
            root / "bdd100k" / "labels" / "box-track" / split,
        ],
        f"box-track label split root for split={split!r}",
    )


def parse_csv_names(text: str, defaults: Sequence[str]) -> List[str]:
    if not text.strip():
        return [value.lower() for value in defaults]
    return [token.strip().lower() for token in text.replace(";", ",").split(",") if token.strip()]


def read_video_frames(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"BDD100K label file should contain a list of frames: {path}")
    return [frame for frame in data if isinstance(frame, dict)]


def frame_sort_key(frame: dict) -> Tuple[int, str]:
    try:
        index = int(frame.get("index", 0))
    except (TypeError, ValueError):
        index = 0
    return index, str(frame.get("name", ""))


def find_frame_image(frame: dict, image_root: Path, video_name: str) -> Path:
    frame_name = str(frame.get("name", "")).strip()
    candidates: List[Path] = []
    if frame_name:
        raw = Path(frame_name)
        if raw.is_absolute():
            candidates.append(raw)
        candidates.extend(
            [
                image_root / video_name / frame_name,
                image_root / frame_name,
            ]
        )
    try:
        index = int(frame.get("index", -1))
    except (TypeError, ValueError):
        index = -1
    if index >= 0:
        candidates.extend(sorted((image_root / video_name).glob(f"*-{index:07d}.*")))
        candidates.extend(sorted((image_root / video_name).glob(f"*{index:07d}.*")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate image for BDD100K frame name={frame_name!r} video={video_name!r}")


def iter_candidate_label_files(label_root: Path) -> Iterable[Path]:
    yield from sorted(label_root.glob("*.json"), key=lambda path: path.name.lower())


def export_videos(
    bdd100k_root: Path,
    output_root: Path,
    split: str,
    max_videos: int,
    min_tracks: int,
    min_annotated_frames: int,
    max_frames: int,
    copy_mode: str,
    categories: Sequence[str],
    distractors: Sequence[str],
    include_distractors: bool,
) -> List[str]:
    output_split = "val" if split in {"valid", "validation"} else split
    target_root = output_root / output_split
    target_root.mkdir(parents=True, exist_ok=True)
    images_root = image_split_root(bdd100k_root, split)
    labels_root = label_split_root(bdd100k_root, split)
    category_set = {value.lower() for value in categories}
    distractor_set = {value.lower() for value in distractors}

    exported: List[str] = []
    for label_file in iter_candidate_label_files(labels_root):
        frames = sorted(read_video_frames(label_file), key=frame_sort_key)
        if max_frames > 0:
            frames = frames[:max_frames]
        if not frames:
            continue
        video_name = str(frames[0].get("videoName") or label_file.stem)
        track_key_to_id: Dict[Tuple[str, str], int] = {}
        gt_rows_by_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
        for frame_number, frame in enumerate(frames, start=1):
            labels = frame.get("labels") or []
            if not isinstance(labels, list):
                continue
            for label in labels:
                if not isinstance(label, dict):
                    continue
                category = str(label.get("category", "")).strip().lower()
                if not category:
                    continue
                if not include_distractors and category in distractor_set:
                    continue
                if category_set and category not in category_set:
                    continue
                attributes = label.get("attributes") or {}
                if isinstance(attributes, dict) and bool(attributes.get("Crowd", False)):
                    continue
                box = label.get("box2d") or {}
                if not isinstance(box, dict):
                    continue
                try:
                    x1 = float(box["x1"])
                    y1 = float(box["y1"])
                    x2 = float(box["x2"])
                    y2 = float(box["y2"])
                except (KeyError, TypeError, ValueError):
                    continue
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 1 or height <= 1:
                    continue
                source_id = str(label.get("id", "")).strip() or f"{category}:{len(track_key_to_id) + 1}"
                key = (category, source_id)
                if key not in track_key_to_id:
                    track_key_to_id[key] = len(track_key_to_id) + 1
                visibility = 1.0
                if isinstance(attributes, dict):
                    if bool(attributes.get("Occluded", False)):
                        visibility = min(visibility, 0.35)
                    if bool(attributes.get("Truncated", False)):
                        visibility = min(visibility, 0.70)
                gt_rows_by_frame.setdefault(frame_number, []).append(
                    (track_key_to_id[key], x1, y1, width, height, visibility)
                )

        annotated_frames = sum(1 for rows in gt_rows_by_frame.values() if rows)
        if len(track_key_to_id) < max(1, int(min_tracks)) or annotated_frames < max(1, int(min_annotated_frames)):
            continue

        safe_name = slugify_sequence_name(f"bdd100k_{video_name}")
        if safe_name in exported:
            safe_name = f"{safe_name}_{len(exported) + 1}"
        sequence_dir = target_root / safe_name
        image_dir = sequence_dir / "img1"
        gt_dir = sequence_dir / "gt"
        image_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        gt_lines: List[str] = []
        width = 0
        height = 0
        copied_frames = 0
        for frame_number, frame in enumerate(frames, start=1):
            src = find_frame_image(frame, images_root, video_name)
            dst = image_dir / f"{frame_number:06d}{src.suffix.lower() or '.jpg'}"
            copy_or_link(src, dst, copy_mode)
            copied_frames += 1
            if width <= 0 or height <= 0:
                probe = cv2.imread(str(src))
                if probe is not None:
                    height, width = probe.shape[:2]
            for track_id, x, y, box_width, box_height, visibility in gt_rows_by_frame.get(frame_number, []):
                # Normalize class id to 1 so selected-object benchmarks can
                # initialize cars, people, bikes, etc. using the same CLI path.
                gt_lines.append(
                    f"{frame_number},{track_id},{x:.2f},{y:.2f},{box_width:.2f},{box_height:.2f},1,1,{visibility:.3f}\n"
                )

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
                    "frameRate=5",
                    f"seqLength={copied_frames}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    "imExt=.jpg",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        exported.append(safe_name)
        print(
            f"Exported BDD100K example {safe_name}: "
            f"frames={copied_frames} tracks={len(track_key_to_id)} gt_rows={len(gt_lines)}"
        )
        if max_videos > 0 and len(exported) >= max_videos:
            break
    if not exported:
        raise RuntimeError("No BDD100K MOT videos met the export criteria.")
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BDD100K MOT videos to MOT-style benchmark folders.")
    parser.add_argument("--bdd100k-root", type=Path, default=mot.PROJECT_ROOT / "data" / "raw" / "BDD100K")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "BDD100K_MOT_EXAMPLES")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--min-tracks", type=int, default=5)
    parser.add_argument("--min-annotated-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--categories", default=",".join(DEFAULT_BDD100K_CATEGORIES))
    parser.add_argument("--distractors", default=",".join(DEFAULT_BDD100K_DISTRACTORS))
    parser.add_argument("--include-distractors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exported = export_videos(
        args.bdd100k_root,
        args.output_root,
        args.split,
        args.max_videos,
        args.min_tracks,
        args.min_annotated_frames,
        args.max_frames,
        args.copy_mode,
        parse_csv_names(args.categories, DEFAULT_BDD100K_CATEGORIES),
        parse_csv_names(args.distractors, DEFAULT_BDD100K_DISTRACTORS),
        args.include_distractors,
    )
    print("Exported sequences:")
    for name in exported:
        print(f"  {name}")
    print(f"Dataset root for benchmark: {args.output_root}")
    print(f"Benchmark split: {'val' if args.split in {'valid', 'validation'} else args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
