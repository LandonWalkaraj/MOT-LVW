"""Export a few TAO/TAO-OW videos into MOTChallenge-style folders.

The benchmark code already knows how to read MOT-style `img1/` and `gt/gt.txt`
folders. This adapter lets TAO examples flow through that same benchmark path
without adding another evaluator-specific code path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

import mot_common as mot


def slugify_video_name(name: str) -> str:
    result = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name.strip())
    result = result.strip("_")
    return result or "tao_video"


def annotation_path(root: Path, split: str, use_freeform: bool) -> Path:
    split_name = "validation" if split == "val" else split
    suffix = "_with_freeform" if use_freeform and split_name in {"train", "validation"} else ""
    candidates = [
        root / "annotations" / f"{split_name}{suffix}.json",
        root / "annotations_public" / f"{split_name}{suffix}.json",
        root / f"{split_name}{suffix}.json",
    ]
    if suffix:
        candidates.extend(
            [
                root / "annotations" / f"{split_name}.json",
                root / "annotations_public" / f"{split_name}.json",
                root / f"{split_name}.json",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find TAO annotation JSON for split={split!r} under {root}")


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


def load_tao_video_records(root: Path, split: str, use_freeform: bool) -> List[Tuple[str, List[dict], Dict[int, List[dict]]]]:
    path = annotation_path(root, split, use_freeform)
    data = json.loads(path.read_text(encoding="utf-8"))
    videos_by_id = {int(video["id"]): video for video in data.get("videos", []) if "id" in video}
    annotations_by_image: Dict[int, List[dict]] = {}
    for annotation in data.get("annotations", []):
        if "image_id" not in annotation or "bbox" not in annotation:
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    images_by_video: Dict[str, List[dict]] = {}
    for image in data.get("images", []):
        file_name = str(image.get("file_name", ""))
        if not file_name:
            continue
        frame_path = root / "frames" / file_name
        if not frame_path.exists():
            continue
        video_id = int(image.get("video_id", -1))
        video = videos_by_id.get(video_id, {})
        video_name = str(image.get("video") or video.get("name") or Path(file_name).parent.as_posix())
        image = dict(image)
        image["_frame_path"] = str(frame_path)
        images_by_video.setdefault(video_name, []).append(image)

    records: List[Tuple[str, List[dict], Dict[int, List[dict]]]] = []
    for video_name, images in sorted(images_by_video.items(), key=lambda item: item[0].lower()):
        images = sorted(
            images,
            key=lambda image: (
                int(image.get("frame_index", image.get("frame_id", 0))),
                str(image.get("file_name", "")),
            ),
        )
        image_ids = {int(image["id"]) for image in images if "id" in image}
        video_annotations = {
            image_id: annotations_by_image.get(image_id, [])
            for image_id in image_ids
            if annotations_by_image.get(image_id)
        }
        if video_annotations:
            records.append((video_name, images, video_annotations))
    return records


def export_videos(
    tao_root: Path,
    output_root: Path,
    split: str,
    max_videos: int,
    min_tracks: int,
    min_annotated_frames: int,
    max_frames: int,
    copy_mode: str,
    use_freeform: bool,
) -> List[str]:
    output_split = "val" if split in {"val", "validation"} else split
    target_root = output_root / output_split
    target_root.mkdir(parents=True, exist_ok=True)
    exported: List[str] = []
    for video_name, images, annotations_by_image in load_tao_video_records(tao_root, split, use_freeform):
        track_ids = {
            int(annotation.get("track_id", annotation.get("id", -1)))
            for annotations in annotations_by_image.values()
            for annotation in annotations
            if "bbox" in annotation
        }
        if len(track_ids) < max(1, int(min_tracks)):
            continue
        annotated_frame_count = sum(1 for image in images if int(image.get("id", -1)) in annotations_by_image)
        if annotated_frame_count < max(1, int(min_annotated_frames)):
            continue
        safe_name = slugify_video_name(video_name)
        if safe_name in exported:
            safe_name = f"{safe_name}_{len(exported) + 1}"
        sequence_dir = target_root / safe_name
        image_dir = sequence_dir / "img1"
        gt_dir = sequence_dir / "gt"
        image_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        selected_images = images[: max_frames] if max_frames > 0 else images
        gt_lines: List[str] = []
        width = 0
        height = 0
        for frame_number, image in enumerate(selected_images, start=1):
            src = Path(str(image["_frame_path"]))
            dst = image_dir / f"{frame_number:06d}{src.suffix.lower() or '.jpg'}"
            copy_or_link(src, dst, copy_mode)
            if width <= 0 or height <= 0:
                probe = cv2.imread(str(src))
                if probe is not None:
                    height, width = probe.shape[:2]
            for annotation in annotations_by_image.get(int(image.get("id", -1)), []):
                bbox = annotation.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x, y, w, h = (float(value) for value in bbox)
                if w <= 0 or h <= 0:
                    continue
                track_id = int(annotation.get("track_id", annotation.get("id", len(gt_lines) + 1)))
                visibility = float(annotation.get("visibility", 1.0))
                # Normalize class id to 1 so existing MOT benchmark defaults
                # select the exported TAO objects without extra CLI flags.
                gt_lines.append(f"{frame_number},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,{visibility:.3f}\n")
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
                    f"seqLength={len(selected_images)}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    "imExt=.jpg",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        exported.append(safe_name)
        print(f"Exported TAO example {safe_name}: frames={len(selected_images)} gt_rows={len(gt_lines)}")
        if max_videos > 0 and len(exported) >= max_videos:
            break
    if not exported:
        raise RuntimeError("No TAO videos met the export criteria.")
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TAO/TAO-OW subset videos to MOT-style benchmark folders.")
    parser.add_argument("--tao-root", type=Path, default=mot.PROJECT_ROOT / "data" / "raw" / "TAO_OW_SUBSET")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "TAO_OW_MOT_EXAMPLES")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--min-tracks", type=int, default=1)
    parser.add_argument("--min-annotated-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--use-freeform", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exported = export_videos(
        args.tao_root,
        args.output_root,
        args.split,
        args.max_videos,
        args.min_tracks,
        args.min_annotated_frames,
        args.max_frames,
        args.copy_mode,
        args.use_freeform,
    )
    print("Exported sequences:")
    for name in exported:
        print(f"  {name}")
    print(f"Dataset root for benchmark: {args.output_root}")
    print(f"Benchmark split: {'val' if args.split in {'val', 'validation'} else args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
