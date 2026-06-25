from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

import exercise_lorat_mot as exercise
import mot_common as mot


BBox = Tuple[float, float, float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week4_proposals"


@dataclass(frozen=True)
class FrameObjects:
    dataset: str
    sequence: str
    frame: int
    image_path: Path
    objects: Sequence[exercise.GroundTruthRow]


@dataclass(frozen=True)
class ProposalMatch:
    dataset: str
    sequence: str
    frame: int
    gt_track_id: int
    gt_bbox: BBox
    gt_area: float
    best_iou: float
    matched: bool
    best_proposal_score: Optional[float]
    best_proposal_source: str
    best_proposal_bbox: Optional[BBox]


@dataclass(frozen=True)
class FrameProposalSummary:
    dataset: str
    sequence: str
    frame: int
    image_path: Path
    gt_count: int
    proposal_count: int
    matched_gt_count: int
    recall: float
    manual_boxes_baseline: int
    manual_boxes_with_proposals: int
    manual_boxes_saved: int
    manual_effort_saved_rate: float
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Week 4 open-world proposal benchmark: class-agnostic proposal recall "
            "and manual box effort saved on DanceTrack/MOT-style data and TAO-style annotations."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "DanceTrack")
    parser.add_argument("--dataset", choices=("dancetrack", "mot17", "tao-ow", "tao"), default="dancetrack")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequence-name", action="append", default=[], help="Restrict to one or more sequence/video names.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--min-gt-area", type=float, default=1.0)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--proposal-source", choices=("auto", "selective_search", "contour"), default="auto")
    parser.add_argument(
        "--max-proposals-per-frame",
        type=int,
        default=1000,
        help="Proposal budget per frame. Large values measure recall; UI queues should use smaller values.",
    )
    parser.add_argument("--proposal-min-area", type=float, default=128.0)
    parser.add_argument("--proposal-iou-suppression", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def selected_names(args: argparse.Namespace) -> Optional[set[str]]:
    return set(args.sequence_name) if args.sequence_name else None


def iter_mot_frames(args: argparse.Namespace) -> Iterable[FrameObjects]:
    sequences = exercise.find_sequences(args.dataset_root, args.split)
    names = selected_names(args)
    if names:
        sequences = [sequence for sequence in sequences if sequence.name in names]
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    stride = max(1, int(args.frame_stride))
    for sequence in sequences:
        images = exercise.get_image_paths(sequence)
        gt_by_frame = exercise.read_gt(sequence)
        emitted = 0
        for frame_number in sorted(gt_by_frame):
            if (frame_number - 1) % stride != 0:
                continue
            image_index = exercise.frame_to_image_index(frame_number)
            if image_index >= len(images):
                continue
            objects = [
                row
                for row in gt_by_frame.get(frame_number, [])
                if row.confidence != 0
                and row.visibility >= args.min_visibility
                and mot.bbox_area(row.bbox) >= args.min_gt_area
            ]
            if not objects:
                continue
            yield FrameObjects(args.dataset, sequence.name, frame_number, images[image_index], objects)
            emitted += 1
            if args.max_frames > 0 and emitted >= args.max_frames:
                break


def find_tao_annotation_path(dataset_root: Path, split: str) -> Optional[Path]:
    candidates = [
        dataset_root / "annotations" / f"{split}.json",
        dataset_root / "annotations" / f"tao_{split}.json",
        dataset_root / f"{split}.json",
        dataset_root / f"tao_{split}.json",
        dataset_root / "TAO" / "annotations" / f"{split}.json",
        dataset_root / "TAO" / "annotations" / f"tao_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(dataset_root.rglob(f"*{split}*.json"))
    return found[0] if found else None


def resolve_tao_image_path(dataset_root: Path, file_name: str, allow_tree_search: bool = False) -> Optional[Path]:
    candidates = [
        dataset_root / file_name,
        dataset_root / "frames" / file_name,
        dataset_root / "TAO" / file_name,
        dataset_root / "TAO" / "frames" / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    if not allow_tree_search:
        return None
    matching = list(dataset_root.rglob(Path(file_name).name))
    return matching[0] if matching else None


def iter_tao_frames(args: argparse.Namespace) -> Iterable[FrameObjects]:
    annotation_path = find_tao_annotation_path(args.dataset_root, args.split)
    if annotation_path is None:
        raise FileNotFoundError(f"Could not find a TAO annotation JSON under {args.dataset_root}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    videos_by_id = {int(video["id"]): video for video in data.get("videos", [])}
    annotations_by_image: Dict[int, List[dict]] = {}
    for annotation in data.get("annotations", []):
        if "bbox" not in annotation or "image_id" not in annotation:
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    names = selected_names(args)
    emitted_by_sequence: Dict[str, int] = {}
    sequence_count = 0
    used_sequences: set[str] = set()
    stride = max(1, int(args.frame_stride))

    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        video = videos_by_id.get(int(image.get("video_id", -1)), {})
        sequence = str(video.get("name") or video.get("file_name") or image.get("video_id") or "tao")
        if names and sequence not in names:
            continue
        if sequence not in used_sequences:
            if args.max_sequences > 0 and sequence_count >= args.max_sequences:
                continue
            used_sequences.add(sequence)
            sequence_count += 1

        frame_number = int(image.get("frame_index", image.get("frame_id", emitted_by_sequence.get(sequence, 0)))) + 1
        if (frame_number - 1) % stride != 0:
            continue
        if args.max_frames > 0 and emitted_by_sequence.get(sequence, 0) >= args.max_frames:
            continue

        image_path = resolve_tao_image_path(args.dataset_root, str(image.get("file_name", "")))
        if image_path is None:
            continue
        objects: List[exercise.GroundTruthRow] = []
        for annotation in annotations_by_image.get(image_id, []):
            bbox_values = annotation.get("bbox", [])
            if len(bbox_values) != 4:
                continue
            bbox = tuple(float(value) for value in bbox_values)  # type: ignore[assignment]
            if mot.bbox_area(bbox) < args.min_gt_area:
                continue
            objects.append(
                exercise.GroundTruthRow(
                    frame=frame_number,
                    track_id=int(annotation.get("track_id", annotation.get("id", len(objects) + 1))),
                    bbox=bbox,
                    confidence=1.0,
                    class_id=int(annotation.get("category_id", 0)),
                    visibility=float(annotation.get("visibility", 1.0)),
                )
            )
        if not objects:
            continue
        emitted_by_sequence[sequence] = emitted_by_sequence.get(sequence, 0) + 1
        yield FrameObjects(args.dataset, sequence, frame_number, image_path, objects)


def iter_frames(args: argparse.Namespace) -> Iterable[FrameObjects]:
    if args.dataset in ("tao", "tao-ow"):
        yield from iter_tao_frames(args)
        return
    yield from iter_mot_frames(args)


def match_proposals(
    frame_objects: FrameObjects,
    proposals: Sequence[Tuple[BBox, float, str]],
    iou_threshold: float,
) -> List[ProposalMatch]:
    rows: List[ProposalMatch] = []
    for gt in frame_objects.objects:
        best_iou = 0.0
        best_score: Optional[float] = None
        best_source = ""
        best_bbox: Optional[BBox] = None
        for bbox, score, source in proposals:
            iou = mot.bbox_iou(gt.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_score = float(score)
                best_source = source
                best_bbox = bbox
        rows.append(
            ProposalMatch(
                dataset=frame_objects.dataset,
                sequence=frame_objects.sequence,
                frame=frame_objects.frame,
                gt_track_id=gt.track_id,
                gt_bbox=gt.bbox,
                gt_area=mot.bbox_area(gt.bbox),
                best_iou=best_iou,
                matched=best_iou >= iou_threshold,
                best_proposal_score=best_score,
                best_proposal_source=best_source,
                best_proposal_bbox=best_bbox,
            )
        )
    return rows


def evaluate_frame(args: argparse.Namespace, frame_objects: FrameObjects) -> Tuple[FrameProposalSummary, List[ProposalMatch]]:
    frame = cv2.imread(str(frame_objects.image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {frame_objects.image_path}")
    started = time.perf_counter()
    proposals = mot.generate_class_agnostic_proposals(
        frame,
        source=args.proposal_source,
        max_proposals=args.max_proposals_per_frame,
        min_area=args.proposal_min_area,
        nms_iou=args.proposal_iou_suppression,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    matches = match_proposals(frame_objects, proposals, args.iou_threshold)
    matched_count = sum(1 for row in matches if row.matched)
    gt_count = len(frame_objects.objects)
    recall = matched_count / gt_count if gt_count else 0.0
    summary = FrameProposalSummary(
        dataset=frame_objects.dataset,
        sequence=frame_objects.sequence,
        frame=frame_objects.frame,
        image_path=frame_objects.image_path,
        gt_count=gt_count,
        proposal_count=len(proposals),
        matched_gt_count=matched_count,
        recall=recall,
        manual_boxes_baseline=gt_count,
        manual_boxes_with_proposals=gt_count - matched_count,
        manual_boxes_saved=matched_count,
        manual_effort_saved_rate=recall,
        elapsed_ms=elapsed_ms,
    )
    return summary, matches


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bbox_columns(prefix: str, bbox: Optional[BBox]) -> Dict[str, str]:
    values = mot.csv_bbox(bbox)
    return {
        f"{prefix}_x": values[0],
        f"{prefix}_y": values[1],
        f"{prefix}_w": values[2],
        f"{prefix}_h": values[3],
    }


def summarize_sequences(frame_rows: Sequence[FrameProposalSummary]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[FrameProposalSummary]] = {}
    for row in frame_rows:
        grouped.setdefault((row.dataset, row.sequence), []).append(row)

    summary_rows: List[dict] = []
    for (dataset, sequence), rows in sorted(grouped.items()):
        gt_total = sum(row.gt_count for row in rows)
        matched_total = sum(row.matched_gt_count for row in rows)
        proposals_total = sum(row.proposal_count for row in rows)
        manual_saved = sum(row.manual_boxes_saved for row in rows)
        elapsed_values = [row.elapsed_ms for row in rows]
        summary_rows.append(
            {
                "dataset": dataset,
                "sequence": sequence,
                "frames": len(rows),
                "gt_objects": gt_total,
                "matched_gt_objects": matched_total,
                "proposal_recall": f"{(matched_total / gt_total) if gt_total else 0.0:.6f}",
                "manual_boxes_baseline": gt_total,
                "manual_boxes_with_proposals": gt_total - manual_saved,
                "manual_boxes_saved": manual_saved,
                "manual_effort_saved_rate": f"{(manual_saved / gt_total) if gt_total else 0.0:.6f}",
                "avg_proposals_per_frame": f"{(proposals_total / len(rows)) if rows else 0.0:.6f}",
                "mean_elapsed_ms": f"{statistics.fmean(elapsed_values) if elapsed_values else 0.0:.6f}",
            }
        )
    return summary_rows


def write_summary_md(path: Path, args: argparse.Namespace, sequence_rows: Sequence[dict], frame_rows: Sequence[FrameProposalSummary]) -> None:
    gt_total = sum(row.gt_count for row in frame_rows)
    matched_total = sum(row.matched_gt_count for row in frame_rows)
    manual_saved = sum(row.manual_boxes_saved for row in frame_rows)
    proposal_total = sum(row.proposal_count for row in frame_rows)
    recall = matched_total / gt_total if gt_total else 0.0
    effort_saved = manual_saved / gt_total if gt_total else 0.0
    lines = [
        "# Week 4 Open-World Proposal Benchmark",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Dataset root: `{args.dataset_root}`",
        f"- Split: `{args.split}`",
        f"- Proposal source: `{args.proposal_source}`",
        f"- IoU match threshold: `{args.iou_threshold:.2f}`",
        f"- Frames evaluated: `{len(frame_rows)}`",
        f"- Ground-truth objects evaluated: `{gt_total}`",
        f"- Proposals generated: `{proposal_total}`",
        f"- Proposal recall: `{recall:.3f}`",
        f"- Manual boxes saved under oracle acceptance: `{manual_saved}/{gt_total}` (`{effort_saved:.3f}`)",
        "",
        "## Per-Sequence Summary",
        "",
        "| Dataset | Sequence | Frames | GT Objects | Matched | Recall | Manual Saved | Avg Proposals/Frame | Mean ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequence_rows:
        lines.append(
            "| {dataset} | {sequence} | {frames} | {gt_objects} | {matched_gt_objects} | {proposal_recall} | "
            "{manual_boxes_saved} | {avg_proposals_per_frame} | {mean_elapsed_ms} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_summaries: List[FrameProposalSummary] = []
    match_rows: List[ProposalMatch] = []
    for frame_objects in iter_frames(args):
        summary, matches = evaluate_frame(args, frame_objects)
        frame_summaries.append(summary)
        match_rows.extend(matches)
        print(
            f"{summary.dataset}/{summary.sequence} frame {summary.frame}: "
            f"recall={summary.recall:.3f} gt={summary.gt_count} proposals={summary.proposal_count}",
            flush=True,
        )

    sequence_summary = summarize_sequences(frame_summaries)

    frame_csv_rows = [
        {
            "dataset": row.dataset,
            "sequence": row.sequence,
            "frame": row.frame,
            "image_path": str(row.image_path),
            "gt_count": row.gt_count,
            "proposal_count": row.proposal_count,
            "matched_gt_count": row.matched_gt_count,
            "recall": f"{row.recall:.6f}",
            "manual_boxes_baseline": row.manual_boxes_baseline,
            "manual_boxes_with_proposals": row.manual_boxes_with_proposals,
            "manual_boxes_saved": row.manual_boxes_saved,
            "manual_effort_saved_rate": f"{row.manual_effort_saved_rate:.6f}",
            "elapsed_ms": f"{row.elapsed_ms:.6f}",
        }
        for row in frame_summaries
    ]
    match_csv_rows = []
    for row in match_rows:
        match_csv_rows.append(
            {
                "dataset": row.dataset,
                "sequence": row.sequence,
                "frame": row.frame,
                "gt_track_id": row.gt_track_id,
                "gt_area": f"{row.gt_area:.6f}",
                "best_iou": f"{row.best_iou:.6f}",
                "matched": "1" if row.matched else "0",
                "best_proposal_score": "" if row.best_proposal_score is None else f"{row.best_proposal_score:.6f}",
                "best_proposal_source": row.best_proposal_source,
                **bbox_columns("gt_bbox", row.gt_bbox),
                **bbox_columns("best_proposal_bbox", row.best_proposal_bbox),
            }
        )

    write_csv(
        args.output_dir / "proposal_frame_summary.csv",
        frame_csv_rows,
        [
            "dataset",
            "sequence",
            "frame",
            "image_path",
            "gt_count",
            "proposal_count",
            "matched_gt_count",
            "recall",
            "manual_boxes_baseline",
            "manual_boxes_with_proposals",
            "manual_boxes_saved",
            "manual_effort_saved_rate",
            "elapsed_ms",
        ],
    )
    write_csv(
        args.output_dir / "proposal_gt_matches.csv",
        match_csv_rows,
        [
            "dataset",
            "sequence",
            "frame",
            "gt_track_id",
            "gt_area",
            "best_iou",
            "matched",
            "best_proposal_score",
            "best_proposal_source",
            "gt_bbox_x",
            "gt_bbox_y",
            "gt_bbox_w",
            "gt_bbox_h",
            "best_proposal_bbox_x",
            "best_proposal_bbox_y",
            "best_proposal_bbox_w",
            "best_proposal_bbox_h",
        ],
    )
    write_csv(
        args.output_dir / "proposal_sequence_summary.csv",
        sequence_summary,
        [
            "dataset",
            "sequence",
            "frames",
            "gt_objects",
            "matched_gt_objects",
            "proposal_recall",
            "manual_boxes_baseline",
            "manual_boxes_with_proposals",
            "manual_boxes_saved",
            "manual_effort_saved_rate",
            "avg_proposals_per_frame",
            "mean_elapsed_ms",
        ],
    )
    write_summary_md(args.output_dir / "summary.md", args, sequence_summary, frame_summaries)
    print(f"Wrote Week 4 proposal benchmark outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
