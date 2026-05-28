from __future__ import annotations

import argparse
import configparser
import csv
import importlib.util
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_V3_PATH = PROJECT_ROOT / "programs" / "bounding_box_v3_lorat.py"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "DanceTrack"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-exercise" / "dancetrack"
LORAT_CONFIG_CHOICES = ("B-224", "B-378", "L-224", "L-378", "g-224", "g-378")


@dataclass(frozen=True)
class GroundTruthRow:
    frame: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    visibility: float


@dataclass(frozen=True)
class RunSummary:
    sequence: str
    backend: str
    lorat_config: str
    backbone: str
    input_size: int
    device: str
    checkpoint_mb: float
    init_frame: int
    tracks: int
    gt_track_ids: str
    frames: int
    seconds: float
    fps: float
    mean_iou: Optional[float]
    hit_rate_50: Optional[float]
    result_path: Path
    preview_path: Optional[Path]


def load_gui_v3_module():
    spec = importlib.util.spec_from_file_location("bounding_box_v3", GUI_V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import v3 GUI module from {GUI_V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise the LoRAT multi-object tracker on DanceTrack sequences."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root folder for DanceTrack.")
    parser.add_argument("--split", default="val", help="DanceTrack split folder to scan, for example train/val/test.")
    parser.add_argument("--sequence", action="append", help="Sequence name to run. Repeat for multiple sequences.")
    parser.add_argument("--list-sequences", action="store_true", help="List detected sequences and exit.")
    parser.add_argument("--extract-zips", action="store_true", help="Extract *.zip files found under --dataset-root.")
    parser.add_argument("--backend", choices=("lorat", "opencv"), default="lorat")
    parser.add_argument("--device", default="cpu", help="LoRAT device, e.g. cpu or cuda:0.")
    parser.add_argument("--lorat-root", type=Path, default=PROJECT_ROOT / "external" / "LoRAT-main")
    parser.add_argument("--lorat-config", default="B-224", choices=LORAT_CONFIG_CHOICES)
    parser.add_argument(
        "--compare-configs",
        nargs="+",
        choices=LORAT_CONFIG_CHOICES + ("all",),
        help="Run several LoRAT configs and write a comparison CSV/Markdown summary.",
    )
    parser.add_argument("--comparison-csv", type=Path, help="Optional path for model comparison CSV output.")
    parser.add_argument("--comparison-md", type=Path, help="Optional path for model comparison Markdown output.")
    parser.add_argument("--weight-path", type=Path, help="Optional LoRAT weight override.")
    parser.add_argument("--max-tracks", type=int, default=8, help="Maximum tracks initialized per sequence.")
    parser.add_argument("--max-frames", type=int, default=150, help="Frames to process per sequence; 0 means full sequence.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Limit number of sequences; 0 means all selected.")
    parser.add_argument("--init-frame", default="auto", help="1-based frame number or auto.")
    parser.add_argument(
        "--min-init-tracks",
        type=int,
        default=2,
        help="Minimum number of visible GT tracks to prefer when --init-frame=auto.",
    )
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to initialize. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Save annotated MP4 previews. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Skip annotated MP4 preview output.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-coordinator", action="store_true", help="Disable overlap/motion coordination.")
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.30)
    parser.add_argument("--normal-proposal-weight", type=float, default=0.90)
    parser.add_argument("--occluded-proposal-weight", type=float, default=0.45)
    parser.add_argument("--suspicious-proposal-weight", type=float, default=0.25)
    parser.add_argument("--max-center-jump", type=float, default=0.45)
    parser.add_argument("--max-scale-change", type=float, default=0.65)
    parser.add_argument("--max-occlusion-frames", type=int, default=20)
    return parser.parse_args()


def extract_zips(dataset_root: Path) -> None:
    zip_paths = sorted(dataset_root.glob("*.zip"))
    if not zip_paths:
        print(f"No zip files found in {dataset_root}")
        return

    for zip_path in zip_paths:
        target = dataset_root / zip_path.stem
        if target.exists():
            print(f"Already extracted: {target}")
            continue
        print(f"Extracting {zip_path} -> {dataset_root}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dataset_root)


def find_sequences(dataset_root: Path, split: str) -> List[Path]:
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    sequences = []
    for image_dir in sorted(dataset_root.rglob("img1")):
        if not image_dir.is_dir():
            continue

        sequence = image_dir.parent
        relative_parts = sequence.relative_to(dataset_root).parts
        if split and split not in relative_parts:
            continue

        sequences.append(sequence)

    deduped = {}
    for sequence in sequences:
        deduped[str(sequence.resolve())] = sequence
    return [deduped[key] for key in sorted(deduped)]


def read_sequence_info(sequence_path: Path) -> Tuple[float, Optional[int]]:
    seqinfo_path = sequence_path / "seqinfo.ini"
    if not seqinfo_path.exists():
        image_count = len(get_image_paths(sequence_path))
        return 30.0, image_count if image_count > 0 else None

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    section = parser["Sequence"] if parser.has_section("Sequence") else {}
    fps = float(section.get("frameRate", 30.0))
    length_text = section.get("seqLength")
    length = int(length_text) if length_text else None
    return fps, length


def get_image_paths(sequence_path: Path) -> List[Path]:
    image_dir = sequence_path / "img1"
    image_paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(suffix))
    return sorted(image_paths)


def read_gt(sequence_path: Path) -> Dict[int, List[GroundTruthRow]]:
    gt_path = sequence_path / "gt" / "gt.txt"
    if not gt_path.exists():
        raise RuntimeError(f"Ground-truth file not found: {gt_path}")

    by_frame: Dict[int, List[GroundTruthRow]] = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        bbox = tuple(float(value) for value in parts[2:6])
        confidence = float(parts[6]) if len(parts) > 6 else 1.0
        class_id = int(float(parts[7])) if len(parts) > 7 else 1
        visibility = float(parts[8]) if len(parts) > 8 else 1.0
        row = GroundTruthRow(frame, track_id, bbox, confidence, class_id, visibility)
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def pick_initial_rows(
    gt_by_frame: Dict[int, List[GroundTruthRow]],
    init_frame_arg: str,
    class_ids: Sequence[int],
    min_visibility: float,
    max_tracks: int,
    min_init_tracks: int,
) -> Tuple[int, List[GroundTruthRow]]:
    class_id_set = set(class_ids or [1])

    def usable_rows(frame: int) -> List[GroundTruthRow]:
        rows = [
            row
            for row in gt_by_frame.get(frame, [])
            if row.confidence != 0 and row.class_id in class_id_set and row.visibility >= min_visibility
        ]
        rows.sort(key=lambda row: row.track_id)
        return rows[:max_tracks]

    if init_frame_arg != "auto":
        frame = int(init_frame_arg)
        rows = usable_rows(frame)
        if not rows:
            raise RuntimeError(f"No usable GT boxes found on requested init frame {frame}.")
        return frame, rows

    candidates = []
    for frame in sorted(gt_by_frame):
        rows = usable_rows(frame)
        if rows:
            candidates.append((frame, rows))

    if candidates:
        required_tracks = max(1, min(min_init_tracks, max_tracks))
        enough_tracks = [(frame, rows) for frame, rows in candidates if len(rows) >= required_tracks]
        ranked_candidates = enough_tracks or candidates
        ranked_candidates.sort(key=lambda item: (-len(item[1]), item[0]))
        return ranked_candidates[0]
    raise RuntimeError("No usable GT boxes found in sequence.")


def frame_to_image_index(frame_number: int) -> int:
    return frame_number - 1


def bbox_iou(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_x2 = lx + lw
    left_y2 = ly + lh
    right_x2 = rx + rw
    right_y2 = ry + rh

    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = (lw * lh) + (rw * rh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


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
        gt_row = gt_rows.get(gt_track_id)
        if gt_row is None:
            continue
        score = bbox_iou(track.bbox, gt_row.bbox)
        metrics["count"] += 1
        metrics["iou_sum"] += score
        if score >= 0.5:
            metrics["hit50"] += 1


def lorat_config_metadata(config: str) -> Tuple[str, int]:
    model_name, input_size_text = config.split("-", 1)
    backbone_by_model = {
        "B": "ViT-B/14",
        "L": "ViT-L/14",
        "g": "ViT-g/14",
    }
    return backbone_by_model.get(model_name, model_name), int(input_size_text)


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    raise RuntimeError("CSRT is unavailable. Install opencv-contrib-python.")


class ExerciseOpenCVMultiObjectTracker:
    backend_name = "OpenCV CSRT"

    def __init__(self, gui_v3):
        self.gui_v3 = gui_v3
        self.tracks = []
        self.next_track_id = 1

    def initialize(self, frame, boxes: Sequence[Tuple[float, float, float, float]], frame_number: int = 1) -> None:
        self.add_tracks(frame, boxes, frame_number)

    def add_tracks(self, frame, boxes: Sequence[Tuple[float, float, float, float]], frame_number: int = 1):
        added_tracks = []
        for bbox in boxes:
            clipped = self.gui_v3.clip_bbox_to_frame(frame, bbox)
            if clipped is None:
                continue

            tracker = create_csrt_tracker()
            initialized = tracker.init(frame, tuple(int(value) for value in clipped))
            if initialized is False:
                continue

            track = self.gui_v3.TrackState(
                track_id=self.next_track_id,
                bbox=tuple(float(value) for value in clipped),
                color=self.gui_v3.color_for_track(self.next_track_id),
                confidence=1.0,
                tracker=tracker,
            )
            self.tracks.append(track)
            added_tracks.append(track)
            self.next_track_id += 1
        return added_tracks

    def update(self, frame, frame_number: int):
        for track in self.tracks:
            if not track.ok or track.tracker is None:
                track.lost_frames += 1
                continue
            ok, bbox = track.tracker.update(frame)
            track.ok = bool(ok)
            if ok:
                track.bbox = tuple(float(value) for value in bbox)
                track.confidence = 1.0
                track.lost_frames = 0
            else:
                track.lost_frames += 1
        return self.tracks

    def close(self) -> None:
        return None


def build_backend(gui_v3, args: argparse.Namespace, sequence_name: str, fps: float, length: Optional[int]):
    if args.backend == "opencv":
        return ExerciseOpenCVMultiObjectTracker(gui_v3)

    weight_path = args.weight_path or gui_v3.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return gui_v3.LoRATMultiObjectTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        args.max_tracks,
        fps,
        length,
        sequence_name,
        args.confidence_threshold,
        args.disable_amp,
        not args.disable_coordinator,
        args.overlap_iou_threshold,
        args.normal_proposal_weight,
        args.occluded_proposal_weight,
        args.suspicious_proposal_weight,
        args.max_center_jump,
        args.max_scale_change,
        args.max_occlusion_frames,
    )


def run_sequence(gui_v3, args: argparse.Namespace, sequence_path: Path) -> RunSummary:
    image_paths = get_image_paths(sequence_path)
    if not image_paths:
        raise RuntimeError(f"No frames found in {sequence_path / 'img1'}")

    gt_by_frame = read_gt(sequence_path)
    fps, sequence_length = read_sequence_info(sequence_path)
    init_frame, init_rows = pick_initial_rows(
        gt_by_frame,
        args.init_frame,
        args.class_id,
        args.min_visibility,
        args.max_tracks,
        args.min_init_tracks,
    )
    init_index = frame_to_image_index(init_frame)
    if init_index >= len(image_paths):
        raise RuntimeError(f"Init frame {init_frame} is outside image sequence length {len(image_paths)}.")

    output_dir = args.output_root / args.split / args.backend
    if args.backend == "lorat" and getattr(args, "output_by_config", False):
        output_dir = output_dir / args.lorat_config
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{sequence_path.name}.txt"

    preview_writer = None
    preview_path = output_dir / f"{sequence_path.name}.mp4" if args.save_video else None

    init_frame_image = cv2.imread(str(image_paths[init_index]))
    if init_frame_image is None:
        raise RuntimeError(f"Unable to read frame: {image_paths[init_index]}")

    mot_lines: List[str] = []
    boxes = [row.bbox for row in init_rows]
    tracker_to_gt_id: Dict[int, int] = {}
    metrics = {"count": 0.0, "iou_sum": 0.0, "hit50": 0.0}

    end_index = len(image_paths) - 1
    if args.max_frames > 0:
        end_index = min(end_index, init_index + args.max_frames - 1)

    started_at = time.perf_counter()
    backend = build_backend(gui_v3, args, sequence_path.name, fps, sequence_length or len(image_paths))
    try:
        backend.initialize(init_frame_image, boxes, init_frame)
        tracker_to_gt_id = {
            track.track_id: init_row.track_id
            for track, init_row in zip(backend.tracks, init_rows)
        }
        gui_v3.append_mot_results(mot_lines, init_frame, backend.tracks)
        update_overlap_metrics(metrics, backend.tracks, gt_by_frame, init_frame, tracker_to_gt_id, args.min_visibility)

        if args.save_video:
            preview_writer = gui_v3.make_video_writer(preview_path, fps, init_frame_image)
            preview_writer.write(gui_v3.draw_tracks(init_frame_image, backend.tracks, init_frame, backend.backend_name))

        for image_index in range(init_index + 1, end_index + 1):
            frame_number = image_index + 1
            frame = cv2.imread(str(image_paths[image_index]))
            if frame is None:
                print(f"Skipping unreadable frame: {image_paths[image_index]}")
                continue
            backend.update(frame, frame_number)
            gui_v3.append_mot_results(mot_lines, frame_number, backend.tracks)
            update_overlap_metrics(metrics, backend.tracks, gt_by_frame, frame_number, tracker_to_gt_id, args.min_visibility)
            if preview_writer is not None:
                preview_writer.write(gui_v3.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name))
    finally:
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    seconds = time.perf_counter() - started_at
    frame_count = end_index - init_index + 1
    run_fps = frame_count / seconds if seconds > 0 else 0.0
    mean_iou = metrics["iou_sum"] / metrics["count"] if metrics["count"] else None
    hit_rate_50 = metrics["hit50"] / metrics["count"] if metrics["count"] else None
    weight_path = args.weight_path if args.backend == "lorat" and args.weight_path else None
    if args.backend == "lorat" and weight_path is None:
        weight_path = gui_v3.LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    checkpoint_mb = weight_path.stat().st_size / (1024 * 1024) if weight_path and weight_path.exists() else 0.0
    backbone, input_size = lorat_config_metadata(args.lorat_config) if args.backend == "lorat" else ("", 0)

    result_path.write_text("".join(mot_lines), encoding="utf-8")
    summary = RunSummary(
        sequence=sequence_path.name,
        backend=args.backend,
        lorat_config=args.lorat_config if args.backend == "lorat" else "",
        backbone=backbone,
        input_size=input_size,
        device=args.device if args.backend == "lorat" else "",
        checkpoint_mb=checkpoint_mb,
        init_frame=init_frame,
        tracks=len(init_rows),
        gt_track_ids=",".join(str(row.track_id) for row in init_rows),
        frames=frame_count,
        seconds=seconds,
        fps=run_fps,
        mean_iou=mean_iou,
        hit_rate_50=hit_rate_50,
        result_path=result_path,
        preview_path=preview_path,
    )
    metric_text = ""
    if mean_iou is not None and hit_rate_50 is not None:
        metric_text = f", mean_iou={mean_iou:.3f}, iou50={hit_rate_50:.3f}, fps={run_fps:.2f}"
    print(
        f"{sequence_path.name}: init_frame={init_frame}, tracks={len(init_rows)}, "
        f"track_ids={','.join(str(row.track_id) for row in init_rows)}, "
        f"frames={frame_count}{metric_text}, result={result_path}"
    )
    return summary


def normalized_compare_configs(configs: Optional[Sequence[str]]) -> List[str]:
    if not configs:
        return []
    if "all" in configs:
        return list(LORAT_CONFIG_CHOICES)

    ordered_configs = []
    seen = set()
    for config in configs:
        if config in seen:
            continue
        ordered_configs.append(config)
        seen.add(config)
    return ordered_configs


def format_optional_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_comparison_outputs(args: argparse.Namespace, summaries: Sequence[RunSummary]) -> Tuple[Path, Path]:
    comparison_dir = args.output_root / args.split / "lorat" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.comparison_csv or comparison_dir / "lorat_model_comparison.csv"
    md_path = args.comparison_md or comparison_dir / "lorat_model_comparison.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sequence",
        "backend",
        "lorat_config",
        "backbone",
        "input_size",
        "device",
        "checkpoint_mb",
        "init_frame",
        "tracks",
        "gt_track_ids",
        "frames",
        "seconds",
        "fps",
        "mean_iou",
        "iou50",
        "result_path",
        "preview_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "sequence": summary.sequence,
                    "backend": summary.backend,
                    "lorat_config": summary.lorat_config,
                    "backbone": summary.backbone,
                    "input_size": summary.input_size,
                    "device": summary.device,
                    "checkpoint_mb": f"{summary.checkpoint_mb:.2f}",
                    "init_frame": summary.init_frame,
                    "tracks": summary.tracks,
                    "gt_track_ids": summary.gt_track_ids,
                    "frames": summary.frames,
                    "seconds": f"{summary.seconds:.3f}",
                    "fps": f"{summary.fps:.3f}",
                    "mean_iou": format_optional_float(summary.mean_iou),
                    "iou50": format_optional_float(summary.hit_rate_50),
                    "result_path": str(summary.result_path),
                    "preview_path": str(summary.preview_path or ""),
                }
            )

    md_lines = [
        "# LoRAT Model Comparison",
        "",
        "These are lightweight local exercise metrics against initialized DanceTrack ground-truth IDs. They are not a replacement for formal TrackEval HOTA/MOTA/IDF1 evaluation.",
        "",
        "Checkpoint MB is the local LoRAT checkpoint size. DINOv2 backbone checkpoints are cached separately by Torch.",
        "",
        "| Sequence | Config | Backbone | Input | Device | Checkpoint MB | Tracks | Frames | Seconds | FPS | Mean IoU | IoU@0.50 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        md_lines.append(
            "| "
            f"{summary.sequence} | "
            f"{summary.lorat_config} | "
            f"{summary.backbone} | "
            f"{summary.input_size} | "
            f"{summary.device} | "
            f"{summary.checkpoint_mb:.2f} | "
            f"{summary.tracks} | "
            f"{summary.frames} | "
            f"{summary.seconds:.3f} | "
            f"{summary.fps:.3f} | "
            f"{format_optional_float(summary.mean_iou, 3)} | "
            f"{format_optional_float(summary.hit_rate_50, 3)} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    args = parse_args()
    if args.extract_zips:
        extract_zips(args.dataset_root)

    sequences = find_sequences(args.dataset_root, args.split)
    if args.list_sequences:
        for sequence in sequences:
            print(sequence)
        return 0

    if not sequences:
        zip_hint = sorted(args.dataset_root.glob("*.zip"))
        hint = " Try --extract-zips first." if zip_hint else ""
        raise RuntimeError(f"No extracted sequences with img1 folders found under {args.dataset_root}.{hint}")

    if args.sequence:
        wanted = set(args.sequence)
        sequences = [sequence for sequence in sequences if sequence.name in wanted]
        missing = wanted - {sequence.name for sequence in sequences}
        if missing:
            raise RuntimeError(f"Requested sequences not found: {sorted(missing)}")

    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    gui_v3 = load_gui_v3_module()
    compare_configs = normalized_compare_configs(args.compare_configs)
    summaries: List[RunSummary] = []

    if compare_configs:
        if args.weight_path:
            raise RuntimeError("--weight-path cannot be combined with --compare-configs.")
        csv_path = None
        md_path = None
        for lorat_config in compare_configs:
            run_args = argparse.Namespace(**vars(args))
            run_args.backend = "lorat"
            run_args.lorat_config = lorat_config
            run_args.output_by_config = True
            for sequence_path in sequences:
                summaries.append(run_sequence(gui_v3, run_args, sequence_path))
                csv_path, md_path = write_comparison_outputs(args, summaries)
        if csv_path is not None and md_path is not None:
            print("Wrote comparison files:")
            print(f"  {csv_path}")
            print(f"  {md_path}")
    else:
        args.output_by_config = False
        for sequence_path in sequences:
            summaries.append(run_sequence(gui_v3, args, sequence_path))

    print("Wrote result files:")
    for summary in summaries:
        print(f"  {summary.result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
