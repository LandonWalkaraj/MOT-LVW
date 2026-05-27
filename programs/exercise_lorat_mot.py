from __future__ import annotations

import argparse
import configparser
import importlib.util
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_V3_PATH = PROJECT_ROOT / "programs" / "bounding box v3.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lorat-exercise"


@dataclass(frozen=True)
class GroundTruthRow:
    frame: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    visibility: float


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
        description="Exercise LoRAT on DanceTrack/MOT17 MOTChallenge-style sequences."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder for DanceTrack or MOT17.")
    parser.add_argument("--dataset", choices=("dancetrack", "mot17"), required=True)
    parser.add_argument("--split", default="val", help="Dataset split folder to scan, for example train/val/test.")
    parser.add_argument("--sequence", action="append", help="Sequence name to run. Repeat for multiple sequences.")
    parser.add_argument("--list-sequences", action="store_true", help="List detected sequences and exit.")
    parser.add_argument("--extract-zips", action="store_true", help="Extract *.zip files found under --dataset-root.")
    parser.add_argument("--backend", choices=("lorat", "opencv"), default="lorat")
    parser.add_argument("--device", default="cpu", help="LoRAT device, e.g. cpu or cuda:0.")
    parser.add_argument("--lorat-root", type=Path, default=PROJECT_ROOT / "external" / "LoRAT-main")
    parser.add_argument("--lorat-config", default="B-224", choices=("B-224", "B-378", "L-224", "L-378", "g-224", "g-378"))
    parser.add_argument("--weight-path", type=Path, help="Optional LoRAT weight override.")
    parser.add_argument("--max-tracks", type=int, default=8, help="Maximum tracks initialized per sequence.")
    parser.add_argument("--max-frames", type=int, default=150, help="Frames to process per sequence; 0 means full sequence.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Limit number of sequences; 0 means all selected.")
    parser.add_argument("--init-frame", default="auto", help="1-based frame number or auto.")
    parser.add_argument("--class-id", type=int, action="append", help="GT class IDs to initialize. Defaults to pedestrian=1.")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-video", action="store_true", help="Save annotated MP4 previews.")
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--disable-amp", action="store_true")
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


def find_sequences(dataset_root: Path, dataset: str, split: str) -> List[Path]:
    search_roots = [dataset_root]
    split_root = dataset_root / split
    if split_root.is_dir():
        search_roots.insert(0, split_root)

    sequences = []
    for root in search_roots:
        for child in sorted(root.iterdir() if root.exists() else []):
            if child.is_dir() and (child / "img1").is_dir():
                sequences.append(child)

    deduped = {}
    for sequence in sequences:
        deduped[sequence.name] = sequence
    return [deduped[name] for name in sorted(deduped)]


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

    for frame in sorted(gt_by_frame):
        rows = usable_rows(frame)
        if rows:
            return frame, rows
    raise RuntimeError("No usable GT boxes found in sequence.")


def frame_to_image_index(frame_number: int) -> int:
    return frame_number - 1


def build_backend(gui_v3, args: argparse.Namespace, sequence_name: str, fps: float, length: Optional[int]):
    if args.backend == "opencv":
        return gui_v3.OpenCVMultiObjectTracker()

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
    )


def run_sequence(gui_v3, args: argparse.Namespace, sequence_path: Path) -> Path:
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
    )
    init_index = frame_to_image_index(init_frame)
    if init_index >= len(image_paths):
        raise RuntimeError(f"Init frame {init_frame} is outside image sequence length {len(image_paths)}.")

    output_dir = args.output_root / args.dataset / args.split / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{sequence_path.name}.txt"

    preview_writer = None
    preview_path = output_dir / f"{sequence_path.name}.mp4"

    init_frame_image = cv2.imread(str(image_paths[init_index]))
    if init_frame_image is None:
        raise RuntimeError(f"Unable to read frame: {image_paths[init_index]}")

    backend = build_backend(gui_v3, args, sequence_path.name, fps, sequence_length or len(image_paths))
    mot_lines: List[str] = []
    boxes = [row.bbox for row in init_rows]

    end_index = len(image_paths) - 1
    if args.max_frames > 0:
        end_index = min(end_index, init_index + args.max_frames - 1)

    try:
        backend.initialize(init_frame_image, boxes, init_frame)
        gui_v3.append_mot_results(mot_lines, init_frame, backend.tracks)

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
            if preview_writer is not None:
                preview_writer.write(gui_v3.draw_tracks(frame, backend.tracks, frame_number, backend.backend_name))
    finally:
        backend.close()
        if preview_writer is not None:
            preview_writer.release()

    result_path.write_text("".join(mot_lines), encoding="utf-8")
    print(
        f"{sequence_path.name}: init_frame={init_frame}, tracks={len(init_rows)}, "
        f"frames={end_index - init_index + 1}, result={result_path}"
    )
    return result_path


def main() -> int:
    args = parse_args()
    if args.extract_zips:
        extract_zips(args.dataset_root)

    sequences = find_sequences(args.dataset_root, args.dataset, args.split)
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
    result_paths = []
    for sequence_path in sequences:
        result_paths.append(run_sequence(gui_v3, args, sequence_path))

    print("Wrote result files:")
    for result_path in result_paths:
        print(f"  {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
