from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
RotatedRect = Tuple[Tuple[float, float], Tuple[float, float], float]
VideoSource = Union[int, str]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gui-tracks"
DEFAULT_LORAT_ROOT = PROJECT_ROOT / "external" / "LoRAT-main"
DEFAULT_LORAT_WEIGHT = PROJECT_ROOT / "models" / "lorat" / "base.bin"


@dataclass
class TrackState:
    track_id: int
    tracker: Optional[object]
    bbox: BBox
    color: Color
    ok: bool = True
    lost_frames: int = 0
    rotated_rect: Optional[RotatedRect] = None
    roi_hist: Optional[np.ndarray] = None
    tracking_window: Optional[Tuple[int, int, int, int]] = None


class FrameSource:
    name: str
    fps: float

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class VideoCaptureSource(FrameSource):
    def __init__(self, source: VideoSource):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.name = str(source)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


class ImageSequenceSource(FrameSource):
    def __init__(self, sequence_path: Path, fps: float):
        image_dir = sequence_path / "img1" if (sequence_path / "img1").is_dir() else sequence_path
        image_paths = sorted(
            path
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in image_dir.glob(suffix)
        )
        if not image_paths:
            raise RuntimeError(f"No image frames found in: {image_dir}")

        self.image_paths = image_paths
        self.index = 0
        self.fps = fps
        self.name = sequence_path.name

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= len(self.image_paths):
            return False, None

        frame = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1
        return frame is not None, frame


class OpenCVMultiObjectTracker:
    def __init__(self, tracker_name: str):
        self.tracker_name = tracker_name.upper()
        self.tracks: List[TrackState] = []
        self.next_track_id = 1

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox]) -> None:
        self.add_tracks(frame, boxes)

    def add_tracks(self, frame: np.ndarray, boxes: Sequence[BBox]) -> List[TrackState]:
        added_tracks = []
        for bbox in boxes:
            init_bbox = tuple(int(round(value)) for value in bbox)
            tracker = create_opencv_tracker(self.tracker_name)
            initialized = tracker.init(frame, init_bbox)
            if initialized is False:
                print(f"Skipping box that OpenCV could not initialize: {init_bbox}")
                continue

            track = TrackState(
                track_id=self.next_track_id,
                tracker=tracker,
                bbox=tuple(float(value) for value in init_bbox),
                color=color_for_track(self.next_track_id),
            )
            self.tracks.append(track)
            added_tracks.append(track)
            self.next_track_id += 1
        return added_tracks

    def update(self, frame: np.ndarray) -> Sequence[TrackState]:
        for track in self.tracks:
            if not track.ok:
                track.lost_frames += 1
                continue
            if track.tracker is None:
                track.ok = False
                track.lost_frames += 1
                continue

            ok, bbox = track.tracker.update(frame)
            track.ok = bool(ok)
            if ok:
                track.bbox = tuple(float(value) for value in bbox)
                track.lost_frames = 0
            else:
                track.lost_frames += 1

        return self.tracks


class CamShiftMultiObjectTracker:
    def __init__(self):
        self.tracks: List[TrackState] = []
        self.next_track_id = 1
        self.term_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            12,
            1,
        )

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox]) -> None:
        self.add_tracks(frame, boxes)

    def add_tracks(self, frame: np.ndarray, boxes: Sequence[BBox]) -> List[TrackState]:
        added_tracks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for bbox in boxes:
            window = clip_bbox_to_frame(frame, bbox)
            if window is None:
                continue

            x, y, w, h = window
            roi = hsv[y : y + h, x : x + w]
            mask = cv2.inRange(roi, np.array((0, 10, 20)), np.array((180, 255, 255)))
            if cv2.countNonZero(mask) == 0:
                mask = None

            roi_hist = cv2.calcHist([roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            rotated_rect = ((x + w / 2.0, y + h / 2.0), (float(w), float(h)), 0.0)
            track = TrackState(
                track_id=self.next_track_id,
                tracker=None,
                bbox=tuple(float(value) for value in window),
                color=color_for_track(self.next_track_id),
                rotated_rect=rotated_rect,
                roi_hist=roi_hist,
                tracking_window=window,
            )
            self.tracks.append(track)
            added_tracks.append(track)
            self.next_track_id += 1

        return added_tracks

    def update(self, frame: np.ndarray) -> Sequence[TrackState]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mask = cv2.inRange(hsv, np.array((0, 10, 20)), np.array((180, 255, 255)))

        for track in self.tracks:
            if track.roi_hist is None or track.tracking_window is None:
                track.ok = False
                track.lost_frames += 1
                continue

            back_project = cv2.calcBackProject([hsv], [0, 1], track.roi_hist, [0, 180, 0, 256], 1)
            back_project = cv2.bitwise_and(back_project, back_project, mask=saturation_mask)
            rotated_rect, window = cv2.CamShift(back_project, track.tracking_window, self.term_criteria)
            clipped_window = clip_bbox_to_frame(frame, window)

            if clipped_window is None:
                track.ok = False
                track.lost_frames += 1
                continue

            track.ok = True
            track.lost_frames = 0
            track.tracking_window = clipped_window
            track.bbox = tuple(float(value) for value in clipped_window)
            track.rotated_rect = (
                (float(rotated_rect[0][0]), float(rotated_rect[0][1])),
                (float(rotated_rect[1][0]), float(rotated_rect[1][1])),
                float(rotated_rect[2]),
            )

        return self.tracks


class LoRATMultiObjectTracker:
    def __init__(self, lorat_root: Path, weight_path: Path, device: str):
        self.lorat_root = lorat_root
        self.weight_path = weight_path
        self.device = device
        self._check_environment()

        raise NotImplementedError(
            "The LoRAT GUI backend is not wired yet. The official LoRAT checkout in this "
            "workspace exposes a training/evaluation pipeline, while this script exposes "
            "the interactive multi-object GUI and MOT-format result writer. Use "
            "--backend opencv now, then wrap LoRAT's OneStreamTrackerPipeline behind this "
            "class for the CUDA tracker backend."
        )

    def _check_environment(self) -> None:
        if not self.lorat_root.exists():
            raise RuntimeError(f"LoRAT repo not found: {self.lorat_root}")
        if not self.weight_path.exists():
            raise RuntimeError(f"LoRAT weight not found: {self.weight_path}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed in this Python environment. Run "
                "scripts/setup-lorat-env.ps1 first."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false in this "
                "environment. Use --device cpu for debugging or run on an NVIDIA/CUDA setup."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version 2 GUI for user-seeded multi-object tracking."
    )
    parser.add_argument(
        "--video",
        default="0",
        help="Path to a video file or camera index. Ignored when --sequence is supplied.",
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        help="Path to a DanceTrack/MOT17-style sequence folder, usually one containing img1.",
    )
    parser.add_argument(
        "--sequence-fps",
        type=float,
        default=30.0,
        help="Playback FPS for image sequence inputs.",
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "lorat"),
        default="opencv",
        help="Tracking backend. OpenCV is runnable now; LoRAT is the CUDA adapter target.",
    )
    parser.add_argument(
        "--opencv-tracker",
        choices=("csrt", "kcf", "mil", "camshift"),
        default="camshift",
        help="OpenCV tracker type. CamShift can resize and rotate boxes, but is color-sensitive.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device string for the future LoRAT backend, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--lorat-root",
        type=Path,
        default=DEFAULT_LORAT_ROOT,
        help="Path to the local LoRAT checkout.",
    )
    parser.add_argument(
        "--weight-path",
        type=Path,
        default=DEFAULT_LORAT_WEIGHT,
        help="Path to the LoRAT model weight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="MOTChallenge-format result file. Defaults to outputs/gui-tracks/<source>.txt.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Optional path for an annotated output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick DanceTrack/MOT17 smoke tests.",
    )
    return parser.parse_args()


def create_opencv_tracker(tracker_name: str):
    candidates = [
        (cv2, f"Tracker{tracker_name}_create"),
        (getattr(cv2, "legacy", None), f"Tracker{tracker_name}_create"),
    ]

    for module, factory_name in candidates:
        if module is not None and hasattr(module, factory_name):
            return getattr(module, factory_name)()

    tracker_class = getattr(cv2, f"Tracker{tracker_name}", None)
    if tracker_class is not None and hasattr(tracker_class, "create"):
        return tracker_class.create()

    raise RuntimeError(
        f"OpenCV tracker {tracker_name} is unavailable. Install opencv-contrib-python "
        "in the selected Python environment."
    )


def clip_bbox_to_frame(frame: np.ndarray, bbox: BBox) -> Optional[Tuple[int, int, int, int]]:
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = [int(round(value)) for value in bbox]

    left = max(0, min(x, frame_width - 1))
    top = max(0, min(y, frame_height - 1))
    right = max(left + 1, min(x + w, frame_width))
    bottom = max(top + 1, min(y + h, frame_height))
    width = right - left
    height = bottom - top

    if width <= 2 or height <= 2:
        return None

    return left, top, width, height


def color_for_track(track_id: int) -> Color:
    palette = (
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 0, 128),
    )
    return palette[(track_id - 1) % len(palette)]


def parse_video_source(value: str) -> VideoSource:
    try:
        return int(value)
    except ValueError:
        return value


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    if args.sequence is not None:
        return ImageSequenceSource(args.sequence, args.sequence_fps)
    return VideoCaptureSource(parse_video_source(args.video))


def select_boxes(frame: np.ndarray, title: str = "Select Objects") -> List[BBox]:
    print("Drag boxes with the mouse. ENTER/SPACE = start tracking. c/q/ESC = cancel. r = reset boxes.")

    boxes: List[BBox] = []
    drawing = False
    start_point: Optional[Tuple[int, int]] = None
    current_point: Optional[Tuple[int, int]] = None

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal drawing, start_point, current_point

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing and start_point is not None:
            drawing = False
            current_point = (x, y)
            x1, y1 = start_point
            x2, y2 = x, y
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            width = right - left
            height = bottom - top
            if width > 2 and height > 2:
                boxes.append((float(left), float(top), float(width), float(height)))
            start_point = None
            current_point = None

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_callback)

    while True:
        preview = frame.copy()
        for index, bbox in enumerate(boxes, start=1):
            x, y, w, h = [int(round(value)) for value in bbox]
            color = color_for_track(index)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                preview,
                f"Box {index}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if drawing and start_point is not None and current_point is not None:
            cv2.rectangle(preview, start_point, current_point, (0, 255, 255), 2)

        cv2.putText(
            preview,
            f"{len(boxes)} boxes | ENTER/SPACE done | c cancel | r reset",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"{len(boxes)} boxes | ENTER/SPACE done | c cancel | r reset",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(title, preview)
        key = cv2.waitKey(16) & 0xFF

        if key in (13, 32):
            cv2.destroyWindow(title)
            return boxes
        if key in (27, ord("c"), ord("q")):
            cv2.destroyWindow(title)
            return []
        if key == ord("r"):
            boxes.clear()

        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return []
        except cv2.error:
            return []


def default_output_path(source_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_OUTPUT_DIR / f"{safe_name}_tracks.txt"


def make_video_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def append_mot_results(lines: List[str], frame_number: int, tracks: Sequence[TrackState]) -> None:
    for track in tracks:
        if not track.ok:
            continue

        x, y, w, h = track.bbox
        lines.append(
            f"{frame_number},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n"
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    frame_number: int,
    backend_label: str,
) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        f"Frame {frame_number} | {backend_label} | q quit | a add boxes | p pause",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Frame {frame_number} | {backend_label} | q quit | a add boxes | p pause",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    for track in tracks:
        x, y, w, h = [int(round(value)) for value in track.bbox]
        color = track.color if track.ok else (0, 0, 255)
        label = f"ID {track.track_id}" if track.ok else f"ID {track.track_id} LOST"

        if track.rotated_rect is not None and track.ok:
            points = cv2.boxPoints(track.rotated_rect)
            points = np.intp(points)
            cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)
            label_x = int(points[:, 0].min())
            label_y = int(points[:, 1].min())
        else:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label_x = x
            label_y = y

        cv2.putText(
            output,
            label,
            (label_x, max(20, label_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def build_backend(args: argparse.Namespace):
    if args.backend == "lorat":
        return LoRATMultiObjectTracker(args.lorat_root, args.weight_path, args.device)

    if args.opencv_tracker == "camshift":
        return CamShiftMultiObjectTracker()

    return OpenCVMultiObjectTracker(args.opencv_tracker)


def main() -> int:
    args = parse_args()
    frame_source = open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok, first_frame = frame_source.read()
    if not ok or first_frame is None:
        print("Unable to read the first frame.")
        return 1

    boxes = select_boxes(first_frame)
    if not boxes:
        print("No bounding boxes selected. Exiting.")
        frame_source.release()
        cv2.destroyAllWindows()
        return 0

    tracker_backend = build_backend(args)
    tracker_backend.initialize(first_frame, boxes)

    writer = make_video_writer(args.save_video, frame_source.fps, first_frame) if args.save_video else None
    mot_lines: List[str] = []
    frame_number = 1
    append_mot_results(mot_lines, frame_number, tracker_backend.tracks)

    paused = False
    backend_label = args.backend
    if args.backend == "opencv":
        backend_label = f"OpenCV {args.opencv_tracker.upper()}"
        if args.opencv_tracker == "camshift":
            print("Using OpenCV CamShift: boxes can resize and rotate when the color model separates the object.")
        else:
            print(f"Using OpenCV {args.opencv_tracker.upper()}: boxes are axis-aligned and will not rotate.")
    else:
        print(f"Using backend: {backend_label}")

    while True:
        if not paused:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                break

            frame_number += 1
            tracker_backend.update(frame)
            append_mot_results(mot_lines, frame_number, tracker_backend.tracks)
        else:
            frame = frame.copy()

        shown = draw_tracks(frame, tracker_backend.tracks, frame_number, backend_label)
        if writer is not None and not paused:
            writer.write(shown)

        cv2.imshow("Multi-Object Bounding Box Tracker v2", shown)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("a"):
            paused = True
            new_boxes = select_boxes(frame, "Add Objects")
            if new_boxes:
                added_tracks = tracker_backend.add_tracks(frame, new_boxes)
                append_mot_results(mot_lines, frame_number, added_tracks)
            paused = False

        if args.max_frames > 0 and frame_number >= args.max_frames:
            break

    output_path.write_text("".join(mot_lines), encoding="utf-8")
    print(f"Wrote MOTChallenge-format tracks to: {output_path}")

    if writer is not None:
        writer.release()
        print(f"Wrote annotated video to: {args.save_video}")

    frame_source.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
