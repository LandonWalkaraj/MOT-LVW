from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
VideoSource = Union[int, str]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lorat-gui"
DEFAULT_LORAT_ROOT = PROJECT_ROOT / "external" / "LoRAT-main"

LORAT_WEIGHT_BY_CONFIG = {
    "B-224": PROJECT_ROOT / "models" / "lorat" / "base.bin",
    "B-378": PROJECT_ROOT / "models" / "lorat" / "base-378.bin",
    "L-224": PROJECT_ROOT / "models" / "lorat" / "large.bin",
    "L-378": PROJECT_ROOT / "models" / "lorat" / "large-378.bin",
    "g-224": PROJECT_ROOT / "models" / "lorat" / "giant.bin",
    "g-378": PROJECT_ROOT / "models" / "lorat" / "giant-378.bin",
}


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    color: Color
    ok: bool = True
    confidence: Optional[float] = None
    lost_frames: int = 0
    tracker: Optional[object] = None
    previous_bbox: Optional[BBox] = None
    predicted_bbox: Optional[BBox] = None
    raw_bbox: Optional[BBox] = None
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)
    occluded_frames: int = 0
    coordinator_state: str = ""


class MultiObjectCoordinator:
    def __init__(
        self,
        enabled: bool = True,
        overlap_iou_threshold: float = 0.30,
        normal_proposal_weight: float = 0.90,
        occluded_proposal_weight: float = 0.45,
        suspicious_proposal_weight: float = 0.25,
        max_center_jump: float = 0.45,
        max_scale_change: float = 0.65,
        max_occlusion_frames: int = 20,
        velocity_smoothing: float = 0.70,
    ):
        self.enabled = enabled
        self.overlap_iou_threshold = overlap_iou_threshold
        self.normal_proposal_weight = normal_proposal_weight
        self.occluded_proposal_weight = occluded_proposal_weight
        self.suspicious_proposal_weight = suspicious_proposal_weight
        self.max_center_jump = max_center_jump
        self.max_scale_change = max_scale_change
        self.max_occlusion_frames = max_occlusion_frames
        self.velocity_smoothing = velocity_smoothing

    def resolve(
        self,
        tracks: Sequence[TrackState],
        proposals: Dict[int, Tuple[BBox, Optional[float]]],
    ) -> Dict[int, Tuple[BBox, Optional[float], str]]:
        if not self.enabled:
            return {
                track_id: (proposal, confidence, "")
                for track_id, (proposal, confidence) in proposals.items()
            }

        track_by_id = {track.track_id: track for track in tracks}
        conflict_ids = self._find_conflicts(track_by_id, proposals)
        updates: Dict[int, Tuple[BBox, Optional[float], str]] = {}

        for track_id, (proposal, confidence) in proposals.items():
            track = track_by_id.get(track_id)
            if track is None:
                continue

            previous = track.bbox
            predicted = predict_bbox(previous, track.velocity)
            suspicious = self._is_suspicious(previous, predicted, proposal)
            in_conflict = track_id in conflict_ids

            if in_conflict:
                proposal_weight = self.occluded_proposal_weight
                state = "OCC"
            else:
                proposal_weight = self.normal_proposal_weight
                state = ""

            if suspicious:
                proposal_weight = min(proposal_weight, self.suspicious_proposal_weight)
                state = "SMOOTH" if not state else state

            if track.occluded_frames >= self.max_occlusion_frames:
                proposal_weight = max(proposal_weight, self.normal_proposal_weight)
                state = ""

            final_bbox = clamp_bbox_size(blend_bbox(predicted, proposal, proposal_weight))
            self._update_track_motion(track, previous, final_bbox, predicted, proposal, in_conflict, state)
            updates[track_id] = (final_bbox, confidence, state)

        return updates

    def _find_conflicts(
        self,
        track_by_id: Dict[int, TrackState],
        proposals: Dict[int, Tuple[BBox, Optional[float]]],
    ) -> set[int]:
        conflict_ids: set[int] = set()
        proposal_items = list(proposals.items())

        for left_index, (left_id, (left_bbox, _)) in enumerate(proposal_items):
            for right_id, (right_bbox, _) in proposal_items[left_index + 1 :]:
                left_track = track_by_id.get(left_id)
                right_track = track_by_id.get(right_id)
                previous_iou = 0.0
                if left_track is not None and right_track is not None:
                    previous_iou = bbox_iou(left_track.bbox, right_track.bbox)

                if max(bbox_iou(left_bbox, right_bbox), previous_iou) >= self.overlap_iou_threshold:
                    conflict_ids.add(left_id)
                    conflict_ids.add(right_id)

        return conflict_ids

    def _is_suspicious(self, previous: BBox, predicted: BBox, proposal: BBox) -> bool:
        previous_diag = max(1.0, bbox_diagonal(previous))
        center_jump = center_distance(predicted, proposal) / previous_diag

        previous_area = max(1.0, previous[2] * previous[3])
        proposal_area = max(1.0, proposal[2] * proposal[3])
        scale_change = abs(proposal_area - previous_area) / previous_area

        return center_jump > self.max_center_jump or scale_change > self.max_scale_change

    def _update_track_motion(
        self,
        track: TrackState,
        previous: BBox,
        final_bbox: BBox,
        predicted_bbox: BBox,
        raw_bbox: BBox,
        in_conflict: bool,
        state: str,
    ) -> None:
        measured_velocity = bbox_delta(previous, final_bbox)
        track.velocity = blend_bbox(measured_velocity, track.velocity, self.velocity_smoothing)
        track.previous_bbox = previous
        track.predicted_bbox = predicted_bbox
        track.raw_bbox = raw_bbox
        track.occluded_frames = track.occluded_frames + 1 if in_conflict else 0
        track.coordinator_state = state


class FrameSource:
    name: str
    fps: float
    length: Optional[int]

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
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps if fps and fps > 1 else 30.0
        self.length = frame_count if frame_count > 0 else None
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

        self.sequence_path = sequence_path
        self.image_paths = image_paths
        self.index = 0
        self.fps = fps
        self.length = len(image_paths)
        self.name = sequence_path.name

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= len(self.image_paths):
            return False, None

        frame = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1
        return frame is not None, frame


class LoRATMultiObjectTracker:
    backend_name = "LoRAT"

    def __init__(
        self,
        lorat_root: Path,
        config_name: str,
        weight_path: Path,
        device: str,
        max_tracks: int,
        fps: Optional[float],
        sequence_length: Optional[int],
        sequence_name: str,
        confidence_threshold: float,
        disable_amp: bool,
        coordinator_enabled: bool = True,
        overlap_iou_threshold: float = 0.30,
        normal_proposal_weight: float = 0.90,
        occluded_proposal_weight: float = 0.45,
        suspicious_proposal_weight: float = 0.25,
        max_center_jump: float = 0.45,
        max_scale_change: float = 0.65,
        max_occlusion_frames: int = 20,
    ):
        self.lorat_root = lorat_root.resolve()
        self.config_name = config_name
        self.weight_path = weight_path.resolve()
        self.device_string = device
        self.max_tracks = max_tracks
        self.fps = fps
        self.sequence_length = sequence_length
        self.sequence_name = sequence_name
        self.confidence_threshold = confidence_threshold
        self.disable_amp = disable_amp
        self.coordinator = MultiObjectCoordinator(
            enabled=coordinator_enabled,
            overlap_iou_threshold=overlap_iou_threshold,
            normal_proposal_weight=normal_proposal_weight,
            occluded_proposal_weight=occluded_proposal_weight,
            suspicious_proposal_weight=suspicious_proposal_weight,
            max_center_jump=max_center_jump,
            max_scale_change=max_scale_change,
            max_occlusion_frames=max_occlusion_frames,
        )
        self.tracks: List[TrackState] = []
        self.track_by_id: Dict[int, TrackState] = {}
        self.next_track_id = 1
        self.closed = False

        self._load_lorat()
        self._build_runtime()

    def _load_lorat(self) -> None:
        if not self.lorat_root.exists():
            raise RuntimeError(f"LoRAT checkout not found: {self.lorat_root}")
        if not self.weight_path.exists():
            raise RuntimeError(f"LoRAT weight not found: {self.weight_path}")

        lorat_root_str = str(self.lorat_root)
        if lorat_root_str not in sys.path:
            sys.path.insert(0, lorat_root_str)

        import torch

        self.torch = torch
        self.device = torch.device(self.device_string)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LoRAT was asked to use cuda, but this PyTorch build reports no CUDA/HIP device. "
                "Use --device cpu on this laptop, or install a working CUDA/ROCm PyTorch build."
            )

    def _build_runtime(self) -> None:
        try:
            from trackit.core.boot.funcs.main.load_config import load_config
            from trackit.core.runtime.global_constant import get_global_constant
            from trackit.data.methods.siamese_tracker_eval.transform.builder import build_data_transform
            from trackit.models import ModelManager
            from trackit.models.compiling.plain.builder import build_plain_inference_engine
            from trackit.models.methods.builder import create_model_build_context
            from trackit.runners.evaluation.distributed.tracker_evaluator import EvaluatorContext
            from trackit.runners.evaluation.distributed.tracker_evaluator.default.evaluator import (
                DefaultTrackerEvaluator,
            )
            from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.builder import (
                build_tracker_evaluator_pipeline,
            )
        except ModuleNotFoundError as exc:
            package_name = exc.name or "unknown"
            raise RuntimeError(
                f"LoRAT dependency '{package_name}' is missing from this Python interpreter: "
                f"{sys.executable}. In VS Code, select the project interpreter at "
                f"{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'}, or run "
                "scripts/setup-lorat-env.ps1."
            ) from exc

        if get_global_constant("TIMM_USE_OLD_CACHE", default=True):
            os.environ["TIMM_USE_OLD_CACHE"] = "1"

        runtime_vars = SimpleNamespace(
            root_path=str(self.lorat_root),
            config_path=str(self.lorat_root / "config"),
            method_name="LoRAT",
            config_name=self.config_name,
            mixin_config=None,
        )
        config = load_config(runtime_vars)
        self.config = config

        self.dtype = self.torch.float32
        transform_config = config["run"]["data"]["eval"]["transform"]
        self.transform = build_data_transform(transform_config, config, self.device, self.dtype)

        model_manager = ModelManager(create_model_build_context(config), rng_fixed_seed=42)
        model_manager.load_state_dict_from_file(str(self.weight_path), strict=False, print_missing=False)
        self.model_manager = model_manager

        inference_config = copy.deepcopy(config["run"]["runner"]["test"]["inference_engine"])
        if self.device.type == "cpu" or self.disable_amp:
            inference_config["auto_mixed_precision"]["enabled"] = False
        inference_config["torch_compile"]["enabled"] = False
        inference_engine = build_plain_inference_engine(inference_config, self.device)

        self.optimized_model = inference_engine(
            model_manager,
            self.device,
            self.dtype,
            self.max_tracks,
            1,
        )

        pipeline_config = config["run"]["runner"]["test"]["evaluator"]["pipeline"]
        pipeline = build_tracker_evaluator_pipeline(
            pipeline_config,
            config,
            self.device,
            config["run"]["num_epochs"],
        )
        self.evaluator = DefaultTrackerEvaluator(pipeline)
        self.evaluator_context = EvaluatorContext(
            epoch=0,
            max_batch_size=self.max_tracks,
            num_input_data_streams=1,
            dtype=self.dtype,
            auto_mixed_precision_dtype=self.optimized_model.auto_mixed_precision_dtype,
            model=self.optimized_model.raw_model,
        )
        self.evaluator.start(self.evaluator_context)
        print(
            f"Loaded LoRAT {self.config_name} on {self.device} with weight {self.weight_path.name}. "
            f"Max simultaneous tracks: {self.max_tracks}"
        )

    def initialize(self, frame: np.ndarray, boxes: Sequence[BBox], frame_number: int = 1) -> None:
        self.add_tracks(frame, boxes, frame_number)

    def add_tracks(
        self,
        frame: np.ndarray,
        boxes: Sequence[BBox],
        frame_number: int = 1,
    ) -> List[TrackState]:
        if len(self.tracks) + len(boxes) > self.max_tracks:
            raise RuntimeError(
                f"Too many tracks selected. Current={len(self.tracks)}, requested={len(boxes)}, "
                f"max={self.max_tracks}. Increase --max-tracks."
            )

        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )
        from trackit.data.protocol import SequenceInfo

        added_tracks = []
        tasks = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for bbox in boxes:
            clipped = clip_bbox_to_frame(frame, bbox)
            if clipped is None:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            track = TrackState(
                track_id=track_id,
                bbox=tuple(float(value) for value in clipped),
                color=color_for_track(track_id),
                confidence=1.0,
                previous_bbox=tuple(float(value) for value in clipped),
                predicted_bbox=tuple(float(value) for value in clipped),
                raw_bbox=tuple(float(value) for value in clipped),
            )
            self.tracks.append(track)
            self.track_by_id[track_id] = track
            added_tracks.append(track)

            bbox_xyxy = xywh_to_xyxy_np(clipped)
            init_context = SiameseTrackerEvalDataWorker_FrameContext(
                frame_number,
                lambda image=rgb_frame.copy(): image,
                bbox_xyxy,
                None,
            )
            sequence_info = SequenceInfo(
                dataset_name="user",
                data_split=None,
                dataset_full_name=None,
                sequence_name=f"{self.sequence_name}-track-{track_id}",
                length=self.sequence_length,
                fps=self.fps,
            )
            tasks.append(
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track_id,
                    do_task_creation=sequence_info,
                    do_tracker_init=init_context,
                    do_tracker_track=init_context,
                    do_task_finalization=False,
                )
            )

        if tasks:
            outputs = self._run_worker_tasks(tasks)
            self._apply_evaluated_frames(outputs)
        return added_tracks

    def update(self, frame: np.ndarray, frame_number: int) -> Sequence[TrackState]:
        from trackit.data.methods.siamese_tracker_eval import (
            SiameseTrackerEvalDataWorker_FrameContext,
            SiameseTrackerEvalDataWorker_Task,
        )

        active_tracks = [track for track in self.tracks if track.ok]
        if not active_tracks:
            return self.tracks

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tasks = []
        for track in active_tracks:
            track_context = SiameseTrackerEvalDataWorker_FrameContext(
                frame_number,
                lambda image=rgb_frame.copy(): image,
                None,
                None,
            )
            tasks.append(
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track.track_id,
                    do_task_creation=None,
                    do_tracker_init=None,
                    do_tracker_track=track_context,
                    do_task_finalization=False,
                )
            )

        outputs = self._run_worker_tasks(tasks)
        self._apply_evaluated_frames(outputs)
        return self.tracks

    def _apply_evaluated_frames(self, outputs: Optional[dict]) -> None:
        evaluated_frames = outputs.get("evaluated_frames", []) if outputs is not None else []
        proposals: Dict[int, Tuple[BBox, Optional[float]]] = {}
        failed_track_ids = set()

        for result in evaluated_frames:
            track = self.track_by_id.get(result.id)
            if track is None:
                continue

            if result.output_box is None:
                failed_track_ids.add(track.track_id)
                continue

            confidence = float(result.output_confidence) if result.output_confidence is not None else None
            if confidence is not None and confidence < self.confidence_threshold:
                track.confidence = confidence
                failed_track_ids.add(track.track_id)
                continue

            proposals[track.track_id] = (xyxy_to_xywh_tuple(result.output_box), confidence)

        for track_id in failed_track_ids - set(proposals):
            track = self.track_by_id.get(track_id)
            if track is None:
                continue
            track.ok = False
            track.lost_frames += 1
            track.coordinator_state = "LOST"

        resolved_updates = self.coordinator.resolve(self.tracks, proposals)
        for track_id, (bbox, confidence, state) in resolved_updates.items():
            track = self.track_by_id.get(track_id)
            if track is None:
                continue

            track.bbox = bbox
            track.confidence = confidence
            track.ok = True
            track.lost_frames = 0
            track.coordinator_state = state

    def _run_worker_tasks(self, worker_tasks: Sequence[object]):
        from trackit.data.protocol.eval_input import TrackerEvalData

        transformed_tasks = tuple(self.transform(task) for task in worker_tasks)
        data = TrackerEvalData(transformed_tasks, {})
        return self.evaluator.run(
            data,
            self.optimized_model.model,
            self.optimized_model.raw_model,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.tracks:
            from trackit.data.methods.siamese_tracker_eval import SiameseTrackerEvalDataWorker_Task

            tasks = [
                SiameseTrackerEvalDataWorker_Task(
                    task_index=track.track_id,
                    do_task_creation=None,
                    do_tracker_init=None,
                    do_tracker_track=None,
                    do_task_finalization=True,
                )
                for track in self.tracks
            ]
            self._run_worker_tasks(tasks)

        self.evaluator.stop(self.evaluator_context)
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version 3 LoRAT-backed multi-object bounding-box GUI."
    )
    parser.add_argument("--video", default="0", help="Path to a video file or camera index.")
    parser.add_argument(
        "--sequence",
        type=Path,
        help="Path to a DanceTrack/MOT17-style sequence folder, usually one containing img1.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root containing DanceTrack or MOT17 sequences. Use with --dataset and --sequence-name.",
    )
    parser.add_argument("--dataset", choices=("dancetrack", "mot17"), help="Dataset layout for --dataset-root.")
    parser.add_argument("--sequence-name", help="Sequence folder name to run from --dataset-root.")
    parser.add_argument("--list-sequences", action="store_true", help="List resolved dataset sequences and exit.")
    parser.add_argument("--sequence-fps", type=float, default=30.0, help="Playback FPS for image sequence inputs.")
    parser.add_argument("--device", default="cpu", help="LoRAT device: cpu, cuda:0, etc.")
    parser.add_argument("--lorat-root", type=Path, default=DEFAULT_LORAT_ROOT, help="Local LoRAT checkout.")
    parser.add_argument("--lorat-config", default="B-224", choices=tuple(LORAT_WEIGHT_BY_CONFIG))
    parser.add_argument("--weight-path", type=Path, help="LoRAT weight. Defaults from --lorat-config.")
    parser.add_argument("--max-tracks", type=int, default=8, help="Maximum simultaneous LoRAT tracks.")
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--disable-amp", action="store_true", help="Disable LoRAT automatic mixed precision.")
    parser.add_argument("--disable-coordinator", action="store_true", help="Disable overlap/motion coordination.")
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.30)
    parser.add_argument("--normal-proposal-weight", type=float, default=0.90)
    parser.add_argument("--occluded-proposal-weight", type=float, default=0.45)
    parser.add_argument("--suspicious-proposal-weight", type=float, default=0.25)
    parser.add_argument("--max-center-jump", type=float, default=0.45)
    parser.add_argument("--max-scale-change", type=float, default=0.65)
    parser.add_argument("--max-occlusion-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, help="MOTChallenge-format result file.")
    parser.add_argument("--save-video", type=Path, help="Optional annotated MP4 output path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for smoke tests.")
    return parser.parse_args()


def resolve_dataset_sequence(args: argparse.Namespace) -> Optional[Path]:
    if args.dataset_root is None:
        return args.sequence

    sequence_dirs = find_dataset_sequences(args.dataset_root, args.dataset)
    if args.list_sequences:
        for path in sequence_dirs:
            print(path)
        return None

    if not args.sequence_name:
        raise RuntimeError("--sequence-name is required when using --dataset-root without --list-sequences.")

    matches = [path for path in sequence_dirs if path.name == args.sequence_name]
    if not matches:
        raise RuntimeError(f"Sequence {args.sequence_name!r} not found under {args.dataset_root}")
    return matches[0]


def find_dataset_sequences(root: Path, dataset: Optional[str]) -> List[Path]:
    if not root.exists():
        raise RuntimeError(f"Dataset root does not exist: {root}")

    candidates = []
    search_roots = [root]
    if dataset == "dancetrack":
        search_roots.extend(path for name in ("train", "val", "test") if (path := root / name).is_dir())
    elif dataset == "mot17":
        search_roots.extend(path for name in ("train", "test") if (path := root / name).is_dir())

    for search_root in search_roots:
        for path in search_root.iterdir():
            if path.is_dir() and (path / "img1").is_dir():
                candidates.append(path)

    return sorted(set(candidates), key=lambda path: str(path).lower())


def parse_video_source(value: str) -> VideoSource:
    try:
        return int(value)
    except ValueError:
        return value


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    sequence = resolve_dataset_sequence(args)
    if args.list_sequences:
        raise SystemExit(0)
    if sequence is not None:
        return ImageSequenceSource(sequence, args.sequence_fps)
    return VideoCaptureSource(parse_video_source(args.video))


def create_backend(args: argparse.Namespace, source: FrameSource):
    weight_path = args.weight_path or LORAT_WEIGHT_BY_CONFIG[args.lorat_config]
    return LoRATMultiObjectTracker(
        args.lorat_root,
        args.lorat_config,
        weight_path,
        args.device,
        args.max_tracks,
        source.fps,
        source.length,
        source.name,
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


def xywh_to_xyxy_np(bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array((x, y, x + w, y + h), dtype=np.float64)


def xyxy_to_xywh_tuple(bbox: np.ndarray) -> BBox:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_iou(left: BBox, right: BBox) -> float:
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


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def bbox_diagonal(bbox: BBox) -> float:
    _, _, w, h = bbox
    return float(np.hypot(w, h))


def center_distance(left: BBox, right: BBox) -> float:
    left_x, left_y = bbox_center(left)
    right_x, right_y = bbox_center(right)
    return float(np.hypot(left_x - right_x, left_y - right_y))


def bbox_delta(previous: BBox, current: BBox) -> BBox:
    return tuple(float(current_value - previous_value) for previous_value, current_value in zip(previous, current))


def predict_bbox(bbox: BBox, velocity: BBox) -> BBox:
    return clamp_bbox_size(tuple(float(value + delta) for value, delta in zip(bbox, velocity)))


def clamp_bbox_size(bbox: BBox) -> BBox:
    x, y, w, h = bbox
    return float(x), float(y), max(1.0, float(w)), max(1.0, float(h))


def blend_bbox(anchor: BBox, proposal: BBox, proposal_weight: float) -> BBox:
    proposal_weight = max(0.0, min(1.0, proposal_weight))
    anchor_weight = 1.0 - proposal_weight
    return tuple(
        float((anchor_value * anchor_weight) + (proposal_value * proposal_weight))
        for anchor_value, proposal_value in zip(anchor, proposal)
    )


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
            cv2.putText(preview, f"Box {index}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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


def append_mot_results(lines: List[str], frame_number: int, tracks: Sequence[TrackState]) -> None:
    for track in tracks:
        if not track.ok:
            continue
        x, y, w, h = track.bbox
        confidence = track.confidence if track.confidence is not None else 1.0
        lines.append(
            f"{frame_number},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{confidence:.4f},-1,-1,-1\n"
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    frame_number: int,
    backend_label: str,
) -> np.ndarray:
    output = frame.copy()
    header = f"Frame {frame_number} | {backend_label} | q quit | a add boxes | p pause"
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    for track in tracks:
        x, y, w, h = [int(round(value)) for value in track.bbox]
        color = track.color if track.ok else (0, 0, 255)
        label = f"ID {track.track_id}"
        if track.confidence is not None:
            label += f" {track.confidence:.2f}"
        if track.coordinator_state:
            label += f" {track.coordinator_state}"
        if not track.ok:
            label += " LOST"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return output


def default_output_path(source_name: str, backend: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_name)
    return DEFAULT_OUTPUT_DIR / f"{safe_name}_{backend}_tracks.txt"


def make_video_writer(path: Path, fps: float, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def main() -> int:
    args = parse_args()
    frame_source = open_frame_source(args)
    output_path = args.output or default_output_path(frame_source.name, "lorat")
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

    backend = create_backend(args, frame_source)
    writer = make_video_writer(args.save_video, frame_source.fps, first_frame) if args.save_video else None
    mot_lines: List[str] = []
    frame_number = 1
    paused = False

    try:
        backend.initialize(first_frame, boxes, frame_number)
        append_mot_results(mot_lines, frame_number, backend.tracks)

        while True:
            if not paused:
                ok, frame = frame_source.read()
                if not ok or frame is None:
                    break

                frame_number += 1
                backend.update(frame, frame_number)
                append_mot_results(mot_lines, frame_number, backend.tracks)
            else:
                frame = frame.copy()

            shown = draw_tracks(frame, backend.tracks, frame_number, backend.backend_name)
            if writer is not None and not paused:
                writer.write(shown)

            cv2.imshow("LoRAT Multi-Object Tracker v3", shown)
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("a"):
                paused = True
                new_boxes = select_boxes(frame, "Add Objects")
                if new_boxes:
                    added_tracks = backend.add_tracks(frame, new_boxes, frame_number)
                    append_mot_results(mot_lines, frame_number, added_tracks)
                paused = False

            if args.max_frames > 0 and frame_number >= args.max_frames:
                break
    finally:
        backend.close()
        frame_source.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    output_path.write_text("".join(mot_lines), encoding="utf-8")
    print(f"Wrote MOTChallenge-format tracks to: {output_path}")
    if args.save_video:
        print(f"Wrote annotated video to: {args.save_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
