from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RunRoot:
    root: Path
    label: str
    candidate_csv: Optional[Path]
    identity_csv: Optional[Path]
    identity_summary_csv: Optional[Path]
    timing_csv: Optional[Path]
    videos: Tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review installed V8/Week 3 benchmark videos and diagnostics, then label failure "
            "types and frequencies by run/config/ReID/object count."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "PRESENTATION MATERIAL" / "v8_failure_review",
    )
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--include-outputs", action="store_true", default=True)
    parser.add_argument("--no-visuals", action="store_true", help="Skip contact sheets and burst sheets.")
    parser.add_argument("--max-contact-videos", type=int, default=80)
    parser.add_argument("--max-burst-sheets", type=int, default=60)
    parser.add_argument("--large-jump-px", type=float, default=80.0)
    parser.add_argument("--large-jump-norm", type=float, default=0.35)
    parser.add_argument("--freeze-pred-jump-px", type=float, default=2.0)
    parser.add_argument("--freeze-gt-jump-px", type=float, default=10.0)
    parser.add_argument("--wrong-object-margin", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fnum(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def bbox_from_text(text: object) -> Optional[BBox]:
    if not text:
        return None
    try:
        parts = [float(part) for part in str(text).split(";")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_diag(bbox: BBox) -> float:
    return math.hypot(max(1.0, bbox[2]), max(1.0, bbox[3]))


def center_distance(left: BBox, right: BBox) -> float:
    lx, ly = bbox_center(left)
    rx, ry = bbox_center(right)
    return math.hypot(lx - rx, ly - ry)


def case_key_from_row(run_label: str, row: Mapping[str, str]) -> Tuple[str, str, str, str, int]:
    return (
        run_label,
        row.get("sequence", ""),
        row.get("lorat_config", ""),
        row.get("reid_mode", "") or "default",
        inum(row.get("target_tracks")),
    )


def track_key_from_row(run_label: str, row: Mapping[str, str]) -> Tuple[str, str, str, str, int, int]:
    case = case_key_from_row(run_label, row)
    return (*case, inum(row.get("tracker_id")))


def find_first(root: Path, name: str) -> Optional[Path]:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def find_run_roots(search_roots: Sequence[Path]) -> List[RunRoot]:
    roots_by_dir: Dict[Path, Dict[str, Optional[Path]]] = {}
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate_csv in search_root.rglob("candidate_diagnostics.csv"):
            roots_by_dir.setdefault(candidate_csv.parent, {})["candidate_csv"] = candidate_csv
        for identity_csv in search_root.rglob("identity_observations_sampled.csv"):
            roots_by_dir.setdefault(identity_csv.parent, {})["identity_csv"] = identity_csv
        for timing_csv in search_root.rglob("timing_by_object_count.csv"):
            roots_by_dir.setdefault(timing_csv.parent, {})["timing_csv"] = timing_csv
        for identity_summary_csv in search_root.rglob("identity_recovery_summary.csv"):
            roots_by_dir.setdefault(identity_summary_csv.parent, {})["identity_summary_csv"] = identity_summary_csv

    run_roots: List[RunRoot] = []
    for root, paths in sorted(roots_by_dir.items()):
        videos = tuple(sorted((root / "videos").glob("*.mp4"))) if (root / "videos").exists() else tuple(sorted(root.rglob("*.mp4")))
        if not videos and not paths.get("candidate_csv") and not paths.get("identity_csv"):
            continue
        run_roots.append(
            RunRoot(
                root=root,
                label=root.name,
                candidate_csv=paths.get("candidate_csv"),
                identity_csv=paths.get("identity_csv"),
                identity_summary_csv=paths.get("identity_summary_csv"),
                timing_csv=paths.get("timing_csv"),
                videos=videos,
            )
        )
    return run_roots


def installed_video_inventory(search_roots: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*.mp4")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "parent": str(path.parent),
                    "size_mb": f"{path.stat().st_size / (1024 * 1024):.2f}",
                    "diagnostic_classifiable": int(
                        (path.parent.parent / "candidate_diagnostics.csv").exists()
                        or (path.parent.parent / "identity_observations_sampled.csv").exists()
                    ),
                }
            )
    return rows


def label_candidate_failure(row: Mapping[str, str], args: argparse.Namespace) -> List[str]:
    labels: List[str] = []
    final_iou = fnum(row.get("final_iou"))
    head_iou = fnum(row.get("head_iou"))
    head_top5 = fnum(row.get("head_top5_best_iou"))
    template_iou = fnum(row.get("template_iou"))
    fused_iou = fnum(row.get("fused_iou"))
    assigned_iou = fnum(row.get("assigned_iou"))
    best_other = fnum(row.get("final_best_other_iou"))
    final_state = str(row.get("final_state", "")).upper()
    bucket = str(row.get("iou_failure_bucket", "") or "")

    if final_iou >= 0.50:
        labels.append("ok_iou50")
    elif final_iou >= 0.30:
        labels.append("partial_iou30")
    else:
        labels.append("low_iou_lt_0_30")

    if best_other > final_iou + float(args.wrong_object_margin):
        labels.append("wrong_object_preferred")
    if bucket:
        labels.append(f"bucket:{bucket}")
    if not truthy(row.get("final_correct_object")) and final_iou < 0.50:
        labels.append("not_correct_object")
    if head_top5 >= 0.50 and final_iou < 0.50:
        labels.append("good_candidate_available_missed")
    if head_iou >= 0.50 and final_iou < 0.50:
        labels.append("head_good_final_bad")
    if assigned_iou >= 0.50 and final_iou < 0.50:
        labels.append("assignment_good_final_bad")
    if fused_iou >= 0.50 and final_iou < 0.50:
        labels.append("fusion_good_final_bad")
    if template_iou >= 0.50 and final_iou < 0.50:
        labels.append("template_good_final_bad")
    if final_iou < 0.30 and head_top5 < 0.30:
        labels.append("head_no_good_candidate")
    if not truthy(row.get("accepted")):
        labels.append("not_accepted")
    if truthy(row.get("held")):
        labels.append("held_previous_box")
    if any(token in final_state for token in ("LOWCONF", "LOST", "MISS", "UNCERTAIN", "HOLD")):
        labels.append("uncertain_or_lost_state")
    return labels


def add_motion_failure_labels(
    rows: Sequence[Mapping[str, str]],
    labels_by_index: Dict[int, List[str]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    by_track: Dict[Tuple[str, str, str, str, int, int], List[Tuple[int, Mapping[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = track_key_from_row("__run__", row)[1:]
        by_track[key].append((index, row))

    for key, group in by_track.items():
        group.sort(key=lambda item: inum(item[1].get("frame")))
        previous_pred: Optional[BBox] = None
        previous_gt: Optional[BBox] = None
        previous_frame: Optional[int] = None
        for index, row in group:
            pred = bbox_from_text(row.get("final_bbox"))
            gt = bbox_from_text(row.get("gt_bbox"))
            if pred is not None and previous_pred is not None:
                pred_jump = center_distance(previous_pred, pred)
                pred_jump_norm = pred_jump / max(1.0, bbox_diag(previous_pred))
                gt_jump = center_distance(previous_gt, gt) if previous_gt is not None and gt is not None else 0.0
                final_iou = fnum(row.get("final_iou"))
                frame = inum(row.get("frame"))
                if pred_jump >= float(args.large_jump_px) or pred_jump_norm >= float(args.large_jump_norm):
                    labels_by_index[index].append("large_box_jump")
                    events.append(
                        event_row(row, "large_box_jump", pred_jump, gt_jump, pred_jump_norm, previous_frame, frame)
                    )
                if pred_jump <= float(args.freeze_pred_jump_px) and gt_jump >= float(args.freeze_gt_jump_px) and final_iou < 0.50:
                    labels_by_index[index].append("frozen_or_stale_box")
                    events.append(
                        event_row(row, "frozen_or_stale_box", pred_jump, gt_jump, pred_jump_norm, previous_frame, frame)
                    )
                if final_iou < 0.30 and gt_jump >= float(args.freeze_gt_jump_px) and pred_jump < gt_jump * 0.25:
                    labels_by_index[index].append("under_following_motion")
            if pred is not None:
                previous_pred = pred
            if gt is not None:
                previous_gt = gt
            previous_frame = inum(row.get("frame"))
    return events


def event_row(
    row: Mapping[str, str],
    event_type: str,
    pred_jump: float,
    gt_jump: float,
    pred_jump_norm: float,
    previous_frame: Optional[int],
    frame: int,
) -> Dict[str, object]:
    return {
        "event_type": event_type,
        "sequence": row.get("sequence", ""),
        "lorat_config": row.get("lorat_config", ""),
        "reid_mode": row.get("reid_mode", "") or "default",
        "target_tracks": inum(row.get("target_tracks")),
        "tracker_id": inum(row.get("tracker_id")),
        "previous_frame": previous_frame or "",
        "frame": frame,
        "pred_jump_px": f"{pred_jump:.3f}",
        "gt_jump_px": f"{gt_jump:.3f}",
        "pred_jump_norm": f"{pred_jump_norm:.6f}",
        "final_iou": row.get("final_iou", ""),
        "head_iou": row.get("head_iou", ""),
        "head_top5_best_iou": row.get("head_top5_best_iou", ""),
        "final_best_other_iou": row.get("final_best_other_iou", ""),
        "candidate_source": row.get("candidate_source", ""),
        "final_state": row.get("final_state", ""),
        "iou_failure_bucket": row.get("iou_failure_bucket", ""),
        "iou_failure_stage": row.get("iou_failure_stage", ""),
    }


def summarize_candidates(run: RunRoot, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows = read_csv(run.candidate_csv)
    labels_by_index: Dict[int, List[str]] = {}
    for index, row in enumerate(rows):
        labels_by_index[index] = label_candidate_failure(row, args)
    motion_events = add_motion_failure_labels(rows, labels_by_index, args)

    counts: Dict[Tuple[str, str, str, str, int], Counter[str]] = defaultdict(Counter)
    totals: Counter[Tuple[str, str, str, str, int]] = Counter()
    event_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows):
        key = case_key_from_row(run.label, row)
        totals[key] += 1
        labels = labels_by_index[index]
        for label in labels:
            counts[key][label] += 1
        if "large_box_jump" in labels or "frozen_or_stale_box" in labels or "wrong_object_preferred" in labels:
            event = event_row(
                row,
                "+".join(label for label in labels if label in {"large_box_jump", "frozen_or_stale_box", "wrong_object_preferred"}),
                0.0,
                0.0,
                0.0,
                "",
                inum(row.get("frame")),
            )
            event["run"] = run.label
            event_rows.append(event)

    for event in motion_events:
        event["run"] = run.label
        event_rows.append(event)

    summary_rows: List[Dict[str, object]] = []
    count_rows: List[Dict[str, object]] = []
    for key in sorted(totals):
        run_label, sequence, config, reid_mode, target_tracks = key
        total = totals[key]
        counter = counts[key]
        summary_rows.append(
            {
                "run": run_label,
                "sequence": sequence,
                "lorat_config": config,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "diagnostic_rows": total,
                "ok_iou50_rate": rate(counter["ok_iou50"], total),
                "partial_iou30_rate": rate(counter["partial_iou30"], total),
                "low_iou_lt_0_30_rate": rate(counter["low_iou_lt_0_30"], total),
                "wrong_object_preferred_rate": rate(counter["wrong_object_preferred"], total),
                "good_candidate_available_missed_rate": rate(counter["good_candidate_available_missed"], total),
                "head_no_good_candidate_rate": rate(counter["head_no_good_candidate"], total),
                "large_box_jump_rate": rate(counter["large_box_jump"], total),
                "frozen_or_stale_box_rate": rate(counter["frozen_or_stale_box"], total),
                "under_following_motion_rate": rate(counter["under_following_motion"], total),
                "not_accepted_rate": rate(counter["not_accepted"], total),
                "held_previous_box_rate": rate(counter["held_previous_box"], total),
                "top_failure_bucket": top_bucket(counter, "bucket:"),
            }
        )
        for label, count in sorted(counter.items()):
            count_rows.append(
                {
                    "run": run_label,
                    "sequence": sequence,
                    "lorat_config": config,
                    "reid_mode": reid_mode,
                    "target_tracks": target_tracks,
                    "failure_type": label,
                    "count": count,
                    "rate": rate(count, total),
                    "diagnostic_rows": total,
                }
            )
    return summary_rows, count_rows, event_rows


def rate(count: int, total: int) -> str:
    return "" if total <= 0 else f"{count / total:.6f}"


def top_bucket(counter: Counter[str], prefix: str) -> str:
    buckets = [(label[len(prefix) :], count) for label, count in counter.items() if label.startswith(prefix)]
    if not buckets:
        return ""
    buckets.sort(key=lambda item: item[1], reverse=True)
    return f"{buckets[0][0]}:{buckets[0][1]}"


def summarize_identity(run: RunRoot) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows = read_csv(run.identity_csv)
    counts: Dict[Tuple[str, str, str, str, int], Counter[str]] = defaultdict(Counter)
    totals: Counter[Tuple[str, str, str, str, int]] = Counter()
    for row in rows:
        key = case_key_from_row(run.label, row)
        totals[key] += 1
        for field in ("correct_object", "identity_jump", "identity_switch", "track_lost", "occluded", "ok"):
            if truthy(row.get(field)):
                counts[key][field] += 1

    summary_rows: List[Dict[str, object]] = []
    count_rows: List[Dict[str, object]] = []
    for key in sorted(totals):
        run_label, sequence, config, reid_mode, target_tracks = key
        total = totals[key]
        counter = counts[key]
        summary_rows.append(
            {
                "run": run_label,
                "sequence": sequence,
                "lorat_config": config,
                "reid_mode": reid_mode,
                "target_tracks": target_tracks,
                "identity_samples": total,
                "correct_object_rate": rate(counter["correct_object"], total),
                "identity_jump_rate": rate(counter["identity_jump"], total),
                "identity_switch_rate": rate(counter["identity_switch"], total),
                "track_lost_rate": rate(counter["track_lost"], total),
                "occluded_rate": rate(counter["occluded"], total),
                "ok_rate": rate(counter["ok"], total),
                "identity_switch_count": counter["identity_switch"],
                "track_lost_count": counter["track_lost"],
            }
        )
        for label, count in sorted(counter.items()):
            count_rows.append(
                {
                    "run": run_label,
                    "sequence": sequence,
                    "lorat_config": config,
                    "reid_mode": reid_mode,
                    "target_tracks": target_tracks,
                    "failure_type": f"identity:{label}",
                    "count": count,
                    "rate": rate(count, total),
                    "identity_samples": total,
                }
            )
    return summary_rows, count_rows


def video_case_from_name(path: Path) -> Tuple[str, str, int]:
    stem = path.stem
    reid_mode = "default"
    if "reid_on" in stem:
        reid_mode = "reid_on"
    elif "reid_off" in stem:
        reid_mode = "reid_off"
    config = ""
    for candidate in ("B-224", "L-224", "g-224", "B_224", "L_224", "g_224"):
        if candidate in stem:
            config = candidate.replace("_", "-")
            break
    target_tracks = 0
    for part in stem.split("_"):
        if part.startswith("N") and part[1:].isdigit():
            target_tracks = int(part[1:])
    return config, reid_mode, target_tracks


def make_contact_sheet(video_path: Path, output_path: Path, frames_per_sheet: int = 15) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return False
    indices = np.linspace(0, max(0, total - 1), frames_per_sheet, dtype=int).tolist()
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        new_width = 280
        new_height = max(1, int(height * new_width / width))
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (new_width, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"{video_path.stem[:26]} f{index + 1}/{total}", (5, 20), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    cap.release()
    if not frames:
        return False
    cols = 5
    rows = math.ceil(len(frames) / cols)
    cell_h = max(frame.shape[0] for frame in frames)
    cell_w = max(frame.shape[1] for frame in frames)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, col = divmod(index, cols)
        y, x = row * cell_h, col * cell_w
        sheet[y : y + frame.shape[0], x : x + frame.shape[1]] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def make_burst_sheet(video_path: Path, event: Mapping[str, object], output_path: Path) -> bool:
    frame = inum(event.get("frame"), 1)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return False
    start = max(1, frame - 4)
    end = min(total, frame + 4)
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame_number in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, image = cap.read()
        if not ok or image is None:
            continue
        height, width = image.shape[:2]
        new_width = 260
        new_height = max(1, int(height * new_width / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (new_width, 48), (0, 0, 0), -1)
        cv2.putText(image, f"{event.get('event_type')} f{frame_number}", (5, 19), font, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"N{event.get('target_tracks')} T{event.get('tracker_id')}", (5, 39), font, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(image)
    cap.release()
    if not frames:
        return False
    sheet = np.hstack(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def match_video(run: RunRoot, event: Mapping[str, object]) -> Optional[Path]:
    config = str(event.get("lorat_config", ""))
    reid_mode = str(event.get("reid_mode", ""))
    target_tracks = int(event.get("target_tracks", 0) or 0)
    candidates = []
    for video in run.videos:
        v_config, v_reid, v_n = video_case_from_name(video)
        if config and v_config and config != v_config:
            continue
        if target_tracks and v_n and target_tracks != v_n:
            continue
        if reid_mode not in ("", "default") and v_reid != "default" and reid_mode != v_reid:
            continue
        candidates.append(video)
    return candidates[0] if candidates else None


def write_report(
    path: Path,
    inventory_rows: Sequence[Mapping[str, object]],
    candidate_summary: Sequence[Mapping[str, object]],
    identity_summary: Sequence[Mapping[str, object]],
    failure_counts: Sequence[Mapping[str, object]],
    event_rows: Sequence[Mapping[str, object]],
    run_roots: Sequence[RunRoot],
) -> None:
    non_failure_labels = {
        "ok_iou50",
        "identity:ok",
        "identity:correct_object",
    }
    top_failures = sorted(
        [
            row
            for row in failure_counts
            if not str(row.get("failure_type", "")).startswith("bucket:")
            and str(row.get("failure_type", "")) not in non_failure_labels
        ],
        key=lambda row: fnum(row.get("rate")),
        reverse=True,
    )[:25]
    lines = [
        "# Installed V8 Video Failure Review",
        "",
        "## Scope",
        "",
        f"- Installed MP4 files inventoried: `{len(inventory_rows)}`",
        f"- Diagnostic benchmark roots analyzed: `{len(run_roots)}`",
        "- Frequency counts come from `candidate_diagnostics.csv` and `identity_observations_sampled.csv`.",
        "- Videos without matching diagnostics are inventoried but not frequency-labeled.",
        "",
        "## Diagnostic Roots",
        "",
    ]
    for run in run_roots:
        lines.append(f"- `{run.root}` ({len(run.videos)} videos)")
    lines += [
        "",
        "## Candidate Failure Summary",
        "",
        "| Run | Config | ReID | N | Rows | OK@0.5 | Low IoU<0.3 | Wrong Object | Good Candidate Missed | Head No Candidate | Large Jump | Frozen/Stale | Under-Follow Motion | Top Bucket |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidate_summary:
        lines.append(
            f"| {row.get('run')} | {row.get('lorat_config')} | {row.get('reid_mode')} | {row.get('target_tracks')} | "
            f"{row.get('diagnostic_rows')} | {fmt_pct(row.get('ok_iou50_rate'))} | {fmt_pct(row.get('low_iou_lt_0_30_rate'))} | "
            f"{fmt_pct(row.get('wrong_object_preferred_rate'))} | {fmt_pct(row.get('good_candidate_available_missed_rate'))} | "
            f"{fmt_pct(row.get('head_no_good_candidate_rate'))} | {fmt_pct(row.get('large_box_jump_rate'))} | "
            f"{fmt_pct(row.get('frozen_or_stale_box_rate'))} | {fmt_pct(row.get('under_following_motion_rate'))} | "
            f"{row.get('top_failure_bucket')} |"
        )
    lines += [
        "",
        "## Identity Failure Summary",
        "",
        "| Run | Config | ReID | N | Samples | Correct | Track Lost | ID Switch | ID Jump | Occluded |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in identity_summary:
        lines.append(
            f"| {row.get('run')} | {row.get('lorat_config')} | {row.get('reid_mode')} | {row.get('target_tracks')} | "
            f"{row.get('identity_samples')} | {fmt_pct(row.get('correct_object_rate'))} | {fmt_pct(row.get('track_lost_rate'))} | "
            f"{fmt_pct(row.get('identity_switch_rate'))} | {fmt_pct(row.get('identity_jump_rate'))} | {fmt_pct(row.get('occluded_rate'))} |"
        )
    lines += [
        "",
        "## Most Frequent Failure Labels",
        "",
        "| Run | Config | ReID | N | Failure Type | Count | Rate |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in top_failures:
        lines.append(
            f"| {row.get('run')} | {row.get('lorat_config')} | {row.get('reid_mode')} | {row.get('target_tracks')} | "
            f"{row.get('failure_type')} | {row.get('count')} | {fmt_pct(row.get('rate'))} |"
        )
    lines += [
        "",
        "## Top Motion Events",
        "",
        "| Run | Type | Config | ReID | N | Tracker | Frame | Pred Jump | GT Jump | Final IoU | Bucket |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(event_rows, key=lambda item: (fnum(item.get("pred_jump_px")), -fnum(item.get("final_iou"))), reverse=True)[:30]:
        lines.append(
            f"| {row.get('run')} | {row.get('event_type')} | {row.get('lorat_config')} | {row.get('reid_mode')} | "
            f"{row.get('target_tracks')} | {row.get('tracker_id')} | {row.get('frame')} | {fnum(row.get('pred_jump_px')):.1f} | "
            f"{fnum(row.get('gt_jump_px')):.1f} | {fnum(row.get('final_iou')):.3f} | {row.get('iou_failure_bucket')} |"
        )
    lines += [
        "",
        "## Output Files",
        "",
        "- `video_inventory.csv`",
        "- `candidate_failure_summary.csv`",
        "- `failure_type_counts.csv`",
        "- `identity_failure_summary.csv`",
        "- `top_failure_events.csv`",
        "- `contact_sheets/`",
        "- `failure_bursts/`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_pct(value: object) -> str:
    if value in ("", None):
        return ""
    return f"{100.0 * fnum(value):.1f}%"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    search_roots = [args.downloads.expanduser().resolve()]
    if args.include_outputs:
        search_roots.append((Path.cwd() / "outputs").resolve())

    inventory_rows = installed_video_inventory(search_roots)
    run_roots = find_run_roots(search_roots)

    candidate_summary: List[Dict[str, object]] = []
    failure_counts: List[Dict[str, object]] = []
    identity_summary: List[Dict[str, object]] = []
    identity_counts: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    for run in run_roots:
        c_summary, c_counts, c_events = summarize_candidates(run, args)
        i_summary, i_counts = summarize_identity(run)
        candidate_summary.extend(c_summary)
        failure_counts.extend(c_counts)
        identity_summary.extend(i_summary)
        identity_counts.extend(i_counts)
        event_rows.extend(c_events)

    write_csv(output_dir / "video_inventory.csv", inventory_rows)
    write_csv(output_dir / "candidate_failure_summary.csv", candidate_summary)
    write_csv(output_dir / "failure_type_counts.csv", [*failure_counts, *identity_counts])
    write_csv(output_dir / "identity_failure_summary.csv", identity_summary)
    write_csv(
        output_dir / "top_failure_events.csv",
        sorted(event_rows, key=lambda row: (fnum(row.get("pred_jump_px")), -fnum(row.get("final_iou"))), reverse=True)[:500],
    )

    if not args.no_visuals:
        contact_dir = output_dir / "contact_sheets"
        made_contacts = 0
        for run in run_roots:
            for video in run.videos:
                if made_contacts >= max(0, int(args.max_contact_videos)):
                    break
                output = contact_dir / f"{safe_name(run.label)}__{video.stem}.jpg"
                if make_contact_sheet(video, output):
                    made_contacts += 1
        burst_dir = output_dir / "failure_bursts"
        events_by_run = {run.label: run for run in run_roots}
        made_bursts = 0
        for event in sorted(event_rows, key=lambda row: (fnum(row.get("pred_jump_px")), -fnum(row.get("final_iou"))), reverse=True):
            if made_bursts >= max(0, int(args.max_burst_sheets)):
                break
            run = events_by_run.get(str(event.get("run")))
            if run is None:
                continue
            video = match_video(run, event)
            if video is None:
                continue
            output = burst_dir / (
                f"{made_bursts + 1:03d}_{safe_name(str(event.get('run')))}_"
                f"{safe_name(str(event.get('event_type')))}_N{event.get('target_tracks')}_"
                f"T{event.get('tracker_id')}_F{event.get('frame')}.jpg"
            )
            if make_burst_sheet(video, event, output):
                made_bursts += 1

    write_report(
        output_dir / "failure_review.md",
        inventory_rows,
        candidate_summary,
        identity_summary,
        [*failure_counts, *identity_counts],
        event_rows,
        run_roots,
    )
    print(f"Wrote failure review to {output_dir / 'failure_review.md'}")
    print(f"Inventoried videos: {len(inventory_rows)}")
    print(f"Diagnostic roots: {len(run_roots)}")
    print(f"Candidate summary rows: {len(candidate_summary)}")
    print(f"Identity summary rows: {len(identity_summary)}")
    print(f"Motion/failure events: {len(event_rows)}")
    return 0


def safe_name(text: str) -> str:
    output = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            output.append(char)
        else:
            output.append("_")
    return "".join(output).strip("_") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
