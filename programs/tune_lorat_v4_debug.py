from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_CSV = PROJECT_ROOT / "outputs" / "debug" / "dancetrack0065_lorat_v4_debug.csv"


@dataclass
class DebugRow:
    frame: int
    track_id: int
    ok: bool
    state: str
    confidence: Optional[float]
    raw_confidence: Optional[float]
    confidence_baseline: Optional[float]
    bbox: Tuple[float, float, float, float]
    raw_bbox: Optional[Tuple[float, float, float, float]]
    pred_bbox: Optional[Tuple[float, float, float, float]]
    prev_bbox: Optional[Tuple[float, float, float, float]]
    assignment_score: Optional[float]
    assignment_margin: Optional[float]
    reid_score: Optional[float]
    motion_score: Optional[float]
    path_score: Optional[float]
    source_score: Optional[float]
    lost_frames: int
    occluded_frames: int
    active_lorat_slot: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and quickly tune V4 LoRAT MOT thresholds from a debug CSV."
    )
    parser.add_argument(
        "debug_csv",
        nargs="*",
        type=Path,
        default=[DEFAULT_DEBUG_CSV],
        help="V4 debug CSV path. Defaults to outputs/debug/dancetrack0065_lorat_v4_debug.csv.",
    )
    parser.add_argument("--jump-pixels", type=float, default=80.0)
    parser.add_argument("--jump-diagonal-factor", type=float, default=1.25)
    parser.add_argument("--top", type=int, default=12, help="Rows to show per diagnostic section.")
    parser.add_argument("--emit-cli", action="store_true", help="Print a trial CLI arg block from the recommendations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: List[DebugRow] = []
    for path in args.debug_csv:
        rows = load_rows(path)
        print(f"\n== {path} ==")
        if not rows:
            print("No rows found.")
            continue
        summarize_rows(rows, args)
        all_rows.extend(rows)

    if len(args.debug_csv) > 1 and all_rows:
        print("\n== combined ==")
        summarize_rows(all_rows, args, compact=True)
    return 0


def load_rows(path: Path) -> List[DebugRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [parse_row(row) for row in reader]


def parse_row(row: dict) -> DebugRow:
    return DebugRow(
        frame=int(float(row["frame"])),
        track_id=int(float(row["track_id"])),
        ok=bool(int(float(row["ok"]))),
        state=row.get("state", ""),
        confidence=to_float(row.get("confidence")),
        raw_confidence=to_float(row.get("raw_confidence")),
        confidence_baseline=to_float(row.get("confidence_baseline")),
        bbox=parse_bbox(row, ""),
        raw_bbox=parse_optional_bbox(row, "raw_"),
        pred_bbox=parse_optional_bbox(row, "pred_"),
        prev_bbox=parse_optional_bbox(row, "prev_"),
        assignment_score=to_float(row.get("assignment_score")),
        assignment_margin=to_float(row.get("assignment_margin")),
        reid_score=to_float(row.get("reid_score")),
        motion_score=to_float(row.get("motion_score")),
        path_score=to_float(row.get("path_score")),
        source_score=to_float(row.get("source_score")),
        lost_frames=int(float(row.get("lost_frames") or 0)),
        occluded_frames=int(float(row.get("occluded_frames") or 0)),
        active_lorat_slot=row.get("active_lorat_slot", ""),
    )


def summarize_rows(rows: Sequence[DebugRow], args: argparse.Namespace, compact: bool = False) -> None:
    print(f"rows: {len(rows)} | frames: {min(row.frame for row in rows)}-{max(row.frame for row in rows)}")
    tracks = sorted({row.track_id for row in rows})
    print(f"tracks: {tracks}")
    print(f"ok frames: {sum(row.ok for row in rows)} | lost/not-ok frames: {sum(not row.ok for row in rows)}")

    state_counts = count_states(rows)
    print("\nstate counts:")
    for state, count in state_counts[:12]:
        print(f"  {state or '(init)'}: {count}")

    accepted = [row for row in rows if is_committed_update(row)]
    rejected = [row for row in rows if is_rejected(row)]
    occluded = [row for row in rows if "OCCLUDED" in row.state]
    suspicious = suspicious_jumps(rows, args.jump_pixels, args.jump_diagonal_factor)

    print("\nscore distributions:")
    print_score_block("committed", accepted)
    print_score_block("rejected/held", rejected)
    if suspicious:
        print_score_block("suspicious jumps", [row for _, row, _ in suspicious])

    print("\nlongest occlusion streaks:")
    for track_id, start, end, length in occlusion_streaks(rows)[: args.top]:
        print(f"  track {track_id}: f{start}-f{end} ({length} frames)")

    if suspicious:
        print("\nsuspicious accepted jumps:")
        for jump, row, previous in suspicious[: args.top]:
            print(
                f"  f{previous.frame}->f{row.frame} track {row.track_id}: "
                f"jump={jump:.1f}px state={row.state} conf={fmt(row.confidence)} "
                f"reid={fmt(row.reid_score)} motion={fmt(row.motion_score)} path={fmt(row.path_score)}"
            )
    else:
        print("\nsuspicious accepted jumps: none by current thresholds")

    suggestions = suggest_thresholds(rows, suspicious)
    print("\nquick tuning suggestions:")
    for key, value, reason in suggestions:
        print(f"  {key} {value}  # {reason}")

    if args.emit_cli and suggestions:
        print("\ntrial CLI block:")
        for key, value, _ in suggestions:
            print(f"  {key} {value} `")

    if not compact:
        print("\nreading tips:")
        print("  many LOWCONF with high reid/motion/path -> lower --lorat-accept-min-score slightly")
        print("  accepted jumps with low path -> raise --identity-min-path")
        print("  PATHLOW during true turns -> lower --identity-min-path or --view-change-min-motion")
        print("  long OCCLUDED then LOST -> raise --occlusion-max-frames or lower recovery thresholds")


def print_score_block(label: str, rows: Sequence[DebugRow]) -> None:
    print(f"  {label}: n={len(rows)}")
    for name, values in (
        ("conf", [row.confidence for row in rows]),
        ("raw_conf", [row.raw_confidence for row in rows]),
        ("reid", [row.reid_score for row in rows]),
        ("motion", [row.motion_score for row in rows]),
        ("path", [row.path_score for row in rows]),
        ("assign", [row.assignment_score for row in rows]),
    ):
        summary = summarize_values(value for value in values if value is not None)
        if summary:
            print(f"    {name}: {summary}")


def suggest_thresholds(
    rows: Sequence[DebugRow],
    suspicious: Sequence[Tuple[float, DebugRow, DebugRow]],
) -> List[Tuple[str, str, str]]:
    suggestions: List[Tuple[str, str, str]] = []
    rejected = [row for row in rows if is_rejected(row)]
    recoverable_lowconf = [
        row
        for row in rejected
        if "LOWCONF" in row.state
        and value_at_least(row.reid_score, 0.60)
        and value_at_least(row.motion_score, 0.60)
        and value_at_least(row.path_score, 0.35)
        and row.confidence is not None
    ]
    if len(recoverable_lowconf) >= 3:
        target = clamp(quantile([row.confidence for row in recoverable_lowconf if row.confidence is not None], 0.20), 0.08, 0.30)
        suggestions.append((
            "--lorat-accept-min-score",
            f"{target:.2f}",
            f"{len(recoverable_lowconf)} held LOWCONF rows still had strong reid/motion/path",
        ))

    suspicious_path = [row.path_score for _, row, _ in suspicious if row.path_score is not None]
    good_path = [
        row.path_score
        for row in rows
        if is_committed_update(row) and row.path_score is not None and "VIEWCHANGE" not in row.state
    ]
    if suspicious_path and good_path:
        bad_p75 = quantile(suspicious_path, 0.75)
        good_p10 = quantile(good_path, 0.10)
        if bad_p75 < good_p10:
            target = clamp((bad_p75 + good_p10) / 2.0, 0.10, 0.70)
            suggestions.append((
                "--identity-min-path",
                f"{target:.2f}",
                "separates suspicious accepted jumps from normal committed path scores",
            ))

    reidlow_smooth = [
        row
        for row in rejected
        if "REIDLOW" in row.state
        and value_at_least(row.motion_score, 0.75)
        and value_at_least(row.path_score, 0.45)
        and value_at_least(row.confidence, 0.20)
    ]
    if len(reidlow_smooth) >= 2:
        suggestions.append((
            "--view-change-min-motion",
            "0.68",
            f"{len(reidlow_smooth)} REIDLOW rows looked motion/path-consistent",
        ))

    max_occlusion = max((length for _, _, _, length in occlusion_streaks(rows)), default=0)
    any_lost_after_occlusion = any((not row.ok and row.occluded_frames >= 25) for row in rows)
    if any_lost_after_occlusion and max_occlusion >= 30:
        suggestions.append((
            "--occlusion-max-frames",
            str(min(60, max_occlusion + 15)),
            "track reached LOST after a long occlusion hold",
        ))

    if not suggestions:
        suggestions.append((
            "--debug-frame-start/--debug-frame-end",
            "around the visible failure",
            "current CSV does not show a clean threshold separation",
        ))
    return suggestions


def suspicious_jumps(
    rows: Sequence[DebugRow],
    jump_pixels: float,
    diagonal_factor: float,
) -> List[Tuple[float, DebugRow, DebugRow]]:
    by_track = group_by_track(rows)
    jumps: List[Tuple[float, DebugRow, DebugRow]] = []
    for track_rows in by_track:
        for previous, row in zip(track_rows, track_rows[1:]):
            if not is_committed_update(row):
                continue
            jump = center_distance(previous.bbox, row.bbox)
            threshold = max(jump_pixels, diagonal_factor * bbox_diagonal(previous.bbox))
            if jump >= threshold:
                jumps.append((jump, row, previous))
    return sorted(jumps, key=lambda item: item[0], reverse=True)


def occlusion_streaks(rows: Sequence[DebugRow]) -> List[Tuple[int, int, int, int]]:
    streaks: List[Tuple[int, int, int, int]] = []
    for track_rows in group_by_track(rows):
        start: Optional[int] = None
        previous_frame: Optional[int] = None
        track_id = track_rows[0].track_id if track_rows else 0
        for row in track_rows:
            if "OCCLUDED" in row.state or "LOST" in row.state:
                if start is None:
                    start = row.frame
                previous_frame = row.frame
            elif start is not None and previous_frame is not None:
                streaks.append((track_id, start, previous_frame, previous_frame - start + 1))
                start = None
                previous_frame = None
        if start is not None and previous_frame is not None:
            streaks.append((track_id, start, previous_frame, previous_frame - start + 1))
    return sorted(streaks, key=lambda item: item[3], reverse=True)


def count_states(rows: Sequence[DebugRow]) -> List[Tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def group_by_track(rows: Sequence[DebugRow]) -> List[List[DebugRow]]:
    groups: dict[int, List[DebugRow]] = {}
    for row in rows:
        groups.setdefault(row.track_id, []).append(row)
    return [sorted(group, key=lambda row: row.frame) for group in groups.values()]


def is_committed_update(row: DebugRow) -> bool:
    if not row.ok or row.lost_frames > 0:
        return False
    if not row.state:
        return False
    bad_tokens = ("LOWCONF", "REIDLOW", "MOTIONLOW", "PATHLOW", "ID_UNCERTAIN", "LORAT_MISS", "LOST")
    return not any(token in row.state for token in bad_tokens)


def is_rejected(row: DebugRow) -> bool:
    tokens = ("LOWCONF", "REIDLOW", "MOTIONLOW", "PATHLOW", "ID_UNCERTAIN", "REACQUIRE_LOWCONF", "LORAT_MISS")
    return any(token in row.state for token in tokens)


def to_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bbox(row: dict, prefix: str) -> Tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}x"]),
        float(row[f"{prefix}y"]),
        float(row[f"{prefix}w"]),
        float(row[f"{prefix}h"]),
    )


def parse_optional_bbox(row: dict, prefix: str) -> Optional[Tuple[float, float, float, float]]:
    values = [to_float(row.get(f"{prefix}{key}")) for key in ("x", "y", "w", "h")]
    if any(value is None for value in values):
        return None
    return values[0], values[1], values[2], values[3]


def summarize_values(values: Iterable[float]) -> str:
    data = sorted(values)
    if not data:
        return ""
    return (
        f"min={data[0]:.3f} p10={quantile(data, 0.10):.3f} "
        f"med={median(data):.3f} p90={quantile(data, 0.90):.3f} max={data[-1]:.3f}"
    )


def quantile(values: Sequence[float], q: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return data[low]
    return data[low] + ((data[high] - data[low]) * (position - low))


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + (w / 2.0), y + (h / 2.0)


def center_distance(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    lx, ly = bbox_center(left)
    rx, ry = bbox_center(right)
    return math.hypot(lx - rx, ly - ry)


def bbox_diagonal(bbox: Tuple[float, float, float, float]) -> float:
    _, _, w, h = bbox
    return math.hypot(w, h)


def value_at_least(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
