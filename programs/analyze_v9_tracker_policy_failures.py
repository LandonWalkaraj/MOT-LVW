"""Analyze V9 tracker-policy failures from candidate diagnostics.

This is narrower than analyze_v9_diagnostic_benchmark.py. It focuses on cases
where the V9 head/oracle candidate looks usable but the final tracker output is
poor, held, scale-modified, or identity-swapped.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


GOOD_IOU = 0.50
PARTIAL_IOU = 0.30
BAD_FINAL_IOU = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V9 tracker-policy failures from candidate_diagnostics.csv files.")
    parser.add_argument("result_root", type=Path, help="Extracted V9 benchmark result folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "benchmark_reviews" / "v9_tracker_policy_analysis",
        help="Folder for the generated review files.",
    )
    parser.add_argument("--good-iou", type=float, default=GOOD_IOU)
    parser.add_argument("--partial-iou", type=float, default=PARTIAL_IOU)
    parser.add_argument("--bad-final-iou", type=float, default=BAD_FINAL_IOU)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_bbox(value: object) -> Optional[Tuple[float, float, float, float]]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.replace(",", ";").split(";")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(part.strip()) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    return x, y, w, h


def bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    if bbox is None:
        return None
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def bbox_height(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    return None if bbox is None else max(0.0, bbox[3])


def bbox_width(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    return None if bbox is None else max(0.0, bbox[2])


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def mode_for_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for mode in ("normal", "gt_window", "gt_identity"):
        if mode in parts:
            return mode
    return "unknown"


def group_key(row: Mapping[str, str], mode: str) -> Tuple[str, str, str, int]:
    return (
        mode,
        str(row.get("sequence") or ""),
        str(row.get("lorat_config") or ""),
        int(float(row.get("target_tracks") or 0)),
    )


def row_flags(row: Mapping[str, str], args: argparse.Namespace) -> Dict[str, bool]:
    head_iou = parse_float(row.get("head_iou")) or 0.0
    oracle_iou = parse_float(row.get("oracle_best_iou")) or 0.0
    final_iou = parse_float(row.get("final_iou")) or 0.0
    head_conf = parse_float(row.get("head_confidence")) or 0.0
    accepted = parse_bool(row.get("accepted"))
    held = parse_bool(row.get("held"))
    final_correct = parse_bool(row.get("final_correct_object"))
    state = str(row.get("final_state") or "")
    failure = str(row.get("iou_failure_bucket") or "")
    scale_state = str(row.get("v9_scale_gate_state") or "")
    scale_reason = str(row.get("v9_scale_gate_reason") or "")
    reject_state = str(row.get("reject_state") or "")

    gt = parse_bbox(row.get("gt_bbox"))
    search = parse_bbox(row.get("search_window"))
    final = parse_bbox(row.get("final_bbox"))
    head = parse_bbox(row.get("head_bbox"))

    search_h_ratio = safe_ratio(bbox_height(search), bbox_height(gt))
    search_w_ratio = safe_ratio(bbox_width(search), bbox_width(gt))
    final_h_ratio = safe_ratio(bbox_height(final), bbox_height(gt))
    final_area_ratio = safe_ratio(bbox_area(final), bbox_area(gt))
    head_area_ratio = safe_ratio(bbox_area(head), bbox_area(gt))

    good_head = head_iou >= args.good_iou
    partial_head = head_iou >= args.partial_iou
    good_oracle = oracle_iou >= args.good_iou
    partial_oracle = oracle_iou >= args.partial_iou
    bad_final = final_iou < args.bad_final_iou

    return {
        "good_head_final_bad": good_head and bad_final,
        "partial_head_final_bad": partial_head and bad_final,
        "good_oracle_final_bad": good_oracle and bad_final,
        "partial_oracle_final_bad": partial_oracle and bad_final,
        "held_good_head": held and good_head,
        "held_good_oracle": held and good_oracle,
        "accepted_bad_final": accepted and bad_final,
        "identity_wrong": not final_correct and final_iou < args.good_iou,
        "scale_limited": "SCALELIMIT" in state or "scale" in scale_state.lower() or "scale" in scale_reason.lower(),
        "scale_override_high_conf": "override_high_confidence" in scale_state or "scale_bad_but_head_confident" in scale_reason,
        "oversized_final": (final_h_ratio is not None and final_h_ratio > 1.35) or (final_area_ratio is not None and final_area_ratio > 1.60),
        "oversized_search": (search_h_ratio is not None and search_h_ratio > 2.25) or (search_w_ratio is not None and search_w_ratio > 2.25),
        "reject_lowconf": "LOWCONF" in reject_state or "LOWCONF" in state,
        "reject_center_jump": "CENTER_JUMP" in reject_state or "CENTER_JUMP" in state,
        "memheld": "MEMHELD" in state,
        "viewchange": "VIEWCHANGE" in state,
        "failure_identity": "identity" in failure,
        "failure_post_head": "post_head" in failure or "template_recovery" in failure,
        "head_confident_bad_final": head_conf >= 0.90 and bad_final,
    }


def summarize_group(rows: Sequence[Mapping[str, str]], mode: str, args: argparse.Namespace) -> Dict[str, object]:
    flag_rows = [row_flags(row, args) for row in rows]

    def count(flag: str) -> int:
        return sum(1 for flags in flag_rows if flags[flag])

    def rate(flag: str) -> float:
        return count(flag) / len(rows) if rows else 0.0

    failures = Counter(str(row.get("iou_failure_bucket") or "") for row in rows)
    states = Counter(str(row.get("final_state") or "") for row in rows)
    scale_states = Counter(str(row.get("v9_scale_gate_state") or "") for row in rows)
    reject_states = Counter(str(row.get("reject_state") or "") for row in rows)
    reid_modes = sorted(set(str(row.get("reid_mode") or "") for row in rows))

    return {
        "mode": mode,
        "sequence": rows[0].get("sequence") if rows else "",
        "config": rows[0].get("lorat_config") if rows else "",
        "target_tracks": rows[0].get("target_tracks") if rows else "",
        "reid_modes": ",".join(reid_modes),
        "samples": len(rows),
        "mean_head_iou": mean(parse_float(row.get("head_iou")) for row in rows),
        "mean_final_iou": mean(parse_float(row.get("final_iou")) for row in rows),
        "mean_oracle_iou": mean(parse_float(row.get("oracle_best_iou")) for row in rows),
        "accepted_rate": sum(1 for row in rows if parse_bool(row.get("accepted"))) / len(rows) if rows else 0.0,
        "held_rate": sum(1 for row in rows if parse_bool(row.get("held"))) / len(rows) if rows else 0.0,
        "good_head_final_bad": count("good_head_final_bad"),
        "good_head_final_bad_rate": rate("good_head_final_bad"),
        "good_oracle_final_bad": count("good_oracle_final_bad"),
        "good_oracle_final_bad_rate": rate("good_oracle_final_bad"),
        "held_good_head": count("held_good_head"),
        "held_good_oracle": count("held_good_oracle"),
        "accepted_bad_final_rate": rate("accepted_bad_final"),
        "scale_limited_rate": rate("scale_limited"),
        "scale_override_high_conf": count("scale_override_high_conf"),
        "oversized_final_rate": rate("oversized_final"),
        "oversized_search_rate": rate("oversized_search"),
        "identity_wrong_rate": rate("identity_wrong"),
        "memheld_rate": rate("memheld"),
        "viewchange_rate": rate("viewchange"),
        "top_failure": ";".join(f"{name}:{value}" for name, value in failures.most_common(5)),
        "top_state": ";".join(f"{name}:{value}" for name, value in states.most_common(5)),
        "top_scale_state": ";".join(f"{name}:{value}" for name, value in scale_states.most_common(5)),
        "top_reject_state": ";".join(f"{name}:{value}" for name, value in reject_states.most_common(5)),
    }


def important_rows(rows: Sequence[Mapping[str, str]], args: argparse.Namespace, limit: int = 30) -> List[Dict[str, object]]:
    scored: List[Tuple[float, Mapping[str, str], Dict[str, bool]]] = []
    for row in rows:
        flags = row_flags(row, args)
        head_iou = parse_float(row.get("head_iou")) or 0.0
        oracle_iou = parse_float(row.get("oracle_best_iou")) or 0.0
        final_iou = parse_float(row.get("final_iou")) or 0.0
        gap = max(head_iou, oracle_iou) - final_iou
        score = gap
        if flags["held_good_head"] or flags["held_good_oracle"]:
            score += 1.0
        if flags["scale_override_high_conf"]:
            score += 0.5
        if flags["oversized_search"]:
            score += 0.25
        if score <= 0:
            continue
        scored.append((score, row, flags))

    out: List[Dict[str, object]] = []
    for score, row, flags in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        out.append(
            {
                "score": score,
                "mode": row.get("v9_diagnostic_mode"),
                "reid": row.get("reid_mode"),
                "frame": row.get("frame"),
                "head_iou": row.get("head_iou"),
                "final_iou": row.get("final_iou"),
                "oracle_iou": row.get("oracle_best_iou"),
                "accepted": row.get("accepted"),
                "held": row.get("held"),
                "reject_state": row.get("reject_state"),
                "final_state": row.get("final_state"),
                "scale_state": row.get("v9_scale_gate_state"),
                "scale_reason": row.get("v9_scale_gate_reason"),
                "failure": row.get("iou_failure_bucket"),
                "gt_bbox": row.get("gt_bbox"),
                "search_window": row.get("search_window"),
                "head_bbox": row.get("head_bbox"),
                "final_bbox": row.get("final_bbox"),
                "flags": ",".join(name for name, value in flags.items() if value),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, object]], fields: Sequence[str], max_rows: int = 30) -> List[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(fmt(row.get(field)).replace("|", "\\|") for field in fields) + " |")
    return lines


def write_summary(path: Path, summaries: Sequence[Mapping[str, object]], examples: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# V9 Tracker-Policy Failure Analysis",
        "",
        "This report focuses on cases where the V9 head/oracle candidate was usable but tracker policy, scale gates, hold logic, or search-window propagation produced a poor final box.",
        "",
        "## Group Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            summaries,
            [
                "mode",
                "reid_modes",
                "samples",
                "mean_head_iou",
                "mean_final_iou",
                "mean_oracle_iou",
                "held_rate",
                "good_head_final_bad",
                "good_oracle_final_bad",
                "scale_override_high_conf",
                "oversized_search_rate",
                "identity_wrong_rate",
            ],
        )
    )
    lines.extend(["", "## Top Tracker-Policy Failure Examples", ""])
    lines.extend(
        markdown_table(
            examples,
            [
                "mode",
                "reid",
                "frame",
                "head_iou",
                "final_iou",
                "oracle_iou",
                "accepted",
                "held",
                "scale_state",
                "scale_reason",
                "failure",
                "final_state",
            ],
            max_rows=30,
        )
    )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `good_head_final_bad`: the local head overlapped GT well, but final output was poor.",
            "- `good_oracle_final_bad`: at least one candidate/oracle source was good, but final output was poor.",
            "- `scale_override_high_conf`: scale sanity was bypassed because the head was very confident.",
            "- `oversized_search_rate`: search windows became much larger than the GT box, often after bad scale propagation.",
            "- If `gt_window` is much better than `normal`, runtime search-window propagation is a bottleneck.",
            "- If `gt_window` has good head/oracle IoU but bad final IoU, final tracker policy is a bottleneck.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = sorted(args.result_root.rglob("candidate_diagnostics.csv"))
    if not paths:
        raise SystemExit(f"No candidate_diagnostics.csv files found under {args.result_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str, str, int], List[Mapping[str, str]]] = defaultdict(list)
    all_rows: List[Mapping[str, str]] = []
    for path in paths:
        mode = mode_for_path(path)
        for row in read_csv(path):
            row = dict(row)
            row.setdefault("v9_diagnostic_mode", mode)
            key = group_key(row, mode)
            grouped[key].append(row)
            all_rows.append(row)

    summaries = [summarize_group(rows, key[0], args) for key, rows in sorted(grouped.items())]
    examples = important_rows(all_rows, args)

    write_csv(args.output_dir / "v9_tracker_policy_summary.csv", summaries)
    write_csv(args.output_dir / "v9_tracker_policy_examples.csv", examples)
    write_summary(args.output_dir / "v9_tracker_policy_summary.md", summaries, examples)

    print(f"Wrote {args.output_dir / 'v9_tracker_policy_summary.md'}")
    print(f"Wrote {args.output_dir / 'v9_tracker_policy_summary.csv'}")
    print(f"Wrote {args.output_dir / 'v9_tracker_policy_examples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
