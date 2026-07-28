from __future__ import annotations

import argparse
import csv
import io
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CSV_BASENAMES = {
    "candidate": "candidate_diagnostics.csv",
    "identity": "identity_recovery_summary.csv",
    "timing": "timing_by_object_count.csv",
}
DIAGNOSTIC_MODES = ("normal", "gt_window", "gt_identity")


@dataclass(frozen=True)
class LoadedCsv:
    kind: str
    run_label: str
    source: Path
    member: str
    rows: List[Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze V9 diagnostic benchmark outputs. The report compares normal, gt_window, "
            "and gt_identity runs and highlights whether failures are search-window, head, "
            "decode, identity arbitration, or ReID related."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="V9 diagnostic result folders or zip files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "benchmark_reviews" / "v9_diagnostic_analysis",
        help="Directory for summary CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--oracle-gap-threshold",
        type=float,
        default=0.15,
        help="Mean oracle-final IoU gap above this value is treated as an arbitration/selection warning.",
    )
    parser.add_argument(
        "--mode-gap-threshold",
        type=float,
        default=0.10,
        help="Mean IoU improvement between diagnostic modes above this value is treated as meaningful.",
    )
    return parser.parse_args()


def read_csv_text(text: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def run_label_for_path(path: Path) -> str:
    return path.stem if path.suffix.lower() == ".zip" else path.name


def infer_mode(member: str, rows: Sequence[Mapping[str, str]]) -> str:
    for row in rows:
        value = str(row.get("v9_diagnostic_mode") or "").strip()
        if value:
            return value
    parts = {part.lower() for part in Path(member).parts}
    for mode in DIAGNOSTIC_MODES:
        if mode in parts:
            return mode
    text = member.lower()
    for mode in DIAGNOSTIC_MODES:
        if f"/{mode}/" in text.replace("\\", "/"):
            return mode
    return "unknown"


def iter_csvs_from_dir(path: Path, run_label: str) -> Iterable[LoadedCsv]:
    for kind, basename in CSV_BASENAMES.items():
        for csv_path in sorted(path.rglob(basename)):
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
            except UnicodeDecodeError:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
                    rows = list(csv.DictReader(file))
            yield LoadedCsv(kind, run_label, path, str(csv_path.relative_to(path)), rows)


def iter_csvs_from_zip(path: Path, run_label: str) -> Iterable[LoadedCsv]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            basename = Path(member).name
            kind = next((key for key, name in CSV_BASENAMES.items() if name == basename), None)
            if kind is None:
                continue
            with archive.open(member) as file:
                rows = read_csv_text(file.read().decode("utf-8-sig"))
            yield LoadedCsv(kind, run_label, path, member, rows)


def load_csvs(paths: Sequence[Path]) -> List[LoadedCsv]:
    loaded: List[LoadedCsv] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            print(f"Skipping missing path: {path}")
            continue
        label = run_label_for_path(path)
        if path.is_dir():
            loaded.extend(iter_csvs_from_dir(path, label))
        elif path.suffix.lower() == ".zip":
            loaded.extend(iter_csvs_from_zip(path, label))
        else:
            print(f"Skipping unsupported path: {path}")
    return loaded


def parse_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: object, default: int = 0) -> int:
    number = parse_float(value)
    if number is None:
        return default
    return int(number)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def rate_bool(rows: Sequence[Mapping[str, str]], field: str) -> Optional[float]:
    if not rows:
        return None
    values: List[float] = []
    for row in rows:
        text = str(row.get(field, "")).strip().lower()
        if text in {"1", "true", "yes"}:
            values.append(1.0)
        elif text in {"0", "false", "no"}:
            values.append(0.0)
    if not values:
        return None
    return statistics.fmean(values)


def group_key(run_label: str, mode: str, row: Mapping[str, str]) -> Tuple[str, str, str, str, str, int]:
    return (
        run_label,
        mode,
        row.get("sequence", ""),
        row.get("lorat_config", ""),
        row.get("reid_mode", ""),
        parse_int(row.get("target_tracks"), 0),
    )


def summarize_candidates(csvs: Sequence[LoadedCsv]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, str, int], List[Mapping[str, str]]] = defaultdict(list)
    for item in csvs:
        if item.kind != "candidate":
            continue
        mode = infer_mode(item.member, item.rows)
        for row in item.rows:
            grouped[group_key(item.run_label, mode, row)].append(row)

    rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        run_label, mode, sequence, config, reid_mode, tracks = key
        failure_counts = Counter(str(row.get("diagnostic_failure_reason") or "unknown").strip() or "unknown" for row in group_rows)
        guard_counts = Counter(str(row.get("v9_accept_guard_state") or "").strip() for row in group_rows)
        hold_counts = Counter(str(row.get("v9_hold_source") or "").strip() for row in group_rows)
        rows.append(
            {
                "run": run_label,
                "diagnostic_mode": mode,
                "sequence": sequence,
                "lorat_config": config,
                "reid_mode": reid_mode,
                "target_tracks": tracks,
                "samples": len(group_rows),
                "mean_final_iou": mean(parse_float(row.get("final_iou")) for row in group_rows),
                "final_iou50": rate_bool(group_rows, "final_correct_object"),
                "mean_head_iou": mean(parse_float(row.get("head_iou")) for row in group_rows),
                "mean_fused_iou": mean(parse_float(row.get("fused_iou")) for row in group_rows),
                "mean_oracle_best_iou": mean(parse_float(row.get("oracle_best_iou")) for row in group_rows),
                "oracle_iou50": rate_bool(group_rows, "oracle_best_iou50"),
                "mean_oracle_runtime_gap": mean(parse_float(row.get("oracle_runtime_iou_gap")) for row in group_rows),
                "local_owner_override_rate": rate_bool(group_rows, "v9_local_owner_override"),
                "accepted_rate": rate_bool(group_rows, "accepted"),
                "held_rate": rate_bool(group_rows, "held"),
                "top_failure_reason": failure_counts.most_common(1)[0][0] if failure_counts else "",
                "top_failure_count": failure_counts.most_common(1)[0][1] if failure_counts else 0,
                "top_accept_guard": guard_counts.most_common(1)[0][0] if guard_counts else "",
                "top_hold_source": hold_counts.most_common(1)[0][0] if hold_counts else "",
            }
        )
    return rows


def summarize_identity(csvs: Sequence[LoadedCsv]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in csvs:
        if item.kind != "identity":
            continue
        mode = infer_mode(item.member, item.rows)
        for row in item.rows:
            output: Dict[str, object] = {
                "run": item.run_label,
                "diagnostic_mode": mode,
            }
            output.update(row)
            rows.append(output)
    return rows


def summarize_timing(csvs: Sequence[LoadedCsv]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in csvs:
        if item.kind != "timing":
            continue
        mode = infer_mode(item.member, item.rows)
        for row in item.rows:
            output: Dict[str, object] = {
                "run": item.run_label,
                "diagnostic_mode": mode,
            }
            output.update(row)
            rows.append(output)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value: object, digits: int = 3) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def markdown_table(rows: Sequence[Mapping[str, object]], fields: Sequence[str], max_rows: int = 20) -> List[str]:
    if not rows:
        return ["_No rows found._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:max_rows]:
        cells: List[str] = []
        for field in fields:
            value = row.get(field, "")
            if value is None:
                value = ""
            if isinstance(value, float):
                value = fmt(value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def by_mode_candidate(candidate_rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    # For the primary N=1 diagnostic run, use the first row per mode after sorting.
    selected: Dict[str, Mapping[str, object]] = {}
    for row in sorted(candidate_rows, key=lambda item: (str(item.get("run")), int(item.get("target_tracks") or 0), str(item.get("diagnostic_mode")))):
        mode = str(row.get("diagnostic_mode") or "")
        selected.setdefault(mode, row)
    return selected


def generate_findings(
    candidate_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    *,
    oracle_gap_threshold: float,
    mode_gap_threshold: float,
) -> List[str]:
    findings: List[str] = []
    by_mode = by_mode_candidate(candidate_rows)
    normal = by_mode.get("normal")
    gt_window = by_mode.get("gt_window")
    gt_identity = by_mode.get("gt_identity")

    if normal and gt_window:
        normal_iou = parse_float(normal.get("mean_final_iou"))
        gt_window_iou = parse_float(gt_window.get("mean_final_iou"))
        if normal_iou is not None and gt_window_iou is not None:
            gap = gt_window_iou - normal_iou
            if gap >= mode_gap_threshold:
                findings.append(
                    f"Search-window propagation is suspicious: gt_window improves mean final IoU by {gap:.3f} over normal."
                )
            else:
                findings.append(
                    f"Search-window propagation is not the dominant issue in this slice: gt_window-normal mean IoU gap is {gap:.3f}."
                )

    if normal:
        oracle_gap = parse_float(normal.get("mean_oracle_runtime_gap"))
        oracle_iou = parse_float(normal.get("mean_oracle_best_iou"))
        final_iou = parse_float(normal.get("mean_final_iou"))
        if oracle_gap is not None and oracle_gap >= oracle_gap_threshold:
            findings.append(
                f"Candidate selection/arbitration is suspicious: normal oracle-final IoU gap is {oracle_gap:.3f}."
            )
        elif oracle_iou is not None and final_iou is not None:
            findings.append(
                f"Oracle gap is modest or unavailable: normal final IoU={final_iou:.3f}, oracle best IoU={oracle_iou:.3f}."
            )

    if gt_window:
        gt_window_oracle = parse_float(gt_window.get("mean_oracle_best_iou"))
        gt_window_final = parse_float(gt_window.get("mean_final_iou"))
        if gt_window_oracle is not None and gt_window_oracle < 0.30:
            findings.append(
                f"Head/local training remains weak even with GT-centered windows: gt_window oracle IoU is {gt_window_oracle:.3f}."
            )
        elif gt_window_final is not None and gt_window_final < 0.30:
            findings.append(
                f"GT windows help candidates but final localization is still weak: gt_window final IoU is {gt_window_final:.3f}."
            )

    if gt_identity and normal:
        gt_identity_iou = parse_float(gt_identity.get("mean_final_iou"))
        normal_iou = parse_float(normal.get("mean_final_iou"))
        if gt_identity_iou is not None and normal_iou is not None:
            gap = gt_identity_iou - normal_iou
            if gap >= mode_gap_threshold:
                findings.append(
                    f"Identity arbitration is suspicious: gt_identity improves mean final IoU by {gap:.3f} over normal."
                )

    if identity_rows:
        worst = sorted(
            identity_rows,
            key=lambda row: parse_float(row.get("correct_rate")) if parse_float(row.get("correct_rate")) is not None else -1.0,
        )[:3]
        for row in worst:
            correct_rate = parse_float(row.get("correct_rate"))
            if correct_rate is not None and correct_rate < 0.50:
                findings.append(
                    "Low correct-object rate: "
                    f"mode={row.get('diagnostic_mode')} N={row.get('target_tracks')} "
                    f"reid={row.get('reid_mode')} correct={correct_rate:.3f}."
                )

    if not findings:
        findings.append("No dominant failure pattern was detected by the automatic rules. Inspect videos and candidate diagnostics.")
    return findings


def write_summary_md(
    path: Path,
    input_paths: Sequence[Path],
    candidate_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    findings: Sequence[str],
) -> None:
    lines: List[str] = [
        "# V9 Diagnostic Benchmark Analysis",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{input_path.expanduser().resolve()}`" for input_path in input_paths)
    lines.extend(
        [
            "",
            "## Automatic Findings",
            "",
            *[f"- {finding}" for finding in findings],
            "",
            "## Candidate / Oracle Summary",
            "",
            *markdown_table(
                candidate_rows,
                [
                    "run",
                    "diagnostic_mode",
                    "sequence",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "samples",
                    "mean_final_iou",
                    "mean_oracle_best_iou",
                    "mean_oracle_runtime_gap",
                    "top_failure_reason",
                    "top_accept_guard",
                    "top_hold_source",
                ],
                max_rows=30,
            ),
            "",
            "## Identity Summary",
            "",
            *markdown_table(
                identity_rows,
                [
                    "run",
                    "diagnostic_mode",
                    "sequence",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "correct_rate",
                    "identity_switches",
                    "track_loss_rate",
                    "jump_rate",
                    "mean_iou",
                ],
                max_rows=30,
            ),
            "",
            "## Timing Snapshot",
            "",
            *markdown_table(
                timing_rows,
                [
                    "run",
                    "diagnostic_mode",
                    "sequence",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "fps_tracking",
                    "tracking_ms_per_bbox",
                    "proof_shared_backbone_ok_rate",
                    "proof_batched_head_ok_rate",
                ],
                max_rows=30,
            ),
            "",
            "## How To Read This",
            "",
            "- `gt_window` improves over `normal`: search-window propagation is likely hurting tracking.",
            "- oracle best is much better than final: candidate selection, identity arbitration, or ReID steering is likely hurting tracking.",
            "- oracle best is poor even under `gt_window`: the V9 head/training target is likely still weak.",
            "- `gt_identity` improves over `normal`: selected-identity association is likely the bottleneck.",
            "",
            "## Output Files",
            "",
            "- `v9_diagnostic_candidate_summary.csv`",
            "- `v9_diagnostic_identity_summary.csv`",
            "- `v9_diagnostic_timing_summary.csv`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    csvs = load_csvs(args.paths)
    if not csvs:
        raise SystemExit("No V9 diagnostic CSVs found.")

    output_dir = args.output_dir.resolve()
    candidate_rows = summarize_candidates(csvs)
    identity_rows = summarize_identity(csvs)
    timing_rows = summarize_timing(csvs)

    write_csv(output_dir / "v9_diagnostic_candidate_summary.csv", candidate_rows)
    write_csv(output_dir / "v9_diagnostic_identity_summary.csv", identity_rows)
    write_csv(output_dir / "v9_diagnostic_timing_summary.csv", timing_rows)
    findings = generate_findings(
        candidate_rows,
        identity_rows,
        oracle_gap_threshold=float(args.oracle_gap_threshold),
        mode_gap_threshold=float(args.mode_gap_threshold),
    )
    write_summary_md(output_dir / "v9_diagnostic_summary.md", args.paths, candidate_rows, identity_rows, timing_rows, findings)

    print(f"Wrote V9 diagnostic summary to {output_dir / 'v9_diagnostic_summary.md'}")
    print(f"Wrote candidate summary to {output_dir / 'v9_diagnostic_candidate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
