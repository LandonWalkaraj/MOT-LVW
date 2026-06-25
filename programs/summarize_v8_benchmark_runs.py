from __future__ import annotations

import argparse
import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CSV_NAMES = {
    "timing": "timing_by_object_count.csv",
    "identity": "identity_recovery_summary.csv",
    "occlusion": "controlled_occlusion_survival.csv",
    "area": "area_reliability.csv",
    "proof": "week2_shared_backbone_proof.csv",
    "candidate": "candidate_diagnostics.csv",
}


@dataclass(frozen=True)
class LoadedCsv:
    run_label: str
    source: Path
    member: str
    rows: List[Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one or more downloaded LoRAT V8/Week 3 benchmark result folders or zip files "
            "into compact comparison CSV/Markdown tables."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Benchmark result folders or .zip files. If omitted with --downloads, recent V8 zips in Downloads are scanned.",
    )
    parser.add_argument(
        "--downloads",
        action="store_true",
        help="Also scan ~/Downloads for v8/lorat-v8 benchmark zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "benchmarks" / "v8_comparison",
        help="Where comparison CSV and Markdown files should be written.",
    )
    parser.add_argument(
        "--limit-downloads",
        type=int,
        default=12,
        help="Maximum recent Downloads zips to include when --downloads is used.",
    )
    return parser.parse_args()


def candidate_download_zips(limit: int) -> List[Path]:
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        return []
    candidates = [
        path
        for path in downloads.glob("*.zip")
        if any(token in path.name.lower() for token in ("v8", "lorat-v8", "week3"))
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[: max(0, int(limit))]


def run_label_for_path(path: Path) -> str:
    return path.stem if path.suffix.lower() == ".zip" else path.name


def read_csv_from_text(text: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def iter_csvs_from_dir(path: Path, run_label: str) -> Iterable[LoadedCsv]:
    for name in CSV_NAMES.values():
        for csv_path in sorted(path.rglob(name)):
            try:
                rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
            except UnicodeDecodeError:
                rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8-sig")))
            yield LoadedCsv(run_label=run_label, source=path, member=str(csv_path.relative_to(path)), rows=rows)


def iter_csvs_from_zip(path: Path, run_label: str) -> Iterable[LoadedCsv]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if Path(member).name not in CSV_NAMES.values():
                continue
            with archive.open(member) as file:
                text = file.read().decode("utf-8-sig")
            yield LoadedCsv(run_label=run_label, source=path, member=member, rows=read_csv_from_text(text))


def load_csvs(paths: Sequence[Path]) -> List[LoadedCsv]:
    loaded: List[LoadedCsv] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            print(f"Skipping missing path: {path}")
            continue
        run_label = run_label_for_path(path)
        if path.is_dir():
            loaded.extend(iter_csvs_from_dir(path, run_label))
        elif path.suffix.lower() == ".zip":
            loaded.extend(iter_csvs_from_zip(path, run_label))
        else:
            print(f"Skipping unsupported path: {path}")
    return loaded


def float_or_blank(value: object, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_int(row: Mapping[str, str], field: str, default: int = 0) -> int:
    try:
        return int(float(row.get(field, "") or default))
    except ValueError:
        return default


def key_from_row(run_label: str, row: Mapping[str, str]) -> Tuple[str, str, str, str, int]:
    return (
        run_label,
        row.get("sequence", ""),
        row.get("lorat_config", ""),
        row.get("reid_mode", ""),
        get_int(row, "target_tracks"),
    )


def by_kind(csvs: Sequence[LoadedCsv]) -> Dict[str, List[LoadedCsv]]:
    grouped: Dict[str, List[LoadedCsv]] = {key: [] for key in CSV_NAMES}
    for item in csvs:
        basename = Path(item.member).name
        for kind, name in CSV_NAMES.items():
            if basename == name:
                grouped[kind].append(item)
                break
    return grouped


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def merge_timing_identity(csvs: Sequence[LoadedCsv]) -> List[Dict[str, object]]:
    grouped = by_kind(csvs)
    rows_by_key: Dict[Tuple[str, str, str, str, int], Dict[str, object]] = {}

    for item in grouped["timing"]:
        for row in item.rows:
            key = key_from_row(item.run_label, row)
            output = rows_by_key.setdefault(
                key,
                {
                    "run": item.run_label,
                    "sequence": key[1],
                    "lorat_config": key[2],
                    "reid_mode": key[3],
                    "target_tracks": key[4],
                    "source": str(item.source),
                },
            )
            output.update(
                {
                    "actual_tracks": row.get("actual_tracks", ""),
                    "frames": row.get("frames", ""),
                    "fps_tracking": row.get("fps_tracking", ""),
                    "tracking_ms_per_bbox": row.get("tracking_ms_per_bbox", ""),
                    "mean_iou": row.get("mean_iou", ""),
                    "iou50": row.get("iou50", ""),
                    "gpu_memory_peak_allocated_mb": row.get("gpu_memory_peak_allocated_mb", ""),
                    "proof_shared_backbone_ok_rate": row.get("proof_shared_backbone_ok_rate", ""),
                    "proof_batched_head_ok_rate": row.get("proof_batched_head_ok_rate", ""),
                    "head_items_per_update_frame": row.get("object_head_items_per_update_frame", ""),
                    "dinov2_crop_reid_items": row.get("dinov2_crop_reid_forward_items", ""),
                    "profile_candidate_extract_ms": row.get("profile_candidate_extract_ms_per_update", ""),
                    "profile_template_match_ms": row.get("profile_template_match_ms_per_update", ""),
                    "profile_dinov2_crop_reid_ms": row.get("profile_dinov2_crop_reid_ms_per_update", ""),
                    "profile_identity_resolve_ms": row.get("profile_identity_resolve_ms_per_update", ""),
                }
            )

    for item in grouped["identity"]:
        for row in item.rows:
            key = key_from_row(item.run_label, row)
            output = rows_by_key.setdefault(
                key,
                {
                    "run": item.run_label,
                    "sequence": key[1],
                    "lorat_config": key[2],
                    "reid_mode": key[3],
                    "target_tracks": key[4],
                    "source": str(item.source),
                },
            )
            output.update(
                {
                    "identity_samples": row.get("samples", ""),
                    "correct_rate": row.get("correct_rate", ""),
                    "jump_rate": row.get("jump_rate", ""),
                    "identity_switches": row.get("identity_switches", ""),
                    "identity_switches_per_1000_samples": row.get("identity_switches_per_1000_samples", ""),
                    "track_loss_rate": row.get("track_loss_rate", ""),
                    "identity_mean_iou": row.get("mean_iou", ""),
                }
            )

    return [rows_by_key[key] for key in sorted(rows_by_key)]


def flatten_csv_kind(csvs: Sequence[LoadedCsv], kind: str) -> List[Dict[str, object]]:
    grouped = by_kind(csvs)
    rows: List[Dict[str, object]] = []
    for item in grouped.get(kind, []):
        for row in item.rows:
            output: Dict[str, object] = {"run": item.run_label, "source": str(item.source)}
            output.update(row)
            rows.append(output)
    return rows


def best_rows(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    sortable: List[Tuple[float, float, Mapping[str, object]]] = []
    for row in rows:
        try:
            iou = float(row.get("mean_iou", "") or row.get("identity_mean_iou", "") or -1.0)
        except (TypeError, ValueError):
            iou = -1.0
        try:
            fps = float(row.get("fps_tracking", "") or -1.0)
        except (TypeError, ValueError):
            fps = -1.0
        sortable.append((iou, fps, row))
    sortable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in sortable[:12]]


def markdown_table(rows: Sequence[Mapping[str, object]], fields: Sequence[str], max_rows: int = 20) -> List[str]:
    if not rows:
        return ["_No rows found._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:max_rows]:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if field in {
                "fps_tracking",
                "tracking_ms_per_bbox",
                "mean_iou",
                "iou50",
                "correct_rate",
                "jump_rate",
                "track_loss_rate",
                "proof_shared_backbone_ok_rate",
                "proof_batched_head_ok_rate",
                "survival_rate",
                "recovery_rate",
                "mean_post_occlusion_iou",
            }:
                value = float_or_blank(value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_summary_md(
    path: Path,
    input_paths: Sequence[Path],
    timing_identity_rows: Sequence[Mapping[str, object]],
    controlled_rows: Sequence[Mapping[str, object]],
    area_rows: Sequence[Mapping[str, object]],
) -> None:
    lines: List[str] = [
        "# V8 Benchmark Comparison",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in input_paths)
    lines.extend(
        [
            "",
            "## Best Timing/Quality Rows",
            "",
            *markdown_table(
                best_rows(timing_identity_rows),
                [
                    "run",
                    "sequence",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "fps_tracking",
                    "mean_iou",
                    "iou50",
                    "correct_rate",
                    "track_loss_rate",
                    "identity_switches",
                ],
                max_rows=12,
            ),
            "",
            "## Week 2 Proof Snapshot",
            "",
            *markdown_table(
                timing_identity_rows,
                [
                    "run",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "proof_shared_backbone_ok_rate",
                    "proof_batched_head_ok_rate",
                    "head_items_per_update_frame",
                ],
                max_rows=20,
            ),
            "",
            "## Controlled Occlusion Snapshot",
            "",
            *markdown_table(
                controlled_rows,
                [
                    "run",
                    "sequence",
                    "lorat_config",
                    "reid_mode",
                    "target_tracks",
                    "duration_frames",
                    "survival_rate",
                    "recovery_rate",
                    "mean_post_occlusion_iou",
                ],
                max_rows=30,
            ),
            "",
            "## Area Reliability Rows",
            "",
            f"Area reliability rows found: {len(area_rows)}",
            "",
            "## Output Files",
            "",
            "- `comparison_timing_identity.csv`",
            "- `comparison_controlled_occlusion.csv`",
            "- `comparison_area_reliability.csv`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_paths: List[Path] = list(args.paths)
    if args.downloads:
        input_paths.extend(candidate_download_zips(args.limit_downloads))
    if not input_paths:
        raise SystemExit("No benchmark paths supplied. Pass folders/zips or use --downloads.")

    csvs = load_csvs(input_paths)
    if not csvs:
        raise SystemExit("No benchmark CSVs found in supplied paths.")

    output_dir = args.output_dir.resolve()
    timing_identity_rows = merge_timing_identity(csvs)
    controlled_rows = flatten_csv_kind(csvs, "occlusion")
    area_rows = flatten_csv_kind(csvs, "area")

    write_csv(
        output_dir / "comparison_timing_identity.csv",
        timing_identity_rows,
        [
            "run",
            "source",
            "sequence",
            "lorat_config",
            "reid_mode",
            "target_tracks",
            "actual_tracks",
            "frames",
            "fps_tracking",
            "tracking_ms_per_bbox",
            "mean_iou",
            "iou50",
            "gpu_memory_peak_allocated_mb",
            "proof_shared_backbone_ok_rate",
            "proof_batched_head_ok_rate",
            "head_items_per_update_frame",
            "dinov2_crop_reid_items",
            "profile_candidate_extract_ms",
            "profile_template_match_ms",
            "profile_dinov2_crop_reid_ms",
            "profile_identity_resolve_ms",
            "identity_samples",
            "correct_rate",
            "jump_rate",
            "identity_switches",
            "identity_switches_per_1000_samples",
            "track_loss_rate",
            "identity_mean_iou",
        ],
    )
    if controlled_rows:
        write_csv(
            output_dir / "comparison_controlled_occlusion.csv",
            controlled_rows,
            sorted({field for row in controlled_rows for field in row}),
        )
    if area_rows:
        write_csv(
            output_dir / "comparison_area_reliability.csv",
            area_rows,
            sorted({field for row in area_rows for field in row}),
        )

    write_summary_md(
        output_dir / "comparison_summary.md",
        [path.expanduser().resolve() for path in input_paths],
        timing_identity_rows,
        controlled_rows,
        area_rows,
    )
    print(f"Wrote comparison summary to {output_dir / 'comparison_summary.md'}")
    print(f"Wrote timing/identity CSV to {output_dir / 'comparison_timing_identity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
