from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import benchmark_lorat_v9_harness as benchmark
import bounding_box_v9_lorat_local_search as v9
import export_bdd100k_to_mot_sequences as bdd100k_export
import export_lasot_to_mot_sequences as lasot_export
import export_tao_to_mot_sequences as tao_export
import mot_common as mot


DEFAULT_V9_OUTPUT_ROOT = mot.PROJECT_ROOT / "outputs" / "benchmarks" / "lorat-v9"
benchmark.v8 = v9
benchmark.EXECUTION_MODE = v9.V9_EXECUTION_MODE
benchmark.DEFAULT_OUTPUT_ROOT = DEFAULT_V9_OUTPUT_ROOT
V9_TRAINED_DEFAULT_LOCAL_GRID_SIZE = 17
V9_BENCHMARK_LOCAL_GRID_SIZE = V9_TRAINED_DEFAULT_LOCAL_GRID_SIZE
V9_ALLOW_CHECKPOINT_FALLBACK = False


def v9_head_config_key(lorat_config: str) -> str:
    return lorat_config.replace("-", "_")


def resolve_v9_head_weights(args: argparse.Namespace, lorat_config: str) -> Optional[Path]:
    if args.v8_head_weights is not None:
        return args.v8_head_weights
    if args.v8_head_weights_root is None:
        return None
    root = args.v8_head_weights_root
    config_key = v9_head_config_key(lorat_config)
    checkpoint_choice = getattr(args, "v8_head_checkpoint", "best")
    if checkpoint_choice == "best":
        suffixes = ("best_by_rollout_identity", "best_by_val_iou", "latest")
    elif checkpoint_choice == "best_by_val_iou":
        suffixes = ("best_by_val_iou", "best_by_rollout_identity", "latest")
    elif checkpoint_choice == "best_by_rollout_identity":
        suffixes = ("best_by_rollout_identity", "best_by_val_iou", "latest")
    else:
        suffixes = ("latest", "best_by_rollout_identity", "best_by_val_iou")
    stems = (f"v9_local_head_{config_key}", f"v9_head_{config_key}")
    candidates = []
    for suffix in suffixes:
        for stem in stems:
            candidates.extend(
                [
                    root / config_key / f"{stem}_{suffix}.pt",
                    root / "checkpoints" / config_key / f"{stem}_{suffix}.pt",
                    root / f"{stem}_{suffix}.pt",
                ]
            )
    candidates.extend(
        [
            root / "models" / "lorat" / f"v9_local_head_{config_key}.pt",
            root / f"v9_local_head_{config_key}.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            suffix = "unknown"
            for choice in suffixes:
                if candidate.name.endswith(f"_{choice}.pt"):
                    suffix = choice
                    break
            strict_choice = checkpoint_choice in {"best_by_rollout_identity", "best_by_val_iou", "latest"}
            if strict_choice and suffix != checkpoint_choice and not V9_ALLOW_CHECKPOINT_FALLBACK:
                expected_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.name.endswith(f"_{checkpoint_choice}.pt")
                ]
                expected_text = ", ".join(str(path) for path in expected_candidates)
                raise RuntimeError(
                    "Resolved V9 checkpoint would fall back away from the requested checkpoint. "
                    f"requested={checkpoint_choice} found_fallback={candidate}. "
                    f"Expected one of: {expected_text}. "
                    "Pass --v9-allow-checkpoint-fallback only for an intentional fallback run."
                )
            try:
                import torch

                checkpoint = torch.load(str(candidate), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    print(
                        "Resolved V9 checkpoint: "
                        f"file={candidate} preference={suffix} "
                        f"epoch={checkpoint.get('epoch')} steps={checkpoint.get('steps')} "
                        f"val_iou={checkpoint.get('val_mean_iou')} "
                        f"best_val_iou={checkpoint.get('best_val_iou')} "
                        f"rollout_correct={checkpoint.get('val_rollout_correct_rate_iou30')} "
                        f"rollout_switch={checkpoint.get('val_rollout_identity_switch_rate')} "
                        f"rollout_track_loss={checkpoint.get('val_rollout_track_loss_rate')} "
                        f"rollout_frames_until_loss={checkpoint.get('val_rollout_mean_frames_until_loss')} "
                        f"rollout_mode={checkpoint.get('rollout_validation_mode')} "
                        f"checkpoint_selection_score={checkpoint.get('checkpoint_selection_score')}",
                        flush=True,
                    )
                else:
                    print(f"Resolved V9 checkpoint: file={candidate} preference={suffix} non_dict_checkpoint", flush=True)
            except Exception as exc:
                print(f"Resolved V9 checkpoint: file={candidate} preference={suffix} metadata_error={exc}", flush=True)
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"No trained V9 local-search head found for {lorat_config}. Checked: {candidate_text}")


benchmark.resolve_v8_head_weights = resolve_v9_head_weights
_benchmark_tracker_args_for_run = benchmark.tracker_args_for_run


def v9_preview_video_path(paths, sequence: str, config: str, target_tracks: int, max_frames: int) -> Path:
    frame_part = f"frames{max_frames}" if max_frames > 0 else "full"
    name = benchmark.bench.slugify(f"{sequence}_{config}_v9_N{target_tracks}_{frame_part}_preview") + ".mp4"
    return benchmark.bench.unique_path(paths.video_dir / name)


benchmark.preview_video_path = v9_preview_video_path


def tracker_args_for_v9_run(
    args: argparse.Namespace,
    lorat_config: str,
    target_tracks: int,
) -> argparse.Namespace:
    run_args = _benchmark_tracker_args_for_run(args, lorat_config, target_tracks)
    run_args.v9_local_grid_size = max(4, int(V9_BENCHMARK_LOCAL_GRID_SIZE))
    return run_args


benchmark.tracker_args_for_run = tracker_args_for_v9_run


def build_v9_front_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tao-root", type=Path, help="Export one or two TAO/TAO-OW videos to MOT format before benchmarking.")
    parser.add_argument("--tao-split", default="validation")
    parser.add_argument("--tao-use-freeform", action="store_true")
    parser.add_argument("--tao-example-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "TAO_OW_MOT_EXAMPLES")
    parser.add_argument("--tao-example-videos", type=int, default=2)
    parser.add_argument("--tao-example-min-tracks", type=int, default=1)
    parser.add_argument("--tao-example-min-annotated-frames", type=int, default=10)
    parser.add_argument("--tao-example-max-frames", type=int, default=300)
    parser.add_argument("--tao-example-copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--lasot-root", type=Path, help="Export one or two LaSOT videos to MOT format before benchmarking.")
    parser.add_argument("--lasot-split", default="val")
    parser.add_argument("--lasot-example-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "LaSOT_MOT_EXAMPLES")
    parser.add_argument("--lasot-example-videos", type=int, default=1)
    parser.add_argument("--lasot-example-min-visible-frames", type=int, default=10)
    parser.add_argument("--lasot-example-max-frames", type=int, default=300)
    parser.add_argument("--lasot-example-copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--lasot-val-fraction", type=float, default=0.20)
    parser.add_argument("--bdd100k-root", type=Path, help="Export BDD100K MOT videos to MOT format before benchmarking.")
    parser.add_argument("--bdd100k-split", default="val")
    parser.add_argument("--bdd100k-example-root", type=Path, default=mot.PROJECT_ROOT / "data" / "derived" / "BDD100K_MOT_EXAMPLES")
    parser.add_argument("--bdd100k-example-videos", type=int, default=2)
    parser.add_argument("--bdd100k-example-min-tracks", type=int, default=5)
    parser.add_argument("--bdd100k-example-min-annotated-frames", type=int, default=10)
    parser.add_argument("--bdd100k-example-max-frames", type=int, default=300)
    parser.add_argument("--bdd100k-example-copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--bdd100k-categories", default=",".join(bdd100k_export.DEFAULT_BDD100K_CATEGORIES))
    parser.add_argument("--bdd100k-include-distractors", action="store_true")
    parser.add_argument("--v9-head-weights", type=Path, help="Alias for --v8-head-weights when benchmarking V9.")
    parser.add_argument("--v9-head-weights-root", type=Path, help="Alias for --v8-head-weights-root when benchmarking V9.")
    parser.add_argument(
        "--v9-head-checkpoint",
        choices=("best", "best_by_val_iou", "best_by_rollout_identity", "latest"),
        default="best_by_rollout_identity",
        help="Alias for --v8-head-checkpoint when benchmarking V9.",
    )
    parser.add_argument(
        "--v9-allow-checkpoint-fallback",
        action="store_true",
        help="Allow benchmarking with the next checkpoint preference when the requested V9 checkpoint is missing.",
    )
    parser.add_argument(
        "--v9-local-grid-size",
        type=int,
        default=V9_TRAINED_DEFAULT_LOCAL_GRID_SIZE,
        help="Local search grid size for V9 benchmark inference. Defaults to the current V9 training grid.",
    )
    parser.add_argument(
        "--draw-v9-diagnostics",
        action="store_true",
        help="Alias for --draw-candidate-diagnostics in the shared benchmark harness.",
    )
    return parser


def parse_v9_front_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = build_v9_front_parser()
    return parser.parse_known_args(argv)


def main() -> int:
    global V9_ALLOW_CHECKPOINT_FALLBACK, V9_BENCHMARK_LOCAL_GRID_SIZE
    original_argv = list(sys.argv)
    if any(arg in {"-h", "--help"} for arg in original_argv[1:]):
        print("V9 wrapper options:")
        build_v9_front_parser().print_help()
        print("\nShared benchmark options:")
    v9_args, remaining = parse_v9_front_args(original_argv[1:])
    V9_BENCHMARK_LOCAL_GRID_SIZE = max(4, int(v9_args.v9_local_grid_size))
    V9_ALLOW_CHECKPOINT_FALLBACK = bool(v9_args.v9_allow_checkpoint_fallback)
    if v9_args.v9_head_weights is not None and "--v8-head-weights" not in remaining:
        remaining = [*remaining, "--v8-head-weights", str(v9_args.v9_head_weights)]
    if v9_args.v9_head_weights_root is not None and "--v8-head-weights-root" not in remaining:
        remaining = [*remaining, "--v8-head-weights-root", str(v9_args.v9_head_weights_root)]
    if v9_args.v9_head_checkpoint is not None and "--v8-head-checkpoint" not in remaining:
        remaining = [*remaining, "--v8-head-checkpoint", v9_args.v9_head_checkpoint]
    if "--draw-candidate-diagnostics" not in remaining:
        remaining = [*remaining, "--draw-candidate-diagnostics"]
    source_roots = [v9_args.tao_root, v9_args.lasot_root, v9_args.bdd100k_root]
    if sum(1 for value in source_roots if value is not None) > 1:
        raise RuntimeError("Use only one source-export root per benchmark invocation: --tao-root, --lasot-root, or --bdd100k-root.")
    if v9_args.tao_root is not None:
        exported = tao_export.export_videos(
            v9_args.tao_root,
            v9_args.tao_example_root,
            v9_args.tao_split,
            v9_args.tao_example_videos,
            v9_args.tao_example_min_tracks,
            v9_args.tao_example_min_annotated_frames,
            v9_args.tao_example_max_frames,
            v9_args.tao_example_copy_mode,
            v9_args.tao_use_freeform,
        )
        benchmark_split = "val" if v9_args.tao_split in {"val", "validation"} else v9_args.tao_split
        remaining = [
            *remaining,
            "--dataset-root",
            str(v9_args.tao_example_root),
            "--split",
            benchmark_split,
            *[item for name in exported for item in ("--sequence", name)],
        ]
    if v9_args.lasot_root is not None:
        exported = lasot_export.export_sequences(
            v9_args.lasot_root,
            v9_args.lasot_example_root,
            v9_args.lasot_split,
            v9_args.lasot_example_videos,
            v9_args.lasot_example_min_visible_frames,
            v9_args.lasot_example_max_frames,
            v9_args.lasot_example_copy_mode,
            v9_args.lasot_val_fraction,
        )
        benchmark_split = "val" if v9_args.lasot_split in {"val", "valid", "validation"} else v9_args.lasot_split
        remaining = [
            *remaining,
            "--dataset-root",
            str(v9_args.lasot_example_root),
            "--split",
            benchmark_split,
            *[item for name in exported for item in ("--sequence", name)],
        ]
    if v9_args.bdd100k_root is not None:
        exported = bdd100k_export.export_videos(
            v9_args.bdd100k_root,
            v9_args.bdd100k_example_root,
            v9_args.bdd100k_split,
            v9_args.bdd100k_example_videos,
            v9_args.bdd100k_example_min_tracks,
            v9_args.bdd100k_example_min_annotated_frames,
            v9_args.bdd100k_example_max_frames,
            v9_args.bdd100k_example_copy_mode,
            bdd100k_export.parse_csv_names(v9_args.bdd100k_categories, bdd100k_export.DEFAULT_BDD100K_CATEGORIES),
            bdd100k_export.DEFAULT_BDD100K_DISTRACTORS,
            v9_args.bdd100k_include_distractors,
        )
        benchmark_split = "val" if v9_args.bdd100k_split in {"valid", "validation"} else v9_args.bdd100k_split
        remaining = [
            *remaining,
            "--dataset-root",
            str(v9_args.bdd100k_example_root),
            "--split",
            benchmark_split,
            *[item for name in exported for item in ("--sequence", name)],
        ]
    try:
        sys.argv = [original_argv[0], *remaining]
        return benchmark.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
