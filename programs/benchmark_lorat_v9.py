from __future__ import annotations

import argparse
import sys
from pathlib import Path

import benchmark_lorat_v8 as benchmark
import bounding_box_v9_lorat_local_search as v9
import export_tao_to_mot_sequences as tao_export
import mot_common as mot


benchmark.v8 = v9
benchmark.EXECUTION_MODE = v9.V9_EXECUTION_MODE


def parse_v9_front_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
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
    return parser.parse_known_args(argv)


def main() -> int:
    original_argv = list(sys.argv)
    v9_args, remaining = parse_v9_front_args(original_argv[1:])
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
    try:
        sys.argv = [original_argv[0], *remaining]
        return benchmark.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
