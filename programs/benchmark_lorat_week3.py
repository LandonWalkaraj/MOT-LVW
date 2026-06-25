from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import benchmark_lorat_v8 as v8_benchmark
import mot_common as mot


DEFAULT_WEEK3_OUTPUT_ROOT = mot.PROJECT_ROOT / "outputs" / "benchmarks" / "week3-reid-recovery"


def _has_option(args: Sequence[str], names: Iterable[str]) -> bool:
    option_names = set(names)
    for arg in args:
        for name in option_names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def _inject_default(args: Sequence[str], name: str, value: str) -> List[str]:
    if _has_option(args, [name]):
        return []
    return [name, value]


def build_week3_args(user_args: Sequence[str]) -> List[str]:
    """Return argv for the comprehensive Week 3 V8 benchmark.

    The underlying benchmark implementation lives in benchmark_lorat_v8.py.
    This entrypoint only sets Week-3-safe defaults:

    - run ReID on/off ablation for the same tracker cases
    - sample identity every frame so natural occlusion diagnostics are reported in frames
    - run a controlled occlusion-duration sweep for deliverable 3(c)
    - keep every-frame area observations for cross-checking lost/occluded periods
    - write into an explicit Week 3 output folder
    """

    injected: List[str] = []
    injected.extend(_inject_default(user_args, "--output-root", str(DEFAULT_WEEK3_OUTPUT_ROOT)))
    injected.extend(_inject_default(user_args, "--identity-sample-interval", "1"))

    if not _has_option(
        user_args,
        [
            "--reid-ablation",
            "--week3-reid-ablation",
            "--disable-reid",
            "--disable-identity-arbitration",
        ],
    ):
        injected.append("--reid-ablation")

    if not _has_option(user_args, ["--full-area-observations"]):
        injected.append("--full-area-observations")

    if not _has_option(user_args, ["--controlled-occlusion-durations"]):
        injected.extend(["--controlled-occlusion-durations", "0,5,10,20,40,80"])
    if not _has_option(user_args, ["--controlled-occlusion-trials-per-duration"]):
        injected.extend(["--controlled-occlusion-trials-per-duration", "3"])
    if not _has_option(user_args, ["--controlled-occlusion-recovery-frames"]):
        injected.extend(["--controlled-occlusion-recovery-frames", "30"])
    if not _has_option(user_args, ["--controlled-occlusion-warmup-frames"]):
        injected.extend(["--controlled-occlusion-warmup-frames", "10"])

    return [*injected, *user_args]


def main() -> int:
    week3_args = build_week3_args(sys.argv[1:])
    print("LoRAT Week 3 benchmark entrypoint", flush=True)
    print(
        "Defaults: ReID ablation on, identity sampled every frame, controlled occlusion sweep on, "
        "full area observations on.",
        flush=True,
    )
    sys.argv = [sys.argv[0], *week3_args]
    return v8_benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
