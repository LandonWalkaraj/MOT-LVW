# Week 3 Benchmark Usage

This benchmark is for the Week 3 ReID/recovery deliverables built on V8:

- DINOv2/LoRAT shared-feature crop embeddings for ReID and recovery.
- Identity-switch and track-loss rates with ReID enabled and disabled.
- Occlusion survival in frames.
- Manual re-anchor support in the interface, with event logging handled by V8 runtime.

## Entry Point

Use:

```powershell
python programs\benchmark_lorat_week3.py --device cuda:0 --sequence dancetrack0065 --track-counts 1,2,3,4,5 --compare-configs B-224 --v8-head-weights-root <checkpoint-root> --v8-head-checkpoint best
```

`benchmark_lorat_week3.py` calls `benchmark_lorat_v8.py` with Week-3-specific defaults:

- `--reid-ablation`
- `--identity-sample-interval 1`
- `--full-area-observations`
- `--output-root outputs/benchmarks/week3-reid-recovery`

Those defaults mean each tracker case runs twice: once with ReID enabled and once with ReID disabled.

## Theia

Submit with Slurm only:

```bash
sbatch /work/$USER/theia_v8_week3_benchmark.sbatch
```

From the local Windows project folder, the ready-to-use submit helper is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\submit_theia_v8_week3_benchmark.ps1 `
  -HeadWeightsRoot /work/landonvw/<training-result-folder>/checkpoints `
  -HeadCheckpoint best `
  -ResultRoot /work/landonvw/lorat-v8-week3-results-<label> `
  -Sequences dancetrack0065 `
  -TrackCounts 1,2,3,4,5 `
  -CompareConfigs B-224
```

If `-HeadWeightsRoot` is omitted, the Theia wrapper searches the newest
`/work/$USER/lorat-v8-train-results*/checkpoints` folder that contains the requested config.
For paper comparisons, prefer setting it explicitly so the checkpoint source is unambiguous.

Useful overrides:

```bash
HEAD_WEIGHTS_ROOT=/work/$USER/lorat-v8-train-results-b224-dcfst-48h-666000/checkpoints \
COMPARE_CONFIGS="B-224" \
SEQUENCES="dancetrack0065" \
TRACK_COUNTS="1,2,3,4,5" \
sbatch /work/$USER/theia_v8_week3_benchmark.sbatch
```

If `HEAD_WEIGHTS_ROOT` does not contain the requested heads, the Slurm script searches the newest
`/work/$USER/lorat-v8-train-results*/checkpoints` folders and uses the newest one that has all requested configs.

By default, both the local benchmark and Theia wrapper prefer `v8_head_<config>_best_by_val_iou.pt`.
Use `--v8-head-checkpoint latest` locally, or `HEAD_CHECKPOINT=latest` on Theia, only when the final
epoch checkpoint is intentionally being tested.

## Main Outputs

Every run writes a timestamped folder under the output root.

- `summary.md`: human-readable tables for timing, candidate diagnostics, area reliability, identity, occlusion survival, and 25 FPS capacity.
- `timing_by_object_count.csv`: FPS, ms/box, GPU memory, shared-backbone proof, and profile timers.
- `identity_observations_sampled.csv`: every sampled tracker/GT identity observation.
- `identity_recovery_summary.csv`: identity-switch count, track-loss rate, correct-object rate, jump rate, occlusion rate, and mean IoU grouped by sequence/config/N/ReID mode.
- `occlusion_survival.csv`: longest observed occlusion/uncertain gap and longest survived gap for each tracker.
- `candidate_diagnostics.csv`: head/template/fused/final candidate IoU diagnostics to debug why boxes drift or jump.
- `area_observations_every_frame.csv`: every-frame visible GT area/IoU observations for cross-checking Week 1/Week 3 failure frames.
- `debug_log.csv`: case-level failures and benchmark progress state.
- `videos/`: annotated videos when `--save-video` is used.

## Comparing Downloaded Runs

After downloading one or more benchmark result folders or zips, create compact comparison tables with:

```powershell
python programs\summarize_v8_benchmark_runs.py `
  C:\Users\lando\Downloads\<week3-result-folder-or-zip> `
  --output-dir outputs\benchmarks\v8_comparison_latest
```

To scan recent V8/Week 3 zips in Downloads automatically:

```powershell
python programs\summarize_v8_benchmark_runs.py --downloads --output-dir outputs\benchmarks\v8_comparison_latest
```

Outputs:

- `comparison_summary.md`: presentation-friendly summary tables.
- `comparison_timing_identity.csv`: FPS, IoU, correct-rate, loss-rate, identity switches, shared-backbone proof, and major profile buckets.
- `comparison_controlled_occlusion.csv`: controlled occlusion duration/recovery rows when present.
- `comparison_area_reliability.csv`: Week 1 small-area reliability rows when present.

## Metric Rules

- Identity switch: a tracker initialized on GT object A is sampled on a visible frame and its box matches a different visible GT object better than A.
- Track loss: the track is marked lost/not-ok or no visible GT object reaches the identity IoU threshold.
- Correct object: the tracker overlaps its initialized GT object by at least `--identity-correct-iou` and no competing visible object beats it by more than `--identity-competitor-margin`.
- Occlusion survival: the longest sampled occluded/uncertain/lost span that later returns to a correct-object match.

The default identity sampling interval is `1`, so occlusion survival is expressed in frames instead of 10-frame chunks.

## Manual Re-Anchor

Manual re-anchor is a live interface behavior, not an automatic benchmark action.

In the V8 interface:

- Press `r`.
- Draw the replacement box for the selected lost/uncertain track.
- The track ID is preserved.
- A `manual_reanchor` event is logged with frame, old box, new box, prior lifecycle, and time spent.

This event format is intended to become the human-cost accounting input for the later active-correction benchmarks.
