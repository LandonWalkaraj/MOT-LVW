# MOT-LVW

LoRaT-based multi-object tracking research code for selected-object video labeling.

The current development line is V9: a local-search tracker that keeps LoRaT as the single-object tracking backbone while adding multi-object coordination, re-identification/recovery logic, and a trainable selected-target head. V8 remains in the repo because V9 reuses its shared utilities, training data adapters, and model-head components.

## Current Working Set

The files that matter for current development are:

- `programs/bounding_box_v9_lorat_local_search.py` - current V9 tracker.
- `programs/train_lorat_v9_local_search_head.py` - current V9 head trainer.
- `programs/benchmark_lorat_v9.py` - current V9 benchmark entrypoint.
- `programs/bounding_box_v9_lorat_open_world.py` - Week 4/open-world scaffold.
- `programs/bounding_box_v8_lorat_quality_batched.py` - V8 tracker/head components reused by V9.
- `programs/train_lorat_v8_head.py` - V8 dataset/training utilities reused by V9.
- `programs/benchmark_lorat_v8.py` - V8 benchmark utilities reused by V9.
- `programs/mot_common.py` - shared MOT/ReID/geometry helpers.
- `programs/exercise_lorat_mot.py` - DanceTrack/MOT sequence helpers.
- `programs/export_tao_to_mot_sequences.py` - TAO export/adapter helper.

Older V2-V7 scripts were useful research history, but they are not the active implementation and should not be treated as current entrypoints.

## Local-Only Assets

Large data and model files are intentionally not tracked:

- `data/raw/` - DanceTrack, MOT17, TAO-OW, LaSOT/GOT-style data.
- `models/lorat/` - LoRaT checkpoint binaries.
- `external/LoRAT-main/` - local LoRaT checkout.
- `outputs/` - benchmark videos, logs, plots, and staging bundles.
- `.venv/` - local virtual environment.

The `.gitignore` is set up to keep those local.

## Environment

Install project dependencies:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Install PyTorch separately for the target machine. For Theia/A100 training we use a CUDA PyTorch wheel inside the Slurm job.

Expected local layout:

```text
MOT-LVW/
  programs/
  scripts/
  docs/
  requirements.txt
  README.md

Local-only:
  data/raw/
  external/LoRAT-main/
  models/lorat/
  outputs/
```

## V9 Training

V9 training mixes MOT-style data and SOT-style selected-target data:

- DanceTrack/MOT17 for multi-object identity/conflict supervision.
- TAO-OW for open-world object diversity.
- LaSOT/GOT-style SOT sequences for selected-target template/search behavior, including small/part-like targets.

Local smoke checks:

```powershell
python -m py_compile programs\train_lorat_v9_local_search_head.py programs\bounding_box_v9_lorat_local_search.py programs\benchmark_lorat_v9.py
python programs\train_lorat_v9_local_search_head.py --smoke-targets
```

Submit the current B-224 V9 48-hour Theia training job:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\submit_theia_v9_b224_tao_48h.ps1
```

That script uploads current V9/V8 code, uploads `data/raw/LaSOT_subset` as `LaSOT_subset.tar.gz`, and submits:

```text
scripts/theia_v9_train_b224_48h_tao.sbatch
```

The Slurm job extracts LaSOT inside the GPU allocation if `/work/landonvw/LaSOT_subset` is not already present.

## V9 Benchmarking

Use:

```powershell
python programs\benchmark_lorat_v9.py --help
```

Benchmark runs should save:

- machine-readable CSV/JSON summaries,
- videos for qualitative proof,
- debug/proof fields for identity switches, track loss, re-acquisition, confidence, IoU, and small-target behavior.

## Dataset Helpers

TAO-OW subset download/export:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_tao_ow_subset.ps1
python programs\export_tao_to_mot_sequences.py --help
```

LaSOT subset download:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_lasot_subset.ps1
```

The current local LaSOT subset is hand-class only, used as an initial SOT-style selected-target training source.

## Notes

The project direction is no longer to keep every numbered prototype in GitHub. The repo should show the active research system and the code needed to reproduce current training/benchmark work. Historical generated outputs, old staging bundles, and retired prototypes should stay local unless a paper appendix explicitly needs them.
