# Multi-Object Tracker Setup Notes

This workspace starts with a reproducible asset manifest and a Windows-friendly downloader for the week-one data, models, and reading list.

## Quick Start

List available asset groups:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -List
```

Fetch the LoRAT code snapshot, TrackEval snapshot, LoRAT B/L/g weights, and core paper PDFs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-core
```

Fetch the smallest useful dataset starter set:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-datasets-small
```

Fetch full dataset groups only when you are ready for the storage/time hit:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset dancetrack
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset mot17
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset tao-ow
```

The TAO AVA/HACS components require a MOTChallenge login and manual terms acceptance. The downloader will warn instead of faking that step.

## Python Environment

Create or update the local LoRAT debug environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-lorat-env.ps1
```

Verify imports, PyTorch device visibility, TurboJPEG, and a CPU LoRAT dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify-lorat-env.ps1
```

This machine currently exposes AMD GPUs, so the default PyTorch install is CPU-only. The week-one CUDA benchmarks need an NVIDIA/CUDA machine or cloud runner.
On an NVIDIA machine, pass the PyTorch CUDA wheel index selected from pytorch.org, for example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-lorat-env.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

## LoRAT GUI v3

Version 3 connects the interactive multi-box GUI to the local LoRAT checkout. It uses OpenCV only for video/image I/O, drawing, and the desktop window; the tracking backend is LoRAT. On this AMD laptop, use CPU mode:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding box v3.py" --device cpu --video 0
```

On a machine where PyTorch exposes an NVIDIA CUDA or AMD ROCm/HIP device, use the same script with:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding box v3.py" --device cuda:0 --video path\to\video.mp4
```

For DanceTrack or MOT17 image sequences, point the script at a sequence folder containing `img1`:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding box v3.py" --device cpu --sequence path\to\DanceTrack\val\dancetrack0001
& ".\.venv\Scripts\python.exe" ".\programs\bounding box v3.py" --device cpu --sequence path\to\MOT17\train\MOT17-02-FRCNN
```

The GUI lets the user place multiple initial boxes. Each box becomes one LoRAT single-object tracking task, batched through LoRAT's evaluator. Results are written in MOTChallenge format under `outputs/lorat-gui` unless `--output` is supplied.

## DanceTrack and MOT17 Exercise Runs

Use `programs/exercise_lorat_mot.py` for non-interactive dataset exercise runs. It reads MOTChallenge-style sequence folders, initializes LoRAT tracks from `gt/gt.txt`, and writes result files in MOTChallenge format.

If a dataset archive is downloaded but not extracted:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\raw\DanceTrack" --dataset dancetrack --extract-zips --list-sequences
```

List extracted sequences:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\raw\DanceTrack" --dataset dancetrack --split val --list-sequences
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root "path\to\MOT17" --dataset mot17 --split train --list-sequences
```

CPU smoke runs on this laptop:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\raw\DanceTrack" --dataset dancetrack --split val --backend lorat --device cpu --max-sequences 1 --max-tracks 2 --max-frames 50
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root "path\to\MOT17" --dataset mot17 --split train --backend lorat --device cpu --max-sequences 1 --max-tracks 2 --max-frames 50
```

On a CUDA or ROCm/HIP PyTorch machine, switch only the device:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root "path\to\MOT17" --dataset mot17 --split train --backend lorat --device cuda:0
```

## Notes

- Asset definitions live in `manifests/assets.json`.
- The downloader saves archives in `data/raw`, repositories in `external`, model weights in `models/lorat`, and paper PDFs in `papers`.
- Use `-NoExtract` if you want to download large zip files without expanding them immediately.
- Python is now available and the LoRAT debug environment is handled by `scripts/setup-lorat-env.ps1`. Git is still optional because the setup uses downloadable repository snapshots.

See `docs/data_models_and_papers.md` for the source links, dataset/model choices, and initial paper reading order.
See `docs/amd_nvidia_platform_notes.md` for platform notes on CUDA/NVIDIA vs ROCm/AMD for the LoRAT benchmark plan.
