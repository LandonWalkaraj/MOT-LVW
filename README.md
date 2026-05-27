# MOT-LVW

LoRAT-based multi-object tracking experiments with an OpenCV desktop GUI.

The current code is a working project scaffold and prototype. It supports an interactive multi-box GUI, a LoRAT-backed tracking path, an OpenCV fallback path, and a headless runner for DanceTrack/MOT17-style sequences.

## What Is Included

- `programs/bounding_box_basic.py`: first single-object OpenCV CSRT tracker.
- `programs/bounding_box_v2_opencv.py`: multi-object OpenCV prototype with ROI selection and optional scale/rotation behavior through CamShift.
- `programs/bounding_box_v3_lorat.py`: multi-object GUI that wraps LoRAT as the tracking backend.
- `programs/exercise_lorat_mot.py`: non-interactive DanceTrack/MOT17 exercise runner.
- `scripts/fetch-assets.ps1`: downloads external code snapshots, model weights, papers, and dataset archives from `manifests/assets.json`.
- `scripts/setup-lorat-env.ps1`: creates the Python environment and installs LoRAT/project dependencies.
- `scripts/verify-lorat-env.ps1`: verifies imports, PyTorch device visibility, and basic LoRAT environment readiness.
- `docs/`: notes about datasets, model assets, papers, and AMD/NVIDIA platform tradeoffs.

Large local assets are intentionally not committed. Datasets, LoRAT weights, downloaded external repositories, generated outputs, PDFs, and `.venv` are ignored.

## Repository Layout

```text
MOT-LVW/
  docs/                       Project notes and platform references
  manifests/assets.json        Download manifest for external assets
  programs/                    Tracker GUIs and dataset exercise scripts
  scripts/                     Windows setup and download helpers
  requirements.txt             Project-level Python dependencies
  README.md
```

After running the asset downloader, local-only folders may also appear:

```text
data/                          DanceTrack, MOT17, TAO, and raw archives
external/                      Local LoRAT and TrackEval snapshots
models/lorat/                  LoRAT checkpoint files
outputs/                       GUI and dataset-run result files
papers/                        Downloaded reference PDFs
```

Those folders are excluded from Git because they can be large or have their own licenses/terms.

## Requirements

- Windows PowerShell
- Python 3.10 or newer
- OpenCV GUI support
- LoRAT assets downloaded through this repo's scripts
- For CUDA runs: an NVIDIA GPU, CUDA-compatible PyTorch, and the matching PyTorch wheel index from pytorch.org

The project can run on CPU for development and smoke tests. CUDA is only needed for the GPU/CUDA benchmark part of the work.

## Setup

Clone the repository:

```powershell
git clone https://github.com/LandonWalkaraj/MOT-LVW.git
cd MOT-LVW
```

Fetch the core LoRAT assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-core
```

Create the Python environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-lorat-env.ps1
```

Verify the environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify-lorat-env.ps1
```

On an NVIDIA/CUDA machine, pass the PyTorch CUDA wheel index before installing dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-lorat-env.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

Use the CUDA index that matches the target machine and the current PyTorch install guidance.

## Asset Downloads

List available asset groups:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -List
```

Common downloads:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-core
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset dancetrack
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset mot17
```

Some MOTChallenge-hosted files may require a browser login or manual terms acceptance. If a scripted download fails, download the archive manually and place it in the matching `data/raw/...` location described in `manifests/assets.json`.

## Run The GUI

Run the LoRAT-backed multi-object GUI from a camera:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --video 0
```

Run the GUI on a video file:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --video ".\path\to\video.mp4"
```

Run the GUI on an image sequence folder containing `img1`:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --sequence ".\data\DanceTrack\val\dancetrack0001"
```

On a CUDA machine:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cuda:0 --video ".\path\to\video.mp4"
```

GUI controls:

- Draw one or more boxes with the mouse.
- Press `Enter` or `Space` to accept a box.
- Press `c` to cancel the current selection.
- Press `q` during playback to quit.

The GUI writes MOTChallenge-format tracking rows under `outputs/lorat-gui` unless `--output` is provided. Use `--save-video` to write an annotated MP4.

## OpenCV Prototype

The v2 OpenCV prototype is useful when LoRAT assets are not ready yet:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v2_opencv.py" --video 0
```

This is not intended to match LoRAT performance. It is mainly for testing the GUI flow and multi-box interaction.

## Exercise DanceTrack And MOT17

The dataset runner initializes tracks from MOTChallenge-style ground truth and writes result files that can be inspected or evaluated later.

List sequences:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\DanceTrack" --dataset dancetrack --split val --list-sequences
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\MOTChallenge\MOT17" --dataset mot17 --split train --list-sequences
```

Run a short CPU smoke test:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\DanceTrack" --dataset dancetrack --split val --device cpu --max-sequences 1 --max-tracks 2 --max-frames 50
```

Run MOT17 on CUDA:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --dataset-root ".\data\MOTChallenge\MOT17" --dataset mot17 --split train --device cuda:0
```

Results are written to `outputs/lorat-exercise` by default.

## Current Status

Implemented:

- Multi-box OpenCV GUI selection.
- LoRAT runtime integration through a local `external/LoRAT-main` checkout.
- One LoRAT task per selected object, batched through LoRAT's evaluator.
- MOTChallenge-format output writing.
- DanceTrack/MOT17-style sequence discovery and smoke-run support.
- CPU development path plus CUDA device option for machines with CUDA-enabled PyTorch.

Still to improve:

- Stronger re-identification when an object leaves and re-enters the frame.
- More robust recovery after long occlusion or tracking loss.
- Formal TrackEval evaluation wiring for final metrics.
- Cleaner dataset download handling for files that require manual MOTChallenge login.
- Packaging the GUI as a friendlier application entry point.

## Notes On LoRAT

LoRAT is fundamentally a single-object tracker. This project adapts it to multi-object tracking by creating one tracking task per user-selected bounding box. That is enough for an interactive multi-object prototype, but it is not the same as a full detector-plus-reID MOT system.
