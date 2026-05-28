# MOT-LVW

LoRAT-based multi-object tracking experiments with an OpenCV desktop GUI.

The current code is a working project scaffold and prototype. It supports an interactive multi-box GUI, a LoRAT-backed tracking path, an OpenCV fallback path, and a headless runner for DanceTrack/MOT17-style sequences.

## What Is Included

- `programs/bounding_box_basic.py`: inital single-object OpenCV CSRT tracker, made for testing.
- `programs/bounding_box_v2_opencv.py`: iterative from bounding box v1, made for testing.
- `programs/bounding_box_v3_lorat.py`: multi-object GUI that wraps LoRAT as the tracking backend.
- `programs/exercise_lorat_mot.py`: DanceTrack exercise runner using bounding box v3.
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

First, 

## Run The GUI

tbd

GUI controls:

- Draw one or more boxes with the mouse.
- Press `Enter` or `Space` to accept a box.
- Press `c` to cancel the current selection.
- Press `q` during playback to quit.

The GUI writes MOTChallenge-format tracking rows under `outputs/lorat-gui` unless `--output` is provided. Use `--save-video` to write an annotated MP4.

## Current Status

Implemented:

- Multi-box OpenCV GUI selection.
- LoRAT runtime integration through a local `external/LoRAT-main` checkout.
- One LoRAT task per selected object, batched through LoRAT's evaluator.

## Notes On LoRAT

LoRAT from my research is fundamentally a single-object tracker. This project is an attempt to adapt it to multi-object tracking by creating one tracking task per user-selected bounding box. 
