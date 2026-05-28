# MOT-LVW

LoRAT-based multi-object tracking with a desktop OpenCV GUI.

Project goal:

> Implement LoRAT for multi-object tracking on GPU/CUDA, exercise it on DanceTrack and MOT17, and provide a graphical interface where a user can place multiple bounding boxes and track all selected objects at the same time.

The current Git-facing deliverable is focused on two program files:

- `programs/bounding_box_v3_lorat.py`: the LoRAT-backed multi-object GUI tracker.
- `programs/exercise_lorat_mot.py`: the DanceTrack exercise/benchmark runner.

Older OpenCV prototypes, local datasets, downloaded models, debug logs, generated videos, papers, and external repository snapshots are development artifacts and should stay local unless there is a specific reason to include them.

## Local Layout

Expected project layout:

```text
MOT-LVW/
  programs/
    bounding_box_v3_lorat.py
    exercise_lorat_mot.py
  requirements.txt
  README.md
```

Local-only folders used while running the project:

```text
data/                          DanceTrack/MOT17 data
external/LoRAT-main/           Local LoRAT checkout
models/lorat/                  LoRAT checkpoint files
outputs/                       Result files, preview videos, debug logs
papers/                        Reference PDFs
.venv/                         Local Python environment
```

## Requirements

- Python 3.10 or newer
- OpenCV with GUI support
- NumPy
- SciPy
- PyYAML
- PyTorch
- A local LoRAT checkout under `external/LoRAT-main`
- LoRAT checkpoint files under `models/lorat`

CPU works for development and short smoke tests. CUDA runs require an NVIDIA GPU and a CUDA-compatible PyTorch install. An AMD laptop can run the code on CPU, but it will not run the CUDA benchmark path.

Install project dependencies:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Install PyTorch separately if needed. For CUDA, use the PyTorch wheel index that matches the target NVIDIA machine.

## Run The GUI

The main GUI script is:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py"
```

When run without arguments, it tries to open the local DanceTrack `dancetrack0065` sequence if the dataset exists.

Run on a webcam:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --video 0
```

Run on a video file:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --video ".\path\to\video.mp4"
```

Run on a DanceTrack/MOT-style image sequence:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cpu --sequence ".\data\raw\DanceTrack\val\val\dancetrack0065"
```

Run on CUDA:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --device cuda:0 --sequence ".\data\raw\DanceTrack\val\val\dancetrack0065"
```

GUI controls:

- Draw a box with the mouse.
- Press `Enter` or `Space` to accept the current box.
- Press `c` to cancel the current box selection.
- Press `a` during playback to add a new box when an object becomes visible.
- Press `q` during playback to quit.

There is no default maximum number of selected boxes. Use `--max-tracks N` only when you intentionally want a cap. `--track-batch-size` controls how many LoRAT tasks are processed together internally.

## Exercise DanceTrack

The exercise script runs the v3 tracker on DanceTrack-style sequences and writes MOTChallenge-format result files.

List available sequences:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --list-sequences
```

Run DanceTrack 0065 with manual GUI initialization:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cpu --sequence dancetrack0065
```

Run a short CPU smoke test:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cpu --sequence dancetrack0065 --max-frames 50
```

Use ground-truth initialization for repeatable tests:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cpu --sequence dancetrack0065 --gt-init --max-frames 50
```

Run on CUDA:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cuda:0 --sequence dancetrack0065
```

Preview videos are enabled by default in the exercise runner. Add `--no-save-video` for faster smoke tests.

## Model Config Comparison

The v3 runner supports multiple LoRAT model configs:

- `B-224`
- `B-378`
- `L-224`
- `L-378`
- `g-224`
- `g-378`

Compare several configs on the same sequence:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --compare-configs B-224 L-224 g-224 --gt-init --device cuda:0 --sequence dancetrack0065 --min-init-tracks 4 --max-frames 150
```

For CPU development, keep comparisons short:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --compare-configs B-224 B-378 --gt-init --device cpu --sequence dancetrack0065 --min-init-tracks 4 --max-frames 3 --no-save-video
```

Comparison CSV and Markdown summaries are written under `outputs/lorat-exercise/dancetrack/.../comparison`.

## Current V3 Behavior

The v3 tracker adapts LoRAT, which is normally single-object tracking, into a multi-object workflow:

- One LoRAT task is created per user-selected bounding box.
- All LoRAT proposals are passed through a shared coordinator.
- The coordinator uses object-agnostic color/texture ReID features.
- SciPy Hungarian assignment matches proposals back to track IDs.
- Each track stores the initial appearance and a conservative rolling appearance bank.
- Each track stores the first visual template and a rolling 10-frame visual memory bank.
- Visual memory is kept for ReID/coordinator reasoning, but it is not used to reset LoRAT every frame.
- When a LoRAT proposal looks like an ID jump, the coordinator can add a `memory` recovery candidate near the predicted track location by comparing local crops against the first-frame and rolling memory examples.
- The coordinator uses trajectory, direction, bottom-edge, overlap, and size guards to reduce ID jumps during close crossings.
- If LoRAT loses a track, coasting preserves the box size instead of letting width/height shrink every frame.
- A trusted size floor prevents a track box from collapsing too small to recover.

The GUI/debug overlay may show state tags:

- `PATH`: proposal disagreed with recent trajectory.
- `DIR`: proposal disagreed with expected movement direction.
- `BOTTOM`: bottom edge/depth consistency was suspicious.
- `MEM`: recent memory was weak.
- `MEMREC`: a memory-recovery candidate was used instead of the raw LoRAT proposal.
- `REID?`: appearance match was weak for a possible source switch.
- `SEP`: overlap guard separated two boxes.
- `SIZE`: box size was clamped to prevent collapse.
- `COAST`: track is being carried forward after a failed/rejected proposal.

## Debug Logs

Write a focused debug CSV when diagnosing a jump:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cpu --sequence dancetrack0065 --debug-log ".\outputs\debug\dancetrack0065_frames60_85.csv" --debug-frame-start 60 --debug-frame-end 85
```

The debug CSV includes:

- final corrected box
- raw LoRAT proposal
- predicted box
- width, height, and area in pixels
- velocity
- assignment score and margin
- ReID, motion, trajectory, memory, direction, bottom, and IoU scores
- state tags
- lost/occluded frame counts
- visual memory counts

The debug log is flushed while the program runs, so pressing `q` should still leave the completed rows on disk.

## Detector Notes

Detector refresh is off by default because the tracker should work for any object the user boxes, not only people.

OpenCV HOG can be enabled for person-specific DanceTrack experiments:

```powershell
--detector hog --detector-interval 5 --max-detections 12
```

Use `--detector none` for the object-agnostic LoRAT + ReID + Hungarian path.

## Important Notes

LoRAT is not being trained by `exercise_lorat_mot.py`. The phrase "exercised on DanceTrack and MOT17" means the tracker is run and evaluated/tested on those datasets. Fine-tuning LoRAT on DanceTrack or MOT17 would be a separate training pipeline.

DanceTrack is the current local test target. MOT17 support is wired as a dataset option, but final MOT17 runs require the MOT17 data to be present locally.

Formal HOTA/MOTA/IDF1 evaluation is not fully wired yet. Current outputs are MOTChallenge-format files and preview/debug artifacts that can be connected to TrackEval later.

## Useful Commands

Show GUI options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --help
```

Show exercise runner options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --help
```

Check that both scripts compile:

```powershell
& ".\.venv\Scripts\python.exe" -m py_compile ".\programs\bounding_box_v3_lorat.py" ".\programs\exercise_lorat_mot.py"
```
