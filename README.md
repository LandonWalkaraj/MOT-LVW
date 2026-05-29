# MOT-LVW

LoRAT-based multi-object tracking with a desktop OpenCV GUI.

Project goal:

> Implement LoRAT for multi-object tracking on GPU/CUDA, exercise it on DanceTrack and MOT17, and provide a graphical interface where a user can place multiple bounding boxes and track all selected objects at the same time.

The current Git-facing deliverable is focused on three program files:

- `programs/bounding_box_v3_lorat.py`: the LoRAT-backed multi-object GUI tracker.
- `programs/bounding_box_v4_lorat_memory.py`: experimental v4 GUI tracker that gives each visible track multiple internal LoRAT memory slots.
- `programs/exercise_lorat_mot.py`: the DanceTrack exercise/benchmark runner.
- `programs/benchmark_lorat_mot.py`: the repeatable Week 1 timing and small-object benchmark runner.

Older OpenCV prototypes, local datasets, downloaded models, debug logs, generated videos, papers, and external repository snapshots are development artifacts and should stay local unless there is a specific reason to include them.

## Local Layout

Expected project layout:

```text
MOT-LVW/
  programs/
    bounding_box_v3_lorat.py
    bounding_box_v4_lorat_memory.py
    exercise_lorat_mot.py
    benchmark_lorat_mot.py
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

CPU works for development and short smoke tests. CUDA runs require an NVIDIA GPU and a CUDA-compatible PyTorch install. On Windows with AMD Radeon hardware, V4 can also use PyTorch DirectML with `--device dml`; this is useful for local iteration, but CUDA remains the target backend for NVIDIA benchmarking.

Install project dependencies:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Install PyTorch separately if needed. For CUDA, use the PyTorch wheel index that matches the target NVIDIA machine. For AMD/Windows DirectML:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install torch-directml
```

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

Run the experimental v4 LoRAT-memory GUI:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v4_lorat_memory.py" --device cuda:0 --sequence ".\data\raw\DanceTrack\val\val\dancetrack0065"
```

Run V4 on AMD/Windows DirectML:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v4_lorat_memory.py" --device dml --sequence ".\data\raw\DanceTrack\val\val\dancetrack0065"
```

DirectML is currently forced to `--track-batch-size 1` by V4 because batched DirectML LoRAT outputs were unstable on the local AMD setup. On CUDA, use a larger `--track-batch-size` such as `16` or `32` to improve GPU utilization.

V4 preserves v3 and changes the control flow: each selected object owns its own LoRAT tracker bank, then a lightweight identity-memory layer decides which LoRAT output still belongs to that track. The default `--lorat-memory-slots 11` means one permanent first-frame LoRAT anchor plus ten rolling recent LoRAT template slots per selected box. For performance, V4 now evaluates only `--lorat-active-slots-per-track 3` slots per track per frame by default. Use `--lorat-memory-slots 1` when you want the older one-LoRAT-task-per-box behavior, or `--lorat-active-slots-per-track 0` when you intentionally want to evaluate the full memory bank every frame.

V4 also tightens LoRAT's own single-object setup by default: `--lorat-search-area-factor 3.0` narrows the search crop so nearby lookalikes are less likely to enter the search region, and `--lorat-window-penalty 0.60` increases LoRAT's bias toward staying near the current target. To compare against upstream LoRAT defaults, use `--lorat-search-area-factor 4.0 --lorat-window-penalty 0.45` for 224 configs, or `--lorat-search-area-factor 5.0 --lorat-window-penalty 0.45` for 378 configs.

V4 also passes conservative state-update thresholds into LoRAT's one-stream pipeline. When a prediction has low score, jumps too far, or changes box area too much, LoRAT still reports the prediction to V4, but LoRAT does not immediately move its internal search/crop state to that suspicious box.

V4 now adds a stricter identity safety layer around each single-object LoRAT tracker. Hungarian assignment is still used across all active LoRAT slot outputs, but the default ReID and motion gates are higher: `--identity-min-score 0.50`, `--identity-min-reid 0.28`, and `--identity-min-motion 0.18`. A LoRAT box also needs `--lorat-accept-min-score 0.20` before it can update the visible track. Lower-confidence outputs are treated as temporary occlusions, held with a per-track Kalman filter for up to `--occlusion-max-frames 30`, and are not allowed to refresh LoRAT memory or contaminate the ReID bank. During occlusion, `--occlusion-velocity-damping 0.65` slows the Kalman velocity each held frame so the box does not coast hundreds of pixels away from the last reliable target.

LoRAT's raw response score is not treated as a universal probability in V4. Each track now calibrates confidence against its own first healthy LoRAT response, so the displayed/debug `confidence` is relative to that target's normal LoRAT score. The CSV also includes `raw_confidence` and `confidence_baseline` so you can see when LoRAT's absolute score is low even though the relative track confidence is healthy.

V4 also has a pose/view-change bridge for targets whose visible appearance genuinely changes, such as a face/front-torso crop becoming the back of a head or torso. If the output comes from the same LoRAT-owned slot and motion is smooth enough, `--view-change-min-motion 0.72` and `--view-change-min-confidence 0.16` allow the update even when ReID temporarily disagrees. Those updates are tagged `VIEWCHANGE` and can refresh rolling LoRAT memory so the tracker learns the new view without replacing the first-frame anchor.

V4 also keeps a reliable center-path vector for each track. Only trusted, non-occluded updates are added to this path, so held Kalman frames cannot drag the guide away. Candidate boxes are scored against that recent direction of travel, and `--identity-min-path 0.40` rejects lower-confidence crossing candidates that jump off the target's expected centerline. When the reliable center path is mostly stationary, V4 switches from directional velocity scoring to a tighter local-center radius, which helps when the selected person is standing or moving toward the camera while a similar person passes behind them.

V4 now lets LoRAT scale boxes again, but clamps scaling aggressively. The default `--lorat-min-box-area 100` prevents accepted boxes and LoRAT's internal one-stream state from shrinking below 100 pixels of area, `--lorat-trusted-size-floor-scale 0.70` prevents width/height from dropping below 70 percent of the initial selected box, and `--lorat-max-area-change-per-frame 1.05` limits frame-to-frame area changes. Use `--fixed-lorat-box-size` only when you want to preserve the initial selected width/height exactly.

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

## Week 1 Benchmarks

Use the dedicated benchmark script for items (b) and (c):

- timing required to produce boxes for one object, two objects, and up to N objects per video
- smallest ground-truth pixel-area bins that remain reliable

Run the default benchmark on DanceTrack 0065 for 200 frames:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --device cpu
```

The default benchmark uses `B-224`, `dancetrack0065`, `--track-counts 1,2,4,8`, and `--max-frames 200`.

Run the same 200-frame benchmark on CUDA:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --device cuda:0
```

Compare model sizes against the same benchmark:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --device cuda:0 --compare-configs B-224 L-224 g-224
```

Write per-frame/per-track area observations for deeper analysis:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --device cpu --write-observations
```

Observation CSVs and annotated preview videos are enabled by default. Use `--no-write-observations` or `--no-save-video` only when you want a lighter run.

The benchmark prints the output folder, CSV paths, video folder, active config, track count, and frame progress in the VS Code terminal. By default it prints progress every 10 processed frames:

```text
[dancetrack0065 B-224 N=4] frame 70/200 (source frame 70), elapsed 123.4s
```

Adjust the status interval with `--progress-interval`:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --device cpu --progress-interval 5
```

Benchmark outputs are written under `outputs/lorat-benchmarks/dancetrack` by default:

- `{run_label}_timing_by_track_count.csv`
- `{run_label}_area_reliability.csv`
- `{run_label}_area_observations.csv`
- `{run_label}_summary.md`
- `videos/*.mp4`

Each benchmark run gets a unique folder and filename prefix such as `dancetrack0065_B-224_N1-2-4-8_frames200_20260529_143000`, so new runs do not overwrite older output and the config plus number of selected tracks are visible in the saved artifacts.

The timing CSV records total seconds, initialization seconds, tracking seconds, FPS, total milliseconds per produced box, tracking-only milliseconds per produced box, and the preview-video path. The area CSV groups each GT-matched tracker observation by ground-truth bounding-box pixel area and reports mean IoU, IoU@0.50, unreliable rate, and whether the bin meets the configured reliability rule.

## V3 Behavior

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
- In V4, box scaling is allowed by default but constrained with a minimum area and per-frame area-change limit.

## Experimental V4 Behavior

The v4 tracker is a LoRAT-led MOT engine. Previous behavior remains in v3, while v4 keeps the active tracking path focused on LoRAT-owned track tasks:

V4 tuning defaults live near the top of `programs/bounding_box_v4_lorat_memory.py` in the `DEFAULT_*` block. The VS Code V4 launch profile only passes device/model/sequence, so editing that block changes normal Run and Debug behavior unless you explicitly pass a CLI override.

- Each user-selected box creates one permanent initial LoRAT tracker task plus rolling recent LoRAT memory tasks by default.
- LoRAT still proposes boxes from each owned tracker task, but V4 now uses a small ReID/Hungarian identity pass to decide which visible track each proposal best belongs to.
- If a proposal looks more like another tracked object, the output can be assigned back to that object's track ID instead of silently swapping identities.
- If LoRAT misses or produces an untrusted low-confidence box, V4 holds the track with a per-track Kalman prediction for up to `--occlusion-max-frames` frames while keeping the LoRAT task alive for recovery.
- `--lorat-memory-slots` controls the first-frame-plus-recent LoRAT memory bank. The default `11` is the 10+1 setup; each recent slot is only refreshed after a trusted identity match, so suspicious overlap frames do not overwrite every memory at once.
- `--lorat-memory-refresh-interval` controls how often the rolling recent LoRAT slots are refreshed. The default `1` keeps the ten recent slots frame-by-frame; higher values spread the memory across a longer time window.
- `--lorat-active-slots-per-track` controls how many memory slots are actually evaluated each frame. The default is currently `10`, which evaluates almost the full 10+1 memory bank; `0` restores full-bank evaluation.
- `--lorat-min-box-area` controls the hard lower area clamp. The default `100` prevents boxes from collapsing below 100 pixels.
- `--lorat-max-area-change-per-frame` controls how quickly boxes can scale. The default `1.05` allows about 5 percent area growth/shrink per accepted frame.
- `--lorat-trusted-size-floor-scale` prevents width/height from shrinking below a fraction of the initial selected box. The default is `0.70`.
- `--fixed-lorat-box-size` preserves the initial selected width/height exactly.
- `--lorat-search-area-factor` and `--lorat-window-penalty` override LoRAT's runtime config before the evaluator is built, making V4's single-object LoRAT tasks more conservative around similar nearby objects.
- `--lorat-state-update-min-score`, `--lorat-state-update-max-center-shift`, and `--lorat-state-update-max-area-change` control whether LoRAT is allowed to update its own internal search/crop state after a prediction.
- `--lorat-accept-min-score` controls whether a LoRAT output is trusted enough to update the visible track; lower scores become Kalman-held occlusion frames instead.
- `--disable-identity-arbitration`, `--identity-min-score`, `--identity-min-reid`, `--identity-min-motion`, `--identity-min-path`, `--identity-bank-size`, and `--identity-memory-min-confidence` tune the lightweight identity layer.
- `--occlusion-max-frames` controls how long an untrusted target is kept alive by Kalman prediction. `--occlusion-iou-threshold` skips memory refresh while a track overlaps another active track, and `--occlusion-velocity-damping` slows the held prediction over time.
- `--reid-recovery-min-score`, `--reid-recovery-min-reid`, `--reid-recovery-min-motion`, and `--reid-recovery-min-confidence` allow a low-confidence LoRAT box to recover a lost track only when appearance and motion are both strong.
- `--view-change-min-score`, `--view-change-min-motion`, `--view-change-min-confidence`, and `--view-change-max-lost-frames` tune same-target pose/view adaptation when a selected face or torso changes appearance as the person turns.
- `--track-batch-size` controls how many LoRAT tasks run per forward chunk; `--lorat-slot-capacity` controls the maximum internal LoRAT task cache size.

The GUI/debug overlay may show state tags:

- `LORAT`: track was updated directly from its owned LoRAT output.
- `LORAT-RECENT-01` through `LORAT-RECENT-10`: a rolling recent LoRAT memory slot produced the selected update.
- `FIXEDSIZE`: V4 accepted LoRAT's center but preserved the initial selected width/height.
- `MINAREA`: V4 expanded a box because its area fell below `--lorat-min-box-area`.
- `SCALELIMIT`: V4 limited the accepted frame-to-frame area change.
- `SIZEFLOOR`: V4 accepted LoRAT's center but expanded a collapsing width/height using trusted size memory.
- `REID-LORAT`: a LoRAT proposal from another owned task was matched back to this track by the identity layer.
- `ID_UNCERTAIN`: LoRAT produced a box, but the identity layer did not trust it enough to write it as this track.
- `LOWCONF`, `REIDLOW`, `MOTIONLOW`, `PATHLOW`, or `REACQUIRE_LOWCONF`: LoRAT produced a box, but V4 held the Kalman prediction instead of committing the suspicious output.
- `REIDRECOVERY`: V4 accepted a low-confidence LoRAT box because ReID and motion were strong enough to recover from occlusion.
- `VIEWCHANGE`: V4 accepted a same-LoRAT-slot update with smooth motion even though appearance changed enough that ordinary ReID would have been suspicious.
- `OCCLUDED`: V4 is holding or carefully accepting a track during a suspected occlusion; memory refresh is blocked.
- `LORAT_MISS`: LoRAT did not produce an output for that track on this frame.

## Debug Logs

Write a focused debug CSV when diagnosing a jump:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --device cpu --sequence dancetrack0065 --debug-log ".\outputs\debug\dancetrack0065_frames60_85.csv" --debug-frame-start 60 --debug-frame-end 85
```

The debug CSV includes:

- final LoRAT-owned box
- raw LoRAT proposal
- predicted box
- calibrated confidence, raw LoRAT confidence, and the per-track confidence baseline

Quick threshold tuning from a V4 debug CSV:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\tune_lorat_v4_debug.py" ".\outputs\debug\dancetrack0065_lorat_v4_debug.csv" --emit-cli
```

This reports state counts, score distributions, suspicious accepted jumps, occlusion streaks, and a small trial CLI block. It is meant for fast local iteration, not formal evaluation.
- width, height, and area in pixels
- velocity
- assignment score, assignment margin, ReID score, motion score, and source score
- state tags
- lost frame count
- active LoRAT slot, LoRAT memory slot count, and appearance-bank size

The debug log is flushed while the program runs, so pressing `q` should still leave the completed rows on disk.

## Important Notes

LoRAT is not being trained by `exercise_lorat_mot.py`. The phrase "exercised on DanceTrack and MOT17" means the tracker is run and evaluated/tested on those datasets. Fine-tuning LoRAT on DanceTrack or MOT17 would be a separate training pipeline.

DanceTrack is the current local test target. MOT17 support is wired as a dataset option, but final MOT17 runs require the MOT17 data to be present locally.

Formal HOTA/MOTA/IDF1 evaluation is not fully wired yet. Current outputs are MOTChallenge-format files and preview/debug artifacts that can be connected to TrackEval later.

## Useful Commands

Show GUI options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v3_lorat.py" --help
```

Show experimental v4 GUI options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\bounding_box_v4_lorat_memory.py" --help
```

Show exercise runner options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\exercise_lorat_mot.py" --help
```

Show benchmark runner options:

```powershell
& ".\.venv\Scripts\python.exe" ".\programs\benchmark_lorat_mot.py" --help
```

Check that the main scripts compile:

```powershell
& ".\.venv\Scripts\python.exe" -m py_compile ".\programs\bounding_box_v3_lorat.py" ".\programs\bounding_box_v4_lorat_memory.py" ".\programs\exercise_lorat_mot.py" ".\programs\benchmark_lorat_mot.py"
```
