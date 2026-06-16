# Project Learning Guide: LoRAT Multi-Object Video Labeler

Date: 2026-06-11

This is the combined study guide for the project. It merges the beginner coding guide, the computer vision/SOT/MOT foundations guide, and the codebase study guide into one path.

Use this as the main learning document. The older guide files can stay as backups, but this file is the one to follow.

## 0. What You Are Building

The project goal is a video labeling tool:

```text
user draws one box on an object
  -> the tool tracks that object through the video
  -> the user can track many objects at the same time
  -> the tool keeps object IDs stable
  -> the tool asks for correction only when uncertain
  -> benchmarks measure quality and human effort
```

The research version is:

```text
LoRAT-based SOT target conditioning
  + multi-object tracking logic
  + appearance ReID
  + open-world object discovery
  + active correction
  + human-effort benchmark suite
```

The key mental model:

```text
frame
  -> shared LoRAT/DINOv2 feature map
  -> per-object memories/templates
  -> object-conditioned scoring
  -> candidate boxes
  -> identity association
  -> accept, hold, lose, or reacquire tracks
  -> GUI, logs, benchmark output
```

## 1. How To Use This Guide

If you are very new, start at Level 0 and do the tiny exercises. If you already know Python/OpenCV, skim Levels 0-4 and start at Level 5.

Learning path:

1. Basic tooling: terminal, Python files, command-line arguments, CSVs.
2. Basic images: pixels, NumPy arrays, OpenCV display, mouse boxes.
3. Box math: bbox formats, IoU, MOT rows.
4. Tracking foundations: SOT, MOT, SOT-based MOT, ReID.
5. Project architecture: V3 through V8.
6. Training: datasets, targets, losses, diagnostics.
7. Benchmarks: timing, small objects, ID switches, HOTA/IDF1, human cost.
8. Week-by-week deliverables.

Do not try to understand every line of V8 immediately. The project becomes much easier if you climb the ladder from toy examples to real code.

## 2. Absolute Basics

### Level 0: Terminal And Project Files

What it means:

The terminal is how you run scripts, inspect files, launch benchmarks, and read errors. In this repo, most commands are PowerShell commands run from:

```text
C:\Users\lando\OneDrive\Documents\Multi-Object Tracker
```

Learn first:

- Working directory: the folder commands run from.
- `Get-ChildItem`: list files.
- `python --version`: check Python.
- `python script.py --help`: inspect script arguments.
- Output folders: where scripts write results.

Sources:

- PowerShell overview: https://learn.microsoft.com/en-us/powershell/scripting/overview
- PowerShell beginner chapter: https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/01-getting-started
- Git book: https://git-scm.com/book/en/v2

Tiny exercises:

```powershell
Get-Location
Get-ChildItem
python --version
python .\programs\bounding_box_basic.py --help
python .\programs\benchmark_lorat_v8.py --help
```

Checkpoint:

You can answer:

- What folder am I in?
- Which Python am I running?
- What script am I running?
- What arguments does the script accept?
- Where did the script write output?

### Level 1: Python From Zero

What it means:

Python is the language used by the project. You need variables, functions, lists, dictionaries, classes, imports, errors, file paths, CLI arguments, and CSV writing.

Learn first:

- Variable: a name that holds a value.
- Function: reusable code like `bbox_area(box)`.
- List: ordered collection like `[box1, box2]`.
- Dictionary: named values like `{"frame": 10, "track_id": 2}`.
- Class: template for a thing, such as `TrackState`.
- Dataclass: compact class for storing data.
- Import: use code from another file or package.
- Traceback: Python's error report.
- `argparse`: command-line arguments.
- `csv.DictWriter`: write debug/benchmark rows.
- `pathlib.Path`: file paths.

Sources:

- Python tutorial: https://docs.python.org/3/tutorial/index.html
- Python argparse: https://docs.python.org/3/library/argparse.html
- Python csv: https://docs.python.org/3/library/csv.html
- Gentle Python Beginners docs: https://python-adv-web-apps.readthedocs.io/

Tiny exercises:

```python
def bbox_area(box):
    x, y, w, h = box
    return w * h

boxes = [(10, 20, 30, 40), (5, 5, 10, 10)]
for box in boxes:
    print(box, bbox_area(box))
```

Then write three debug rows to a CSV:

```python
import csv

rows = [
    {"frame": 1, "track_id": 1, "confidence": 0.8},
    {"frame": 2, "track_id": 1, "confidence": 0.7},
]

with open("scratch_debug.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "confidence"])
    writer.writeheader()
    writer.writerows(rows)
```

Project files to study:

- `programs/bounding_box_basic.py`
- `programs/exercise_lorat_mot.py`
- `programs/benchmark_lorat_mot.py`

Checkpoint:

You can identify imports, functions, classes, `main()`, argument parsing, and output writing in a Python script.

### Level 2: NumPy Arrays And Images

What it means:

Images are grids of numbers. In OpenCV, a color frame is usually:

```text
height x width x channels
```

Access pixels as:

```python
frame[y, x]
```

Important:

- `x` is horizontal.
- `y` is vertical.
- NumPy indexing uses `y` first.
- OpenCV color order is usually BGR, not RGB.

Sources:

- NumPy absolute basics: https://numpy.org/doc/stable/user/absolute_beginners.html
- NumPy learn page: https://numpy.org/learn/
- OpenCV image tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
- OpenCV drawing tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html

Tiny exercises:

```python
import numpy as np

image = np.zeros((200, 300, 3), dtype=np.uint8)
print(image.shape)  # 200 high, 300 wide, 3 channels

image[50:100, 80:160] = (255, 0, 0)
crop = image[50:100, 80:160]
print(crop.shape)
```

Project files to study:

- `clip_bbox_to_frame()` helpers in tracker files.
- `bbox_area()` and `bbox_center()` in `programs/bounding_box_v5_lorat_shared.py`.
- V8 feature pooling helpers in `programs/bounding_box_v8_lorat_quality_batched.py`.

Checkpoint:

You can explain why an image crop is:

```python
crop = frame[y:y+h, x:x+w]
```

### Level 3: OpenCV GUI And Video

What it means:

OpenCV is how the current tool displays frames, lets the user draw boxes, reads video/image sequences, and saves preview videos.

Learn first:

- `cv2.VideoCapture()` reads video.
- Image sequences are ordered `.jpg` files.
- `cv2.imshow()` displays frames.
- `cv2.waitKey()` reads keypresses.
- `cv2.rectangle()` draws boxes.
- `cv2.putText()` draws labels.
- `cv2.setMouseCallback()` handles mouse events.
- `cv2.VideoWriter()` saves annotated video.

Sources:

- OpenCV video tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
- OpenCV mouse tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
- OpenCV drawing tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html

Tiny exercises:

1. Show one image in a window.
2. Draw a fixed rectangle.
3. Print `(x, y)` when the user clicks.
4. Add drag-to-draw rectangle behavior.
5. Add keys:
   - `q`: quit
   - `a`: add box
   - `c`: cancel
6. Save an annotated image.
7. Read a video or image sequence and draw frame numbers.

Project files to study:

- `select_boxes()` in `programs/bounding_box_v2_opencv.py`.
- `draw_tracks()` in `programs/bounding_box_v5_lorat_shared.py`.
- The `main()` loops at the bottom of tracker files.

Checkpoint:

You can explain:

```text
read frame -> draw overlays -> show frame -> wait for key/mouse -> update state -> repeat
```

### Level 4: Bounding Boxes And IoU

What it means:

A bounding box is a rectangle around an object.

Two common formats:

```text
xywh = x, y, width, height
xyxy = left, top, right, bottom
```

This project mostly uses `xywh` for visible track boxes.

Learn first:

- Convert `xywh` to `xyxy`.
- Convert `xyxy` to `xywh`.
- Area: `width * height`.
- Center: `(x + width / 2, y + height / 2)`.
- Intersection area.
- Union area.
- IoU: intersection area divided by union area.
- GIoU: IoU with an extra penalty for bad spatial alignment.

Sources:

- IoU tutorial: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
- MOTChallenge format: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt

Tiny exercises:

1. Draw two boxes on graph paper and compute IoU manually.
2. Write `xywh_to_xyxy()`.
3. Write `xyxy_to_xywh()`.
4. Write `bbox_iou()`.
5. Test:
   - same boxes -> IoU 1
   - no overlap -> IoU 0
   - partial overlap -> between 0 and 1
6. Put fake boxes into area bins:

```text
0-256
257-1024
1025-4096
4097-16384
>16384
```

Project files to study:

- `bbox_iou()` in `programs/exercise_lorat_mot.py`.
- `collect_area_observations()` in benchmark files.
- `area_reliability.csv` outputs.

Checkpoint:

You can explain why a box can look close visually but still have bad IoU if its size is wrong.

## 3. Computer Vision And Tracking Foundations

### Computer Vision In This Project

Computer vision here means:

- locating objects with boxes;
- following boxes over video;
- keeping identities stable;
- noticing uncertainty;
- asking the human for correction;
- measuring quality and human cost.

The project-specific ladder:

```text
images -> frames -> boxes -> SOT -> MOT -> ReID -> active correction -> benchmark
```

### Detection, Segmentation, Tracking

Object detection:

```text
image -> boxes, class names, confidence scores
```

Segmentation:

```text
image -> pixel masks
```

Tracking:

```text
video + object initialization -> object locations across time
```

This project is mainly a tracker/labeler, not just a detector. The object may have no name and no prior examples. That is why user-box initialization and SOT-style target conditioning matter.

### Single-Object Tracking

SOT starts with:

```text
first frame + one initial box
```

and outputs:

```text
one box per later frame
```

Common SOT terms:

- Target: the selected object.
- Template: saved target example.
- Search region: part of the next frame to search.
- Response map: score map for likely target location.
- Box regression: predicting the box around the target.
- Template update: refreshing target memory.
- Drift: tracker starts following the wrong object.
- Occlusion: target is hidden.

Why SOT fits this project:

- The user gives a first box.
- The object can be unnamed.
- The tracker should follow "this selected thing."

Why SOT alone is not enough:

- It usually assumes one target.
- It can drift during crossings.
- It does not solve global multi-object assignment.
- Running full SOT once per object scales poorly.

Sources:

- SOT survey: https://arxiv.org/abs/2201.13066
- LoRAT paper: https://arxiv.org/abs/2403.05231
- LoRAT repo: https://github.com/LitingLin/LoRAT

Local papers:

- `papers/SingleObjectTrackingASurveyofMethodsDatasetsandEvaluation Metrics.pdf`
- `papers/lorat_tracking_meets_lora.pdf`
- `papers/lorat_supplemental_training_details.pdf`
- `papers/ostrack_one_stream_tracking.pdf`
- `papers/mixformer_end_to_end_tracking.pdf`
- `papers/transt_transformer_tracking.pdf`

Project files:

- `programs/bounding_box_v3_lorat.py`
- `programs/bounding_box_v4_lorat_memory.py`
- `programs/bounding_box_v5_lorat_shared.py`

### Multi-Object Tracking

MOT tracks many objects and keeps their IDs stable.

Standard output:

```text
frame, track_id, box
```

MOT has two jobs:

1. Localization: where are the objects?
2. Association: which box belongs to which identity?

Common MOT terms:

- Track ID: identity number for one object over time.
- Candidate: possible box for a track this frame.
- Association: matching candidates to tracks.
- Assignment matrix: table of track/candidate scores.
- Identity switch: a track starts following the wrong object.
- Fragmentation: one real object becomes many broken track IDs.
- False positive: tracking something that is not a real object.
- False negative: missing a real object.
- Occlusion hold: keep a track alive while hidden.

Sources:

- DanceTrack: https://arxiv.org/abs/2111.14690
- DanceTrack repo: https://github.com/DanceTrack/DanceTrack
- TrackEval: https://github.com/JonathonLuiten/TrackEval
- HOTA: https://arxiv.org/abs/2009.07736

Local papers:

- `papers/dancetrack_multi_object_tracking_uniform_appearance_diverse_motion.pdf`
- `papers/deep_sort.pdf`
- `papers/oc_sort_observation_centric_sort.pdf`
- `papers/hota_metric.pdf`
- `papers/bytetrack.pdf`
- `papers/strongsort_make_deepsort_great_again.pdf`
- `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf`

Project files:

- `programs/benchmark_lorat_v8.py`
- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`

### SOT Versus MOT

| Question | SOT | MOT |
| --- | --- | --- |
| How many objects? | One selected object. | Many objects. |
| Initialization | Usually a first-frame box. | Usually detector boxes, sometimes user boxes. |
| Main question | Where did this target go? | Which boxes belong to which identities? |
| Strength | Target-specific and class-agnostic. | Handles many identities. |
| Weakness | Drift and one-target assumption. | Association complexity and detector dependence. |
| Project relevance | User selects unnamed objects. | Tool must track many boxes at once. |

This project lives between them:

```text
SOT-style user initialization + MOT-style multi-object identity management
```

### SOT-Based MOT Does Exist

SOT-based MOT is less common than tracking-by-detection, but it is real and directly relevant.

Pattern A: run one SOT tracker per object.

```text
object 1 -> SOT tracker 1
object 2 -> SOT tracker 2
object 3 -> SOT tracker 3
```

Pros:

- simple;
- class-agnostic;
- works naturally with user boxes.

Cons:

- expensive as object count grows;
- independent trackers can steal objects;
- no global assignment by default.

Project mapping:

- V3/V4 are closest to this.
- V4 adds memory and identity checks.

Pattern B: detection plus SOT branch.

SOTMOT, "Improving Multiple Object Tracking With Single Object Tracking," adds an SOT branch to MOT so target-specific discrimination helps association.

Local papers:

- `papers/SOT_For_MOT.pdf`
- `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`

Why they matter:

- They show SOT can reduce missed detections.
- They show SOT target discrimination can improve MOT.
- They support the research direction.

Pattern C: shared backbone plus per-object target heads.

This is V7/V8:

```text
frame -> one shared feature map
object memories -> batched per-object scoring
identity logic -> assign, accept, hold, recover
```

Pros:

- keeps target-conditioned tracking;
- avoids a full backbone pass per object;
- supports unnamed user-selected objects;
- scales better than one full SOT per object.

Cons:

- requires careful head training;
- identity logic is still required;
- memory updates can poison tracks.

Project files:

- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`

Local papers:

- `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf`
- `papers/DCFST.pdf`
- `papers/lorat_tracking_meets_lora.pdf`

### Why SOT-Based MOT Is Limited/Rare

Reason 1: SOT assumes one target.

```text
target vs background
```

MOT needs:

```text
target A vs target B vs target C vs background
```

Reason 2: independent SOT trackers can steal objects during crossings.

Reason 3: full SOT per object is expensive.

```text
N objects -> N model passes
```

Week 2 aims for:

```text
N objects -> 1 shared frame pass + batched object scoring
```

Reason 4: bad memory updates cause drift.

Reason 5: many MOT benchmarks assume detector boxes, while this project assumes user-selected unknown objects.

### ReID Embeddings

ReID asks:

```text
does this new crop look like the object I saw earlier?
```

Important terms:

- Embedding: vector representing appearance.
- Normalization: scale vector to unit length.
- Dot product: basic similarity measure.
- Cosine similarity: angle-based similarity.
- Positive pair: same object.
- Negative pair: different object.
- Hard negative: different object that looks similar or is nearby.
- Feature bank: saved embeddings over time.

Sources:

- Cosine similarity docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- SciPy assignment docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

Local papers:

- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- `papers/fairmot_detection_reid_mot.pdf`
- `papers/strongsort_make_deepsort_great_again.pdf`
- `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf`
- `papers/boosttrack_similarity_confidence_mot.pdf`

Project files:

- `extract_reid_histogram()` in `programs/bounding_box_v5_lorat_shared.py`
- `V8FeatureIdentityArbitrator` in `programs/bounding_box_v8_lorat_quality_batched.py`
- `contrastive_reid_loss()` in `programs/train_lorat_v8_head.py`

Tiny exercise:

```python
import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

a = normalize([1.0, 0.0])
b = normalize([0.9, 0.1])
c = normalize([0.0, 1.0])

print("a vs b", float(a @ b))
print("a vs c", float(a @ c))
```

## 4. PyTorch, Training, And Models

### PyTorch Tensors

PyTorch is used for model inference and training. A tensor is like a NumPy array that can live on GPU and track gradients.

Learn first:

- Tensor shape.
- Batch dimension.
- Device: CPU, CUDA, DirectML.
- `torch.no_grad()` for inference.
- `requires_grad`.
- `nn.Module`.
- Loss.
- Optimizer.
- Dataset and DataLoader.

Sources:

- PyTorch basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch quickstart: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- PyTorch tensors: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- PyTorch autograd: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- PyTorch DataLoaders: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch training loop: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

Tiny exercises:

```python
import torch

x = torch.randn(4, 8)
y = torch.randn(8, 3)
z = x @ y
print(z.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
print(x.device)
```

Practice V8-like shapes:

```python
objects = 5
locations = 196
dim = 768

features = torch.randn(objects, locations, dim)
scores = torch.randn(objects, locations)
best_location = scores.argmax(dim=1)
print(best_location.shape)
```

### Training A Model

Training loop:

```text
load batch -> model predicts -> compute loss -> backward pass -> optimizer step -> validate -> save checkpoint
```

Terms:

- Dataset sample: one training example.
- Batch: group of examples.
- Ground truth: correct answer.
- Target map: model supervision over a grid.
- Prediction: model output.
- Loss: error number to minimize.
- Backpropagation: compute gradients.
- Optimizer: updates weights.
- Epoch: one pass over training data.
- Validation: evaluate without training.
- Overfit smoke test: prove the model can memorize a tiny dataset.
- Checkpoint: saved model weights.
- Diagnostic CSV: training history.

Project files:

- `programs/train_lorat_v7_head.py`
- `programs/train_lorat_v8_head.py`

Important V8 functions:

- `MOTFrameHeadDataset`
- `make_lorat_style_targets()`
- `decode_box_maps_xyxy()`
- `contrastive_reid_loss()`
- `validate_head()`
- `save_checkpoint()`
- `append_diagnostic_row()`

Local papers:

- `papers/giou_generalized_intersection_over_union.pdf`
- `papers/varifocalnet_iou_aware_dense_detector.pdf`
- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- `papers/detr_end_to_end_object_detection.pdf`
- `papers/deformable_detr.pdf`

Checkpoint:

You can explain why "loss went down" is not enough. Tracking also needs IoU, correct-object rate, ID switches, track-loss rate, validation quality, and human cost.

## 5. Codebase Map

### Runtime Files

| File | What It Teaches | Priority |
| --- | --- | --- |
| `programs/bounding_box_basic.py` | Minimal OpenCV tracking loop. | First warmup. |
| `programs/bounding_box_v2_opencv.py` | Multiple boxes, frame sources, GUI, MOT output. | First serious read. |
| `programs/bounding_box_v3_lorat.py` | Early LoRAT-backed multi-object tracker. | LoRAT integration history. |
| `programs/bounding_box_v4_lorat_memory.py` | Memory slots, identity arbitration, Kalman holds. | ReID/control logic ideas. |
| `programs/bounding_box_v5_lorat_shared.py` | Shared utility layer reused later. | Essential reference. |
| `programs/bounding_box_v6_lorat_gated.py` | Slot gating and recovery slot selection. | Short bridge. |
| `programs/bounding_box_v7_lorat_frame_shared.py` | First shared-frame architecture. | Read before V8. |
| `programs/bounding_box_v8_lorat_quality_batched.py` | Current main tracker. | Main implementation target. |

### Training And Benchmark Files

| File | What It Teaches | Priority |
| --- | --- | --- |
| `programs/exercise_lorat_mot.py` | DanceTrack loading, GT init, basic evaluation. | Early. |
| `programs/benchmark_lorat_mot.py` | Week 1 timing and small-object benchmark. | Early. |
| `programs/benchmark_lorat_v5.py` | Identity/debug benchmarking. | Medium. |
| `programs/benchmark_lorat_v6_forced_area.py` | Controlled small-object stress test. | Medium. |
| `programs/benchmark_lorat_v7.py` | Week 2 shared-frame proof. | High. |
| `programs/benchmark_lorat_v8.py` | Current V8 benchmark runner. | High. |
| `programs/train_lorat_v7_head.py` | Smaller head training script. | Read before V8 training. |
| `programs/train_lorat_v8_head.py` | Current head training script. | Main training target. |
| `programs/tune_lorat_v4_debug.py` | Debug CSV reading and threshold tuning. | Useful. |

### Scripts

| File | Use |
| --- | --- |
| `scripts/setup-lorat-env.ps1` | Environment setup. |
| `scripts/verify-lorat-env.ps1` | Environment verification. |
| `scripts/fetch-assets.ps1` | Fetch datasets, papers, models. |
| `scripts/theia-stage-v*.ps1` | Package for HPC runs. |
| `scripts/theia_v*_benchmark.sbatch` | HPC benchmark jobs. |
| `scripts/theia_v8_train_heads.sbatch` | HPC V8 head training. |
| `scripts/make_v6_v8_presentation_visuals.py` | Presentation charts/visuals. |

### Existing Docs

| Doc | Use |
| --- | --- |
| `docs/v8_code_walkthrough.md` | Detailed V8 file walkthrough. |
| `docs/week3_reid_recovery_todo.md` | Week 3 implementation checklist. |
| `docs/v8_training_methods_research.md` | Training strategy and papers. |
| `docs/v8_papers_training_onboarding_review.md` | Review of V8 and onboarding alignment. |
| `docs/week3_gap_fill_papers.md` | Gap-fill paper catalog. |
| `docs/sotmot_lorat_mot_notes.md` | SOTMOT interpretation for this project. |
| `docs/data_models_and_papers.md` | Datasets, models, paper starter kit. |
| `docs/core_training_dataset_candidates.md` | Expanded core training dataset shortlist. |
| `docs/amd_nvidia_platform_notes.md` | AMD/NVIDIA platform notes. |

## 6. V8 Architecture

Short version:

V8 is not upstream LoRAT simply running once per object. V8 is a LoRAT-backbone shared-frame MOT branch.

V8 flow:

```text
main()
  -> open video/image source
  -> user selects initial boxes
  -> create V8 tracker
  -> initialize tracks and memory
  -> for each frame:
       encode frame once
       score all object memories in one batched head
       decode candidate boxes
       rerank/recover candidates
       resolve identity
       accept or hold each track
       update memory only when safe
       draw/log/output
```

Important V8 classes:

- `SharedFrameLoRATEncoder`: one shared LoRAT/DINOv2 frame encoding.
- `BatchedObjectConditionedHead`: scores object memories against frame features.
- `V8FeatureIdentityArbitrator`: batched identity association.
- `V8QualityBatchedLoRATTracker`: main runtime tracker.

Important V8 state:

- `TrackState`: visible track state.
- `V8TemplateMemorySlot`: feature memory for a target.
- `V8HeadCandidate`: decoded candidate box.
- `RuntimeStatus`: FPS/GPU/object count/status.

Read:

- `docs/v8_code_walkthrough.md`
- `programs/bounding_box_v8_lorat_quality_batched.py`

Trace exercise:

Find and summarize these functions:

```powershell
rg -n "def update|def _score_and_update_tracks|def _accept_candidate|def _hold_track|class V8QualityBatchedLoRATTracker" .\programs\bounding_box_v8_lorat_quality_batched.py
```

## 7. Benchmarks And Metrics

A benchmark is a repeatable experiment. Metrics are numbers that describe tracker behavior.

Important metrics:

- FPS.
- Seconds per output box.
- GPU memory.
- IoU.
- Correct-object rate.
- ID switches.
- Track-loss rate.
- Occlusion survival.
- HOTA.
- IDF1.
- MOTA.
- Human-cost events.

Sources:

- TrackEval: https://github.com/JonathonLuiten/TrackEval
- MOTChallenge format: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt
- HOTA paper: https://arxiv.org/abs/2009.07736
- HOTA overview: https://autonomousvision.github.io/hota-metrics/

Project files:

- `programs/benchmark_lorat_mot.py`
- `programs/benchmark_lorat_v8.py`
- `external/TrackEval-master/trackeval/metrics/hota.py`
- `external/TrackEval-master/trackeval/metrics/identity.py`

Tiny benchmark exercise:

1. Make fake GT for 3 frames.
2. Make fake tracker output for 3 frames.
3. Compute IoU per frame.
4. Add one identity switch.
5. Count the switch.
6. Write a tiny `summary.md`.

## 8. Summer Weeks As Learning Tasks

### Week 1: Multi-Object Tracker, GUI, Core Benchmarks

Need to know:

- OpenCV image/video loops.
- Mouse box selection.
- Bbox formats and IoU.
- MOT rows.
- Seconds per box.
- Small-object pixel-area bins.

Learn from zero:

1. Show one image with OpenCV.
2. Draw one rectangle.
3. Drag one rectangle with the mouse.
4. Save that rectangle as `xywh`.
5. Write one MOT row.
6. Compute area and IoU.
7. Run a 10-frame benchmark.

Files:

- `programs/bounding_box_basic.py`
- `programs/bounding_box_v2_opencv.py`
- `programs/exercise_lorat_mot.py`
- `programs/benchmark_lorat_mot.py`

Exercises:

- Add a command-line flag to change output folder.
- Print selected boxes before tracking starts.
- Add initial box area to a CSV.

Checkpoint:

You can explain how a mouse drag becomes a track box and how a benchmark row is written.

### Week 2: Shared Backbone And Throughput Scaling

Need to know:

- Why one LoRAT per object is slow.
- What a shared ViT feature map is.
- Tensor shapes for objects and locations.
- FPS and GPU memory measurement.

Learn from zero:

1. Make a NumPy matrix shaped `[objects, features]`.
2. Compute pairwise dot products.
3. Make PyTorch tensors shaped `[objects, locations, dim]`.
4. Find best location per object with `argmax`.
5. Compare V5 and V8 control flow.
6. Read V8 shared encoder and batched head only.

Files:

- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/benchmark_lorat_v7.py`
- `programs/benchmark_lorat_v8.py`
- `docs/v8_code_walkthrough.md`

Exercises:

- Find where V8 encodes the frame once.
- Find where V8 scores all objects in one head batch.
- Add one timing/debug field.

Checkpoint:

You can explain "one shared backbone pass" versus "one object head batch."

### Week 3: ReID And Track Recovery

Need to know:

- Appearance embeddings.
- Cosine similarity.
- Hungarian assignment.
- Identity switches.
- Track states: healthy, uncertain, lost, reacquired.
- Manual reanchor event logging.

Learn from zero:

1. Make three 2D vectors and compute cosine similarity.
2. Make a 3-track by 3-candidate score matrix.
3. Solve it with `linear_sum_assignment()`.
4. Count ID switches from fake rows.
5. Add one `track_state` debug field.
6. Write one `manual_reanchor` event CSV row.

Files:

- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`
- `programs/benchmark_lorat_v8.py`
- `docs/week3_reid_recovery_todo.md`
- `docs/week3_gap_fill_papers.md`

Exercises:

- Log when a track is held instead of accepted.
- Pool one feature vector inside a bbox.
- Add same-ID vs different-ID ReID diagnostic.
- Run with ReID-like identity on/off once metrics are wired.

Checkpoint:

You can explain what caused a track to accept, hold, become lost, or reacquire.

### Week 4: Open-World Object Discovery

Need to know:

- Proposal versus track.
- Class-agnostic proposals.
- Open-vocabulary detection.
- Proposal recall.
- Proposal accept/reject UI.

Learn from zero:

1. Define a `Proposal` dataclass.
2. Generate fake proposal boxes.
3. Draw proposals differently from tracks.
4. Let accepted proposals spawn tracks.
5. Compute proposal recall against GT boxes.

Files and papers:

- `docs/week3_gap_fill_papers.md`
- `papers/grounding_dino.pdf`
- `papers/segment_anything.pdf`
- `papers/sam2_segment_anything_images_videos.pdf`
- `papers/owlvit_simple_open_vocabulary_detection_vit.pdf`
- `papers/video_owlvit_open_world_video_localization.pdf`

Exercises:

- Add proposal queue CSV.
- Add accepted/rejected proposal logging.

Checkpoint:

You can explain why proposals should not contaminate tracks before user acceptance.

### Week 5: Active Correction Loop

Need to know:

- Uncertainty scoring.
- Review queues.
- Human-cost events.
- Correction propagation.
- Cost-to-quality curves.

Learn from zero:

1. Make fake debug rows with confidence, margin, jitter, ReID distance.
2. Compute uncertainty score.
3. Sort a review queue.
4. Log correction events.
5. Simulate correcting one bad bbox with GT.
6. Measure quality before and after correction.

Files and papers:

- `docs/week3_gap_fill_papers.md`
- `papers/video_annotation_tracking_active_learning.pdf`
- `papers/vatic_efficient_crowdsourced_video_annotation.pdf`
- `papers/efficient_video_annotation_visual_interpolation_frame_selection.pdf`
- `papers/hdamot_active_learning_multi_object_tracking.pdf`

Exercises:

- Write `review_queue.csv`.
- Add event types: `initial_box`, `manual_reanchor`, `correction`, `verify`.

Checkpoint:

You can explain how uncertainty-ranked correction saves human effort.

### Weeks 6-12: Export, Simulated Annotation, Baselines, Paper

Need to know:

- MOT export.
- COCO-video export.
- TrackEval folder expectations.
- HOTA/IDF1/MOTA.
- Baseline tracker interfaces.
- Ablation tables.
- Reproducible run metadata.

Learn from zero:

1. Write a tiny MOT-format output file.
2. Read it back and verify boxes.
3. Make one fake TrackEval-ready sequence.
4. Add `run_metadata.json`.
5. Turn one feature on/off and compare metrics.
6. Make an ablation table.

Files and papers:

- `external/TrackEval-master`
- `papers/hota_metric.pdf`
- `papers/bytetrack.pdf`
- `papers/boosttrack_similarity_confidence_mot.pdf`
- `papers/boosttrack_plus_plus_tracklet_information_mot.pdf`

Checkpoint:

You can separate speed, quality, and human-effort claims.

## 9. Four-Week Self-Study Plan

### Week A: Python And OpenCV Tracking Basics

Read:

- `programs/bounding_box_basic.py`
- `programs/bounding_box_v2_opencv.py`
- first half of `README.md`

Do:

- Run `--help` for basic and V2 scripts.
- Draw a box.
- Find where output rows are written.
- Write and test your own `bbox_iou()`.

Checkpoint:

You can explain how a frame source differs from a tracker backend.

### Week B: SOT, MOT, And Identity Logic

Read:

- tracking foundations in this guide;
- `programs/bounding_box_v5_lorat_shared.py`;
- `programs/bounding_box_v6_lorat_gated.py`;
- `docs/sotmot_lorat_mot_notes.md`.

Papers:

- `papers/SOT_For_MOT.pdf`
- `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`
- `papers/deep_sort.pdf`
- `papers/oc_sort_observation_centric_sort.pdf`

Do:

- Trace one track through accept, hold, and memory update.
- Add one debug column for rejection reason.
- Use `tune_lorat_v4_debug.py` on an existing CSV.

Checkpoint:

You can explain confidence, ReID, motion, path, and occlusion gates.

### Week C: V8 Shared-Frame Architecture

Read:

- `docs/v8_code_walkthrough.md`
- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`

Papers:

- `papers/lorat_tracking_meets_lora.pdf`
- `papers/dinov2_learning_robust_visual_features_without_supervision.pdf`
- `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf`

Do:

- Draw V8 data flow.
- Add notes for tensor shapes.
- Run a short V8 smoke test.

Checkpoint:

You can point to shared encoder, batched head, identity resolver, accept/hold logic, and memory refresh logic.

### Week D: Training And Week 3 Features

Read:

- `programs/train_lorat_v8_head.py`
- `docs/v8_training_methods_research.md`
- `docs/week3_reid_recovery_todo.md`
- `docs/week3_gap_fill_papers.md`

Papers:

- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- `papers/fairmot_detection_reid_mot.pdf`
- `papers/strongsort_make_deepsort_great_again.pdf`
- `papers/xmem_long_term_video_object_segmentation.pdf`

Do:

- Run `train_lorat_v8_head.py --help`.
- Find target creation, loss, validation, diagnostics.
- Add or verify an overfit smoke mode.
- Add one ReID diagnostic.
- Add a manual reanchor event row.

Checkpoint:

You can explain what the head is trained to predict and what Week 3 is still missing.

## 10. Paper-To-Code Map

| Local paper/doc | Main idea | Code connection |
| --- | --- | --- |
| `papers/CV_MOT.pdf` | Broad CV/MOT overview. | Read before tracker files. |
| `papers/SingleObjectTrackingASurveyofMethodsDatasetsandEvaluation Metrics.pdf` | SOT vocabulary and failures. | V3/V4 LoRAT per-object behavior. |
| `papers/SOT_For_MOT.pdf` | SOT can reduce MOT false negatives. | V4/V5 memory recovery and held tracks. |
| `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf` | SOT branch can improve MOT discrimination. | `docs/sotmot_lorat_mot_notes.md`, V8 design argument. |
| `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf` | Unify motion and affinity features. | V8 shared-frame encoder plus feature identity. |
| `papers/DCFST.pdf` | Discriminative SOT features. | Future ReID/head embedding training. |
| `papers/lorat_tracking_meets_lora.pdf` | LoRAT SOT backbone/training. | V3/V4 runtime, V8 backbone. |
| `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf` | Contrastive ReID training. | `contrastive_reid_loss()` and Week 3/V9 plan. |
| `papers/dancetrack_multi_object_tracking_uniform_appearance_diverse_motion.pdf` | Similar appearance stresses MOT. | ReID/motion ablations. |
| `papers/hota_metric.pdf` | Detection and association evaluation. | TrackEval and benchmark summaries. |
| `papers/grounding_dino.pdf` | Open-world proposals. | Week 4 proposal queue. |
| `papers/sam2_segment_anything_images_videos.pdf` | Video-aware segmentation/correction. | Future correction propagation. |
| `papers/video_annotation_tracking_active_learning.pdf` | Active frame selection. | Week 5 review queue. |
| `papers/vatic_efficient_crowdsourced_video_annotation.pdf` | Human video annotation cost. | Human-effort benchmark design. |

## 11. Good First Code Contributions

These are realistic tasks that build confidence:

1. Add `track_state` to V8 debug output.
2. Add a `manual_reanchor` event CSV schema.
3. Add a small V8 training overfit smoke command.
4. Add same-ID versus different-ID ReID diagnostic.
5. Add a benchmark switch for ReID on/off.
6. Add an occlusion-gap stress mode.
7. Add TrackEval-compatible export validation for one DanceTrack sequence.
8. Add a `run_metadata.json` writer for benchmarks.
9. Add a summary section that splits speed, quality, and human-cost metrics.

## 12. Command Habits

Use these frequently:

```powershell
rg -n "class V8QualityBatchedLoRATTracker|def update|def _score_and_update_tracks" .\programs\bounding_box_v8_lorat_quality_batched.py
rg -n "def make_lorat_style_targets|def contrastive_reid_loss|def validate_head" .\programs\train_lorat_v8_head.py
rg -n "Identity|ReID|occlusion|manual_reanchor|HOTA|IDF1" .\docs .\programs
python .\programs\bounding_box_v8_lorat_quality_batched.py --help
python .\programs\benchmark_lorat_v8.py --help
python .\programs\train_lorat_v8_head.py --help
```

After code changes:

```powershell
python -m py_compile .\programs\bounding_box_v8_lorat_quality_batched.py
python -m py_compile .\programs\train_lorat_v8_head.py
python -m py_compile .\programs\benchmark_lorat_v8.py
```

## 13. Debug Routine

When stuck:

1. Shrink it.
   Run fewer frames, fewer tracks, smaller tensors, or fake data.

2. Print shapes.
   For model bugs, inspect tensor shapes before values.

3. Print one row.
   For CSV/benchmark bugs, print the first row and read it by hand.

4. Freeze randomness.
   Use a fixed seed when comparing behavior.

5. Name the invariant.
   Example: "There should be one shared frame encode per frame."

6. Compare against a toy case.
   If V8 is confusing, solve a 2-track example on paper.

7. Ask a specific question.
   Good: "Why did track 3 hold on frame 72?"
   Vague: "Why is tracking bad?"

## 14. Minimum Vocabulary

Basic coding:

- script
- function
- class
- dataclass
- argument
- path
- CSV

Images:

- image
- pixel
- frame
- channel
- crop
- array
- shape

Boxes:

- bbox
- `xywh`
- `xyxy`
- area
- center
- IoU
- GIoU

SOT:

- target
- template
- search region
- response map
- box regression
- drift
- template update

MOT:

- track ID
- candidate
- association
- assignment matrix
- identity switch
- track loss
- occlusion
- fragmentation

Models:

- tensor
- batch
- device
- embedding
- cosine similarity
- ReID
- loss
- optimizer
- checkpoint

Project:

- LoRAT
- shared backbone
- object-conditioned head
- memory slot
- ReID bank
- active correction
- manual reanchor
- human-cost event

## 15. What To Avoid

- Do not start by rewriting V8 from scratch.
- Do not add open-world proposals before identity recovery is stable.
- Do not trust a trained head until it can overfit a tiny slice.
- Do not use FPS alone as proof of success.
- Do not refresh memory from uncertain boxes.
- Do not compare runs unless sequence, frames, tracks, model size, and seed match.

## 16. North-Star Skill

The skill you are building is:

```text
Given a video frame, object memory, model outputs, and uncertainty signals,
write code that makes a careful track-state update and proves it improved
label quality per unit of human effort.
```

That is the center of this project.
