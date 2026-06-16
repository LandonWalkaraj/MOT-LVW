# Absolute Beginner Layer For The LoRAT MOT Labeler

Date: 2026-06-11

Note: this guide has been consolidated into `docs/project_learning_guide.md`. Use that file as the main study path.

This is the "start from the very beginning" companion to `docs/coding_study_guide.md`.

Use this when the main guide says something like "learn OpenCV video loops" or "learn embeddings" and you want to know exactly what that means, where to learn it, and what tiny exercise proves you understand it.

For a project-focused explanation of computer vision, single-object tracking, multi-object tracking, and SOT-based MOT, read `docs/computer_vision_sot_mot_foundations.md` after Level 4 or alongside Levels 5-6.

## How To Use This

Work through the levels in order. Each level has:

- What the concept means.
- Why it matters for this project.
- What to learn first.
- Sources to study.
- Tiny exercises.
- A checkpoint for knowing you are ready to move on.

You do not need to master all of computer vision before helping with this project. You need a ladder from "I can run and edit a Python file" to "I can make a small, tested change to the tracker."

## Level 0: Using The Terminal And Project Files

What this means:

The terminal is how you run scripts, install packages, inspect files, and launch benchmarks. In this repo, most commands are PowerShell commands run from:

```text
C:\Users\lando\OneDrive\Documents\Multi-Object Tracker
```

Why it matters:

If you cannot confidently run scripts and inspect outputs, every coding task feels foggy. The tracker is not one app button; it is a collection of scripts, arguments, data folders, models, and output files.

Learn first:

- What a working directory is.
- How to list files.
- How to run Python.
- How to pass command-line arguments.
- How to read script help with `--help`.
- How to locate output files.

Good sources:

- Microsoft PowerShell overview: https://learn.microsoft.com/en-us/powershell/scripting/overview
- Microsoft PowerShell beginner chapter: https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/01-getting-started
- Git book, especially "Getting Started" and "Git Basics": https://git-scm.com/book/en/v2

Tiny exercises:

1. Open PowerShell in the project folder.
2. Run:

   ```powershell
   Get-Location
   Get-ChildItem
   python --version
   python .\programs\bounding_box_basic.py --help
   python .\programs\benchmark_lorat_v8.py --help
   ```

3. Find the `docs`, `programs`, `papers`, `outputs`, and `models` folders.
4. Create a scratch file named `scratch_terminal_notes.txt` outside the source code or in a scratch folder, then delete it manually after confirming you know where it is.

Checkpoint:

You can answer:

- What folder am I in?
- Which Python am I running?
- What script am I running?
- What arguments does this script accept?
- Where did this script write output?

## Level 1: Python From Zero

What this means:

Python is the language used by almost all project code. At the beginner level, you need to understand variables, functions, lists, dictionaries, classes, imports, errors, and files.

Why it matters:

The project code is large, but it is still built from basic Python pieces. A tracker is a class. A bbox is a tuple. A debug row is a dictionary. A benchmark is a loop that writes CSV rows.

Learn first:

- Variables: names that hold values.
- Lists: ordered collections like `[box1, box2, box3]`.
- Dictionaries: named values like `{"frame": 10, "track_id": 2}`.
- Functions: reusable actions like `bbox_area(box)`.
- Classes: templates for objects like `TrackState`.
- Dataclasses: compact classes mostly used to store data.
- Imports: using code from another file or package.
- Exceptions: errors and how to read tracebacks.
- `pathlib.Path`: clean file paths.
- `argparse`: command-line arguments.
- `csv`: writing benchmark/debug rows.

Good sources:

- Python official tutorial: https://docs.python.org/3/tutorial/index.html
- Python `argparse` tutorial/reference: https://docs.python.org/3/library/argparse.html
- Python `csv` module docs: https://docs.python.org/3/library/csv.html
- Python Beginners docs, gentler than the official tutorial: https://python-adv-web-apps.readthedocs.io/

Tiny exercises:

1. Write a function:

   ```python
   def bbox_area(box):
       x, y, w, h = box
       return w * h
   ```

2. Make a list of boxes and print each area.
3. Make a dictionary for one debug row:

   ```python
   row = {"frame": 1, "track_id": 7, "confidence": 0.82}
   ```

4. Write three rows to a CSV file using `csv.DictWriter`.
5. Make a tiny CLI:

   ```powershell
   python scratch_bbox_area.py --width 20 --height 10
   ```

   It should print `200`.

Project files to connect:

- `programs/bounding_box_basic.py`: easiest script.
- `programs/exercise_lorat_mot.py`: clean examples of dataclasses and file parsing.
- `programs/benchmark_lorat_mot.py`: CSV writing and benchmark loops.

Checkpoint:

You can open a Python file and identify:

- imports;
- constants;
- functions;
- classes;
- the `main()` function;
- where command-line arguments are parsed;
- where output files are written.

## Level 2: NumPy Arrays And Images

What this means:

Images are just grids of numbers. In this project, frames are usually NumPy arrays shaped like:

```text
height x width x channels
```

For a color OpenCV image, channels are usually BGR, not RGB.

Why it matters:

Every frame, crop, bbox, and template comes from slicing arrays. If you understand array shapes, image crops stop feeling magical.

Learn first:

- What an array is.
- What shape means.
- How image coordinates work: `x` goes sideways, `y` goes downward.
- How to crop an image: `frame[y1:y2, x1:x2]`.
- Difference between `height` and `width`.
- Difference between row/column indexing and x/y coordinates.
- Data types like `uint8` for images and `float32` for model tensors.

Good sources:

- NumPy absolute basics: https://numpy.org/doc/stable/user/absolute_beginners.html
- NumPy learning resources: https://numpy.org/learn/
- OpenCV image basics: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
- OpenCV drawing functions: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html

Tiny exercises:

1. Create a black image:

   ```python
   import numpy as np
   image = np.zeros((200, 300, 3), dtype=np.uint8)
   print(image.shape)
   ```

2. Set a rectangular region to blue:

   ```python
   image[50:100, 80:160] = (255, 0, 0)
   ```

3. Use OpenCV to draw a rectangle around that region.
4. Crop that region and print the crop shape.
5. Write a function that clips a bbox so it cannot go outside the image.

Project files to connect:

- `clip_bbox_to_frame()` in older tracker files.
- `bbox_center()`, `bbox_area()`, and crop helpers in `programs/bounding_box_v5_lorat_shared.py`.
- V8 feature pooling helpers in `programs/bounding_box_v8_lorat_quality_batched.py`.

Checkpoint:

You can explain why this crop uses `y` first and `x` second:

```python
crop = frame[y:y+h, x:x+w]
```

## Level 3: OpenCV GUI And Video Basics

What this means:

OpenCV gives the project its basic desktop interface: show a frame, draw boxes, read mouse clicks, advance video frames, and write annotated videos.

Why it matters:

The user-facing tool begins with simple GUI mechanics. A fancy tracker is useless if the user cannot place boxes and see results.

Learn first:

- `cv2.VideoCapture()` for reading video.
- Image sequences as a list of `.jpg` frame paths.
- `cv2.imshow()` to display a frame.
- `cv2.waitKey()` to keep the window alive and read keys.
- `cv2.rectangle()` and `cv2.putText()` for drawing.
- `cv2.setMouseCallback()` for mouse events.
- `cv2.VideoWriter()` for saving preview videos.

Good sources:

- OpenCV videos tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
- OpenCV mouse handling tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
- OpenCV drawing functions: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html

Tiny exercises:

1. Show one image in a window.
2. Draw a fixed rectangle on the image.
3. Add a mouse callback that prints `(x, y)` when you click.
4. Add drag-to-draw rectangle behavior.
5. Add a keypress:

   - `q` quits.
   - `a` starts adding a new box.
   - `c` cancels current selection.

6. Save an annotated copy of the image.
7. Read a video or image sequence and draw frame numbers on each frame.

Project files to connect:

- `select_boxes()` in `programs/bounding_box_v2_opencv.py`.
- `draw_tracks()` in `programs/bounding_box_v5_lorat_shared.py`.
- `main()` loops at the bottom of tracker files.

Checkpoint:

You can explain the lifecycle:

```text
read frame -> draw overlays -> show frame -> wait for key/mouse -> update state -> repeat
```

## Level 4: Bounding Boxes And Object Detection Math

What this means:

A bounding box is a rectangle around an object. Most of this project measures whether predicted boxes match ground-truth boxes.

Why it matters:

The tracker can only be judged by geometry and identity. Geometry starts with boxes.

Learn first:

- `xywh`: top-left x, top-left y, width, height.
- `xyxy`: left, top, right, bottom.
- Area.
- Intersection.
- Union.
- IoU: intersection area divided by union area.
- Center distance.
- Aspect ratio.
- Pixel-area bins for small-object reliability.

Good sources:

- IoU tutorial from PyImageSearch: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
- MOTChallenge result format via TrackEval docs: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt
- Ultralytics object detection docs for the basic idea of boxes/classes/confidence: https://docs.ultralytics.com/tasks/detect/

Tiny exercises:

1. Draw two boxes on graph paper and compute IoU by hand.
2. Write `xywh_to_xyxy()` and `xyxy_to_xywh()`.
3. Write `bbox_iou()`.
4. Make test cases:

   - identical boxes should have IoU 1.
   - non-overlapping boxes should have IoU 0.
   - partial overlap should be between 0 and 1.

5. Create area bins:

   ```text
   0-256
   257-1024
   1025-4096
   4097-16384
   >16384
   ```

6. Put five fake boxes into those bins.

Project files to connect:

- `bbox_iou()` in `programs/exercise_lorat_mot.py`.
- `collect_area_observations()` in benchmark files.
- `area_reliability.csv` outputs.

Checkpoint:

You can explain why a predicted box can look "close" visually but still have low IoU if the size is wrong.

## Level 5: Multi-Object Tracking From Scratch

What this means:

Single-object tracking asks:

```text
Where did this one object go?
```

Multi-object tracking asks:

```text
Where did every object go, and which object is which?
```

Why it matters:

The hard part is not only finding boxes. The hard part is keeping the same ID on the same physical object when objects cross, disappear, reappear, or look similar.

Learn first:

- Track ID.
- Frame number.
- Candidate box.
- Track state.
- Motion prediction.
- Appearance similarity.
- Assignment matrix.
- Greedy matching versus optimal assignment.
- Identity switch.
- Track loss.
- Occlusion hold.

Good sources:

- SciPy `linear_sum_assignment()` docs for assignment matrices: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- TrackEval README for tracking evaluation context: https://github.com/JonathonLuiten/TrackEval
- DeepSORT local paper: `papers/deep_sort.pdf`
- OC-SORT local paper: `papers/oc_sort_observation_centric_sort.pdf`

Tiny exercises:

1. Make two tracks:

   ```python
   tracks = [
       {"id": 1, "bbox": (10, 10, 20, 40)},
       {"id": 2, "bbox": (80, 10, 20, 40)},
   ]
   ```

2. Make two candidate boxes in the next frame.
3. Score each track/candidate pair by center distance.
4. Build a 2x2 score matrix.
5. Assign candidates to tracks using:

   - greedy best score;
   - `scipy.optimize.linear_sum_assignment()`.

6. Create a crossing case where greedy matching makes the wrong choice.
7. Add a fake appearance score and combine it with motion score.

Project files to connect:

- `LightweightIdentityArbitrator` in `programs/bounding_box_v5_lorat_shared.py`.
- `V8FeatureIdentityArbitrator` in `programs/bounding_box_v8_lorat_quality_batched.py`.
- `summarize_identity()` in benchmark files.

Project-focused reading:

- `docs/computer_vision_sot_mot_foundations.md`

Checkpoint:

You can explain what each row and column means in a track-candidate assignment matrix.

## Level 6: ReID Embeddings And Similarity

What this means:

An embedding is a vector that represents appearance. ReID uses embeddings to answer:

```text
Does this new crop look like the object I saw earlier?
```

Why it matters:

Week 3 is about re-identification and recovery. If a track is lost, the tool needs to reattach it when the same object reappears.

Learn first:

- Vector.
- Normalization.
- Dot product.
- Cosine similarity.
- Positive pair: same object.
- Negative pair: different object.
- Hard negative: different object that looks similar or is nearby.
- Feature bank: saved embeddings over time.

Good sources:

- Scikit-learn cosine similarity docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- QDTrack local paper: `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- StrongSORT local paper: `papers/strongsort_make_deepsort_great_again.pdf`
- BoT-SORT local paper: `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf`

Tiny exercises:

1. Make three vectors:

   ```python
   a = [1.0, 0.0]
   b = [0.9, 0.1]
   c = [0.0, 1.0]
   ```

2. Normalize them.
3. Compute cosine similarity by hand or NumPy.
4. Decide which vector is most similar to `a`.
5. Make a tiny feature bank:

   ```python
   track_memory = {
       1: [embedding_a1, embedding_a2],
       2: [embedding_b1, embedding_b2],
   }
   ```

6. Score a new candidate against each track memory.

Project files to connect:

- `extract_reid_histogram()` in `programs/bounding_box_v5_lorat_shared.py`.
- `V8FeatureIdentityArbitrator` in `programs/bounding_box_v8_lorat_quality_batched.py`.
- `contrastive_reid_loss()` in `programs/train_lorat_v8_head.py`.

Checkpoint:

You can explain the difference between:

- a bbox score;
- a motion score;
- an appearance/ReID score;
- an assignment margin.

## Level 7: PyTorch Tensors And GPU Basics

What this means:

PyTorch is the library used for model inference and training. A tensor is like a NumPy array that can live on the GPU and can track gradients for learning.

Why it matters:

V8's shared-frame head and V8 training are tensor code. Bugs often come from wrong shapes, wrong devices, or accidentally mixing CPU and GPU tensors.

Learn first:

- Tensor.
- Shape.
- Batch dimension.
- Device: CPU, CUDA, DirectML.
- `float32`.
- `torch.no_grad()` for inference.
- `requires_grad`.
- `nn.Module`.
- Loss.
- Optimizer.
- Training loop.
- Dataset and DataLoader.

Good sources:

- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch Quickstart: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- PyTorch tensors tutorial: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- PyTorch autograd tutorial: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- PyTorch Datasets and DataLoaders: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch training loop video/tutorial page: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

Tiny exercises:

1. Create tensors:

   ```python
   import torch
   x = torch.randn(4, 8)
   y = torch.randn(8, 3)
   z = x @ y
   print(z.shape)
   ```

2. Move a tensor to CUDA if available:

   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   x = x.to(device)
   ```

3. Create a fake batch:

   ```python
   features = torch.randn(5, 196, 768)  # objects, locations, dim
   scores = torch.randn(5, 196)         # objects, locations
   ```

4. Find the best location per object:

   ```python
   best = scores.argmax(dim=1)
   ```

5. Write a tiny model with `nn.Linear`.
6. Train it on a toy problem until the loss decreases.

Project files to connect:

- `BatchedObjectConditionedHead` in `programs/bounding_box_v8_lorat_quality_batched.py`.
- `MOTFrameHeadDataset` in `programs/train_lorat_v8_head.py`.
- `make_lorat_style_targets()` in `programs/train_lorat_v8_head.py`.

Checkpoint:

You can look at a tensor shape and say what each dimension means.

## Level 8: Training A Model Without Panic

What this means:

Training is a loop:

```text
load batch -> model predicts -> compute loss -> backward pass -> optimizer step -> validate -> save checkpoint
```

Why it matters:

The current V8 head needs better training. To improve it, you need to understand targets, losses, validation, overfitting, and diagnostics.

Learn first:

- Dataset sample.
- Batch.
- Ground truth.
- Target map.
- Prediction.
- Loss.
- Backpropagation.
- Optimizer.
- Epoch.
- Validation.
- Overfit smoke test.
- Checkpoint.
- Diagnostic CSV.

Good sources:

- PyTorch training loop tutorial: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
- PyTorch autograd tutorial: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- PyTorch Datasets and DataLoaders: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- GIoU local paper: `papers/giou_generalized_intersection_over_union.pdf`
- VarifocalNet local paper: `papers/varifocalnet_iou_aware_dense_detector.pdf`

Tiny exercises:

1. Train a one-layer model on a fake problem:

   ```text
   input: width, height
   target: area
   ```

2. Print loss every epoch.
3. Save the model.
4. Load it back.
5. Make an overfit test with only 8 samples.
6. Confirm the model can memorize those 8 samples.
7. Add a CSV with epoch, train loss, val loss.

Project files to connect:

- `train_lorat_v8_head.py`
- `validate_head()`
- `save_checkpoint()`
- `append_diagnostic_row()`
- `outputs/v8_training_results_*`

Checkpoint:

You can explain why "the loss went down" is not enough. The tracker also needs IoU, correct-object rate, ID switches, and validation behavior.

## Level 9: Benchmarks And Metrics

What this means:

A benchmark is a repeatable experiment. Metrics are numbers that say how the tracker behaved.

Why it matters:

The final deliverable is not just a tool. It is a benchmark suite that measures human effort required to reach target labeling quality.

Learn first:

- FPS.
- Seconds per output box.
- GPU memory.
- IoU.
- Correct-object rate.
- ID switch.
- Track loss.
- Occlusion survival.
- HOTA.
- IDF1.
- MOTA.
- Human cost events.

Good sources:

- TrackEval README: https://github.com/JonathonLuiten/TrackEval
- MOTChallenge format docs: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt
- HOTA paper: https://arxiv.org/abs/2009.07736
- HOTA overview: https://autonomousvision.github.io/hota-metrics/

Tiny exercises:

1. Make a fake tracker output with 3 frames.
2. Make fake ground truth with 3 frames.
3. Compute IoU frame by frame.
4. Add one identity switch and count it.
5. Add one missing frame and count track loss.
6. Write a tiny `summary.md` with:

   - frames processed;
   - boxes output;
   - mean IoU;
   - ID switches;
   - seconds per box.

Project files to connect:

- `programs/benchmark_lorat_mot.py`
- `programs/benchmark_lorat_v8.py`
- `external/TrackEval-master/trackeval/metrics/hota.py`
- `external/TrackEval-master/trackeval/metrics/identity.py`

Checkpoint:

You can explain the difference between:

- speed benchmark;
- quality benchmark;
- human-effort benchmark.

## Level 10: Reading This Project's Large Files

What this means:

Some files are thousands of lines long. You need a strategy for reading them without getting buried.

Why it matters:

V8 is large because it has model loading, GUI, tracking state, tensor scoring, debug logging, CLI args, and benchmark hooks in one script. You need to read by responsibility, not from line 1 to line 4000.

Learn first:

- Search for class names.
- Search for function names.
- Read dataclasses first.
- Read `main()` second.
- Read the core update function third.
- Ignore CLI options until you need them.
- Add temporary prints or debug rows, then remove them once understood.

Good sources:

- Python tutorial modules section: https://docs.python.org/3/tutorial/modules.html
- Python tutorial classes section: https://docs.python.org/3/tutorial/classes.html
- Python tutorial errors/exceptions section: https://docs.python.org/3/tutorial/errors.html

Tiny exercises:

1. Run:

   ```powershell
   rg -n "class V8QualityBatchedLoRATTracker|def update|def _score_and_update_tracks" .\programs\bounding_box_v8_lorat_quality_batched.py
   ```

2. Write a one-sentence note for each function you find.
3. Find where V8 creates tracks.
4. Find where V8 accepts a candidate.
5. Find where V8 holds a track.
6. Find where V8 writes debug rows.

Checkpoint:

You can answer:

- Where does the frame enter?
- Where does the model run?
- Where are candidates decoded?
- Where is identity resolved?
- Where is memory updated?
- Where are outputs written?

## Beginner Roadmap By Summer Week

### Before Week 1 Work

Study:

- Python basics.
- OpenCV images/video/mouse.
- Bounding boxes and IoU.
- CSV writing.

Build:

- A one-image rectangle drawing script.
- A drag-to-draw bbox script.
- A CSV writer for fake MOT rows.

Then read:

- `programs/bounding_box_basic.py`
- `programs/bounding_box_v2_opencv.py`
- `programs/benchmark_lorat_mot.py`

### Before Week 2 Work

Study:

- NumPy shapes.
- PyTorch tensor shapes.
- Batch dimensions.
- Timing code.
- GPU device basics.

Build:

- A fake batched scoring matrix.
- A timing wrapper using `time.perf_counter()`.
- A tiny script comparing one-by-one scoring versus batched scoring.

Then read:

- `docs/v8_code_walkthrough.md`
- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`

### Before Week 3 Work

Study:

- Cosine similarity.
- Assignment matrices.
- Hungarian assignment.
- Track state.
- ReID memory banks.
- Debug CSV design.

Build:

- A fake 3-track assignment solver.
- A fake embedding memory bank.
- A function that counts ID switches from simple rows.
- A `manual_reanchor` event CSV writer.

Then read:

- `docs/week3_reid_recovery_todo.md`
- `programs/bounding_box_v5_lorat_shared.py`
- `programs/train_lorat_v8_head.py`
- `programs/benchmark_lorat_v8.py`

### Before Week 4 Work

Study:

- Object detection outputs.
- Proposal boxes.
- Proposal recall.
- Difference between accepted tracks and unaccepted proposals.

Build:

- A fake proposal queue.
- A proposal recall calculator.
- A GUI overlay that draws proposals in a different color than tracks.

Then read:

- `docs/week3_gap_fill_papers.md`
- Grounding DINO, SAM, SAM 2 notes/papers.

### Before Week 5 Work

Study:

- Uncertainty scoring.
- Sorting review queues.
- Active learning idea: ask the human where the model is most unsure.
- Cost events.

Build:

- A review queue from fake debug rows.
- A cost CSV with correction events.
- A simple plot or CSV showing quality versus number of corrections.

Then read:

- Active annotation section in `docs/week3_gap_fill_papers.md`.

### Before Weeks 6-12 Work

Study:

- MOT export format.
- TrackEval expected folders.
- HOTA, IDF1, and MOTA at a conceptual level.
- Git/versioning basics.
- Reproducible benchmark metadata.

Build:

- A tiny MOT-format file.
- A run metadata JSON writer.
- One ablation table with a feature on/off.

Then read:

- `external/TrackEval-master`
- HOTA paper and TrackEval docs.

## "I Am Stuck" Debug Routine

When something does not make sense:

1. Shrink it.
   Run fewer frames, fewer tracks, smaller tensors, or fake data.

2. Print shapes.
   For model bugs, print tensor shapes before values.

3. Print one row.
   For CSV/benchmark bugs, print the first row written and read it by hand.

4. Freeze randomness.
   Use a fixed seed when comparing behavior.

5. Name the invariant.
   Example: "There should be one shared frame encode per frame." Then write a debug check for that.

6. Compare against a toy case.
   If V8 is confusing, make a 2-track fake example and solve it on paper.

7. Write down the question.
   Good question: "Why did track 3 hold on frame 72?"
   Hard-to-debug question: "Why is tracking bad?"

## The Minimum Beginner Vocabulary

You should be able to define these in your own words:

- script
- function
- class
- dataclass
- argument
- path
- CSV
- frame
- image sequence
- pixel
- array
- shape
- channel
- bbox
- crop
- IoU
- tracker
- track ID
- candidate
- assignment
- confidence
- embedding
- cosine similarity
- ReID
- tensor
- batch
- device
- model
- loss
- optimizer
- checkpoint
- benchmark
- metric
- human-cost event

If any word in that list feels fuzzy, use the levels above to turn it into a tiny script.
