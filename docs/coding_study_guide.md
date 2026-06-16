# Coding Study Guide For The LoRAT MOT Labeler

Date: 2026-06-11

Note: this guide has been consolidated into `docs/project_learning_guide.md`. Use that file as the main study path.

This guide is for learning how to code more of this project yourself. It is organized around the actual repository, the summer onboarding statement of work, and the research papers already collected locally.

The big picture: you are building a video annotation tool where a user draws one box on an object, and the system tracks that object and other user-selected objects across the video. The research version is a multi-object extension of LoRAT with ReID, open-world discovery, and active correction.

If any term here feels too advanced, start with the beginner companion guide: `docs/coding_study_guide_beginner_layer.md`. It breaks the basics down from terminal use, Python, arrays, OpenCV, and bounding boxes up through tensors, ReID, training, and benchmarks.

For the computer-vision concepts behind this project specifically, read `docs/computer_vision_sot_mot_foundations.md`. It explains images, boxes, SOT, MOT, SOT-based MOT, and why our LoRAT direction is different from ordinary detector-based MOT.

## What To Learn First

Learn in this order:

1. Python project navigation and debugging.
2. Bounding boxes, video frames, and OpenCV GUI/event loops.
3. Single-object tracking versus multi-object tracking.
4. Identity association: motion, appearance, Hungarian assignment, occlusion holds.
5. PyTorch tensor batching and GPU device handling.
6. LoRAT/Vision Transformer features and the V8 shared-frame architecture.
7. Training a tracker head: datasets, targets, losses, diagnostics.
8. Benchmarking: timing, IoU, ID switches, track loss, HOTA/IDF1, human effort.

Do not try to understand every line of V8 first. V8 is easier after you have read the smaller prototypes.

## Repository Map

### Core Runtime Files

| File | What It Teaches | Study Priority |
| --- | --- | --- |
| `programs/bounding_box_basic.py` | Minimal OpenCV object selection and tracking loop. | First warmup. |
| `programs/bounding_box_v2_opencv.py` | Multiple boxes, frame sources, GUI controls, MOT output. | First serious read. |
| `programs/bounding_box_v3_lorat.py` | First large LoRAT-backed multi-object tracker. | Read for LoRAT integration history. |
| `programs/bounding_box_v4_lorat_memory.py` | LoRAT memory slots, identity arbitration, Kalman holding. | Read for ReID/control logic ideas. |
| `programs/bounding_box_v5_lorat_shared.py` | Shared utility layer reused by later versions. | Essential reference. |
| `programs/bounding_box_v6_lorat_gated.py` | Slot gating and recovery slot selection. | Short, useful bridge. |
| `programs/bounding_box_v7_lorat_frame_shared.py` | First shared-frame LoRAT architecture. | Read before V8. |
| `programs/bounding_box_v8_lorat_quality_batched.py` | Current main tracker: shared frame encoder, object-conditioned batched head, feature identity. | Main implementation target. |

### Training And Benchmark Files

| File | What It Teaches | Study Priority |
| --- | --- | --- |
| `programs/exercise_lorat_mot.py` | DanceTrack sequence loading, GT initialization, basic evaluation. | Early. |
| `programs/benchmark_lorat_mot.py` | Week 1 timing and small-object benchmarks. | Early. |
| `programs/benchmark_lorat_v5.py` | More detailed identity/debug benchmarking. | Medium. |
| `programs/benchmark_lorat_v6_forced_area.py` | Controlled small-object area stress testing. | Medium. |
| `programs/benchmark_lorat_v7.py` | Week 2 shared-frame proof and identity observations. | High. |
| `programs/benchmark_lorat_v8.py` | Current benchmark runner for V8 quality, candidates, throughput, and proof logs. | High. |
| `programs/train_lorat_v7_head.py` | Small first version of head training. | Read before V8 training. |
| `programs/train_lorat_v8_head.py` | Current head training script: LoRAT-style targets, augmentation, box loss, ReID loss hooks, diagnostics. | Main training target. |
| `programs/tune_lorat_v4_debug.py` | How to read debug CSVs and tune thresholds. | Useful for debugging. |

### Scripts And Local Operations

| File | What It Is For |
| --- | --- |
| `scripts/setup-lorat-env.ps1` | Local Python/LoRAT environment setup. |
| `scripts/verify-lorat-env.ps1` | Checks whether the local LoRAT environment works. |
| `scripts/fetch-assets.ps1` | Pulls datasets, models, papers, and external repos from the manifest. |
| `scripts/theia-stage-v*.ps1` | Packages local code/data for HPC runs. |
| `scripts/theia_v*_benchmark.sbatch` | HPC benchmark jobs. |
| `scripts/theia_v8_train_heads.sbatch` | HPC training job for V8 heads. |
| `scripts/make_v6_v8_presentation_visuals.py` | Converts benchmark outputs into presentation visuals. |

### Study Notes Already In The Repo

Read these in this order:

1. `docs/v8_code_walkthrough.md`
2. `docs/week3_reid_recovery_todo.md`
3. `docs/v8_training_methods_research.md`
4. `docs/v8_papers_training_onboarding_review.md`
5. `docs/week3_gap_fill_papers.md`
6. `docs/computer_vision_sot_mot_foundations.md`
7. `docs/sotmot_lorat_mot_notes.md`
8. `docs/data_models_and_papers.md`
9. `docs/amd_nvidia_platform_notes.md`

The extracted summer onboarding text is at `outputs/summer_onboarding_extracted.txt`.

## Summer Onboarding Translated Into Coding Skills

### Week 1: Multi-Object Tracker, GUI, Core Benchmarks

You need to know:

- How OpenCV reads video frames and image sequences.
- How mouse callbacks collect user-drawn boxes.
- How bounding boxes move between `(x, y, w, h)` and `(x1, y1, x2, y2)`.
- How to write MOTChallenge-format rows.
- How to measure seconds per output box.
- How to bin objects by pixel area and compute IoU reliability.

Ways to learn it:

- Start with a tiny OpenCV script before touching LoRAT. Load one image, draw a rectangle on it, display it, and save it. Then load a video or image sequence and show frames in a loop.
- Rebuild the box math by hand. Write `xywh_to_xyxy()`, `xyxy_to_xywh()`, `bbox_area()`, `bbox_center()`, and `bbox_iou()` in a scratch file. Test them with boxes you can reason about visually.
- Trace one GUI event. In `bounding_box_v2_opencv.py`, follow the mouse-down, mouse-move, mouse-up, accept-box path until a selected box becomes a `TrackState`.
- Learn MOT output format by writing five fake rows yourself, then compare them to rows written by `append_mot_results()`.
- Use CSVs as your feedback loop. Run a 5-10 frame smoke test, open the timing and area CSVs, and explain each column in plain English.
- Build one micro-feature: add a printed message or CSV column showing initial box area. This forces you to touch CLI, box state, and output without deep model code.

Files to study:

- `programs/bounding_box_basic.py`
- `programs/bounding_box_v2_opencv.py`
- `programs/exercise_lorat_mot.py`
- `programs/benchmark_lorat_mot.py`

Practice tasks:

- Add a command-line flag to change the output folder.
- Print the selected boxes before tracking starts.
- Write a tiny function that computes IoU and test it on three hand-made boxes.
- Run a 10-frame smoke benchmark and explain each CSV column.

### Week 2: Shared Backbone And Throughput Scaling

You need to know:

- Why running one LoRAT instance per object is slow.
- How a shared ViT feature map lets all objects reuse one frame encoding.
- How tensor shapes represent locations, objects, and embeddings.
- How GPU memory and FPS are measured.

Ways to learn it:

- First learn batching without neural networks. Make a NumPy array shaped `[objects, features]`, compute pairwise dot products, and print the resulting `[objects, objects]` matrix.
- Then learn PyTorch tensor shapes. Create random tensors shaped like V8's comments: `[locations, dim]`, `[objects, dim]`, and `[objects, locations]`. Practice `unsqueeze`, `expand`, `matmul`, `softmax`, and `argmax`.
- Compare V5 and V8 mentally. V5 asks "run tracker slot A, then tracker slot B." V8 asks "encode frame once, then score all object memories." Write that flow on paper before reading the code.
- In `bounding_box_v8_lorat_quality_batched.py`, find the exact methods that encode the frame, score the batched head, and append Week 2 proof rows. Read only those methods first.
- Run the benchmark with `N=1` and `N=2`, then compare proof logs. The learning target is to see the backbone call count stay fixed while object-head items grow.
- Build one micro-feature: add a timing/debug field for one internal step, then make sure it appears in the runtime status or benchmark CSV.

Files to study:

- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/benchmark_lorat_v7.py`
- `programs/benchmark_lorat_v8.py`
- `docs/v8_code_walkthrough.md`

Practice tasks:

- Find where V8 encodes the current frame once.
- Find where V8 batches all object heads together.
- Add one new timing bucket and report it in the runtime status.
- Explain the difference between "one shared backbone pass" and "one object head batch."

### Week 3: ReID And Track Recovery

You need to know:

- What an appearance embedding is.
- Why identity switches happen.
- How a tracker decides between "accept this candidate" and "hold the track."
- How Hungarian assignment works.
- How to log a human correction event.

Ways to learn it:

- Learn embeddings with a toy example. Make three 2D vectors by hand, normalize them, compute cosine similarity, and decide which two are most alike. That is the tiny version of ReID.
- Watch an identity switch happen. Pick a debug CSV or short video where two people cross. Find the frame where the tracker chooses the wrong object, then list what signals should have warned it: low ReID, bad motion, low margin, overlap, or jumpy size.
- Learn Hungarian assignment as a table. Draw a 3-track by 3-candidate score matrix on paper. Pick the best global assignment, then compare with what greedy matching would do wrong.
- Read V5's identity resolver before V8's. V5 is more Python/scalar and easier to understand. Then read V8's matrix version as the batched upgrade.
- Implement one metric in isolation. Given rows of `(frame, tracker_id, matched_gt_id)`, count identity switches. Do this in a scratch script before wiring it into a benchmark.
- Build one micro-feature: add a `track_state` or `reject_reason` debug field. This teaches the state machine without requiring a better model.
- For human-cost logging, design the CSV first. Write down columns for `event_type`, `frame`, `track_id`, `old_bbox`, `new_bbox`, and `seconds_spent`; then make code write one event.

Files to study:

- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`
- `programs/benchmark_lorat_v8.py`
- `docs/week3_reid_recovery_todo.md`
- `docs/week3_gap_fill_papers.md`

Practice tasks:

- Add a `track_state` text field to a debug row.
- Log when a track is held instead of accepted.
- Write a small function that pools a feature vector inside a bbox.
- Add a CSV event row for `manual_reanchor` without changing tracking behavior yet.
- Run V8 with ReID-like identity features enabled and disabled, then compare ID switch counts once the metric is wired.

### Week 4: Open-World Object Discovery

You need to know:

- Difference between tracking an accepted object and proposing an unknown object.
- Class-agnostic proposals versus open-vocabulary detection.
- How a proposal queue should avoid becoming part of the tracker until the user accepts it.

Ways to learn it:

- Separate the concepts in code. A `TrackState` is committed and has an ID; a `Proposal` is just a candidate with a bbox, score, source, and frame number. Write those as two dataclasses and compare fields.
- Mock proposals before installing a big model. Generate three fake boxes per frame and display them differently from tracked boxes. This teaches the UI and queue behavior without Grounding DINO or SAM complexity.
- Learn proposal recall with ground truth. For each GT box, ask whether any proposal overlaps it above an IoU threshold. This is the core Week 4 metric.
- Read one open-world paper for interface implications, not model details. Ask: what does it output, how confident is it, and how would a user accept/reject it?
- Build one micro-feature: add a proposal queue CSV with `frame`, `bbox`, `score`, `source`, `accepted`, and `spawned_track_id`.
- Keep the tracker boundary clean. Practice making a proposal become a track only when a user accepts it, because this prevents open-world discovery from contaminating tracking logic.

Files and papers to study:

- `docs/week3_gap_fill_papers.md`
- `papers/grounding_dino.pdf`
- `papers/segment_anything.pdf`
- `papers/sam2_segment_anything_images_videos.pdf`
- `papers/owlvit_simple_open_vocabulary_detection_vit.pdf`
- `papers/video_owlvit_open_world_video_localization.pdf`

Practice tasks:

- Define a proposal dataclass with bbox, score, source, and frame number.
- Add an empty proposal queue to the GUI overlay.
- Write CSV output for accepted/rejected proposals.

### Week 5: Active Correction Loop

You need to know:

- How to turn uncertainty into a review queue.
- Why "most uncertain frame" can save human effort.
- How correction propagation changes future boxes.
- How to measure human cost.

Ways to learn it:

- Start with a sortable list. Make a fake list of frame/track events with confidence, margin, jitter, and ReID distance. Compute an uncertainty score and sort it.
- Learn each uncertainty term separately. Plot or print examples of low confidence, low assignment margin, high bbox jitter, and high ReID distance. Do not combine them until each one makes sense alone.
- Simulate correction by editing data, not the GUI. Take one track row, replace its bbox with GT, and recompute IoU for following frames. That is the simplest version of propagation.
- Read active learning papers by looking for "what did the system ask the human to label next?" Then map that to our review queue.
- Build one micro-feature: write `review_queue.csv` from existing debug rows. Include `frame`, `track_id`, `uncertainty`, `reason`, and `suggested_action`.
- Measure cost in units you can defend: number of initial boxes, number of correction boxes, number of manual reanchors, number of verification-only frames, and wall-clock seconds if available.

Files and papers to study:

- `docs/week3_gap_fill_papers.md`
- `papers/video_annotation_tracking_active_learning.pdf`
- `papers/vatic_efficient_crowdsourced_video_annotation.pdf`
- `papers/efficient_video_annotation_visual_interpolation_frame_selection.pdf`
- `papers/hdamot_active_learning_multi_object_tracking.pdf`

Practice tasks:

- Add an uncertainty score made from confidence, margin, and box jitter.
- Sort track/frame events by uncertainty.
- Create an event CSV schema for `initial_box`, `manual_reanchor`, `correction`, and `verify`.

### Weeks 6-12: Benchmarks, Baselines, Paper

You need to know:

- TrackEval/HOTA/IDF1/MOTA.
- MOT and COCO-video export formats.
- Baseline tracker interfaces.
- Ablation tables.
- Reproducible scripts and frozen outputs.

Ways to learn it:

- Start with one metric, not all metrics. Learn IoU first, then ID switches/IDF1, then HOTA. HOTA becomes much easier when you already understand detection quality and association quality separately.
- Treat TrackEval as an input/output contract. Find what folder structure and text files it expects, then make one tiny fake sequence pass through it.
- Learn exports by round trip. Write a small MOT-format file, read it back, and verify every bbox matches what you wrote.
- Build baselines behind the same interface. Define a small tracker API such as `initialize(frame, boxes)`, `update(frame)`, and `tracks()`. ByteTrack, BoostTrack, and V8 should become swappable behind that shape.
- Learn ablations as controlled experiments. Change one feature, keep sequence/frames/model/seed fixed, and record what changed in speed, quality, and human cost.
- Practice reproducibility early. Every benchmark output should include command, date, dataset path, model size, device, seed, and git/version info when available.
- Build one micro-feature: add a small `run_metadata.json` next to benchmark outputs with device, command, and key arguments.

Files and papers to study:

- `external/TrackEval-master/trackeval/metrics/hota.py`
- `external/TrackEval-master/trackeval/metrics/identity.py`
- `papers/hota_metric.pdf`
- `papers/bytetrack.pdf`
- `papers/boosttrack_similarity_confidence_mot.pdf`
- `papers/boosttrack_plus_plus_tracklet_information_mot.pdf`
- `papers/oc_sort_observation_centric_sort.pdf`

Practice tasks:

- Convert one tracker output into TrackEval's expected layout.
- Add a "tracker backend" field to benchmark CSVs.
- Write a small ablation table by turning one feature on/off.

## Project Concepts To Master

### Bounding Boxes

You must be fluent with:

- `xywh`: `(x, y, width, height)`
- `xyxy`: `(x1, y1, x2, y2)`
- area: `width * height`
- center: `(x + width / 2, y + height / 2)`
- IoU: overlap area divided by union area
- GIoU: IoU plus a penalty for boxes far apart

Code locations:

- `programs/bounding_box_v5_lorat_shared.py`: geometry helpers.
- `programs/train_lorat_v8_head.py`: target boxes and GIoU.
- `programs/benchmark_lorat_v8.py`: quality measurements.

### Track State

A tracker is not just a box predictor. Each object needs state:

- current bbox
- previous bbox and velocity
- confidence
- appearance memory
- occlusion count
- recent reliable path
- whether memory should refresh
- whether the object is healthy, uncertain, lost, or reacquired

Code locations:

- `TrackState` in `programs/bounding_box_v5_lorat_shared.py`
- V8 accept/hold logic in `programs/bounding_box_v8_lorat_quality_batched.py`

### Identity Association

Multi-object tracking needs to answer:

> Which candidate box belongs to which existing track?

The score usually mixes:

- appearance similarity
- motion agreement
- path agreement
- overlap/occlusion checks
- assignment margin
- confidence

Code locations:

- `LightweightIdentityArbitrator` in `programs/bounding_box_v5_lorat_shared.py`
- `V8FeatureIdentityArbitrator` in `programs/bounding_box_v8_lorat_quality_batched.py`

Papers:

- `papers/deep_sort.pdf`
- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- `papers/strongsort_make_deepsort_great_again.pdf`
- `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf`
- `papers/oc_sort_observation_centric_sort.pdf`

### LoRAT And V8

LoRAT is a single-object tracker. V8 is our experimental multi-object branch.

V8 changes the shape:

```text
old pattern:
  object 1 -> LoRAT frame pass
  object 2 -> LoRAT frame pass
  object N -> LoRAT frame pass

V8 pattern:
  frame -> one shared LoRAT/DINOv2 feature map
  all object memories -> one batched object-conditioned head
```

Code locations:

- `SharedFrameLoRATEncoder` in `programs/bounding_box_v8_lorat_quality_batched.py`
- `BatchedObjectConditionedHead` in `programs/bounding_box_v8_lorat_quality_batched.py`
- `V8QualityBatchedLoRATTracker` in `programs/bounding_box_v8_lorat_quality_batched.py`

Papers:

- `papers/lorat_tracking_meets_lora.pdf`
- `papers/lorat_supplemental_training_details.pdf`
- `papers/dinov2_learning_robust_visual_features_without_supervision.pdf`
- `papers/vision_transformer_an_image_is_worth_16x16_words.pdf`

### Training

The training script should teach you:

- how a dataset class returns samples;
- how target score maps and box maps are built;
- how predictions decode back into pixel boxes;
- how losses combine objectness, box quality, and ReID;
- why diagnostics matter.

Code locations:

- `MOTFrameHeadDataset` in `programs/train_lorat_v8_head.py`
- `make_lorat_style_targets()` in `programs/train_lorat_v8_head.py`
- `decode_box_maps_xyxy()` in `programs/train_lorat_v8_head.py`
- `contrastive_reid_loss()` in `programs/train_lorat_v8_head.py`
- `validate_head()` in `programs/train_lorat_v8_head.py`

Papers:

- `papers/giou_generalized_intersection_over_union.pdf`
- `papers/varifocalnet_iou_aware_dense_detector.pdf`
- `papers/detr_end_to_end_object_detection.pdf`
- `papers/deformable_detr.pdf`
- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`

## Four-Week Self-Study Plan

### Week A: Python And OpenCV Tracking Basics

Goal: understand the GUI loop and box data flow.

Read:

- `programs/bounding_box_basic.py`
- `programs/bounding_box_v2_opencv.py`
- first half of `README.md`

Do:

- Run `python programs/bounding_box_basic.py --help`.
- Run `python programs/bounding_box_v2_opencv.py --help`.
- Draw a box, track it for a few frames, and find where the output rows are written.
- Write your own `bbox_iou()` in a scratch file, then compare it to the project helper.

Checkpoint:

- You can explain how a mouse drag becomes a stored bbox.
- You can explain how a frame source differs from a tracker backend.

### Week B: Multi-Object Tracking Logic

Goal: understand why MOT is mostly identity management.

Read:

- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v6_lorat_gated.py`
- `docs/sotmot_lorat_mot_notes.md`

Papers:

- `papers/deep_sort.pdf`
- `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`
- `papers/oc_sort_observation_centric_sort.pdf`

Do:

- Trace one track through accept, hold, and memory update.
- Add one debug column to show why a candidate was rejected.
- Use `programs/tune_lorat_v4_debug.py` on an existing debug CSV and summarize what went wrong.

Checkpoint:

- You can explain confidence, ReID, motion, path, and occlusion gates.
- You can explain why a bad memory update can ruin a track.

### Week C: V8 Shared-Frame Architecture

Goal: understand the current main tracker.

Read:

- `docs/v8_code_walkthrough.md`
- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`

Papers:

- `papers/lorat_tracking_meets_lora.pdf`
- `papers/dinov2_learning_robust_visual_features_without_supervision.pdf`
- `papers/ostrack_one_stream_tracking.pdf`

Do:

- Draw the V8 data flow from frame to final track.
- Add one comment or local note explaining each tensor shape in the V8 head.
- Run a very short V8 smoke test with `--max-frames` and inspect the proof/debug outputs.

Checkpoint:

- You can explain why V8 is not just "multiple LoRATs."
- You can point to the shared frame encoder, batched head, identity resolver, accept/hold logic, and memory refresh logic.

### Week D: Training And Week 3 Features

Goal: become able to modify the training/recovery path.

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

- Run `python programs/train_lorat_v8_head.py --help`.
- Find the code that builds targets, computes losses, validates, and writes diagnostics.
- Add or verify an overfit smoke mode.
- Add one ReID diagnostic: same-ID cosine mean, different-ID cosine mean, or retrieval top-1.
- Add one event log row for manual reanchor.

Checkpoint:

- You can explain what the head is trained to predict.
- You can explain why ReID needs positives, negatives, and hard negatives.
- You can identify what is missing for the Week 3 deliverable.

## Command Habits

Use these while learning:

```powershell
rg -n "class V8QualityBatchedLoRATTracker|def update|def _score_and_update_tracks" .\programs\bounding_box_v8_lorat_quality_batched.py
rg -n "def make_lorat_style_targets|def contrastive_reid_loss|def validate_head" .\programs\train_lorat_v8_head.py
rg -n "Identity|ReID|occlusion|manual_reanchor|HOTA|IDF1" .\docs .\programs
python .\programs\bounding_box_v8_lorat_quality_batched.py --help
python .\programs\benchmark_lorat_v8.py --help
python .\programs\train_lorat_v8_head.py --help
```

When you change code:

```powershell
python -m py_compile .\programs\bounding_box_v8_lorat_quality_batched.py
python -m py_compile .\programs\train_lorat_v8_head.py
python -m py_compile .\programs\benchmark_lorat_v8.py
```

When debugging:

- Prefer short smoke runs before long training runs.
- Save CSVs for anything you cannot explain from the screen.
- Add one debug column at a time.
- Compare the same sequence, same frame range, same selected object IDs.
- Separate throughput results from quality results.

## Good First Code Contributions

These are realistic tasks that build confidence without requiring a full rewrite.

1. Add a `track_state` field to V8 debug output.
2. Add a `manual_reanchor` event CSV schema and writer.
3. Add a small overfit smoke command to V8 training.
4. Add a same-ID versus different-ID ReID diagnostic to training.
5. Add a benchmark switch for ReID on/off.
6. Add an occlusion-gap stress mode that hides a GT object for N frames.
7. Add TrackEval-compatible export validation for one DanceTrack sequence.
8. Add a short `summary.md` section that splits speed, quality, and human-cost metrics.

## How To Read Papers For Coding

Do not read every paper cover to cover at first. Read with one coding question in mind.

Use this pattern:

1. Read the abstract and method diagram.
2. Find the training objective or scoring formula.
3. Find what data structure the method keeps over time.
4. Find what metric proves it worked.
5. Write a 5-line note: "What would this change in our code?"

Examples:

- DeepSORT: What appearance vector does it store, and how is assignment scored?
- QDTrack: What are positives and negatives for identity learning?
- OC-SORT: What can motion solve without appearance?
- XMem: What kinds of memory should be short-term versus long-term?
- VATIC/Efficient Video Annotation: What counts as human effort?
- LoRAT: What is template/search training, and what parts can V8 preserve?

## Mental Model Of The Whole System

Keep this picture in your head:

```text
User draws box
  -> create track and memory
  -> frame arrives
  -> encode frame once
  -> score each object against frame features
  -> decode candidate boxes
  -> associate candidates to tracks
  -> accept, hold, lose, or reacquire
  -> update memory only from trusted frames
  -> draw boxes and write outputs
  -> benchmark speed, quality, and human cost
```

Every feature in the summer plan fits into one part of that loop.

## What To Avoid While Learning

- Do not start by rewriting V8 from scratch.
- Do not add open-world proposals before the tracker can reliably keep identity.
- Do not trust a trained head until it can overfit a tiny slice.
- Do not use FPS alone as proof of success.
- Do not refresh appearance memory from uncertain boxes.
- Do not compare runs unless they use the same sequence, frames, tracks, and model size.

## North-Star Skill

The skill you are building is not "memorize LoRAT." It is:

> Given a video frame, tracked object memory, and uncertain model outputs, write code that makes a careful state update and proves the update improved labeling quality per unit of human effort.

That is the heart of this project.
