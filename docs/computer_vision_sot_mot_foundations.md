# Computer Vision, SOT, MOT, And SOT-Based MOT Foundations

Date: 2026-06-11

Note: this guide has been consolidated into `docs/project_learning_guide.md`. Use that file as the main study path.

This note explains the computer vision ideas that matter for this project, from very basic image concepts up through single-object tracking, multi-object tracking, and the more specific idea of building MOT from SOT-style target-conditioned tracking.

The goal is not to become a general computer vision expert first. The goal is to learn exactly enough computer vision to understand and modify this tool:

```text
user box -> object memory -> video frames -> tracked boxes -> identity recovery -> correction queue -> benchmark
```

## 1. Computer Vision In This Project

Computer vision means writing programs that interpret images and video.

For this project, "interpret" mostly means:

- locate objects with bounding boxes;
- keep each object ID consistent over time;
- notice when tracking is uncertain;
- ask the human for help only when useful;
- measure quality and human effort.

You do not need to begin with all of computer vision. Start with these project-specific building blocks:

1. Images are arrays of pixel values.
2. Videos are ordered lists of images.
3. Objects are represented by bounding boxes.
4. A tracker predicts where a selected object moved.
5. A multi-object tracker also preserves identity.
6. A ReID system compares appearance across time.
7. A benchmark compares predictions to ground truth.

## 2. Images, Pixels, Frames, And Coordinates

An image is a grid:

```text
height rows x width columns x color channels
```

OpenCV usually stores color images as:

```text
frame[y, x, channel]
```

Important detail:

- `x` means horizontal position.
- `y` means vertical position.
- NumPy indexing uses `frame[y, x]`, not `frame[x, y]`.
- OpenCV color order is usually BGR, not RGB.

A video is a sequence:

```text
frame 1, frame 2, frame 3, ...
```

In DanceTrack and MOT17, a video sequence is often stored as many image files:

```text
img1/00000001.jpg
img1/00000002.jpg
img1/00000003.jpg
```

Project files:

- `programs/bounding_box_v2_opencv.py`: frame sources and image sequence reading.
- `programs/exercise_lorat_mot.py`: DanceTrack sequence discovery and ground-truth reading.
- `programs/bounding_box_v5_lorat_shared.py`: reusable frame source classes.

Tiny exercise:

Write a script that opens one DanceTrack frame, prints its shape, draws the frame number on it, and saves it.

## 3. Bounding Boxes

A bounding box is a rectangle around an object.

Two common formats:

```text
xywh = x, y, width, height
xyxy = left, top, right, bottom
```

This project mostly uses `xywh` for visible track boxes.

You need these operations:

- convert between `xywh` and `xyxy`;
- compute area;
- compute center;
- clip a box to image bounds;
- crop image pixels inside a box;
- compute IoU against ground truth.

IoU means intersection over union:

```text
IoU = overlapping area / total area covered by both boxes
```

Why it matters:

- Week 1 small-object reliability uses box area.
- Training loss uses box overlap ideas like GIoU.
- Benchmarks use IoU to decide whether a predicted box is good.
- Correction cost matters only if final boxes are good.

Project files:

- `programs/exercise_lorat_mot.py`: simple `bbox_iou()`.
- `programs/bounding_box_v5_lorat_shared.py`: geometry helpers.
- `programs/train_lorat_v8_head.py`: target maps and GIoU.
- `programs/benchmark_lorat_v8.py`: benchmark IoU summaries.

Source to learn:

- IoU tutorial: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
- MOTChallenge format: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt

Tiny exercise:

Make three boxes by hand:

- same box: IoU should be 1;
- no overlap: IoU should be 0;
- partial overlap: IoU should be between 0 and 1.

Then write a function that returns those values.

## 4. Object Detection, Segmentation, And Tracking

These are related but different.

Object detection:

```text
image -> boxes, class names, confidence scores
```

Example output:

```text
person, box=(100, 50, 40, 120), confidence=0.93
```

Segmentation:

```text
image -> pixel mask for an object
```

Tracking:

```text
video + initial object -> boxes across time
```

This project is not primarily a detector. It is a user-initialized tracker and labeler. The user can select unnamed objects. That matters because a detector usually asks "what known class is this?" while this project asks "where did this selected thing go?"

Later open-world discovery adds proposals:

```text
frame -> possible unlabeled object boxes -> user accepts -> new track
```

Project mapping:

- Weeks 1-3: tracking selected objects.
- Week 4: propose new unknown objects.
- Week 5: ask for corrections on uncertain frames.

Tiny exercise:

Create two dataclasses:

```python
@dataclass
class Proposal:
    frame: int
    bbox: tuple
    score: float
    source: str

@dataclass
class Track:
    track_id: int
    bbox: tuple
    state: str
```

Explain why a proposal should not become a track until the user accepts it.

## 5. What Single-Object Tracking Means

Single-object tracking, or SOT, starts with one object in one frame.

Input:

```text
first frame + initial box
```

Output:

```text
one box per later frame
```

The tracker usually stores a target representation:

- a template crop;
- a feature vector;
- patch tokens;
- or model memory.

Then each new frame asks:

```text
Where is the thing that looked like my initial target?
```

Common SOT ideas:

- Template: saved example of target appearance.
- Search region: area of the new frame where the target might be.
- Response map: score map saying "target likely here."
- Box regression: predicts the exact target box.
- Template update: refreshes memory when the target changes appearance.
- Drift: tracker starts following the wrong object.
- Occlusion: target is hidden for a while.

Why SOT is attractive for this project:

- The user gives the first box, exactly like SOT initialization.
- The object may have no class name.
- SOT is naturally target-specific.
- SOT can follow "this selected thing" instead of "a person class."

Why SOT alone is not enough:

- It usually assumes one target.
- It may not handle multiple similar objects crossing.
- It can drift if memory updates from bad frames.
- Running one full SOT model per object can be slow.
- It does not automatically solve global identity assignment.

Project files:

- `programs/bounding_box_v3_lorat.py`: LoRAT used more like one SOT tracker per selected object.
- `programs/bounding_box_v4_lorat_memory.py`: SOT memory slots and identity safety.
- `programs/bounding_box_v5_lorat_shared.py`: reusable SOT-style memory and state utilities.

Sources:

- SOT survey: https://arxiv.org/abs/2201.13066
- LoRAT paper: https://arxiv.org/abs/2403.05231
- Official LoRAT repo: https://github.com/LitingLin/LoRAT

Tiny exercise:

Draw the SOT loop:

```text
initial box -> save template -> search next frame -> choose best box -> maybe update template -> repeat
```

Then identify where that loop appears in V3 or V4.

## 6. What Multi-Object Tracking Means

Multi-object tracking, or MOT, tracks multiple objects and keeps their identities stable.

Input:

```text
video frames
```

Often in standard MOT:

```text
detector boxes per frame
```

Output:

```text
frame, track_id, box
```

MOT must solve two problems at once:

1. Localization: where are the objects?
2. Association: which box belongs to which identity?

The second problem is the one that makes MOT difficult.

Standard tracking-by-detection pipeline:

```text
detect boxes in current frame
predict existing track positions
compute track-candidate scores
assign candidates to tracks
update matched tracks
hold or delete unmatched tracks
spawn new tracks from unmatched detections
```

Common MOT signals:

- detector confidence;
- motion prediction;
- IoU overlap with predicted position;
- appearance/ReID similarity;
- assignment margin;
- occlusion state;
- track age.

Common MOT failures:

- identity switch: track ID jumps to another object;
- fragmentation: one real object gets many broken track IDs;
- false positive: tracker follows something that is not a real object;
- false negative: tracker misses a real object;
- drift: a track gradually moves away from the selected object.

Project files:

- `programs/benchmark_lorat_v8.py`: identity and quality observations.
- `programs/bounding_box_v5_lorat_shared.py`: track state, association, occlusion holding.
- `programs/bounding_box_v8_lorat_quality_batched.py`: current multi-object V8 control logic.

Sources:

- DanceTrack paper: https://arxiv.org/abs/2111.14690
- DanceTrack official repo: https://github.com/DanceTrack/DanceTrack
- TrackEval: https://github.com/JonathonLuiten/TrackEval
- HOTA metric: https://arxiv.org/abs/2009.07736

Tiny exercise:

Make a fake 3-frame MOT table:

```text
frame, id, x, y, w, h
1, 1, 10, 10, 20, 40
2, 1, 12, 10, 20, 40
3, 1, 15, 10, 20, 40
```

Then add a second object and create an identity switch on frame 3. Explain what went wrong.

## 7. SOT Versus MOT

| Question | SOT | MOT |
| --- | --- | --- |
| How many objects? | One selected object. | Many objects. |
| Initialization | Usually first-frame box. | Usually detections, sometimes user boxes. |
| Main question | Where did this target go? | Which boxes belong to which identities? |
| Strength | Target-specific and class-agnostic. | Handles many objects and global assignment. |
| Weakness | Drift and one-target assumption. | Association complexity and detector dependence. |
| Project relevance | User gives first box, object may be unnamed. | Tool must track many boxes simultaneously. |

The project lives between SOT and MOT:

```text
SOT-style user initialization + MOT-style multi-object identity management
```

That is why "multi-object extension of LoRAT" is not simply "run LoRAT N times." It means preserving LoRAT's target-conditioned strength while solving MOT's identity and scaling problems.

## 8. SOT-Based MOT Does Exist

SOT-based MOT is less common than tracking-by-detection, but it is real and relevant.

There are several ways to combine SOT and MOT.

### Pattern A: Run One SOT Tracker Per Object

Basic idea:

```text
object 1 -> SOT tracker 1
object 2 -> SOT tracker 2
object 3 -> SOT tracker 3
```

Pros:

- simple to understand;
- works naturally with user-drawn boxes;
- no class labels needed;
- each object gets its own target memory.

Cons:

- expensive when object count grows;
- each SOT tracker may drift independently;
- crossing objects can swap identities;
- no global assignment unless you add it.

Project mapping:

- V3 and V4 are closest to this pattern.
- V4 adds memory and identity arbitration to reduce drift.

### Pattern B: Detection Plus SOT Branch

SOTMOT, from "Improving Multiple Object Tracking with Single Object Tracking," is the key paper here.

Basic idea:

```text
detector finds objects
SOT-style branch helps discriminate each target
MOT association uses the target-specific information
```

This is not just running old SOT trackers separately. It adds an SOT branch to an MOT architecture so MOT can benefit from SOT-style target discrimination.

Why it matters:

- It proves the research direction is legitimate.
- It shows SOT can improve MOT association.
- It warns us that efficiency and target discrimination must be designed together.

Project mapping:

- `docs/sotmot_lorat_mot_notes.md`
- `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`

Source:

- SOTMOT paper PDF: https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf

Another local paper, `papers/SOT_For_MOT.pdf`, is even more direct as a beginner bridge. Its core idea is to stay within tracking-by-detection MOT but use SOT to reduce missed detections. In plain English:

```text
detector misses object in a frame
SOT prediction can still suggest where the object went
appearance/ReID model helps associate detections into long tracks
```

That makes it a good stepping stone between "run one SOT per target" and "design a learned SOT branch inside MOT."

### Pattern C: Shared Backbone Plus Per-Object Target Heads

This is the direction of V7/V8.

Basic idea:

```text
frame -> one shared feature map
object memories -> batched per-object scoring head
MOT identity logic -> assign, accept, hold, recover
```

Pros:

- keeps SOT's target-conditioned idea;
- avoids a full frame-backbone pass per object;
- can track unnamed user-selected objects;
- naturally supports multiple user boxes;
- gives a path toward ReID and active correction.

Cons:

- head must be trained carefully;
- identity logic is still needed;
- memory updates can still contaminate tracks;
- this is more researchy than off-the-shelf MOT.

Project mapping:

- V7 proves the shared-frame shape.
- V8 adds stronger quality logic and a trainable object-conditioned head.
- V9 should add stronger ReID training and recovery.

Files:

- `programs/bounding_box_v7_lorat_frame_shared.py`
- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`
- `docs/v8_code_walkthrough.md`
- `docs/v8_training_methods_research.md`

The local UMA paper, `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf`, is relevant here because it explicitly complains about the overhead of using separate SOT and affinity networks. UMA tries to learn one compact feature that supports both object motion and affinity matching. That is not our exact architecture, but the motivation is very close to V8:

```text
avoid repeated feature extraction
share representation
still support target motion and identity affinity
```

### Pattern D: Hybrid SOT With Re-Detection/ReID Recovery

Basic idea:

```text
SOT follows target locally
ReID searches for target after loss
MOT association prevents target stealing
human correction repairs uncertain frames
```

This is close to the final tool.

It is not pure SOT and not pure detector-MOT. It is an annotation-focused hybrid:

- user creates tracks;
- LoRAT-style tracking propagates boxes;
- ReID recovers lost identities;
- open-world proposals suggest new objects;
- active correction chooses frames for human review.

## 9. Why SOT-Based MOT Is Limited/Rare

SOT-based MOT is promising, but it has real difficulties.

### Reason 1: SOT Assumes A Single Target

Most SOT trackers are trained to distinguish:

```text
target versus local background
```

MOT needs:

```text
target A versus target B versus target C versus background
```

That is a harder identity problem.

### Reason 2: Independent SOT Trackers Can Steal Objects

If two similar objects cross, tracker A may start following object B. Without global assignment, neither tracker knows the global identity conflict.

### Reason 3: Computation Scales With Object Count

Running a full SOT model per object can be expensive:

```text
N objects -> N model passes
```

Week 2 exists because the project needs:

```text
N objects -> 1 shared frame pass + batched object scoring
```

### Reason 4: Memory Updates Can Poison The Track

If the tracker updates its template from a wrong box, future frames become worse. That is why V4/V5/V8 have conservative learning gates.

### Reason 5: Standard MOT Benchmarks Often Start From Detections

Many MOT systems assume detector boxes are available. This project assumes the user can select any unnamed object, so detector-first MOT does not directly solve the scope.

## 10. How Our Project Should Describe Its Approach

A careful description:

```text
We build a user-initialized, SOT-inspired MOT labeler around LoRAT. The runtime extends LoRAT from one target to multiple target memories by sharing the frame backbone and batching object-conditioned scoring heads, then adding MOT-style identity association, ReID recovery, open-world proposal intake, and active correction.
```

Avoid saying:

```text
We simply run LoRAT on multiple boxes.
```

That was closer to the early prototype, but V8 is different.

Also avoid saying:

```text
LoRAT is already a complete MOT tracker.
```

LoRAT is a SOT tracker. The MOT extension is our research system.

## 11. Concept Map For This Project

```text
Computer vision
  -> images and video frames
  -> object localization with boxes
  -> SOT: one selected object over time
  -> MOT: many identities over time
  -> ReID: appearance memory for identity recovery
  -> open-world discovery: propose unnamed objects
  -> active correction: ask humans only when uncertain
  -> benchmark: quality per unit human effort
```

V8 system map:

```text
frame
  -> LoRAT/DINOv2 shared feature map
  -> per-track memory/template slots
  -> batched object-conditioned head
  -> candidate boxes
  -> identity association
  -> accept / hold / lost / reacquired
  -> debug logs + MOT output + GUI display
```

## 12. Reading Path

### Beginner Computer Vision

1. OpenCV image/video/mouse tutorials:
   - https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
   - https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
   - https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
2. IoU tutorial:
   - https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
3. MOTChallenge format:
   - https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-format.txt

### Tracking Basics

1. `papers/CV_MOT.pdf`
   This is a broad computer-vision/MOT overview. Use it as a beginner warmup, not as the main research citation.
2. `papers/SingleObjectTrackingASurveyofMethodsDatasetsandEvaluation Metrics.pdf`
   Use this for SOT vocabulary: target, template, search, drift, benchmarks.
3. `papers/deep_sort.pdf`
   Use this for appearance/ReID association basics.
4. `papers/oc_sort_observation_centric_sort.pdf`
   Use this for the motion-heavy side of MOT.
5. `papers/dancetrack_multi_object_tracking_uniform_appearance_diverse_motion.pdf`
   Use this to understand why similar appearance makes association hard.
6. `papers/hota_metric.pdf`
   Use this to understand why MOT quality needs detection and association metrics.

### SOT-Based MOT And LoRAT

1. `papers/SOT_For_MOT.pdf`
   Start here for the plainest "SOT can help MOT" bridge.
2. `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`
   Read next for the more formal SOTMOT architecture.
3. `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf`
   Read for the shared motion/affinity motivation that resembles the V8 shared-feature argument.
4. `papers/DCFST.pdf`
   Read for discriminative feature learning in SOT; useful background for target-specific embeddings.
5. `docs/sotmot_lorat_mot_notes.md`
   Read our local interpretation of SOTMOT for this project.
6. `papers/lorat_tracking_meets_lora.pdf`
   Read for LoRAT itself.
7. `docs/v8_code_walkthrough.md`
   Read for how our code turns LoRAT into a shared-frame MOT branch.
8. `docs/v8_training_methods_research.md`
   Read for how to train the V8 head.

### ReID, Association, And Recovery

1. `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
   Dense contrastive identity learning; important for V8/V9 training.
2. `papers/fairmot_detection_reid_mot.pdf`
   Joint detection and ReID baseline.
3. `papers/strongsort_make_deepsort_great_again.pdf`
   Practical modern ReID association.
4. `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf`
   Strong MOT baseline with robust association.
5. `papers/boosttrack_similarity_confidence_mot.pdf`
   Confidence-aware association baseline.
6. `papers/boosttrack_plus_plus_tracklet_information_mot.pdf`
   Tracklet-history association baseline.

### Open-World And Active Correction Later

1. `papers/grounding_dino.pdf`
2. `papers/segment_anything.pdf`
3. `papers/sam2_segment_anything_images_videos.pdf`
4. `papers/video_annotation_tracking_active_learning.pdf`
5. `papers/efficient_video_annotation_visual_interpolation_frame_selection.pdf`
6. `papers/vatic_efficient_crowdsourced_video_annotation.pdf`

These matter after Week 3. They should not distract from learning SOT/MOT/ReID first.

## 13. Paper-To-Code Map

| Local paper/doc | Main idea to learn | Where it connects in code |
| --- | --- | --- |
| `papers/CV_MOT.pdf` | Broad CV/MOT overview. | Use before reading tracker files. |
| `papers/SingleObjectTrackingASurveyofMethodsDatasetsandEvaluation Metrics.pdf` | SOT vocabulary and failure modes. | V3/V4 LoRAT per-object tracker behavior. |
| `papers/SOT_For_MOT.pdf` | SOT can reduce MOT false negatives under tracking-by-detection. | V4/V5 memory recovery and held tracks. |
| `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf` | SOT branch can improve MOT discrimination. | `docs/sotmot_lorat_mot_notes.md`, V8 design argument. |
| `papers/A Unified Object Motion and Affinity Model for Online Multi-Object Tracking.pdf` | Unify motion and affinity features for efficiency. | V8 shared-frame encoder plus feature identity. |
| `papers/DCFST.pdf` | Learn discriminative SOT features for class-agnostic targets. | Future ReID/head embedding training ideas. |
| `papers/lorat_tracking_meets_lora.pdf` | LoRAT SOT backbone and training. | V3/V4 LoRAT runtime, V8 LoRAT backbone. |
| `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf` | Contrastive ReID training. | `contrastive_reid_loss()` and Week 3/V9 ReID plan. |
| `papers/dancetrack_multi_object_tracking_uniform_appearance_diverse_motion.pdf` | Similar appearance makes MOT hard. | ReID/motion ablation benchmarks. |
| `papers/hota_metric.pdf` | Evaluate detection and association together. | TrackEval and benchmark summary outputs. |

## 14. Coding Exercises In Project Order

### Exercise A: Image And Box

Write a script that:

1. loads one frame;
2. draws one hard-coded bbox;
3. prints bbox area and center;
4. saves the annotated frame.

### Exercise B: User Box

Write or modify a script so:

1. user drags a box with the mouse;
2. the box is saved as `xywh`;
3. the crop inside the box is saved as an image.

### Exercise C: One-Object Tracker Skeleton

Without LoRAT, make a fake SOT tracker:

1. user draws a box;
2. each frame moves the box 2 pixels right;
3. output MOT-format rows.

The point is to learn the data flow, not tracking quality.

### Exercise D: Two-Object Assignment

Make fake tracks and fake candidates:

1. compute center-distance scores;
2. run Hungarian assignment;
3. update matched tracks;
4. hold unmatched tracks.

### Exercise E: Appearance ReID Toy

Make fake appearance vectors:

1. one memory vector per track;
2. one vector per candidate;
3. cosine similarity matrix;
4. combine with motion score.

This is the toy version of V8 identity scoring.

### Exercise F: SOT-Based MOT Sketch

Build a tiny architecture diagram from code comments:

```text
TrackState
  -> bbox
  -> velocity
  -> confidence
  -> memory vectors
  -> state
```

Then identify matching fields/functions in `programs/bounding_box_v5_lorat_shared.py`.

### Exercise G: V8 Trace

Trace one frame through V8:

1. `main()`
2. `update()`
3. `_encode_frame()`
4. `_score_and_update_tracks()`
5. `_candidates_from_head_output()`
6. identity resolver
7. `_accept_candidate()` or `_hold_track()`
8. debug/MOT output

Write one sentence for what each step does.

## 15. Minimum Vocabulary For This Project

Computer vision:

- image
- pixel
- frame
- channel
- crop
- feature
- embedding
- proposal
- detection
- segmentation

Box geometry:

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

Project-specific:

- LoRAT
- shared backbone
- object-conditioned head
- memory slot
- ReID bank
- active correction
- manual reanchor
- human-cost event

## 16. The Key Mental Shift

Generic computer vision asks:

```text
What is in this image?
```

SOT asks:

```text
Where did this selected object go?
```

MOT asks:

```text
Where did all objects go, and which identity is each one?
```

This project asks:

```text
How can a human select any object once, even if it has no name, and get high-quality boxes across the video with the fewest corrections?
```

That is why SOT-based MOT is the right conceptual neighborhood, even if it is a less common path than standard detector-based MOT.
