# Benchmark Protocol

This document defines the benchmark language for the LoRAT multi-object labeling project. Its purpose is to make the results usable in a paper: each benchmark should say what is measured, why it is measured, how it is computed, and which established computer-vision metrics it relates to.

## Benchmark Philosophy

The system is not only a conventional MOT tracker. It is a user-initialized video labeling tool built from a LoRAT-style SOT tracker, extended to multiple objects, ReID recovery, open-world proposal intake, and active correction. Because of that, we need two layers of benchmarks:

1. Standard external metrics.
   These are the numbers used by existing SOT/MOT/open-world tracking papers and allow comparison against other systems.

2. Tool-specific diagnostics.
   These explain human effort, first-box initialization behavior, small-object failure, and internal architectural claims such as shared-backbone batching.

The paper should clearly label which results are externally comparable and which are system diagnostics.

## Core Geometry

### Intersection over Union

IoU is the core localization measure used throughout the benchmarks.

For predicted box `P` and ground-truth box `G`:

```text
IoU(P, G) = area(P intersection G) / area(P union G)
```

Interpretation:

- `1.0`: perfect overlap.
- `0.5`: common detection/tracking match threshold.
- `0.0`: no overlap.

Why we use it:

- It is standard in object detection, SOT, MOT, and proposal recall.
- MOTChallenge-style evaluation commonly treats predictions with IoU at or above a threshold as valid matches.
- OTB-style SOT success plots are based on overlap thresholds across frames.

Project use:

- Per-frame tracking quality.
- Small-object reliability.
- Proposal recall.
- Track loss and identity switch decisions in benchmark scripts.

## Week 1 Benchmarks

### 1. Time to Produce Bounding Boxes vs Object Count

Question:

How long does the tracker take to output boxes for one object, two objects, and up to `N` objects?

What we measure:

- Total frame update time.
- Time per frame.
- Time per object/box.
- FPS.
- Optional internal timing buckets.

How we compute it:

For each model/config and object count `N`:

```text
time_per_box_ms = frame_update_time_ms / active_object_count
fps = processed_frames / elapsed_seconds
```

Why it matters:

- The labeling tool must scale beyond one selected object.
- It directly tests whether the multi-object extension is practical.

External comparison:

- FPS is standard in SOT and MOT papers, including VOT/SOT tracker reports.
- Time-per-object is more project-specific but useful because our system starts from SOT and scales to MOT.

Paper phrasing:

Report both FPS and ms/box. FPS is externally familiar; ms/box explains scaling behavior for a user-initialized multi-object tool.

### 2. Small-Object Reliability by Pixel Area

Question:

How small can an object become before the tracker becomes unreliable?

What we measure:

- GT object area in pixels.
- Mean IoU for predictions in that area bin.
- IoU success rate, usually `IoU >= 0.5`.
- Minimum area bin that satisfies the reliability rule.

Current reliability rule:

```text
Reliable if:
  mean IoU >= 0.50
  and IoU@0.5 >= 0.50
  and sample count >= 10
```

Definitions:

```text
IoU@0.5 = number of evaluated frames with IoU >= 0.5 / total evaluated frames
```

Why `0.5`:

- IoU `0.5` is a common detection/tracking matching threshold.
- It is strict enough to indicate meaningful box overlap, but not so strict that small annotation noise dominates.

Why minimum 10 samples:

- A single tiny object frame can be misleading.
- Requiring at least 10 frame/object samples prevents declaring a size bin reliable from too little evidence.

10-frame method:

For temporal reliability, sample every 10 frames or evaluate windows spaced by 10 frames. Compare the tracker prediction at the sampled frame against the dataset ground truth at that frame, not merely against the first frame. The first frame initializes the object identity; ground truth determines whether the tracker still follows the correct object later.

External comparison:

- IoU success and overlap-style success rates are standard in SOT benchmarks such as OTB.
- Area-binned analysis is a project-specific stress test, but it uses standard IoU overlap.

### 3. Model Size Comparison

Question:

How do LoRAT model sizes trade accuracy and runtime?

Models:

- `B-224`
- `L-224`
- `g-224`

What we measure:

- FPS / frame time.
- Mean IoU.
- IoU@0.5.
- Smallest reliable pixel area.
- GPU memory if available.

Why it matters:

- Larger ViT backbones may improve robustness but cost speed/memory.
- The project needs a practical default model, not just the highest-capacity model.

External comparison:

- Accuracy/speed tradeoff tables are common in tracker papers.
- Model-size comparison is especially important because LoRAT reports multiple ViT backbone sizes.

## Week 2 Benchmarks

### 4. Shared-Backbone Proof

Question:

Did the refactor actually make all objects share one Vision Transformer frame pass?

What we measure:

- Shared frame backbone calls per frame.
- Object head batch count per frame.
- Object head batch size.
- Number of active objects.

Expected Week 2 property:

```text
shared_frame_backbone_calls_per_frame ~= 1
object_head_batch_size ~= active_object_count
```

Meaning:

- Object count should increase the batched head workload.
- Object count should not multiply the ViT frame backbone pass.

External comparison:

- This is mostly an architectural proof, not a standard leaderboard metric.
- It supports the system claim required by the deliverable.

### 5. FPS vs Object Count Before/After Refactor

Question:

Did shared-backbone batching improve throughput scaling?

What we measure:

- FPS at `N=1...Nmax`.
- ms/frame.
- ms/object.
- GPU memory.

How we compare:

- Pre-refactor: V6 or earlier multi-object LoRAT wrapper.
- Post-refactor: V8 shared-frame/batched-head tracker.

External comparison:

- FPS is standard.
- Scaling curve vs object count is project-specific but directly supports the week 2 deliverable.

### 6. GPU Memory vs Object Count

Question:

How many objects can run while sustaining at least 25 FPS?

What we measure:

- GPU allocated/reserved memory.
- Peak memory.
- FPS.
- Max `N` where FPS >= 25.

Why 25 FPS:

- It is the deliverable threshold.
- It approximates interactive video-rate tracking.

External comparison:

- GPU memory reporting is common in engineering papers and ablation studies.
- The 25 FPS target is project-specific.

## Week 3 Benchmarks

### 7. Identity Switch Count

Question:

Does ReID reduce cases where a track changes identity?

Standard concept:

An identity switch occurs when a predicted track ID that was associated with one ground-truth object becomes associated with a different ground-truth object.

What we measure:

- ID switches per video.
- ID switches per 1000 evaluated samples.
- With ReID vs without ReID.

External comparison:

- ID switches are standard in MOTChallenge/CLEAR MOT-style evaluation.
- IDF1 and HOTA AssA are stronger external metrics for identity consistency.

### 8. Track-Loss Rate

Question:

How often does the tracker lose the selected object?

What we measure:

- Frames where the target is visible in GT but the tracker is lost or below match threshold.
- Loss rate:

```text
track_loss_rate = lost_visible_samples / visible_gt_samples
```

External comparison:

- Related to false negatives in MOT metrics.
- Related to failure/robustness in VOT-style SOT evaluation.

### 9. Occlusion Survival

Question:

How many occluded frames can the tracker tolerate before identity is lost?

What we measure:

- Controlled or observed occlusion duration.
- Whether the same track reattaches after reappearance.
- Survival duration before loss/switch.

External comparison:

- VOT long-term tracking evaluates robustness/re-detection behavior.
- HOTA/IDF1 reflect association quality after occlusion, but do not directly express “frames tolerated,” so this remains a tool-specific diagnostic.

### 10. Manual Reanchor Cost

Question:

How much human correction does the system require?

What we measure:

- Manual reanchor events.
- Frame number.
- Track ID.
- Old box/new box.
- Time spent, if measured.

External comparison:

- Human-effort cost is not a standard MOT leaderboard metric.
- It is essential for this labeling tool because final quality depends on human correction workload.

## Week 4 Benchmarks

### 11. Proposal Recall

Question:

Do class-agnostic proposals cover unlabeled/unknown objects before the user draws them?

What we measure:

For each GT object in a frame:

```text
matched = max IoU(proposal, gt_object) >= threshold
proposal_recall = matched_gt_objects / total_gt_objects
```

Recommended thresholds:

- Primary: IoU >= 0.5.
- Optional: recall at IoU >= 0.3 for loose proposal coverage, especially for rough class-agnostic proposal methods.

Proposal budgets:

Report recall at several proposal counts:

```text
Recall@25
Recall@100
Recall@500
Recall@1000
```

Why budgets matter:

- A queue with 1000 proposals may cover objects but is not usable by a human.
- A queue with 25 proposals is more usable but may miss objects.

External comparison:

- Proposal recall is standard in object proposal and open-world detection work.
- TAO-OW/open-world tracking emphasizes class-agnostic discovery/recall before classification.

### 12. Manual Bounding-Box Effort Saved

Question:

How many manual boxes can the user avoid drawing if proposals are accepted?

Oracle calculation:

```text
manual_boxes_baseline = number_of_gt_objects
manual_boxes_with_proposals = number_of_gt_objects - matched_gt_objects
manual_boxes_saved = matched_gt_objects
manual_effort_saved_rate = matched_gt_objects / number_of_gt_objects
```

Meaning:

- This is not yet a live user study.
- It is an upper-bound/oracle estimate of proposal usefulness.

Next version:

Run a simulated proposal queue:

- only top `K` proposals shown,
- user accepts proposals overlapping GT,
- rejected proposals count as review burden,
- accepted proposals spawn tracks.

## Standard Metrics to Add for Paper Comparability

### TrackEval / MOT Metrics

We should export V8/V9 predictions in MOTChallenge format and run TrackEval or py-motmetrics for:

- HOTA
- DetA
- AssA
- LocA
- MOTA
- MOTP
- IDF1
- ID switches
- FP/FN

Why:

- HOTA balances localization, detection, and association.
- MOTA is historically common but can overemphasize detection errors.
- IDF1 focuses identity consistency.

Recommended use in paper:

- Main MOT table: HOTA, DetA, AssA, IDF1, MOTA, IDs.
- Diagnostic tables: FPS, memory, small-object area, manual effort.

### SOT Metrics

For user-initialized single target behavior, add:

- success AUC over IoU thresholds,
- precision plot / center error,
- normalized precision if image size varies,
- failure count or robustness.

Why:

- The project extends a SOT tracker, so SOT-style evaluation tells whether the selected object remains trackable.

Recommended use in paper:

- Report SOT-style success/precision for `N=1`.
- Report MOT metrics for `N>1`.

## Dataset Roles

### DanceTrack

Use for:

- crowded multi-person tracking,
- identity switches,
- throughput scaling,
- visual jitter under similar-looking targets.

Risk:

- Person-only distribution can overstate general-object ability.

### MOT17

Use for:

- standard pedestrian MOT comparison,
- MOTChallenge-style metrics.

Risk:

- Also person-focused.

### TAO / TAO-OW

Use for:

- general object tracking,
- open-world proposal recall,
- non-person selected-object generalization.

Risk:

- Sparse/federated annotations mean not every visible object is labeled.
- Must be careful not to count unlabeled objects as false positives too aggressively.

## Reporting Template

Each benchmark section in the paper should include:

1. Purpose.
2. Dataset and split.
3. Initialization protocol.
4. Ground-truth matching rule.
5. Metrics.
6. Thresholds.
7. Sample count.
8. Hardware.
9. Failure cases.

Example:

```text
We evaluate small-object reliability by binning GT object instances by pixel area.
For each visible GT object assigned to a tracker, we compute IoU between the
tracker output and GT box every 10 frames. A bin is reliable if it has at least
10 samples, mean IoU >= 0.50, and IoU@0.5 >= 0.50.
```

## Sources to Cite

- Wu, Lim, Yang, "Online Object Tracking: A Benchmark," CVPR 2013. Defines OTB-style precision and success plots.
- Kristan et al., VOT challenge reports. Defines VOT-style accuracy/robustness and EAO.
- Bernardin and Stiefelhagen, CLEAR MOT. Defines MOTP/MOTA family.
- Ristani et al., IDF1 / identity metrics.
- Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking," IJCV 2020. Defines HOTA/DetA/AssA.
- Dave et al., "TAO: A Large-Scale Benchmark for Tracking Any Object," ECCV 2020. Defines TAO general-object tracking benchmark.
- Liu et al., "Opening Up Open World Tracking," CVPR 2022. Defines TAO-OW/open-world tracking framing.
- TrackEval repository. Reference implementation for HOTA and common MOT metrics.

