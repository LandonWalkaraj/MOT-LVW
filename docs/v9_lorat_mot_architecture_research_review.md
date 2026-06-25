# V9 LoRaT-MOT Architecture Research Review

Date: 2026-06-23

This note collects what we have built so far, what the benchmark evidence says, and what related tracking/detection work suggests before we make the next architecture move. The goal is to avoid another round of isolated patches that improve one symptom while moving us farther from the original project.

## Project Goal

The tool is not meant to be a normal pedestrian tracker. The intended workflow is:

1. A user draws one bounding box around any object or object part.
2. The system tracks that selected target through the video.
3. The target may be unnamed and may not belong to a fixed training class.
4. Multiple selected targets must be tracked at once.
5. The system should recover after loss/occlusion with appearance-based ReID.
6. Human correction should happen only when uncertainty is high, and those corrections should be measured.

This means the core identity is the selected box, not a class label like `person`.

## What We Have Built

### Week 1 / V4-V6

V4-V6 kept more of LoRaT's original single-object tracking behavior.

- Each selected object was effectively handled by a LoRaT-like target/search process.
- The tracker wrapped LoRaT with motion, memory, identity, and quality gates.
- This was closer to the selected-target behavior we wanted.
- Quality was much better than the early V7/V8 shared-frame head path.
- The cost scaled poorly because too much LoRaT work still happened per object.

Useful local references:

- `programs/bounding_box_v4_lorat_memory.py`
- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v6_lorat_quality.py`
- `programs/mot_common.py`
- `docs/benchmark_protocol.md`
- `docs/paper_evidence_inventory.md`

### Week 2 / V7-V8

The Week 2 requirement forced the deeper refactor:

> all objects share a single Vision Transformer backbone forward pass, with per-object LoRA heads batched on the GPU

V8 is the current branch that implements this structure:

- one shared LoRaT/DINOv2/ViT frame encoder pass
- frozen shared frame feature map
- per-object template/memory vectors
- batched object-conditioned low-rank heads
- DINOv2 crop embeddings for ReID/recovery
- benchmark proof counters showing one shared frame-backbone pass and N batched head items

Useful local references:

- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`
- `programs/benchmark_lorat_v8.py`
- `docs/v8_training_methods_research.md`
- `outputs/PRESENTATION MATERIAL/v8_training_repair_presentation_notes_20260623.md`

The good news: V8 satisfies the structure and is fast. Recent A100 runs stayed above 25 FPS through N=5.

The bad news: V8 did not preserve LoRaT's selected-target localization quality. Recent videos and diagnostics show:

- N=1 can begin plausibly, then jump during crowding.
- N=5 compounds identity ownership conflicts.
- ReID keeps tracks alive, but sometimes on the wrong target.
- Small selected regions, such as heads/faces, are especially weak.
- The head often predicts body-scale boxes after a small user selection.

Useful local evidence:

- `outputs/benchmark_reviews/v8_week3_largest_smallest_20260622_140749/analysis_summary.md`
- `outputs/analysis/v8_week3_rerun3_review.md`

## The Core Diagnosis

V8 solved the shared-compute requirement by moving from target-centered LoRaT tracking to a full-frame shared feature map plus a new dense object-conditioned head. That created a coordinate-system mismatch.

LoRaT's original strength comes from template/search tracking:

- The selected object is represented by a template crop.
- The next frame is searched in a normalized local search crop.
- The object is effectively resized into a useful token/pixel scale.
- The head predicts the target inside that search region.

V8 instead asks the new head to localize arbitrary selected targets directly on one full-frame feature map. This is fast and meets the shared-backbone proof, but it loses the local normalization that made LoRaT robust across scale.

This explains the repeated pattern:

- Full-body boxes can sometimes work.
- Head/face/small boxes collapse because they occupy too few frame-level tokens.
- Longer training helps some validation IoU, but does not fully fix runtime identity jumps.
- ReID recognizes appearance but cannot repair bad localization if the candidate box is already wrong.
- Conflict resolution helps N>1, but it cannot fully compensate for weak target-local geometry.

## What Related Work Says

### LoRaT And Siamese SOT

LoRaT and classic Siamese trackers are trained around target-conditioned tracking, not class detection. LoRaT is a DINOv2/ViT tracker with LoRA adaptation, while classic SiamFC established the template/search matching pattern for generic object tracking.

Relevant takeaways:

- The user's first box should remain the tracking query.
- Training should teach "same selected target" versus distractors.
- Localization should happen in a normalized search coordinate system.
- Template/search crop jitter is not a side detail. It is part of why scale works.

Sources:

- [LoRaT: Tracking Meets LoRA](https://arxiv.org/abs/2403.05231)
- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549)
- Local LoRaT config/training notes: `docs/v8_training_methods_research.md`

### Shared Backbone With Local Precision

Detection systems solved a closely related problem years ago: how to compute a shared image feature map, then make local predictions for many regions. Faster R-CNN shares convolutional features and evaluates region proposals; Mask R-CNN adds RoIAlign so region features are sampled precisely from the shared feature map; FPN adds multi-scale feature maps so small objects are not erased by low resolution.

Relevant takeaways:

- Shared full-frame compute and local target precision are not mutually exclusive.
- Per-object local features can be extracted from shared frame features.
- RoIAlign-like feature extraction is a better fit than per-object full ViT passes if we want to preserve the Week 2 constraint.
- Small objects need higher-resolution or multi-scale features, not just more epochs.

Sources:

- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [Mask R-CNN / RoIAlign](https://arxiv.org/abs/1703.06870)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

### MOT Association And ReID

MOT systems often separate detection/localization from identity association. DeepSORT uses appearance embeddings for association, QDTrack trains dense similarity for tracking, and ByteTrack/BoT-SORT/OC-SORT show how much association logic matters when detections are ambiguous.

Relevant takeaways:

- ReID should decide identity ownership, not replace localization.
- Association must use appearance, motion, margin, and conflict resolution.
- Hard negatives from nearby similar objects are crucial for crowded scenes.
- A tracker can have plausible boxes but bad identity, which is exactly what V8 videos show.

Sources:

- [DeepSORT](https://arxiv.org/abs/1703.07402)
- [QDTrack](https://arxiv.org/abs/2006.06664)
- [ByteTrack](https://arxiv.org/abs/2110.06864)
- [OC-SORT](https://arxiv.org/abs/2203.14360)
- [BoT-SORT](https://arxiv.org/abs/2206.14651)

### Query-Based MOT

TrackFormer and MOTR carry track queries through time. This is conceptually close to our per-object memories, but their training directly teaches persistent object queries under temporal changes.

Relevant takeaways:

- V8's memory slots should be trained as persistent target queries.
- Training needs sequence windows, not only independent frame pairs.
- Query dropout, stale templates, missing targets, and false memories should be part of training.

Sources:

- [TrackFormer](https://arxiv.org/abs/2101.02702)
- [MOTR](https://arxiv.org/abs/2105.03247)

### Small Object Tracking And Detection

Small-object methods preserve detail through higher input resolution, multi-scale features, or tiled inference. SAHI is one practical example: slice the image so small objects occupy more pixels relative to the model input.

Relevant takeaways:

- Small object failure is expected if a target becomes too few tokens on a full-frame feature map.
- Synthetic small-target loss weighting helps, but the feature representation also needs enough spatial detail.
- A local search/RoI feature grid is the most direct way to recover LoRaT-like small-target behavior while staying close to Week 2.

Sources:

- [SAHI: Slicing Aided Hyper Inference](https://arxiv.org/abs/2202.06934)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

### Open-World / Open-Vocabulary Tracking

TAO and Video OWL-ViT point toward the later project goal: tracking arbitrary object categories and open-world proposals. They are relevant for Week 4 and beyond, but they do not replace selected-target localization.

Relevant takeaways:

- TAO-style data is useful for object diversity.
- Open-vocabulary/objectness branches should propose candidates.
- The selected-target tracker still needs to localize the exact user-selected instance/part.

Sources:

- [TAO: Tracking Every Thing in the Wild](https://arxiv.org/abs/2005.10356)
- [OWL-ViT](https://arxiv.org/abs/2205.06230)
- [Video OWL-ViT](https://arxiv.org/abs/2308.11093)

## Why The Current V8 Path Feels Circular

The V8 patches we have been adding are reasonable in isolation:

- more small-target training samples
- small-target scale lock
- template rescue
- assignment conflict resolution
- stricter ReID gates
- longer training
- TAO/YFCC data
- hard negatives and ReID losses

But these still operate on top of the same basic full-frame head problem. If the head never sees the selected target at a preserved useful scale, then each repair only handles one visible symptom:

- scale lock prevents expansion but cannot create precise localization
- ReID finds likely identity but cannot define the correct box alone
- conflict resolution prevents two tracks from stealing the same candidate but cannot guarantee either candidate is right
- longer training may improve validation IoU but may keep learning a body-scale prior
- TAO adds diversity but does not fix the coordinate mismatch

The non-whackamole correction is to restore LoRaT's local target/search geometry inside the shared-backbone framework.

## Proposed Direction: V9 LoRaT-Preserving Shared Backbone

V9 should not abandon the Week 2 requirement. It should satisfy it in a way that keeps LoRaT's selected-target behavior.

### Runtime Architecture

1. Encode the frame once with the shared LoRaT/DINOv2/ViT backbone.
2. Keep immutable first-frame template memory for each selected target.
3. Keep recent high-confidence memory separately from the immutable anchor.
4. For each active track, define a local search window from previous box, velocity, and uncertainty.
5. Extract a fixed-size local feature grid for each search window from the shared frame features.
   - Use RoIAlign-style sampling or token-grid interpolation.
   - Batch all tracks' RoI/search grids together.
6. Feed each local grid plus its template/memory embedding to a batched object-conditioned head.
7. Predict objectness and box offsets in local search-window coordinates.
8. Map the local prediction back to full-frame coordinates.
9. Use ReID only to arbitrate identity/recovery, not as the primary localization mechanism.
10. Apply multi-track conflict resolution after all candidate scores are available.

This keeps:

- one shared ViT frame pass
- batched per-object heads
- selected-target conditioning
- LoRaT-like search localization
- better small-object scale handling

### Training Architecture

The training data and runtime should use the same coordinate system.

For each sample:

1. Choose a selected target region from a frame.
   - full object
   - head/face/part crop
   - small object crop
   - TAO/YFCC/open-world object crop
2. Build a template from an earlier frame or the first frame.
3. Build a local search window in the current frame.
4. Encode the full frame once.
5. Extract the local search feature grid from shared features.
6. Train the head to predict the selected target inside the local search grid.
7. Include same-frame and same-sequence distractors as hard negatives.
8. Train ReID/association separately with same-ID positives and nearby different-ID negatives.

Losses:

- LoRaT-style dense objectness over the local search grid
- GIoU / L1 / ltrb box regression in local coordinates
- scale/center losses for small selected targets
- contrastive ReID loss
- assignment/ranking loss among candidate tracks
- optional no-target/lost-state supervision for occlusion windows

### Why This Better Fits The Original Goal

If the user selects a dancer's head, the local search window makes the head occupy enough feature resolution to localize. The system still shares the expensive ViT pass, but the per-object head sees a normalized target-search problem instead of trying to identify a tiny object directly from a full-frame feature map.

That is closer to:

- LoRaT's original selected-target premise
- Siamese SOT training
- RoI detector shared-backbone practice
- MOT association practice
- the project's open-world user-box workflow

## What To Keep From V8

Keep:

- shared frame encoder machinery
- proof counters for shared calls and batched head items
- `mot_common.py`
- DINOv2 crop ReID feature bank
- lost/uncertain/manual re-anchor states and cost logging
- conflict-resolution diagnostics
- benchmark summaries and video output
- training diagnostics and checkpoint metadata
- TAO/YFCC adapter work

Modify or replace:

- full-frame dense V8 head as the main localization path
- training that labels targets on the full-frame score map
- small-target scale lock as a primary solution
- ReID recovery acting before local geometry is solid

Keep as fallback or diagnostics:

- V8 full-frame candidate generation can remain as a coarse proposal source.
- Template matching can remain as a rescue source.
- Small-target scale lock can remain as a safety rail.

## Immediate Next Steps

1. Write a V9 design stub before more training.
   - The important choice is RoI/token-grid local search from shared frame features.

2. Implement the local feature extractor.
   - Input: shared frame features plus search-window boxes.
   - Output: fixed-size batched search grids.
   - First version can use `torch.nn.functional.grid_sample` on the shared feature map.

3. Implement a V9 local search head.
   - Template/memory conditioned.
   - Batched across tracks.
   - Predicts local score map and l/t/r/b offsets.

4. Add an overfit test before another 48h run.
   - One sequence slice.
   - One full-body target and one small/head target.
   - Must overfit to good IoU in local coordinates.

5. Rework training data generation around local search windows.
   - The target should be supervised inside the search grid, not full-frame.

6. Benchmark against V8, not instead of it.
   - V8 remains the Week 2 shared-backbone branch.
   - V9 becomes the LoRaT-preserving shared-backbone correction.

7. Present the current results honestly.
   - Week 2 throughput/shared-backbone success.
   - Week 3 ReID/recovery implemented with mixed results.
   - Current blocker: full-frame head lost LoRaT's scale-normalized local search.
   - Next solution: shared backbone plus batched local-search RoI heads.

## Paper-Friendly Benchmark Additions

The current benchmark suite is useful, but the paper should add standard tracking metrics and clearer protocol language.

Add:

- HOTA
- DetA / AssA
- IDF1
- MOTA
- TrackEval-compatible exports
- forced small-area protocol
- manual small-box protocol
- ReID on/off ablation
- occlusion survival with natural and artificial gaps
- human correction cost: manual reanchors per minute and per target

The existing internal metrics should stay:

- FPS
- ms/box
- GPU memory
- one-backbone-call proof
- batched-head-item proof
- IoU and IoU@0.5
- identity switches
- track-loss rate
- correct-object rate

## Decision Point

The key decision is not "train longer or tune more." It is:

> Do we keep forcing selected-target tracking into a full-frame dense head, or do we restore LoRaT's local search formulation while keeping the Week 2 shared backbone?

The evidence points to the second option.

The next plan should be V9: shared frame encoder plus batched RoI/local-search object heads. This is the path that works with LoRaT instead of fighting it.
