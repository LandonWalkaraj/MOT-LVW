# V8 Code Walkthrough

This note explains `programs/bounding_box_v8_lorat_quality_batched.py` as a study guide.
It focuses on what V8 does, why it still imports V5, and how to read the file without getting lost.

## Short Answer: Why V8 Imports V5

V8 is standalone as a tracker architecture, but it reuses V5 as a shared utility module.

V8 does not subclass the V5 tracker and does not call the V5 per-object LoRaT evaluator in its frame update path. The V8-specific tracker is `V8QualityBatchedLoRATTracker`, the V8-specific frame encoder is `SharedFrameLoRATEncoder`, and the V8-specific head is `BatchedObjectConditionedHead`.

V8 uses V5 for:

- Shared data structures: `BBox`, `TrackState`, `LoRATSlotOutput`, `LoRATMemorySlot`, `IdentityScore`, `IdentityAssignment`, `RuntimeStatus`, `CropInformation`, and `FrameSource`.
- Shared defaults: LoRaT memory size, identity thresholds, occlusion thresholds, shrink guard thresholds, path recovery thresholds, and model weight lookup.
- Geometry helpers: bbox area, center, diagonal, IoU, clamping, deltas, motion affinity, path affinity, and overlap checks.
- State helpers: Kalman prediction/reference, `BBoxKalmanFilter`, trajectory recording, and state-token formatting.
- GUI and output helpers: frame source opening, box selection UI, video writer, drawing tracks, MOT output rows, and debug CSV writing.
- CSV and status formatting: `csv_float`, `csv_text`, `csv_bbox`, `bytes_to_mb`, output path helpers.

So the split is:

- V5 provides stable building blocks.
- V8 owns the tracking algorithm.

## High-Level V8 Flow

Runtime flow:

1. `main()` parses CLI arguments and opens a video/image source.
2. The user provides one or more initial boxes, or boxes are passed by CLI.
3. `create_backend()` builds a `V8QualityBatchedLoRATTracker`.
4. The tracker loads LoRaT internals once through `_load_lorat_shared_backbone()`.
5. `initialize()` creates one `TrackState` per selected box and caches first-frame feature memory.
6. For each video frame, `update()` runs one shared frame encoder pass.
7. `_score_and_update_tracks()` runs the batched object-conditioned head for all active tracks.
8. Candidate boxes are decoded, reranked, optionally rescued by template matching, and identity-resolved.
9. Each track is either accepted with `_accept_candidate()` or held with `_hold_track()`.
10. The GUI draws tracks and writes optional videos/debug/MOT outputs.

## Line-Range Walkthrough

### Lines 1-60: Imports, aliases, constants

The file imports ordinary Python modules, OpenCV, NumPy, and then imports V5 as:

```python
import bounding_box_v5_lorat_shared as v5
```

`BBox = v5.BBox` means V8 uses the same bounding-box type convention as V5:

```text
(x, y, width, height)
```

The `DEFAULT_V8_*` constants tune V8-specific behavior such as recovery heads, template matching, memory refresh gates, and window penalty. `V8_PROFILE_BUCKETS` names the timing buckets used by V8 benchmarks.

### Lines 62-100: Small data containers

These dataclasses carry structured values between parts of the tracker:

- `SharedFrameEncoding`: a feature map plus timing.
- `V8TemplateMemorySlot`: one per-object memory entry. It stores a vector and optional patch tokens.
- `BatchedHeadOutput`: score maps and box deltas from the V8 head.
- `V8HeadCandidate`: one decoded box candidate from a score map.
- `V8HeadCandidateInfo`: best candidate, margin, ROI token count, and top candidates.

This is a good coding pattern: use dataclasses when a function needs to return several related values.

### Lines 103-119: Debug/proof headers

`WEEK2_PROOF_LOG_HEADER` defines columns proving the Week 2 property:

- one shared frame-backbone call per tracked frame;
- one batched object-head operation per tracked frame;
- object count changes head batch size, not the number of backbone passes.

`V8_DEBUG_LOG_HEADER` reuses the V5 debug format and adds V8 feature-bank size.

### Lines 122-173: `SharedFrameLoRATEncoder`

This is the key Week 2 refactor.

Before V8, LoRaT was used as a single-object tracker: each object got its own search crop and LoRaT pipeline. V8 instead preprocesses the whole frame once and runs the LoRaT/DINOv2 ViT blocks once:

```python
tokens = self.lorat_model._x_feat(x)
for block in self.lorat_model.blocks:
    tokens = block(tokens)
tokens = self.lorat_model.norm(tokens)
```

The result is reshaped into a grid:

```text
feature_map: [grid_height, grid_width, embed_dim]
```

This is the shared frame representation used by every object.

### Lines 176-453: `BatchedObjectConditionedHead`

This is the per-object head bank.

Important idea:

- The frame feature tokens are shared.
- Each object has memory/template embeddings.
- The head scores every object against the shared feature grid in one batched operation.

The inner `V8ObjectConditionedLoRAHead` has:

- layer norms for frame features, template features, and object embeddings;
- template attention so object memory can condition frame tokens;
- a shared base projection;
- a generated low-rank object-conditioned delta;
- a score head and box head.

The key tensor idea:

```text
feature_tokens:      [locations, dim]
object_embeddings:   [objects, dim]
template_tokens:     [objects, template_tokens, dim]
score_logits:        [objects, locations]
box_deltas:          [objects, locations, 4]
```

`_build_head_tensor()` packs Python memory slots into tensors.

`score()` chooses one of two modes:

- If no trained head weights are loaded, use zero-shot cosine similarity.
- If weights are loaded, run the trainable template-conditioned head.

This design is why V8 can run a single object-head batch for N objects.

### Lines 456-1132: `V8FeatureIdentityArbitrator`

This class replaces V5's crop-histogram identity logic with feature-tensor identity logic.

It inherits from `v5.LightweightIdentityArbitrator` to reuse the configuration and view-change logic, but overrides the actual appearance path.

Important pieces:

- `normalize_feature()`: keeps appearance vectors normalized on device.
- `_output_feature_stack()`: packs candidate features into one tensor.
- `_track_memory_similarity_rows()`: compares track memory against candidate features in batch.
- `_motion_matrix()`: scores geometric motion agreement.
- `_path_matrix()`: scores whether a candidate follows the recent reliable trajectory.
- `_occlusion_matrices()`: checks which other tracks overlap a candidate.
- `_identity_score_matrices()`: builds all score components as matrices.
- `resolve()`: runs Hungarian assignment over the score matrix.

The shape to remember:

```text
score_matrix: [track_count, candidate_count]
```

This is the "scoring brain moved into batched math" part. Python still decides what to do with assignments, but score construction is matrix-based.

The old scalar `score()` remains as a compatibility/fallback method, but normal V8 resolution now uses `_identity_score_matrices()`.

### Lines 1135-1344: `V8QualityBatchedLoRATTracker.__init__`

This constructor stores all knobs and clamps them into safe ranges.

Examples:

- `max(0.0, min(1.0, confidence))` keeps confidence thresholds between 0 and 1.
- `max(1, int(...))` prevents zero or negative memory sizes.
- V5 constants are used as defaults so V8 keeps the same tuning vocabulary as V5.

It creates the main tracker state:

- `self.tracks`
- `self.track_by_id`
- `self.runtime_status`
- debug/proof buffers
- profiling buckets

Then it calls `_load_lorat_shared_backbone()`.

### Lines 1346-1510: Loading LoRaT internals

`_load_lorat_shared_backbone()` is where V8 reaches into the LoRaT checkout.

It:

1. Validates paths.
2. Imports LoRaT runtime modules.
3. Sets CPU/CUDA/DirectML device.
4. Loads the LoRaT config and checkpoint.
5. Builds the optimized inference model.
6. Extracts the underlying LoRaT model that exposes `_x_feat`, `blocks`, `norm`, and `x_size`.
7. Builds:
   - `SharedFrameLoRATEncoder`
   - `BatchedObjectConditionedHead`
   - `V8FeatureIdentityArbitrator`

This is the part that turns LoRaT from a packaged single-object tracker into a usable shared-frame backbone.

### Lines 1527-1656: Encoding, timing, proof logging

`_encode_frame()` calls the shared encoder and increments proof counters.

The profiling methods add timing into named buckets.

`_append_week2_proof_row()` writes per-frame proof that V8 is doing:

- one shared backbone pass;
- one object-head batch;
- N object-head items.

### Lines 1658-1747: Initialization and track creation

`initialize()` clears old tracks and creates one track per selected box.

`_create_track()` does the first-frame setup:

- clips the user box to the frame;
- encodes the frame if needed;
- creates an initial template slot from the selected box;
- creates a `v5.TrackState`;
- stores initial V8 feature memory;
- starts Kalman tracking;
- records trajectory history.

This is where the user's first click becomes a track object.

### Lines 1749-1994: Main per-frame tracking update

`update()` is the frame loop entry point.

For every frame:

1. Reset per-frame profiling.
2. Encode the frame once.
3. Gather active tracks.
4. Call `_score_and_update_tracks()`.
5. Update FPS/status.
6. Optionally write proof rows.

`_score_and_update_tracks()` is the heart of V8:

1. Select object memory heads for each track.
2. Run the batched object-conditioned head once.
3. Predict each track's location.
4. Decode score maps into candidate boxes.
5. Rerank candidates with feature similarity.
6. Optionally run feature-template rescue.
7. Create synthetic `LoRATSlotOutput` objects for identity assignment.
8. Resolve identity with the batched identity matrix.
9. Accept or hold each track.

This is the main function to study if you want to understand how MOT control logic is built.

### Lines 1996-2172: Head selection and debug rows

The head bank can contain multiple memory slots per object:

- first-frame anchor;
- recent reliable updates;
- possibly recovery memories.

`_select_track_heads()` chooses how many heads to use this frame. Normally it uses a small primary set. If the track is uncertain, stale, occluded, or recovering, it selects more recovery heads.

`_append_head_debug_rows()` is opt-in. It writes debug details but should not be enabled for FPS benchmarking unless needed.

### Lines 2174-2494: Prediction, candidate extraction, reranking

`_predict_track()` uses Kalman if available, otherwise a simple bbox delta.

`_candidates_from_head_output()` converts score maps and box deltas into actual `(x, y, w, h)` candidates. This is mostly tensor-side now:

- apply ROI mask around predicted location;
- apply optional window penalty;
- find top scoring grid cells;
- decode box deltas into pixel boxes;
- pack final candidates back into Python dataclasses.

`_rerank_head_candidate()` uses batched feature similarity to choose a better candidate from the top-K head outputs.

### Lines 2496-2786: Acceptance rules and learning gates

These methods decide whether a candidate is trustworthy.

Important concepts:

- `_apply_identity_scores()` writes score components onto the track for debugging/status.
- `_candidate_reject_state()` rejects low-confidence, bad-ReID, bad-motion, bad-path, or anchor-stealing candidates.
- `_is_reid_recovery()` and `_is_path_recovery()` allow recovery cases that would otherwise look risky.
- `_assess_learning_hold()` decides whether the visible bbox may update the track but should not refresh memory.
- `_apply_scale_limits()` prevents catastrophic size jumps or collapse.
- `_is_strong_memory_update()` decides whether this frame is good enough to refresh memory.

This is where quality is protected. A tracker usually gets bad when it learns from bad frames. These functions try to stop that.

### Lines 2788-2989: Accept or hold

`_accept_candidate()` commits a candidate to a track.

It:

- clips and scale-limits the box;
- rejects unsafe candidates;
- updates bbox, velocity, confidence, identity scores, occlusion state, and Kalman state;
- decides whether memory should refresh;
- records trajectory and size history.

`_hold_track()` is the fallback when a candidate is unsafe. It keeps the track alive using prediction instead of learning from a suspicious box.

This is a common MOT pattern:

```text
bad observation -> do not immediately delete the track
bad observation -> hold/coast briefly
good observation -> accept and possibly refresh memory
```

### Lines 2991-3273: Feature patches, template rescue, candidate fusion

These functions operate on shared frame features.

- `_bbox_to_grid_slices()` maps pixel boxes to feature-grid slices.
- `_feature_mean_for_bbox()` gets one vector for a box.
- `_feature_patch_and_foreground_mask_for_bbox()` gets patch tokens and a foreground mask.
- `_template_slot_for_bbox()` creates a memory slot from a selected or accepted box.
- `_feature_template_candidate()` searches nearby frame tokens using stored template tokens.
- `_fuse_head_and_template_candidate()` decides whether to use the head candidate, template candidate, or a blend.

This is how V8 keeps some of LoRaT's template-matching benefit without calling LoRaT separately for every object.

### Lines 3275-3428: Memory management

These methods manage V8's per-track memory bank.

The memory bank stores:

- the first-frame anchor;
- recent reliable template slots;
- patch tokens;
- foreground masks;
- confidence/frame metadata.

The rules are conservative: V8 only refreshes memory when confidence, identity, motion, path, and stability evidence are good enough.

### Lines 3436-3529: Status and cleanup

`_update_gpu_status()` reads CUDA memory stats.

`status_lines()` returns human-readable overlay lines for the GUI.

`runtime_status_snapshot()` returns a copy used by benchmarks.

`close()` releases the model manager.

### Lines 3532-3982: CLI, backend creation, GUI loop

The bottom of the file is the executable script layer.

- `parse_initial_boxes()` parses `--initial-boxes`.
- `parse_args()` defines CLI flags.
- `create_backend()` maps CLI args into the tracker constructor.
- `main()` opens the source, gets initial boxes, creates the backend, loops frames, handles `a` to add objects, writes MOT output, writes debug/proof logs, writes video, and draws the GUI.

This layer reuses many V5 utilities because GUI/video/MOT output behavior is not the point of V8.

## How To Think About This As Code

The file is large, but it follows a normal tracking architecture:

```text
Input frame
  -> shared frame encoder
  -> per-object batched head
  -> candidate decode
  -> identity association
  -> accept or hold
  -> memory refresh
  -> output/render/log
```

When writing code like this yourself, separate the system into layers:

1. Data containers: simple dataclasses for passing results.
2. Model wrappers: code that loads and runs neural networks.
3. Tensor scoring: batched math on GPU/NumPy arrays.
4. State machine: Python logic for tracks, decisions, and memory.
5. IO/UI layer: files, video, GUI, debug logs.

V8 is currently doing this split more cleanly than older versions. The main thing to keep improving is making sure expensive scoring stays in layer 3, not scattered through layer 4.

## What V8 Still Gets From V5 By Category

### Data Structures

- `BBox`
- `TrackState`
- `LoRATSlotOutput`
- `LoRATMemorySlot`
- `IdentityScore`
- `IdentityAssignment`
- `RuntimeStatus`
- `CropInformation`
- `FrameSource`

### Geometry and Tracking Math

- `bbox_center`
- `bbox_area`
- `bbox_diagonal`
- `bbox_iou`
- `bbox_delta`
- `clamp_bbox_size`
- `clamp_bbox_to_frame_bounds`
- `clip_bbox_to_frame`
- `motion_affinity`
- `center_path_affinity`
- `strongest_track_overlap`
- `kalman_prediction_reference`
- `predict_bbox`
- `BBoxKalmanFilter`

### Memory, Learning, and Identity Defaults

- LoRaT memory slot counts.
- LoRaT accept and scale thresholds.
- Shrink guard thresholds.
- Identity, ReID, motion, path, occlusion, and view-change thresholds.
- Path recovery thresholds.
- Center-path constants.

### UI and Output Helpers

- `open_frame_source`
- `select_boxes`
- `draw_tracks`
- `make_video_writer`
- `append_mot_results`
- `write_slot_debug_log`
- default output/debug/video paths

### Formatting and Status Helpers

- `csv_float`
- `csv_text`
- `csv_bbox`
- `csv_bbox_measurements`
- `bytes_to_mb`

## Is Importing V5 Bad?

No, not by itself.

It would be a problem if V8 imported V5 and then quietly called V5's old tracker update path or per-object evaluator. It does not do that in the V8 frame update path.

The current import is mostly code reuse. Long term, if the project matures, these shared utilities should probably move into a neutral module, for example:

```text
programs/tracking_core.py
programs/tracking_geometry.py
programs/tracking_io.py
```

Then V5 and V8 could both import those utilities without V8 appearing dependent on an older tracker version.

## Good Study Exercises

To learn this code, try these in order:

1. Read `SharedFrameLoRATEncoder.encode()` and write down the tensor shape after each step.
2. Read `BatchedObjectConditionedHead.score()` and trace how `selected_banks` becomes `score_maps`.
3. Read `_score_and_update_tracks()` and draw its control flow on paper.
4. Read `_identity_score_matrices()` and identify every score term: appearance, motion, path, source, confidence, IoU.
5. Read `_accept_candidate()` and list every condition required before memory is refreshed.
6. Read `main()` last. It is mostly UI and IO glue.

