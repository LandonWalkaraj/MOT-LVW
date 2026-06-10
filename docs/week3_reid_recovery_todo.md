# Week 3 To-Do: ReID and Track Recovery

Date: 2026-06-10

Week 3 scope from the summer statement of work: add appearance-based re-identification and track recovery, benchmark identity/recovery behavior, and expose lost-track/manual-reanchor behavior in the interface.

Current context: V8 has the right shared-frame architecture, but the trained head quality is still low. The retraining overhaul should be treated as part of Week 3 because ReID will not rescue a tracker whose localization head is not learning stable boxes.

## Week 3 Statement-Of-Work Requirements

- (a) Integrate DINOv2 crop embeddings so a lost track re-attaches automatically when the object reappears, operating within the existing interface.
- (b) Benchmark identity-switch count and track-loss rate per video, with and without re-identification.
- (c) Benchmark track survival against occlusion duration, expressed as the number of occluded frames tolerated before identity is lost.
- (d) Add an interface indicator for lost tracks together with a single-action manual re-anchor, with each manual re-anchor recorded as a measured cost event.

## Week 3 Definition Of Done

- A V8/V9 tracker can mark tracks as healthy, uncertain, lost, and reacquired.
- DINOv2/LoRAT feature embeddings are used for re-identification when a track is lost or contested.
- The GUI visibly marks lost tracks and supports a single-action manual re-anchor.
- Manual re-anchor events are logged as human-cost events.
- Benchmarks report identity-switch count, track-loss rate, and occlusion survival with ReID on and off.
- The retrained head passes a small quality gate before being used for headline Week 3 results.

## P0: Training Overhaul Before More Claims

- [ ] Add a tiny overfit smoke test for the V8 head.
  - Train on one sequence slice with 8-16 object/frame samples.
  - Expected result: the head should overfit to high IoU quickly.
  - If it cannot overfit, stop and fix loss/head/data before launching long jobs.

- [ ] Make training diagnostics sensitive.
  - Save train loss, objectness loss, box loss, mean IoU, IoU@0.50, and ReID loss per epoch.
  - Evaluate every checkpoint on the same validation slice.
  - Save `best_by_val_iou.pt` separately from `latest.pt`.
  - Investigate why previous train/val diagnostic values repeated exactly across epochs.

- [ ] Strengthen localization supervision.
  - Keep LoRAT-style BCE plus GIoU.
  - Add SmoothL1/L1 on normalized left/top/right/bottom distances for positive cells.
  - Add center sampling or centerness so interior positive cells are not all equal.
  - Try focal or Varifocal-style classification if dense negatives dominate.
  - Use hard negative cells from nearby distractor objects.

- [ ] Make the retraining run reproducible.
  - Pin dataset root, train/val split, max sequences, stride, seed, LoRAT config, and head config in the output metadata.
  - Store the exact command used for each checkpoint.
  - Log CUDA device name and PyTorch/CUDA versions.

- [ ] Quality gate for using a trained V8 head.
  - Minimum smoke gate: mean IoU >= 0.5 and IoU@0.50 >= 0.5 on the tiny overfit slice.
  - Minimum validation gate for demos: improve over zero-shot V8 and over previous trained V8 on the same DanceTrack slice.

## P0: ReID Training

- [ ] Add a QDTrack-style contrastive ReID objective.
  - Positive pairs: same GT track across frames.
  - Negatives: other visible tracks in the same frame or nearby frames.
  - Hard negatives: nearby boxes, overlapping boxes, and same-sequence lookalikes.
  - Candidate embeddings: V8 template vectors, pooled frame features, or a small ReID projection head.

- [ ] Add sequence-window training.
  - Train on 2-5 frame windows instead of independent frames only.
  - Include stale templates, query dropout, noisy previous boxes, and temporary missing targets.
  - Add a "no confident target" or lost-target target where appropriate.

- [ ] Add ReID diagnostics.
  - Same-ID cosine distribution.
  - Different-ID cosine distribution.
  - Hard-negative accuracy.
  - ReID retrieval top-1/top-5 across a validation sequence.
  - Reacquisition success after artificial gaps.

## P0: Runtime ReID And Recovery Features

- [ ] Make a clear embedding module.
  - Pool DINOv2/LoRAT shared-frame features inside a bbox.
  - Normalize embeddings.
  - Keep initial, recent, and high-confidence feature banks per track.
  - Store feature timestamp, confidence, and source state.

- [ ] Upgrade lost-track recovery.
  - When localization confidence is low, search around predicted motion and recent feature matches.
  - Allow recovery candidates to reattach only when appearance, motion, and assignment margin pass thresholds.
  - Do not refresh memory from low-confidence or overlapping frames.

- [ ] Add track states.
  - `HEALTHY`: accepted high-confidence update.
  - `UNCERTAIN`: low margin, weak ReID, high jitter, or contested candidate.
  - `LOST`: no trusted candidate for N frames.
  - `REACQUIRED`: lost track recovered by ReID.
  - `MANUAL_REANCHOR`: user corrected or reattached the track.

- [ ] Reduce identity arbitration cost.
  - Vectorize pairwise track-candidate ReID scores.
  - Keep feature banks in tensors where practical.
  - Prune candidates before expensive scoring.

## P0: GUI And Human-Cost Logging

- [ ] Show lost and uncertain tracks in the GUI.
  - Healthy tracks: normal color.
  - Uncertain tracks: warning color or dashed box.
  - Lost tracks: visible label and held/predicted box if available.
  - Reacquired tracks: brief state label.

- [ ] Add a single-action manual re-anchor.
  - User selects a lost/uncertain track.
  - User draws a new box on the current frame.
  - Existing track ID is preserved.
  - ReID and template memory update from the new anchor.

- [ ] Log correction-cost events.
  - Event type: `manual_reanchor`.
  - Frame number, track ID, old bbox, new bbox.
  - Time spent if available.
  - Whether the track was lost, uncertain, or healthy before correction.

- [ ] Add an event CSV.
  - Use this now so Week 5 active-correction and Week 7 simulated annotator can reuse the same cost accounting.

## P0: Week 3 Benchmarks

- [ ] With/without ReID ablation.
  - Same tracker, same sequences, same seeds.
  - Run with ReID enabled and disabled.
  - Report identity switches, track-loss rate, correct-object rate, mean IoU, and IoU@0.50.

- [ ] Occlusion survival benchmark.
  - Measure how many occluded/missing frames a track survives before identity is lost.
  - Use natural occlusions from DanceTrack/MOT17 where available.
  - Add artificial dropout gaps if natural occlusions are too sparse.

- [ ] Track-loss benchmark.
  - A track is lost when it has no trusted object match for N frames or no longer overlaps its initialized GT object.
  - Report loss rate per video and mean frames until loss.

- [ ] Identity-switch benchmark.
  - Count a switch when a tracker ID matches a different GT object than its initialized GT.
  - Report switches per sequence and per 1000 frames.

- [ ] Minimum benchmark matrix for this week.
  - DanceTrack: at least 3 validation sequences.
  - MOT17: at least 2 train sequences if local data is ready.
  - V8/V9 configs: start with B-224, then L-224/g-224 only after B-224 quality is acceptable.

## P1: Evaluation And Reporting

- [ ] Wire TrackEval or a TrackEval-compatible export path.
  - HOTA, DetA, AssA, IDF1, MOTA should become first-class outputs.
  - Current internal IoU/correct-object metrics are useful but not enough for the paper.

- [ ] Split throughput and quality claims.
  - Throughput status: shared-frame proof, FPS, memory.
  - Quality status: IoU, IDF1/HOTA, switches, losses.
  - Human-cost status: manual re-anchor events.

- [ ] Add a Week 3 summary generator.
  - One command writes `summary.md`, CSVs, and plots for ReID on/off and occlusion survival.

## P1: Paper And Research Follow-Ups

- [x] Add missing baseline papers.
  - BoostTrack, required by Week 9.
  - StrongSORT and BoT-SORT, now present locally.
  - OC-SORT as another motion-heavy baseline.
  - Gap-fill catalog: `docs/week3_gap_fill_papers.md`.

- [x] Add active-correction references.
  - Active learning for object detection/tracking.
  - Interactive video object segmentation or memory propagation papers.
  - Human annotation cost measurement papers.
  - Gap-fill catalog: `docs/week3_gap_fill_papers.md`.

- [ ] Keep open-world discovery separate this week.
  - Do not block ReID/recovery on OWL-ViT/SAM proposals.
  - Only add proposal papers or scaffolding if ReID work is stable.

## Suggested Work Order

1. Make the training overfit smoke pass.
2. Add stronger box loss and rerun B-224 head training.
3. Add contrastive ReID loss and ReID diagnostics.
4. Integrate pooled DINOv2/LoRAT feature ReID into runtime recovery.
5. Add GUI lost/uncertain/reanchor states and event logging.
6. Run ReID on/off and occlusion survival benchmarks.
7. Package Week 3 summary with honest throughput/quality separation.

## Stretch Goals

- TrackEval/HOTA fully wired for DanceTrack and MOT17.
- Tensorized identity arbitration.
- Short sequence-window training with no-object/lost-target supervision.
- Manual re-anchor propagation forward after correction.
- First draft of the Week 5 uncertainty score using confidence margin, box jitter, and ReID distance.
