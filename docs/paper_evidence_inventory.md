# Paper Evidence Inventory

Date: 2026-06-18

This document gathers the project evidence that is useful for a final paper about a LoRAT-based, user-initialized multi-object video labeling tool. It separates paper-ready evidence from diagnostic evidence and known gaps.

## Project Thesis

The system starts from LoRAT as a strong single-object tracker and extends it toward a video labeling tool where a user draws one box for an unnamed object, then the system tracks all matching instances across the video with bounded human correction.

The intended final system has four main pieces:

- Multi-object LoRAT tracking from user boxes.
- Shared Vision Transformer frame computation with batched per-object heads.
- Appearance-based re-identification and recovery after loss/occlusion.
- Open-world proposal and active correction to reduce human labeling effort.

## Current Architecture Evidence

### V6/V4 Baseline: Whole-LoRAT Per Track

Claim supported:

- The earlier system can run multiple LoRAT-backed track states, one per selected object or per memory slot, with identity/motion/path gates around LoRAT outputs.

Relevant files:

- `programs/bounding_box_v4_lorat_memory.py`
- `programs/bounding_box_v5_lorat_shared.py`
- `programs/bounding_box_v6_lorat_quality.py`
- `programs/mot_common.py`

Paper role:

- Baseline for quality-preserving, wrapper-level SOT-to-MOT behavior.
- Baseline for Week 1 and "before shared-backbone refactor" comparisons.

Caveat:

- This path preserves more of upstream LoRAT behavior but scales poorly because it still performs too much per-object LoRAT work.

### V8: Shared Frame Encoder and Batched Object Head

Claim supported:

- V8 uses one shared LoRA-adapted DINOv2/ViT frame pass and then batches object-conditioned low-rank heads across active objects.

Relevant files:

- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/train_lorat_v8_head.py`
- `programs/benchmark_lorat_v8.py`

Important wording:

- V8 is not "upstream LoRAT unchanged, run with many objects."
- It is better described as a LoRAT-backbone shared-frame MOT branch with a new template-conditioned low-rank head.

Evidence:

- `outputs/v8_benchmark_results_20260610_092523/summary.md`
- `outputs/v8_benchmark_results_20260610_092523/week2_shared_backbone_proof.csv`

Representative Week 2 proof numbers:

| Config | N | FPS | Peak GPU Reserved MB | Shared Calls/Frame | Head Items/Update | 25 FPS |
|---|---:|---:|---:|---:|---:|---|
| B-224 | 1 | 53.296 | 478 | 1.000 | 1.000 | Yes |
| B-224 | 2 | 43.492 | 840 | 1.000 | 1.631 | Yes |
| B-224 | 3 | 30.439 | 1210 | 1.000 | 2.625 | Yes |
| B-224 | 4 | 15.809 | 1582 | 1.000 | 4.000 | No |
| B-224 | 5 | 12.456 | 1952 | 1.000 | 4.752 | No |
| L-224 | 1 | 43.492 | 3216 | 1.000 | 1.000 | Yes |
| L-224 | 2 | 37.633 | 4144 | 1.000 | 1.526 | Yes |
| g-224 | 1 | 35.931 | 9612 | 1.000 | 1.000 | Yes |
| g-224 | 2 | 27.636 | 14036 | 1.000 | 2.000 | Yes |

Interpretation:

- The shared-backbone requirement is structurally supported: object count does not multiply the ViT frame pass.
- Throughput is good for B-224 up to N=3 in that benchmark.
- Quality was poor in this run, so speed and tracking quality must be reported separately.

## Benchmark Protocol Evidence

Core protocol document:

- `docs/benchmark_protocol.md`

Metrics currently defined:

- IoU: `area(predicted box intersection GT box) / area(predicted box union GT box)`.
- IoU@0.50: fraction of evaluated boxes with IoU at least 0.50.
- FPS and ms/box.
- GPU memory allocated/reserved/peak.
- Shared-backbone proof counters.
- Identity switch count.
- Track-loss rate.
- Correct-object rate.
- Natural occlusion survival.
- Controlled occlusion survival.
- Proposal recall and manual boxes saved.

Established metrics already aligned with existing literature:

- IoU / overlap success.
- FPS.
- GPU memory.
- ID switches.
- Track loss.
- Proposal recall.

Metrics still needed for stronger paper comparability:

- HOTA.
- DetA / AssA.
- IDF1.
- MOTA.
- TrackEval-compatible export.

Relevant references already stored locally:

- `papers/lorat_tracking_meets_lora.pdf`
- `papers/lorat_supplemental_training_details.pdf`
- `papers/dinov2_learning_robust_visual_features_without_supervision.pdf`
- `papers/vision_transformer_an_image_is_worth_16x16_words.pdf`
- `papers/SOT_For_MOT.pdf`
- `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`
- `papers/DCFST.pdf`
- `papers/deep_sort.pdf`
- `papers/bytetrack.pdf`
- `papers/hota_metric.pdf`
- `papers/tao_tracking_any_object.pdf`

## Week 1 Evidence: Multi-Object LoRAT and Core Benchmarks

### Object Count Timing

Source:

- `outputs/PRESENTATION MATERIAL/benchmark_conclusions.md`
- `outputs/PRESENTATION MATERIAL/v4_dancetrack0065_B-224-L-224-g-224_N1-2-3-4-5-6_frames200_20260601_103248/summary.md`

Representative DirectML/local timing:

| Model | Completed Object Counts | Tracking ms/box Range | FPS Range | Mean IoU Range | IoU@0.50 Range |
|---|---:|---:|---:|---:|---:|
| B-224 | N=1 through N=5 | 747.5 to 949.8 | 1.05 to 0.27 | 0.320 to 0.488 | 0.315 to 0.535 |
| L-224 | N=1 through N=5 | 1088.6 to 1272.1 | 0.79 to 0.18 | 0.308 to 0.602 | 0.358 to 0.730 |
| g-224 | N=1 only | 2683.3 | 0.37 | 0.439 | 0.450 |

Paper interpretation:

- This is useful as a local baseline and a "why shared backbone was needed" result.
- It should not be used as final CUDA throughput because it used local DirectML constraints.

### Controlled Small-Object Area

Sources:

- `outputs/PRESENTATION MATERIAL/benchmark_conclusions.md`
- `programs/benchmark_lorat_v6_forced_area.py`

Reliability rule used in the controlled area report:

- IoU@0.50 >= 0.80.
- Mean IoU >= 0.50.
- Lost rate <= 0.20.
- At least 10 samples.

Controlled area conclusion:

| Model | Smallest Reliable Area | Notes |
|---|---:|---|
| B-224 | About 1500 px | 1000 px was not reliable under the final rule. |
| L-224 | About 1500 px | Did not clearly beat B-224 on minimum area. |
| g-224 | Incomplete | Not enough completed evidence for a full area claim. |

Paper interpretation:

- This is the cleanest evidence for "how small before unreliable" because the target area was explicitly forced.
- The natural-video area bins answer a different question and were confounded by identity switches.

## Week 2 Evidence: Shared Backbone and Throughput Scaling

Source:

- `outputs/v8_benchmark_results_20260610_092523/summary.md`
- `outputs/PRESENTATION MATERIAL/benchmark_visual_assets/`
- `outputs/PRESENTATION MATERIAL/lorat_week2_shared_backbone_diagram.svg`

Main paper-ready architectural result:

- V8 proves one shared frame-backbone call per frame.
- V8 proves batched head item count increases with object count.
- The B-224 HPC run sustained at least 25 FPS through N=3.

Main caveat:

- V8 head quality was low in the initial benchmark:
  - B-224 N=1 mean IoU 0.107, IoU@0.50 0.071.
  - B-224 N=3 mean IoU 0.121, IoU@0.50 0.111.
  - B-224 N=5 mean IoU 0.149, IoU@0.50 0.119.

Paper interpretation:

- Week 2 should be written as a computational refactor success, not as final tracking quality success.
- The architectural tradeoff created the V8 training problem: splitting from upstream LoRAT required training a new object-conditioned head.

## V8 Training Evidence

Current V8 training file:

- `programs/train_lorat_v8_head.py`

Training mechanisms implemented:

- Frozen LoRAT/DINOv2 ViT backbone.
- Trainable V8 template-conditioned low-rank head.
- Template/search style sampling.
- Staged search-crop to full-frame training.
- Synthetic selected-region training for full body and part boxes.
- Mixed template sampling from first, previous, and short-window frames.
- Hard negatives from other objects.
- DCFST-style candidate discrimination.
- Assignment discrimination.
- Contrastive ReID loss.
- Closed-loop simulated previous-box training.
- TAO/TAO-OW dataset adapter.
- Diagnostics, debug visuals, best-by-validation-IoU checkpoints.

Representative 48-hour B-224 training evidence:

| Run | Train Mean IoU | Train IoU@0.50 | Val Mean IoU | Val IoU@0.50 | Val ReID Top-1 | Best Val Mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| `lorat-v8-train-b224-48h-666917.out` | 0.5082 | 0.6064 | 0.4219 | 0.4800 | 0.7573 | 0.4274 |
| `lorat-v8-train-b224-48h-668222.out` | 0.4669 | 0.5459 | 0.3499 | 0.3784 | 0.6204 | 0.3540 |

Paper interpretation:

- Training is learning something real; it is not stuck at zero.
- Validation quality is still below a strong demo/paper quality gate.
- Longer training and more diverse data may help, but the current benchmark videos show that improved validation IoU has not yet fully translated into stable selected-object tracking.

Active/pending training:

- One 48h B-224 run focused on extended epochs.
- One 48h B-224 run mixing prior data with TAO/YFCC material.

## Week 3 Evidence: ReID, Recovery, and Manual Reanchor

Implementation evidence:

- `programs/bounding_box_v8_lorat_quality_batched.py`
- `programs/benchmark_lorat_v8.py`
- `docs/week3_benchmark_usage.md`

Implemented features:

- DINOv2 crop ReID is enabled by default in V8.
- Per-track feature/crop memory banks.
- Identity arbitration with appearance, motion, path, IoU, and initial-anchor margin.
- Lost/uncertain/reacquired/manual lifecycle states.
- Interface lost/uncertain status display.
- Single-action manual reanchor (`r` in interface).
- Manual reanchor cost event logging.
- ReID-on/ReID-off benchmark ablation.
- Identity switch and track-loss benchmark.
- Natural and controlled occlusion survival benchmark.

Latest downloaded Week 3 benchmark evidence:

- `C:\Users\lando\Downloads\lorat-v8-week3-results-crop-reid-full-20260616.zip`
- `C:\Users\lando\Downloads\lorat-v8-week3-670188.out`

Representative log values:

| Case | FPS | IoU@0.50 | Notes |
|---|---:|---:|---|
| B-224 N=4, ReID on | 51.431 | 0.196 | Shared frame proof still passes. |
| B-224 N=5, ReID on | 20.802 | 0.187 | Crop ReID adds major runtime cost. |
| B-224 N=5, ReID off | 49.060 | 0.166 | Faster, but quality still poor. |

Failure review evidence:

- `outputs/PRESENTATION MATERIAL/v8_failure_review/failure_review.md`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/candidate_failure_summary.csv`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/identity_failure_summary.csv`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/contact_sheets/`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/failure_bursts/`

Failure review scope:

- 103 installed MP4s inventoried.
- 35 videos had matching diagnostics.

Key failure findings:

- Dominant issue is not only jitter.
- The most common failure mode is plausible-looking boxes on the wrong object or wrong body part.
- `head_no_good_candidate` remains high in current Week 3 diagnostic runs.
- ReID reduces some "track lost" behavior but does not yet preserve identity reliably.

Representative Week 3 diagnostic ranges:

| Metric | Typical Recent B-224 Range |
|---|---:|
| OK@0.50 | 11% to 19% |
| Low IoU < 0.30 | 72% to 86% |
| Wrong object preferred | 64% to 87% |
| Head no good candidate | 60% to 85% |

Representative ReID effect:

| Run | N | ReID | Correct | Track Lost | ID Switch |
|---|---:|---|---:|---:|---:|
| 20260615_102945 | 1 | off | 9.4% | 49.3% | 63.2% |
| 20260615_102945 | 1 | on | 9.4% | 5.7% | 85.5% |
| 20260615_114109 | 1 | off | 13.2% | 3.6% | 83.3% |
| 20260615_114109 | 1 | on | 13.5% | 1.6% | 85.2% |

Paper interpretation:

- ReID is currently better at keeping a track alive than keeping it correct.
- The next paper claim should be diagnostic: "crop ReID lowers loss in some cases but can amplify identity switches when the localization head is weak."
- A publishable ReID claim needs stronger head quality or stronger identity gating.

## Week 4 Evidence: Open-World Proposal Scaffolding

Implementation evidence:

- `programs/benchmark_lorat_week4.py`
- `outputs/week4_smoke_selective_1000/summary.md`
- `outputs/week4_tao_yfcc_smoke/summary.md`

Current benchmark definition:

- Proposal recall at IoU >= 0.50.
- Manual boxes saved under oracle acceptance.
- Proposal count and runtime per frame.

Representative smoke results:

| Dataset | Source | Frames | GT Objects | Proposals | Recall | Mean ms/frame |
|---|---|---:|---:|---:|---:|---:|
| DanceTrack0065 | selective search | 2 | 10 | 1748 | 0.900 | 9760.2 |
| DanceTrack0065 | selective search fast/low proposal | 2 | 10 | 400 | 0.200 | 8781.3 |
| TAO-OW/YFCC | contour | 1 | 4 | 11 | 0.000 | 40.4 |

Paper interpretation:

- Open-world proposal scaffolding exists.
- Selective search can reach high oracle recall on a tiny DanceTrack sample, but runtime is far too slow for an interactive system.
- TAO-OW contour smoke did not work as a useful proposal method.
- This is not yet a final open-world discovery result.

## Visual and Presentation Assets

Useful existing visuals:

- `outputs/PRESENTATION MATERIAL/lorat_week2_shared_backbone_diagram.svg`
- `outputs/PRESENTATION MATERIAL/benchmark_visual_assets/`
- `outputs/PRESENTATION MATERIAL/week3_v8_visuals/`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/contact_sheets/`
- `outputs/PRESENTATION MATERIAL/v8_failure_review/failure_bursts/`
- `outputs/PRESENTATION MATERIAL/v6_v8_visual_assets/`

Paper figure candidates:

- V6 whole-LoRAT memory tracker vs V8 shared-frame head architecture.
- Shared-backbone proof diagram.
- FPS vs object count.
- GPU memory vs object count.
- Controlled small-object reliability curve.
- ReID on/off track-loss and identity-switch chart.
- Failure taxonomy contact sheet.
- Open-world proposal recall vs proposal count/runtime chart.

## Strong Paper Claims Available Now

These are defensible with current evidence:

1. A LoRAT SOT tracker can be wrapped into a multi-object tracker, but per-object LoRAT scaling is expensive.
2. Controlled small-object stress testing suggests a practical reliability floor around 1500 px for B-224/L-224 in the tested setup.
3. V8 achieves the Week 2 computational structure: one shared frame ViT pass and batched object-conditioned head work.
4. On A100/CUDA, V8 B-224 sustains 25 FPS up to N=3 in the initial full-sequence benchmark.
5. V8 quality is currently limited by head localization and wrong-object association, not only by jitter.
6. DINOv2 crop ReID can reduce some track-loss behavior but does not yet solve identity preservation.
7. Open-world proposal scaffolding is present, but current proposal methods are either too slow or too weak for final claims.

## Claims Not Yet Paper-Ready

These should be treated as active work:

1. "V8 tracks selected objects well." Current evidence does not support this yet.
2. "ReID improves identity accuracy." Current evidence is mixed: track loss can improve, identity switches remain high.
3. "TAO-OW open-world discovery works." Only smoke/scaffolding exists.
4. "The system reduces human effort to a target labeling quality." Manual-cost logging exists, but active-correction benchmarks are not complete.
5. "The tracker is comparable to MOT baselines." TrackEval/HOTA/IDF1/MOTA are not wired yet.

## Recommended Paper Tables

### Table 1: System Versions

Columns:

- Version.
- Architecture.
- Preserves upstream LoRAT head?
- Shared frame backbone?
- ReID?
- Open-world proposal?
- Main purpose.

### Table 2: Week 1 Timing and Small Object Reliability

Columns:

- Config.
- Object count.
- FPS.
- ms/box.
- Mean IoU.
- IoU@0.50.
- Smallest reliable forced area.

### Table 3: Week 2 Shared Backbone Scaling

Columns:

- Config.
- N.
- FPS.
- Peak GPU reserved MB.
- Shared calls/frame.
- Head items/update.
- 25 FPS sustained.

### Table 4: V8 Training Runs

Columns:

- Training run.
- Dataset mix.
- Epochs/steps completed.
- Train mean IoU.
- Val mean IoU.
- Val IoU@0.50.
- ReID top-1.
- Chosen checkpoint.

### Table 5: ReID Ablation

Columns:

- Config.
- N.
- ReID mode.
- Correct-object rate.
- Track-loss rate.
- Identity-switch count/rate.
- Mean IoU.
- FPS.

### Table 6: Failure Taxonomy

Columns:

- Failure type.
- Frequency.
- Example frame/video.
- Likely cause.
- Proposed repair.

### Table 7: Open-World Proposal Recall

Columns:

- Dataset.
- Proposal method.
- Frames.
- GT objects.
- Proposals/frame.
- Recall@0.50.
- ms/frame.
- Manual boxes saved under oracle acceptance.

## Recommended Experiments Before Manuscript Lock

Highest priority:

1. Finish the active 48h B-224 training runs.
2. Benchmark those exact checkpoints with `benchmark_lorat_v8.py`.
3. Compare against the failure taxonomy:
   - Did `head_no_good_candidate` decrease?
   - Did `wrong_object_preferred` decrease?
   - Did ReID track-loss decrease without raising ID switches?
4. Wire TrackEval-compatible exports.
5. Run at least:
   - 3 DanceTrack validation sequences.
   - 2 MOT17 train sequences.
   - A small TAO-OW subset for proposal recall.

Paper-critical:

1. Add HOTA/IDF1/MOTA table or clearly state that current metrics are internal diagnostics.
2. Add human-correction cost benchmark once active-correction loop is complete.
3. Use one consistent reliability rule across text, code, and tables.
4. Keep throughput and quality claims separate.

## Current Narrative

The cleanest paper story right now is:

1. Start from LoRAT because user-initialized SOT naturally matches the "draw one box, track this object" labeling workflow.
2. Show that a straightforward multi-object wrapper preserves SOT behavior but scales poorly.
3. Refactor to a shared-frame ViT and batched object-conditioned head, proving the desired compute structure and improving throughput.
4. Show that this architectural split creates a new learning problem: the new head must relearn selected-object localization and identity discrimination.
5. Add ReID and recovery infrastructure, but diagnose that ReID cannot fix weak localization by itself.
6. Move toward object-agnostic and open-world training with TAO/YFCC plus stronger selected-region supervision.
7. Use failure taxonomy, identity metrics, and human-cost metrics to guide the final active-correction system.
