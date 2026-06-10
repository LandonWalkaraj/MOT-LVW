# V8 Papers, Training, and Summer Onboarding Review

Date: 2026-06-10

Scope reviewed:

- All PDFs currently under `papers/`.
- V8 runtime: `programs/bounding_box_v8_lorat_quality_batched.py`.
- V8 training: `programs/train_lorat_v8_head.py`.
- V8 benchmark/training outputs under `outputs/v8_benchmark_results_20260610_092523` and `outputs/v8_training_results_666089`.
- Summer onboarding / statement-of-work text extracted at `outputs/summer_onboarding_extracted.txt`.

## Short Verdict

V8 is a useful and defensible research branch: it has the right shared-frame computation shape for Week 2, and its head training borrows the right broad loss family from LoRAT. The problem is that the current evidence supports "we built the throughput architecture" much more strongly than "we built a reliable tracker." The trained head is not yet good enough for the summer scope's labeling tool, and the paper set says the missing piece is not another speed refactor. The missing piece is identity-aware, sequence-aware training.

The biggest recommended pivot is:

1. Keep V8's shared-frame LoRAT backbone.
2. Fix the training objective around target localization quality.
3. Add QDTrack/FairMOT-style ReID contrastive learning.
4. Add MOTR/TrackFormer-style sequence windows and track-query robustness.
5. Add open-world proposal generation as a separate branch after the tracker has reliable propagation.

## Highest Priority Problems

### P0: V8 quality is currently too low for the labeling objective.

The full DanceTrack0065 run proves that the architecture runs, but not that it tracks well. In `outputs/v8_benchmark_results_20260610_092523/summary.md`, final mean IoU is mostly around 0.1-0.2, IoU@0.50 is mostly below 0.25, and the smallest reliable area table is empty. Examples:

- B-224, N=1: 53.296 FPS, but mean IoU 0.107 and IoU@0.50 0.071.
- g-224, N=5: 12.551 FPS, mean IoU 0.172 and IoU@0.50 0.181.
- No small-object bin reaches the reliability rule.

This means the current Week 2 story should be: "shared-backbone throughput achieved, tracking head still under quality threshold." It should not be presented as a completed high-quality MOT tracker yet.

Recommended changes:

- Add a short "quality gate" before future demos: do not call a V8 head usable until it reaches at least mean IoU >= 0.5 and IoU@0.50 >= 0.5 on a small held-out DanceTrack slice.
- Make the benchmark report a red/yellow/green status for throughput and quality separately.
- Keep V5/V6 as fallback baselines until V8 head quality catches up.

### P0: V8 training lacks an explicit identity/ReID objective.

The paper set is almost unanimous on this point: MOT is not just target localization. The difficult part is target-versus-nearby-target discrimination over time.

Most relevant papers:

- QDTrack: dense contrastive similarity between same/different instances.
- FairMOT: joint detection and ReID.
- DeepSORT: appearance metric as association backbone.
- SOTMOT / "Improving MOT with SOT": local target-specific discrimination.
- MOTR / TrackFormer / TransTrack / OVTR: persistent queries and assignment over video.

Current V8 has a runtime identity arbitrator in `V8FeatureIdentityArbitrator`, but `train_lorat_v8_head.py` trains only objectness and boxes. There is no same-ID versus different-ID loss.

Recommended changes:

- Add a ReID projection head on top of V8 template/head embeddings or pooled frame features.
- Train with same track ID across frames as positives and other visible tracks as negatives.
- Mine hard negatives from nearby boxes and same-sequence lookalikes, especially DanceTrack crossing cases.
- Report IDF1, ID switches, track-loss rate, and occlusion survival immediately after adding this.

### P0: V8 is not exactly upstream LoRAT inference with multiple objects.

This is not a bad thing, but it must be stated precisely.

Upstream LoRAT is a one-stream template/search tracker that fuses template and search tokens through the transformer. V8's `SharedFrameLoRATEncoder` encodes the current frame once through the LoRAT/DINOv2 ViT path, then a new object-conditioned head scores template memories over that shared feature map.

That means V8 is best described as:

> a LoRAT-backbone shared-frame MOT branch with a new object-conditioned low-rank head

not:

> upstream LoRAT simply running with multiple per-object LoRA heads

Recommended changes:

- Keep the architecture, but be careful in writing and presentations.
- Add an ablation named "upstream-LoRAT-per-track" versus "V8 shared-frame head" so the paper can show what was preserved and what was changed.
- Add a diagram that explicitly shows that LoRAT's original fused SOT head is replaced.

### P1: Training diagnostics look suspiciously insensitive.

The diagnostic CSVs show exactly identical train/val numbers for every epoch after epoch 1, down to many decimal places. For example, `train_L_224_diagnostics.csv` repeats `val_mean_iou=0.1687312583000982` and `val_iou50=0.09740259740259741` for all 12 epochs. The same pattern appears for B-224 and g-224.

This may mean the model converges by epoch 1, but exact repetition across 12 epochs is suspicious. It could also mean the diagnostic set is too small, too deterministic, or not sensitive to the changes being learned.

Recommended changes:

- Add an overfit smoke test on one sequence and 8-16 frame/object samples. It should reach high IoU quickly. If it cannot overfit, the head/loss is wrong.
- Save and evaluate every checkpoint on the same validation slice to confirm metrics change with weights.
- Add direct loss curves per epoch, not only rolling step logs.
- Add a random validation subset option with a fixed seed per epoch group.
- Save "best by validation IoU" separately from "latest."

### P1: The box loss is probably under-constrained.

The current V8 training loss is LoRAT-style BCE objectness plus GIoU on positive cells. That is a reasonable base, but for a fresh head predicting left/top/right/bottom distances over a full frame, GIoU alone can be slow and weak.

Recommended changes:

- Keep GIoU, but add direct positive-cell L1 or SmoothL1 on normalized l/t/r/b distances.
- Add centerness or center sampling so not every interior cell is an equally important positive.
- Try Varifocal/Focal loss instead of plain BCE for the dense class imbalance.
- Extend the IoU-aware classification warmup well beyond 250 steps or make it epoch-based.
- Add hard-negative mining inside nearby distractor boxes from other tracks.

### P1: Identity arbitration is a throughput bottleneck.

The Week 2 architecture succeeds on shared backbone calls, but the profile table shows identity time grows quickly. In the V8 profile, identity time is around 1-4 ms at N=1-2 but grows to roughly 30-40 ms/update at N=5 for B-224/L-224/g-224. That hurts the "up to N objects" claim before the GPU head itself is the bottleneck.

Recommended changes:

- Vectorize identity scoring across tracks/candidates.
- Keep ReID features in a single tensor bank instead of repeatedly scoring Python objects.
- Prune candidates before full identity arbitration.
- Add a benchmark column separating "head compute" from "identity arbitration" from "debug/proof logging."

### P1: The benchmark is too narrow for the onboarding scope.

The current V8 run is a full-sequence run on DanceTrack0065, which is useful, but the summer scope calls for DanceTrack and MOT17, then TAO-OW for open-world.

Recommended changes:

- Add a minimum benchmark matrix: 3 DanceTrack val sequences, 2 MOT17 train sequences, 1 TAO-OW validation subset once open-world starts.
- Add TrackEval integration now so HOTA, DetA, AssA, IDF1, and MOTA are not bolted on at the end.
- Keep quick smoke runs, but label them as smoke runs and do not mix them with headline results.

## Paper-Driven Suggestions

### LoRAT, OSTrack, MixFormer, TransT

These papers support V8's backbone/head direction, but they also warn that target-conditioned tracking usually benefits from template/search training structure. V8 currently uses full-frame shared features, which is good for throughput but harder for localization.

Suggestions:

- Keep SiamFC-like crop jitter in training.
- Add a crop-mode training ablation where the shared frame is a search crop rather than the full video frame.
- Compare full-frame V8, crop-frame V8, and upstream LoRAT-per-track to isolate what quality is lost by splitting LoRAT.

### SOTMOT and Unified Motion/Affinity MOT

These papers say MOT needs target-specific discrimination against nearby targets, not just target-versus-background tracking.

Suggestions:

- Add local distractor negatives from other GT boxes in the same frame.
- Add a "nearby target confusion" metric: when the tracker is wrong, did it choose another annotated object?
- Make uncertainty high when multiple candidate tracks have similar score/motion/ReID values.

### QDTrack, DeepSORT, FairMOT

These are the strongest guidance for the next training step.

Suggestions:

- Add a contrastive ReID loss.
- Store per-track feature banks with timestamps and confidence.
- Evaluate with and without ReID as required by Week 3.
- Report IDF1 and identity switches, not only IoU.

### MOTR, TrackFormer, TransTrack, DETR, Deformable DETR, OVTR

These papers are not one-to-one matches for a user-box-initialized tracker, but their persistent object query idea maps well onto V8's template memory slots.

Suggestions:

- Treat V8 template memories as track queries.
- Train 2-5 frame windows, not independent frames only.
- Include query dropout, stale templates, noisy previous boxes, disappeared objects, and no-object targets.
- Add an unmatched/no-track score so the head can say "uncertain/lost" instead of always emitting a box.

### ByteTrack, BoostTrack, CenterTrack

ByteTrack/BoostTrack are important as baselines, not as the core method. CenterTrack is more directly useful for previous-frame conditioning and motion-offset supervision.

Suggestions:

- Implement ByteTrack/BoostTrack as harness baselines for Week 9.
- Add a CenterTrack-like auxiliary current-center offset loss if V8 boxes drift.
- Add a simple detector/proposal baseline so open-world proposal recall has context.

### OWL-ViT, Video OWL-ViT, OVTrack, RO-ViT, Class-Agnostic Detection, Odin

These support the open-world object discovery branch, not the first V8 tracking-head objective.

Suggestions:

- Keep open-world discovery as a proposal queue feeding the tracker, not as a replacement for tracking.
- Use OWL-ViT or class-agnostic proposal methods for "unknown object" suggestions.
- Use Video OWL-ViT/OVTrack/OVTR for TAO-OW protocol ideas and temporal consistency.
- Use Odin/RO-ViT as background for objectness and region-aware representations, but do not implement them from scratch unless required.

### Shuffle-Then-Assemble

This is the weakest fit. It is about object-agnostic visual relationship features, not MOT head training.

Suggestion:

- Keep it as conceptual support for reducing category bias, but do not center the training plan on it.

### HOTA, MOTChallenge, TAO, Open-World Tracking

These define the evaluation standard.

Suggestions:

- Wire TrackEval now.
- Make HOTA/DetA/AssA/IDF1/MOTA first-class benchmark outputs.
- For TAO-OW, separate known-class tracking, unknown-object proposal recall, and human-correction cost.

## Added Gap-Fill Papers

The missing-topic sweep is now captured in [week3_gap_fill_papers.md](week3_gap_fill_papers.md), and the exact PDF set is registered as `papers-week3-gap-fill` in `manifests/assets.json`.

Added local PDFs for:

- BoostTrack, BoostTrack++, StrongSORT, BoT-SORT, and OC-SORT.
- Grounding DINO, Segment Anything, SAM 2, and Crowd-SAM.
- STM, STCN, XMem, AOT, and DeAOT for video memory/propagation.
- HD-AMOT, Interactive Self-Annotation, Video Annotation and Tracking with Active Learning, VATIC, Efficient Video Annotation, Localization-Aware Active Learning, Plug and Play Active Learning, Extreme Clicking, Snapper, Best of Both Worlds, What Do I Annotate Next, and Polygon-RNN++.

One optional annotation-cost paper, Crowdsourcing Annotations for Visual Object Detection, is citation-only for now because the available Stanford PDF mirror returned an HTML page rather than a valid PDF. The local set is already strong enough for the Week 3-7 human-effort benchmark design.

## Onboarding Alignment

### Week 1

Partially met. The project has LoRAT-backed multi-object GUI paths and benchmarks, but formal MOT17 evaluation and TrackEval/HOTA are still not fully wired. The current V8 quality also means "exercised" should not be read as "solved."

### Week 2

Computationally mostly met. V8 proves one shared frame-backbone call per tracked frame and batched object-head scoring. But the wording should be careful: the head is a new object-conditioned low-rank head, not LoRAT's original per-object LoRA head. Also, 25 FPS capacity on A100 is only N=3 for B-224 and N=2 for L-224/g-224 in the current full run.

### Week 3

Not yet met. Runtime has feature-based identity arbitration, but the required deliverable asks for DINOv2 crop embeddings, track recovery benchmarks, occlusion survival, lost-track indicators, and manual re-anchor cost events. This is the next major milestone.

### Week 4

Not yet met. The paper set is ready for open-world proposal design, but there is not yet a class-agnostic proposal queue or TAO-OW proposal recall benchmark.

### Week 5

Not yet met. V8 has uncertainty-like signals such as confidence, margin, motion, path, and appearance, but there is no active correction queue, no correction cost-to-quality benchmark, and no re-propagation loop.

### Weeks 6-12

Mostly future work. The key risk is that the schedule assumes a reliable tracker before active correction, simulated annotation, hardware sweeps, baselines, ablations, and manuscript writing. The current priority should be to stabilize V8 quality and evaluation before adding many downstream features.

## Concrete Next Changes

1. Add a V8 quality gate and overfit smoke command.
   The head must overfit a tiny sequence slice before more full training runs are trusted.

2. Add direct l/t/r/b regression supervision.
   Keep GIoU but add SmoothL1/L1 on normalized offsets for positive cells.

3. Add QDTrack-style ReID loss.
   Same track across frames is positive; different visible tracks are negatives; nearby tracks are hard negatives.

4. Add sequence-window training.
   Train on short clips with stale templates, missing targets, and noisy previous boxes.

5. Add no-object/uncertain output behavior.
   The model needs to learn when not to trust a box.

6. Vectorize identity arbitration.
   The current Python-heavy identity layer becomes a bottleneck as N grows.

7. Wire TrackEval and MOT17 before expanding the paper claims.
   Current outputs are useful, but paper-ready results need HOTA/IDF1/MOTA over multiple sequences.

8. Split claims into "throughput achieved" and "quality pending."
   This keeps demos honest and makes the next work obvious.

9. Start the Week 3 UI/event logging now.
   Lost-track state, manual re-anchor, and correction-cost events should be logged early because later human-effort benchmarks depend on them.

10. Treat open-world discovery as a proposal module.
   Use class-agnostic/open-vocabulary proposals to spawn tracks; keep tracking quality work separate.

## Suggested V9/V10 Direction

The next version should not be a from-scratch rewrite of everything. It should keep the V8 shared-frame encoder and head batching, but change the training and evaluation spine:

- V9: V8 architecture plus better localization training and ReID contrastive loss.
- V10: V9 plus sequence windows, no-object/lost-target training, and active-correction event logging.
- Open-world proposal branch: separate module using OWL-ViT/class-agnostic proposals, attached after V9 is stable.

This fits the paper set better than a purely larger V8 head or another runtime-only optimization pass.
