# SOTMOT Notes for LoRAT-Based MOT

Paper: [Improving Multiple Object Tracking With Single Object Tracking](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.html), Zheng et al., CVPR 2021.

Local PDF: `papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf`

## Core Idea

SOTMOT argues that directly running a full independent SOT tracker for every MOT target is both inefficient and not quite the right discrimination problem. A normal SOT tracker learns target-versus-background discrimination. MOT needs target-versus-nearby-target discrimination because the difficult identity errors happen when similar objects cross, occlude, or move close together.

Their solution is not "N separate heavy trackers." It extends a detector with a lightweight SOT branch. For each target, the SOT branch trains an online ridge-regression model against neighboring detected foreground objects, then uses those models as association scores in a DeepSORT/FairMOT-style online tracker.

## What We Should Borrow

1. Treat LoRAT outputs as proposals, not as final identities.

   Our current v3/v4 direction is correct here. LoRAT should produce candidate boxes per selected object, but a shared MOT coordinator should own final ID assignment, rejection, coasting, and correction requests.

2. Score each target against nearby competitors.

   SOTMOT's strongest insight is "specific discrimination": target versus surrounding targets. For LoRAT, the practical wrapper-level version is to add a neighborhood candidate set around each predicted track and score LoRAT proposals, memory proposals, detector/proposal boxes, and nearby competing track boxes against that target.

3. Use motion and IoU as first-class association signals.

   SOTMOT uses Kalman prediction, motion information, IoU, and Hungarian matching around the SOT scores. Our coordinator already has motion, IoU, trajectory, bottom-edge, direction, and appearance terms. The next improvement is to make that association explicitly two-stage: first LoRAT/appearance/motion matching, then IoU fallback for unmatched tracks and candidates.

4. Batch multi-object work.

   SOTMOT gets efficiency because many small per-target models can be trained/scored in a batch on GPU. Our `--track-batch-size` does chunked LoRAT task execution, but it still runs multiple LoRAT tracker tasks rather than sharing one current-frame feature computation. This is fine for the week-one working tool; the research upgrade is shared-frame LoRAT inference for all active templates.

5. Maintain a sample pool, but update it conservatively.

   SOTMOT updates each target model with moving-average sample history. Our v4 memory slots approximate this by keeping initial, recent, and diverse LoRAT slots. That is useful as a wrapper experiment, but the cleaner design is a per-track memory bank that updates only on high-confidence, high-margin, non-occluded frames.

6. Make uncertainty a first-class output.

   The paper's association structure gives natural uncertainty signals: low best score, low assignment margin, conflicting nearby tracks, weak motion agreement, and low confidence. These map directly to our active-correction loop.

## What We Should Not Copy Directly

- SOTMOT depends on a class detector, specifically a CenterNet pedestrian-style MOT setup. Our final tool must track nameless unknown objects, so detections/proposals must come from user boxes, objectness/open-world discovery, segmentation proposals, or optional class detectors.
- Their SOT model is a differentiable ridge-regression branch trained with foreground negatives. LoRAT is a transformer SOT tracker with its own template/search pipeline. We can emulate SOTMOT at the coordinator layer first, but a true architectural version would require modifying LoRAT internals.
- SOTMOT reports axis-aligned MOT boxes. Rotated boxes can be a GUI/refinement feature for us, but MOT17/DanceTrack/TrackEval should stay axis-aligned for standard metrics.

## Current Code Mapping

- `programs/bounding_box_v3_lorat.py` already follows the right high-level shape: one LoRAT task per selected object, then a global coordinator uses Hungarian assignment, motion, IoU, object-agnostic appearance, trajectory guards, and memory recovery.
- `programs/bounding_box_v4_lorat_memory.py` explores a stronger memory path: each visible track can have initial/recent/diverse LoRAT slots. This is not SOTMOT's ridge-regression sample pool, but it is a good ablation before patching LoRAT.
- `programs/exercise_lorat_mot.py` and `programs/benchmark_lorat_mot.py` currently exercise v3. To evaluate the SOTMOT-inspired v4 memory idea properly, the benchmark should be able to choose v3 versus v4.

## Recommended Next Implementation Path

1. Make v4 benchmarkable.

   Add a tracker-module or tracker-version option to the exercise and benchmark scripts so the same DanceTrack/MOT17 seeds can run `v3`, `v4 --lorat-memory-slots 1`, and `v4 --lorat-memory-slots 3`.

2. Add a SOTMOT-style association mode.

   For each track, build a local neighborhood around its predicted center. Score only nearby candidates first, then run Hungarian matching. Keep a second IoU fallback pass for unmatched tracks/candidates.

3. Add explicit active-correction flags.

   Emit a correction request when assignment margin is low, best candidate is contested by a neighboring track, LoRAT confidence is low, the object falls below the small-area reliability threshold, or the accepted box required a guard/resync.

4. Add a shared-frame LoRAT research branch.

   Once the wrapper is benchmarked, inspect LoRAT internals for a way to encode the current frame once and score multiple templates/tasks from that shared feature map. This is the true SOTMOT efficiency lesson applied to LoRAT.

5. Add open-world proposal sources later.

   For unknown-object discovery, unmatched proposals should come from objectness/segmentation/open-world proposal modules, not a person detector. Human correction should decide whether a proposal becomes a new tracked object.

## Benchmark Implications

- Item (b), time per box: report scaling for v3, v4 one-slot, and v4 three-slot runs. If v4 improves ID stability but costs more per box, the benchmark will make the tradeoff visible.
- Item (c), small-object reliability: include uncertainty-trigger rate per area bin, not only IoU. Small objects may still produce boxes while becoming too uncertain for low-human-effort labeling.
- Item (d), LoRAT model sizes: run B/L/g with the same association settings first. Then run the SOTMOT-inspired ablation separately so model-size comparisons do not get confounded.

## Useful Reading Hooks

- Specific discrimination: target versus neighboring targets, not target versus arbitrary background.
- Online inference: SOT scores combine with Kalman/motion, IoU, and Hungarian matching.
- Efficiency: many lightweight per-target models are solved/scored in batch on GPU.
- Supplement: their ID-switch discussion says sparse scenes can hurt because new tracks have too few nearby samples; adapted-size neighborhoods are a likely fix.
