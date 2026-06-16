# Core Training Dataset Candidates

Date: 2026-06-11

Purpose: identify additional datasets that could help train the LoRAT-based multi-object video labeler. This is focused on practical usefulness for our system, not just famous benchmark names.

The project needs several different kinds of training signal:

- SOT-style target localization: given a first box/template, find the same object later.
- MOT identity association: keep the same ID across multiple nearby objects.
- ReID/recovery: reattach a lost object after occlusion or disappearance.
- Open-world proposals: suggest unnamed objects without fixed class labels.
- Human-effort benchmarking: measure correction cost versus final label quality.

No single dataset solves all of that. The best training mix is layered.

## Cross-Reference To Summer Onboarding Goals

Source: `outputs/summer_onboarding_extracted.txt`.

The onboarding statement of work defines the final tool as:

- a user-initialized box labeler for objects with no required class name;
- a multi-object extension of LoRAT;
- appearance-based re-identification;
- open-world object discovery;
- an active-correction loop;
- benchmarks that measure human effort required to reach target quality;
- a final paper with reproducible results.

That means dataset choices should be judged by whether they help one of the scheduled deliverables, not by whether they are broadly popular.

| Onboarding Goal | Required / Best-Match Datasets | Dataset Implication |
| --- | --- | --- |
| Week 1: LoRAT MOT GUI, timing, small-object size, model-size comparison | DanceTrack, MOT17 | These remain the benchmark-aligned core. Do not replace them with extra datasets when reporting Week 1/2 claims. |
| Week 2: shared backbone throughput against object count and memory | DanceTrack, MOT17 | No extra dataset is required. The important variable is object count, model size, and hardware. |
| Week 3: DINOv2 ReID, identity switches, track loss, occlusion survival, manual re-anchor cost | DanceTrack, MOT17, plus MOT20 or SportsMOT; LaSOT/OVIS for occlusion stress | This is the strongest reason to add hard identity datasets now. Prioritize same-frame negatives, lookalikes, occlusion gaps, and reappearance cases. |
| Week 4: open-world object discovery and proposal recall on TAO-OW and DanceTrack | TAO-OW, TAO, BURST, COCO/LVIS/Objects365 as proposal support | TAO-OW is mandatory for the open-world claim. BURST helps because masks can become boxes. Static datasets help proposal/objectness pretraining only. |
| Week 5: uncertainty-ranked active correction and cost-to-quality | DanceTrack, MOT17, TAO-OW/BURST | Use datasets with reliable ground truth so a simulated annotator can count corrections and quality gain. |
| Week 6: bidirectional propagation, smoothing, MOT and COCO-video export | DanceTrack, MOT17, TAO/BURST, YouTube-VIS/VOS as optional format/propagation practice | Do not let export support dictate early training. Use mask datasets later if correction propagation needs mask-to-box supervision. |
| Week 7: complete benchmark harness across DanceTrack, MOT17, TAO-OW | DanceTrack, MOT17, TAO-OW | The harness should regenerate all headline numbers on the designated datasets before adding more. |
| Week 8: LoRAT model-size and hardware sweep | Same benchmark suite from Weeks 1-7 | Extra datasets should not enter the main model-size frontier unless the earlier required benchmarks are already stable. |
| Week 9: ByteTrack/BoostTrack baselines and lab footage | DanceTrack, MOT17, TAO-OW, lab footage | Baselines need the same benchmark data and metrics as our tracker. Lab footage is for failure categorization, not training. |
| Week 10: robustness across blur, low light, crowding, small objects | MOT20, BDD100K, VisDrone, UAVDT, CrowdHuman as optional stress/support | These are useful robustness stress sets, but they belong after the main tracker and harness are working. |
| Weeks 11-12: manuscript, figures, reproducibility | Frozen benchmark suite from Weeks 1-10 | The paper should report a stable, reproducible dataset matrix instead of a late-expanded one. |

Cross-referenced conclusion:

1. For the immediate Week 3 overhaul, add only a small amount of new data: one SOT recovery source and one hard MOT/ReID source.
2. For Week 4, make TAO-OW/TAO the first open-world dataset; use BURST if mask-to-box conversion is manageable.
3. Treat large static proposal datasets as support data, not headline tracking datasets.
4. Keep VOTS2025, VisDrone, UAVDT, BDD100K, LVIS, Objects365, and SA-1B as later stress/proposal/evaluation resources unless a specific deliverable needs them.

## Short Recommendation

Across the whole summer, use this as the serious training/evaluation stack:

1. DanceTrack and MOT17.
   Already required. Keep these as the benchmark-aligned core.

2. GOT-10k, LaSOT, TrackingNet, and COCO.
   These match LoRAT-style SOT training and should help target-conditioned localization. LoRAT itself uses this family of data.

3. MOT20, PersonPath22, and SportsMOT.
   These strengthen identity association beyond DanceTrack/MOT17: crowded people, large-scale person tracks, fast similar-player motion.

4. TAO and BURST.
   These are the best fit for open-world/general-object tracking. TAO gives "tracking any object" boxes; BURST adds high-quality masks on diverse object tracks.

5. YouTube-VIS, YouTube-VOS, and OVIS.
   These are not box-MOT datasets first, but masks can be converted to boxes. They are useful for occlusion/recovery and future correction propagation.

Use VOTS2025 mostly for evaluation/integration, not core supervised training, because the main benchmark does not expose full ground truth locally.

## Tier 1: Highest Value Additions

| Dataset | Best Use | Why It Fits | Main Caveat |
| --- | --- | --- | --- |
| GOT-10k | SOT localization and class-general tracking | Large generic object tracking with one-shot protocol and diverse object classes/motion. | Single-object style, not MOT identity association. |
| LaSOT | Long-term SOT, disappearance/reappearance | Dense boxes, long sequences, diverse classes, good for recovery behavior. | Single target per sequence. |
| TrackingNet | Large-scale SOT training | Very large number of videos and dense boxes; strong for deep tracker training. | Large download; SOT, not MOT. |
| COCO | Static object crops for target/head pretraining | LoRAT-style training commonly uses detection images as object crops; useful for objectness and selected-region augmentation. | No temporal identity. |
| MOT20 | Crowded pedestrian MOT | Stress-tests occlusion/crowding beyond MOT17. | Small number of sequences, pedestrian-only. |
| PersonPath22 | Large-scale multi-person ReID/MOT | Much larger than MOT17/MOT20, box + track IDs in real videos. | Person-only; data access/storage planning needed. |
| SportsMOT | Similar appearance + fast motion | Players in uniforms are good hard negatives for ReID and motion. | Sports-specific; mostly player tracking. |
| TAO | Open-world/general-object tracking | Many categories and videos; tracks "any object" with post-hoc category naming. | Sparse/federated annotations require careful sampling. |
| BURST | General object masks/tracks | Built around unified recognition, segmentation, and tracking in video; masks can become boxes. | Segmentation format conversion required. |

## Tier 2: Useful After The Core Stack

| Dataset | Best Use | Why It Fits | Main Caveat |
| --- | --- | --- | --- |
| YouTube-VIS | General video instance tracking via masks | Multi-instance video masks and object categories; convert masks to boxes. | VIS protocol, not user-initialized box tracking. |
| YouTube-VOS | Semi-supervised video object segmentation | First-frame object propagation resembles user-initialized tracking. | Mask-centric; not MOT box benchmark. |
| OVIS | Occlusion-heavy recovery | Designed around severe occlusions; good for recovery and lost-track logic. | Mask/VIS format; category set is limited. |
| BDD100K MOT | Multi-category driving MOT | Weather/time-of-day/generalization and vehicle/person categories. | Driving domain; not generic user-object tracking. |
| GMOT-40 | Generic one-shot MOT concept | Very aligned with "generic multiple object tracking" and one-shot protocol. | Small: better for validation/analysis than large training. |
| CrowdHuman | Detection/proposal pretraining | Very strong for crowded person boxes and occlusion. | Static images, no track IDs. |
| VastTrack | Large general SOT | Huge category coverage and videos; potentially excellent for general SOT pretraining. | Huge scale and newer access path; not MOT identity. |

## Tier 3: Specialized Or Later

| Dataset | Best Use | Why It Fits | Main Caveat |
| --- | --- | --- | --- |
| VisDrone MOT | Small aerial objects | Good for small-object and camera-motion robustness. | Domain shift from current DanceTrack/MOT17 goals. |
| UAVDT | Aerial vehicle DET/SOT/MOT | Useful for small vehicles and aerial motion. | Vehicle/aerial domain. |
| VOTS2025 dev / VOTSt val | Tracker integration and SOT/MOTS evaluation | Matches general target tracking and disappear/reappear behavior. | Main VOTS2025 GT is server-side; dev set is small. |
| LVIS | Open-world proposal/objectness | Long-tail categories and masks on images. | Static images, no temporal tracks. |
| Objects365 | Broad object detector/proposal pretraining | Many detection categories and boxes. | Static images; very large. |
| SA-1B | Promptable segmentation/proposal research | Huge open-world mask data. | Enormous; no temporal identity; likely overkill for this summer. |
| TNL2K | Language-guided tracking | Useful if the tool later accepts text prompts. | Not core for box-initialized tracking. |

## Dataset-To-Training-Objective Map

| Objective | Best Datasets | Training Use |
| --- | --- | --- |
| V8/V9 target-conditioned localization | GOT-10k, LaSOT, TrackingNet, COCO, DanceTrack crops | Template/search pairs, score maps, box regression, selected-region augmentation. |
| Multi-object ReID contrastive loss | DanceTrack, MOT17, MOT20, PersonPath22, SportsMOT | Same-track positives, same-frame negatives, nearby hard negatives. |
| Lost-track recovery | LaSOT, OVIS, YouTube-VOS, BURST, TAO, DanceTrack occlusion cases | Artificial gaps, stale templates, reappearance positives, recovery diagnostics. |
| Open-world proposal training | TAO, BURST, COCO, LVIS, Objects365, SA-1B | Objectness/proposal recall, mask-to-box conversion, unknown-object suggestions. |
| Small-object reliability | DanceTrack/MOT17 bins, VisDrone, UAVDT, BDD100K | Size bins, scale augmentation, small-object failure analysis. |
| Human-effort benchmark | DanceTrack, MOT17, TAO/BURST | Simulated annotator, cost-to-quality, correction event replay. |

## Practical Training Mix

### Phase 1: Fix The Current V8/V9 Head

Use:

- DanceTrack train/val.
- MOT17 train split.
- A small sampled subset of GOT-10k or LaSOT.

Goal:

- Prove localization quality.
- Pass an overfit smoke test.
- Add ReID positives/negatives from DanceTrack and MOT17.

Why not bigger yet:

- If the head cannot overfit a tiny slice, more data will only hide the bug.

### Phase 2: ReID And Recovery Scale-Up

Add:

- MOT20.
- PersonPath22 if storage/access is manageable.
- SportsMOT.
- More LaSOT long-term sequences.

Goal:

- Reduce identity switches.
- Improve hard-negative handling.
- Benchmark occlusion survival.

### Phase 3: Open-World General Objects

Add:

- TAO.
- BURST.
- YouTube-VIS or YouTube-VOS.

Goal:

- General objects beyond people.
- Mask-to-box proposal/correction paths.
- Open-world proposal recall.

### Phase 4: Proposal/Detector Support

Add selectively:

- COCO.
- LVIS.
- Objects365.
- CrowdHuman for person-heavy crowd proposals.

Goal:

- Proposal queue.
- Class-agnostic/objectness training.
- Better initial candidate boxes.

## Notes On Specific Datasets

### GOT-10k

Strong fit for LoRAT-style SOT training. It has many generic moving-object classes and a one-shot tracking protocol. Use it to train "given this target, find this target later" behavior.

### LaSOT

Strong fit for long-term tracking. Long videos and disappear/reappear cases make it especially useful for lost-track recovery and conservative memory update policies.

### TrackingNet

Good high-volume SOT training data. Useful once the training pipeline is stable, but storage and preprocessing may be non-trivial.

### VastTrack

Potentially excellent for general SOT because it has many categories and many sequences. Treat it as a scale-up candidate, not the first dataset to add.

### DanceTrack

Still central. It was designed to stress association when people have similar appearance and diverse motion. It is one of the best datasets for proving ReID/motion association actually helps.

### MOT17 / MOT20

MOT17 stays the standard baseline. MOT20 adds dense crowds, which is useful for occlusion and target stealing.

### PersonPath22

Very attractive if we want much more multi-person identity data. It has boxes and track IDs in each frame and is much larger than MOT17/MOT20. Good for ReID and track-loss training, but it is person-only.

### SportsMOT

Useful because players look similar and move quickly. It gives hard negatives similar to DanceTrack but in sports scenes.

### TAO

Best match for "tracking any object" and open-world generalization. However, its annotations are sparse/federated, so use it carefully for open-world evaluation and proposal/recovery rather than naive dense frame-by-frame training.

### BURST

Very useful for general object masks/tracks. Convert masks to boxes for the current labeler, and keep masks available for future correction/snap/proposal work.

### YouTube-VOS / YouTube-VIS / OVIS

Useful when the project moves toward correction propagation and occlusion recovery. They are mask-first datasets, but boxes can be derived from masks.

### CrowdHuman

Not an MOT dataset, but very useful for crowded human detection/proposal pretraining. Do not use it for ReID unless pairing/pseudo-labeling is added.

## What Not To Treat As Core Supervised Training

- VOTS2025 main benchmark: good evaluation/integration target, but full ground truth is server-side.
- VOTSt2025 validation: useful dev data, but not enough to be a main training source.
- LVIS/Objects365/SA-1B: excellent for proposals/objectness, but static image datasets do not provide temporal identity.
- TNL2K: interesting for future language-guided tracking, not needed for the current box-initialized tracker.

## Immediate Actionable Ranking

If we add only three things next:

1. GOT-10k or LaSOT subset.
   Helps the V8 head learn target-conditioned localization in the same spirit as LoRAT.

2. MOT20 or SportsMOT.
   Adds harder association/recovery cases than MOT17 alone.

3. TAO subset.
   Starts the open-world path without waiting until Week 4.

If storage and time allow a fourth:

4. PersonPath22.
   Best scale-up for multi-person ID training.
