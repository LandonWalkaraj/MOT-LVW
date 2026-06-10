# Week 3 Gap-Fill Paper Catalog

Date: 2026-06-10

Purpose: fill the missing paper/topics list from the V8 review with papers that are useful for Week 3 ReID and recovery, plus the nearby Week 4-7 work on open-world proposals, active correction, and human-effort benchmarks.

## Best First Reads

1. StrongSORT, BoT-SORT, BoostTrack, and BoostTrack++.
   Read these as association and ReID baselines. They are the closest practical references for "ReID is the backbone" in a tracker that still needs motion and confidence checks.

2. OC-SORT.
   Read this as the motion-only counterweight. It is useful because DanceTrack stresses similar appearance; if ReID is weak or misleading, OC-SORT-style observation-centric motion gives a clean baseline.

3. XMem, STCN, STM, AOT, and DeAOT.
   Read these for memory design, not because we should switch to mask tracking immediately. Their memory-bank and propagation logic maps well to "human corrected one frame, now repair nearby frames."

4. Efficient Video Annotation, Video Annotation and Tracking with Active Learning, VATIC, and Interactive Self-Annotation.
   These are the most direct references for the final benchmark claim: how much human effort is needed to reach target labeling quality.

5. Grounding DINO, SAM, SAM 2, and Crowd-SAM.
   These should feed the open-world proposal and correction-assist branch. They should not block Week 3 ReID/recovery.

## What This Adds To The Plan

The project now has a clearer split:

- Week 3 core: ReID, lost-track recovery, ID-switch and occlusion benchmarks.
- Week 4 open-world branch: Grounding DINO/SAM/SAM 2 style proposals feed candidate unknown objects into the tracker.
- Week 5 active correction: uncertainty-ranked frames use VOS memory and active annotation papers as the design base.
- Week 7 human-effort benchmark: report wall-clock time, number of boxes drawn, number of correction/reanchor events, and target-quality curves.

The strongest implementation direction is still to keep V8's shared-frame LoRAT backbone, add a contrastive ReID head, and treat object discovery/correction as modules around the tracker.

## MOT And ReID Baselines

| Paper | Local PDF | Why It Is Useful | Use In This Project |
| --- | --- | --- | --- |
| BoostTrack | `papers/boosttrack_similarity_confidence_mot.pdf` | Improves association by combining similarity and detection confidence. | Baseline and inspiration for confidence-weighted ReID/motion association. |
| BoostTrack++ | `papers/boosttrack_plus_plus_tracklet_information_mot.pdf` | Adds tracklet-level information and stronger association context. | Later baseline for Week 9, plus ideas for using longer track history after occlusion. |
| StrongSORT | `papers/strongsort_make_deepsort_great_again.pdf` | Modernizes DeepSORT with stronger appearance, motion, and post-processing. | Reference for appearance memory and camera/motion compensation style association. |
| BoT-SORT | `papers/bot_sort_robust_associations_multipedestrian_tracking.pdf` | Strong practical tracker using appearance and motion association. | ReID baseline and sanity check for our identity arbitration. |
| OC-SORT | `papers/oc_sort_observation_centric_sort.pdf` | Strong motion-centric tracker that does not rely on appearance as heavily. | Baseline for "does ReID actually help?" and fallback logic when appearance is ambiguous. |

## Open-World Proposal And Correction Assist

| Paper | Local PDF | Why It Is Useful | Use In This Project |
| --- | --- | --- | --- |
| Grounding DINO | `papers/grounding_dino.pdf` | Open-set phrase-grounded detection with strong proposal behavior. | Optional proposal source for known/unknown object candidates. |
| Segment Anything | `papers/segment_anything.pdf` | Promptable segmentation foundation model. | Convert user boxes or proposal boxes into tighter masks/boxes during correction. |
| SAM 2 | `papers/sam2_segment_anything_images_videos.pdf` | Extends SAM-style prompting to images and videos. | Video-aware proposal/correction propagation reference for Week 4+. |
| Crowd-SAM | `papers/sam_smart_annotator_object_detection_crowded_scenes.pdf` | Adapts SAM to crowded/occluded scenes. | Useful caution for DanceTrack/MOT17 crowd cases where vanilla SAM may merge people. |

## Video Memory And Propagation

| Paper | Local PDF | Why It Is Useful | Use In This Project |
| --- | --- | --- | --- |
| STM | `papers/stm_video_object_segmentation_space_time_memory.pdf` | Classic space-time memory for VOS. | Reference for a correction memory bank that can repair future/past frames. |
| STCN | `papers/stcn_rethinking_space_time_networks_vos.pdf` | Efficient memory correspondence without per-object re-encoding. | Good fit for multi-object shared-frame thinking. |
| XMem | `papers/xmem_long_term_video_object_segmentation.pdf` | Long-term, mid-term, and short-term memory design. | Most useful VOS memory paper for long videos and occlusion recovery. |
| AOT | `papers/aot_associating_objects_with_transformers_vos.pdf` | Transformer-based multi-object association in VOS. | Helps reason about multiple object memories competing in the same frame. |
| DeAOT | `papers/deaot_decoupling_features_hierarchical_propagation_vos.pdf` | Decouples object-agnostic and object-specific features. | Conceptually close to LoRAT backbone plus object-specific head/memory. |

## Active Annotation And Human Effort

| Paper | Local PDF | Why It Is Useful | Use In This Project |
| --- | --- | --- | --- |
| HD-AMOT | `papers/hdamot_active_learning_multi_object_tracking.pdf` | Active learning specifically for MOT frame selection. | Direct reference for uncertainty/diversity sampling over video frames. |
| Interactive Self-Annotation | `papers/interactive_self_annotation_video_object_bounding_box.pdf` | Human-in-the-loop video box annotation with recurrent correction. | Direct precedent for our manual reanchor and correction loop. |
| Video Annotation and Tracking with Active Learning | `papers/video_annotation_tracking_active_learning.pdf` | Uses active frame selection to reduce manual effort for tracks. | Core citation for "ask only uncertain frames." |
| VATIC | `papers/vatic_efficient_crowdsourced_video_annotation.pdf` | Classic large-scale video annotation tool and cost study. | Benchmark design for real annotation time, QA, and UI costs. |
| Efficient Video Annotation | `papers/efficient_video_annotation_visual_interpolation_frame_selection.pdf` | Interpolation/extrapolation plus frame selection for boxes. | Strongest direct paper for measuring boxes drawn versus quality. |
| Localization-Aware Active Learning | `papers/localization_aware_active_learning_object_detection.pdf` | Active learning scores include localization tightness/stability. | Use localization instability as an uncertainty term in active correction. |
| Plug and Play Active Learning | `papers/plug_and_play_active_learning_object_detection.pdf` | Detector-agnostic uncertainty/diversity active learning. | Useful if we later rank proposal frames without changing the tracker. |
| Extreme Clicking | `papers/extreme_clicking_efficient_object_annotation.pdf` | Measures box annotation alternatives and time per object. | Human-effort baseline for single-object box entry cost. |
| Snapper | `papers/snapper_accelerating_bounding_box_annotation.pdf` | Interactive snapping for faster bounding-box creation. | UI idea for reducing manual correction time. |
| Best of Both Worlds | `papers/best_of_both_worlds_human_machine_object_annotation.pdf` | Human-machine collaboration for object annotation. | Higher-level framework for combining model proposals and human verification. |
| What Do I Annotate Next? | `papers/what_do_i_annotate_next_active_learning_video.pdf` | Empirical active learning for selecting video annotations. | Helps justify frame/query prioritization. |
| Polygon-RNN++ | `papers/polygon_rnn_plus_plus_interactive_annotation.pdf` | Interactive segmentation annotation, starting from object crops. | Optional reference for future mask-to-box correction assistance. |

## Citation-Only Or Optional

- Crowdsourcing Annotations for Visual Object Detection is useful for annotation-cost framing, but the Stanford PDF mirror returned an HTML page during download. We have enough local cost papers for now, so this is optional unless we need a larger annotation-cost section later.
- EVA-VOS and newer VOS active annotation papers may be useful later if the tool expands from boxes to masks. They are not needed for Week 3's bounding-box ReID/recovery deliverable.

## Implementation Implications

- ReID should be trained, not only hand-coded. Add a projection head and contrastive loss with same-track positives, different-track negatives, and nearby-track hard negatives.
- Association should be ablated against motion-only and ReID-assisted modes. OC-SORT and BoT-SORT make this comparison easy to explain.
- Lost-track recovery should maintain memory at multiple horizons. XMem is the best conceptual reference for short/mid/long memory.
- Active correction should log cost events from the start: initial box, correction box, reanchor, verification-only frame, and skipped uncertain frame.
- Open-world discovery should remain a proposal queue feeding the tracker. Grounding DINO/SAM/SAM 2 are proposal/correction tools, not replacements for the LoRAT MOT tracker.

