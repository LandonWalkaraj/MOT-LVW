# V8 LoRAT-MOT Head Training Research

Date: 2026-06-10

This note answers the practical question for the current V8 branch: if we keep the LoRAT ViT backbone and replace LoRAT's original single-object tracking head with an object-conditioned MOT-style head, what published training recipes apply?

## Current V8 Context

V8 is no longer just one LoRAT single-object tracker per user box. It has a new architecture branch:

- `programs/bounding_box_v8_lorat_quality_batched.py` encodes the current frame once through LoRAT's DINOv2/ViT path in `SharedFrameLoRATEncoder`.
- `BatchedObjectConditionedHead` scores multiple object memories against that shared frame feature map.
- The head predicts a dense score map plus box deltas for each tracked object.
- The object conditioning comes from template/memory slots and a generated low-rank projection, so the head is LoRA-like but distinct from upstream LoRAT's original fused template/search path.
- `programs/train_lorat_v8_head.py` freezes the LoRAT frame encoder and trains only this V8 object-conditioned head.

The current V8 training script already follows the most important part of LoRAT's own loss: positive cells over the target box, BCE objectness, IoU-aware positive labels after warmup, and GIoU box regression on positive cells.

## How LoRAT Trained

LoRAT's official configs are the closest direct reference:

- Backbone: DINOv2 ViT-B/L/g with LoRA adapters.
- Task: Siamese generic object tracking with template/search pairs.
- Training data: GOT-10k, LaSOT, TrackingNet, and COCO sampled with equal weights.
- Crop recipe: SiamFC-style template and search crops, template area factor 2, search area factor 4, scale jitter 0.25, translation jitter 3, and minimum object size 10 pixels.
- Augmentation: horizontal flip, color jitter, and DeiT-style augmentation.
- Optimizer: AdamW, learning rate 1e-4, weight decay 0.1, cosine schedule, warmup, AMP fp16.
- Scale: 170 epochs, global batch size 128, 131072 samples per epoch.
- Loss: `box_with_score_map`, BCE score-map classification with IoU-aware positive labels, plus GIoU box regression.

Local references:

- `external/LoRAT-main/config/LoRAT/run.yaml`
- `external/LoRAT-main/config/LoRAT/B-224/config.yaml`
- `external/LoRAT-main/trackit/criteria/methods/box_with_score_map/__init__.py`

Implication for V8: keep the LoRAT-style dense score/box objective as the base objective. The architectural difference is that V8 trains a shared-frame, object-conditioned head instead of the original one-stream fused SOT head.

## Did An Exact Method Exist?

I did not find a paper that exactly says: "take LoRAT, freeze its ViT backbone, replace its SOT head with a multi-object target-conditioned head, and train that head on MOT data."

What does exist is a strong set of adjacent training methods. The V8 training plan should be presented as a synthesis of these:

| Method family | Most relevant papers | Training idea | V8 applicability |
| --- | --- | --- | --- |
| LoRAT and one-stream SOT | LoRAT, OSTrack, MixFormer, TransT | Train a target-conditioned tracker with dense response maps and box regression from template/search pairs. | Directly applicable for the score/box loss, crop jitter, and staged freezing/unfreezing. |
| Query-based MOT | MOTR, TrackFormer, TransTrack, OVTR | Carry persistent object queries across frames and train them with detection/track assignment over video. | Very relevant conceptually. V8 template memories are our track queries; add sequence-window training, query dropout, missed-object cases, and identity continuity losses. |
| MOT appearance/ReID training | QDTrack, FairMOT, DeepSORT, CenterTrack | Train association embeddings, dense contrastive similarity, joint detection/ReID, or previous-frame offset heads. | This should be the next major V8 addition. Add a contrastive ReID loss over V8 template/head embeddings and hard negatives from nearby tracks. |
| Open-world and object-agnostic localization | OWL-ViT, Video OWL-ViT, OVTrack, class-agnostic detection, RO-ViT, Odin | Learn objectness or open-vocabulary localization without a fixed closed class set. | Applicable to discovery/proposal generation and TAO-OW, but not a replacement for target-conditioned tracking loss. |
| Detection loss design | VarifocalNet, GIoU, DETR, Deformable DETR | IoU-aware classification, box losses, and set prediction/assignment. | Useful for improving V8's dense head labels and future proposal/object-discovery branch. |
| Object-agnostic visual relationship pretraining | Shuffle-Then-Assemble | Reduce object-category bias in visual relationship features. | Weakly applicable. It supports the idea of category-bias reduction, but it is not a direct recipe for MOT head training. |

## Object-Agnostic Vision Training Judgment

The phrase "Object-Agnostic Vision Training" does not appear to refer to one canonical MOT-head training method. The closest paper already collected, "Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features," is about relationship features, not target-conditioned box tracking. It should not be treated as the main V8 training recipe.

For our project, "object-agnostic" should mean:

- The tracker follows an instance selected by a user box, not a named class.
- Training supervision should not require object names.
- The head should learn "same selected thing in this frame" versus distractors, not "person/car/dog."
- Open-world discovery should produce class-agnostic object proposals or one-shot/open-vocabulary candidates that the user can accept.

Better paper support for that part comes from:

- Class-agnostic object detection: objectness independent of known classes.
- OWL-ViT: one-shot image-conditioned and text-conditioned open-vocabulary detection.
- Video OWL-ViT: recurrent object representations across video for TAO-OW-like localization.
- OVTrack and OVTR: open-vocabulary MOT training and association.
- Odin: self-supervised object discovery and representation learning.

The current V8 `--target-region-mode mixed` option is a useful local approximation because it trains the head on full objects and selected sub-regions without category labels. I would describe it as "class-agnostic selected-region augmentation," not as a complete object-agnostic pretraining method.

## Recommended V8 Training Plan

1. Keep the current head-only stage.
   Freeze the LoRAT ViT path and train only `BatchedObjectConditionedHead` with LoRAT-style BCE plus GIoU. First prove overfit on one DanceTrack sequence, then train DanceTrack train and validate on DanceTrack val.

2. Add QDTrack-style contrastive ReID.
   For every training frame pair, use same track ID across frames as positives and other visible tracks as negatives. Mine hard negatives from nearby boxes and similar-looking tracks. Apply the loss to template vectors, head hidden summaries, or both.

3. Add sequence-window MOT training.
   Move from independent frame samples to 2-5 frame windows. Randomly stale or drop templates, include occlusion/missing labels, and force the head/memory bank to recover identity after appearance change.

4. Add TrackFormer/MOTR-style query robustness.
   Treat each V8 memory slot as a track query. Train with query dropout, noisy previous boxes, false positive memories, and unmatched-track cases so the head learns uncertainty instead of always emitting a box.

5. Add CenterTrack-style motion auxiliary loss if needed.
   V8 already uses previous boxes to define search regions. If motion remains unstable, add a small offset head or auxiliary loss that predicts current center from previous center.

6. Add open-world proposal/discovery after the tracker is stable.
   Use OWL-ViT/Video OWL-ViT/OVTrack/RO-ViT/Odin as references for proposal generation and TAO-OW discovery. Keep this as a side branch feeding candidates into the active-correction loop, not as the first V8 head objective.

7. Only then unfreeze backbone adapters.
   If head-only training saturates, unfreeze LoRA/adapters in upper ViT blocks with a much smaller learning rate. Do not full-finetune the backbone until the head loss, ReID loss, and evaluation protocol are stable.

## Papers Added For This Question

Core LoRAT/head training:

- `papers/lorat_tracking_meets_lora.pdf`
- `papers/lorat_supplemental_training_details.pdf`
- `papers/ostrack_one_stream_tracking.pdf`
- `papers/mixformer_end_to_end_tracking.pdf`
- `papers/transt_transformer_tracking.pdf`

MOT head, query, and association training:

- `papers/motr_end_to_end_multi_object_tracking_transformer.pdf`
- `papers/trackformer_multi_object_tracking_transformers.pdf`
- `papers/transtrack_multiple_object_tracking_transformer.pdf`
- `papers/ovtr_end_to_end_open_vocabulary_mot.pdf`
- `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf`
- `papers/fairmot_detection_reid_mot.pdf`
- `papers/centertrack_tracking_objects_as_points.pdf`

Open-world, class-agnostic, and object discovery:

- `papers/owlvit_simple_open_vocabulary_detection_vit.pdf`
- `papers/video_owlvit_open_world_video_localization.pdf`
- `papers/ovtrack_open_vocabulary_mot.pdf`
- `papers/class_agnostic_object_detection_wacv2021.pdf`
- `papers/class_agnostic_object_detection_multimodal_transformer.pdf`
- `papers/ro_vit_region_aware_pretraining_open_vocabulary_detection.pdf`
- `papers/object_discovery_and_representation_networks.pdf`
- `papers/shuffle_then_assemble_object_agnostic_visual_relationship_features.pdf`

Losses and backbone foundations:

- `papers/varifocalnet_iou_aware_dense_detector.pdf`
- `papers/giou_generalized_intersection_over_union.pdf`
- `papers/detr_end_to_end_object_detection.pdf`
- `papers/deformable_detr.pdf`
- `papers/dinov2_learning_robust_visual_features_without_supervision.pdf`
- `papers/vision_transformer_an_image_is_worth_16x16_words.pdf`

## Short Research Claim We Can Safely Make

V8 is a LoRAT-backbone, target-conditioned dense MOT head trained with LoRAT-style IoU-aware score-map supervision and GIoU box regression. The next scientifically grounded upgrade is to add QDTrack-style identity contrastive learning and TrackFormer/MOTR-style sequence/query robustness training, then attach an open-world objectness/proposal branch for TAO-OW.
