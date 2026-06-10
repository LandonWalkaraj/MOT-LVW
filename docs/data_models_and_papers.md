# Week 1 Data, Models, and Paper Starter Kit

This note pins the assets needed for the first week: LoRAT multi-object tracking on DanceTrack and MOT17, plus the open-world path toward TAO-OW.

## Immediate Pull

Run this first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-core
```

That pulls:

- The official LoRAT repository snapshot from [LitingLin/LoRAT](https://github.com/LitingLin/LoRAT).
- The official TrackEval repository snapshot from [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).
- LoRAT model weights for ViT-B, ViT-L, and ViT-g in both 224 and 378 search-size variants from the official [LoRAT Google Drive folder](https://drive.google.com/drive/folders/1FvViP0MCSiAu2FSrNjg7XEORn74yOBdD).
- The core PDF reading list under `papers`.

Then pull the smallest dataset starter set:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-assets.ps1 -Asset week1-datasets-small
```

That gives you DanceTrack validation data, MOT17 labels, and TAO labels. Pull full image archives after verifying storage and download time.

## Datasets

| Dataset | Source | What to use this week | Notes |
| --- | --- | --- | --- |
| DanceTrack | [official repo](https://github.com/DanceTrack/DanceTrack), [Hugging Face mirror](https://huggingface.co/datasets/noahcao/dancetrack) | Use train and val for local timing, quality, and small-object benchmarks. | Designed to stress association under similar appearance and diverse motion, which is exactly useful for multi-object LoRAT plus re-ID. |
| MOT17 | [MOTChallenge MOT17](https://motchallenge.net/data/MOT17/) | Use train split locally because GT is public; use test split only for challenge-style submission. | MOT17 includes detector-specific variants. For a user-initialized tracker, treat frames as the primary input and pick one canonical sequence variant for reporting. |
| TAO / TAO-OW | [TAO repo download docs](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md), [MOTChallenge TAO download page](https://motchallenge.net/tao_download.php) | Pull labels first; pull val/non-AVA/HACS image data when open-world experiments begin. | AVA/HACS video portions require MOTChallenge login and terms acceptance. Use TAO-OW/OWTB protocol for open-world unknown-object reporting. |

MOTChallenge direct zip downloads may refuse command-line connections from some networks. If `fetch-assets.ps1` times out on `motchallenge.net`, open the official MOT17 or TAO page in the browser, download the same zip named in `manifests/assets.json`, and place it under the manifest path in `data/raw`.

Recommended local layout:

```text
data/
  DanceTrack/
  MOTChallenge/
  TAO/
data/raw/
models/
  lorat/
external/
papers/
```

## LoRAT Models

The manifest contains the exact public Google Drive file IDs from the official LoRAT weight folder:

| Manifest asset | LoRAT weight | Intended comparison |
| --- | --- | --- |
| `lorat-base-224` | `base.bin` | ViT-B throughput baseline |
| `lorat-base-378` | `base-378.bin` | ViT-B small-object stress test |
| `lorat-large-224` | `large.bin` | ViT-L quality/latency tradeoff |
| `lorat-large-378` | `large-378.bin` | ViT-L small-object stress test |
| `lorat-giant-224` | `giant.bin` | ViT-g upper quality baseline |
| `lorat-giant-378` | `giant-378.bin` | ViT-g upper quality, highest compute |

Use all six for week-one item (d). For item (c), the 378 variants are especially important because the larger search/template setting may shift the object-area failure point.

## Benchmark Setup Guidance

For item (b), report wall-clock time per generated box as:

```text
seconds_per_box = total_tracking_runtime_seconds / number_of_output_boxes
```

Run the benchmark with user-seeded tracks for `1, 2, 4, 8, ... N` initial boxes per video. Keep the same sampled videos and same object IDs across model sizes.

For item (c), bin GT object boxes by pixel area before evaluation:

```text
area = width * height
bins = [0-256, 257-1024, 1025-4096, 4097-16384, >16384]
```

Within each bin, report HOTA/AssA/DetA plus failure rate under your chosen minimum quality threshold. DanceTrack and MOT17 are enough for the week-one deliverable; TAO becomes more useful when you start true open-world object discovery.

## Initial Papers

Read in this order:

1. [Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance](https://arxiv.org/abs/2403.05231) - core tracker implementation and model variants.
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - adapter mechanism used by LoRAT.
3. [DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion](https://arxiv.org/abs/2111.14690) - primary association-stress benchmark.
4. [MOTChallenge 2015](https://arxiv.org/abs/1504.01942) and [MOT16](https://arxiv.org/abs/1603.00831) - MOTChallenge protocol background for MOT17.
5. [TAO: A Large-Scale Benchmark for Tracking Any Object](https://arxiv.org/abs/2005.10356) - open-vocabulary/open-world dataset base.
6. [Opening Up Open-World Tracking](https://arxiv.org/abs/2104.11221) - TAO-OW/OWTB framing and open-world metrics.
7. [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://arxiv.org/abs/2009.07736) - primary quality metric for association-heavy tracking.
8. [Improving Multiple Object Tracking With Single Object Tracking](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.html) - SOTMOT design pattern for turning SOT-style target discrimination into scalable MOT association.
9. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402) - DeepSORT/re-ID baseline.
10. [ByteTrack](https://arxiv.org/abs/2110.06864), [StrongSORT](https://arxiv.org/abs/2202.13514), and [BoT-SORT](https://arxiv.org/abs/2206.14651) - modern MOT association/re-ID baselines.
11. [DINOv2](https://arxiv.org/abs/2304.07193), [Vision Transformer](https://arxiv.org/abs/2010.11929), [Grounding DINO](https://arxiv.org/abs/2303.05499), [Segment Anything](https://arxiv.org/abs/2304.02643), [OVTrack](https://arxiv.org/abs/2304.08408), and [Video OWL-ViT](https://arxiv.org/abs/2308.11093) - background for open-world discovery, appearance embeddings, and zero-shot object proposals.

The downloader groups this as `papers-core` and `papers-extended`.

## V8 Head Training Papers

The V8 branch changes the question from "can we run LoRAT?" to "can we keep the LoRAT ViT backbone and train a multi-object, object-conditioned head?" The detailed notes are in [v8_training_methods_research.md](v8_training_methods_research.md).

## Week 3 Gap-Fill Papers

The Week 3 literature gap-fill is cataloged in [week3_gap_fill_papers.md](week3_gap_fill_papers.md). The PDFs are registered under the manifest group `papers-week3-gap-fill`.

Use this group for:

- ReID and association baselines: BoostTrack, BoostTrack++, StrongSORT, BoT-SORT, and OC-SORT.
- Open-world proposals and correction assist: Grounding DINO, SAM, SAM 2, and Crowd-SAM.
- Correction propagation and memory design: STM, STCN, XMem, AOT, and DeAOT.
- Active correction and human-effort measurement: HD-AMOT, VATIC, Efficient Video Annotation, Video Annotation and Tracking with Active Learning, Snapper, Extreme Clicking, and related annotation papers.

Direct LoRAT/SOT-head references:

| Paper | Local file | Why it matters |
| --- | --- | --- |
| [Tracking Meets LoRA](https://arxiv.org/abs/2403.05231) | `papers/lorat_tracking_meets_lora.pdf` | Core backbone, LoRA adaptation, and tracker design. |
| [LoRAT supplemental](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00113-supp.pdf) | `papers/lorat_supplemental_training_details.pdf` | Extra training and model details. |
| [OSTrack](https://arxiv.org/abs/2203.11991) | `papers/ostrack_one_stream_tracking.pdf` | One-stream transformer SOT training pattern. |
| [MixFormer](https://arxiv.org/abs/2203.11082) | `papers/mixformer_end_to_end_tracking.pdf` | End-to-end target-template mixing for tracking. |
| [TransT](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Transformer_Tracking_CVPR_2021_paper.html) | `papers/transt_transformer_tracking.pdf` | Transformer-based target/search fusion. |

MOT/query/ReID training references:

| Paper | Local file | Why it matters |
| --- | --- | --- |
| [MOTR](https://arxiv.org/abs/2105.03247) | `papers/motr_end_to_end_multi_object_tracking_transformer.pdf` | Track-query training over video. |
| [TrackFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.html) | `papers/trackformer_multi_object_tracking_transformers.pdf` | Transformer queries for persistent object identities. |
| [TransTrack](https://arxiv.org/abs/2012.15460) | `papers/transtrack_multiple_object_tracking_transformer.pdf` | Query propagation from previous frame to current frame. |
| [OVTR](https://openreview.net/forum?id=GDS5eN65QY) | `papers/ovtr_end_to_end_open_vocabulary_mot.pdf` | Open-vocabulary MOT with end-to-end tracking. |
| [QDTrack](https://arxiv.org/abs/2006.06664) | `papers/qdtrack_quasi_dense_similarity_learning_mot.pdf` | Dense contrastive ReID training, the strongest next fit for V8. |
| [FairMOT](https://arxiv.org/abs/2004.01888) | `papers/fairmot_detection_reid_mot.pdf` | Joint detection and ReID baseline. |
| [CenterTrack](https://arxiv.org/abs/2004.01177) | `papers/centertrack_tracking_objects_as_points.pdf` | Previous-frame conditioning and motion offsets. |

Open-world and object-agnostic references:

| Paper | Local file | Why it matters |
| --- | --- | --- |
| [OWL-ViT](https://arxiv.org/abs/2205.06230) | `papers/owlvit_simple_open_vocabulary_detection_vit.pdf` | One-shot image-conditioned and text-conditioned localization. |
| [Video OWL-ViT](https://arxiv.org/abs/2308.11093) | `papers/video_owlvit_open_world_video_localization.pdf` | Open-world video localization and recurrent object representations. |
| [OVTrack](https://arxiv.org/abs/2304.08408) | `papers/ovtrack_open_vocabulary_mot.pdf` | Open-vocabulary MOT and association. |
| [Class-Agnostic Object Detection](https://arxiv.org/abs/2011.14204) | `papers/class_agnostic_object_detection_wacv2021.pdf` | Objectness without a fixed category set. |
| [Class-Agnostic Object Detection with Multi-modal Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1367_ECCV_2022_paper.php) | `papers/class_agnostic_object_detection_multimodal_transformer.pdf` | Generic object localization with multimodal transformer supervision. |
| [RO-ViT](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Region-Aware_Pretraining_for_Open-Vocabulary_Object_Detection_With_Vision_Transformers_CVPR_2023_paper.html) | `papers/ro_vit_region_aware_pretraining_open_vocabulary_detection.pdf` | Region-aware ViT pretraining for open-vocabulary detection. |
| [Object Discovery and Representation Networks](https://arxiv.org/abs/2203.08777) | `papers/object_discovery_and_representation_networks.pdf` | Self-supervised object discovery and representation learning. |
| [Shuffle-Then-Assemble](https://openaccess.thecvf.com/content_ECCV_2018/html/Xu_Yang_Shuffle-Then-Assemble_Learning_ECCV_2018_paper.html) | `papers/shuffle_then_assemble_object_agnostic_visual_relationship_features.pdf` | Relevant mainly as a cautionary object-agnostic feature reference, not as a direct V8 MOT-head recipe. |

Losses and backbone foundations:

| Paper | Local file | Why it matters |
| --- | --- | --- |
| [VarifocalNet](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) | `papers/varifocalnet_iou_aware_dense_detector.pdf` | IoU-aware dense classification alternatives. |
| [Generalized IoU](https://giou.stanford.edu/) | `papers/giou_generalized_intersection_over_union.pdf` | Box loss used by LoRAT and V8 training. |
| [DETR](https://arxiv.org/abs/2005.12872) | `papers/detr_end_to_end_object_detection.pdf` | Set prediction and object queries. |
| [Deformable DETR](https://arxiv.org/abs/2010.04159) | `papers/deformable_detr.pdf` | Efficient transformer detection over spatial features. |
| [DINOv2](https://arxiv.org/abs/2304.07193) | `papers/dinov2_learning_robust_visual_features_without_supervision.pdf` | Backbone family used by LoRAT. |
| [Vision Transformer](https://arxiv.org/abs/2010.11929) | `papers/vision_transformer_an_image_is_worth_16x16_words.pdf` | ViT foundation for the LoRAT backbone. |
