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
8. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402) - DeepSORT/re-ID baseline.
9. [ByteTrack](https://arxiv.org/abs/2110.06864), [StrongSORT](https://arxiv.org/abs/2202.13514), and [BoT-SORT](https://arxiv.org/abs/2206.14651) - modern MOT association/re-ID baselines.
10. [DINOv2](https://arxiv.org/abs/2304.07193), [Vision Transformer](https://arxiv.org/abs/2010.11929), [Grounding DINO](https://arxiv.org/abs/2303.05499), [Segment Anything](https://arxiv.org/abs/2304.02643), [OVTrack](https://arxiv.org/abs/2304.08408), and [Video OWL-ViT](https://arxiv.org/abs/2308.11093) - background for open-world discovery, appearance embeddings, and zero-shot object proposals.

The downloader groups this as `papers-core` and `papers-extended`.
