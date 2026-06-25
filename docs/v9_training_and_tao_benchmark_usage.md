# V9 Training And TAO Benchmark Usage

## Why V9 Exists

V8 proved the shared-frame backbone idea, but it moved too far away from LoRaT's original selected-target localization setup. LoRaT is trained around template/search pairs, where the chosen object is represented by a template and localized in a local search region. That preserves scale and spatial detail, especially for selected parts such as heads or small objects.

V9 keeps the shared-backbone MOT requirement, but restores the local-search geometry:

- one frozen LoRaT/DINOv2 frame encoder pass per frame;
- one local feature grid sampled per object from that shared feature map;
- one batched object-conditioned LoRA head call across all objects;
- local l/t/r/b box regression targets normalized by the per-object search window.

This is closer to the SOT behavior we actually want: the user selects an object or part, and the head learns to localize that selected target inside a local search region rather than learning a whole-frame detector.

## Relevant Training Ideas From Prior Work

- LoRaT trains a low-rank adapted transformer tracker with template/search supervision, 170 epochs, and large paired-image sampling. The important lesson for V9 is not just epoch count; it is that the supervised task must match template/search inference.
- QDTrack uses dense region-level similarity/ranking supervision for association. V9 keeps hard-negative and assignment ranking losses so selected tracks do not collapse onto nearby objects.
- MOTR/TrackFormer-style MOT systems keep object-specific query state across time. V9 keeps per-object template/memory banks and trains with previous/template sampling, missing targets, and closed-loop rollout.

## Trainer

Main file:

```powershell
python programs\train_lorat_v9_local_search_head.py --help
```

Smoke target check:

```powershell
python programs\train_lorat_v9_local_search_head.py --smoke-targets
```

Example local smoke training:

```powershell
python programs\train_lorat_v9_local_search_head.py `
  --overfit-smoke `
  --lorat-config B-224 `
  --device cuda:0 `
  --max-steps 100 `
  --output models\lorat\v9_local_head_smoke.pt
```

Example full mixed-data training:

```powershell
python programs\train_lorat_v9_local_search_head.py `
  --dataset-root data\raw\DanceTrack `
  --mot17-root data\raw\MOTChallenge\MOT17 `
  --tao-root data\raw\TAO_OW_SUBSET `
  --tao-use-freeform `
  --lasot-root data\raw\LaSOT_subset `
  --lorat-config B-224 `
  --device cuda:0 `
  --epochs 250 `
  --max-wall-hours 47.5 `
  --max-train-samples-per-epoch 2048 `
  --output models\lorat\v9_local_head_B_224.pt
```

The trainer reuses V8's dataset adapters:

- DanceTrack and MOT17 via MOT-style `img1/gt/gt.txt` folders;
- TAO/TAO-OW via TAO JSON annotations and extracted frames;
- LaSOT-style SOT folders via `img/`, `groundtruth.txt`, `full_occlusion.txt`, and `out_of_view.txt`;
- selected-target variants: full body, upper body, head-like, face-like, tiny head, center, and half-body crops;
- template sampling from first, previous, mixed, or recent-window frames.

The V9-specific pieces are:

- `make_v9_local_search_targets(...)`;
- `decode_v9_box_maps_xyxy(...)`;
- `v9_training_head_output(...)`;
- local hard-negative and assignment ranking losses.

## TAO Example Export

To benchmark TAO examples, export one or two videos into MOT format:

```powershell
python programs\export_tao_to_mot_sequences.py `
  --tao-root data\raw\TAO_OW_SUBSET `
  --split validation `
  --output-root data\derived\TAO_OW_MOT_EXAMPLES `
  --max-videos 2 `
  --max-frames 300
```

This creates:

```text
data/derived/TAO_OW_MOT_EXAMPLES/val/<sequence>/img1
data/derived/TAO_OW_MOT_EXAMPLES/val/<sequence>/gt/gt.txt
data/derived/TAO_OW_MOT_EXAMPLES/val/<sequence>/seqinfo.ini
```

The exporter normalizes TAO category IDs to MOT class ID `1` so the current benchmark default can select them without extra `--class-id` arguments.

## V9 Benchmark

Main wrapper:

```powershell
python programs\benchmark_lorat_v9.py --help
```

Benchmark regular MOT-style sequences:

```powershell
python programs\benchmark_lorat_v9.py `
  --dataset-root data\raw\DanceTrack `
  --sequence dancetrack0065 `
  --track-counts 1,2,3,4,5 `
  --lorat-config B-224 `
  --device cuda:0 `
  --v8-head-weights models\lorat\v9_local_head_B_224.pt `
  --save-video
```

Export and benchmark TAO examples in one command:

```powershell
python programs\benchmark_lorat_v9.py `
  --tao-root data\raw\TAO_OW_SUBSET `
  --tao-example-videos 2 `
  --tao-example-max-frames 300 `
  --track-counts 1,2 `
  --lorat-config B-224 `
  --device cuda:0 `
  --v8-head-weights models\lorat\v9_local_head_B_224.pt `
  --save-video
```

The benchmark wrapper exports TAO examples, then calls the existing V8 benchmark engine with the V9 tracker substituted in. The command still writes timing, area reliability, identity, occlusion, candidate diagnostics, proof CSVs, summary markdown, and videos.

## What To Watch In Results

- `train_mean_iou` and `val_mean_iou`: whether the V9 head is learning the local search task.
- `seconds_per_step`: whether mixed datasets and template caching are keeping training time manageable.
- candidate diagnostics and videos: whether multi-track conflict resolution is reducing duplicate jumps.
- small-object videos: whether local search restores selected-head/part tracking instead of collapsing to full-body boxes.
