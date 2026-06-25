# V9 Local Search Implementation Notes

Date: 2026-06-23

V9 starts the architecture correction described in `docs/v9_lorat_mot_architecture_research_review.md`.

## Why V9 Exists

V8 proved the shared-backbone idea, but its localization head scores the whole frame feature map. That kept throughput high but moved away from LoRaT's selected-target template/search behavior. The result was predictable: full-body tracking sometimes worked, but selected head/face/small boxes and crowded multi-track identity were unstable.

V9 keeps the expensive shared frame encoder and moves the object-specific prediction back into a local search coordinate system.

## Files Added

- `programs/bounding_box_v9_lorat_local_search.py`
- `programs/train_lorat_v9_local_search_head.py`
- `programs/benchmark_lorat_v9.py`

## Tracker Change

`bounding_box_v9_lorat_local_search.py` reuses the V8 tracker lifecycle:

- LoRaT/DINOv2/ViT frame encoder
- per-track template/memory banks
- DINOv2 crop ReID
- identity arbitration
- multi-track conflict handling
- lost/manual reanchor states
- debug/video/proof outputs

The replaced piece is candidate scoring:

1. The frame is encoded once.
2. Each track predicts a local search window from its previous box and motion state.
3. V9 samples a fixed-size feature grid from the shared frame feature map using `torch.nn.functional.grid_sample`.
4. The object-conditioned head scores those local grids in a batch.
5. l/t/r/b box offsets decode in local search-window coordinates, not whole-frame coordinates.
6. The predicted local box maps back into full-frame coordinates.

This is the intended compromise: one shared frame backbone pass, but LoRaT-like local search geometry.

## Training Change

`train_lorat_v9_local_search_head.py` currently provides the local target-generation scaffold.

It creates:

- local score labels
- positive masks
- l/t/r/b targets normalized by the search window
- forced nearest-positive cells for tiny targets

Smoke test:

```powershell
python programs\train_lorat_v9_local_search_head.py --smoke-targets
```

This passed locally and produced 16x16 local target maps.

## Benchmark Change

`benchmark_lorat_v9.py` is a light shim over the V8 benchmark runner. It swaps the backend module to V9, so the existing timing, identity, ReID, occlusion, video, and proof metrics can be reused.

The argument names still say `v8-*` because the compatibility surface is intentionally reused for now.

## Current Limitations

- V9 can load V8 head checkpoints because the module parameter names are compatible, but those checkpoints were trained on full-frame coordinates. They are not expected to be optimal for V9.
- The full V9 trainer is not finished yet. The next step is to adapt the V8 training loop to call V9's local feature-grid extractor and `make_v9_local_search_targets`.
- The local-grid size defaults to 16. We should test 16, 24, and 32 for small-object quality versus FPS.
- The inherited proof log still uses the old Week 2 CSV schema. The V9 mode is exposed through the backend/status path, but the proof log writer can be made more explicit later.
- V9 is separate from `bounding_box_v9_lorat_open_world.py`, which was an earlier open-world fork of the V8 full-frame head path.

## Next Engineering Steps

1. Build the full V9 trainer from the V8 trainer, but replace full-frame target maps with local search-window target maps.
2. Add a tiny overfit test with one full-body target and one selected head/face target.
3. Train B-224 V9 head only after the overfit test passes.
4. Run `benchmark_lorat_v9.py` against V8 on the same DanceTrack sequence.
5. Compare:
   - N1 full-object quality
   - N1 small/manual selected region quality
   - N5 conflict behavior
   - ReID on/off
   - FPS and GPU memory
6. Only after V9 local quality is healthy, bring TAO/open-world proposals into this architecture.
