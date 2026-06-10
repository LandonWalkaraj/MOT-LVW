# V7 LoRAT Split Status

## Done In V7

- Split LoRAT usage into a v7-only `SharedFrameLoRATEncoder`.
- Added v7-only cached per-object template memory via `V7TemplateMemorySlot`.
- Added v7-only `BatchedObjectConditionedHead` with a trainable per-object LoRA-conditioned score and box-delta module.
- Replaced the per-object search-crop evaluator pipeline for v7. V7 does not call `DefaultTrackerEvaluator` or `self.evaluator.run`.
- Kept v7 standalone: it does not import v6 and does not subclass v5/v6. It imports v5 only for shared utility/data/output helpers.
- Added a zero-shot shared-feature similarity head for no-training runtime, plus `--v7-head-weights` for the trained LoRA-conditioned head path.
- Added `--week2-proof-log`, which records per-frame shared-backbone and object-head batch deltas.
- Preserved v4/v5/v6 behavior by leaving their upstream LoRAT evaluator call path untouched.
- Added `programs/train_lorat_v7_head.py`, a DanceTrack/MOT frame-level training adapter and loop that freezes the shared encoder and trains only the v7 head.

## Partially Done

- Head redesign: the module exists and predicts objectness plus box deltas, but it needs trained weights before it can be considered a reliable replacement for LoRAT's original fused template/search box head.

## Still To Do

- Train or fine-tune the v7 object-conditioned head on frame-level box-prompt tracking data.
- Decide whether to fine-tune only the new head first, or also fine-tune small LoRA adapters after the head is stable.
- Benchmark v7 against v4/v5/v6 only after the trained head exists, so FPS and reliability numbers are meaningful.
