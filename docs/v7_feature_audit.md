# V7 Feature Audit

## Carried Forward From V5/V6

| V5/V6 Feature | V7 Implementation |
| --- | --- |
| Multiple selected tracks | Same GUI/headless selection flow, with `--initial-boxes` for headless runs. |
| MOTChallenge output | Same result writer through v5 helpers. |
| Annotated video output | Same MP4 writer and overlay path. |
| Per-frame debug CSV | Same track debug rows through v5 helpers. |
| Slot debug CSV | V7 writes slot-shaped debug rows for head-bank candidates. |
| Kalman prediction/hold | Kept; low-confidence tracks coast with damped Kalman motion. |
| Occlusion hold | Kept through `--occlusion-max-frames`, `--occlusion-iou-threshold`, and velocity damping. |
| Identity arbitration | Kept through `LightweightIdentityArbitrator`, now run over v7 shared-frame head candidates. |
| ReID appearance memory | Kept through the same histogram memory bank and update thresholds. |
| ReID/path recovery checks | Kept for low-confidence reacquisition decisions. |
| Shrink/crop learning holds | Kept; controls whether v7 head memory is refreshed from the current ROI. |
| Trusted size memory | Kept; used as v7's box-size safety layer. |
| LoRAT memory slots | Reinterpreted as a per-object v7 head bank. |
| V6 gated SOT memory | Reinterpreted as gated primary/recovery head-bank scoring. |
| V6 CLI recovery flags | Removed from v7. V7 now exposes only v7 recovery/head-bank flags. |
| GPU/memory runtime counters | Kept and extended with shared-frame backbone and object-head counters. |

## Improved By The Shared-Frame Split

- Memory recovery no longer multiplies ViT forward passes; recovery only widens the object head bank.
- Debugging can now separate frame-backbone time from per-object head time.
- Object-count scaling is cleaner to measure because frame encoding is fixed and object work is explicit.
- Identity arbitration sees all v7 candidates from one shared frame representation instead of separate SOT task outputs.

## Still Needing Deeper Model Work

- The current v7 `BatchedObjectConditionedHead` has a trainable LoRA-conditioned score and box-delta path when weights are loaded. Without `--v7-head-weights`, runtime uses a deterministic zero-shot shared-feature similarity head so v7 can be exercised before training.
- Best next step after the Week 2 architecture proof: train or fine-tune the v7 object-conditioned LoRA head on frame-level box-prompt tracking samples while keeping the LoRA-adapted ViT mostly frozen.
