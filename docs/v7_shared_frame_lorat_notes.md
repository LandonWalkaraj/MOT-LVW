# LoRAT v7 Shared-Frame Tracker Notes

## What Changed

`programs/bounding_box_v7_lorat_frame_shared.py` is a new experimental tracker branch for the requirement:

> all objects share a single Vision Transformer backbone forward pass, with per-object LoRA heads batched on the GPU

The current v5/v6 path batches LoRAT tracker tasks, but each task still carries its own template/search pair through LoRAT. That improves Python-side overhead, yet it does not make all objects share one current-frame ViT pass.

v7 changes the execution shape:

- The current video frame is resized to the LoRAT search tensor size and encoded once through the LoRA-adapted DINOv2/ViT blocks.
- Each selected object stores a small per-object template/head memory bank derived from ROI features on that shared frame feature map.
- All object heads are scored together with one batched GPU operation over the shared feature map.
- The v7 head is now a trainable object-conditioned LoRA module that predicts objectness and box deltas for each object/grid location.
- Kalman prediction limits each object's search to a local ROI on the shared feature grid.
- v5 identity arbitration, ReID memory, shrink/crop learning holds, trusted-size memory, occlusion holds, MOT/debug outputs, and slot-style debug output are carried forward as v7 head-bank equivalents.
- v6 gated recovery is carried forward as primary/recovery head-bank selection: normal frames score the freshest primary head, while low confidence, weak assignment, stale memory, occlusion, or periodic checks expand to a larger head bank without adding another ViT frame pass.

## Important Limitation

Upstream LoRAT is not naturally separable into "one search backbone pass plus independent object heads." In `external/LoRAT-main/trackit/models/methods/LoRAT/lorat.py`, LoRAT concatenates template and search tokens and runs the ViT blocks over the fused token sequence. That means the original LoRAT SOT head cannot be used exactly without doing per-object transformer work.

So v7 is the first research branch that matches the desired shared-frame computation pattern, but it is not an exact reproduction of upstream LoRAT inference. It should be evaluated as a new architecture branch.

## Current Pieces That Benefit From Moving Past Wrapper-Level Batching

- Memory slots: v5/v6 memory slots become expensive when every slot is another full LoRAT task. In v7, memory is a per-object head bank scored cheaply after the shared frame pass.
- V6 gating: instead of deciding whether to run one or five LoRAT tasks, gating now decides whether to use a small primary head or a larger recovery head bank.
- Identity arbitration: v7 can expose per-object score maps and margins directly, which should give better debug data than only accepted boxes.
- Debug logs: useful v7 counters are shared-frame backbone calls, object-head batch size, ROI tokens scored, head-rank, backbone time, head time, and confidence margin.
- Box size estimation: v7 now has a shared-frame box-delta head, but it needs trained weights before the deltas should be trusted as deliverable-quality localization.

## Next Internal Refactor Target

The deeper LoRAT-internal version should split the model into:

1. a shared frame encoder producing current-frame tokens once;
2. a cached template/object-head representation per tracked object;
3. a batched object-conditioned head that predicts score and box deltas for all objects on the shared tokens.

That split now exists in v7. The next quality step is training the new head on frame-level MOT/DanceTrack samples.
