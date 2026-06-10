# AMD vs NVIDIA Notes for LoRAT Tracking

Current date: 2026-05-27

## Executive Takeaway

For this project, the AMD/NVIDIA divide is not mainly an algorithmic divide. LoRAT, DanceTrack, MOT17, TAO, and HOTA are not inherently NVIDIA-only. The divide is practical: CUDA/NVIDIA is the most direct path for reproducing LoRAT's reported GPU results and for running week-one benchmarks with the least integration risk; AMD/ROCm can be viable on supported hardware, but the supported-device matrix is narrower and more version-sensitive, especially on Windows laptops.

This means:

- Use NVIDIA/CUDA as the first official benchmark platform for week-one LoRAT model-size comparisons.
- Use the current AMD laptop for UI, data setup, CPU smoke tests, and possibly ROCm experiments only if the exact GPU is supported by AMD's ROCm matrix.
- Treat AMD-vs-NVIDIA as a separate platform ablation, not as the first baseline for LoRAT-B/L/g comparison.

## What the Collected Papers Imply

### LoRAT

The LoRAT paper frames the method as a way to make large ViT trackers practical with parameter-efficient fine-tuning. The relevant hardware signals are:

- LoRAT reports that ViT-g training becomes practical with about 25.8 GB GPU memory for batch size 16.
- It reports reducing L-224 training from 35.0 to 10.8 GPU-hours and raising L-224 inference from 52 to 119 FPS.
- The local PDF also reports training variants from B-224 through g-378 on 8 NVIDIA V100 GPUs and notes B-224 can train on a consumer NVIDIA RTX 4090.

Interpretation: LoRAT is model/transformer compute bound, and the paper's timing expectations are anchored in NVIDIA CUDA hardware. This does not make AMD impossible, but it does mean the cleanest reproduction target is NVIDIA.

Source: [Tracking Meets LoRA](https://arxiv.org/abs/2403.05231)

### DanceTrack and MOT17

DanceTrack exists because normal MOT datasets over-reward appearance-heavy re-ID. It stresses association when people look similar and move/articulate differently. MOT17 is the standard pedestrian MOT baseline. Neither paper depends on a vendor GPU, but both make runtime comparisons meaningful only when hardware and software are pinned.

Interpretation: For item (b), "time per box" must record GPU vendor, GPU model, framework build, precision, and video decode path. For item (c), object-size reliability should be reported separately from speed so hardware differences do not masquerade as tracking quality differences.

Sources: [DanceTrack](https://arxiv.org/abs/2111.14690), [MOTChallenge 2015](https://arxiv.org/abs/1504.01942), [MOT16](https://arxiv.org/abs/1603.00831)

### TAO / Open-World Tracking / HOTA

TAO and open-world tracking push beyond closed known classes. HOTA emphasizes separating detection and association behavior. These papers imply that the AMD/NVIDIA comparison should not change the labeling/evaluation protocol; it should only change runtime/platform metadata.

Interpretation: When we later compare AMD and NVIDIA, the same seeded boxes, sequences, uncertainty thresholds, model checkpoints, and evaluation code should be used. The platform variable should be isolated.

Sources: [TAO](https://arxiv.org/abs/2005.10356), [Opening up Open-World Tracking](https://arxiv.org/abs/2104.11221), [HOTA](https://arxiv.org/abs/2009.07736)

## Current Project State

This workspace currently has:

- Local GPU adapters: AMD Radeon RX 7700S and AMD integrated Radeon Graphics.
- Local PyTorch: `2.12.0+cpu`.
- `torch.cuda.is_available()`: `False`.
- LoRAT GUI v3 uses LoRAT as the tracking backend. OpenCV remains only for frame I/O, drawing, and desktop windowing.

Important PyTorch/ROCm detail: ROCm PyTorch still exposes the device through the `torch.cuda` API surface. If a supported AMD ROCm build is installed correctly, `torch.cuda.is_available()` should become `True`, and `--device cuda:0` can refer to an AMD GPU.

## The Practical Divide

### NVIDIA / CUDA

Strengths for this project:

- LoRAT's official helper script is explicitly documented as "Linux with NVIDIA GPU only."
- PyTorch's official Windows and Linux install pages directly support CUDA wheels.
- The LoRAT paper's reported timing and training hardware are NVIDIA-centered.
- Tooling around AMP, `torch.compile`, CUDA profiling, and cloud rentals is more predictable.

Risks:

- CUDA results should not be generalized to AMD speed without measurement.
- Larger variants, especially `g-378`, still need enough VRAM and careful batch/max-track settings.

Best use:

- Reproduce LoRAT-B/L/g on DanceTrack and MOT17.
- Run week-one timing and small-object reliability benchmarks.
- Establish the paper-facing baseline.

### AMD / ROCm

Strengths for this project:

- ROCm is AMD's open GPU stack and supports PyTorch on selected AMD hardware.
- AMD's WSL support matrix lists PyTorch 2.9.1 + ROCm 7.2.1 with official production support for supported Radeon GPUs.
- AMD's Windows matrix now lists PyTorch 2.9 + ROCm 7.2, Python 3.12, and FP16 support for supported devices.

Risks:

- AMD's Windows note says PyTorch on Windows includes ROCm components, but the full ROCm stack is not yet supported on Windows.
- The current laptop GPU is `AMD Radeon RX 7700S`; AMD's public Windows and WSL support lists include several desktop Radeon 7000/9000 cards and `RX 7700`, but not explicitly `RX 7700S`. Do not assume support until verified.
- Some LoRAT-adjacent code paths are CUDA/NVIDIA-specific, especially optional fast segmentation/flash components and DeepSpeed-style training paths.
- `torch.compile`, mixed precision, and fused kernels can behave differently across CUDA and ROCm.

Best use:

- CPU/debug today.
- ROCm smoke test only after checking exact GPU support.
- Later platform ablation if the same LoRAT run path works reliably.

## Benchmark Design Implications

Every benchmark row should include:

```text
platform: windows|linux|wsl
gpu_vendor: NVIDIA|AMD|CPU
gpu_model
driver_version
torch_version
torch_cuda_version
torch_hip_version
device_string
precision: fp32|fp16_amp|bf16_amp
torch_compile_enabled
lorat_config: B-224|B-378|L-224|L-378|g-224|g-378
max_tracks
dataset
sequence
num_seed_boxes
frames_processed
output_boxes
seconds_total
seconds_per_box
peak_vram_mb
quality_metrics: HOTA, DetA, AssA, IDF1, MOTA where applicable
small_object_area_bin
```

For week one:

1. First compare LoRAT B/L/g on one NVIDIA CUDA platform.
2. Then run the same scripts on CPU as a correctness/debug baseline.
3. Only add AMD if ROCm support is confirmed for the exact GPU and the same LoRAT code path runs.

## Recommended Setup Paths

### NVIDIA Baseline

Use a CUDA machine and install the matching PyTorch wheel:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-lorat-env.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
& ".\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### AMD Experiment

Use AMD's current ROCm install path, not the generic CPU PyPI wheel. On Windows, AMD currently documents ROCm 7.2.1 PyTorch wheels requiring Python 3.12 and a specific Adrenalin driver. Verify before running:

```powershell
& ".\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If `torch.version.hip` is `None` or `torch.cuda.is_available()` is `False`, LoRAT will run CPU-only.

## Recommendation for the Paper

Use NVIDIA/CUDA for the primary LoRAT claims unless an AMD ROCm run is demonstrably stable and reproducible. In the paper, report AMD as a portability/platform study:

- "CUDA baseline" for official model-size and object-size benchmarks.
- "ROCm portability" table if supported hardware is available.
- A short caveat that all algorithmic comparisons are done within the same platform, while cross-platform numbers are reported as systems measurements.

## Sources

- LoRAT paper: https://arxiv.org/abs/2403.05231
- LoRAT repo: https://github.com/LitingLin/LoRAT
- DanceTrack paper: https://arxiv.org/abs/2111.14690
- TAO paper: https://arxiv.org/abs/2005.10356
- Open-world tracking paper: https://arxiv.org/abs/2104.11221
- HOTA paper: https://arxiv.org/abs/2009.07736
- PyTorch local install docs: https://pytorch.org/get-started/locally/
- NVIDIA CUDA Linux install docs: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- AMD ROCm overview: https://www.amd.com/en/products/software/rocm.html
- AMD ROCm Windows PyTorch install docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html
- AMD ROCm Windows support matrix: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
- AMD ROCm WSL support matrix: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html
