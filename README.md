# DCT-Cauvis on S-DGOD

## Method Summary

This release packages a Cauvis-based single-source domain generalized object detection method
called **DCT-Cauvis** (Dynamic Causal-Temporal Causal Visual Prompt evolution).

- Baseline method: **Cauvis** (DINOv2 backbone + causal visual prompts).
- Dataset/benchmark: **S-DGOD / Diverse Weather** urban-scene split.
- Task: single-domain domain-generalized detection with source domain `Daytime-Sunny` and target domains
  `Daytime-Foggy`, `Dusk-rainy`, `Night-rainy`, `Night-Sunny`.
- Metric: **mAP@50** and **mean AP50** over five weather domains under the official SDGOD-style evaluator.

## Release Layout

- `code/` — patched implementation files.
- `configs/` — runnable configs for baseline and DCT-Cauvis experiments.
- `scripts/` — environment-aware launch scripts.
- `results/` — summary JSON + markdown table.
- `results/metrics-comparison.json` — official verification payload used for final claim.

## What is Included

1. Modified Cauvis prompt module:
   - `code/Cauvis/mmdet/models/layers/cauvis.py`
   - `code/Cauvis/mmdet/models/backbones/cauvis_backbone.py`
   - `code/Cauvis/mmdet/models/layers/__init__.py`
   - `code/Cauvis/mmdet/evaluation/metrics/s_dgod_voc_metric.py`
2. SDGOD base dataset config:
   - `code/Cauvis/configs/_base_/datasets/Sdgod.py`
3. Baseline + candidate configs:
   - `code/Cauvis/configs/cauvis/cauvis_dinov2_dinohead_bs1x4_sdgod.py`
   - `code/Cauvis/configs/cauvis/dct_cauvis_dinov2_dinohead_bs1x4_sdgod_project.py`
4. Reproducible script wrappers under `scripts/`.

## Setup (Environment)

- Prepare the upstream Cauvis repo (commit from upstream at your end), Python 3.10, CUDA-enabled PyTorch, and install dependencies.
- This release uses official repo defaults plus the files above.

Recommended install (adapt to your local setup):

```bash
cd /path/to/Cauvis
pip install -r requirements.txt
pip install -v -e .
```

## Required Environment Variables

Set before running:

- `SDGOD_DATA_ROOT` — path to `Single-DGOD` root directory.
  - Example: `/data/Single-DGOD`
- `CAUVIS_PRETRAINED` — path to Cauvis DINOv2-SDGOD baseline checkpoint.
  - Example: `/data/pretrained_weights/cauvis/cauvis_dinohead.pth`
- `PROJECT_ROOT` — project root containing this release folder (used by configs).
- Optional: `CAUVIS_WORK_DIR` override output directory.

## Run Commands

### Baseline Evaluation (Cauvis)

```bash
cd /path/to/release/cv-single-domain-object-detection-2-1780324605924
export PROJECT_ROOT=/path/to/release/cv-single-domain-object-detection-2-1780324605924
export SDGOD_DATA_ROOT=/data/Single-DGOD
export CAUVIS_PRETRAINED=/data/pretrained_weights/cauvis/cauvis_dinohead.pth
bash scripts/run_baseline_eval.sh
```

### DCT-Cauvis Fine-tune (from baseline checkpoint)

```bash
cd /path/to/release/cv-single-domain-object-detection-2-1780324605924
export PROJECT_ROOT=/path/to/release/cv-single-domain-object-detection-2-1780324605924
export SDGOD_DATA_ROOT=/data/Single-DGOD
export CAUVIS_PRETRAINED=/data/pretrained_weights/cauvis/cauvis_dinohead.pth
bash scripts/run_dct_train.sh
```

### DCT-Cauvis Evaluation

```bash
cd /path/to/release/cv-single-domain-object-detection-2-1780324605924
export PROJECT_ROOT=/path/to/release/cv-single-domain-object-detection-2-1780324605924
export SDGOD_DATA_ROOT=/data/Single-DGOD
bash scripts/run_dct_eval.sh /path/to/checkpoint/epoch_12.pth
```

## Evaluation Protocol

- Fair protocol baseline and candidate use the same config family and SDGOD evaluator.
- In this project verification, the final baseline/eval comparisons were run with batch size 1 for SDGOD per-domain keying correctness.
- Exact claim scope: same-protocol S-DGOD mAP@50 comparison between official Cauvis baseline and DCT-Cauvis.

## Results Summary (Finalized Claim Scope)

| Model | Daytime-Sunny | Daytime-Foggy | Dusk-rainy | Night_rainy | Night-Sunny | Target mean AP50 | All-domain mean AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cauvis (corrected batch_size=1) | 0.7410 | 0.5650 | 0.6500 | 0.4750 | 0.6060 | 0.5740 | 0.6074 |
| DCT-Cauvis (12 epochs) | 0.7530 | 0.5710 | 0.6420 | 0.4800 | 0.6190 | 0.5780 | 0.6130 |
| Δ (DCT - Baseline) | +0.0120 | +0.0060 | -0.0080 | +0.0050 | +0.0130 | +0.0040 | +0.0056 |

Supported claim: modest improvement in all-domain and target mean AP50 under the fixed protocol.
Unsupported by final verification: +1.0 absolute target mean gain and universal target-domain gains.

## Citations

- Chen Li et al., **Towards Single-Source Domain Generalized Object Detection via Causal Visual Prompts (Cauvis)**, arXiv:2510.19487 / OpenReview (repository and paper).
- Aming Wu et al., **Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation**, CVPR 2022.
- Aming Wu et al., **Cauvis repository**, https://github.com/lichen1015/Cauvis.
- S-DGOD dataset repository, https://github.com/AmingWu/Single-DGOD.
- Official benchmark-style checkpoint link (as reported in repo docs): `https://drive.google.com/file/d/1ZjddVB0h4ZYjZ3g2ve6BsQUamvOidTxj/view?usp=sharing`

## Removed/Excluded in this public bundle

- Full local dataset images, checkpoint binaries, logs, run artifacts, pipeline metadata, and local-infrastructure paths.
- `research-pipeline-artifacts` internal reports and session traces.
