# Continual, Quantization-Aware All-in-One Restoration (OUR project)

One compact network, no pretrained generative backbone, that learns
**SR → denoising → deblurring → JPEG** one task at a time, then gets INT8 QAT.

![architecture](docs/architecture.png)

## Layout

```
models/net.py        RestoreNet: degradation encoder (FDI) → 8 SE-gated window-attention blocks → AdaIN decoder
models/lora.py       per-task low-rank adapters (TaskLoRALinear) + switching helpers
models/teacher.py    train-time-only FDI teacher (DINOv2), not part of the deployed model
models/quant.py      INT8 fake-quant layers (per-channel W, per-tensor A, attention matmuls)
data/degrade.py      synthetic SR / noise / JPEG (and a toy blur) degradations
data/datasets.py     synthetic + paired datasets, per-task replay buffer
losses.py            Charbonnier, frequency self-distillation, FDI target normaliser
cl.py                stage-wise freezing + EWC
metrics.py           fixed PSNR/SSIM/LPIPS protocol + ACC/BWT/FGT  (verified against scikit-image)
train.py             runs all stages, evaluates every task after every stage → R matrix
evaluate.py          standalone eval, oracle vs inferred task routing, compound sets
qat.py               PTQ baseline → QAT → layer sensitivity sweep → ONNX QDQ export + parity
probe.py             RQ2: linear probe / nearest centroid on FDI embeddings, confusion matrix
profile_model.py     params, per-task adapter cost, GMACs/GFLOPs, size, latency
common.py            checkpoint loading (FP32 and QAT), patch collection
scripts/make_arch_diagram.py  regenerates docs/architecture.svg
configs/smoke.yaml   2-minute CPU end-to-end test on toy data
configs/cl_adapters.yaml  real run
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_toy_data.py --out toy_data
python train.py --config configs/smoke.yaml          # sanity check, numbers meaningless
python train.py --config configs/cl_adapters.yaml    # real run (needs datasets/ below)
python evaluate.py --config configs/cl_adapters.yaml --ckpt runs/cl_adapters/stage3.pt --routing inferred

# INT8
python qat.py --config configs/cl_adapters.yaml --sensitivity --export
python evaluate.py --config configs/cl_adapters.yaml --ckpt runs/cl_adapters/qat/qat.pt --qat

# analysis
python probe.py --ckpt runs/cl_adapters/stage3.pt --hq_dir datasets/DIV2K/valid_HR
python profile_model.py --ckpt runs/cl_adapters/stage3.pt --onnx runs/cl_adapters/qat/model_sr_int8_qdq.onnx
```

Outputs go to `runs/<name>/`: one checkpoint per stage, plus `results.json` with the
R matrix (PSNR, task × stage), ACC/BWT/FGT, and oracle and inferred-routing results.

## Design decisions (worth stating in the report)

* **Same resolution for all tasks.** For SR, the LQ image is bicubic-upsampled to HR size,
  so one network and one output head cover every task.
* **FDI without a train/test mismatch.** The student encoder sees only LQ and is regressed
  onto the teacher difference T(LQ) − T(HQ), normalised by a running per-dimension RMS.
  Train and test therefore share one code path. The teacher is train-time only, so it
  adds nothing to deployed params or FLOPs.
* **Adapters.** Stage 0 trains everything. Each later stage freezes the base and adds
  LoRA pairs to the SE gates and AdaIN MLPs. With `lora_backbone: true`, adapters also go
  on attention and FFN; ablate both settings.
* **The shared degradation encoder stays trainable** so it can learn new degradation types.
  It is therefore the main source of forgetting, which replay and EWC are there to limit.
  To isolate it, ablate with `shared_trainable: ["task_head.*"]`.

## Experiments to run (configs differ only in `cl` and `stages`)

| Run | Setting |
|---|---|
| specialists ×4 | one-stage config per task |
| joint | `cl.mode: joint` |
| naive sequential | `cl.mode: finetune` |
| proposed | `cl.mode: adapters`, with ablations over replay size, EWC λ, `lora_backbone`, `teacher: none`, `loss.freq: 0`, `loss.dual: 0` |
| order sensitivity | e.g. JPEG → deblur → denoise → SR |

## Datasets (expected under `datasets/`)

| Path | Source |
|---|---|
| `DF2K/HR` | DIV2K train HR + Flickr2K HR merged |
| `BSD400`, `WED` | standard denoising training sets |
| `GoPro/{train,test}/{blur,sharp}` | GoPro_Large, flattened so filenames match |
| `benchmark/{Set5,Set14,BSD100,Urban100,Manga109}/{HR,LR_bicubic/X4}` | official MATLAB-bicubic LR files (do **not** regenerate with PIL) |
| `CBSD68`, `Kodak24`, `McMaster`, `LIVE1`, `Classic5` | clean images; noise / JPEG applied with fixed seeds |
| `SIDD`, `RealSR`, `DRealSR`, `HIDE`, `RealBlur` | add as `type: paired` entries |

## Evaluation protocol (metrics.py)

* SR and JPEG: Y channel (BT.601). SR crops `scale` pixels from the border.
* Denoising and deblurring: RGB, no crop.
* Outputs are quantised to uint8 before scoring.
* SSIM uses an 11×11 Gaussian window with σ = 1.5.

Before trusting any number, reproduce one published baseline with this script
(e.g. download released SwinIR outputs on Set5 ×4 and score them here).

## INT8 scheme (qat.py / models/quant.py)

* **Weights:** per-output-channel symmetric INT8, with scales recomputed every step.
* **Activations:** per-tensor asymmetric uint8 with EMA min/max observers. The observers
  are calibrated on replay patches from **all** tasks, each passed through its own adapter,
  and are frozen after `observer_warmup` steps.
* **What gets quantized:** the inputs and weights of every conv and linear layer, the
  per-task adapter intermediates, and the q, k, v and softmax inputs to the attention
  matmuls.
* **What stays in floating point:** the layers in `keep_fp` (first and last conv by
  default), the norms, the activations and the elementwise ops. Report these in the paper.
  `results.json` lists them along with the INT8 weight coverage.
* **QAT recipe:** task-balanced batches, Charbonnier loss plus L1 distillation from the
  frozen FP32 model, LR 5e-6 with a cosine schedule.
* **Export:** one ONNX graph per task containing QDQ nodes. The script checks that ONNX
  Runtime output matches the PyTorch fake-quant output. Build the real engine on the
  device with `trtexec --onnx=model_sr_int8_qdq.onnx --int8`.

## Not done yet

* QAT after every stage (the secondary RQ3 setting). The current `qat.py` quantizes once
  at the end.
* Real CCTV frames (no-reference metrics only).
* Real motion blur in the probe. `probe.py` uses synthetic Gaussian blur for the deblur class.
