# Runbook: from a fresh GPU machine to paper tables

Run every command from the repo root unless a step says otherwise. Steps marked
**(check)** depend on a third-party site. Those sites change, so confirm the link or
flag before relying on it.

---

## 0. Before you book the GPU

* Disk: about 150 GB (DF2K ≈ 15 GB, GoPro_Large ≈ 30 GB unpacked, test sets, checkpoints;
  each run is about 0.5–2 GB of checkpoints).
* Ask whether the machine has internet access. If it doesn't, do step 4 (teacher and metric
  weights) on a machine that does and copy `~/.cache/torch` and `~/.cache/huggingface` over.
* Ask whether it uses SLURM. If so, see step 7c.

## 1. Get the code

```bash
git clone https://github.com/gauravrai1704/our.git
cd our
git checkout claude/compassionate-hawking-bj55nv
git log --oneline -3
```

The repo is private, so authenticate first. Either add an SSH key to GitHub and clone
`git@github.com:gauravrai1704/our.git`, or run `gh auth login`, or use a personal access
token when git asks for a password. After the branch is merged into `main`, skip the
`checkout`. Later, to get updates: `git pull`.

Work inside `tmux` so training survives when your SSH connection drops:

```bash
tmux new -s our          # detach: Ctrl-b d     reattach: tmux attach -t our
```

## 2. Python environment

```bash
nvidia-smi                                    # note the driver's "CUDA Version"
conda create -n our python=3.11 -y && conda activate our
# pick the wheel matching that CUDA version at https://pytorch.org/get-started/locally/  (check)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name.

## 3. Datasets

### 3a. Direct downloads (scripted)

```bash
bash scripts/download_data.sh     # DIV2K, Flickr2K, SR benchmarks (Set5/Set14/B100/Urban100), Kodak24
```

### 3b. Manual downloads → put each in `datasets/raw/` (check every link)

| Dataset | Used for | Where |
|---|---|---|
| GoPro_Large | deblur train / val / test | https://seungjunnah.github.io/Datasets/gopro (Google Drive; `pip install gdown`, `gdown <file id>`) → unzip to `datasets/raw/GOPRO_Large/{train,test}/<seq>/{blur,sharp}` |
| BSD400, WED, CBSD68, McMaster | denoise train / test | Restormer's script: `git clone https://github.com/swz30/Restormer ../Restormer && cd ../Restormer/Denoising && python download_data.py --help` (Gaussian-noise train + test sets), then copy the folders to `datasets/raw/BSD400`, `WED`, `CBSD68`, `McMaster` |
| LIVE1, Classic5 | JPEG test | test-set folders of the FBCNN / SwinIR repos → `datasets/raw/LIVE1`, `datasets/raw/Classic5` |
| Manga109 (optional) | SR test | request access at manga109.org; downscale with **MATLAB** `imresize` (not PIL) to make LR_bicubic/X4 |
| RealSR, DRealSR | real-world SR test (for the comparison with [2]–[6]) | https://github.com/csjcai/RealSR, the DRealSR repo by Wei et al. |
| SIDD, RealBlur, HIDE (optional) | real-world denoise / deblur test | official project pages |

Folder names matter: `scripts/build_splits.py` finds datasets by name anywhere under
`datasets/raw/`.

### 3c. Build the train / val / test layout

```bash
python scripts/build_splits.py --raw datasets/raw --out datasets
```

| Split | Contents | Used by |
|---|---|---|
| **train** | `DF2K/HR` (DIV2K train + Flickr2K minus 100), `BSD400`, `WED`, `GoPro/train` (minus 2 sequences) | `stages:` |
| **val** | `val/HR` (100 held-out Flickr2K images, synthetic ×4 / σ25 / q20), `GoPro/val` (2 held-out train sequences, every 10th frame) | `val:` → `--eval_on val`, for **all** hyperparameter choices |
| **test** | Set5, Urban100 (+Set14, B100, Manga109), CBSD68, Kodak24, McMaster, GoPro test, LIVE1, Classic5, RealSR/DRealSR; DIV2K valid for the held-out compound degradations | `eval:`, for **final numbers only** |

### 3d. Check everything before training

```bash
python scripts/check_data.py --config configs/cl_adapters.yaml
```

It must end with `ALL GOOD`. It reports missing folders, LQ/HQ count or name mismatches,
images smaller than the training patch, and val/test images that also appear in training
(leakage). Fix every line it prints. If you choose not to use a test set, delete its entry
from `eval:` in `configs/cl_adapters.yaml`.

Recommended additions to `eval:` for the paper (copy the existing entries' format):
Set14_x4, B100_x4, Manga109_x4, McMaster_s25, CBSD68_s15/s50, Kodak24_s25, Classic5_q10/q30,
and RealSR / DRealSR as `type: paired` with `sr_scale: 4`. For the comparison with the
generative methods, also add `nr_metrics: [musiq, maniqa, clipiqa, niqe]` at the top level.

## 4. Pre-download teacher and metric weights

```bash
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
python -c "from transformers import Blip2VisionModelWithProjection as M; M.from_pretrained('Salesforce/blip2-itm-vit-g')"
python -c "import lpips; lpips.LPIPS(net='alex')"
python -c "import pyiqa; [pyiqa.create_metric(m) for m in ('musiq','maniqa','clipiqa','niqe')]"
```

## 5. GPU sanity checks (≈ 15 min)

```bash
python scripts/make_toy_data.py --out toy_data
python train.py --config configs/smoke.yaml                 # whole pipeline on toy data, on the GPU
python qat.py --config configs/smoke.yaml --export          # INT8 + ONNX on the GPU
python scripts/estimate_gpu_time.py --base configs/cl_adapters.yaml --suite all --seeds 0 1 2
```

The last command times real iterations and prints GPU-hours per variant and the total in
days. Plan your booking from that output, not from my earlier estimate.

## 6. Pilot run (≈ 2–4 h), to catch problems on real data cheaply

```bash
python experiments.py --base configs/cl_adapters.yaml --suite main --only proposed finetune \
    --seeds 0 --scale 0.05 --eval_on val --run
python aggregate.py --val --out runs/pilot_tables
```

Check that the loss goes down, `R row` PSNRs look sensible (SR ×4 above bicubic, denoising
clearly above the noisy input), and `routing acc.` is well above 0.25.

## 7. Experiments

All runs write to `runs/<variant>_s<seed>/`. Re-running the same command skips finished runs
and resumes unfinished ones from the last completed task stage (`--resume` is passed
automatically). A crash costs at most one stage.

### 7a. Hyperparameter search on the val split (1 seed, shortened schedule)

```bash
python experiments.py --base configs/cl_adapters.yaml --suite ablation \
    --only abl_ewc10 abl_ewc1000 abl_no_ewc abl_replay100 abl_replay1000 abl_rank4 abl_rank16 \
    --seeds 0 --scale 0.25 --eval_on val --run
python aggregate.py --val --out runs/val_tables
```

Put the winning values (`ewc_lambda`, `replay_per_task`, `lora_rank`, …) into
`configs/cl_adapters.yaml`, then **commit that change**. Do not tune anything on test numbers.

### 7b. Final runs on the test sets

One GPU:

```bash
python experiments.py --base configs/cl_adapters.yaml --suite main     --seeds 0 1 2 --run --qat
python experiments.py --base configs/cl_adapters.yaml --suite proposal --seeds 0 1 2 --run
python experiments.py --base configs/cl_adapters.yaml --suite quant    --seeds 0 1 2 --run
python experiments.py --base configs/cl_adapters.yaml --suite ablation --seeds 0 --scale 0.5 --run
python experiments.py --base configs/cl_adapters.yaml --suite order    --seeds 0 --run
```

Several GPUs: run the same command once per GPU, each with its own shard:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments.py --base configs/cl_adapters.yaml --suite main --seeds 0 1 2 --run --qat --shard 0/2
CUDA_VISIBLE_DEVICES=1 python experiments.py --base configs/cl_adapters.yaml --suite main --seeds 0 1 2 --run --qat --shard 1/2
```

(each in its own tmux window). Log to a file if you like: append `2>&1 | tee runs/main_shard0.log`.

What each suite contains:

| Suite | Runs |
|---|---|
| `main` | specialists ×4, joint, finetune, ER, EWC, MAS, LwF, adapters-only, proposed |
| `proposal` | ICVGIP [1] reproduction and the cumulative ladder: [1] + CL, + DINOv2 [4], + frequency [5] (+ dual [6] = proposed) |
| `quant` | INT8 after every task: snapshot and fully quantized, for proposed / ER / finetune |
| `ablation` | one component changed at a time |
| `order` | reversed and shuffled task orders |

### 7c. SLURM instead of tmux (if your lab uses it)

```bash
cat > run_main.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=our-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=runs/slurm-%j.log
source ~/miniconda3/etc/profile.d/conda.sh && conda activate our
cd $SLURM_SUBMIT_DIR
python experiments.py --base configs/cl_adapters.yaml --suite main --seeds 0 1 2 --run --qat --shard ${SHARD:-0/1}
EOF
SHARD=0/2 sbatch run_main.sbatch
SHARD=1/2 sbatch run_main.sbatch
```

When a job hits its time limit, resubmit it; it resumes from the last finished stage.
Partition and account names are cluster-specific, so ask your admin.

### 7d. Monitoring

```bash
watch -n 30 nvidia-smi
tail -f runs/main_shard0.log
ls runs/*/results.json | wc -l                  # finished runs
python -c "import json;r=json.load(open('runs/proposed_s0/results_partial.json'));print([[round(x,2) for x in row] for row in r['R']])"
```

## 8. Testing / evaluation of a finished model

The checkpoint after the last task is `runs/<run>/stage3.pt`.

```bash
C=configs/cl_adapters.yaml; K=runs/proposed_s0/stage3.pt

# per-task PSNR / SSIM / LPIPS, with the four routing modes
python evaluate.py --config $C --ckpt $K --routing oracle     # task label given
python evaluate.py --config $C --ckpt $K --routing inferred   # learned task head
python evaluate.py --config $C --ckpt $K --routing ncm        # nearest prototype
python evaluate.py --config $C --ckpt $K --routing soft       # adapter mixture (compound degradations)

# INT8 (quantize once at the end; already done by --qat in 7b)
python qat.py --config $C --ckpt $K --sensitivity --export
python evaluate.py --config $C --ckpt runs/proposed_s0/qat/qat.pt --qat --routing ncm

# INT8 learned per task (quant suite) needs no extra step: its stage3.pt is already INT8
python evaluate.py --config $C --ckpt runs/proposed_q_quantized_s0/stage3.pt --routing ncm

# RQ2 probe: does the FDI embedding identify the degradation? (content-controlled)
python probe.py --ckpt $K --hq_dir datasets/DIV2K/valid_HR

# efficiency: params, per-task adapter cost, GFLOPs, latency, ONNX Runtime CPU latency
python profile_model.py --ckpt $K --sizes 256x256,720x1280 --onnx runs/proposed_s0/qat/model_sr_int8_qdq.onnx
```

### Edge device (Jetson etc.)

Copy `runs/proposed_s0/qat/model_<task>_int8_qdq.onnx` to the device, then:

```bash
trtexec --onnx=model_sr_int8_qdq.onnx --int8 --saveEngine=sr_int8.plan
trtexec --loadEngine=sr_int8.plan --iterations=200 --avgRuns=50
```

Report the median latency printed under "GPU Compute Time".

## 9. External methods ([2]–[6], RwF, PromptIR, AirNet, InstructIR, SwinIR, Restormer)

For every method, `configs/external_baselines.yaml` records the reference, code status and
how to compare (`outputs` / `retrain` / `reported`).

```bash
# methods with code: run their official inference on OUR test inputs, save outputs as
#   results_external/<Method>/<eval dataset name>/<image name>.png
python eval_external.py --config configs/cl_adapters.yaml --method VOSR --outputs results_external/VOSR

# all-in-one models: retrain with their official code on OUR training data
# (datasets/DF2K/HR, BSD400, WED, GoPro/train), then run inference and eval_external.py as above

# methods without code: type the paper's numbers under `reported:` in
# configs/external_baselines.yaml (dataset keys = our eval dataset names)
```

The LQ inputs for synthetic test sets are generated on the fly. To give another method
exactly the same inputs, dump them first:

```bash
python - <<'PY'
import os, yaml, numpy as np
from PIL import Image
from evaluate import build_eval_sets
cfg = yaml.safe_load(open("configs/cl_adapters.yaml"))
for task, items in build_eval_sets(cfg).items():
    for name, ds, _ in items:
        os.makedirs(f"inputs_lq/{name}", exist_ok=True)
        for i in range(len(ds)):
            b = ds[i]
            x = (b["lq"].clamp(0, 1).numpy().transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
            Image.fromarray(x).save(f"inputs_lq/{name}/{os.path.splitext(b['name'])[0]}.png")
PY
```

For SR methods that expect a small LR input, give them the official `LR_bicubic` files
instead (these dumps are bicubic-upsampled to HR size).

## 10. Paper tables

```bash
python aggregate.py                      # -> runs/tables.md and runs/tables.tex
```

The output contains:
* final per-dataset PSNR / SSIM, ACC, BWT, FGT and the drop versus specialists
* task-free inference: oracle vs task head vs NCM vs soft routing
* held-out compound degradations
* the per-stage R matrices
* INT8 at the end, and INT8 after every task
* the comparison with external methods (params, PSNR / SSIM / LPIPS / NR metrics)

Copy `runs/tables.tex` into the paper (needs `\usepackage{booktabs}`).

## 11. Back up results

Training outputs are not in git (`runs/` is ignored). Copy them off the lab machine
regularly. JSON files and logs are enough for the tables; checkpoints are only needed to
re-evaluate.

```bash
rsync -av --include='*/' --include='*.json' --include='*.yaml' --include='*.log' --exclude='*' runs/ you@your-laptop:our_runs/
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | lower `batch` (and raise `iters` proportionally) or `patch` in the stage; set `eval_tile: 192` |
| DataLoader is the bottleneck (GPU util < 70 %) | raise `workers` in the config; put datasets on a local SSD, not NFS |
| `torch.hub` / Hugging Face download fails | no internet: do step 4 elsewhere and copy the caches |
| `MISMATCH` or a pairing warning from check_data | LQ/HQ filenames differ; rename so stems match (`x4` suffixes are fine) |
| A run died | re-run the same command; it resumes from the last finished stage |
| Loss becomes `nan` | `amp: false` in the config, or lower `lr`; report which stage |
