"""INT8 stage: PTQ baseline -> QAT -> (optional) sensitivity sweep and ONNX export.

    python qat.py --config configs/cl_adapters.yaml                   # uses runs/<name>/stage<last>.pt
    python qat.py --config configs/cl_adapters.yaml --sensitivity --export

Key points (see models/quant.py for the quantization scheme):
  * Observers are calibrated on patches from ALL tasks, each run through its own adapter.
    Calibrating only on the last task makes quantization itself cause forgetting.
  * QAT batches are task-balanced (replay buffer from the checkpoint by default), and the
    loss adds output distillation from the frozen FP32 model.
  * Results: FP32 vs PTQ vs QAT per task/dataset, plus BWT before vs after QAT
    (QAT-induced forgetting), written to runs/<name>/qat/results.json.
"""
import argparse
import copy
import json
import math
import os
import random

import torch
import torch.nn.functional as F
import yaml

from common import collect_patches, load_checkpoint
from data.degrade import TASKS
from evaluate import build_eval_sets, evaluate
from losses import charbonnier
from metrics import psnr
from models.lora import set_active_task
from models.quant import (prepare_qat, quant_coverage, set_act_mode,
                          set_layer_quant)

DEFAULTS = {
    "ckpt": None, "per_task": 500, "calib_batches": 16, "batch": 16,
    "iters": 5000, "lr": 5.0e-6, "observer_warmup": 500, "kd": 1.0,
    "train": "all",                      # all | adapters
    "keep_fp": ["shallow.0", "out"],     # layers left in FP16/FP32 (report these!)
    "quant_attn": True, "export_size": 256, "log_every": 250,
}


def balanced_batches(patches, adapter_keys, batch):
    """Infinite generator of {task_id: (lq, hq)} with batch//n_tasks samples per task."""
    tids = [t for t in sorted(patches) if t in adapter_keys]
    per = max(1, batch // len(tids))
    while True:
        out = {}
        for t in tids:
            lq, hq = patches[t]
            idx = torch.randint(0, len(lq), (per,))
            out[t] = (lq[idx], hq[idx])
        yield out


@torch.no_grad()
def calibrate(model, patches, adapter_keys, device, n_batches, bs=8):
    model.eval()
    set_act_mode(model, "observe")
    for t, (lq, _) in patches.items():
        if t not in adapter_keys:
            continue
        set_active_task(model, adapter_keys[t])
        for i in range(min(n_batches, math.ceil(len(lq) / bs))):
            model(lq[i * bs:(i + 1) * bs].to(device))
    set_active_task(model, None)
    set_act_mode(model, "quant")


def qat_params(q, which="all"):
    """Trainable params for QAT: all | adapters. Sets requires_grad accordingly."""
    if which == "adapters":
        params = [p for n, p in q.named_parameters() if ".A." in n or ".B." in n]
    else:
        params = [p for n, p in q.named_parameters() if not n.startswith(("task_head", "fdi_head"))]
    for p in q.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    return params


def qat_train(q, fp, patches, keys, qc, dev, params, iters, observe=True):
    """Task-balanced QAT: Charbonnier + kd * L1 to the FP reference `fp` (None = no KD).

    observe=True: all activation observers keep updating for the first
    qc["observer_warmup"] iters, then freeze. observe=False leaves observer modes as set.
    """
    opt = torch.optim.AdamW(params, lr=qc["lr"], weight_decay=0.0)
    gen = balanced_batches(patches, keys, qc["batch"])
    if observe:
        set_act_mode(q, "observe_quant")
    for it in range(iters):
        if observe and it == min(qc["observer_warmup"], iters - 1):
            set_act_mode(q, "quant")  # freeze observers
        q.train()
        lr = 0.5 * qc["lr"] * (1 + math.cos(math.pi * it / iters))
        for g in opt.param_groups:
            g["lr"] = lr
        loss, groups = 0.0, next(gen)
        for t, (lq, hq) in groups.items():
            lq, hq = lq.to(dev), hq.to(dev)
            set_active_task(q, keys[t])
            out = q(lq)
            loss = loss + charbonnier(out, hq)
            if fp is not None and qc["kd"] > 0:
                set_active_task(fp, keys[t])
                with torch.no_grad():
                    ref = fp(lq)
                loss = loss + qc["kd"] * F.l1_loss(out, ref)
        loss = loss / len(groups)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if it % qc["log_every"] == 0:
            print(f"  qat it {it:6d}  loss {loss.item():.4f}  lr {lr:.2e}")
    set_active_task(q, None)
    if fp is not None:
        set_active_task(fp, None)
    if observe:
        set_act_mode(q, "quant")
    q.eval()


@torch.no_grad()
def holdout_psnr(model, patches, adapter_keys, device, n=32):
    """Fast proxy for the sensitivity sweep: mean RGB PSNR on a few patches of every task."""
    model.eval()
    vals = []
    for t, (lq, hq) in patches.items():
        if t not in adapter_keys:
            continue
        set_active_task(model, adapter_keys[t])
        out = model(lq[-n:].to(device))
        vals.append(sum(psnr(out, hq[-n:].to(device), "denoise")) / len(out))
    set_active_task(model, None)
    return sum(vals) / len(vals)


def sensitivity_sweep(qmodel, qlayers, patches, keys, device):
    """Quantize ONE layer at a time (everything else FP) and record the PSNR drop."""
    set_layer_quant(qmodel, False)
    attn = [q for n, m in qmodel.named_modules() if n.endswith(("q_q", "k_q", "v_q", "p_q"))
            for q in [m]]
    for q in attn:
        q.mode = "off"
    base = holdout_psnr(qmodel, patches, keys, device)
    rows = []
    for name, _ in qlayers:
        set_layer_quant(qmodel, True, only={name})
        rows.append((name, base - holdout_psnr(qmodel, patches, keys, device)))
    set_layer_quant(qmodel, False)
    for q in attn:
        q.mode = "quant"
    rows.append(("<attention matmuls>", base - holdout_psnr(qmodel, patches, keys, device)))
    set_layer_quant(qmodel, True)
    return sorted(rows, key=lambda r: -r[1])


def export_onnx(qmodel, keys, out_dir, size, device):
    """One ONNX graph per task (adapter baked in), with QDQ nodes. Checks ORT vs PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping export")
        return {}
    qmodel.eval().cpu()
    report = {}
    x = torch.rand(1, 3, size, size)
    for tid, key in keys.items():
        task = TASKS[tid]
        set_active_task(qmodel, key)
        path = os.path.join(out_dir, f"model_{task}_int8_qdq.onnx")
        torch.onnx.export(qmodel, (x,), path, input_names=["lq"], output_names=["out"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
        with torch.no_grad():
            ref = qmodel(x)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = torch.from_numpy(sess.run(None, {"lq": x.numpy()})[0])
        report[task] = {"onnx": path, "max_abs_diff": float((got - ref).abs().max()),
                        "psnr_onnx_vs_torch": psnr(got, ref, "denoise")[0]}
        print(f"  exported {path}: max|diff|={report[task]['max_abs_diff']:.2e}  "
              f"PSNR(onnx, torch)={report[task]['psnr_onnx_vs_torch']:.1f} dB")
    set_active_task(qmodel, None)
    qmodel.to(device)
    return report


def flat(res):
    return {f"{t}/{d}": v["psnr"] for t, ds in res.items() for d, v in ds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--skip_qat", action="store_true", help="PTQ only")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    qc = {**DEFAULTS, **cfg.get("qat", {})}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))
    run_dir = os.path.join("runs", cfg["name"])
    ckpt = a.ckpt or qc["ckpt"] or os.path.join(run_dir, f"stage{len(cfg['stages']) - 1}.pt")
    out_dir = os.path.join(run_dir, "qat")
    os.makedirs(out_dir, exist_ok=True)

    fp, ck, keys = load_checkpoint(ckpt, dev)
    if ck.get("quantized"):
        raise SystemExit(f"{ckpt} was trained with qat.continual=quantized and is already INT8 (fake-quant); "
                         "evaluate it with evaluate.py --ckpt instead")
    for p in fp.parameters():
        p.requires_grad_(False)
    seen = ck["seen_tasks"]
    eval_cfg = {**ck["cfg"], "eval": cfg.get("eval", ck["cfg"].get("eval", {}))}
    eval_sets = build_eval_sets(eval_cfg)
    tile = cfg.get("eval_tile", 0)
    patches = collect_patches(ck["cfg"], ck, qc["per_task"])
    print(f"loaded {ckpt}; tasks {seen}; patches/task "
          f"{ {TASKS[t]: len(v[0]) for t, v in patches.items()} }")

    def ev(m):
        o = evaluate(m, eval_sets, keys, seen, dev, "oracle", tile)
        i = evaluate(m, eval_sets, keys, seen, dev, "inferred", tile) if len(seen) > 1 else None
        return o, i

    results = {"ckpt": ckpt, "qat_cfg": qc}
    print("\n[FP32]")
    results["fp32"], results["fp32_inferred"] = ev(fp)

    # ---------------- PTQ baseline
    q = copy.deepcopy(fp)
    qlayers = prepare_qat(q, qc["keep_fp"], qc["quant_attn"])
    q.to(dev)
    cov, left_fp = quant_coverage(q, qlayers)
    print(f"\nquantized layers: {len(qlayers)}  INT8 weight coverage: {cov:.1%}  FP layers: {left_fp}")
    results["int8_weight_coverage"], results["fp_layers"] = cov, left_fp
    calibrate(q, patches, keys, dev, qc["calib_batches"])
    print("[PTQ]")
    results["ptq"], results["ptq_inferred"] = ev(q)

    if a.sensitivity:
        print("[sensitivity sweep]")
        rows = sensitivity_sweep(q, qlayers, patches, keys, dev)
        results["sensitivity_db"] = rows
        for n, d in rows[:10]:
            print(f"  {d:+.3f} dB  {n}")

    # ---------------- QAT
    if not a.skip_qat and qc["iters"] > 0:
        params = qat_params(q, qc["train"])
        print(f"\n[QAT] iters={qc['iters']} lr={qc['lr']} trainable={sum(p.numel() for p in params):,}")
        qat_train(q, fp, patches, keys, qc, dev, params, qc["iters"])
        set_active_task(q, None)
        set_active_task(fp, None)
        set_act_mode(q, "quant")
        print("[QAT eval]")
        results["qat"], results["qat_inferred"] = ev(q)
        torch.save({"state_dict": q.state_dict(), "adapter_keys": keys, "seen_tasks": seen,
                    "keep_fp": qc["keep_fp"], "quant_attn": qc["quant_attn"], "cfg": ck["cfg"],
                    "teacher_dim": ck.get("teacher_dim", 0)},
                   os.path.join(out_dir, "qat.pt"))

    # ---------------- summary
    final = results.get("qat") or results["ptq"]
    tag = "qat" if results.get("qat") else "ptq"
    f32, ptq, fq = flat(results["fp32"]), flat(results["ptq"]), flat(final)
    table = {k: {"fp32": f32[k], "ptq": ptq[k], tag: fq[k], f"d_{tag}": fq[k] - f32[k]} for k in f32}
    results["table"] = table
    rpath = os.path.join(run_dir, "results.json")
    if os.path.exists(rpath):  # BWT before vs after quantization, using the FP32 diagonal R[j][j]
        tr = json.load(open(rpath))
        R, order = tr["R"], tr["eval_tasks"]

        def row(res):
            return [sum(v["psnr"] for v in res[t].values()) / len(res[t]) for t in order]
        fq_row = row(final)
        T = len(order)
        if T > 1 and len(R) == T:  # one eval column per stage, in stage order
            results["bwt_fp32"] = sum(R[-1][j] - R[j][j] for j in range(T - 1)) / (T - 1)
            results[f"bwt_{tag}"] = sum(fq_row[j] - R[j][j] for j in range(T - 1)) / (T - 1)
    print(f"\n{'dataset':28s} {'FP32':>7s} {'PTQ':>7s} {tag.upper():>7s} {'Δ':>7s}")
    for k, v in table.items():
        print(f"{k:28s} {v['fp32']:7.2f} {v['ptq']:7.2f} {v[tag]:7.2f} {v[f'd_{tag}']:+7.2f}")
    if "bwt_fp32" in results:
        print(f"BWT fp32 {results['bwt_fp32']:+.3f} dB  ->  {tag} {results[f'bwt_{tag}']:+.3f} dB")

    if a.export:
        print("\n[ONNX export]")
        results["export"] = export_onnx(q, keys, out_dir, qc["export_size"], dev)
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=2, default=str)
    print(f"\nwrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
