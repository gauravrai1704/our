"""Efficiency report: params, per-task adapter cost, MACs/GFLOPs, model size, latency.

    python profile_model.py --config configs/cl_adapters.yaml
    python profile_model.py --ckpt runs/cl_adapters/stage3.pt --onnx runs/cl_adapters/qat/model_sr_int8_qdq.onnx

MACs are counted with forward hooks on Conv2d / Linear / adapter matmuls, plus the two
attention matmuls per window (computed analytically). GFLOPs = 2 x GMACs. Norms,
activations and elementwise ops are not counted (state this in the paper).
Latency numbers from this script on a desktop GPU are NOT edge numbers: for the paper,
build the TensorRT INT8 engine from the exported ONNX on the target device (e.g. Jetson
Orin Nano: trtexec --onnx=... --int8) and report that.
"""
import argparse
import time

import torch
import torch.nn as nn
import yaml

from data.degrade import TASKS
from models.lora import (TaskLoRALinear, active_weights, add_task_adapters, adapter_parameters,
                         set_active_task)
from models.net import WindowAttention


def count_macs(model, x):
    macs = [0]
    hooks = []
    bases = {id(m.base) for m in model.modules() if isinstance(m, TaskLoRALinear)}

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += o.numel() * k

    def lin_hook(m, i, o):
        macs[0] += o.numel() * m.in_features

    def lora_hook(m, i, o):
        rows = o.numel() // m.base.out_features
        macs[0] += rows * m.base.in_features * m.base.out_features
        for t, _ in active_weights(m.active):
            if t in m.A:
                macs[0] += rows * m.rank * (m.base.in_features + m.base.out_features)

    def attn_hook(m, i, o):
        B, H, W, C = i[0].shape
        n_win, N = B * (H // m.ws) * (W // m.ws), m.ws * m.ws
        macs[0] += 2 * n_win * N * N * C  # QK^T and AV (summed over heads)

    for n, m in model.named_modules():
        if n.startswith(("task_head", "fdi_head")) or id(m) in bases:
            continue
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, TaskLoRALinear):
            hooks.append(m.register_forward_hook(lora_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, WindowAttention):
            hooks.append(m.register_forward_hook(attn_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs[0]


@torch.no_grad()
def latency(fn, x, warmup=5, reps=20):
    for _ in range(warmup):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", default=None, help="exported INT8 QDQ model for an ORT latency check")
    ap.add_argument("--sizes", default="256x256,720x1280")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.ckpt:
        from common import load_checkpoint
        model, ck, keys = load_checkpoint(a.ckpt, dev)
    else:
        from train import build_model
        cfg = yaml.safe_load(open(a.config))
        model = build_model(cfg).to(dev).eval()
        keys = {0: None}
        for st in cfg["stages"][1:]:
            add_task_adapters(model, st["task"])
            keys[TASKS.index(st["task"])] = st["task"]
        model.to(dev)

    heads = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("task_head", "fdi_head")))
    adapters = {k: sum(p.numel() for p in adapter_parameters(model, k)) for k in keys.values() if k}
    total = sum(p.numel() for p in model.parameters()) - heads
    shared = total - sum(adapters.values())
    print(f"shared params        : {shared / 1e6:.3f} M")
    for k, n in adapters.items():
        print(f"adapter[{k:8s}]    : {n / 1e3:.1f} K ({n / shared:.1%} of shared)")
    print(f"deployable total     : {total / 1e6:.3f} M  (all tasks; training-only heads excluded: {heads:,})")
    print(f"size FP32 / FP16     : {total * 4 / 2**20:.2f} MB / {total * 2 / 2**20:.2f} MB")
    print(f"size INT8 (approx.)  : {total / 2**20:.2f} MB  (+ FP layers in keep_fp, scales)")

    for s in a.sizes.split(","):
        h, w = map(int, s.split("x"))
        x = torch.rand(1, 3, h, w, device=dev)
        set_active_task(model, None)
        g0 = count_macs(model, x) / 1e9
        ms = latency(model, x, reps=a.reps)
        print(f"\n{h}x{w}: {g0:.2f} GMACs = {2 * g0:.2f} GFLOPs (base task)   FP32 latency on {dev}: {ms:.1f} ms")
        for tid, key in keys.items():
            if key:
                set_active_task(model, key)
                extra = count_macs(model, x) / 1e9 - g0
                print(f"   adapter {key:8s} adds {extra * 1e3:.2f} MMACs ({extra / g0:.3%})")
        set_active_task(model, None)

    if a.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        shp = sess.get_inputs()[0].shape
        x = torch.rand(*shp).numpy()
        for _ in range(3):
            sess.run(None, {"lq": x})
        t = time.perf_counter()
        for _ in range(a.reps):
            sess.run(None, {"lq": x})
        print(f"\nONNX Runtime CPU ({a.onnx}, input {shp}): {(time.perf_counter() - t) / a.reps * 1000:.1f} ms")


if __name__ == "__main__":
    main()
