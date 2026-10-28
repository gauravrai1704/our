"""Estimate how long the study takes on THIS machine, before reserving GPU time.

Times real training iterations (same compute_loss / backward / optimizer step as train.py,
on random tensors, so no datasets are needed) for every distinct stage setting, then
multiplies by the iteration counts of every generated run. Eval is timed on one tiled
full-size image and multiplied by --eval_images.

    python scripts/estimate_gpu_time.py --base configs/cl_adapters.yaml --suite all --seeds 0 1 2
    python scripts/estimate_gpu_time.py --base configs/cl_adapters.yaml --suite main --seeds 0 --gpus 2

Run from the repo root. Needs the teacher weights (DINOv2 / BLIP-2) to download, or pass
--teacher random to time without them (underestimates by the teacher's forward cost).
"""
import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments  # noqa: E402
from cl import configure_trainable  # noqa: E402
from data.degrade import TASKS  # noqa: E402
from evaluate import tiled_forward  # noqa: E402
from losses import FreqSelfDistill, RunningNorm  # noqa: E402
from models.lora import add_task_adapters  # noqa: E402
from models.teacher import build_teacher  # noqa: E402
from train import build_model, compute_loss  # noqa: E402

_teachers = {}


def teacher_for(name, dev):
    if name not in _teachers:
        t = build_teacher(name)
        _teachers[name] = t.to(dev) if t is not None else None
    return _teachers[name]


def time_stage(cfg, si, dev, iters, warmup):
    """Seconds per training iteration for stage si of cfg."""
    st, clc, lw = cfg["stages"][si], cfg.get("cl", {}), cfg.get("loss", {})
    mode = clc.get("mode", "adapters")
    teacher = teacher_for(cfg.get("teacher", "none"), dev)
    model = build_model(cfg, teacher.dim if teacher else 0).to(dev)
    keys = {TASKS.index(s["task"]): None for s in cfg["stages"][:si + 1]}
    if mode == "adapters" and si > 0:
        for s in cfg["stages"][1:si + 1]:
            add_task_adapters(model, s["task"])
            keys[TASKS.index(s["task"])] = s["task"]
        model.to(dev)
    params = configure_trainable(model, si, st["task"], mode, clc.get("shared_trainable", ["degr.*", "task_head.*", "fdi_head.*"]))
    freq = FreqSelfDistill(cfg.get("model", {}).get("dim", 96)).to(dev) if lw.get("freq", 0) > 0 else None
    if freq is not None:
        params += list(freq.parameters())
    norm = RunningNorm(teacher.dim).to(dev) if teacher else None
    opt = torch.optim.AdamW(params, lr=1e-4)
    lwf = None
    if clc.get("lwf", 0) > 0 and si > 0:
        lwf = (copy.deepcopy(model).eval(), sorted(set(keys.values()), key=lambda k: k or "")[:-1] or [None], clc["lwf"])
    n_rep = max(1, int(st["batch"] * clc.get("replay_ratio", 0.25))) if clc.get("replay_per_task", 0) and si > 0 else 0
    tid = TASKS.index(st["task"]) if st["task"] in TASKS else 0
    old = [t for t in keys if t != tid] or [tid]
    p = st["patch"]
    amp = dev == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def batch(n, t):
        return {"lq": torch.rand(n, 3, p, p), "hq": torch.rand(n, 3, p, p), "task": torch.tensor(t)}

    model.train()
    for i in range(warmup + iters):
        if i == warmup:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        bs = [batch(st["batch"], [tid] * st["batch"])]
        if n_rep:
            bs.append(batch(n_rep, [old[j % len(old)] for j in range(n_rep)]))
        with torch.autocast(dev, enabled=amp):
            loss, _ = compute_loss(model, bs, keys, teacher, freq, lw, dev, norm, lwf)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def time_eval(cfg, dev, size, tile):
    model = build_model(cfg).to(dev).eval()
    x = torch.rand(1, 3, *size, device=dev)
    for i in range(3):
        if i == 1:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        tiled_forward(model, x, tile)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--suite", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per stage setting")
    ap.add_argument("--teacher", default=None, help="override the teacher (e.g. random) for timing")
    ap.add_argument("--eval_images", type=int, default=400, help="eval images per stage (all eval sets)")
    ap.add_argument("--eval_size", default="720x1280")
    ap.add_argument("--qat_minutes", type=float, default=None, help="QAT minutes per run if you also run qat.py (time one run, then pass it here)")
    ap.add_argument("--gpus", type=int, default=1, help="runs execute in parallel, one per GPU")
    a = ap.parse_args()
    import yaml
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = yaml.safe_load(open(a.base))
    cfgs = experiments.build(base, a.suite, a.seeds)
    if a.only:
        cfgs = [c for c in cfgs if c["variant"] in a.only]
    cache, total, per_variant = {}, 0.0, {}
    eh, ew = map(int, a.eval_size.split("x"))
    t_eval = None
    for cfg in cfgs:
        if a.teacher:
            cfg["teacher"] = a.teacher
        secs = 0.0
        mode = cfg["cl"].get("mode", "adapters")
        stages = cfg["stages"]
        if mode == "joint":  # one long stage; cost ~ the first stage's setting
            stages = [{**stages[0], "iters": sum(s["iters"] for s in stages)}]
            cfg = {**cfg, "stages": stages}
        for si, st in enumerate(stages):
            key = json.dumps([cfg.get("model"), cfg.get("teacher"), cfg.get("loss"), mode, si > 0,
                              st["patch"], st["batch"], bool(cfg["cl"].get("replay_per_task")),
                              bool(cfg["cl"].get("lwf"))], sort_keys=True)
            if key not in cache:
                cache[key] = time_stage(cfg, si, dev, a.iters, 3)
                print(f"  timed {cfg['name']} stage {si} ({st['task']}, patch {st['patch']}, batch {st['batch']}): "
                      f"{cache[key] * 1000:.0f} ms/iter")
            secs += cache[key] * st["iters"]
        if t_eval is None:
            t_eval = time_eval(cfg, dev, (eh, ew), cfg.get("eval_tile", 256))
            print(f"  eval: {t_eval:.2f} s per {eh}x{ew} image (tiled)")
        n_routings = 4 if mode == "adapters" else 1   # oracle (+ inferred, ncm, soft)
        secs += len(stages) * a.eval_images * t_eval * n_routings
        per_variant[cfg["variant"]] = per_variant.get(cfg["variant"], 0.0) + secs
        total += secs
    print(f"\n{'variant':26s} {'GPU-hours (all seeds)':>22s}")
    for v, s in per_variant.items():
        print(f"{v:26s} {s / 3600:22.1f}")
    qat_h = 0.0
    if a.qat_minutes is not None:
        qat_h = a.qat_minutes / 60 * len(cfgs)
    print(f"\nruns: {len(cfgs)}   training + eval: {total / 3600:.0f} GPU-hours   QAT: {qat_h:.0f} GPU-hours")
    wall = (total / 3600 + qat_h) / a.gpus
    print(f"wall clock on {a.gpus} GPU(s): {wall:.0f} h = {wall / 24:.1f} days "
          f"(+30% for debugging / reruns: {wall * 1.3 / 24:.1f} days)")


if __name__ == "__main__":
    main()
