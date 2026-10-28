"""Sequential (continual) training: SR -> denoise -> deblur -> JPEG.

    python train.py --config configs/smoke.yaml

cl.mode:
  adapters : stage 0 trains everything; later stages train LoRA adapters + shared
             degradation encoder/heads, with replay (+ optional EWC)        [proposed]
  finetune : full fine-tuning per stage, no adapters                        [lower bound]
  joint    : one stage on all tasks at once, no adapters                    [joint baseline]
  (specialists = run with a single-stage config per task)

After every stage, all tasks in cfg.eval are evaluated -> R matrix in runs/<name>/results.json
"""
import argparse
import copy
import fnmatch
import json
import math
import os
import time

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from cl import EWC, configure_trainable
from data.datasets import PairedDataset, ReplayBuffer, SyntheticTaskDataset
from data.degrade import TASKS
from evaluate import build_eval_sets, evaluate
from losses import FreqSelfDistill, RunningNorm, charbonnier
from metrics import cl_summary
from models.lora import add_task_adapters, set_active_task
from models.net import RestoreNet, count_params
from models.teacher import build_teacher


# ---------------------------------------------------------------- builders
def build_model(cfg, teacher_dim=0):
    m = cfg.get("model", {})
    return RestoreNet(teacher_dim=teacher_dim, n_tasks=len(TASKS), **m)


def build_train_set(task, items, patch):
    sets = []
    for it in items:
        if it["type"] == "synthetic":
            sets.append(SyntheticTaskDataset(it["hq_dir"], task, patch, True, repeat=it.get("repeat", 1)))
        else:
            sets.append(PairedDataset(it["lq_dir"], it["hq_dir"], task, patch, True,
                                      sr_scale=it.get("sr_scale"), repeat=it.get("repeat", 1)))
    return ConcatDataset(sets)


def infinite(loader):
    while True:
        yield from loader


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def lr_at(it, total, base, warmup):
    if it < warmup:
        return base * (it + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup)))


# ---------------------------------------------------------------- one step
def compute_loss(model, batch, adapter_keys, teacher, freq_mod, lw, device, fdi_norm=None):
    """Batch may mix tasks (current + replay): forward each task group with its own adapter."""
    lq, hq, tid = batch["lq"].to(device), batch["hq"].to(device), batch["task"].to(device)
    logs, total = {}, 0.0
    for t in tid.unique().tolist():
        sel = tid == t
        set_active_task(model, adapter_keys.get(t))
        x, y = lq[sel], hq[sel]
        out, aux = model(x, return_feats=True)
        loss = charbonnier(out, y)
        logs["rec"] = logs.get("rec", 0) + loss.item() * sel.float().mean().item()

        if lw.get("task", 0) > 0:
            ce = torch.nn.functional.cross_entropy(model.task_head(aux["fdi"]), tid[sel])
            loss = loss + lw["task"] * ce
        if teacher is not None and lw.get("fdi", 0) > 0:
            with torch.no_grad():
                target = teacher(x) - teacher(y)
            if model.training:
                fdi_norm.update(target)
            pred = model.fdi_head(aux["fdi"])
            fl = torch.nn.functional.mse_loss(pred, fdi_norm(target))
            loss = loss + lw["fdi"] * fl
        if freq_mod is not None and lw.get("freq", 0) > 0:
            f = aux["feats"]
            loss = loss + lw["freq"] * freq_mod(f[len(f) // 2 - 1], f[-1])
        if lw.get("dual", 0) > 0:
            x2 = x + torch.randn_like(x) * (lw.get("dual_sigma", 2.0) / 255.0)
            out2 = model(x2)
            loss = loss + lw["dual"] * (charbonnier(out2, y) + charbonnier(out2, out.detach()))
        total = total + loss * sel.float().mean()
    set_active_task(model, None)
    return total, logs


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg.get("seed", 0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join("runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "config.yaml"), "w"))

    clc = cfg.get("cl", {})
    mode = clc.get("mode", "adapters")
    lw = cfg.get("loss", {})
    teacher = build_teacher(cfg.get("teacher", "none"))
    if teacher is not None:
        teacher = teacher.to(dev)
    model = build_model(cfg, teacher.dim if teacher else 0).to(dev)
    fdi_norm = RunningNorm(teacher.dim).to(dev) if teacher else None
    freq_mod = FreqSelfDistill(cfg.get("model", {}).get("dim", 96)).to(dev) if lw.get("freq", 0) > 0 else None
    print(f"deployable params: {count_params(model) - (sum(p.numel() for p in model.fdi_head.parameters()) if model.fdi_head else 0):,}")

    eval_sets = build_eval_sets(cfg)
    # R-matrix columns follow STAGE order so R[j][j] = task j right after it was learned
    eval_tasks = [s["task"] for s in cfg["stages"] if s["task"] in eval_sets]
    replay = ReplayBuffer(clc.get("replay_per_task", 0)) if mode == "adapters" and clc.get("replay_per_task", 0) else None
    ewc = EWC(clc.get("ewc_lambda", 0.0))
    shared = clc.get("shared_trainable", ["degr.*", "task_head.*", "fdi_head.*"])

    stages = cfg["stages"]
    if mode == "joint":  # merge all stages into one multi-task stage
        stages = [{**stages[0], "task": "joint", "train": None,
                   "_joint": [(s["task"], s["train"]) for s in stages],
                   "iters": sum(s["iters"] for s in stages)}]

    adapter_keys, seen, R, history = {}, [], [], []
    for si, st in enumerate(stages):
        t0 = time.time()
        if st["task"] == "joint":
            tasks_here = [t for t, _ in st["_joint"]]
            train_set = ConcatDataset([build_train_set(t, items, st["patch"]) for t, items in st["_joint"]])
        else:
            tasks_here = [st["task"]]
            train_set = build_train_set(st["task"], st["train"], st["patch"])
        for t in tasks_here:
            tid = TASKS.index(t)
            if mode == "adapters" and si > 0:
                add_task_adapters(model, t)
                adapter_keys[tid] = t
            else:
                adapter_keys.setdefault(tid, None)
            seen.append(t)
        model.to(dev)

        params = configure_trainable(model, si, tasks_here[-1], mode, shared)
        if freq_mod is not None:
            params += list(freq_mod.parameters())
        n_train = sum(p.numel() for p in params)
        print(f"\n=== stage {si}: {tasks_here}  trainable={n_train:,}  iters={st['iters']}")
        opt = torch.optim.AdamW(params, lr=st["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
        ema = EMA(model, cfg.get("ema", 0.999))
        loader = infinite(DataLoader(train_set, batch_size=st["batch"], shuffle=True,
                                     num_workers=cfg.get("workers", 4), drop_last=True))
        n_rep = int(st["batch"] * clc.get("replay_ratio", 0.25)) if replay and replay.tasks() else 0
        scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and cfg.get("amp", True)))

        model.train()
        for it in range(st["iters"]):
            batch = next(loader)
            if n_rep:
                rb = replay.sample(n_rep)
                batch = {k: torch.cat([batch[k], rb[k]]) for k in ("lq", "hq", "task")}
            for g in opt.param_groups:
                g["lr"] = lr_at(it, st["iters"], st["lr"], st.get("warmup", 1000))
            with torch.autocast(dev, enabled=(dev == "cuda" and cfg.get("amp", True))):
                loss, logs = compute_loss(model, batch, adapter_keys, teacher, freq_mod, lw, dev, fdi_norm)
                pen = ewc.penalty(model)
                if pen is not None:
                    loss = loss + pen
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            if it % cfg.get("log_every", 100) == 0:
                print(f"  it {it:6d}  loss {loss.item():.4f}  rec {logs['rec']:.4f}  lr {opt.param_groups[0]['lr']:.2e}")

        # ---- end of stage: fill replay, EWC consolidation, eval with EMA weights
        live = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        if replay is not None:
            fill_loader = DataLoader(train_set, batch_size=st["batch"], shuffle=True, num_workers=0)
            need = replay.per_task
            for b in fill_loader:
                replay.add_batch(b["lq"], b["hq"], b["task"])
                need -= len(b["lq"])
                if need <= 0:
                    break
        if ewc.lam > 0 and mode == "adapters":
            names = {n for n, p in model.named_parameters()
                     if ".A." not in n and ".B." not in n and any(fnmatch.fnmatch(n, s) for s in shared)}
            fb = [next(loader) for _ in range(clc.get("ewc_batches", 20))]
            model.eval()
            ewc.consolidate(model, lambda b: compute_loss(model, b, adapter_keys, teacher, None, {"task": lw.get("task", 0), "fdi": lw.get("fdi", 0)}, dev, fdi_norm)[0], fb, names)

        res = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="oracle", tile=cfg.get("eval_tile", 0))
        res_inf = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="inferred", tile=cfg.get("eval_tile", 0)) \
            if mode == "adapters" and len(seen) > 1 else None
        # R row: mean PSNR over that task's eval sets
        R.append([float(sum(v["psnr"] for v in res[t].values()) / len(res[t])) if t in res else float("nan")
                  for t in eval_tasks])
        history.append({"stage": si, "tasks": tasks_here, "oracle": res, "inferred": res_inf,
                        "minutes": (time.time() - t0) / 60})
        torch.save({"model": live, "ema": ema.shadow, "adapter_keys": adapter_keys, "seen_tasks": seen,
                    "teacher_dim": teacher.dim if teacher else 0, "fdi_norm": fdi_norm.state_dict() if fdi_norm else None, "replay": replay.state_dict() if replay else None,
                    "ewc": ewc.state_dict(), "cfg": cfg}, os.path.join(out_dir, f"stage{si}.pt"))
        model.load_state_dict(live)
        print("  R row (PSNR):", [round(x, 2) for x in R[-1]])

    summary = cl_summary(R) if mode != "joint" else {"ACC": R[-1]}
    json.dump({"eval_tasks": eval_tasks, "R": R, "summary": summary, "history": history},
              open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
