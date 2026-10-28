"""Generate (and optionally run) every config the paper needs from one base config.

    python experiments.py --base configs/cl_adapters.yaml --suite main --seeds 0 1 2          # list
    python experiments.py --base configs/cl_adapters.yaml --suite main --seeds 0 1 2 --run    # run
    python aggregate.py                                                                        # tables

Configs go to configs/generated/<variant>_s<seed>.yaml and runs to runs/<variant>_s<seed>/.
Runs whose results.json already exists are skipped, so --run can be resumed.

Suites
  main     : specialists (one per task), joint, finetune, ER, EWC, LwF, adapters-only, proposed
  ablation : the proposed method with one component changed at a time
  order    : the proposed method and finetune under reversed and shuffled task orders
"""
import argparse
import copy
import os
import subprocess
import sys

import yaml


def _cl(cfg, **kw):
    cfg["cl"] = {**cfg.get("cl", {}), **kw}
    return cfg


def _loss(cfg, **kw):
    cfg["loss"] = {**cfg.get("loss", {}), **kw}
    return cfg


def _model(cfg, **kw):
    cfg["model"] = {**cfg.get("model", {}), **kw}
    return cfg


def _no_cl(cfg, mode):
    return _cl(cfg, mode=mode, replay_per_task=0, ewc_lambda=0.0, lwf=0.0)


def _baseline(cfg, **kw):
    """finetune + one CL component. kw values must be read BEFORE _no_cl resets them."""
    return _cl(_no_cl(cfg, "finetune"), **kw)


def specialist(task):
    def f(cfg):
        st = [s for s in cfg["stages"] if s["task"] == task]
        if not st:
            return None
        cfg["stages"] = st
        cfg["eval"] = {task: cfg["eval"][task]} if task in cfg.get("eval", {}) else {}
        return _no_cl(cfg, "adapters")
    return f


def reorder(order):
    def f(cfg):
        old = cfg["stages"]
        by_task = {s["task"]: s for s in old}
        # each task keeps its data/patch settings; the schedule follows the position
        # (position 0 trains the whole base network, so it keeps the long schedule)
        sched = [{k: s[k] for k in ("iters", "lr", "warmup") if k in s} for s in old]
        cfg["stages"] = [{**by_task[t], **sched[i]} for i, t in enumerate(t for t in order if t in by_task)]
        return cfg
    return f


def _then(*fs):
    def f(cfg):
        for g in fs:
            cfg = g(cfg)
        return cfg
    return f


SUITES = {
    "main": {
        **{f"specialist_{t}": specialist(t) for t in ("sr", "denoise", "deblur", "jpeg")},
        "joint": lambda c: _no_cl(c, "joint"),
        "finetune": lambda c: _no_cl(c, "finetune"),
        "er": lambda c: _baseline(c, replay_per_task=c["cl"].get("replay_per_task") or 500),
        "ewc": lambda c: _baseline(c, ewc_lambda=c["cl"].get("ewc_lambda") or 100.0),
        "lwf": lambda c: _baseline(c, lwf=1.0),
        "adapters_only": lambda c: _no_cl(c, "adapters"),
        "proposed": lambda c: c,
    },
    "ablation": {
        "abl_no_replay": lambda c: _cl(c, replay_per_task=0),
        "abl_no_ewc": lambda c: _cl(c, ewc_lambda=0.0),
        "abl_replay100": lambda c: _cl(c, replay_per_task=100),
        "abl_replay1000": lambda c: _cl(c, replay_per_task=1000),
        "abl_ewc10": lambda c: _cl(c, ewc_lambda=10.0),
        "abl_ewc1000": lambda c: _cl(c, ewc_lambda=1000.0),
        "abl_lwf": lambda c: _cl(c, lwf=1.0),
        "abl_lora_backbone": lambda c: _model(c, lora_backbone=True),
        "abl_rank4": lambda c: _model(c, lora_rank=4),
        "abl_rank16": lambda c: _model(c, lora_rank=16),
        "abl_frozen_encoder": lambda c: _cl(c, shared_trainable=["task_head.*"]),
        "abl_no_teacher": lambda c: {**c, "teacher": "none"},
        "abl_no_freq": lambda c: _loss(c, freq=0.0),
        "abl_no_dual": lambda c: _loss(c, dual=0.0),
        "abl_no_taskce": lambda c: _loss(c, task=0.0),
    },
    "order": {
        "order_rev_proposed": reorder(["jpeg", "deblur", "denoise", "sr"]),
        "order_rev_finetune": _then(reorder(["jpeg", "deblur", "denoise", "sr"]), lambda c: _no_cl(c, "finetune")),
        "order_mix_proposed": reorder(["denoise", "jpeg", "sr", "deblur"]),
        "order_mix_finetune": _then(reorder(["denoise", "jpeg", "sr", "deblur"]), lambda c: _no_cl(c, "finetune")),
    },
}


def build(base, suite, seeds, scale=1.0):
    out = []
    names = list(SUITES) if suite == "all" else [suite]
    for sname in names:
        for vname, fn in SUITES[sname].items():
            for seed in seeds:
                cfg = fn(copy.deepcopy(base))
                if cfg is None:
                    continue
                cfg["name"] = f"{vname}_s{seed}"
                cfg["seed"] = seed
                cfg["variant"], cfg["suite"] = vname, sname
                if scale != 1.0:  # quick pilot runs: shrink every schedule
                    for st in cfg["stages"]:
                        st["iters"] = max(1, int(st["iters"] * scale))
                        st["warmup"] = max(1, int(st.get("warmup", 1000) * scale))
                out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--suite", default="main", choices=list(SUITES) + ["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*", default=None, help="variant names to keep")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all iters (pilot runs)")
    ap.add_argument("--out", default="configs/generated")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--qat", action="store_true", help="also run qat.py after each training run")
    a = ap.parse_args()
    base = yaml.safe_load(open(a.base))
    os.makedirs(a.out, exist_ok=True)
    cfgs = build(base, a.suite, a.seeds, a.scale)
    if a.only:
        cfgs = [c for c in cfgs if c["variant"] in a.only]
    for cfg in cfgs:
        path = os.path.join(a.out, cfg["name"] + ".yaml")
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        done = os.path.exists(os.path.join("runs", cfg["name"], "results.json"))
        print(f"{'[done] ' if done else ''}{path}")
        if not a.run or done:
            continue
        subprocess.run([sys.executable, "train.py", "--config", path], check=True)
        if a.qat and cfg["cl"].get("mode") != "joint":
            subprocess.run([sys.executable, "qat.py", "--config", path], check=True)


if __name__ == "__main__":
    main()
