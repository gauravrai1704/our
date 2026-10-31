"""Sequential (continual) training: SR -> denoise -> deblur -> JPEG.

    python train.py --config configs/smoke.yaml

cl.mode:
  adapters : stage 0 trains everything; later stages train LoRA adapters + shared
             degradation encoder/heads, with replay (+ optional EWC)        [proposed]
  finetune : full fine-tuning per stage, no adapters                        [lower bound]
  joint    : one stage on all tasks at once, no adapters                    [joint baseline]
  (specialists = run with a single-stage config per task)

Replay (cl.replay_per_task), EWC / MAS (cl.ewc_lambda, cl.importance: fisher | mas)
and LwF (cl.lwf) work in both adapters and finetune mode, so finetune + one of them
gives the classic ER / EWC / MAS / LwF baselines (see experiments.py).

After every stage, all tasks in cfg.eval are evaluated -> R matrix in runs/<name>/results.json

qat.continual (INT8 after every task, RQ3):
  none      : FP32 continual training; quantize once at the end with qat.py
  snapshot  : after every stage, copy the FP32 model -> PTQ calibration on replay patches of
              all seen tasks -> qat.stage_iters of task-balanced QAT with KD from FP32 ->
              evaluate. Gives an INT8 R matrix (R_int8) next to the FP32 one; FP32 training
              continues from the FP32 weights. Saves stage<k>_int8.pt.
  quantized : the model itself is INT8 from task 1 on. After stage 0 it is converted to
              fake-quant (calibration + qat.stage_iters QAT); every later task is learned
              directly in fake-quant (new adapters get their own observers, all others stay
              frozen). This is the deploy-once setting: the INT8 base never changes and each
              new task ships as INT8 adapters. R is then the INT8 R matrix.
"""
import argparse
import copy
import fnmatch
import json
import math
import os
import random
import time

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from cl import EWC, configure_trainable
from qat import DEFAULTS as QAT_DEFAULTS
from qat import calibrate, qat_params, qat_train
from data.datasets import PairedDataset, ReplayBuffer, SyntheticTaskDataset
from data.degrade import TASKS
from evaluate import build_eval_sets, evaluate
from losses import FreqSelfDistill, RunningNorm, charbonnier
from metrics import NRMetrics, cl_summary
from models.lora import adapter_parameters, add_task_adapters, set_active_task
from models.net import RestoreNet, count_params, update_prototypes
from models.quant import prepare_qat, set_act_mode
from models.teacher import build_teacher
from models.lora import lora_modules


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
    """EMA of the parameters. Buffers (quantizer ranges, prototypes) are copied, not averaged:
    an observer range that starts at +-inf would otherwise stay inf in the shadow."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.params = {n for n, _ in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.params and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def lr_at(it, total, base, warmup):
    if it < warmup:
        return base * (it + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (it - warmup) / max(1, total - warmup)))


# ---------------------------------------------------------------- one step
def compute_loss(model, batches, adapter_keys, teacher, freq_mod, lw, device, fdi_norm=None, lwf=None):
    """batches: list of batches (current task + replay; patch sizes may differ between them).

    Each task group is forwarded with its own adapter; group losses are weighted by
    their share of all samples. lwf = (prev_model, old_keys, weight): L1 distillation of
    the previous stage's outputs under each old adapter, on the current-task inputs.
    """
    n_total = sum(len(b["lq"]) for b in batches)
    # the task CE is class-balanced (each task group weighs 1/G): weighting by sample share
    # would let the current task dominate and bias routing toward the newest task
    n_groups = sum(len(b["task"].unique()) for b in batches)
    logs, total = {"rec": 0.0}, 0.0
    for batch in batches:
        lq, hq, tid = batch["lq"].to(device), batch["hq"].to(device), batch["task"].to(device)
        for t in tid.unique().tolist():
            sel = tid == t
            w = sel.sum().item() / n_total
            set_active_task(model, adapter_keys.get(t))
            x, y = lq[sel], hq[sel]
            out, aux = model(x, return_feats=True)
            loss = charbonnier(out, y)
            logs["rec"] += loss.item() * w

            if lw.get("task", 0) > 0:
                ce = torch.nn.functional.cross_entropy(model.task_head(aux["fdi"]), tid[sel])
                total = total + lw["task"] * ce / n_groups
            if teacher is not None and lw.get("fdi", 0) > 0:
                # fdi_max: teacher on only the first k samples of each group (BLIP-2 ViT-g is
                # ~100x the student's FLOPs; this keeps [1]-style runs affordable)
                k = lw.get("fdi_max") or len(x)
                with torch.no_grad():
                    target = teacher(x[:k]) - teacher(y[:k])
                if model.training:
                    fdi_norm.update(target)
                pred = model.fdi_head(aux["fdi"][:k])
                fl = torch.nn.functional.mse_loss(pred, fdi_norm(target))
                loss = loss + lw["fdi"] * fl
            if freq_mod is not None and lw.get("freq", 0) > 0:
                f = aux["feats"]
                loss = loss + lw["freq"] * freq_mod(f[len(f) // 2 - 1], f[-1])
            if lw.get("dual", 0) > 0:
                x2 = x + torch.randn_like(x) * (lw.get("dual_sigma", 2.0) / 255.0)
                out2 = model(x2)
                loss = loss + lw["dual"] * (charbonnier(out2, y) + charbonnier(out2, out.detach()))
            total = total + loss * w
    if lwf is not None:
        prev, old_keys, wt = lwf
        x = batches[0]["lq"].to(device)
        dl = 0.0
        for k in old_keys:
            set_active_task(model, k)
            set_active_task(prev, k)
            with torch.no_grad():
                ref = prev(x)
            dl = dl + torch.nn.functional.l1_loss(model(x), ref)
        set_active_task(prev, None)
        dl = dl / len(old_keys)
        logs["lwf"] = dl.item()
        total = total + wt * dl
    set_active_task(model, None)
    return total, logs


# ---------------------------------------------------------------- INT8 helpers
def stage_patches(replay, train_set, per_task):
    """{task_id: (lq, hq)} for calibration / QAT: the replay buffer (all seen tasks) if there
    is one, else patches of the current stage only (no access to old data)."""
    if replay is not None and replay.store:
        return {t: (torch.stack([l for l, _ in v]), torch.stack([h for _, h in v])) for t, v in replay.store.items()}
    out = {}
    for b in DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0):
        for t in b["task"].unique().tolist():
            sel = b["task"] == t
            lq, hq = out.get(t, (b["lq"][:0], b["hq"][:0]))
            out[t] = (torch.cat([lq, b["lq"][sel]])[:per_task], torch.cat([hq, b["hq"][sel]])[:per_task])
        if all(len(v[0]) >= per_task for v in out.values()):
            break
    return out


def new_observers(model, task):
    """Activation quantizers created for a new task's adapters (QTaskLoRALinear.mid_q[task])."""
    return [m.mid_q[task] for m in lora_modules(model) if hasattr(m, "mid_q") and task in m.mid_q]


def r_row(res, eval_tasks):
    """R row: mean PSNR over each task's eval sets."""
    return [float(sum(v["psnr"] for v in res[t].values()) / len(res[t])) if t in res else float("nan")
            for t in eval_tasks]


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))
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
    replay = ReplayBuffer(clc.get("replay_per_task", 0)) if mode != "joint" and clc.get("replay_per_task", 0) else None
    ewc = EWC(clc.get("ewc_lambda", 0.0), clc.get("importance", "fisher"))
    shared = clc.get("shared_trainable", ["degr.*", "task_head.*", "fdi_head.*"])

    stages = cfg["stages"]
    if mode == "joint":  # merge all stages into one multi-task stage
        stages = [{**stages[0], "task": "joint", "train": None,
                   "_joint": [(s["task"], s["train"]) for s in stages],
                   "iters": sum(s["iters"] for s in stages)}]

    qc = {**QAT_DEFAULTS, "stage_iters": 1000, "continual": "none", **cfg.get("qat", {})}
    qmode = qc["continual"] if mode != "joint" else "none"
    assert qmode in ("none", "snapshot", "quantized"), qmode
    quantized = False  # True once the model itself is fake-quant (qmode == "quantized")

    adapter_keys, seen, R, R_int8, history = {}, [], [], [], []
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
        n_rep = max(1, int(st["batch"] * clc.get("replay_ratio", 0.25))) if replay and replay.tasks() else 0
        lwf = None
        if clc.get("lwf", 0) > 0 and si > 0 and mode != "joint":
            prev = copy.deepcopy(model).eval()
            for p in prev.parameters():
                p.requires_grad_(False)
            old_keys = sorted({adapter_keys[TASKS.index(t)] for t in seen[:-1]}, key=lambda k: k or "")
            lwf = (prev, old_keys, clc["lwf"])
        # fake-quant ops are run in FP32, so AMP is off once the model is quantized
        amp = dev == "cuda" and cfg.get("amp", True) and not quantized
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        fresh_obs = [q for t in tasks_here for q in new_observers(model, t)] if quantized else []
        for q in fresh_obs:
            q.mode = "observe_quant"

        model.train()
        for it in range(st["iters"]):
            batches = [next(loader)]
            if n_rep:
                batches.append(replay.sample(n_rep))
            for g in opt.param_groups:
                g["lr"] = lr_at(it, st["iters"], st["lr"], st.get("warmup", 1000))
            if fresh_obs and it == min(qc["observer_warmup"], st["iters"] - 1):
                for q in fresh_obs:  # the new adapters' activation ranges are now fixed
                    q.mode = "quant"
            with torch.autocast(dev, enabled=amp):
                loss, logs = compute_loss(model, batches, adapter_keys, teacher, freq_mod, lw, dev, fdi_norm, lwf)
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
                extra = f"  lwf {logs['lwf']:.4f}" if "lwf" in logs else ""
                print(f"  it {it:6d}  loss {loss.item():.4f}  rec {logs['rec']:.4f}{extra}  lr {opt.param_groups[0]['lr']:.2e}")

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
        # NCM prototypes: from the replay buffer when there is one (old tasks re-embedded with
        # the current encoder), otherwise only the tasks of this stage are (re)computed
        if replay is not None:
            proto_src = {t: torch.stack([l for l, _ in v]) for t, v in replay.store.items()}
        else:
            proto_src = {}
            for b in DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0):
                for t in b["task"].unique().tolist():
                    proto_src.setdefault(t, []).append(b["lq"][b["task"] == t])
                if sum(len(x) for v in proto_src.values() for x in v) >= 256 * len(tasks_here):
                    break
            proto_src = {t: torch.cat(v) for t, v in proto_src.items()}
        model.eval()
        update_prototypes(model, proto_src)
        for k in ("protos", "proto_n"):
            ema.shadow[k] = live[k] = model.state_dict()[k].clone()

        if qmode == "snapshot" or (qmode == "quantized" and not quantized):
            patches = stage_patches(replay, train_set, qc["per_task"])
        if qmode == "quantized" and not quantized:
            # convert the model itself to INT8 fake-quant (stage 0, or the first stage of a run)
            fp_ref = copy.deepcopy(model).eval()
            prepare_qat(model, qc["keep_fp"], qc["quant_attn"])
            model.to(dev)
            calibrate(model, patches, adapter_keys, dev, qc["calib_batches"])
            print(f"  [INT8] converting model: {qc['stage_iters']} QAT iters on {sorted(patches)}")
            qat_train(model, fp_ref, patches, adapter_keys, qc, dev, qat_params(model, "all"), qc["stage_iters"])
            set_act_mode(model, "quant")
            update_prototypes(model, proto_src)  # QAT moved the encoder a little
            quantized, fp_ref = True, None
            live = copy.deepcopy(model.state_dict())
            ema = EMA(model, cfg.get("ema", 0.999))

        if ewc.lam > 0 and mode != "joint":
            # adapters: protect the shared trainable params; finetune: protect everything
            protect = shared if mode == "adapters" else ["*"]
            names = {n for n, p in model.named_parameters()
                     if ".A." not in n and ".B." not in n and any(fnmatch.fnmatch(n, s) for s in protect)}
            fb = [next(loader) for _ in range(clc.get("ewc_batches", 20))]
            model.eval()
            if ewc.importance == "mas":
                def imp_loss(b):
                    set_active_task(model, adapter_keys.get(int(b["task"][0])))
                    out = model(b["lq"].to(dev))
                    set_active_task(model, None)
                    return out.pow(2).sum((1, 2, 3)).mean()
            else:
                def imp_loss(b):
                    return compute_loss(model, [b], adapter_keys, teacher, None,
                                        {"task": lw.get("task", 0), "fdi": lw.get("fdi", 0)}, dev, fdi_norm)[0]
            ewc.consolidate(model, imp_loss, fb, names)

        # no-reference metrics only on the final stage (slow); oracle routing
        nr = NRMetrics(cfg.get("nr_metrics"), dev) if cfg.get("nr_metrics") and si == len(stages) - 1 else None
        res = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="oracle", tile=cfg.get("eval_tile", 0), nr_fn=nr)
        multi = mode == "adapters" and len(seen) > 1
        res_inf = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="inferred", tile=cfg.get("eval_tile", 0)) \
            if multi else None
        res_soft = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="soft", tile=cfg.get("eval_tile", 0)) \
            if multi else None
        res_ncm = evaluate(model, eval_sets, adapter_keys, seen, dev, routing="ncm", tile=cfg.get("eval_tile", 0)) \
            if multi else None
        R.append(r_row(res, eval_tasks))
        res_q = res_q_ncm = None
        if qmode == "snapshot":
            q = copy.deepcopy(model)
            prepare_qat(q, qc["keep_fp"], qc["quant_attn"])
            q.to(dev)
            calibrate(q, patches, adapter_keys, dev, qc["calib_batches"])
            if qc["stage_iters"] > 0:
                print(f"  [INT8 snapshot] {qc['stage_iters']} QAT iters on {sorted(patches)}")
                qat_train(q, model, patches, adapter_keys, qc, dev, qat_params(q, qc["train"]), qc["stage_iters"])
            set_act_mode(q, "quant")
            res_q = evaluate(q, eval_sets, adapter_keys, seen, dev, routing="oracle", tile=cfg.get("eval_tile", 0))
            res_q_ncm = evaluate(q, eval_sets, adapter_keys, seen, dev, routing="ncm", tile=cfg.get("eval_tile", 0)) \
                if multi else None
            R_int8.append(r_row(res_q, eval_tasks))
            torch.save({"state_dict": q.state_dict(), "adapter_keys": adapter_keys, "seen_tasks": seen,
                        "keep_fp": qc["keep_fp"], "quant_attn": qc["quant_attn"], "cfg": cfg,
                        "teacher_dim": teacher.dim if teacher else 0}, os.path.join(out_dir, f"stage{si}_int8.pt"))
            print("  R_int8 row (PSNR):", [round(x, 2) for x in R_int8[-1]])
            del q
        history.append({"stage": si, "tasks": tasks_here, "oracle": res, "inferred": res_inf, "ncm": res_ncm, "soft": res_soft,
                        "int8_oracle": res_q, "int8_ncm": res_q_ncm, "minutes": (time.time() - t0) / 60})
        torch.save({"model": live, "ema": ema.shadow, "adapter_keys": adapter_keys, "seen_tasks": seen,
                    "teacher_dim": teacher.dim if teacher else 0, "fdi_norm": fdi_norm.state_dict() if fdi_norm else None, "replay": replay.state_dict() if replay else None,
                    "ewc": ewc.state_dict(), "cfg": cfg,
                    "quantized": {"keep_fp": qc["keep_fp"], "quant_attn": qc["quant_attn"]} if quantized else None},
                   os.path.join(out_dir, f"stage{si}.pt"))
        model.load_state_dict(live)
        lwf = prev = None  # free the previous-stage copy
        print("  R row (PSNR):", [round(x, 2) for x in R[-1]])

    summary = cl_summary(R) if mode != "joint" else {"ACC": R[-1]}
    heads = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("task_head", "fdi_head")))
    params = {"deployable": count_params(model) - heads,
              "per_adapter": {k: sum(p.numel() for p in adapter_parameters(model, k)) for k in adapter_keys.values() if k}}
    out = {"eval_tasks": eval_tasks, "R": R, "summary": summary, "history": history, "params": params,
           "qat_continual": qmode}  # with "quantized", R itself is the INT8 R matrix
    if R_int8:
        out["R_int8"], out["summary_int8"] = R_int8, cl_summary(R_int8)
        print("summary INT8 (snapshot):", out["summary_int8"])
    json.dump(out,
              open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
