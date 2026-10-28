"""Evaluation: per-task PSNR/SSIM/LPIPS, with oracle or inferred task routing.

routing="oracle"   : use the ground-truth task's adapter
routing="inferred" : pick the adapter from argmax of the FDI task head over seen tasks
                     (RQ2: can the degradation embedding replace an explicit task label?)

Standalone use:
    python evaluate.py --config configs/smoke.yaml --ckpt runs/smoke/stage3.pt --routing inferred
    python evaluate.py --config configs/smoke.yaml --ckpt runs/smoke/qat/qat.pt --qat

Eval entries under a non-task key (e.g. `compound:`) are held-out mixed degradations;
they are always routed by the FDI task head, and the routing histogram is recorded.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import PairedDataset, SyntheticTaskDataset
from data.degrade import TASKS
from metrics import LPIPSMetric, psnr, ssim
from models.lora import set_active_task


def build_eval_sets(cfg):
    """cfg['eval'][task] = list of {name, type, ...}. Returns {task: [(name, dataset, scale)]}."""
    sets = {}
    for task, items in cfg.get("eval", {}).items():
        for it in items:
            if it["type"] == "synthetic":
                ds = SyntheticTaskDataset(it["hq_dir"], task, train=False, eval_params=it.get("params", {}))
                scale = it.get("params", {}).get("scale", 4)
            else:
                ds = PairedDataset(it["lq_dir"], it["hq_dir"], task, train=False, sr_scale=it.get("sr_scale"))
                scale = it.get("sr_scale") or 4
            sets.setdefault(task, []).append((it["name"], ds, scale))
    return sets


@torch.no_grad()
def tiled_forward(model, x, tile=0, overlap=32):
    if not tile or (x.shape[-1] <= tile and x.shape[-2] <= tile):
        return model(x)
    B, C, H, W = x.shape
    out, wsum = torch.zeros_like(x), torch.zeros_like(x[:, :1])
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if H > tile and (H - tile) % step else [])
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if W > tile and (W - tile) % step else [])
    for y in ys:
        for x0 in xs:
            patch = x[..., y:y + tile, x0:x0 + tile]
            out[..., y:y + tile, x0:x0 + tile] += model(patch)
            wsum[..., y:y + tile, x0:x0 + tile] += 1
    return out / wsum
    # NOTE: FDI is computed per tile here. For the paper, also report whole-image FDI.


@torch.no_grad()
def evaluate(model, eval_sets, adapter_keys, seen_tasks, device, routing="oracle", tile=0, lpips_fn=None):
    """adapter_keys: {task_id: adapter key or None}. Returns {task: {dataset: metrics}}."""
    model.eval()
    lpips_fn = lpips_fn or LPIPSMetric(device)
    seen_ids = [TASKS.index(t) for t in seen_tasks]
    results = {}
    for task, items in eval_sets.items():
        tid = TASKS.index(task) if task in TASKS else -1   # -1 = compound / unseen
        for name, ds, scale in items:
            P, S, L, hit, routed = [], [], [], [], []
            for b in DataLoader(ds, batch_size=1, num_workers=2):
                lq, hq = b["lq"].to(device), b["hq"].to(device)
                if (routing == "inferred" or tid < 0) and len(seen_ids) > 1:
                    set_active_task(model, None)
                    logits = model.task_head(model.degr(lq))[:, seen_ids]
                    pred_tid = seen_ids[int(logits.argmax(1))]
                else:
                    pred_tid = tid if tid in adapter_keys else max(adapter_keys)
                hit.append(pred_tid == tid)
                routed.append(TASKS[pred_tid])
                set_active_task(model, adapter_keys.get(pred_tid))
                out = tiled_forward(model, lq, tile)
                P += psnr(out, hq, task, scale)
                S += ssim(out, hq, task, scale)
                L += lpips_fn(out, hq)
            results.setdefault(task, {})[name] = {
                "psnr": float(np.mean(P)), "ssim": float(np.mean(S)), "lpips": float(np.nanmean(L)) if not all(np.isnan(L)) else None,
                "task_acc": float(np.mean(hit)) if tid >= 0 else None,
                "routed_to": {t: routed.count(t) for t in set(routed)}, "n": len(P)}
    set_active_task(model, None)
    return results


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--routing", default="oracle", choices=["oracle", "inferred"])
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--qat", action="store_true", help="ckpt is runs/<name>/qat/qat.pt")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.qat:
        from common import load_qat_checkpoint
        model, ck, keys = load_qat_checkpoint(a.ckpt, dev)
    else:
        from common import load_checkpoint
        model, ck, keys = load_checkpoint(a.ckpt, dev)
    res = evaluate(model, build_eval_sets(cfg), keys, ck["seen_tasks"], dev, a.routing, a.tile)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
