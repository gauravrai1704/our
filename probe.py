"""RQ2: does the degradation embedding (FDI) identify the task on its own?

Extracts frozen embeddings  f = model.degr(LQ)  and fits two simple classifiers on
half the images, testing on the other half:
  * linear probe (multinomial logistic regression on standardised features)
  * nearest centroid
Also reports the model's own task head. Prints confusion matrices, saves embeddings
(for a t-SNE/UMAP figure) to runs/<name>/probe/.

Content control. Use --hq_dir to apply EVERY task's degradation to the SAME clean
images; otherwise the probe can cheat by recognising the test-set content
(e.g. Set5 vs CBSD68) instead of the degradation.

    python probe.py --ckpt runs/cl_adapters/stage3.pt --hq_dir datasets/DIV2K/valid_HR
"""
import argparse
import json
import os

import numpy as np
import torch

from common import load_checkpoint
from data.datasets import list_images, load_rgb, to_tensor
from data.degrade import TASKS, degrade


@torch.no_grad()
def embed_controlled(model, hq_dir, tasks, device, crop=256, max_images=200, seed=0):
    rng = np.random.default_rng(seed)
    X, y, img = [], [], []
    for i, path in enumerate(list_images(hq_dir)[:max_images]):
        hq = load_rgb(path)
        h, w = hq.shape[:2]
        if h < crop or w < crop:
            continue
        r, c = (h - crop) // 2, (w - crop) // 2
        hq = hq[r:r + crop, c:c + crop]
        for t in tasks:
            lq = degrade(hq, t, rng)  # training-distribution parameters
            X.append(model.degr(to_tensor(lq)[None].to(device)).cpu().numpy()[0])
            y.append(TASKS.index(t))
            img.append(i)
    return np.array(X), np.array(y), np.array(img)


@torch.no_grad()
def embed_eval_sets(model, cfg, tasks, device):
    from evaluate import build_eval_sets
    X, y, img = [], [], []
    k = 0
    for t, items in build_eval_sets(cfg).items():
        if t not in tasks:
            continue
        for _, ds, _ in items:
            for b in torch.utils.data.DataLoader(ds, batch_size=1):
                X.append(model.degr(b["lq"].to(device)).cpu().numpy()[0])
                y.append(TASKS.index(t))
                img.append(k)
                k += 1
    return np.array(X), np.array(y), np.array(img)


def split_by_image(img, seed=0):
    """Train/test split on IMAGE ids so the same picture never appears on both sides."""
    ids = np.unique(img)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    tr = set(ids[: len(ids) // 2].tolist())
    m = np.array([i in tr for i in img])
    return m, ~m


def logistic_probe(Xtr, ytr, Xte, n_cls, steps=500, wd=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    lin = torch.nn.Linear(A.shape[1], n_cls)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps)
    yt = torch.tensor(ytr)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(A), yt) + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lin(B).argmax(1).numpy()


def nearest_centroid(Xtr, ytr, Xte, classes):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    C = np.stack([((Xtr[ytr == c] - mu) / sd).mean(0) for c in classes])
    d = (((Xte - mu) / sd)[:, None, :] - C[None]) ** 2
    return np.array(classes)[d.sum(-1).argmin(1)]


def confusion(y, p, classes):
    M = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, p):
        M[classes.index(a), classes.index(b)] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hq_dir", default=None, help="clean images for the content-controlled probe")
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--crop", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, keys = load_checkpoint(a.ckpt, dev)
    tasks = ck["seen_tasks"]
    if a.hq_dir:
        X, y, img = embed_controlled(model, a.hq_dir, tasks, dev, a.crop, a.max_images)
    else:
        print("WARNING: no --hq_dir; probing on eval sets (content differs per task, accuracy is optimistic)")
        X, y, img = embed_eval_sets(model, ck["cfg"], tasks, dev)
    assert len(X), f"no images usable (all smaller than --crop {a.crop}?)"
    classes = sorted(set(y.tolist()))
    tr, te = split_by_image(img)
    names = [TASKS[c] for c in classes]

    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "tasks": names}
    preds = {
        "linear_probe": logistic_probe(X[tr], y[tr], X[te], len(TASKS)),
        "nearest_centroid": nearest_centroid(X[tr], y[tr], X[te], classes),
    }
    with torch.no_grad():
        logits = model.task_head(torch.tensor(X[te], dtype=torch.float32, device=dev))[:, classes]
        preds["model_task_head"] = np.array(classes)[logits.argmax(1).cpu().numpy()]
    for k, p in preds.items():
        acc = float((p == y[te]).mean())
        M = confusion(y[te].tolist(), p.tolist(), classes)
        res[k] = {"acc": acc, "confusion": M.tolist()}
        print(f"\n{k}: acc {acc:.3f}   (rows = true, cols = predicted: {names})")
        for n, row in zip(names, M):
            print(f"  {n:8s} {row}")

    out_dir = os.path.join(os.path.dirname(a.ckpt), "probe")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "embeddings.npz"), X=X, y=y, img=img)
    json.dump(res, open(os.path.join(out_dir, "probe.json"), "w"), indent=2)
    print(f"\nwrote {out_dir}/probe.json and embeddings.npz")


if __name__ == "__main__":
    main()
