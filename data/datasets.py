"""Datasets + rehearsal buffer.

SyntheticTaskDataset : HQ folder -> (LQ, HQ) by on-the-fly degradation (SR/denoise/JPEG)
PairedDataset        : LQ folder + HQ folder matched by sorted filename (GoPro, SIDD,
                       RealSR, benchmark LR_bicubic, ...)
ReplayBuffer         : fixed-size per-task reservoir of (LQ, HQ) patches kept on CPU
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degrade import TASK_ID, degrade, sr_bicubic

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def list_images(d):
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(d, "**", e), recursive=True)
    return sorted(files)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def augment(*arrs, rng):
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    out = []
    for a in arrs:
        a = np.rot90(a, k)
        out.append(a[:, ::-1] if flip else a)
    return out


def random_crop(rng, patch, *arrs):
    h, w = arrs[0].shape[:2]
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than patch {patch}")
    y, x = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
    return [a[y:y + patch, x:x + patch] for a in arrs]


class SyntheticTaskDataset(Dataset):
    def __init__(self, hq_dir, task, patch=128, train=True, eval_params=None, repeat=1, seed=0):
        self.files = list_images(hq_dir)
        assert self.files, f"no images in {hq_dir}"
        self.task, self.patch, self.train = task, patch, train
        self.eval_params, self.repeat, self.seed = eval_params, repeat, seed

    def __len__(self):
        return len(self.files) * (self.repeat if self.train else 1)

    def __getitem__(self, i):
        path = self.files[i % len(self.files)]
        hq = load_rgb(path)
        if self.train:
            rng = np.random.default_rng()
            (hq,) = random_crop(rng, self.patch, hq)
            (hq,) = augment(hq, rng=rng)
            lq = degrade(np.ascontiguousarray(hq), self.task, rng)
        else:
            rng = np.random.default_rng(self.seed + i)  # deterministic eval degradation
            if self.task == "sr":  # make HQ divisible by scale so LR/HR align
                s = self.eval_params["scale"]
                hq = hq[: hq.shape[0] // s * s, : hq.shape[1] // s * s]
            lq = degrade(hq, self.task, rng, self.eval_params)
        return {"lq": to_tensor(lq), "hq": to_tensor(hq), "task": TASK_ID.get(self.task, -1),
                "name": os.path.basename(path)}


class PairedDataset(Dataset):
    """lq_dir/hq_dir with matching sorted filenames.

    sr_scale: if set, LQ is low-res and gets bicubic-upsampled to HQ size
    (benchmark LR_bicubic folders). For pre-aligned pairs (GoPro, SIDD) leave None.
    """

    def __init__(self, lq_dir, hq_dir, task, patch=128, train=True, sr_scale=None, repeat=1):
        self.lq, self.hq = list_images(lq_dir), list_images(hq_dir)
        assert self.lq and len(self.lq) == len(self.hq), f"{lq_dir}: {len(self.lq)} vs {hq_dir}: {len(self.hq)}"
        self.task, self.patch, self.train, self.sr_scale, self.repeat = task, patch, train, sr_scale, repeat

    def __len__(self):
        return len(self.lq) * (self.repeat if self.train else 1)

    def __getitem__(self, i):
        j = i % len(self.lq)
        lq, hq = load_rgb(self.lq[j]), load_rgb(self.hq[j])
        if self.sr_scale:
            s = self.sr_scale
            hq = hq[: lq.shape[0] * s, : lq.shape[1] * s]
            from PIL import Image as _I
            im = _I.fromarray((lq * 255 + 0.5).astype(np.uint8))
            lq = np.asarray(im.resize((hq.shape[1], hq.shape[0]), _I.BICUBIC), dtype=np.float32) / 255.0
        assert lq.shape == hq.shape, f"shape mismatch {self.lq[j]} {lq.shape} vs {hq.shape}"
        if self.train:
            rng = np.random.default_rng()
            lq, hq = random_crop(rng, self.patch, lq, hq)
            lq, hq = augment(lq, hq, rng=rng)
        return {"lq": to_tensor(lq), "hq": to_tensor(hq), "task": TASK_ID.get(self.task, -1),
                "name": os.path.basename(self.hq[j])}


class ReplayBuffer:
    """Per-task reservoir. Stores fixed patches (LQ+HQ) so replay is exactly reproducible."""

    def __init__(self, per_task=500):
        self.per_task = per_task
        self.store = {}   # task_id -> list of (lq, hq)
        self.seen = {}

    def add_batch(self, lq, hq, task_ids):
        for l, h, t in zip(lq.cpu(), hq.cpu(), task_ids.tolist()):
            buf = self.store.setdefault(t, [])
            n = self.seen.get(t, 0)
            if len(buf) < self.per_task:
                buf.append((l.clone(), h.clone()))
            else:
                k = random.randint(0, n)
                if k < self.per_task:
                    buf[k] = (l.clone(), h.clone())
            self.seen[t] = n + 1

    def tasks(self):
        return sorted(self.store)

    def sample(self, n, exclude=()):
        pool = [(t, x) for t in self.store if t not in exclude for x in self.store[t]]
        if not pool:
            return None
        pick = random.sample(pool, min(n, len(pool)))
        return {"lq": torch.stack([p[1][0] for p in pick]), "hq": torch.stack([p[1][1] for p in pick]),
                "task": torch.tensor([p[0] for p in pick])}

    def state_dict(self):
        return {"per_task": self.per_task, "store": self.store, "seen": self.seen}

    def load_state_dict(self, s):
        self.per_task, self.store, self.seen = s["per_task"], s["store"], s["seen"]
