"""Synthetic degradations. All functions take / return float32 HxWx3 arrays in [0, 1].

Every task keeps LQ and HQ at the SAME size (SR LQ is bicubic-upsampled back),
so one network handles all tasks.

Note on SR benchmarks: PIL bicubic != MATLAB imresize. For Set5/Set14/Urban100/
Manga109 use the official pre-made LR_bicubic files via PairedDataset; use the
synthetic path here for training only.
"""
import io

import numpy as np
from PIL import Image

TASKS = ["sr", "denoise", "deblur", "jpeg"]
TASK_ID = {t: i for i, t in enumerate(TASKS)}


def _to_pil(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def _from_pil(im):
    return np.asarray(im, dtype=np.float32) / 255.0


def sr_bicubic(hq, scale):
    h, w = hq.shape[:2]
    im = _to_pil(hq)
    lr = im.resize((w // scale, h // scale), Image.BICUBIC)
    return _from_pil(lr.resize((w, h), Image.BICUBIC))


def gaussian_noise(hq, sigma, rng):
    """sigma on the 0-255 scale (15/25/50 are the standard eval levels)."""
    return hq + rng.normal(0.0, sigma / 255.0, hq.shape).astype(np.float32)  # unclipped, as in the literature


def jpeg(hq, quality):
    buf = io.BytesIO()
    _to_pil(hq).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return _from_pil(Image.open(buf).convert("RGB"))


def gaussian_blur(hq, sigma):
    """Fallback blur for smoke tests only — real deblurring training uses GoPro pairs."""
    from PIL import ImageFilter
    return _from_pil(_to_pil(hq).filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def degrade(hq, task, rng, params=None):
    """Sample a training degradation (params=None) or apply fixed eval params."""
    p = params or {}
    if task == "sr":
        return sr_bicubic(hq, p.get("scale", int(rng.choice([2, 3, 4]))))
    if task == "denoise":
        return gaussian_noise(hq, p.get("sigma", float(rng.uniform(0, 50))), rng)
    if task == "jpeg":
        return jpeg(hq, p.get("quality", int(rng.integers(10, 41))))
    if task == "deblur":
        return gaussian_blur(hq, p.get("sigma", float(rng.uniform(0.5, 3.0))))
    if task == "compound":
        return compound(hq, p["chain"], rng)
    raise ValueError(task)


def compound(hq, chain, rng):
    """Apply a fixed chain, e.g. [["deblur", {"sigma": 1.5}], ["denoise", {"sigma": 15}], ["jpeg", {"quality": 30}]].

    Used only for the held-out generalization test (RQ4). Fix the chains BEFORE training
    and never train on them.
    """
    x = hq
    for task, params in chain:
        x = degrade(x, task, rng, params)
        if task == "denoise":
            x = np.clip(x, 0, 1)  # later stages (JPEG, resize) need a valid image
    return x
