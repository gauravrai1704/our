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
    """Isotropic defocus-like blur (used in compound chains and as a synthetic fallback)."""
    from PIL import ImageFilter
    return _from_pil(_to_pil(hq).filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def motion_kernel(rng, size=21, length=None, steps=64):
    """Random camera-shake kernel: a smoothed random-walk trajectory rasterised on a grid.

    Simplified version of the trajectory model of Boracchi & Foi (2012), as used in
    DeblurGAN's synthetic blur. length = trajectory extent in pixels.
    """
    length = float(rng.uniform(size / 4, size - 2)) if length is None else float(length)
    v = rng.normal(size=2)
    v /= np.linalg.norm(v) + 1e-8
    pts = [np.zeros(2)]
    for _ in range(steps - 1):
        v = v + rng.normal(scale=0.35, size=2) - 0.05 * pts[-1] / (length + 1e-8)
        v /= np.linalg.norm(v) + 1e-8
        pts.append(pts[-1] + v * length / steps)
    pts = np.asarray(pts)
    pts -= (pts.max(0) + pts.min(0)) / 2
    k = np.zeros((size, size), np.float64)
    c = size // 2
    for x, y in pts:  # bilinear splat
        x0, y0 = int(np.floor(x + c)), int(np.floor(y + c))
        fx, fy = x + c - x0, y + c - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < size and 0 <= y0 + dy < size:
                k[y0 + dy, x0 + dx] += w
    return (k / k.sum()).astype(np.float32)


def convolve(hq, k):
    """Same-size 2D convolution per channel, reflect padding, via FFT."""
    p = k.shape[0] // 2
    x = np.pad(hq, ((p, p), (p, p), (0, 0)), mode="reflect")
    H, W = x.shape[:2]
    K = np.fft.rfft2(k, s=(H, W))
    y = np.fft.irfft2(np.fft.rfft2(x, axes=(0, 1)) * K[..., None], s=(H, W), axes=(0, 1))
    # the FFT product is a circular conv anchored at the kernel's corner; shift back by p
    return y[2 * p:H, 2 * p:W].astype(np.float32)


def motion_blur(hq, rng, size=21, length=None):
    return np.clip(convolve(hq, motion_kernel(rng, size, length)), 0, 1)


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
        # eval params: {"sigma": s} -> Gaussian, {"length": l} -> motion. Training: 50/50 mix.
        if "sigma" in p:
            return gaussian_blur(hq, p["sigma"])
        if "length" in p or rng.random() < 0.5:
            return motion_blur(hq, rng, p.get("size", 21), p.get("length"))
        return gaussian_blur(hq, float(rng.uniform(0.5, 3.0)))
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
