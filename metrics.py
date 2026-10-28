"""Evaluation protocol + continual-learning metrics.

PROTOCOL (fixed for the whole project; state it in the paper):
  * images quantised to uint8 before scoring (as saved PNGs would be)
  * SR, JPEG   : PSNR/SSIM on Y channel (BT.601, MATLAB rgb2ycbcr), crop `scale` px border (JPEG: 0)
  * denoise, deblur : PSNR/SSIM on RGB, no crop
  * SSIM: 11x11 Gaussian window, sigma 1.5, K=(0.01, 0.03) — same as BasicSR / MATLAB ssim
  * LPIPS (AlexNet) on RGB in [-1, 1], if the `lpips` package is installed
Before trusting any number: reproduce one published baseline with this file.
"""
import numpy as np
import torch
import torch.nn.functional as F

PROTOCOL = {
    "sr": {"y": True, "crop": "scale"},
    "jpeg": {"y": True, "crop": 0},
    "denoise": {"y": False, "crop": 0},
    "deblur": {"y": False, "crop": 0},
    "compound": {"y": False, "crop": 0},   # held-out mixed degradations (RQ4)
}


def quantize(x):
    """float tensor (B,3,H,W) in [0,1] -> float64 tensor on 0-255 grid."""
    return (x.clamp(0, 1) * 255.0).round().double()


def rgb_to_y(x255):
    r, g, b = x255[:, 0], x255[:, 1], x255[:, 2]
    return (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0)[:, None]


def _prep(pred, gt, task, scale=4):
    p, g = quantize(pred), quantize(gt)
    cfg = PROTOCOL[task]
    if cfg["y"]:
        p, g = rgb_to_y(p), rgb_to_y(g)
    c = scale if cfg["crop"] == "scale" else cfg["crop"]
    if c:
        p, g = p[..., c:-c, c:-c], g[..., c:-c, c:-c]
    return p, g


def psnr(pred, gt, task, scale=4):
    p, g = _prep(pred, gt, task, scale)
    mse = ((p - g) ** 2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).tolist()


def _gauss(size=11, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float64) - size // 2
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return (k[:, None] @ k[None, :])[None, None]


def ssim(pred, gt, task, scale=4):
    p, g = _prep(pred, gt, task, scale)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    w = _gauss().to(p.device)
    out = []
    for ch in range(p.shape[1]):  # per-channel, valid conv (no padding), then mean
        x, y = p[:, ch:ch + 1], g[:, ch:ch + 1]
        mx, my = F.conv2d(x, w), F.conv2d(y, w)
        sxx = F.conv2d(x * x, w) - mx ** 2
        syy = F.conv2d(y * y, w) - my ** 2
        sxy = F.conv2d(x * y, w) - mx * my
        m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
        out.append(m.mean(dim=(1, 2, 3)))
    return torch.stack(out, 1).mean(1).tolist()


class LPIPSMetric:
    def __init__(self, device):
        try:
            import lpips
            self.fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except ImportError:
            self.fn = None

    @torch.no_grad()
    def __call__(self, pred, gt):
        if self.fn is None:
            return [float("nan")] * pred.shape[0]
        p = quantize(pred).float() / 127.5 - 1
        g = quantize(gt).float() / 127.5 - 1
        return self.fn(p, g).flatten().tolist()


class NRMetrics:
    """No-reference IQA via pyiqa (MUSIQ, MANIQA, CLIPIQA: higher is better; NIQE: lower).

    The generative SR baselines (FluxIR, DiffRes, VOSR, FRAMER, DNF-SR) report these on
    RealSR / DRealSR, so we need them to compare on their terms. Optional dependency.
    """

    def __init__(self, names, device):
        self.fns = {}
        if not names:
            return
        try:
            import pyiqa
        except ImportError:
            print("pyiqa not installed; skipping no-reference metrics")
            return
        for n in names:
            self.fns[n] = pyiqa.create_metric(n, device=device, as_loss=False)

    @torch.no_grad()
    def __call__(self, pred):
        x = quantize(pred).float() / 255.0
        return {n: f(x).flatten().tolist() for n, f in self.fns.items()}


class Scorer:
    """Accumulates every metric for one dataset under the fixed protocol."""

    def __init__(self, task, scale, lpips_fn=None, nr_fn=None):
        self.task, self.scale, self.lpips_fn, self.nr_fn = task, scale, lpips_fn, nr_fn
        self.vals = {"psnr": [], "ssim": [], "lpips": []}

    def add(self, pred, gt):
        self.vals["psnr"] += psnr(pred, gt, self.task, self.scale)
        self.vals["ssim"] += ssim(pred, gt, self.task, self.scale)
        if self.lpips_fn is not None:
            self.vals["lpips"] += self.lpips_fn(pred, gt)
        if self.nr_fn is not None:
            for n, v in self.nr_fn(pred).items():
                self.vals.setdefault(n, []).extend(v)

    def summary(self):
        out = {}
        for k, v in self.vals.items():
            v = [x for x in v if not np.isnan(x)]
            out[k] = float(np.mean(v)) if v else None
        out["n"] = len(self.vals["psnr"])
        return out


# ---------------------------------------------------------------- continual learning
def cl_summary(R, specialist=None):
    """R[i][j] = PSNR on task j after finishing training stage i (T x T, lower triangle used).

    ACC : mean final-row performance
    BWT : mean_{j<T-1} (R[T-1][j] - R[j][j])          (negative = forgetting, in dB)
    FGT : mean_{j<T-1} (max_{i<T-1} R[i][j] - R[T-1][j])
    rel : optional per-task drop vs specialist model (dB), comparable across tasks
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    out = {"ACC": float(np.nanmean(R[-1]))}
    if T > 1:
        out["BWT"] = float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))
        out["FGT"] = float(np.mean([np.nanmax(R[j:T - 1, j]) - R[-1, j] for j in range(T - 1)]))
    if specialist is not None:
        out["drop_vs_specialist"] = (R[-1] - np.asarray(specialist, float)).tolist()
    return out
