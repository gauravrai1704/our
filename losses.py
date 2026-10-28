"""Training losses.

  rec   : Charbonnier (smooth L1) on pixels                       — main loss
  fdi   : MSE( head(fdi_student(LQ)), T(LQ) - T(HQ) )             — teacher-guided FDI (needs teacher)
  freq  : low/high-band self-distillation, early block <- late block (stop-grad)   [FRAMER-style]
  task  : CE on task head (only to make FDI task-discriminative; used for RQ2 routing)
  dual  : consistency between outputs for LQ and LQ + small noise   [DNF-SR-style dual input]

The freq / dual terms are simplified adaptations of the cited ideas for a
from-scratch regression model — describe them as such, not as reimplementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def split_bands(f, k=5):
    """f: (B, H, W, C) tokens -> (low, high) via box blur in the spatial dims."""
    x = f.permute(0, 3, 1, 2)
    low = F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    return low, x - low


class FreqSelfDistill(nn.Module):
    """Align early-block bands to stop-grad late-block bands through a 1x1 projector."""

    def __init__(self, dim, w_high=2.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)
        self.w_high = w_high

    def forward(self, early, late):
        el, eh = split_bands(early)
        ll, lh = split_bands(late.detach())
        return F.l1_loss(self.proj(el), ll) + self.w_high * F.l1_loss(self.proj(eh), lh)


class RunningNorm(nn.Module):
    """Per-dimension running RMS of the FDI target T(LQ)-T(HQ).

    Normalising by the *batch* magnitude explodes when degradations are mild
    (target ~ 0). A running per-dim RMS keeps the loss scale stable across tasks.
    """

    def __init__(self, dim, momentum=0.01, eps=1e-4):
        super().__init__()
        self.register_buffer("rms", torch.ones(dim))
        self.register_buffer("init", torch.zeros(()))
        self.m, self.eps = momentum, eps

    @torch.no_grad()
    def update(self, x):
        r = x.pow(2).mean(0).sqrt()
        if self.init == 0:
            self.rms.copy_(r)
            self.init.fill_(1)
        else:
            self.rms.mul_(1 - self.m).add_(r, alpha=self.m)

    def forward(self, x):
        return x / (self.rms + self.eps)
