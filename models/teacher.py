"""Frozen teachers for the FDI-regression target  T(LQ) - T(HQ).

The teacher is used at TRAIN time only; it is never part of the deployed model,
so it does not count toward params / GFLOPs / INT8 size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoTeacher(nn.Module):
    """DINOv2 CLS + mean patch token. variant: dinov2_vits14 (384-d) / dinov2_vitb14 (768-d)."""

    def __init__(self, variant="dinov2_vits14", size=224):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", variant)
        self.size = size
        self.dim = 2 * self.model.embed_dim
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False)
        x = (x - _MEAN.to(x)) / _STD.to(x)
        out = self.model.forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1)


class RandomConvTeacher(nn.Module):
    """Stand-in for smoke tests / offline runs. NOT a meaningful teacher."""

    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 4, 4), nn.GELU(), nn.Conv2d(32, dim, 4, 4), nn.GELU())
        self.dim = dim
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f = self.net(x)
        return f.mean((2, 3))


def build_teacher(name):
    if name in (None, "none"):
        return None
    if name.startswith("dinov2"):
        return DinoTeacher(name)
    if name == "random":
        return RandomConvTeacher()
    raise ValueError(name)
