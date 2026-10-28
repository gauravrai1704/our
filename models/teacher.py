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


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Blip2QFormerTeacher(nn.Module):
    """BLIP-2 ViT-g + Q-Former (the FDI extractor of our ICVGIP model [1] and of DiffRes [3]).

    Uses the image-text-matching checkpoint, which holds the vision encoder and Q-Former
    without the LLM. Feature = mean and std over the 32 projected query embeddings (512-d).
    """

    def __init__(self, ckpt="Salesforce/blip2-itm-vit-g", size=224):
        super().__init__()
        from transformers import Blip2VisionModelWithProjection
        self.model = Blip2VisionModelWithProjection.from_pretrained(ckpt)
        self.size = size
        self.dim = 2 * self.model.config.image_text_hidden_size
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x.float(), size=(self.size, self.size), mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - _CLIP_MEAN.to(x)) / _CLIP_STD.to(x)
        q = self.model(pixel_values=x).image_embeds  # (B, 32, 256)
        return torch.cat([q.mean(1), q.std(1)], 1)


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
    if name.startswith("blip2"):  # "blip2" or "blip2:<hf checkpoint>"
        return Blip2QFormerTeacher(*name.split(":", 1)[1:])
    if name == "random":
        return RandomConvTeacher()
    raise ValueError(name)
