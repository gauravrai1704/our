"""Compact degradation-conditioned restoration network (backbone-free).

Pipeline (all tasks share one input/output resolution; for SR the LQ image is
bicubic-upsampled to HR size first, so one network covers all four tasks):

    LQ ──► DegradationEncoder ──► fdi (B, d_fdi)  ─┬─► SE gates in every block
     │                                              └─► AdaIN in the decoder
     └─► stem (pixel-unshuffle x4) ─► 8 x GatedBlock ─► decoder (x4 up) ─► + LQ

FDI target. At train time we regress the student embedding onto a feature
*difference*  T(LQ) - T(HQ)  from a frozen teacher T (e.g. DINOv2). At inference
the student only sees LQ, so train and test use the same code path (no
"running mean of HQ features" approximation).

Adapters. SE-gate MLPs and AdaIN MLPs are TaskLoRALinear layers; each new task
adds its own low-rank pair there (see models/lora.py). lora_backbone=True also puts
adapters on attention qkv/proj and FFN linears (more capacity per task; ablate both).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import TaskLoRALinear


# ---------------------------------------------------------------- conditioning
class DegradationEncoder(nn.Module):
    """Small CNN: LQ image -> degradation embedding. Computed once per image."""

    def __init__(self, d_fdi=256, width=64):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, 2, 1), nn.GELU(),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.GELU(),
        )
        # mean + std pooling keeps degradation statistics (noise level, blur), not just content
        self.mlp = nn.Sequential(nn.Linear(8 * c, 2 * d_fdi), nn.GELU(), nn.Linear(2 * d_fdi, d_fdi))

    def forward(self, x):
        f = self.net(x)
        pooled = torch.cat([f.mean((2, 3)), f.flatten(2).std(2)], 1)
        return self.mlp(pooled)


class SEGate(nn.Module):
    """Channel gate from the FDI vector: sigmoid(W2 relu(W1 fdi)). LoRA-adaptable."""

    def __init__(self, d_fdi, dim, r=32, lora_rank=4):
        super().__init__()
        self.fc1 = TaskLoRALinear(d_fdi, r, rank=lora_rank)
        self.fc2 = TaskLoRALinear(r, dim, rank=lora_rank)
        nn.init.zeros_(self.fc2.base.weight)
        nn.init.constant_(self.fc2.base.bias, 2.0)  # starts near-open (sigmoid(2)=0.88)

    def forward(self, fdi):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(fdi))))


class AdaIN(nn.Module):
    """gamma(fdi) * InstanceNorm(x) + beta(fdi), with residual so it starts ~identity."""

    def __init__(self, d_fdi, ch, hidden=128, lora_rank=4):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.fc1 = TaskLoRALinear(d_fdi, hidden, rank=lora_rank)
        self.fc2 = TaskLoRALinear(hidden, 2 * ch, rank=lora_rank)
        nn.init.zeros_(self.fc2.base.weight)
        nn.init.zeros_(self.fc2.base.bias)

    def forward(self, x, fdi):
        g, b = self.fc2(F.relu(self.fc1(fdi))).chunk(2, 1)
        g, b = g[..., None, None], b[..., None, None]
        # residual form: identity at init (g = b = 0), pure AdaIN modulation on top
        return x + g * self.norm(x) + b


# ---------------------------------------------------------------- backbone
def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(w, ws, B, H, W):
    C = w.shape[-1]
    x = w.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


class WindowAttention(nn.Module):
    """Window MHSA (Swin-style; odd blocks use a half-window cyclic shift, no mask).

    Written out explicitly so QAT can wrap every matmul input.
    """

    def __init__(self, dim, heads, ws, shift=False, lora_rank=0):
        super().__init__()
        self.h, self.ws, self.shift = heads, ws, shift
        Lin = (lambda i, o: TaskLoRALinear(i, o, rank=lora_rank)) if lora_rank else nn.Linear
        self.qkv = Lin(dim, 3 * dim)
        self.proj = Lin(dim, dim)
        self.scale = (dim // heads) ** -0.5
        # quantization hook points for the two attention matmuls (Identity in FP32; qat.py swaps in ActQuant)
        self.q_q, self.k_q, self.v_q, self.p_q = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):  # x: (B, H, W, C), H, W multiples of ws
        B, H, W, C = x.shape
        s = self.ws // 2 if self.shift else 0
        if s:
            x = torch.roll(x, (-s, -s), (1, 2))
        w = window_partition(x, self.ws)  # (nW*B, N, C)
        q, k, v = self.qkv(w).view(w.shape[0], w.shape[1], 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_q(q * self.scale), self.k_q(k), self.v_q(v)
        a = q @ k.transpose(-2, -1)
        o = (self.p_q(a.softmax(-1)) @ v).transpose(1, 2).reshape(w.shape[0], w.shape[1], C)
        x = window_reverse(self.proj(o), self.ws, B, H, W)
        if s:
            x = torch.roll(x, (s, s), (1, 2))
        return x


class GatedBlock(nn.Module):
    """Pre-norm transformer block; attention and FFN outputs are channel-gated by the FDI."""

    def __init__(self, dim, d_fdi, heads=4, ws=8, shift=False, mlp_ratio=2.0, se_rank=32, lora_rank=4,
                 lora_backbone=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        br = lora_rank if lora_backbone else 0
        self.attn = WindowAttention(dim, heads, ws, shift, br)
        hid = int(dim * mlp_ratio)
        Lin = (lambda i, o: TaskLoRALinear(i, o, rank=br)) if br else nn.Linear
        self.ffn = nn.Sequential(Lin(dim, hid), nn.GELU(), Lin(hid, dim))
        self.g_attn = SEGate(d_fdi, dim, se_rank, lora_rank)
        self.g_ffn = SEGate(d_fdi, dim, se_rank, lora_rank)

    def forward(self, x, fdi):  # x: (B, H, W, C)
        ga = self.g_attn(fdi)[:, None, None, :]
        gf = self.g_ffn(fdi)[:, None, None, :]
        x = x + ga * self.attn(self.n1(x))
        x = x + gf * self.ffn(self.n2(x))
        return x


# ---------------------------------------------------------------- full model
class RestoreNet(nn.Module):
    def __init__(self, dim=96, depth=8, heads=4, ws=8, d_fdi=256, enc_width=64,
                 se_rank=32, lora_rank=4, lora_backbone=False, teacher_dim=0, n_tasks=4):
        super().__init__()
        self.ws, self.down = ws, 4
        self.degr = DegradationEncoder(d_fdi, enc_width)
        # head that maps the student embedding into teacher space for the FDI-regression loss
        self.fdi_head = nn.Linear(d_fdi, teacher_dim) if teacher_dim else None
        # optional task-ID head (RQ2 probe during training; never used to route at inference)
        self.task_head = nn.Linear(d_fdi, n_tasks)

        self.shallow = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.GELU())   # full-res features
        self.stem = nn.Sequential(nn.PixelUnshuffle(4), nn.Conv2d(32 * 16, dim, 1))
        self.blocks = nn.ModuleList(
            [GatedBlock(dim, d_fdi, heads, ws, shift=(i % 2 == 1), se_rank=se_rank,
                        lora_rank=lora_rank, lora_backbone=lora_backbone)
             for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.ada1 = AdaIN(d_fdi, dim, lora_rank=lora_rank)
        self.up = nn.Sequential(nn.Conv2d(dim, 32 * 16, 1), nn.PixelShuffle(4), nn.GELU())
        self.ada2 = AdaIN(d_fdi, 32, lora_rank=lora_rank)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)  # starts ~identity (output ~ LQ)

    def forward(self, lq, return_feats=False):
        B, _, H, W = lq.shape
        m = self.ws * self.down
        ph, pw = (m - H % m) % m, (m - W % m) % m
        x_in = F.pad(lq, (0, pw, 0, ph), mode="reflect") if (ph or pw) else lq

        fdi = self.degr(lq)
        s0 = self.shallow(x_in)
        z = self.stem(s0).permute(0, 2, 3, 1)  # (B, h, w, C)
        feats = []
        for blk in self.blocks:
            z = blk(z, fdi)
            feats.append(z)
        z = self.norm(z).permute(0, 3, 1, 2)
        z = self.ada1(z, fdi)
        y = self.ada2(self.up(z) + s0, fdi)  # full-res skip: latent alone can't carry 4x4 detail
        out = (x_in + self.out(y))[..., :H, :W]
        if return_feats:
            return out, {"fdi": fdi, "feats": feats}
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())
