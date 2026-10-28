"""INT8 fake-quantization for QAT / PTQ.

Uses torch.fake_quantize_per_{tensor,channel}_affine, which (a) give straight-through
gradients and (b) export to ONNX QuantizeLinear/DequantizeLinear (QDQ) pairs that
TensorRT / ONNX Runtime fuse into real INT8 kernels.

Scheme (standard for INT8 inference engines):
  weights      : per-output-channel, symmetric, [-127, 127], scale recomputed from the
                 current weight every forward (so QAT can move weights freely)
  activations  : per-tensor, asymmetric uint8 [0, 255], EMA min/max observer that is
                 FROZEN after calibration / observer warm-up
  quantized    : inputs + weights of every Conv2d / Linear, the low-rank adapter
                 intermediate (per task), and q, k, v, softmax(qk) for attention matmuls
  left in FP   : LayerNorm, InstanceNorm, GELU, sigmoid gates, residual adds,
                 and any layer listed in keep_fp (default: first and last conv)

prepare_qat(model, keep_fp) converts the model in place. State-dict keys of the
original weights are unchanged; quantizer buffers are added as new keys.
"""
import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import TaskLoRALinear, active_weights


class ActQuant(nn.Module):
    """Per-tensor asymmetric uint8 fake-quant with an EMA min/max observer.

    mode: "off" (pass-through) | "observe" (update stats, no quant) |
          "quant" (fake-quant with frozen stats) | "observe_quant" (both; QAT warm-up)
    """

    def __init__(self, momentum=0.05):
        super().__init__()
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))
        self.m = momentum
        self.mode = "off"

    @torch.no_grad()
    def observe(self, x):
        lo, hi = x.detach().min().float(), x.detach().max().float()
        if torch.isinf(self.lo):
            self.lo.copy_(lo)
            self.hi.copy_(hi)
        else:
            self.lo.mul_(1 - self.m).add_(lo, alpha=self.m)
            self.hi.mul_(1 - self.m).add_(hi, alpha=self.m)

    def qparams(self):
        lo, hi = torch.clamp(self.lo, max=0.0), torch.clamp(self.hi, min=0.0)  # range must contain 0
        scale = ((hi - lo) / 255.0).clamp_min(1e-8).reshape(1)
        zp = torch.clamp(torch.round(-lo / scale), 0, 255).to(torch.int32).reshape(1)
        return scale, zp

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode in ("observe", "observe_quant"):
            self.observe(x)
        if self.mode == "observe" or torch.isinf(self.lo):
            return x
        s, z = self.qparams()
        return torch.fake_quantize_per_tensor_affine(x, s.to(x.device), z.to(x.device), 0, 255)


def fq_weight(w, enabled=True):
    """Per-output-channel symmetric int8 fake-quant (axis 0)."""
    if not enabled:
        return w
    amax = w.detach().abs().flatten(1).amax(1).clamp_min(1e-8)
    scale = (amax / 127.0).float()
    zp = torch.zeros_like(scale, dtype=torch.int32)
    # range -128..127 is what ONNX QDQ accepts; with scale = amax/127 the value -128 is never used,
    # so this is still symmetric [-127, 127] (TensorRT-compatible)
    return torch.fake_quantize_per_channel_affine(w, scale, zp, 0, -128, 127)


# ---------------------------------------------------------------- quantized layers
# Implemented by swapping __class__ so parameter names (and FP32 checkpoints) stay valid.
class QConv2d(nn.Conv2d):
    def forward(self, x):
        on = self.q_enabled
        return self._conv_forward(self.in_q(x) if on else x, fq_weight(self.weight, on), self.bias)


class QLinear(nn.Linear):
    def forward(self, x):
        on = self.q_enabled
        return F.linear(self.in_q(x) if on else x, fq_weight(self.weight, on), self.bias)


class QTaskLoRALinear(TaskLoRALinear):
    def forward(self, x):
        on = self.q_enabled
        xq = self.in_q(x) if on else x
        y = F.linear(xq, fq_weight(self.base.weight, on), self.base.bias)
        for t, w in active_weights(self.active):
            if t not in self.A:
                continue
            mid = xq @ fq_weight(self.A[t], on).t()
            if on:
                mid = self.mid_q[t](mid)
            y = y + (w * self.alpha / self.rank) * (mid @ fq_weight(self.B[t], on).t())
        return y

    def add_task(self, task):
        super().add_task(task)
        if task not in self.mid_q:
            q = ActQuant()
            q.mode = self.in_q.mode
            self.mid_q[task] = q.to(self.base.weight.device)


def _attach(m, cls):
    m.__class__ = cls
    m.in_q = ActQuant()
    m.q_enabled = True
    if cls is QTaskLoRALinear:
        m.mid_q = nn.ModuleDict({t: ActQuant() for t in m.A})


def prepare_qat(model, keep_fp=(), quant_attn=True):
    """Convert model in place. Returns list of (name, module) quantized compute layers."""
    qlayers = []
    lora_bases = {id(m.base) for m in model.modules() if isinstance(m, TaskLoRALinear)}
    for name, m in list(model.named_modules()):
        if any(fnmatch.fnmatch(name, p) for p in keep_fp):
            continue
        if name.startswith("task_head") or name.startswith("fdi_head"):
            continue  # training-only heads (task_head is tiny; probe/routing can stay FP)
        if isinstance(m, TaskLoRALinear):
            _attach(m, QTaskLoRALinear)
            qlayers.append((name, m))
        elif type(m) is nn.Linear and id(m) not in lora_bases:
            _attach(m, QLinear)
            qlayers.append((name, m))
        elif type(m) is nn.Conv2d:
            _attach(m, QConv2d)
            qlayers.append((name, m))
    if quant_attn:
        from .net import WindowAttention
        for name, m in model.named_modules():
            if isinstance(m, WindowAttention):
                m.q_q, m.k_q, m.v_q, m.p_q = ActQuant(), ActQuant(), ActQuant(), ActQuant()
    return qlayers


def act_quantizers(model):
    return [m for m in model.modules() if isinstance(m, ActQuant)]


def set_act_mode(model, mode):
    for q in act_quantizers(model):
        q.mode = mode


def set_layer_quant(model, enabled, only=None):
    """Enable/disable weight+input quant per layer. only: set of module names to enable."""
    for name, m in model.named_modules():
        if hasattr(m, "q_enabled"):
            m.q_enabled = enabled if only is None else (name in only)


def _wcount(m):
    if isinstance(m, TaskLoRALinear):
        return m.base.weight.numel() + sum(p.numel() for p in m.A.values()) + sum(p.numel() for p in m.B.values())
    return m.weight.numel()


def compute_layers(model):
    """All Conv2d / Linear / TaskLoRALinear layers of the deployable model (heads excluded)."""
    bases = {id(m.base) for m in model.modules() if isinstance(m, TaskLoRALinear)}
    out = []
    for n, m in model.named_modules():
        if n.startswith(("task_head", "fdi_head")) or id(m) in bases:
            continue
        if isinstance(m, (TaskLoRALinear, nn.Linear, nn.Conv2d)):
            out.append((n, m))
    return out


def quant_coverage(model, qlayers):
    """Fraction of compute-layer weights that are INT8, and the names left in FP."""
    qnames = {n for n, _ in qlayers}
    allc = compute_layers(model)
    tot = sum(_wcount(m) for _, m in allc)
    q = sum(_wcount(m) for n, m in allc if n in qnames)
    return q / max(tot, 1), [n for n, _ in allc if n not in qnames]
