"""Per-task low-rank adapters.

`TaskLoRALinear` wraps an nn.Linear. Task 0 trains the base weight; every later
task adds its own (A, B) pair, so switching tasks never overwrites earlier ones:

    y = W x + b + (alpha / r) * B_t (A_t x)

Only the adapter for the active task is applied. `set_task` is called on the
whole model via `set_active_task(model, t)`.
"""
import math

import torch
import torch.nn as nn


class TaskLoRALinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True, rank=4, alpha=4.0):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=bias)
        self.rank, self.alpha = rank, alpha
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.active = None  # str task key, or None = base only

    def add_task(self, task: str):
        if task in self.A:
            return
        a = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = nn.Parameter(torch.zeros(self.base.out_features, self.rank))  # zero => starts as identity
        self.A[task], self.B[task] = a, b

    def forward(self, x):
        y = self.base(x)
        t = self.active
        if t is not None and t in self.A:
            y = y + (self.alpha / self.rank) * (x @ self.A[t].t() @ self.B[t].t())
        return y


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, TaskLoRALinear)]


def add_task_adapters(model, task: str):
    for m in lora_modules(model):
        m.add_task(task)


def set_active_task(model, task):
    """task: str key of an adapter, or None for the base (first) task."""
    for m in lora_modules(model):
        m.active = task


def adapter_parameters(model, task: str):
    ps = []
    for m in lora_modules(model):
        if task in m.A:
            ps += [m.A[task], m.B[task]]
    return ps
