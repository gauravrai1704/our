"""Continual-learning helpers: parameter freezing per stage + Elastic Weight Consolidation.

Stage policy (mode="adapters"):
  stage 0 : everything trainable (base task, adapters key None)
  stage k : base frozen; trainable = new task's LoRA pairs + params matching
            cfg.shared_trainable (default: degradation encoder + heads). Those shared
            params are the ones that can forget, so EWC and replay protect them.
mode="finetune" : everything trainable every stage, no adapters (forgetting lower bound)
"""
import fnmatch

import torch

from models.lora import adapter_parameters


def configure_trainable(model, stage, task_key, mode, shared_patterns):
    if mode == "finetune" or stage == 0:
        for p in model.parameters():
            p.requires_grad_(True)
        # adapters of other tasks (none exist yet at stage 0) stay frozen anyway
        return [p for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for n, p in model.named_parameters():
        if ".A." in n or ".B." in n:
            continue
        if any(fnmatch.fnmatch(n, pat) for pat in shared_patterns):
            p.requires_grad_(True)
            params.append(p)
    for p in adapter_parameters(model, task_key):
        p.requires_grad_(True)
        params.append(p)
    return params


class EWC:
    """Accumulates per-parameter importance + anchor weights after each task; quadratic penalty.

    importance="fisher" : EWC (Kirkpatrick et al. 2017), squared gradients of the task loss
    importance="mas"    : MAS (Aljundi et al. 2018), absolute gradients of the squared L2
                          norm of the output (label-free; pass that as loss_fn)
    """

    def __init__(self, lam=0.0, importance="fisher"):
        self.lam, self.importance = lam, importance
        self.fisher, self.anchor = {}, {}

    @torch.no_grad()
    def _snapshot(self, model, names):
        return {n: p.detach().clone() for n, p in model.named_parameters() if n in names}

    def consolidate(self, model, loss_fn, batches, names):
        """loss_fn(batch) -> scalar loss with model in eval mode; names: params to protect."""
        if self.lam <= 0:
            return
        params = {n: p for n, p in model.named_parameters() if n in names}
        fis = {n: torch.zeros_like(p) for n, p in params.items()}
        req = {n: p.requires_grad for n, p in params.items()}
        for p in params.values():
            p.requires_grad_(True)
        k = 0
        for b in batches:
            model.zero_grad(set_to_none=True)
            loss_fn(b).backward()
            for n, p in params.items():
                if p.grad is not None:
                    g = p.grad.detach()
                    fis[n] += g.abs() if self.importance == "mas" else g ** 2
            k += 1
        for n, p in params.items():
            p.requires_grad_(req[n])
            f = fis[n] / max(k, 1)
            self.fisher[n] = self.fisher.get(n, 0) + f  # online accumulation across tasks
        self.anchor.update(self._snapshot(model, names))
        model.zero_grad(set_to_none=True)

    def penalty(self, model):
        if self.lam <= 0 or not self.fisher:
            return None
        tot = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher and p.requires_grad:
                tot = tot + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return 0.5 * self.lam * tot

    def state_dict(self):
        return {"lam": self.lam, "importance": self.importance, "fisher": self.fisher, "anchor": self.anchor}

    def load_state_dict(self, s):
        self.lam, self.fisher, self.anchor = s["lam"], s["fisher"], s["anchor"]
        self.importance = s.get("importance", "fisher")
