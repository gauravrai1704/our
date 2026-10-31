"""Shared helpers: checkpoint loading and per-task patch collection."""
import torch
from torch.utils.data import DataLoader

from data.degrade import TASKS
from models.lora import add_task_adapters


def load_checkpoint(path, device, use_ema=True):
    """Rebuild the model (with all adapters) from a train.py stage checkpoint.

    FP32 unless the run used qat.continual = quantized, in which case the fake-quant
    INT8 model is rebuilt."""
    from train import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg, teacher_dim=ck.get("teacher_dim", 0))
    keys = {int(k): v for k, v in ck["adapter_keys"].items()}
    for k in keys.values():
        if k is not None:
            add_task_adapters(model, k)
    if ck.get("quantized"):  # trained with qat.continual = quantized: rebuild the fake-quant model
        from models.quant import prepare_qat, set_act_mode
        prepare_qat(model, ck["quantized"]["keep_fp"], ck["quantized"]["quant_attn"])
    model.load_state_dict(ck["ema"] if use_ema and ck.get("ema") else ck["model"])
    if ck.get("quantized"):
        set_act_mode(model, "quant")
    return model.to(device).eval(), ck, keys


def collect_patches(cfg, ck, per_task, patch=None):
    """{task_id: (lq, hq)} patches for calibration / QAT.

    Prefers the replay buffer stored in the checkpoint (the realistic continual setting:
    old tasks' data is only available through replay). Falls back to sampling from the
    stage training sets (finetune / joint / specialist runs have no buffer).
    """
    from train import build_train_set
    out = {}
    rep = ck.get("replay")
    if rep and rep.get("store"):
        for t, items in rep["store"].items():
            items = items[:per_task]
            out[int(t)] = (torch.stack([a for a, _ in items]), torch.stack([b for _, b in items]))
    for st in cfg["stages"]:
        tid = TASKS.index(st["task"])
        if tid in out:
            continue
        ds = build_train_set(st["task"], st["train"], patch or st["patch"])
        lq, hq, n = [], [], 0
        for b in DataLoader(ds, batch_size=8, shuffle=True, num_workers=0):
            lq.append(b["lq"])
            hq.append(b["hq"])
            n += len(b["lq"])
            if n >= per_task:
                break
        out[tid] = (torch.cat(lq)[:per_task], torch.cat(hq)[:per_task])
    return out


def load_qat_checkpoint(path, device):
    """Rebuild the fake-quant INT8 model saved by qat.py."""
    from train import build_model
    from models.quant import prepare_qat, set_act_mode
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"], teacher_dim=ck.get("teacher_dim", 0))
    keys = {int(k): v for k, v in ck["adapter_keys"].items()}
    for k in keys.values():
        if k is not None:
            add_task_adapters(model, k)
    prepare_qat(model, ck["keep_fp"], ck["quant_attn"])
    model.load_state_dict(ck["state_dict"])
    set_act_mode(model, "quant")
    return model.to(device).eval(), ck, keys
