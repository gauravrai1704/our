"""Score another method's restored images with OUR protocol (metrics.py), so every row of
the comparison table is computed the same way.

    python eval_external.py --config configs/cl_adapters.yaml --method VOSR \
        --outputs results_external/VOSR          # <outputs>/<eval dataset name>/<image>.png

Each eval dataset in the config is matched by name (e.g. Set5_x4, RealSR_x4, GoPro_test),
and each output image is matched to the HQ image by filename stem (a suffix after the stem,
e.g. "baby_SwinIR.png" for "baby.png", is allowed). Datasets with no output folder are
skipped. Outputs larger than HQ (some SR codes pad) are cropped to the HQ size.

Results go to runs/external/<method>/results.json and appear in aggregate.py's
comparison table, together with the metadata from configs/external_baselines.yaml.
"""
import argparse
import glob
import json
import os

import torch
import yaml

from data.datasets import load_rgb, to_tensor
from evaluate import build_eval_sets
from metrics import LPIPSMetric, NRMetrics, Scorer


def find_output(folder, stem):
    exact = [f for f in glob.glob(os.path.join(folder, stem + ".*"))]
    if exact:
        return exact[0]
    loose = sorted(glob.glob(os.path.join(folder, stem + "[_-]*")))
    return loose[0] if loose else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="supplies the eval datasets and nr_metrics")
    ap.add_argument("--method", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--registry", default="configs/external_baselines.yaml")
    ap.add_argument("--out", default="runs/external")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reg = yaml.safe_load(open(a.registry)).get("methods", {}) if os.path.exists(a.registry) else {}
    lp = LPIPSMetric(dev)
    nr = NRMetrics(cfg.get("nr_metrics"), dev) if cfg.get("nr_metrics") else None

    results, missing = {}, {}
    for task, items in build_eval_sets(cfg).items():
        for name, ds, scale in items:
            folder = os.path.join(a.outputs, name)
            if not os.path.isdir(folder):
                continue
            sc, miss = Scorer(task, scale, lp, nr), []
            for i in range(len(ds)):
                b = ds[i]
                stem = os.path.splitext(b["name"])[0]
                f = find_output(folder, stem)
                if f is None:
                    miss.append(stem)
                    continue
                hq = b["hq"][None].to(dev)
                out = to_tensor(load_rgb(f))[None].to(dev)
                if out.shape[-2] < hq.shape[-2] or out.shape[-1] < hq.shape[-1]:
                    raise ValueError(f"{f}: output {tuple(out.shape[-2:])} smaller than HQ {tuple(hq.shape[-2:])}")
                sc.add(out[..., :hq.shape[-2], :hq.shape[-1]], hq)
            if miss:
                missing[name] = miss
                print(f"WARNING {name}: {len(miss)} of {len(ds)} outputs missing, e.g. {miss[:3]}")
            if sc.vals["psnr"]:
                results.setdefault(task, {})[name] = sc.summary()
                r = results[task][name]
                print(f"{name:24s} PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  n={r['n']}")
    if not results:
        raise SystemExit(f"no dataset folders under {a.outputs} match the eval sets in {a.config}")
    out_dir = os.path.join(a.out, a.method)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"method": a.method, "source": "outputs", "meta": reg.get(a.method, {}), "results": results,
               "missing": missing}, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    print(f"wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
