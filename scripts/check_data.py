"""Check every dataset a config refers to BEFORE starting a multi-day run.

    python scripts/check_data.py --config configs/cl_adapters.yaml

For each train / val / eval entry: does the folder exist, how many images, do LQ/HQ
counts and filenames pair up, are training images at least one patch large. Also warns
if a val or eval image also appears in a training folder (test leakage). Exit code 1 if
anything required is missing.
"""
import argparse
import os
import sys

import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import list_images, pair_by_stem  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--size_check", type=int, default=50, help="images per folder to open for the size check")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    bad = 0
    train_files = set()

    def check(label, item, patch=None):
        nonlocal bad
        dirs = [item[k] for k in ("lq_dir", "hq_dir") if k in item]
        for d in dirs:
            if not os.path.isdir(d):
                print(f"  MISSING  {label}: {d}")
                bad += 1
                return []
        files = list_images(item["hq_dir"])
        if not files:
            print(f"  EMPTY    {label}: {item['hq_dir']}")
            bad += 1
            return []
        msg = f"  ok       {label}: {len(files)} images"
        if "lq_dir" in item:
            lq = list_images(item["lq_dir"])
            if len(lq) != len(files):
                print(f"  MISMATCH {label}: {len(lq)} LQ vs {len(files)} HQ")
                bad += 1
                return files
            pair_by_stem(lq, files, item["lq_dir"], item["hq_dir"])
        if patch:
            small = 0
            for f in files[:a.size_check]:
                w, h = Image.open(f).size
                small += min(w, h) < patch
            if small:
                print(f"  SMALL    {label}: {small}/{min(len(files), a.size_check)} checked images < patch {patch}")
                bad += 1
        print(msg)
        return files

    print("train")
    for st in cfg["stages"]:
        for it in st["train"]:
            train_files.update(os.path.realpath(f) for f in check(f"{st['task']:8s} {it['hq_dir']}", it, st["patch"]))
    for key in ("val", "eval"):
        if key not in cfg:
            continue
        print(key)
        for task, items in cfg[key].items():
            for it in items:
                files = check(f"{task:8s} {it['name']}", it)
                leak = sum(os.path.realpath(f) in train_files for f in files)
                if leak:
                    print(f"  LEAK     {it['name']}: {leak} images are also training images")
                    bad += 1
    print("\nALL GOOD" if not bad else f"\n{bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
