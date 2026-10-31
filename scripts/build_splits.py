"""Arrange downloaded datasets into the layout the configs use, with a held-out val split.

    python scripts/build_splits.py --raw datasets/raw --out datasets

Builds (symlinks, so no extra disk):
  datasets/DF2K/HR             DIV2K train (800) + Flickr2K minus the val images
  datasets/val/HR              last --val_flickr Flickr2K images (default 100): val split for
                               SR / denoise / JPEG, never trained on, not a test set
  datasets/DIV2K/valid_HR      DIV2K valid (100): compound-degradation test set (RQ4) + probe
  datasets/benchmark/...       Set5 / Set14 / B100 / Urban100 (+ Manga109 if present), HR + LR_bicubic
  datasets/Kodak24             Kodak24
  datasets/GoPro/{train,val,test}/{blur,sharp}
                               flattened GoPro_Large; the last --gopro_val_seqs train sequences
                               (every --gopro_val_every-th frame) become val
Any other folder in --raw named like a dataset in the config (BSD400, WED, CBSD68, McMaster,
LIVE1, Classic5, SIDD, RealSR, ...) is linked into --out unchanged.
Run scripts/check_data.py afterwards.
"""
import argparse
import glob
import os

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def images(d):
    return sorted(f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if f.lower().endswith(EXT))


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def link_dir(src, dst):
    if os.path.isdir(src) and not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        os.symlink(os.path.abspath(src), dst)


def first(raw, *names):
    for n in names:
        hits = glob.glob(os.path.join(raw, "**", n), recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--val_flickr", type=int, default=100)
    ap.add_argument("--gopro_val_seqs", type=int, default=2)
    ap.add_argument("--gopro_val_every", type=int, default=10)
    a = ap.parse_args()
    out = a.out

    # ---- DF2K + val
    div_tr = first(a.raw, "DIV2K_train_HR")
    fl = first(a.raw, "Flickr2K_HR")
    if div_tr and fl:
        fl_imgs = images(fl)
        for f in images(div_tr):
            link(f, os.path.join(out, "DF2K/HR", os.path.basename(f)))
        for f in fl_imgs[:-a.val_flickr]:
            link(f, os.path.join(out, "DF2K/HR", os.path.basename(f)))
        for f in fl_imgs[-a.val_flickr:]:
            link(f, os.path.join(out, "val/HR", os.path.basename(f)))
        print(f"DF2K/HR: {len(images(os.path.join(out, 'DF2K/HR')))}  val/HR: {len(images(os.path.join(out, 'val/HR')))}")
    else:
        print("MISSING DIV2K_train_HR or Flickr2K_HR in", a.raw)
    div_va = first(a.raw, "DIV2K_valid_HR")
    if div_va:
        link_dir(div_va, os.path.join(out, "DIV2K/valid_HR"))

    # ---- SR benchmarks (EDSR benchmark.tar layout: benchmark/<Set>/{HR,LR_bicubic/X4})
    bench = first(a.raw, "benchmark")
    if bench:
        for s in sorted(os.listdir(bench)):
            link_dir(os.path.join(bench, s), os.path.join(out, "benchmark", s))
        print("benchmark:", sorted(os.listdir(os.path.join(out, "benchmark"))))

    # ---- GoPro
    gp = first(a.raw, "GOPRO_Large", "GoPro_Large", "GOPRO")
    if gp:
        seqs = {sp: sorted(d for d in glob.glob(os.path.join(gp, sp, "*")) if os.path.isdir(d)) for sp in ("train", "test")}
        val_seqs = seqs["train"][-a.gopro_val_seqs:] if a.gopro_val_seqs else []
        for sp, ss in seqs.items():
            for seq in ss:
                split = "val" if seq in val_seqs else sp
                for kind in ("blur", "sharp"):
                    frames = sorted(glob.glob(os.path.join(seq, kind, "*.png")))
                    if split == "val":
                        frames = frames[::a.gopro_val_every]
                    for f in frames:
                        link(f, os.path.join(out, "GoPro", split, kind, f"{os.path.basename(seq)}_{os.path.basename(f)}"))
        for sp in ("train", "val", "test"):
            n = len(images(os.path.join(out, "GoPro", sp, "sharp")))
            print(f"GoPro/{sp}: {n} pairs")
    else:
        print("GoPro_Large not found in", a.raw, "(deblurring needs it)")

    # ---- everything else: link by name
    for name in ("BSD400", "WED", "CBSD68", "Kodak24", "McMaster", "LIVE1", "Classic5", "Urban100",
                 "SIDD", "RealSR", "DRealSR", "HIDE", "RealBlur"):
        d = first(a.raw, name)
        if d and not os.path.lexists(os.path.join(out, name)):
            link_dir(d, os.path.join(out, name))
            print(f"{name}: {len(images(d))} images")


if __name__ == "__main__":
    main()
