"""Flatten GoPro_Large into the layout PairedDataset expects.

GoPro_Large/{train,test}/GOPRxxxx_xx_xx/{blur,sharp}/000001.png reuses frame names
across sequences, so pairing by sorted filename only works after flattening:

    datasets/GoPro/{train,test}/{blur,sharp}/GOPRxxxx_xx_xx_000001.png

Files are symlinked (use --copy to copy). The same script works for HIDE / RealBlur
if their folders are arranged as <split>/<sequence>/{blur,sharp}/.

    python scripts/prepare_gopro.py --src /data/GOPRO_Large --dst datasets/GoPro
"""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="datasets/GoPro")
    ap.add_argument("--blur", default="blur", help="use blur_gamma for the gamma-corrected variant")
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()
    for split in ("train", "test"):
        n = 0
        for seq in sorted(glob.glob(os.path.join(a.src, split, "*"))):
            if not os.path.isdir(seq):
                continue
            for kind, sub in (("blur", a.blur), ("sharp", "sharp")):
                out = os.path.join(a.dst, split, kind)
                os.makedirs(out, exist_ok=True)
                for f in sorted(glob.glob(os.path.join(seq, sub, "*.png"))):
                    tgt = os.path.join(out, f"{os.path.basename(seq)}_{os.path.basename(f)}")
                    if os.path.lexists(tgt):
                        continue
                    if a.copy:
                        shutil.copy2(f, tgt)
                    else:
                        os.symlink(os.path.abspath(f), tgt)
                    n += kind == "sharp"
        nb = len(os.listdir(os.path.join(a.dst, split, "blur"))) if os.path.isdir(os.path.join(a.dst, split, "blur")) else 0
        ns = len(os.listdir(os.path.join(a.dst, split, "sharp"))) if os.path.isdir(os.path.join(a.dst, split, "sharp")) else 0
        print(f"{split}: {nb} blur / {ns} sharp ({n} new pairs)")
        assert nb == ns, "blur/sharp counts differ"


if __name__ == "__main__":
    main()
