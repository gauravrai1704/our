"""Generate a tiny procedural dataset so the whole pipeline can be smoke-tested
without downloading anything. Results on it mean nothing.

    python scripts/make_toy_data.py --out toy_data
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_image(rng, size=160):
    im = Image.new("RGB", (size, size), tuple(int(c) for c in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(8, 20))):
        x0, y0 = rng.integers(0, size, 2)
        x1, y1 = x0 + rng.integers(5, 60), y0 + rng.integers(5, 60)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([int(x0), int(y0), int(x1), int(y1)], fill=col)
    for _ in range(10):
        d.line([tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(2)],
               fill=tuple(int(c) for c in rng.integers(0, 255, 3)), width=int(rng.integers(1, 4)))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy_data")
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        for sub in ("hq", "blur_lq", "blur_hq"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        for i in range(n):
            im = make_image(rng)
            im.save(os.path.join(a.out, split, "hq", f"{i:03d}.png"))
            im.save(os.path.join(a.out, split, "blur_hq", f"{i:03d}.png"))
            im.filter(ImageFilter.BoxBlur(int(rng.integers(1, 4)))).save(
                os.path.join(a.out, split, "blur_lq", f"{i:03d}.png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
