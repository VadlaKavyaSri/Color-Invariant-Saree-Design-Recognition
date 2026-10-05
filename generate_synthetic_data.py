"""
generate_synthetic_data.py
--------------------------
Creates a small synthetic 'saree design' dataset so the project runs end to
end even before you collect real photos.

Six design classes: stripes, checks, polka, floral, paisley, zari_border.

IMPORTANT TRICK: training images use WARM colours only (red/orange/yellow/pink)
while test images use COOL colours only (blue/green/teal/purple).
A model that learned 'colour' will fail on the test set; a truly
colour-invariant model will still work.
"""
import argparse
from pathlib import Path
import colorsys
import cv2
import numpy as np

S = 128
CLASSES = ["stripes", "checks", "polka", "floral", "paisley", "zari_border"]
WARM = [(0.0, 0.12), (0.92, 1.0)]       # hue ranges (0..1)
COOL = [(0.42, 0.80)]


def rand_color(rng, hue_ranges):
    lo, hi = hue_ranges[rng.integers(len(hue_ranges))]
    h = rng.uniform(lo, hi)
    r, g, b = colorsys.hsv_to_rgb(h, rng.uniform(0.5, 1), rng.uniform(0.45, 1))
    return int(b * 255), int(g * 255), int(r * 255)       # BGR


def draw(cls, rng, hues):
    bg, fg = rand_color(rng, hues), rand_color(rng, hues)
    img = np.full((S, S, 3), bg, np.uint8)
    if cls == "stripes":
        w = int(rng.integers(8, 16))
        for x in range(0, S, 2 * w):
            cv2.rectangle(img, (x, 0), (x + w, S), fg, -1)
    elif cls == "checks":
        w = int(rng.integers(12, 24))
        for y in range(0, S, w):
            for x in range(0, S, w):
                if ((x // w) + (y // w)) % 2 == 0:
                    cv2.rectangle(img, (x, y), (x + w, y + w), fg, -1)
    elif cls == "polka":
        r, gap = int(rng.integers(4, 8)), int(rng.integers(20, 30))
        for y in range(gap // 2, S, gap):
            for x in range(gap // 2, S, gap):
                cv2.circle(img, (x, y), r, fg, -1)
    elif cls == "floral":
        gap = 44
        for y in range(gap // 2, S, gap):
            for x in range(gap // 2, S, gap):
                for a in range(0, 360, 60):
                    px = int(x + 9 * np.cos(np.radians(a)))
                    py = int(y + 9 * np.sin(np.radians(a)))
                    cv2.circle(img, (px, py), 6, fg, -1)
                cv2.circle(img, (x, y), 4, bg, -1)
    elif cls == "paisley":
        gap = 42
        for y in range(gap // 2, S, gap):
            for x in range(gap // 2, S, gap):
                cv2.ellipse(img, (x, y), (14, 8), 45, 0, 360, fg, -1)
                cv2.ellipse(img, (x + 9, y - 9), (8, 5), -30, 0, 360, fg, -1)
                cv2.ellipse(img, (x, y), (6, 3), 45, 0, 360, bg, -1)
    elif cls == "zari_border":
        h = int(rng.integers(26, 38))
        cv2.rectangle(img, (0, S - h), (S, S), fg, -1)
        for x in range(6, S, 18):
            pts = np.array([[x, S - h + 4], [x + 7, S - h // 2], [x, S - 4], [x - 7, S - h // 2]])
            cv2.fillPoly(img, [pts], bg)
        cv2.line(img, (0, S - h - 4), (S, S - h - 4), fg, 2)
    # random small rotation + noise (like a photo of a real saree)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(-8, 8), rng.uniform(0.95, 1.15))
    img = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for split, n, hues in [("train", a.n_train, WARM), ("test", a.n_test, COOL)]:
        for c in CLASSES:
            d = Path(a.out) / split / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(d / f"{c}_{i:03d}.png"), draw(c, rng, hues))
    print(f"Dataset written to {a.out}/ (train = warm colours, test = cool colours)")


if __name__ == "__main__":
    main()
