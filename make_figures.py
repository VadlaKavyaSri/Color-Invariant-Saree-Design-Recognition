"""make_figures.py - produces figures for your report/presentation."""
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from color_invariance import load_image, to_invariant, random_recolor

rng = np.random.default_rng(3)
classes = sorted(p.name for p in Path("data/train").iterdir())
fig, ax = plt.subplots(len(classes), 4, figsize=(8, 2 * len(classes)))
for r, c in enumerate(classes):
    img = load_image(sorted((Path("data/train") / c).glob("*"))[0])
    views = [img, random_recolor(img, rng), random_recolor(img, rng), to_invariant(img)]
    for k, v in enumerate(views):
        ax[r, k].imshow(v if v.ndim == 2 else cv2.cvtColor(v, cv2.COLOR_BGR2RGB), cmap="gray")
        ax[r, k].axis("off")
        if r == 0:
            ax[r, k].set_title(["Original", "Recolour 1", "Recolour 2", "Invariant (L+CLAHE)"][k], fontsize=8)
    ax[r, 0].text(-5, 64, c, ha="right", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("fig_color_invariance.png", dpi=150)
print("saved fig_color_invariance.png")
