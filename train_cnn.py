"""
train_cnn.py  (deep-learning version)
-------------------------------------
MobileNetV2 transfer-learning classifier made colour-invariant by:

  1. Colour augmentation  - every training image is shown in a random new
                            colour (hue shift, channel shuffle, greyscale...).
  2. Consistency loss     - the network is forced to give the SAME prediction
                            for an image and its recoloured twin
                            (symmetric KL divergence between the two outputs).
  3. (optional) --gray_input feeds only the colour-free L channel.

   total_loss = CE(original) + CE(recoloured) + lambda * KL(original || recoloured)

Usage:
    python train_cnn.py --data data --epochs 15
    python train_cnn.py --data data --epochs 15 --gray_input
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from color_invariance import load_image, random_recolor, to_invariant

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img_bgr, gray_input=False):
    if gray_input:
        g = to_invariant(img_bgr)
        img_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def geo_aug(img, rng):
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.15))
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


class SareeDataset(Dataset):
    def __init__(self, root, train, gray_input):
        self.classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        self.items = [(f, i) for i, c in enumerate(self.classes)
                      for f in sorted((Path(root) / c).glob("*"))]
        self.train, self.gray = train, gray_input
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = load_image(path)
        if self.train:
            img = geo_aug(img, self.rng)
            twin = random_recolor(img, self.rng)       # same design, new colours
            return to_tensor(img, self.gray), to_tensor(twin, self.gray), label
        return to_tensor(img, self.gray), label


def build_model(n_classes):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.last_channel, n_classes)
    return m


def sym_kl(a, b):
    pa, pb = F.log_softmax(a, 1), F.log_softmax(b, 1)
    return 0.5 * (F.kl_div(pa, pb.exp(), reduction="batchmean") +
                  F.kl_div(pb, pa.exp(), reduction="batchmean"))


@torch.no_grad()
def evaluate(model, loader, dev, recolor_seed=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0, help="consistency-loss weight")
    ap.add_argument("--gray_input", action="store_true")
    ap.add_argument("--out", default="saree_cnn.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = SareeDataset(Path(a.data) / "train", True, a.gray_input)
    te = SareeDataset(Path(a.data) / "test", False, a.gray_input)
    tl = DataLoader(tr, a.bs, shuffle=True)
    el = DataLoader(te, a.bs)

    model = build_model(len(tr.classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    for ep in range(1, a.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, xt, y in tl:
            x, xt, y = x.to(dev), xt.to(dev), y.to(dev)
            lo, lt = model(x), model(xt)
            loss = F.cross_entropy(lo, y) + F.cross_entropy(lt, y) + a.lam * sym_kl(lo, lt)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(y)
        sched.step()
        acc = evaluate(model, el, dev)
        print(f"epoch {ep:2d}  loss {run_loss/len(tr):.4f}  unseen-colour test acc {acc*100:.1f}%")

    torch.save({"state": model.state_dict(), "classes": tr.classes,
                "gray_input": a.gray_input}, a.out)
    print("saved", a.out)
