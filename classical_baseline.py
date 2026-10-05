"""
classical_baseline.py
---------------------
Classical (no deep learning) colour-invariant recogniser.

Features, all computed on the colour-free image from to_invariant():
  * HOG   (unsigned gradients)  -> shape / motif layout
  * LBP   (uniform, multi-scale)-> fine weave texture; invariant to any
                                   monotonic brightness change
  * Gabor filter-bank energies  -> repeating pattern frequency/orientation

Classifier: RBF-SVM.

For comparison we ALSO train a 'colour-histogram' SVM. It should do well when
train and test colours match, and collapse when colours change - which is
exactly the problem this project solves.

Usage:
    python classical_baseline.py --data data
"""
import argparse
from pathlib import Path
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from color_invariance import load_image, to_invariant, random_recolor


# ---------- feature extractors ----------
def hog_feat(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def lbp_feat(gray):
    feats = []
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
        feats.append(h)
    return np.concatenate(feats)


_GABOR = [cv2.getGaborKernel((21, 21), 4.0, th, lam, 0.5, 0)
          for th in np.arange(0, np.pi, np.pi / 6) for lam in (4, 8, 16)]


def gabor_feat(gray):
    g = gray.astype(np.float32) / 255.0
    out = []
    for k in _GABOR:
        r = cv2.filter2D(g, cv2.CV_32F, k)
        out += [np.abs(r).mean(), r.std()]
    return np.array(out)


def invariant_features(img_bgr):
    gray = to_invariant(img_bgr)
    return np.concatenate([hog_feat(gray), lbp_feat(gray), gabor_feat(gray)])


def color_hist_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


# ---------- data ----------
def load_split(root, extractor, recolor=False, seed=0):
    rng = np.random.default_rng(seed)
    classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    X, y = [], []
    for ci, c in enumerate(classes):
        for f in sorted((Path(root) / c).glob("*")):
            img = load_image(f)
            if recolor:
                img = random_recolor(img, rng)
            X.append(extractor(img))
            y.append(ci)
    return np.array(X), np.array(y), classes


def run(name, extractor, data, augment_colors):
    Xtr, ytr, classes = load_split(Path(data) / "train", extractor)
    if augment_colors:      # add recoloured copies of the training images
        Xa, ya, _ = load_split(Path(data) / "train", extractor, recolor=True, seed=1)
        Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
    Xte, yte, _ = load_split(Path(data) / "test", extractor)     # unseen colours
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n=== {name} ===\nAccuracy on UNSEEN-COLOUR test set: {acc*100:.1f}%")
    return clf, classes, yte, pred, acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--save", default="classical_model.joblib")
    a = ap.parse_args()

    run("Baseline: colour histogram", color_hist_features, a.data, False)
    clf, classes, yte, pred, acc = run(
        "Proposed: colour-invariant HOG+LBP+Gabor", invariant_features, a.data, True)

    print(classification_report(yte, pred, target_names=classes, digits=3))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    joblib.dump({"clf": clf, "classes": classes}, a.save)
    print(f"Saved model -> {a.save}")
