"""
color_invariance.py
-------------------
Core ideas used by the whole project:

1. to_invariant()  : turn an RGB image into a representation that does NOT
                     depend on the saree's colour (luminance only + contrast
                     normalisation).
2. random_recolor(): simulate the same design woven in a different colour
                     (hue rotation, channel shuffle, saturation change,
                     greyscale). Used for data augmentation and for testing.
"""
import cv2
import numpy as np

IMG_SIZE = 128


def load_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_invariant(img_bgr):
    """BGR uint8 -> single-channel uint8 image with colour removed.

    Steps
      a) Convert to CIE-LAB and keep only L (lightness). Hue/chroma (a, b)
         carry most of the colour information, so they are dropped.
      b) CLAHE (local histogram equalisation) so a dark-coloured saree and a
         light-coloured saree end up with a similar contrast range.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(L)


def random_recolor(img_bgr, rng=None):
    """Return a randomly re-coloured copy of the image (same design, new colours)."""
    rng = rng or np.random.default_rng()
    out = img_bgr.copy()

    mode = rng.integers(0, 4)
    if mode == 0:                       # hue rotation
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(0, 180)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.6, 1.4), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif mode == 1:                     # shuffle B,G,R channels
        out = out[:, :, rng.permutation(3)]
    elif mode == 2:                     # random per-channel gain (colour cast)
        gains = rng.uniform(0.5, 1.5, size=3)
        out = np.clip(out.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    else:                               # greyscale (3-channel)
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # brightness / contrast jitter
    alpha, beta = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
