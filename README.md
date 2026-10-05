# Color-Invariant Saree Design Recognition

Recognise the **design** of a saree (stripes, checks, polka, floral, paisley, zari border …)
no matter what **colour** it is woven in.

---

## 1. Problem statement

A weaver or retailer often has one design (say, a paisley body with a zari border) in 20 colourways.
A normal image classifier trained on red paisley sarees will often fail on a blue one, because it
learns *colour* as a shortcut instead of *pattern*. This project builds recognisers whose output depends
on the **design** and not on the **colour**.

**Objectives**
1. Build a colour-free representation of a saree image.
2. Train a classifier on it (classical ML and deep learning).
3. Prove colour invariance with a test where *train colours ≠ test colours*.

## 2. Key ideas (the "why it works" part)

| Idea | What it does |
|---|---|
| **LAB → L channel only** | LAB separates lightness (L) from colour (a, b). Dropping a, b removes the hue. |
| **CLAHE** | Local contrast equalisation, so dark and light sarees look equally contrasty. |
| **HOG (unsigned gradients)** | Captures motif shape and layout. "Unsigned" means a dark-on-light motif and a light-on-dark motif give the same histogram. |
| **LBP (uniform, multi-scale)** | Texture code based on *comparisons* between pixels, so it is unchanged by any monotonic brightness change. |
| **Gabor filter bank** | Measures repeating pattern frequency and orientation (stripes, checks, weave). |
| **Colour augmentation** | Each training image is shown in random new colours (hue rotate, channel shuffle, tint, greyscale). |
| **Consistency loss (CNN)** | Forces the network to output the same prediction for an image and its recoloured twin. |

## 3. System pipeline

```
Saree image ──► resize 128×128 ──► [colour removal: LAB-L + CLAHE]
                                          │
              ┌───────────────────────────┴──────────────────────────┐
              ▼                                                      ▼
   Classical: HOG + LBP + Gabor ──► SVM              Deep: MobileNetV2 with recolour
                                                     augmentation + consistency loss
              └───────────────────────────┬──────────────────────────┘
                                          ▼
                              Design class + confidence
```

## 4. Project files

| File | Purpose |
|---|---|
| `color_invariance.py` | `to_invariant()` (colour removal) and `random_recolor()` (colour simulation) |
| `generate_synthetic_data.py` | Makes a 6-class synthetic dataset. Train = **warm** colours, test = **cool** colours |
| `classical_baseline.py` | Colour-histogram baseline vs. proposed colour-invariant HOG+LBP+Gabor+SVM |
| `train_cnn.py` | MobileNetV2 transfer learning with recolour augmentation + consistency loss |
| `predict.py` | Predict the design of any image |
| `make_figures.py` | Makes `fig_color_invariance.png` for your report |

## 5. How to run

```bash
pip install -r requirements.txt

python generate_synthetic_data.py          # skip if you use real photos (see §8)
python classical_baseline.py --data data   # classical experiment + comparison
python make_figures.py                     # figure for the report
python train_cnn.py --data data --epochs 15             # deep model (GPU recommended)
python train_cnn.py --data data --epochs 15 --gray_input  # variant: feed only colour-free input
python predict.py path/to/saree.jpg                      # classical model
python predict.py path/to/saree.jpg --cnn saree_cnn.pt   # CNN model
```

## 6. Experiment and results

**Design of the experiment.** Training sarees only use warm colours (red/orange/yellow/pink).
Test sarees only use cool colours (blue/green/teal/purple), so the test colours were **never seen**.
This is the strictest possible test of colour invariance.

Results from running `classical_baseline.py` on the synthetic data (6 classes, 80 train / 30 test per class):

| Method | Accuracy on unseen-colour test set |
|---|---|
| Colour-histogram + SVM (baseline) | **17.2 %** (about chance, since chance = 16.7 %) |
| **Colour-invariant HOG + LBP + Gabor + SVM (proposed)** | **92.2 %** |

The baseline collapses to chance level because it memorised colour. The proposed method barely notices the colour change.
Stripes and floral are near-perfect; the remaining confusion is mostly between polka, checks and paisley.

> Your numbers will differ slightly on re-runs and on real data. Report your own results.
> `train_cnn.py` prints the unseen-colour test accuracy every epoch. Add that number to your table.
> (I could not run the CNN in my environment because PyTorch isn't installed there, so run it on your machine and report what you get.)

## 7. Limitations (include these in your report, examiners like honesty)

1. **Equal-luminance colours lose the design.** In `fig_color_invariance.png`, the paisley example has a
   foreground and background with almost the same lightness, so the L-channel is nearly pure noise. Colour was
   the *only* thing separating the motif. *Fix:* also compute gradients on the a/b chroma channels
   and take the maximum gradient magnitude across L, a, b (a colour-invariant edge map).
2. **Synthetic data is easy.** Real sarees have folds, drape, lighting, shadows, multi-colour zari, and
   complex backgrounds. Expect lower accuracy until you train on real photos.
3. **Colour is sometimes meaningful.** Two designs that differ only by colour scheme can't be told apart. That is by design here.
4. **Gold zari reflectance** changes with the lighting angle, which can confuse texture features.

## 8. Using real saree images

Arrange photos like this (any number of classes, any image names):

```
data/
  train/  kanjivaram/  banarasi/  paithani/  ikat/ ...
  test/   kanjivaram/  banarasi/  paithani/  ikat/ ...
```

Good sources: online catalogues, your own photos, or a local weaver's collection (with permission).
Crop to the saree body or border, avoid models/backgrounds, and put different colourways of the
same design in *both* train and test, or use the warm/cool split above to make the test stricter.

## 9. Possible extensions

- **Retrieval**: "show me all sarees with this design" by comparing the feature vectors of two images (cosine similarity).
- Add the chroma-gradient edge map (limitation 1) and measure the improvement.
- Replace MobileNetV2 with a Vision Transformer or a self-supervised model (DINO).
- Segment the body, pallu and border and recognise each separately.
- Wrap `predict.py` in a Streamlit/Flask app for a demo.

## 10. Likely viva questions

**Q. What does "color-invariant" mean?**
The output doesn't change when only the colour of the input changes.

**Q. Why LAB and not just grayscale?**
LAB's L channel is built to match perceived lightness and is separated cleanly from colour. Plain
grayscale is a weighted RGB sum, and works similarly, but L is more perceptually uniform.

**Q. Why is LBP colour-invariant?**
LBP compares each pixel with its neighbours (greater/smaller). Any brightness change that keeps the order
of pixel values leaves the code unchanged.

**Q. Why did the colour-histogram model get ~17 %?**
It learned "warm colours → class X". On cool-coloured test images that shortcut is useless, so it guesses.

**Q. What does the consistency loss do?**
It adds a penalty (KL divergence) when the network's predicted probabilities for an image and its recoloured
version differ, pushing the learned features to ignore colour.

**Q. Where does your method fail?**
When motif and background have the same lightness (see §7.1).

**Q. Why SVM for the classical part?**
Works well with small datasets and high-dimensional features like HOG.

## 11. References (starting points)

- Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005
- Ojala et al., *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP*, TPAMI 2002
- Zuiderveld, *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV, 1994
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
- van de Sande et al., *Evaluating Color Descriptors for Object and Scene Recognition*, TPAMI 2010
