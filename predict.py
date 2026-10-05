"""
predict.py - recognise the design of a saree image (any colour).

    python predict.py my_saree.jpg                       # classical SVM model
    python predict.py my_saree.jpg --cnn saree_cnn.pt    # deep-learning model
"""
import argparse
import numpy as np
from color_invariance import load_image, to_invariant


def predict_classical(path, model_file):
    import joblib
    from classical_baseline import invariant_features
    d = joblib.load(model_file)
    x = invariant_features(load_image(path))[None]
    scores = d["clf"].decision_function(x)[0]
    p = np.exp(scores - scores.max()); p /= p.sum()
    return d["classes"], p


def predict_cnn(path, weights):
    import torch
    from train_cnn import build_model, to_tensor
    ck = torch.load(weights, map_location="cpu")
    m = build_model(len(ck["classes"])); m.load_state_dict(ck["state"]); m.eval()
    x = to_tensor(load_image(path), ck["gray_input"])[None]
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0].numpy()
    return ck["classes"], p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--cnn", default=None)
    ap.add_argument("--model", default="classical_model.joblib")
    a = ap.parse_args()
    classes, p = predict_cnn(a.image, a.cnn) if a.cnn else predict_classical(a.image, a.model)
    for i in np.argsort(-p)[:3]:
        print(f"{classes[i]:15s} {p[i]*100:5.1f}%")
