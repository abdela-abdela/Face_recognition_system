"""Matching faces by cosine similarity: a gallery of known people, with an 'unknown' answer."""
import json

import numpy as np

from .config import DEFAULT_MODEL, FALLBACK_THRESHOLDS, MODELS_DIR

GALLERY_FILE = MODELS_DIR / "gallery.npz"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"
UNKNOWN = -1


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-12)


def build_gallery(E, y):
    """One reference vector per person: the normalised mean of their embeddings."""
    ids = np.unique(y)
    return ids, l2_normalize(np.stack([E[y == i].mean(axis=0) for i in ids]))


def identify(E, ids, centroids, threshold, unknown=UNKNOWN):
    """Best-matching identity per embedding, or `unknown` if the similarity is below threshold."""
    S = l2_normalize(E) @ centroids.T
    best = S.argmax(axis=1)
    score = S[np.arange(len(S)), best]
    return np.where(score >= threshold, ids[best], unknown), score


class CentroidMatcher:
    """Nearest-centroid classifier with cosine similarity: the rule the webcam demo uses."""

    def __init__(self, threshold=-1.0):  # -1 = never answer 'unknown'
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_, self.centroids_ = build_gallery(X, y)
        return self

    def predict(self, X):
        return identify(X, self.classes_, self.centroids_, self.threshold)[0]


def load_threshold(model_name=None):
    """(model, threshold) chosen by `python -m src.train`, or DeepFace's default if not run yet."""
    if THRESHOLD_FILE.exists():
        t = json.loads(THRESHOLD_FILE.read_text())
        if model_name in (None, t["model"]):
            return t["model"], t["threshold"]
    model_name = model_name or DEFAULT_MODEL
    print(f"No tuned threshold for {model_name}; using DeepFace's default "
          f"{FALLBACK_THRESHOLDS[model_name]}. Run `python -m src.train` to tune it on LFW.")
    return model_name, FALLBACK_THRESHOLDS[model_name]


def save_gallery(names, centroids, model_name, threshold, path=GALLERY_FILE):
    path.parent.mkdir(exist_ok=True)
    np.savez(path, names=np.asarray(names), centroids=centroids,
             model_name=model_name, threshold=threshold)


def load_gallery(path=GALLERY_FILE):
    """Returns (names, centroids, model_name, threshold)."""
    d = np.load(path)
    return d["names"], d["centroids"], str(d["model_name"]), float(d["threshold"])
