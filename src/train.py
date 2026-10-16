"""Benchmark Eigenfaces vs deep embeddings on LFW, and pick the match threshold for the demo.

Run from the project root (computes embeddings first if they aren't cached):
    python -m src.train
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import DEFAULT_MODEL, EMBEDDING_MODELS, MODELS_DIR, REPORTS_DIR, SEED
from .data import load_lfw, split
from .matching import THRESHOLD_FILE, CentroidMatcher, build_gallery, l2_normalize
from .metrics import all_pairs, verification_metrics

CLASSIFIERS = {
    "SVM": lambda: SVC(kernel="rbf", C=10, class_weight="balanced"),
    "1-NN": lambda: KNeighborsClassifier(n_neighbors=1, metric="cosine"),
    "Nearest centroid": lambda: CentroidMatcher(),
}
FAR_TARGET = 0.01  # demo threshold: accept at most 1 in 100 impostor pairs


def eigenfaces(images, train_idx, n_components=150):
    """Classic baseline: PCA fitted on training images (grayscale pixels), for all images."""
    X = images.mean(axis=-1).reshape(len(images), -1)
    pca = PCA(n_components=n_components, whiten=True, random_state=SEED).fit(X[train_idx])
    return l2_normalize(pca.transform(X))


def feature_sets(images, train_idx, models=EMBEDDING_MODELS):
    """{name: (N, d) L2-normalised features} for Eigenfaces and each embedding model."""
    from .embed import get_embeddings  # imports TensorFlow, so only when needed

    feats = {"Eigenfaces": eigenfaces(images, train_idx)}
    feats.update({m: get_embeddings(m) for m in models})
    return feats


def gallery_threshold(E, y, far=FAR_TARGET):
    """Threshold for gallery matching such that only `far` of impostors get a name.

    Each image's best similarity to *another* person's average face plays the stranger.
    Pair-based thresholds are too low here: similarity to an average face is higher than
    to a single photo, for strangers too.
    """
    ids, centroids = build_gallery(E, y)
    S = E @ centroids.T
    best_impostor = np.where(y[:, None] == ids[None, :], -np.inf, S).max(axis=1)
    return float(np.quantile(best_impostor, 1 - far))


def run_identification(feats, y, train_idx, test_idx):
    """Closed-set identification: every feature set x every classifier."""
    rows = []
    for fname, F in feats.items():
        for cname, make in CLASSIFIERS.items():
            clf = make()
            t0 = time.perf_counter()
            clf.fit(F[train_idx], y[train_idx])
            fit_s = time.perf_counter() - t0
            pred = clf.predict(F[test_idx])
            rows.append({"features": fname, "classifier": cname,
                         "accuracy": accuracy_score(y[test_idx], pred),
                         "macro_f1": f1_score(y[test_idx], pred, average="macro"),
                         "fit_seconds": fit_s})
    return pd.DataFrame(rows)


def run_verification(feats, y, idx):
    """Verification on every pair of images in idx, one row per feature set."""
    rows = []
    for fname, F in feats.items():
        m = verification_metrics(*all_pairs(F[idx], y[idx]))
        rows.append({"features": fname, **{k: v for k, v in m.items() if k != "roc"}})
    return pd.DataFrame(rows)


def main():
    images, y, _ = load_lfw()
    train_idx, test_idx = split(y)
    feats = feature_sets(images, train_idx)

    ident = run_identification(feats, y, train_idx, test_idx)
    verif = run_verification(feats, y, test_idx)
    REPORTS_DIR.mkdir(exist_ok=True)
    ident.to_csv(REPORTS_DIR / "identification.csv", index=False)
    verif.to_csv(REPORTS_DIR / "verification.csv", index=False)

    # Choose the demo threshold on *training* pairs so the test set stays untouched.
    threshold = gallery_threshold(feats[DEFAULT_MODEL][train_idx], y[train_idx])
    MODELS_DIR.mkdir(exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(
        {"model": DEFAULT_MODEL, "threshold": threshold, "target_far": FAR_TARGET}, indent=2))

    pd.set_option("display.width", 120)
    print("\nIdentification (test set)\n", ident.round(3).to_string(index=False))
    print("\nVerification (all test pairs)\n", verif.round(3).to_string(index=False))
    print(f"\n{DEFAULT_MODEL} gallery threshold at FAR={FAR_TARGET:g} (training data): {threshold:.3f}"
          f" -> {THRESHOLD_FILE.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
