"""Metrics for verification (1:1, 'same person?') and open-set identification (1:N + unknown)."""
import numpy as np
from sklearn.metrics import auc, roc_curve

from .matching import UNKNOWN


def all_pairs(E, y):
    """Cosine similarity and same-person label (0/1) for every pair. E must be L2-normalised."""
    S = E @ E.T
    i, j = np.triu_indices(len(y), k=1)
    return S[i, j], (y[i] == y[j]).astype(int)


def verification_metrics(scores, same):
    """AUC, equal error rate, and true-accept rate at fixed false-accept rates (with thresholds)."""
    fpr, tpr, thr = roc_curve(same, scores)
    fnr = 1 - tpr
    k = np.argmin(np.abs(fpr - fnr))
    out = {"auc": auc(fpr, tpr), "eer": (fpr[k] + fnr[k]) / 2, "eer_threshold": float(thr[k])}
    for far in (0.01, 0.001):
        m = np.searchsorted(fpr, far, side="right") - 1
        out[f"tar@far={far:g}"] = tpr[m]
        out[f"threshold@far={far:g}"] = float(thr[m])
    out["roc"] = (fpr, tpr, thr)
    return out


def open_set_metrics(y_true, y_pred, known_ids):
    """Anyone whose identity is not in known_ids should be predicted UNKNOWN."""
    known = np.isin(y_true, known_ids)
    return {
        "known_accuracy": np.mean(y_pred[known] == y_true[known]),     # named correctly
        "false_reject_rate": np.mean(y_pred[known] == UNKNOWN),        # known person called unknown
        "false_accept_rate": np.mean(y_pred[~known] != UNKNOWN),       # stranger given a name
    }
