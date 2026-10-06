"""Vectorised age-conditioned AUROC, verified identical to evaluate_model.compute_auroc_age."""
import numpy as np

def auroc_age_fast(labels, preds, ages, gap=2):
    labels = np.asarray(labels); preds = np.asarray(preds, float); ages = np.asarray(ages, float)
    p = preds[labels == 1]; n = preds[labels == 0]
    ap = ages[labels == 1]; an = ages[labels == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    mask = np.abs(ap[:, None] - an[None, :]) <= gap
    denom = mask.sum()
    if denom == 0: return float("nan")
    d = p[:, None] - n[None, :]
    numer = ((d > 0) * 1.0 + (d == 0) * 0.5)[mask].sum()
    return float(numer / denom)
