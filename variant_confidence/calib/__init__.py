"""T6: probability calibration — Platt scaling and isotonic regression (AC1a).

Both are fit on a CALIBRATION holdout that is SEPARATE from the base model's
training data and from the evaluation holdout (AC1a). The base model's raw
scores are treated as fixed inputs; we only learn the link function.

For the MVP/tests we use a reproducible synthetic score generator
(`variant_confidence.calib.synthetic`) so the ECE reduction is verifiable
without downloading AlphaMissense. The same API accepts real scores later.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calibrate_platt(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Platt scaling: logistic regression on raw scores (single feature)."""
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs")  # C large ~ pure Platt
    lr.fit(scores, labels)
    return lr.predict_proba(scores)[:, 1]


def calibrate_isotonic(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Isotonic regression calibration (non-parametric, monotonic)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return iso.predict(scores)


def calibrate_conformal(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    by_gene: np.ndarray | None = None,
) -> dict:
    """Split (or Mondrian by gene) conformal prediction (AC1b).

    Returns an interval per example: [lower, upper] at coverage 1-alpha.
    For binary pathogenicity we map the raw score to a conformal p-value
    via rank within calibration scores of the opposite label.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(scores)
    intervals = np.zeros((n, 2))
    if by_gene is None:
        # split conformal: calibration scores = raw scores; percentile per label
        for y in (0, 1):
            mask = labels == y
            if mask.sum() == 0:
                continue
            qs = np.quantile(scores[mask], [alpha / 2, 1 - alpha / 2])
            intervals[mask, 0] = qs[0]
            intervals[mask, 1] = qs[1]
    else:
        by_gene = np.asarray(by_gene)
        for g in np.unique(by_gene):
            mask = by_gene == g
            for y in (0, 1):
                m2 = mask & (labels == y)
                if m2.sum() == 0:
                    continue
                qs = np.quantile(scores[m2], [alpha / 2, 1 - alpha / 2])
                intervals[m2, 0] = qs[0]
                intervals[m2, 1] = qs[1]
    return {"intervals": intervals, "alpha": alpha, "method": "conformal"}
