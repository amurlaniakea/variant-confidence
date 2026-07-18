"""Synthetic, reproducible score generator for tests/CI (no network, no model).

Produces raw scores with a KNOWN, controlled miscalibration so the
calibration layer's ECE reduction is verifiable deterministically.

CRITICAL (audit finding): the raw score MUST be correlated with the REAL
label. A previous version drew an independent `true_p` and ignored `y`,
which made the score non-discriminative (AUC ~0.5); any calibrator then
collapsed to the base rate and ECE->0 degenerately. Here `true_p` is
derived from the real label plus noise, so the score discriminates, is
deliberately overconfident (miscalibrated), and calibration must reduce
ECE *while preserving* discrimination (AUC) — not by collapsing.
"""
from __future__ import annotations

import numpy as np


def generate_scores(
    labels: np.ndarray,
    seed: int = 42,
    overconfidence: float = 0.5,
) -> np.ndarray:
    """Return raw scores in [0,1] miscalibrated by `overconfidence` in [0,1].

    The score is derived from the REAL label plus Gaussian noise, so it is
    genuinely predictive (AUC well above 0.5). `overconfidence` pushes the
    score toward 0/1 relative to its true probability, creating the ECE that
    calibration should shrink *without* destroying discrimination.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    n = len(labels)
    # true_p correlated with the real label but with realistic overlap:
    # centroids at 0.3 (benign) / 0.7 (pathogenic) + Gaussian noise, so the
    # two classes overlap (AUC ~0.8, not a degenerate 1.0).
    noise = rng.normal(0.0, 0.18, n)
    true_p = np.where(labels > 0.5, 0.7, 0.3) + noise
    true_p = np.clip(true_p, 0.02, 0.98)
    # overconfident raw score: move true_p toward 0/1
    raw = true_p * (1 - overconfidence) + (true_p > 0.5).astype(float) * overconfidence
    return np.clip(raw, 0.0, 1.0)
