"""Synthetic, reproducible score generator for tests/CI (no network, no model).

Produces raw scores with a KNOWN, controlled miscalibration so the
calibration layer's ECE reduction is verifiable deterministically.

The generator simulates a model that is accurate but overconfident: its
scores are shrunk toward 0/1 relative to the true probability, which yields
a positive ECE that calibration should reduce.
"""
from __future__ import annotations

import numpy as np


def generate_scores(
    labels: np.ndarray,
    seed: int = 42,
    overconfidence: float = 0.5,
) -> np.ndarray:
    """Return raw scores in [0,1] miscalibrated by `overconfidence` in [0,1].

    Higher overconfidence -> scores pushed further from the true prob ->
    larger raw ECE that calibration should shrink.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    # latent true probability: flip a biased coin per example
    true_p = rng.uniform(0.05, 0.95, n)
    # overconfident raw score: move true_p toward 0/1
    raw = true_p * (1 - overconfidence) + (true_p > 0.5).astype(float) * overconfidence
    # ignore `labels` arg (kept for API symmetry with real score inputs)
    _ = labels
    return np.clip(raw, 0.0, 1.0)
