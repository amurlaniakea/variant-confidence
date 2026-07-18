# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 Pedro Sordo Martínez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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


def inject_missing(
    scores: np.ndarray,
    *,
    missing_fraction: float = 0.0,
    missing_genes: list[str] | None = None,
    genes: np.ndarray | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Return a COPY of `scores` with controlled NaN injection.

    Two independent mechanisms (mirroring real AlphaMissense gaps):
      - missing_fraction: random X% of variants set to NaN.
      - missing_genes: ALL variants of these genes set to NaN (entire
        proteins/isoforms without prediction).

    Args:
        scores: base scores in [0,1].
        missing_fraction: fraction of rows (excluding missing_genes) to NaN.
        missing_genes: gene names whose every variant becomes NaN.
        genes: gene per variant (required if missing_genes given).
        seed: RNG seed for reproducible random drop.

    Returns NaN-injected scores. Deterministic given the seed.
    """
    scores = np.asarray(scores, dtype=float).copy()
    rng = np.random.default_rng(seed)
    if missing_genes and genes is not None:
        genes = np.asarray(genes)
        mask = np.isin(genes, list(missing_genes))
        scores[mask] = np.nan
    if missing_fraction > 0.0:
        n = len(scores)
        n_rand = int(round(n * missing_fraction))
        if n_rand > 0:
            idx = rng.choice(n, size=n_rand, replace=False)
            scores[idx] = np.nan
    return scores
