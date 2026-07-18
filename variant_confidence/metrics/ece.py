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

"""T8: ECE metric (AC2, AC9).

Expected Calibration Error with:
  - 10 equal-width bins (minimum required by AC2)
  - adaptive binning (reference)
  - bootstrap CI (1000 resamples) on the ECE estimate
  - per-bin count report (flag bins with < 25 samples as low-reliability)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EceReport:
    ece: float
    ece_adaptive: float
    ci_low: float
    ci_high: float
    n_bins: int
    bin_counts: list[int]
    low_reliability_bins: int  # bins with < 25 samples


def _ece_from_probs(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += cnt / n * abs(avg_conf - avg_acc)
    return float(ece)


def _adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Adaptive (equal-mass) binning ECE (reference metric)."""
    order = np.argsort(probs)
    n = len(probs)
    chunk = max(1, n // n_bins)
    ece = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = order[start:end]
        cnt = len(idx)
        avg_conf = probs[idx].mean()
        avg_acc = labels[idx].mean()
        ece += cnt / n * abs(avg_conf - avg_acc)
    return float(ece)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    bootstrap: int = 1000,
    min_bin: int = 25,
    seed: int = 42,
) -> EceReport:
    """Compute ECE (equal-width) + adaptive, with bootstrap CI and bin counts.

    Args:
        probs: calibrated/raw probabilities in [0,1].
        labels: binary labels (0/1).
        n_bins: equal-width bin count (AC2 minimum 10).
        bootstrap: number of bootstrap resamples for CI.
        min_bin: bins with fewer samples flagged low-reliability (AC9).
        seed: RNG seed for reproducible CI.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)

    base_ece = _ece_from_probs(probs, labels, n_bins)
    base_adaptive = _adaptive_ece(probs, labels, n_bins)

    # bootstrap CI on the equal-width ECE
    n = len(probs)
    boot = np.empty(bootstrap)
    for i in range(bootstrap):
        idx = rng.integers(0, n, n)
        boot[i] = _ece_from_probs(probs[idx], labels[idx], n_bins)
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))

    # per-bin counts (equal-width)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    counts = [int((bin_idx == b).sum()) for b in range(n_bins)]
    low = sum(1 for c in counts if c < min_bin)

    return EceReport(
        ece=base_ece,
        ece_adaptive=base_adaptive,
        ci_low=ci_low,
        ci_high=ci_high,
        n_bins=n_bins,
        bin_counts=counts,
        low_reliability_bins=low,
    )


if __name__ == "__main__":
    # Sanity: a perfectly calibrated model has ECE ~ 0.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    rep = compute_ece(p, y)
    print(f"ECE (calibrated-sim): {rep.ece:.4f}  CI[{rep.ci_low:.4f},{rep.ci_high:.4f}]")
