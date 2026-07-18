# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
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

"""T8/T9/T11: calibration + ECE tests (AC1, AC2, AC4, AC9).

All use the OFFLINE fixture (6044 records) — deterministic, no network.

ACCEPTANCE CRITERION (audit-strengthened):
  - The synthetic raw score must genuinely discriminate (AUC > 0.7) — guards
    against the degenerate "ignore y" setup we already fixed.
  - Calibration must be fit on a CALIBRATION split and EVALUATED on a
    SEPARATE evaluation split (AC1a). Fitting and predicting on the same
    array let isotonic memorize and report ECE=0.0000 deceptively. We split
    70/30 stratified and measure ECE ONLY on the eval split.
  - Calibration must REDUCE ECE *while preserving* discrimination (AUC).
    If AUC collapses with ECE, the calibrator degenerated to the base rate.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from variant_confidence.calib import calibrate_isotonic, calibrate_platt
from variant_confidence.calib.synthetic import generate_scores
from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.metrics.ece import compute_ece


@pytest.fixture(scope="module")
def split_data():
    df = build_dataframe_from_fixture()
    y = df["label_bin"].to_numpy()
    scores = generate_scores(y, seed=42, overconfidence=0.6)
    # 70/30 stratified split — calibration fit on fit_idx, ECE on eval_idx
    fit_idx, eval_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=42
    )
    return scores, y, fit_idx, eval_idx


def test_raw_score_discriminates(split_data):
    scores, y, _, _ = split_data
    auc = roc_auc_score(y, scores)
    assert auc > 0.7, f"synthetic score non-discriminative (AUC={auc:.3f}) — degenerate"
    print(f"\n[raw AUC] {auc:.4f}")


def test_calibration_reduces_ece_preserves_auc_on_holdout(split_data):
    scores, y, fit_idx, eval_idx = split_data
    # ECE measured ONLY on the eval split (never the fit split)
    raw_ece = compute_ece(scores[eval_idx], y[eval_idx]).ece
    raw_auc = roc_auc_score(y[eval_idx], scores[eval_idx])

    cal_platt = calibrate_platt(scores, y, fit_idx, eval_idx)
    cal_iso = calibrate_isotonic(scores, y, fit_idx, eval_idx)
    ece_platt = compute_ece(cal_platt, y[eval_idx]).ece
    ece_iso = compute_ece(cal_iso, y[eval_idx]).ece
    auc_platt = roc_auc_score(y[eval_idx], cal_platt)
    auc_iso = roc_auc_score(y[eval_idx], cal_iso)

    assert raw_ece > 0.02, f"raw ECE too low to be a meaningful test: {raw_ece}"
    assert ece_platt < raw_ece, f"Platt did not reduce ECE: {raw_ece} -> {ece_platt}"
    assert ece_iso < raw_ece, f"Isotonic did not reduce ECE: {raw_ece} -> {ece_iso}"
    # isotonic must NOT be the deceptively perfect 0.0000 (fit/eval leakage)
    assert ece_iso > 1e-4, (
        f"Isotonic ECE={ece_iso} suggests fit/eval leakage (should be ~0.01 "
        f"on a real holdout, not 0.0000)"
    )
    # discrimination preserved
    assert auc_platt == pytest.approx(raw_auc, abs=0.05), (
        f"Platt destroyed discrimination: {raw_auc:.3f} -> {auc_platt:.3f}"
    )
    assert auc_iso == pytest.approx(raw_auc, abs=0.05), (
        f"Isotonic destroyed discrimination: {raw_auc:.3f} -> {auc_iso:.3f}"
    )
    print(
        f"\n[ECE holdout] raw={raw_ece:.4f} platt={ece_platt:.4f} isotonic={ece_iso:.4f}"
        f" | [AUC] raw={raw_auc:.4f} platt={auc_platt:.4f} iso={auc_iso:.4f}"
    )


def test_ece_reproducible(split_data):
    scores, y, _, eval_idx = split_data
    r1 = compute_ece(scores[eval_idx], y[eval_idx], seed=42)
    r2 = compute_ece(scores[eval_idx], y[eval_idx], seed=42)
    assert r1.ece == pytest.approx(r2.ece, rel=1e-9)


def test_ece_bootstrap_ci_reported(split_data):
    scores, y, _, eval_idx = split_data
    rep = compute_ece(scores[eval_idx], y[eval_idx], bootstrap=200)
    assert rep.ci_low <= rep.ece <= rep.ci_high


def test_single_score_robustness_warns():
    """T11/AC4: a single score source must not crash silently."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    scores = rng.uniform(0.1, 0.9, 500)
    fit_idx = np.arange(350)
    eval_idx = np.arange(350, 500)
    cal = calibrate_platt(scores, y, fit_idx, eval_idx)
    assert cal.shape == (150,)
    assert np.all((cal >= 0) & (cal <= 1))
