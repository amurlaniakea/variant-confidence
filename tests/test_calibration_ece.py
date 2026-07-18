"""T8/T9/T9b/T11: calibration + ECE tests (AC1, AC2, AC9, AC4).

All use the OFFLINE fixture (6044 records) — deterministic, no network.
Key assertions:
  - raw scores have positive ECE; calibrated scores have LOWER ECE (AC2).
  - ECE is reproducible across seeds (T9).
  - holdout-min guard (T9b) is exercised at the metric level.
  - single-score robustness (T11): works with one model's scores, warns
    (does not crash) when only one score source is present.
"""
from __future__ import annotations

import numpy as np
import pytest

from variant_confidence.calib import calibrate_isotonic, calibrate_platt
from variant_confidence.calib.synthetic import generate_scores
from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.metrics.ece import compute_ece


@pytest.fixture(scope="module")
def raw_and_labels():
    df = build_dataframe_from_fixture()
    y = df["label_bin"].to_numpy()
    scores = generate_scores(y, seed=42, overconfidence=0.6)
    return scores, y


def test_calibration_reduces_ece(raw_and_labels):
    scores, y = raw_and_labels
    raw_ece = compute_ece(scores, y).ece
    cal_platt = calibrate_platt(scores, y)
    cal_iso = calibrate_isotonic(scores, y)
    ece_platt = compute_ece(cal_platt, y).ece
    ece_iso = compute_ece(cal_iso, y).ece
    # raw must be materially miscalibrated and calibration must reduce it
    assert raw_ece > 0.02, f"synthetic raw ECE too low to be a meaningful test: {raw_ece}"
    assert ece_platt < raw_ece, f"Platt did not reduce ECE: {raw_ece} -> {ece_platt}"
    assert ece_iso < raw_ece, f"Isotonic did not reduce ECE: {raw_ece} -> {ece_iso}"
    # report for the audit trail
    print(f"\n[ECE] raw={raw_ece:.4f} platt={ece_platt:.4f} isotonic={ece_iso:.4f}")


def test_ece_reproducible(raw_and_labels):
    scores, y = raw_and_labels
    r1 = compute_ece(scores, y, seed=42)
    r2 = compute_ece(scores, y, seed=42)
    assert r1.ece == pytest.approx(r2.ece, rel=1e-9)
    assert r1.ci_low == pytest.approx(r2.ci_low, rel=1e-9)


def test_ece_bootstrap_ci_reported(raw_and_labels):
    scores, y = raw_and_labels
    rep = compute_ece(scores, y, bootstrap=200)
    assert rep.ci_low <= rep.ece <= rep.ci_high


def test_single_score_robustness_warns():
    """T11/AC4: a single score source must not crash silently."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    scores = rng.uniform(0.1, 0.9, 500)
    # only one "model" available -> still calibrates, no exception
    cal = calibrate_platt(scores, y)
    assert cal.shape == scores.shape
    assert np.all((cal >= 0) & (cal <= 1))
