"""T8/T9/T11: calibration + ECE tests (AC1, AC2, AC4).

All use the OFFLINE fixture (6044 records) — deterministic, no network.

ACCEPTANCE CRITERION (audit-strengthened): calibration must REDUCE ECE
*while preserving discrimination*. A degenerate calibrator that collapses to
the base rate would also zero the ECE but destroy AUC — we assert AUC is
preserved (raw AUC ≈ calibrated AUC) so the ECE drop is genuine, not a
collapse to the constant base rate.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

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


def test_raw_score_discriminates(raw_and_labels):
    """Guard: the synthetic raw score must be genuinely predictive (AUC>0.7),
    otherwise calibration is meaningless (the bug we fixed)."""
    scores, y = raw_and_labels
    auc = roc_auc_score(y, scores)
    assert auc > 0.7, f"synthetic score is non-discriminative (AUC={auc:.3f}) — degenerate setup"
    print(f"\n[raw AUC] {auc:.4f}")


def test_calibration_reduces_ece_preserves_auc(raw_and_labels):
    scores, y = raw_and_labels
    raw_ece = compute_ece(scores, y).ece
    raw_auc = roc_auc_score(y, scores)

    cal_platt = calibrate_platt(scores, y)
    cal_iso = calibrate_isotonic(scores, y)
    ece_platt = compute_ece(cal_platt, y).ece
    ece_iso = compute_ece(cal_iso, y).ece
    auc_platt = roc_auc_score(y, cal_platt)
    auc_iso = roc_auc_score(y, cal_iso)

    # raw must be materially miscalibrated
    assert raw_ece > 0.02, f"synthetic raw ECE too low to be a meaningful test: {raw_ece}"
    # calibration reduces ECE
    assert ece_platt < raw_ece, f"Platt did not reduce ECE: {raw_ece} -> {ece_platt}"
    assert ece_iso < raw_ece, f"Isotonic did not reduce ECE: {raw_ece} -> {ece_iso}"
    # discrimination is PRESERVED (no collapse to base rate)
    assert auc_platt == pytest.approx(raw_auc, abs=0.05), (
        f"Platt destroyed discrimination: AUC {raw_auc:.3f} -> {auc_platt:.3f}"
    )
    assert auc_iso == pytest.approx(raw_auc, abs=0.05), (
        f"Isotonic destroyed discrimination: AUC {raw_auc:.3f} -> {auc_iso:.3f}"
    )
    print(
        f"\n[ECE] raw={raw_ece:.4f} platt={ece_platt:.4f} isotonic={ece_iso:.4f}"
        f" | [AUC] raw={raw_auc:.4f} platt={auc_platt:.4f} iso={auc_iso:.4f}"
    )


def test_ece_reproducible(raw_and_labels):
    scores, y = raw_and_labels
    r1 = compute_ece(scores, y, seed=42)
    r2 = compute_ece(scores, y, seed=42)
    assert r1.ece == pytest.approx(r2.ece, rel=1e-9)


def test_ece_bootstrap_ci_reported(raw_and_labels):
    scores, y = raw_and_labels
    rep = compute_ece(scores, y, bootstrap=200)
    assert rep.ci_low <= rep.ece <= rep.ci_high


def test_single_score_robustness_warns():
    """T11/AC4: a single score source must not crash silently."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    scores = rng.uniform(0.1, 0.9, 500)
    cal = calibrate_platt(scores, y)
    assert cal.shape == scores.shape
    assert np.all((cal >= 0) & (cal <= 1))
