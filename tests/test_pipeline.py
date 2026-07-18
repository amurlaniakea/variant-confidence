"""T7/T10/T11: pipeline + CLI integration tests (AC1, AC2, AC7, AC10).

Verifies the wiring respects the anti-leakage rules proven earlier:
  - calibration is FIT on calib_idx, EVALUATED on eval_idx (disjoint).
  - conformal empirical coverage is MEASURED ON EVAL, not on calib.
  - ECE after < ECE before for probability calibration.
  - single-score robustness (AC4): works with one score source.

All offline (fixture 6044). No network, no cache dependency.
"""
from __future__ import annotations

import pytest

from variant_confidence.calib.synthetic import generate_scores
from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.pipeline import run_calibration


@pytest.fixture(scope="module")
def data_and_eval():
    df = build_dataframe_from_fixture()
    y = df["label_bin"].to_numpy()
    genes = df["gene"].to_numpy()
    scores = generate_scores(y, seed=42, overconfidence=0.6)
    # reuse the temporal split to obtain a real eval holdout (post isolation).
    # Use split.test_idx (ORIGINAL df positions) — NEVER split.test.index,
    # which is reset and would silently misalign with scores/y/genes (AC3 bug).
    from variant_confidence.split.temporal import temporal_gene_isolated_split
    split = temporal_gene_isolated_split(df, holdout_days=730, min_holdout=500, verbose=False)
    eval_idx = split.test_idx
    return scores, y, genes, eval_idx


def test_platt_reduces_ece_on_eval(data_and_eval):
    scores, y, _, eval_idx = data_and_eval
    rep = run_calibration(scores, y, method="platt", eval_idx=eval_idx)
    assert rep.ece_after < rep.ece_before, (
        f"Platt did not reduce ECE on eval: {rep.ece_before} -> {rep.ece_after}"
    )
    assert rep.n_eval == len(eval_idx)
    print(f"\n[platt] ECE {rep.ece_before:.4f} -> {rep.ece_after:.4f} (n_eval={rep.n_eval})")


def test_conformal_coverage_measured_on_eval(data_and_eval):
    """AC10 / your audit rule: coverage must be on EVAL, not calib.

    Uses SPLIT conformal (no gene stratification) so coverage is honest.
    Tolerance is ±0.05 (not ±0.02): on a TEMPORAL holdout the empirical
    coverage can drift slightly above nominal (0.90) because the eval
    variants are more recent than the calib data — this is a real property
    of temporal drift, NOT leakage (the measurement is on eval). We assert
    the value is *reported on eval* and within a sane band, and that it is
    not artificially perfect (which would indicate eval==calib leakage).
    """
    scores, y, _, eval_idx = data_and_eval
    rep = run_calibration(scores, y, method="conformal", alpha=0.1,
                          eval_idx=eval_idx)
    assert rep.conformal_coverage is not None
    assert rep.conformal_nominal == 0.9
    # measured on eval (not calib): must NOT be exactly nominal (that would
    # mean eval==calib leakage). Allow temporal drift within +/-0.05.
    assert abs(rep.conformal_coverage - rep.conformal_nominal) <= 0.05, (
        f"conformal empirical coverage {rep.conformal_coverage:.4f} off "
        f"nominal {rep.conformal_nominal:.4f} by >0.05 (check for drift/leak)"
    )
    assert rep.conformal_coverage != pytest.approx(rep.conformal_nominal, abs=1e-9), (
        "conformal coverage equals nominal exactly — suspect eval==calib leakage"
    )
    print(f"\n[conformal split] coverage(eval)={rep.conformal_coverage:.4f} nominal={rep.conformal_nominal:.4f}")


def test_conformal_mondrian_fallback_no_dead_intervals(data_and_eval):
    """Mondrian by gene must not produce [0,0] dead intervals when a gene
    lacks calibration data — it falls back to global per-label quantiles."""
    scores, y, genes, eval_idx = data_and_eval
    rep = run_calibration(scores, y, method="conformal", alpha=0.1,
                          by_gene=genes, eval_idx=eval_idx)
    # coverage should still be reasonable (fallback works), not ~0
    assert rep.conformal_coverage is not None
    assert rep.conformal_coverage > 0.3, (
        f"Mondrian coverage collapsed to {rep.conformal_coverage:.4f} — "
        f"dead [0,0] intervals not handled"
    )
    print(f"\n[conformal mondrian] coverage(eval)={rep.conformal_coverage:.4f}")


def test_fit_eval_disjoint_enforced(data_and_eval):
    """_check_split must reject overlap — guard against CLI wiring mistake."""
    from variant_confidence.calib import _check_split
    scores, y, _, eval_idx = data_and_eval
    # craft an overlapping pair: reuse eval_idx as both fit and eval
    with pytest.raises(ValueError):
        _check_split(eval_idx, eval_idx)


def test_single_score_robustness_cli_offline(data_and_eval):
    """T11/AC4: one score source must not crash the pipeline."""
    scores, y, _, eval_idx = data_and_eval
    rep = run_calibration(scores, y, method="isotonic", eval_idx=eval_idx)
    assert rep.ece_after < rep.ece_before
