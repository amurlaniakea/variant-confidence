"""T7/T10: end-to-end calibration pipeline + reporting.

Orchestrates the full flow while ENFORCING the anti-leakage rules already
proven in T5 (gene-isolation) and AC10 (fit/eval separation):

  - The temporal holdout (post gene-isolation) is the EVALUATION set.
  - Calibration (Platt/isotonic/conformal) is FIT on a calibration split
    DRAWN FROM THE TRAIN portion (never the eval holdout).
  - ECE and conformal empirical coverage are MEASURED ONLY on the eval
    holdout — never on the calibration/fit set (AC10, your audit condition).

The CLI passes explicit fit_idx/eval_idx; this module never lets them
overlap (delegates to _check_split).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from variant_confidence.calib import (
    _check_split,
    calibrate_conformal,
    calibrate_isotonic,
    calibrate_platt,
)
from variant_confidence.metrics.ece import compute_ece


@dataclass
class CalibrationReport:
    method: str
    ece_before: float
    ece_after: float
    ece_before_ci: tuple[float, float] = (0.0, 0.0)
    ece_after_ci: tuple[float, float] = (0.0, 0.0)
    conformal_coverage: float | None = None  # empirical coverage on eval
    conformal_nominal: float | None = None  # 1 - alpha
    coverage_within_tolerance: bool | None = None
    n_eval: int = 0
    n_calib: int = 0


def run_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "platt",
    alpha: float = 0.1,
    by_gene: np.ndarray | None = None,
    eval_idx: np.ndarray | None = None,
    calib_fraction: float = 0.5,
    seed: int = 42,
) -> CalibrationReport:
    """Calibrate `scores` and report ECE before/after on the EVAL holdout.

    Args:
        scores: raw model scores in [0,1].
        labels: binary labels.
        method: "platt" | "isotonic" | "conformal".
        alpha: conformal miscoverage level (coverage target = 1-alpha).
        by_gene: gene per variant (for Mondrian conformal / gene-isolation).
        eval_idx: indices of the TEMPORAL HOLDOUT (evaluation set). If None,
            a stratified 30% eval split is drawn from within this call — but
            in production the CLI passes the real temporal holdout.
        calib_fraction: fraction of the non-eval data used to FIT calib.
        seed: RNG seed for reproducible calib/eval split.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(scores)

    if eval_idx is None:
        _, eval_idx = train_test_split(
            np.arange(n), test_size=0.3, stratify=labels, random_state=seed
        )
    eval_idx = np.asarray(eval_idx)
    # calibration indices: a subset of the NON-eval data
    non_eval = np.setdiff1d(np.arange(n), eval_idx)
    calib_idx, _ = train_test_split(
        non_eval, test_size=1.0 - calib_fraction, random_state=seed
    )
    _check_split(calib_idx, eval_idx)  # enforces no overlap (AC10)

    e_before = compute_ece(scores[eval_idx], labels[eval_idx], seed=seed)
    rep = CalibrationReport(
        method=method,
        ece_before=e_before.ece,
        ece_before_ci=(e_before.ci_low, e_before.ci_high),
        ece_after=0.0,
        n_eval=int(len(eval_idx)),
        n_calib=int(len(calib_idx)),
    )

    if method in ("platt", "isotonic"):
        fn = calibrate_platt if method == "platt" else calibrate_isotonic
        cal = fn(scores, labels, calib_idx, eval_idx)
        e_after = compute_ece(cal, labels[eval_idx], seed=seed)
        rep.ece_after = e_after.ece
        rep.ece_after_ci = (e_after.ci_low, e_after.ci_high)
    elif method == "conformal":
        conf = calibrate_conformal(
            scores, labels, calib_idx, eval_idx, alpha=alpha, by_gene=by_gene
        )
        intervals = conf["intervals"]
        # empirical coverage MEASURED ON EVAL (not calib) — your audit rule
        lo = intervals[:, 0]
        hi = intervals[:, 1]
        covered = (scores[eval_idx] >= lo) & (scores[eval_idx] <= hi)
        empirical = float(covered.mean())
        rep.conformal_coverage = empirical
        rep.conformal_nominal = 1.0 - alpha
        rep.coverage_within_tolerance = abs(empirical - (1.0 - alpha)) <= 0.02
        # For conformal there is no single "calibrated probability"; report
        # ECE of the raw score (unchanged) and coverage instead.
        rep.ece_after = e_before.ece
        rep.ece_after_ci = (e_before.ci_low, e_before.ci_high)
    else:
        raise ValueError(f"unknown method {method}")

    return rep
