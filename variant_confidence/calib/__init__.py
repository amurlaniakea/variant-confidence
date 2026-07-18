"""T6: probability calibration — Platt scaling, isotonic, conformal (AC1a).

CRITICAL (audit finding, twice): every calibrator MUST be fit on a
CALIBRATION split and evaluated on a SEPARATE evaluation split. Fitting and
predicting on the same array lets non-parametric methods (isotonic,
conformal) memorize the calibration curve of those exact points and report
a deceptively perfect ECE (e.g. 0.0000). This is the same information
leakage class as T5's gene-isolation, applied to calibration.

Therefore each `calibrate_*` function takes explicit `fit_idx` /
`eval_idx` (or a `calib_mask`) so the CALLER (the pipeline) imposes the
split. The function never decides the split itself — the pipeline owns it,
mirroring how AC3 owns the temporal split.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _check_split(fit_idx: np.ndarray, eval_idx: np.ndarray) -> None:
    overlap = np.intersect1d(fit_idx, eval_idx)
    if overlap.size > 0:
        raise ValueError(
            f"calibration fit/eval overlap ({overlap.size} shared indices) — "
            f"this is information leakage (same class as T5 gene-isolation)"
        )


def calibrate_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> np.ndarray:
    """Platt scaling fit on `fit_idx`, evaluated on `eval_idx`."""
    _check_split(fit_idx, eval_idx)
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(scores[fit_idx], labels[fit_idx])
    return lr.predict_proba(scores[eval_idx])[:, 1]


def calibrate_isotonic(
    scores: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> np.ndarray:
    """Isotonic regression fit on `fit_idx`, evaluated on `eval_idx`."""
    _check_split(fit_idx, eval_idx)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores[fit_idx], labels[fit_idx])
    return iso.predict(scores[eval_idx])


def calibrate_conformal(
    scores: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    alpha: float = 0.1,
    by_gene: np.ndarray | None = None,
) -> dict:
    """Split (or Mondrian by gene) conformal prediction (AC1b).

    Calibration quantiles are computed on `fit_idx` ONLY; intervals are
    returned for `eval_idx`. Evaluating on the same set used to compute
    quantiles would inflate empirical coverage to exactly 1-alpha by
    construction — a leakage we forbid.
    """
    _check_split(fit_idx, eval_idx)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_eval = len(eval_idx)
    intervals = np.zeros((n_eval, 2))
    mondrian_fallback_rate = 0.0

    def _quantiles(sub_scores: np.ndarray) -> tuple[float, float]:
        return tuple(np.quantile(sub_scores, [alpha / 2, 1 - alpha / 2]))

    if by_gene is None:
        for y in (0, 1):
            mask = labels[fit_idx] == y
            if mask.sum() == 0:
                continue
            qs = _quantiles(scores[fit_idx][mask])
            m_eval = labels[eval_idx] == y
            intervals[m_eval, 0] = qs[0]
            intervals[m_eval, 1] = qs[1]
    else:
        by_gene = np.asarray(by_gene)
        # precompute global per-label quantiles as FALLBACK for eval genes
        # that have no calibration data in fit (avoids [0,0] dead intervals)
        global_q = {}
        for y in (0, 1):
            mask = labels[fit_idx] == y
            global_q[y] = _quantiles(scores[fit_idx][mask]) if mask.sum() else (0.0, 1.0)
        n_fallback = 0
        for g in np.unique(by_gene):
            m_fit = by_gene[fit_idx] == g
            for y in (0, 1):
                m2 = m_fit & (labels[fit_idx] == y)
                if m2.sum() == 0:
                    qs = global_q[y]
                    n_fallback += int(((by_gene[eval_idx] == g) & (labels[eval_idx] == y)).sum())
                else:
                    qs = _quantiles(scores[fit_idx][m2])
                m_eval = (by_gene[eval_idx] == g) & (labels[eval_idx] == y)
                intervals[m_eval, 0] = qs[0]
                intervals[m_eval, 1] = qs[1]
        # With AC3 gene-isolation, NO test gene ever appears in train, so
        # calib_idx (subset of train) never has per-gene data for eval genes.
        # The Mondrian fallback therefore fires for ~100% of eval — Mondrian
        # is then identical to split conformal by construction. We report the
        # rate so the user is not misled into thinking per-gene stratification
        # actually happened.
        mondrian_fallback_rate = n_fallback / n_eval if n_eval else 0.0

    return {
        "intervals": intervals,
        "alpha": alpha,
        "method": "conformal",
        "mondrian_fallback_rate": mondrian_fallback_rate,
    }
