"""T7/T11b: Mondrian conformal must declare its fallback rate (audit note).

With AC3 gene-isolation, no eval gene has calib data of its own, so the
Mondrian fallback fires for ~100% of eval. The CLI must surface this so the
user is not misled into thinking per-gene stratification happened. This
test asserts the report carries the fallback rate and that it is ~1.0 on
the temporal holdout.
"""
from __future__ import annotations

import numpy as np
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
    from variant_confidence.split.temporal import temporal_gene_isolated_split
    split = temporal_gene_isolated_split(df, holdout_days=730, min_holdout=500, verbose=False)
    return scores, y, genes, np.asarray(split.test_idx)


def test_mondrian_fallback_declared_and_full(data_and_eval):
    scores, y, genes, eval_idx = data_and_eval
    rep = run_calibration(scores, y, method="conformal", alpha=0.1,
                          by_gene=genes, eval_idx=eval_idx)
    assert rep.mondrian_fallback_rate is not None
    # On the temporal holdout, AC3 guarantees no eval gene has calib data,
    # so Mondrian == split conformal by construction.
    assert rep.mondrian_fallback_rate >= 0.999, (
        f"Mondrian fallback rate {rep.mondrian_fallback_rate:.3f} not ~1.0 — "
        f"unexpected per-gene stratification on an AC3-isolated holdout"
    )
    print(f"\n[mondrian] fallback_rate={rep.mondrian_fallback_rate:.3f} (expected ~1.0)")
