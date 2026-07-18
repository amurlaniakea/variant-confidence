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

"""T12: missing-score handling (AC4 reinforcement, anti-bug #5).

AlphaMissense does NOT cover 100% of ClinVar variants. A missing score MUST
NOT be treated as 0 / a confident prediction. This test verifies:
  - on_missing="fail" raises with an explicit, non-silent message.
  - on_missing="degrade" excludes missing rows, reports n_missing /
    fraction_missing, and the ECE is computed ONLY on present scores (not
    inflated by implicit zeros).
  - Degrade result matches computing ECE on the present subset directly
    (no silent leakage of missing-as-0).
"""
from __future__ import annotations

import numpy as np
import pytest

from variant_confidence.calib.synthetic import generate_scores, inject_missing
from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.pipeline import run_calibration


@pytest.fixture(scope="module")
def base():
    df = build_dataframe_from_fixture()
    y = df["label_bin"].to_numpy()
    genes = df["gene"].to_numpy()
    scores = generate_scores(y, seed=42, overconfidence=0.6)
    from variant_confidence.split.temporal import temporal_gene_isolated_split
    split = temporal_gene_isolated_split(df, holdout_days=730, min_holdout=500, verbose=False)
    return scores, y, genes, np.asarray(split.test_idx)


def test_fail_mode_raises_on_missing(base):
    scores, y, _, eval_idx = base
    miss = inject_missing(scores, missing_fraction=0.15, seed=7)
    with pytest.raises(ValueError) as exc:
        run_calibration(miss, y, method="platt", eval_idx=eval_idx, on_missing="fail")
    assert "Missing scores" in str(exc.value)
    assert "AC4" in str(exc.value) or "silent" in str(exc.value).lower()


def test_degrade_excludes_missing_and_reports(base):
    scores, y, _, eval_idx = base
    miss = inject_missing(scores, missing_fraction=0.15, seed=7)
    rep = run_calibration(miss, y, method="platt", eval_idx=eval_idx, on_missing="degrade")
    # n_missing must be reported and non-zero
    assert rep.n_missing > 0
    assert rep.fraction_missing > 0
    assert rep.degraded is True
    # ECE_after must be computed on PRESENT scores only. Compare against
    # running on the present subset directly (no implicit zeros).
    present = ~np.isnan(miss)
    present_scores = miss[present]
    present_y = y[present]
    # eval_present must be positions in the TRIMMED present_scores array, not
    # original df positions. Map via the surviving-original positions.
    survived_orig = np.where(present)[0]
    orig_to_new = np.full(len(miss), -1, dtype=int)
    orig_to_new[survived_orig] = np.arange(len(survived_orig))
    eval_present_orig = np.array([i for i in eval_idx if present[i]])
    eval_present = orig_to_new[eval_present_orig]
    ece_direct = run_calibration(
        present_scores, present_y, method="platt", eval_idx=eval_present, on_missing="fail"
    )
    assert rep.ece_after == pytest.approx(ece_direct.ece_after, rel=1e-9), (
        f"degrade ECE {rep.ece_after} != direct-on-present {ece_direct.ece_after} "
        f"— missing rows leaking into ECE"
    )
    print(f"\n[degrade] n_missing={rep.n_missing} frac={rep.fraction_missing:.3f} "
          f"ECE_after={rep.ece_after:.4f}")


def test_degrade_with_missing_genes(base):
    """Entire proteins without prediction (real AlphaMissense gap)."""
    scores, y, genes, eval_idx = base
    # pick a gene that actually appears in the eval holdout
    from variant_confidence.split.temporal import temporal_gene_isolated_split
    df = build_dataframe_from_fixture()
    split = temporal_gene_isolated_split(df, holdout_days=730, min_holdout=500, verbose=False)
    eval_genes = set(df.iloc[np.asarray(split.test_idx)]["gene"].unique())
    some_gene = sorted(eval_genes)[0]
    miss = inject_missing(scores, missing_genes=[some_gene], genes=genes, seed=7)
    rep = run_calibration(miss, y, method="isotonic", eval_idx=eval_idx, on_missing="degrade")
    # all variants of that gene must be excluded; ECE still valid
    assert rep.n_missing > 0
    assert rep.ece_after < rep.ece_before, "degrade ECE must still reduce"
    print(f"\n[degrade-gene] gene={some_gene} n_missing={rep.n_missing}")
