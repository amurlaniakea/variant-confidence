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

"""Guard test for the AC3 index-alignment bug (audit finding).

Verifies that SplitResult.test_idx maps to positions in the ORIGINAL df
whose genes are truly disjoint from train genes. Using split.test.index
(after reset_index) silently mapped eval to the first N rows of df and
let 238/1088 "eval" genes actually belong to train — invalidating AC3.
"""
from __future__ import annotations

import numpy as np
import pytest

from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.split.temporal import temporal_gene_isolated_split


@pytest.fixture(scope="module")
def split_and_df():
    df = build_dataframe_from_fixture()
    split = temporal_gene_isolated_split(df, holdout_days=730, min_holdout=500, verbose=False)
    return df, split


def test_test_idx_genes_disjoint_from_train(split_and_df):
    df, split = split_and_df
    test_idx = np.asarray(split.test_idx)
    train_idx = np.asarray(split.train_idx)
    test_genes = set(df.iloc[test_idx]["gene"].unique())
    train_genes = set(df.iloc[train_idx]["gene"].unique())
    overlap = test_genes & train_genes
    assert not overlap, (
        f"AC3 VIOLATION: {len(overlap)} genes appear in BOTH eval and train "
        f"via test_idx. This is the audit-found index-misalignment bug."
    )
    # also: test_idx must be a strict subset of the temporal candidates
    # (date > cutoff), i.e. genuinely the recent holdout.
    cutoff = split.cutoff_date
    recent = df.iloc[test_idx]["clinvar_date"] > cutoff
    assert recent.all(), "test_idx contains non-recent variants (not the temporal holdout)"


def test_test_idx_differs_from_reset_index(split_and_df):
    """The bug was using split.test.index (reset -> [0,1,2,...]). Assert that
    test_idx is NOT the trivial range [0..n_test), proving it carries real
    original positions."""
    df, split = split_and_df
    test_idx = np.asarray(split.test_idx)
    n = len(test_idx)
    is_trivial_range = np.array_equal(test_idx, np.arange(n))
    assert not is_trivial_range, (
        "test_idx equals [0,1,..n) — that's the reset_index trap; it must "
        "carry ORIGINAL df positions"
    )
