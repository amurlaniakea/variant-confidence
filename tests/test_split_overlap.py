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

"""T5: test_split_overlap — CI-obligatory (AC3, AC9).

Fails explicitly if any gene appears in BOTH train and test (leakage), and
asserts the temporal holdout meets the AC9 minimum WITH A REAL MARGIN.
Uses the OFFLINE, versioned fixture (tests/fixtures/clinvar_sample.json,
6044 real ClinVar records) so the test is deterministic and
network-independent.
"""
from __future__ import annotations

import pytest

from variant_confidence.data.dataset import build_dataframe_from_fixture
from variant_confidence.split.temporal import temporal_gene_isolated_split


@pytest.fixture(scope="module")
def split_res():
    # OFFLINE fixture: 6044 real ClinVar variants (1591 genes).
    df = build_dataframe_from_fixture()
    return temporal_gene_isolated_split(
        df, holdout_days=730, min_holdout=500, verbose=False
    )


def test_no_gene_overlap_between_splits(split_res):
    train_genes = set(split_res.train["gene"].unique())
    test_genes = set(split_res.test["gene"].unique())
    overlap = train_genes & test_genes
    assert not overlap, f"Gene leakage detected across train/test: {sorted(overlap)[:10]}"


def test_temporal_ordering_preserved(split_res):
    assert (split_res.test["clinvar_date"] > split_res.cutoff_date).all()


def test_holdout_meets_minimum_with_real_margin(split_res):
    # AC9: not just >=500, but a HOLGADO margin. With the 6044-variant fixture
    # the temporal holdout after gene-isolation is ~3000 (margin ~2500),
    # proving robustness — not the fragile 502+/-2 seen at limit=2000.
    assert split_res.n_test_after_isolation >= 500, (
        f"Holdout {split_res.n_test_after_isolation} below AC9 minimum 500"
    )
    assert split_res.n_test_after_isolation >= 1000, (
        f"Holdout {split_res.n_test_after_isolation} is too close to the 500 "
        f"minimum — fragile (AC9 requires a real margin)."
    )
