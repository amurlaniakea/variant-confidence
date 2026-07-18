"""T5: test_split_overlap — CI-obligatory (AC3).

Fails explicitly if any gene appears in BOTH train and test. This is the
unit test that prevents a "pretty but lying" ECE caused by homology leakage.
"""
from __future__ import annotations

import pytest

from variant_confidence.data.dataset import build_dataframe
from variant_confidence.split.temporal import temporal_gene_isolated_split


@pytest.fixture(scope="module")
def split_res():
    df = build_dataframe(limit=2000)
    return temporal_gene_isolated_split(
        df, holdout_days=730, min_holdout=500, verbose=False
    )


def test_no_gene_overlap_between_splits(split_res):
    train_genes = set(split_res.train["gene"].unique())
    test_genes = set(split_res.test["gene"].unique())
    overlap = train_genes & test_genes
    assert not overlap, f"Gene leakage detected across train/test: {sorted(overlap)[:10]}"


def test_temporal_ordering_preserved(split_res):
    # All test variants must be strictly newer than the cutoff.
    assert (split_res.test["clinvar_date"] > split_res.cutoff_date).all()


def test_holdout_meets_minimum(split_res):
    assert split_res.n_test_after_isolation >= 500, (
        f"Holdout {split_res.n_test_after_isolation} below AC9 minimum 500"
    )
