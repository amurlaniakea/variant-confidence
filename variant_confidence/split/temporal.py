# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
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

"""T4: temporal + gene-isolated split (anti-leakage, AC3).

The split is by ClinVar release date: the most recent variants (after a
cutoff) form the TEMPORAL TEST set. Critically, to avoid homology/label
leakage, NO gene present in the test set may appear in the train set.

CRITICAL (audit finding): we compute `test_idx` / `train_idx` as POSITIONS
IN THE ORIGINAL `df` (before any reset_index). The returned DataFrames are
reset for convenience, but `SplitResult.test_idx` / `train_idx` carry the
original positions so downstream code (pipeline / CLI) can index the
aligned `scores` / `y` / `genes` arrays WITHOUT reconstructing indices by
hand. Using `split.test.index` after a reset_index would silently point at
the wrong rows and reintroduce gene leakage (AC3) — this is exactly the
bug that was caught and fixed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_date: pd.Timestamp
    n_test_after_isolation: int
    n_test_candidates: int
    genes_in_test: int
    genes_in_train: int
    genes_overlap: int
    # Positions in the ORIGINAL df (pre-reset). Use THESE to index aligned
    # arrays (scores/y/genes). Never use .test.index (it was reset).
    test_idx: list[int] = field(default_factory=list)
    train_idx: list[int] = field(default_factory=list)


def temporal_gene_isolated_split(
    df: pd.DataFrame,
    holdout_days: int = 365,
    min_holdout: int = 500,
    verbose: bool = True,
) -> SplitResult:
    """Split df into train/test by date with gene isolation.

    Args:
        df: must have columns gene, clinvar_date, label_bin.
        holdout_days: test = variants with clinvar_date > (max_date - holdout_days).
        min_holdout: if post-isolation test < min_holdout, raise (AC9).
        verbose: print a diagnostic block (used by T5 / audit).
    """
    # Keep the original positions. We capture them from df_sorted (post-sort)
    # so orig_pos is aligned with candidates_mask / kept_cand_mask below.
    df = df.reset_index(drop=False)  # 'index' column = original position
    df_sorted = df.sort_values("clinvar_date").reset_index(drop=True)
    orig_pos = df_sorted["index"].to_numpy()
    max_date = df_sorted["clinvar_date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)

    candidates_mask = df_sorted["clinvar_date"] > cutoff
    train_genes = set(df_sorted.loc[~candidates_mask, "gene"].unique())
    candidate_genes = set(df_sorted.loc[candidates_mask, "gene"].unique())
    overlap_genes = train_genes & candidate_genes

    # Test keeps only candidates whose gene is NOT in train.
    kept_cand_mask = candidates_mask & ~df_sorted["gene"].isin(train_genes)
    test = df_sorted[kept_cand_mask].reset_index(drop=True)
    test_idx = orig_pos[kept_cand_mask.to_numpy()].tolist()

    # Train = everything else (base + candidate genes dropped from test to
    # preserve isolation — they go back to train to avoid leakage either way).
    train_pos = orig_pos[(~kept_cand_mask).to_numpy()].tolist()

    if verbose:
        print(
            f"[split] max_date={max_date.date()} cutoff={cutoff.date()} "
            f"holdout_days={holdout_days}\n"
            f"  candidates(date>{cutoff.date()}): {int(candidates_mask.sum())}\n"
            f"  after gene-isolation: {len(test_idx)} (need >= {min_holdout})\n"
            f"  genes in test={len(candidate_genes - train_genes)}, "
            f"overlap genes dropped={len(overlap_genes)}"
        )

    if len(test_idx) < min_holdout:
        raise ValueError(
            f"Temporal holdout after gene-isolation is {len(test_idx)} < "
            f"min_holdout={min_holdout}. Increase holdout_days or relax "
            f"gene-isolation, or reduce min_holdout. AC9: do NOT emit ECE on "
            f"an unreliable holdout."
        )

    return SplitResult(
        train=df_sorted[~kept_cand_mask].reset_index(drop=True),
        test=test,
        cutoff_date=cutoff,
        n_test_after_isolation=len(test_idx),
        n_test_candidates=int(candidates_mask.sum()),
        genes_in_test=len(set(test["gene"].unique())),
        genes_in_train=len(train_genes),
        genes_overlap=len(overlap_genes),
        test_idx=test_idx,
        train_idx=train_pos,
    )


if __name__ == "__main__":
    from .dataset import build_dataframe

    df = build_dataframe(limit=2000)
    res = temporal_gene_isolated_split(df, holdout_days=365, min_holdout=500)
