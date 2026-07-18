"""T4: temporal + gene-isolated split (anti-leakage, AC3).

The split is by ClinVar release date: the most recent variants (after a
cutoff) form the TEMPORAL TEST set. Critically, to avoid homology/label
leakage, NO gene present in the test set may appear in the train set.

Because ClinVar is dominated by a few large genes (e.g. BRCA1/2, TP53),
gene-isolation can shrink the temporal test set dramatically. We therefore:
  - pick a date cutoff (default: latest date minus `holdout_days`);
  - collect candidate test variants (date > cutoff);
  - KEEP a test variant only if its gene is absent from the train portion;
  - report the resulting test size so AC9 (n>=500) can be checked.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    df = df.sort_values("clinvar_date").reset_index(drop=True)
    max_date = df["clinvar_date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)

    candidates = df[df["clinvar_date"] > cutoff]
    train_base = df[df["clinvar_date"] <= cutoff]

    # Genes already represented in train must NOT appear in test.
    train_genes = set(train_base["gene"].unique())
    candidate_genes = set(candidates["gene"].unique())
    overlap_genes = train_genes & candidate_genes

    # Test keeps only candidates whose gene is NOT in train.
    test = candidates[~candidates["gene"].isin(train_genes)].reset_index(drop=True)
    # Train = everything else (base + candidate genes that were dropped from test
    # to preserve isolation — they go back to train to avoid leakage either way).
    dropped = candidates[candidates["gene"].isin(train_genes)]
    train = pd.concat([train_base, dropped], ignore_index=True)

    if verbose:
        print(
            f"[split] max_date={max_date.date()} cutoff={cutoff.date()} "
            f"holdout_days={holdout_days}\n"
            f"  candidates(date>{cutoff.date()}): {len(candidates)}\n"
            f"  after gene-isolation: {len(test)} (need >= {min_holdout})\n"
            f"  genes in test={len(candidate_genes - train_genes)}, "
            f"overlap genes dropped={len(overlap_genes)}"
        )

    if len(test) < min_holdout:
        raise ValueError(
            f"Temporal holdout after gene-isolation is {len(test)} < "
            f"min_holdout={min_holdout}. Increase holdout_days or relax "
            f"gene-isolation, or reduce min_holdout. AC9: do NOT emit ECE on "
            f"an unreliable holdout."
        )

    return SplitResult(
        train=train,
        test=test,
        cutoff_date=cutoff,
        n_test_after_isolation=len(test),
        n_test_candidates=len(candidates),
        genes_in_test=len(set(test["gene"].unique())),
        genes_in_train=len(set(train["gene"].unique())),
        genes_overlap=len(overlap_genes),
    )


if __name__ == "__main__":
    from .dataset import build_dataframe

    df = build_dataframe(limit=2000)
    res = temporal_gene_isolated_split(df, holdout_days=365, min_holdout=500)
