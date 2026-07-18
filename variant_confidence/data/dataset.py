"""T3: dataset schema.

Wraps the raw ClinVar records into a DataFrame with columns:
variant_id, gene, clinvar_date (YYYY-MM-DD), label (0=benign, 1=pathogenic).
"""
from __future__ import annotations

import pandas as pd

from .loader import fetch_clinvar_missense

LABEL_MAP = {"benign": 0, "pathogenic": 1}


def build_dataframe(limit: int = 5000, cache: bool = True) -> pd.DataFrame:
    records = fetch_clinvar_missense(limit=limit, cache=cache)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["label_bin"] = df["label"].map(LABEL_MAP)
    df["clinvar_date"] = pd.to_datetime(df["clinvar_date"], errors="coerce")
    return df[["variant_id", "gene", "clinvar_date", "label", "label_bin"]]


if __name__ == "__main__":
    df = build_dataframe(limit=2000)
    print(df.shape)
    print(df.head())
