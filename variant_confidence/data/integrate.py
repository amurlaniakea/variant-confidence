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

"""T13 integration: align AlphaMissense scores to a ClinVar-style df.

Bridges the real AlphaMissense TSV (loaded locally, never committed) to the
variant_confidence pipeline. The join output feeds run_calibration's
on_missing handling (T12) — so a variant with no AlphaMissense prediction
becomes NaN, never 0.

Reports the MISSING pattern (Sil audit note #2): is it random, or
structured (entire proteins/isoforms absent)? That affects whether
on_missing='degrade' stays unbiased on the temporal holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from variant_confidence.data.alphamissense import join_scores, load_alphamissense


def align_scores(df: pd.DataFrame, am_path: str, on: str = "position") -> dict:
    """Return {scores, n_missing, missing_by_protein} for `df` vs AlphaMissense.

    Args:
        df: ClinVar-style DataFrame with chrom/pos/ref/alt (+ optional
            uniprot_id/protein_variant for on='protein').
        am_path: local path to AlphaMissense TSV(.gz) (user-downloaded).
        on: "position" or "protein" join key.

    Returns dict with:
        scores: float array aligned to df (np.nan where unmatched)
        n_missing: count of unmatched variants
        fraction_missing: n_missing / len(df)
        missing_proteins: set of uniprot_ids whose variants are fully absent
            (structured missing, if any)
    """
    am = load_alphamissense(am_path)
    scores = join_scores(df, am, on=on)

    n_missing = int(np.isnan(scores).sum())
    fraction_missing = n_missing / len(scores) if len(scores) else 0.0

    missing_proteins: set[str] = set()
    if on == "protein" and "uniprot_id" in df.columns:
        matched = ~np.isnan(scores)
        # proteins with ZERO matched variants among those present in df
        present_proteins = set(df["uniprot_id"].astype(str))
        matched_proteins = set(df.loc[matched, "uniprot_id"].astype(str))
        missing_proteins = present_proteins - matched_proteins

    return {
        "scores": scores,
        "n_missing": n_missing,
        "fraction_missing": fraction_missing,
        "missing_proteins": missing_proteins,
    }
