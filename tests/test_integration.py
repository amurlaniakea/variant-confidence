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

"""T13 integration test (AGPL-clean): join + on_missing pipeline path.

Uses the structural fixture (no real AlphaMissense data committed). Verifies
the full bridge: align scores -> NaN for unmatched -> run_calibration
on_missing='degrade' excludes them and reports n_missing. This is the exact
path that will run on real data; here it is proven without committing the
71M-row NC-ambiguous catalog (Opción A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from variant_confidence.data.alphamissense import join_scores, load_alphamissense
from variant_confidence.data.integrate import align_scores
from variant_confidence.pipeline import run_calibration

FIXTURE = "tests/fixtures/alphamissense_sample.tsv"


@pytest.fixture
def clinvar_like():
    # 4 variants: 3 match the fixture, 1 (chrX:9:A>T) does not -> NaN
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chrX"],
        "pos": [69094, 69103, 100000, 9],
        "ref": ["G", "T", "A", "A"],
        "alt": ["T", "C", "G", "T"],
        "label_bin": [0, 1, 0, 1],
    })


def test_align_reports_missing(clinvar_like):
    out = align_scores(clinvar_like, FIXTURE)
    assert out["n_missing"] == 1  # chrX:9:A>T unmatched
    assert out["fraction_missing"] == pytest.approx(1 / 4)
    assert np.isnan(out["scores"][3])  # NaN, never 0


def test_pipeline_degrade_on_real_join(clinvar_like):
    # Build a larger df from the 6-variant fixture plus the 4 from
    # clinvar_like, so stratification has enough rows after 1 missing.
    base = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chr1", "chr2", "chr2"],
        "pos": [69094, 69103, 100000, 69104, 100000, 100001],
        "ref": ["G", "T", "A", "T", "A", "C"],
        "alt": ["T", "C", "G", "G", "G", "T"],
        "label_bin": [0, 1, 0, 1, 0, 0],
    })
    df = pd.concat([base, clinvar_like], ignore_index=True)
    am = load_alphamissense(FIXTURE)
    scores = join_scores(df, am, on="position")
    y = df["label_bin"].to_numpy()
    rep = run_calibration(scores, y, method="platt", on_missing="degrade")
    # clinvar_like contributes 1 missing (chrX:9); base all match
    assert rep.n_missing == 1
    assert rep.degraded is True
    assert rep.ece_after < rep.ece_before


def test_full_alignment_against_fixture():
    """All fixture variants present -> zero missing, ECE reducible."""
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chr1", "chr2", "chr2"],
        "pos": [69094, 69103, 100000, 69104, 100000, 100001],
        "ref": ["G", "T", "A", "T", "A", "C"],
        "alt": ["T", "C", "G", "G", "G", "T"],
        "label_bin": [0, 1, 0, 1, 0, 0],
    })
    out = align_scores(df, FIXTURE)
    assert out["n_missing"] == 0
    assert out["fraction_missing"] == 0.0
