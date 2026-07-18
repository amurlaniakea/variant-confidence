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

"""T13 guard test: AlphaMissense variant->score join (bug #4 class).

The join by (chrom,pos,ref,alt) is exactly where a silent index
misalignment could hide: if the key construction is wrong (e.g. isoform
ambiguity, column shift, 0/1-based pos), scores would align to the WRONG
variants with NOTHING failing. This test pins KNOWN (variant -> score)
pairs from the structural fixture and asserts the join reproduces them,
plus that an unmatched variant yields NaN (never a silent 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from variant_confidence.data.alphamissense import join_scores, load_alphamissense

FIXTURE = "tests/fixtures/alphamissense_sample.tsv"


@pytest.fixture
def am():
    return load_alphamissense(FIXTURE)


def test_load_columns(am):
    # score column "***" must be parsed to float, not dropped
    assert "am_score" in am.columns
    assert am["am_score"].dtype == float
    assert len(am) == 6


def test_join_known_variants(am):
    """Pinned (variant -> score) pairs from the fixture."""
    variants = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2", "chr1"],
        "pos": [69094, 69103, 100000, 69104],
        "ref": ["G", "T", "A", "T"],
        "alt": ["T", "C", "G", "G"],
    })
    scores = join_scores(variants, am, on="position")
    # exact known scores (no rounding, no misalignment)
    assert scores[0] == pytest.approx(0.2937, abs=1e-6)
    assert scores[1] == pytest.approx(0.9110, abs=1e-6)
    assert scores[2] == pytest.approx(0.5123, abs=1e-6)
    assert scores[3] == pytest.approx(0.6917, abs=1e-6)
    # all matched (no NaN) for these known rows
    assert not np.isnan(scores).any()


def test_join_case_insensitive_and_no_silent_zero():
    """REF/ALT mixed case must still match; unmatched gives NaN, not 0."""
    am = load_alphamissense(FIXTURE)
    variants = pd.DataFrame({
        "chrom": ["CHR1", "chrX"],  # chrX not in fixture
        "pos": [69094, 999999],
        "ref": ["g", "A"],           # lowercase ref
        "alt": ["t", "T"],
    })
    scores = join_scores(variants, am, on="position")
    assert scores[0] == pytest.approx(0.2937, abs=1e-6)  # matched despite case
    assert np.isnan(scores[1])  # unmatched -> NaN, NOT 0 (anti-bug #5)
    print(f"\n[join] matched={not np.isnan(scores[0])} unmatched_nan={np.isnan(scores[1])}")


def test_join_protein_key(am):
    variants = pd.DataFrame({
        "uniprot_id": ["Q8NH21", "P12345"],
        "protein_variant": ["V2L", "L11S"],
    })
    scores = join_scores(variants, am, on="protein")
    assert scores[0] == pytest.approx(0.2937, abs=1e-6)
    assert scores[1] == pytest.approx(0.0444, abs=1e-6)
