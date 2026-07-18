# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""T14e guard tests for ESM-1v/EVE join — same rigor as AlphaMissense join.

Structural fixture only (tests/fixtures/esm_eve_sample.tsv). No real model
weights are committed (Opción A). We assert KNOWN variants map to KNOWN
scores, case-insensitivity works, and unmatched -> NaN (never 0).
"""
import os

import numpy as np
import pandas as pd

from variant_confidence.data import esm_eve

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "esm_eve_sample.tsv")


def _variants():
    return pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr3", "CHRM", "chr9"],
        "pos": [100, 200, 300, 400, 999],
        "ref": ["A", "G", "C", "A", "T"],
        "alt": ["T", "C", "T", "G", "A"],
    })


def test_load_and_known_pins():
    df = esm_eve.load_esm1v(FIXTURE)
    assert len(df) == 5
    assert (df["source"] == "esm1v").all()
    # known pins from the fixture
    v = _variants()
    s = esm_eve.join_scores(v, df, on_missing="degrade")
    # chr1:100:A:T -> 0.91 ; chr2:200:G:C -> 0.42 ; chr3:300:C:T -> 0.77
    assert np.isclose(s[0], 0.91)
    assert np.isclose(s[1], 0.42)
    assert np.isclose(s[2], 0.77)
    # chr9 has no entry -> NaN, not 0
    assert np.isnan(s[4])


def test_case_insensitivity():
    df = esm_eve.load_eve(FIXTURE)
    # variant CHRM:400:A:G (uppercase in variants) must match chrM:400:A:G
    v = _variants()
    s = esm_eve.join_scores(v, df, on_missing="degrade")
    assert np.isclose(s[3], 0.15)  # the mixed-case pin


def test_unmatched_is_nan_not_zero():
    df = esm_eve.load_esm1v(FIXTURE)
    v = _variants()
    s = esm_eve.join_scores(v, df, on_missing="degrade")
    # chr9:999:T:A has no entry in the fixture -> NaN, NOT 0
    assert np.isnan(s[4])
    assert not np.isnan(s[0])


def test_on_missing_fail_raises():
    df = esm_eve.load_esm1v(FIXTURE)
    v = _variants()
    try:
        esm_eve.join_scores(v, df, on_missing="fail")
        raise AssertionError("expected ValueError on unmatched variant")
    except ValueError:
        pass


def test_eve_source_tag():
    df = esm_eve.load_eve(FIXTURE)
    assert (df["source"] == "eve").all()
