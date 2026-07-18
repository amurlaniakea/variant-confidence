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
    s = esm_eve.join_scores(v, df)
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
    s = esm_eve.join_scores(v, df)
    assert np.isclose(s[3], 0.15)  # the mixed-case pin


def test_unmatched_is_nan_not_zero():
    df = esm_eve.load_esm1v(FIXTURE)
    v = _variants()
    s = esm_eve.join_scores(v, df)
    # chr9:999:T:A has no entry in the fixture -> NaN, NOT 0
    assert np.isnan(s[4])
    assert not np.isnan(s[0])


def test_join_scores_always_nan_never_raises():
    # join_scores must NOT decide fail/degrade (that lives in align_scores /
    # run_calibration, AC12). It returns NaN for every unmatched variant
    # and never raises on partial match — consistent with alphamissense.
    df = esm_eve.load_esm1v(FIXTURE)
    v = _variants()
    s = esm_eve.join_scores(v, df)
    assert np.isnan(s[4])  # unmatched stays NaN
    assert np.isclose(s[0], 0.91)  # matched still correct


def test_empty_scores_returns_all_nan():
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt", "score", "source"])
    v = _variants()
    s = esm_eve.join_scores(v, empty)
    assert np.isnan(s).all()  # no IndexError on empty scores


def test_eve_source_tag():
    df = esm_eve.load_eve(FIXTURE)
    assert (df["source"] == "eve").all()


def test_align_scores_esm_eve_wiring():
    # The integration layer (integrate.align_scores_esm_eve) must connect
    # the join module to a ClinVar-style df, returning NaN (not 0) for
    # unmatched and delegating fail/degrade to run_calibration (AC12).
    from variant_confidence.data.integrate import align_scores_esm_eve

    df = _variants()
    res = align_scores_esm_eve(df, FIXTURE, source="esm1v")
    assert res["source"] == "esm1v"
    assert np.isclose(res["scores"][0], 0.91)
    assert np.isnan(res["scores"][4])  # unmatched -> NaN, never 0
    assert res["n_missing"] == 1
    assert res["fraction_missing"] == 1 / 5
