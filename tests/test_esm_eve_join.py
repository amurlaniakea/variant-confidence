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


def test_cli_wiring_esm1v_end_to_end(capsys, monkeypatch):
    # Full CLI flow with --source esm1v. We mock run_calibration (its own
    # behaviour is covered by test_pipeline / test_calibration_ece) so this
    # test isolates the CONTRACT T14g adds: source selection -> align_scores_esm_eve
    # -> scores with NaN -> `source` propagated to the report -> report prints
    # source=esm1v and never imputes a missing score as 0.
    #
    # The dataset loader's ClinVar rows lack chrom/pos/ref/alt, so we inject a df
    # WITH those columns (synthetic) so the join has keys — that exercises the
    # real wiring (source selection -> align -> report) without depending on the
    # temporal split / holdout being valid for this synthetic coord mapping.
    from types import SimpleNamespace

    import numpy as np

    from variant_confidence import cli
    from variant_confidence.data.dataset import build_dataframe_from_fixture

    real = build_dataframe_from_fixture().reset_index(drop=True)
    rng = np.random.default_rng(0)
    real["chrom"] = "chr" + rng.integers(1, 23, len(real)).astype(str)
    real["pos"] = rng.integers(1, 100000, len(real))
    real["ref"] = rng.choice(list("ACGT"), len(real))
    real["alt"] = rng.choice(list("ACGT"), len(real))
    monkeypatch.setattr(
        "variant_confidence.data.dataset.build_dataframe_from_fixture",
        lambda *a, **k: real,
    )
    monkeypatch.setattr(cli, "build_dataframe_from_fixture",
                        lambda *a, **k: real)

    captured = {}

    def fake_run_calibration(scores, labels, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["scores"] = np.asarray(scores, dtype=float)
        return SimpleNamespace(
            method=kwargs.get("method", "platt"),
            source=kwargs.get("source", "synthetic"),
            n_eval=1, n_calib=1, n_missing=int(np.isnan(scores).sum()),
            fraction_missing=0.0, degraded=bool(np.isnan(scores).any()),
            ece_before=0.1, ece_after=0.05,
            ece_before_ci=(0.0, 0.0), ece_after_ci=(0.0, 0.0),
            conformal_coverage=None, conformal_nominal=None,
            coverage_within_tolerance=None, mondrian_fallback_rate=None,
        )

    monkeypatch.setattr(cli, "run_calibration", fake_run_calibration)

    rc = cli.main([
        "--source", "esm1v",
        "--score-path", FIXTURE,
        "--on-missing", "degrade",
        "--offline",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # T14g contract: the report declares which source produced the scores.
    assert "source=esm1v" in out, f"report did not declare source=esm1v:\n{out}"
    assert captured["source"] == "esm1v"
    # NaN guard preserved: a missing score reaches run_calibration as NaN,
    # never silently converted to 0 by the wiring.
    assert np.isnan(captured["scores"]).any()
    assert not np.any(captured["scores"] == 0)


def test_cli_wiring_eve_source_tag(capsys, monkeypatch):
    # Same contract for --source eve: report must declare source=eve.
    from types import SimpleNamespace

    import numpy as np

    from variant_confidence import cli
    from variant_confidence.data.dataset import build_dataframe_from_fixture

    real = build_dataframe_from_fixture().reset_index(drop=True)
    rng = np.random.default_rng(1)
    real["chrom"] = "chr" + rng.integers(1, 23, len(real)).astype(str)
    real["pos"] = rng.integers(1, 100000, len(real))
    real["ref"] = rng.choice(list("ACGT"), len(real))
    real["alt"] = rng.choice(list("ACGT"), len(real))
    monkeypatch.setattr(
        "variant_confidence.data.dataset.build_dataframe_from_fixture",
        lambda *a, **k: real,
    )
    monkeypatch.setattr(cli, "build_dataframe_from_fixture",
                        lambda *a, **k: real)
    captured = {}

    def fake_run_calibration(scores, labels, **kwargs):
        captured["source"] = kwargs.get("source")
        return SimpleNamespace(
            method="platt", source=kwargs.get("source", "synthetic"),
            n_eval=1, n_calib=1, n_missing=0, fraction_missing=0.0,
            degraded=False, ece_before=0.1, ece_after=0.05,
            ece_before_ci=(0.0, 0.0), ece_after_ci=(0.0, 0.0),
            conformal_coverage=None, conformal_nominal=None,
            coverage_within_tolerance=None, mondrian_fallback_rate=None,
        )

    monkeypatch.setattr(cli, "run_calibration", fake_run_calibration)
    rc = cli.main([
        "--source", "eve",
        "--score-path", FIXTURE,
        "--on-missing", "degrade",
        "--offline",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "source=eve" in out, f"report did not declare source=eve:\n{out}"
    assert captured["source"] == "eve"


def test_cli_requires_score_path_for_real_source():
    # Selecting a real source without --score-path must fail fast, not
    # silently fall back to synthetic scores.
    from variant_confidence.cli import main

    rc = main(["--source", "esm1v", "--offline", "--quiet"])
    assert rc == 1  # explicit error, no silent fallback
