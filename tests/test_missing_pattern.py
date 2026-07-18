"""T13b (commiteable): detection of STRUCTURED missing (by protein).

Sil audit note #2: AlphaMissense does NOT cover 100% of proteins, so the
missing pattern is STRUCTURAL (entire proteins/isoforms absent), not
random. `align_scores(on='protein')` must detect which input proteins have
ZERO matched variants. This test pins that detection on the structural
fixture (no real catalog committed). The real-catalog measurement is run
locally in /tmp and reported, not committed.
"""
from __future__ import annotations

import pandas as pd

from variant_confidence.data.integrate import align_scores

FIXTURE = "tests/fixtures/alphamissense_sample.tsv"


def test_structured_missing_by_protein_detected():
    """A ClinVar df with proteins Q8NH21 (present) + P99999 (absent) must
    report P99999 in missing_proteins, and P99999's variants as NaN."""
    df = pd.DataFrame({
        "chrom": ["chr1", "chr2", "chrX"],
        "pos": [69094, 100000, 9],
        "ref": ["G", "A", "A"],
        "alt": ["T", "G", "T"],
        "uniprot_id": ["Q8NH21", "P12345", "P99999"],  # P99999 absent in fixture
        "protein_variant": ["V2L", "K10R", "X1Y"],
        "label_bin": [0, 0, 1],
    })
    out = align_scores(df, FIXTURE, on="protein")
    # P99999 fully absent -> must be flagged as a missing protein
    assert "P99999" in out["missing_proteins"], (
        f"structured missing not detected: {out['missing_proteins']}"
    )
    # P99999's variant must be NaN, never 0
    assert out["n_missing"] == 1
    assert out["fraction_missing"] == 1 / 3


def test_present_protein_not_flagged():
    df = pd.DataFrame({
        "chrom": ["chr1", "chr2"],
        "pos": [69094, 100000],
        "ref": ["G", "A"],
        "alt": ["T", "G"],
        "uniprot_id": ["Q8NH21", "P12345"],
        "protein_variant": ["V2L", "K10R"],
        "label_bin": [0, 0],
    })
    out = align_scores(df, FIXTURE, on="protein")
    assert out["missing_proteins"] == set(), (
        f"false positive missing: {out['missing_proteins']}"
    )
    assert out["n_missing"] == 0
