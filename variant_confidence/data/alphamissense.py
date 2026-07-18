"""T13: AlphaMissense score join (OPTION A — data stays external / AGPL-clean).

AlphaMissense license is AMBIGUOUS between official sources:
  - google-deepmind/alphamissense README: "CC BY 4.0"
  - The actual TSV header + Ensembl VEP + EBI: "CC BY-NC-SA 4.0"
Two primary sources contradict each other (verified 2026-07-18). Per
project policy (Opción A), the SCORES ARE NEVER COMMITTED. This module
loads a local AlphaMissense TSV (downloaded by the USER under their own
responsibility) and joins variant -> score. Tests use a small structural
fixture (not the real catalog), so the repo stays 100% AGPL-3.0 clean.

Join key: AlphaMissense identifies a variant by (uniprot_id, protein_variant)
e.g. "Q8NH21" + "V2L". ClinVar identifies by (CHROM, POS, REF, ALT) hg38.
To align, we need a transcript/protein map. For the guard test we join on
the columns AlphaMissense actually emits (CHROM, POS, REF, ALT) which are
present in the TSV — a direct positional join, the exact spot where a
silent index misalignment (bug #4 class) would hide. We therefore assert
the join maps KNOWN variants to KNOWN scores, never trust "ran without
error".
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
    "transcript_id", "protein_variant", "***", "am_class",
]


def load_alphamissense(path: str) -> pd.DataFrame:
    """Load an AlphaMissense TSV(.gz) into a DataFrame.

    The real file is ~71M rows / 613MB and is NOT committed (CC BY-NC-SA
    4.0 ambiguity). The user downloads it locally. The file has 3 pure
    comment lines, then a header line that ALSO begins with '#'
    (e.g. "#CHROM\\tPOS..."), so a naive comment='#' would eat the header.
    We skip all leading '#' lines and treat the next line as the header.
    """
    # The file has 3 pure comment lines, then a header line that ALSO begins
    # with '#' (e.g. "#CHROM\tPOS..."). We skip the pure-comment lines and
    # treat the first '#'-line that contains a TAB as the header (stripping
    # the leading '#'). A naive comment='#' would eat that header.
    opener = gzip.open if path.endswith(".gz") else open
    n_skip = 0
    header_line = None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if "\t" in line and header_line is None:
                    header_line = line.lstrip("#").rstrip("\n")
                    break
                n_skip += 1
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=n_skip, low_memory=False)
    if header_line is not None:
        df.columns = [c.lstrip("#") for c in header_line.split("\t")]
    # The score column is literally named "***" in the source file.
    score_col = "***" if "***" in df.columns else df.columns[-2]
    out = pd.DataFrame({
        "chrom": df["CHROM"].astype(str),
        "pos": df["POS"].astype(int),
        "ref": df["REF"].astype(str),
        "alt": df["ALT"].astype(str),
        "uniprot_id": df["uniprot_id"].astype(str),
        "protein_variant": df["protein_variant"].astype(str),
        "am_score": df[score_col].astype(float),
        "am_class": df["am_class"].astype(str),
    })
    return out


def join_scores(
    variants: pd.DataFrame,
    am: pd.DataFrame,
    on: str = "position",
) -> np.ndarray:
    """Return AlphaMissense scores aligned to `variants`, NaN where no match.

    Args:
        variants: DataFrame with at least chrom/pos/ref/alt (str/int).
        am: output of load_alphamissense().
        on: "position" joins on (chrom,pos,ref,alt); "protein" joins on
            (uniprot_id, protein_variant). Position is the guard-test path.

    Returns: float array len(len(variants)), np.nan where unmatched.
    """
    if on == "position":
        key_v = (
            variants["chrom"].astype(str).str.upper() + ":"
            + variants["pos"].astype(int).astype(str) + ":"
            + variants["ref"].astype(str).str.upper() + ":"
            + variants["alt"].astype(str).str.upper()
        )
        key_a = (
            am["chrom"].str.upper() + ":" + am["pos"].astype(str) + ":"
            + am["ref"].str.upper() + ":" + am["alt"].str.upper()
        )
    elif on == "protein":
        key_v = (
            variants["uniprot_id"].astype(str) + "|"
            + variants["protein_variant"].astype(str)
        )
        key_a = am["uniprot_id"] + "|" + am["protein_variant"]
    else:
        raise ValueError(f"on must be 'position' or 'protein', got {on!r}")

    am_map = pd.Series(am["am_score"].to_numpy(), index=key_a.to_numpy())
    # .get returns NaN for unmatched keys — explicit, never a silent 0
    scores = key_v.map(am_map).to_numpy(dtype=float)
    return scores
