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

"""T3: dataset schema.

Wraps the raw ClinVar records into a DataFrame with columns:
variant_id, gene, clinvar_date (YYYY-MM-DD), label (0=benign, 1=pathogenic).

Two load paths:
  - build_dataframe(limit=...)  -> live ClinVar fetch (real use).
  - build_dataframe_from_fixture(path) -> OFFLINE, deterministic (tests/CI).
    Never hits the network; the fixture is a versioned snapshot in
    tests/fixtures/ so the split/calibration tests are reproducible and
    do not depend on network or cache state.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .loader import fetch_clinvar_missense

LABEL_MAP = {"benign": 0, "pathogenic": 1}

FIXTURE_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent
    / "tests" / "fixtures" / "clinvar_sample.json"
)


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["label_bin"] = df["label"].map(LABEL_MAP)
    df["clinvar_date"] = pd.to_datetime(df["clinvar_date"], errors="coerce")
    return df[["variant_id", "gene", "clinvar_date", "label", "label_bin"]]


def build_dataframe(limit: int = 5000, cache: bool = True) -> pd.DataFrame:
    """Live ClinVar fetch (real usage)."""
    records = fetch_clinvar_missense(limit=limit, cache=cache)
    return _records_to_df(records)


def build_dataframe_from_fixture(path: str | Path = FIXTURE_DEFAULT) -> pd.DataFrame:
    """Offline, deterministic load from a versioned fixture (tests/CI)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} missing. Tests must be reproducible offline; "
            f"regenerate it from live ClinVar if intentionally updated."
        )
    with path.open() as f:
        records = json.load(f)
    return _records_to_df(records)


if __name__ == "__main__":
    df = build_dataframe(limit=2000)
    print(df.shape)
    print(df.head())
