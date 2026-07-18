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

"""T14: ESM-1v and EVE score join (OPTION A — data stays external / AGPL-clean).

Licenses (verified 2026-07-18, primary sources):
  - ESM-1v: MIT (facebookresearch/esm, main). Copyright Meta Platforms, Inc.
    Código y pesos se distribuyen bajo la misma declaración MIT del repo
    (README + setup.py). Atribución requerida; citar Meier et al. 2021
    (doi:10.1101/2021.07.09.450648). NO asumir restricciones tipo "Meta
    Open Source Terms of Use" — no se pudo reproducir en fuente primaria.
  - EVE: MIT (OATML/EVE, master). Copyright (c) 2021 Pascal Notin.
    Atribución requerida; citar Notin et al. 2022 (arXiv:2110.04624).

Ambas fuentes son MIT, uso comercial permitido. Sin ambigüedad tipo
AlphaMissense (AC13).

IMPORTANTE — formato de entrada:
  ESM-1v y EVE NO emiten un TSV plano de "variante -> patogenicidad" listo
  para usar (como sí hace AlphaMissense con su columna '***'). Producen
  log-likelihood ratios por mutación que el USUARIO debe convertir a un
  archivo de variantes con score. Este módulo acepta ese archivo YA CONVERTIDO
  con columnas (chrom, pos, ref, alt, score) — NO asume ni reimplementa el
  pipeline de inferencia de ninguno de los dos modelos.

Al igual que AlphaMissense (Opción A): los pesos/scores NUNCA se commitean.
El usuario los descarga localmente bajo su responsabilidad. El repo queda
100% AGPL-3.0 limpio y los tests usan fixtures estructurales sintéticos.

Join key: (CHROM, POS, REF, ALT) hg38, case-insensitive. Unmatched -> NaN
(nunca 0). Igual que alphamissense.py, se afirma el join mapea variantes
CONOCIDAS a scores CONOCIDOS — nunca confiar en "corrió sin error".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columnas que el USUARIO debe producir al convertir la salida del modelo.
EXPECTED_COLUMNS = ["CHROM", "POS", "REF", "ALT", "score"]

# Origen de cada score, para trazabilidad en reportes.
SOURCE_ESM1V = "esm1v"
SOURCE_EVE = "eve"


def _load_converted(path: str, source: str) -> pd.DataFrame:
    """Load a USER-CONVERTED (chrom,pos,ref,alt,score) TSV(.gz).

    The real model weights/scores are NOT committed (MIT, external). The
    user converts model output to this flat schema. The file is read with
    a normal header (no leading '#' comment lines, unlike AlphaMissense).
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} converted file missing columns {missing}; "
            f"expected {EXPECTED_COLUMNS}"
        )
    out = pd.DataFrame({
        "chrom": df["CHROM"].astype(str),
        "pos": df["POS"].astype(int),
        "ref": df["REF"].astype(str),
        "alt": df["ALT"].astype(str),
        "score": df["score"].astype(float),
        "source": source,
    })
    return out


def load_esm1v(path: str) -> pd.DataFrame:
    """Load a user-converted ESM-1v score file (chrom,pos,ref,alt,score)."""
    return _load_converted(path, SOURCE_ESM1V)


def load_eve(path: str) -> pd.DataFrame:
    """Load a user-converted EVE score file (chrom,pos,ref,alt,score)."""
    return _load_converted(path, SOURCE_EVE)


def join_scores(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    on_missing: str = "fail",
) -> np.ndarray:
    """Return model scores aligned to `variants`, NaN where no match.

    Args:
        variants: DataFrame with at least chrom/pos/ref/alt (str/int).
        scores: output of load_esm1v() or load_eve().
        on_missing: "fail" (default) raises if ANY variant is unmatched;
            "degrade" returns NaN for unmatched and reports n_missing.

    Returns: float array len(len(variants)), np.nan where unmatched.
    """
    key_v = (
        variants["chrom"].astype(str).str.upper() + ":"
        + variants["pos"].astype(int).astype(str) + ":"
        + variants["ref"].astype(str).str.upper() + ":"
        + variants["alt"].astype(str).str.upper()
    )
    key_s = (
        scores["chrom"].str.upper() + ":"
        + scores["pos"].astype(int).astype(str) + ":"
        + scores["ref"].str.upper() + ":"
        + scores["alt"].str.upper()
    )
    score_map = pd.Series(scores["score"].to_numpy(), index=key_s.to_numpy())
    # .get returns NaN for unmatched keys — explicit, never a silent 0.
    aligned = key_v.map(score_map)
    n_missing = int(aligned.isna().sum())
    if on_missing == "fail" and n_missing > 0:
        raise ValueError(
            f"{n_missing} variants have no {scores['source'].iloc[0]} score; "
            f"use on_missing='degrade' to exclude them (never silently 0)"
        )
    return aligned.to_numpy(dtype=float)
