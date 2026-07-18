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

"""T2: data layer — loaders for ClinVar, AlphaMissense, dbNSFP.

All network access goes through urllib with explicit User-Agent.
Data files are cached locally under data/cache (gitignored).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "variant-confidence/0.1 (amurlaniakea@gmail.com)"}


def _get_json(url: str, data: bytes | None = None, retries: int = 4) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - we retry on any network error
            last_err = e
            time.sleep(2 + attempt * 3)
    raise last_err  # type: ignore[misc]


def _norm_date(s: str) -> str:
    """ClinVar last_evaluated is 'YYYY/MM/DD' or '' -> 'YYYY-MM-DD' or ''."""
    if not s or s.startswith("1/01/01"):
        return ""
    parts = s.split(" ")[0].split("/")
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return ""


def _label_from_desc(desc: str) -> str | None:
    d = (desc or "").lower()
    if "pathogenic" in d and "benign" not in d:
        return "pathogenic"
    if "benign" in d and "pathogenic" not in d:
        return "benign"
    # "likely pathogenic" / "likely benign" handled above; discard the rest
    return None


def fetch_clinvar_missense(limit: int = 5000, cache: bool = True) -> list[dict]:
    """Fetch ClinVar missense variants with gene + last-evaluated date + label.

    Uses E-utilities (esearch + esummary POST). Returns list of dicts:
    {variant_id, gene, clinvar_date (YYYY-MM-DD), label (pathogenic/benign)}.
    Only clean pathogenic/benign labels are kept (Uncertain significance etc.
    dropped), mirroring the AnnotateMissense binary setup.
    """
    cache_path = CACHE_DIR / f"clinvar_missense_{limit}.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # esearch: just 'missense' — field-level clinical_significance filtering is
    # unreliable in E-utilities; we filter by description during parsing.
    # Paginate esearch in blocks of 10000 (retmax cap + NCBI rate limits);
    # large single requests return HTTP 502.
    ids: list[str] = []
    page = 0
    while len(ids) < limit:
        retstart = page * 10000
        search = (
            f"{base}/esearch.fcgi?db=clinvar&retstart={retstart}&retmax=10000"
            "&term=missense&retmode=json"
        )
        s = _get_json(search)
        batch = s.get("esearchresult", {}).get("idlist", [])
        if not batch:
            break
        ids.extend(batch)
        if len(batch) < 10000:
            break
        page += 1
        # polite delay to respect NCBI rate limits
        time.sleep(0.34)
    ids = ids[:limit]

    summary_url = f"{base}/esummary.fcgi?db=clinvar&retmode=json"
    out = []
    # Paginate esummary in chunks of 200 ids — a single POST with all ids is
    # rejected/truncated by E-utilities.
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        post_data = urllib.parse.urlencode({"id": ",".join(chunk)}).encode()
        sm = _get_json(summary_url, post_data)
        result = sm.get("result", {})
        for vid in chunk:
            v = result.get(vid)
            if not v:
                continue
            gene = v.get("gene_sort", "")
            date = _norm_date(v.get("germline_classification", {}).get("last_evaluated", ""))
            desc = v.get("germline_classification", {}).get("description", "")
            label = _label_from_desc(desc)
            if label is None or not gene:
                continue
            if not date:
                date = "1900-01-01"
            out.append({
                "variant_id": vid,
                "gene": gene,
                "clinvar_date": date,
                "label": label,
            })
        time.sleep(0.1)  # respect NCBI rate limits across 550 chunks
    if cache:
        cache_path.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    data = fetch_clinvar_missense(limit=2000)
    print(f"fetched {len(data)} labelled variants")
    if data:
        print("sample:", data[0])
        from collections import Counter
        print("by label:", Counter(d["label"] for d in data))
        print("unique genes:", len({d["gene"] for d in data}))
