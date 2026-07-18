# variant-confidence

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-blue.svg)](https://github.com/amurlaniakea/variant-confidence/releases/tag/v0.1.0)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG-blue.svg)](CHANGELOG.md)
[![Investigación](https://img.shields.io/badge/investigación-INVESTIGACION-blue.svg)](INVESTIGACION.md)

Calibrated confidence layer for protein **variant-effect pathogenicity**
predictions (AlphaMissense / ESM-1v / EVE). The project does **not** train a
new model — it adds a reusable, auditable **calibration layer** on top of
existing predictors, because the real gap is not accuracy but *knowing how
much to trust the score* (see AnnotateMissense 2026: MCC 0.94 CV → 0.76
temporal ClinVar; accuracy 0.8798).

## Features

- Dual, selectable calibration (AC1):
  - **Probability calibration**: Platt scaling / isotonic regression over a
    separate holdout.
  - **Conformal prediction**: coverage `1−α` intervals (split or Mondrian by
    gene).
- **ECE** (Expected Calibration Error) reported before/after calibration,
  with bootstrap CI and per-bin counts (AC2, AC9).
- **Temporal, leakage-free split** by ClinVar release date + gene isolation
  (same gene never in both train and test) — unit-tested in CI (AC3).
- **Robust to missing scores**: works with AlphaMissense or ESM-1v alone;
  emits an explicit warning instead of failing silently (AC4).
- **Non-misleading output**: every result includes interval/ECE + method +
  threshold, never a bare calibrated score (AC7).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Offline demo (synthetic scores, versioned fixture — no network)
variant-confidence --method platt --offline

# Real AlphaMissense scores (download once, see "AlphaMissense data" below)
variant-confidence --method conformal --on-missing degrade
```

The CLI reports: raw score, calibrated probability (or conformal interval),
method used, ECE before/after, and (conformal) empirical vs nominal coverage
— measured on the **temporal holdout**, never a bare score (AC7, AC10).

## AlphaMissense data (external download — NOT redistributed)

AlphaMissense prediction scores are **never committed** to this repository.
Download them locally under your own responsibility:

```bash
curl -L "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz" \
     -o AlphaMissense_hg38.tsv.gz
```

**License ambiguity — READ BEFORE USE.** Official primary sources disagree
(verified 2026-07-18):

- `google-deepmind/alphamissense` README states **CC BY 4.0**.
- The actual TSV header, Ensembl VEP plugin, and EBI page state
  **CC BY-NC-SA 4.0**.

This contradiction is not resolved here. Until clarified in writing
(contact `alphamissense@google.com`), **treat the data as restricted
(non-commercial)**. The software remains AGPL-3.0-or-later and 100%
self-contained; only the external score file is affected by this ambiguity.

## Data Licenses

The **software** is AGPL-3.0-or-later. Input **data** carry their own
licenses, which the end user is responsible for complying with:

- **AlphaMissense predictions**: license is **AMBIGUOUS** between official
  sources — README says CC BY 4.0, but the distributed TSV + Ensembl VEP +
  EBI say CC BY-NC-SA 4.0 (verified 2026-07-18). Not redistributed here;
  download externally per the section above. Do not assume commercial use
  is permitted until the contradiction is resolved by the rights holder.
- **ClinVar / dbNSFP / Ensembl VEP**: subject to NCBI / EMBL-EBI terms
  (free use with attribution).
- **gnomAD**: query + terms to be verified at implementation time (not
  assumed accessible by default).

This project does not redistribute data under a license different from its
own.

## Documentation

- **[SDD.md](SDD.md)** — full specification (Constitution, Spec, Plan, Tasks,
  Acceptance Criteria AC1–AC13b). The technical anchor of the project.
- **[INVESTIGACION.md](INVESTIGACION.md)** — research basis (ES): the calibration
  gap, the four silent-leakage bugs found in audit, the AlphaMissense license
  ambiguity, and the architectural findings.
- **[CHANGELOG.md](CHANGELOG.md)** — version history (v0.1.0 entry).

### References consulted

- **AnnotateMissense (2026)** — arXiv:2605.24520. Baseline without calibration:
  MCC 0.7613, accuracy 0.8798, F1 0.8750 on temporal ClinVar validation. The
  project's target is to reduce ECE, not improve this accuracy.
- **AlphaMissense** — Cheng et al., *Science* 2023 (adg7492);
  `google-deepmind/alphamissense`. Scores are CC BY 4.0 per README but
  CC BY-NC-SA 4.0 per the distributed TSV/VEP/EBI — contradiction unresolved
  (see AlphaMissense data section).
- **Data sources verified 2026-07-18 (real HTTP calls):** ClinVar E-utilities
  (200, no token), dbNSFP (200), Ensembl VEP REST (200), AlphaMissense repo
  (200); gnomAD GraphQL endpoint live but returned 400 — not assumed accessible.

## License

AGPL-3.0-or-later — Pedro Sordo Martínez (amurlaniakea@gmail.com), 2026.
See [LICENSE](LICENSE).
