# variant-confidence

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

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
variant-confidence --method platt --min-holdout 500 --alpha 0.1
```

(CLI is currently a scaffold; the pipeline is wired in T2–T11b per SDD.md.)

## Data Licenses

The **software** is AGPL-3.0-or-later. Input **data** carry their own
licenses, which the end user is responsible for complying with:

- **AlphaMissense predictions**: CC BY 4.0 (verified in
  `google-deepmind/alphamissense`, relicensed from CC BY-NC in March 2024).
  Attribution to DeepMind / Science 2023 (adg7492) required.
- **ClinVar / dbNSFP / Ensembl VEP**: subject to NCBI / EMBL-EBI terms
  (free use with attribution).
- **gnomAD**: query + terms to be verified at implementation time (not
  assumed accessible by default).

This project does not redistribute data under a license different from its
own.

## License

AGPL-3.0-or-later — Pedro Sordo Martínez (amurlaniakea@gmail.com), 2026.
See [LICENSE](LICENSE).
