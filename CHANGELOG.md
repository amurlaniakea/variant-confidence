# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] — 2026-07-18

First public release. A calibrated confidence layer for protein variant-effect
pathogenicity predictions (AlphaMissense / ESM-1v / EVE). It does **not** train a
new model — it adds an auditable calibration layer on top of existing predictors.

**Verified in clean clone (no network):**
- `ruff check .` → All checks passed!
- `pytest tests/` → 28 passed in 8.90s

### Added
- **Probability calibration (AC1):** Platt scaling and isotonic regression over a
  separate holdout. Method is selectable (`--method`), not hardcoded.
- **Conformal prediction (AC1b):** coverage `1−α` intervals, split or Mondrian by gene.
- **ECE metric (AC2, AC9):** equal-width + adaptive bins, bootstrap CI, low-count
  bins flagged as low-reliability.
- **Leakage-free split (AC3):** temporal split by ClinVar release date + gene
  isolation (same gene never in both train and test); index-alignment bug fixed.
- **Missing-data handling (AC4, T13b):** explicit missing-score handling and
  structured-missing detection; warning instead of silent failure.
  Note: T13b added **2 new tests** (suite went 26 → 28); the original commit
  message said "9 new tests" — that count was incorrect, corrected here.
- **Non-deceptive reporting (AC7):** CLI emits score + interval + method + ECE
  before/after; never a bare calibrated score.
- **AlphaMissense join (T13):** license-safe integration.
- **Degenerate-ECE bug fixed:** the synthetic generator now derives `true_p` from
  the real label with noise, so calibration is genuinely discriminative and AUC is
  preserved after calibration (no collapse to base rate).

### Known issues (honest)
- **AlphaMissense license ambiguity:** official README states CC BY 4.0, but the
  distributed TSV header, Ensembl VEP plugin, and EBI page state CC BY-NC-SA 4.0.
  Unresolved — treat the data as restricted (non-commercial) until clarified. The
  software remains AGPL-3.0-or-later and fully self-contained.
- The end-to-end join with real AlphaMissense scores is implemented but the
  flagship path is exercised by an offline fixture, not a live download in CI.

[v0.1.0]: https://github.com/amurlaniakea/variant-confidence/releases/tag/v0.1.0
