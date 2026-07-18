# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.1] — 2026-07-18

Maintenance release: adds ESM-1v/EVE score integration (T14) without changing
the existing AlphaMissense calibration behaviour. Backwards compatible.

**Verified in clean clone (no network):**
- `ruff check .` → All checks passed!
- `pytest tests/` → 38 passed in ~10s

### Added
- **ESM-1v / EVE integration (T14):** `variant_confidence/data/esm_eve.py` loads
  user-converted (chrom,pos,ref,alt,score) TSVs for ESM-1v (MIT, Meta) and EVE
  (MIT, Pascal Notin). `join_scores` returns NaN (never 0) for unmatched, same
  pattern as AlphaMissense. `integrate.align_scores_esm_eve` wires it to the
  pipeline; weights are NEVER committed (Opción A). Licenses verified 2026-07-18
  (AC13c).
- **CLI wiring (T14g):** `variant-confidence --source
  {synthetic,alphamissense,esm1v,eve}` with `--score-path` selects the score
  source end-to-end. Reuses `align_scores` / `align_scores_esm_eve` (no
  reimplemented join); fail/degrade stays centralized in `run_calibration`
  (AC12); the report declares `source=` and a missing score is never imputed
  as 0 (NaN guard preserved). +8 tests (CLI wiring + integration layer).

### Fixed
- T13b commit message said "9 new tests" — the diff added **2** (suite 26 → 28);
  corrected in CHANGELOG (v0.1.0 entry).
- AC13b coverage figure (19.118 proteins) marked as NOT independently audited
  (measured by Hermes against the local AlphaMissense TSV; user audit pending).

### Documentation
- CHANGELOG.md, INVESTIGACION.md (research basis, ES), README Documentation
  section with primary references (AnnotateMissense arXiv:2605.24520,
  AlphaMissense Cheng et al. Science 2023).

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
- **ESM-1v / EVE join (T14):** `variant_confidence/data/esm_eve.py` loads
  user-converted (chrom,pos,ref,alt,score) TSVs for ESM-1v (MIT, Meta) and EVE
  (MIT, Pascal Notin); `join_scores` returns NaN (never 0) for unmatched, same
  pattern as AlphaMissense. `integrate.align_scores_esm_eve` wires it to the
  pipeline; weights are NEVER committed (Opción A). Licenses verified 2026-07-18
  (AC13c).
  **T14g (DONE):** CLI wiring — `variant-confidence --source {synthetic,alphamissense,esm1v,eve}`
  with `--score-path` selects the score source end-to-end. Reuses
  `align_scores` / `align_scores_esm_eve` (no reimplemented join); fail/degrade
  stays centralized in `run_calibration` (AC12); the report declares
  `source=` and a missing score is never imputed as 0 (NaN guard preserved).
  +8 tests (CLI wiring + integration layer), suite 38 passed, ruff clean.
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
[v0.1.1]: https://github.com/amurlaniakea/variant-confidence/compare/v0.1.0...v0.1.1
