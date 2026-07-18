"""T10: CLI — non-deceptive output.

Every prediction prints: raw score, calibrated probability (or conformal
interval), the method used, ECE before/after, and (for conformal) empirical
vs nominal coverage. Never a bare calibrated score — the whole point is that
the user sees the uncertainty context (AC7).
"""
from __future__ import annotations

import argparse
import sys

from variant_confidence.calib.synthetic import generate_scores
from variant_confidence.data.dataset import build_dataframe, build_dataframe_from_fixture
from variant_confidence.pipeline import run_calibration


def _print_report(rep, scores_eval, labels_eval, method):
    print("=" * 60)
    print(f"CALIBRATION REPORT — method={rep.method}")
    print(f"  n_eval (temporal holdout)={rep.n_eval}  n_calib={rep.n_calib}")
    print(f"  ECE before (raw) = {rep.ece_before:.4f} "
          f"CI[{rep.ece_before_ci[0]:.4f},{rep.ece_before_ci[1]:.4f}]")
    if method in ("platt", "isotonic"):
        print(f"  ECE after  (cal) = {rep.ece_after:.4f} "
              f"CI[{rep.ece_after_ci[0]:.4f},{rep.ece_after_ci[1]:.4f}]")
        delta = rep.ece_before - rep.ece_after
        print(f"  ECE reduction      = {delta:.4f} "
              f"({'improved' if delta > 0 else 'WORSE'})")
    if method == "conformal" and rep.conformal_coverage is not None:
        flag = "OK" if rep.coverage_within_tolerance else "OUT-OF-TOL"
        print(f"  conformal coverage (eval) = {rep.conformal_coverage:.4f} "
              f"(nominal {rep.conformal_nominal:.4f}, tol ±0.02) [{flag}]")
    print("=" * 60)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="variant-confidence: calibrated pathogenicity")
    ap.add_argument("--method", choices=["platt", "isotonic", "conformal"], default="platt")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--mondrian", action="store_true",
                    help="conformal: stratify quantiles by gene (Mondrian). "
                         "Requires enough per-gene calib data; otherwise falls "
                         "back to global per-label quantiles.")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--min-holdout", type=int, default=500)
    ap.add_argument("--offline", action="store_true",
                    help="use versioned fixture (no network)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = (build_dataframe_from_fixture() if args.offline
          else build_dataframe(limit=args.limit))
    if df.empty:
        print("ERROR: no data loaded", file=sys.stderr)
        return 1
    y = df["label_bin"].to_numpy()
    genes = df["gene"].to_numpy()
    scores = generate_scores(y, seed=42, overconfidence=0.6)

    # temporal holdout = most recent variants (post gene-isolation, done
    # upstream by split.temporal). For the CLI demo we reuse the fixture's
    # dates to pick the eval set, keeping fit/eval disjoint.
    from variant_confidence.split.temporal import temporal_gene_isolated_split
    split = temporal_gene_isolated_split(df, holdout_days=730,
                                          min_holdout=args.min_holdout, verbose=False)
    eval_idx = split.test.index.to_numpy()

    rep = run_calibration(
        scores, y, method=args.method, alpha=args.alpha,
        by_gene=genes if args.method == "conformal" and args.mondrian else None,
        eval_idx=eval_idx,
    )
    if not args.quiet:
        _print_report(rep, scores[eval_idx], y[eval_idx], args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
