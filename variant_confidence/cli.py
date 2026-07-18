"""CLI entry point. MVP stub — la lógica de calibración/metrics/split se
implementa en T2-T11b. No debe fallar silenciosamente (AC4/AC7)."""


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="variant-confidence",
        description="Calibrated confidence layer for variant-effect pathogenicity scores.",
    )
    parser.add_argument("--version", action="version", version="variant-confidence 0.1.0")
    parser.add_argument(
        "--method",
        choices=["platt", "isotonic", "conformal"],
        default="platt",
        help="Calibration method (AC1). Selectable, not hardcoded.",
    )
    parser.add_argument(
        "--min-holdout",
        type=int,
        default=500,
        help="Minimum temporal-holdout size before emitting ECE (AC9, parametrizable).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal coverage target 1-alpha (AC1b/AC2).",
    )
    # Placeholder: real pipeline wired in T2-T11b.
    args = parser.parse_args(argv)
    print(
        "variant-confidence 0.1.0 (AGPL-3.0-or-later)\n"
        f"  method={args.method} min-holdout={args.min_holdout} alpha={args.alpha}\n"
        "  STATUS: scaffold only — pipeline not yet wired (see SDD T2-T11b)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
