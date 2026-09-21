"""numpy-poisson-variance-guard CLI: probe the installed numpy for the
large-lam Poisson variance-inflation bug (numpy/numpy#31986) and provide
a corrected drop-in sampler.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from . import __version__
from .core import diagnose, safe_poisson
from .style import print_fields, resolve_style, status_headline


def cmd_diagnose(args: argparse.Namespace) -> int:
    result = diagnose(lam=args.lam, n_samples=args.samples, tolerance=args.tolerance)
    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy Poisson large-lam variance probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("lam", f"{result.lam:.0e}"),
            ("samples", str(result.n_samples)),
            ("numpy var/lam", f"{result.numpy_var_over_lam:.4f}"),
            ("safe_poisson var/lam", f"{result.safe_var_over_lam:.4f}"),
            ("numpy mean", f"{result.numpy_mean:.6e}"),
            ("safe_poisson mean", f"{result.safe_mean:.6e}"),
            ("affected", "yes" if result.affected else "no"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def cmd_sample(args: argparse.Namespace) -> int:
    import numpy as np

    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    samples = safe_poisson(args.lam, args.size, rng=rng)
    if args.json:
        print(json.dumps({"lam": args.lam, "size": args.size, "samples": samples.tolist()}))
        return 0

    style = resolve_style(args.no_color)
    print(status_headline(style, "info", "safe_poisson sample summary"))
    print_fields(
        [
            ("lam", f"{args.lam:.0e}"),
            ("size", str(args.size)),
            ("empirical mean", f"{samples.mean():.6e}"),
            ("empirical var/lam", f"{samples.astype(float).var() / args.lam:.4f}"),
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpy-poisson-variance-guard",
        description=(
            "Detect and work around numpy/numpy#31986: "
            "Generator.poisson(lam) silently inflates variance for large lam "
            "(roughly lam >= 1e15) due to float64 log-pmf cancellation in its "
            "rejection sampler's acceptance test."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"numpy-poisson-variance-guard {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="empirically probe the installed numpy for the bug")
    diag.add_argument("--lam", type=float, default=1e16)
    diag.add_argument("--samples", type=int, default=200_000)
    diag.add_argument("--tolerance", type=float, default=0.05)
    diag.add_argument("--json", action="store_true")
    diag.add_argument("--no-color", action="store_true")
    diag.set_defaults(func=cmd_diagnose)

    sample = sub.add_parser("sample", help="draw corrected Poisson(lam) samples via safe_poisson")
    sample.add_argument("--lam", type=float, required=True)
    sample.add_argument("--size", type=int, required=True)
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--json", action="store_true")
    sample.add_argument("--no-color", action="store_true")
    sample.set_defaults(func=cmd_sample)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
