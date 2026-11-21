#!/usr/bin/env python3
"""Reproducible k=2 prompt-grouped OOF check for the V8-TRACE arm vs the V7 control.

Deterministic by construction: fixed seeds, single-threaded BLAS, whole prompt blocks per fold.
    python run_oof.py --data data/passages_clean.csv.gz --out evidence/oof_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ntl_v8.data import classes_of, load_corpus  # noqa: E402
from ntl_v8.oof import run_gates  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/passages_clean.csv.gz")
    ap.add_argument("--out", default="evidence/oof_report.json")
    ap.add_argument("--reps", type=int, default=5, help="subsampled k=2 runs for the stability gate")
    ap.add_argument("--subsample", type=float, default=0.8, help="share of prompt groups per run")
    ap.add_argument("--n-splits", type=int, default=2, help="k of the prompt-grouped OOF folds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=400, help="bootstrap draws over prompt groups")
    ap.add_argument("--alpha", type=float, default=0.15, help="damping strength (calibration arm)")
    ap.add_argument("--no-variants", action="store_true", help="skip the unscaled-TRACE variant gate")
    ap.add_argument("--quick", action="store_true", help="small battery for smoke tests")
    args = ap.parse_args(argv)

    if args.quick:
        args.reps, args.boot = 1, 40

    t0 = time.time()
    df = load_corpus(args.data)
    classes = classes_of(df)
    report = {
        "bounty": "DAiL bnty_0716 - V8 arm bounty (TRACE rank-transition features)",
        "arm_under_test": "V8-TRACE (rank-transition features) vs V7-control",
        "data_file": args.data,
        "data_sha256": sha256(args.data),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    report.update(
        run_gates(
            df,
            classes,
            reps=args.reps,
            subsample=args.subsample,
            n_splits=args.n_splits,
            seed=args.seed,
            variants=not args.no_variants,
            alpha=args.alpha,
            n_boot=args.boot,
        )
    )
    report["runtime_seconds"] = round(time.time() - t0, 1)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out} ({os.path.getsize(out)} bytes) in {report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
