"""Accuracy and calibration versus training-set size on non-stationary 1-D functions.

python experiments/synthetic/sample_size.py --functions burst multi_step --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from srwf import RFFGP, RWFGP
from srwf.baselines import ExactGP
from srwf.data.synthetic import FUNCTIONS, sample
from srwf.metrics import evaluate, rmse

MODELS = {
    "exact": lambda s: ExactGP(kernel="rbf", steps=300),
    "exact-m12": lambda s: ExactGP(kernel="matern0.5", steps=300),
    "rff": lambda s: RFFGP(num_features=1024, steps=500, seed=s),
    "rwf": lambda s: RWFGP(num_features=1024, levels=9, steps=500, seed=s),
    "rwf-adaptive": lambda s: RWFGP(num_features=1024, levels=9, anchors=16, steps=500, seed=s),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", nargs="+", default=list(FUNCTIONS))
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    RWFGP(num_features=32, levels=2, steps=2).fit(np.zeros((4, 1)), np.arange(4.0))

    rows = []
    for fn in args.functions:
        f, _ = FUNCTIONS[fn]
        for n in args.sizes:
            for seed in args.seeds:
                xtr, ytr, xte, yte = sample(fn, n, seed=seed)
                for name in args.models:
                    m = MODELS[name](seed).fit(xtr, ytr)
                    p = m.predict(xte)
                    sec = m.info.seconds if hasattr(m, "info") else m.seconds
                    row = dict(function=fn, n=n, model=name, seed=seed, seconds=sec, rmse_f=rmse(f(xte[:, 0]), p.mean))
                    row.update(evaluate(yte, p.mean, p.std))
                    rows.append(row)
                    print(
                        json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True
                    )
                    if seed == args.seeds[0]:
                        np.savez(
                            args.out / f"fit_{fn}_{n}_{name}.npz",
                            x=xte[:, 0],
                            mean=p.mean,
                            std=p.std,
                            f_std=p.f_std,
                            x_train=xtr[:, 0],
                            y_train=ytr,
                        )
            (args.out / "sample_size.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
