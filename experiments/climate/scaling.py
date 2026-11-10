"""Space-time reconstruction at scale: N = stations x time steps up to 10^6.

A fixed area-weighted station network observes the field every 6 h over a window starting at
``--start``; the window length sets N. Inputs are ``(x, y, z, orography, land, t)`` with ``t``
in days. The test set is a fixed random sample of (held-out cell, time) pairs in the window,
scored with area weights. The exact GP is run only up to ``--exact-max`` points (O(N^3) time,
O(N^2) memory). The random-feature models use D = clip(N, 16384, 32768) features.

    python experiments/climate/scaling.py --sizes 10000 100000 1000000 --out results/climate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import build_model, fit_predict, sphere_centres, weighted_scores

from srwf.data import era5


def window(target: str, start: str, n_steps: int, grid: era5.Grid) -> np.ndarray:
    s0 = era5.step_index(start)
    raw = era5.load_variable(target)[s0 : s0 + n_steps]
    return era5.transform(target, raw).reshape(n_steps, grid.n_cells)


def space_time_inputs(grid: era5.Grid, cells: np.ndarray, steps: np.ndarray) -> np.ndarray:
    days = steps * era5.STEP_HOURS / 24.0
    return np.column_stack([grid.inputs(cells, covariates=True), days])


def models_for(n: int, args: argparse.Namespace, centres: np.ndarray) -> dict[str, tuple[str, dict]]:
    d = int(np.clip(n, args.min_features, args.max_features))
    rwf = dict(num_features=d, levels=args.levels, centres=centres, subsample=args.subsample, steps=args.steps)
    batch = 4096
    epochs = max(2, math.ceil(args.svgp_steps * batch / n))
    out = {
        "rwf": ("rwf", rwf),
        "rwf-adaptive": ("rwf", {**rwf, "anchors": args.anchors}),
        "rff": ("rff", dict(num_features=d, subsample=args.subsample, steps=args.steps)),
        "svgp": ("svgp", dict(num_inducing=args.inducing, epochs=epochs, batch_size=batch, lr=0.01)),
    }
    if n <= args.exact_max:
        out["exact"] = ("exact", dict(kernel="matern2.5", steps=200, lr=0.05))
    return {k: v for k, v in out.items() if k in args.models}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="temperature")
    ap.add_argument("--start", default="2005-01-15T00")
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5000, 20000, 100000, 300000, 1000000])
    ap.add_argument("--models", nargs="+", default=["rwf", "rwf-adaptive", "rff", "svgp", "exact"])
    ap.add_argument("--min-features", type=int, default=16384)
    ap.add_argument("--max-features", type=int, default=32768, help="D = clip(N, min, max)")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--subsample", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--inducing", type=int, default=1024)
    ap.add_argument("--svgp-steps", type=int, default=6000)
    ap.add_argument("--exact-max", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="scaling")
    ap.add_argument("--out", type=Path, default=Path("results/climate"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = era5.load_grid()
    stations, held_out = era5.sample_stations(grid, args.fraction, args.seed)
    rows = []
    for n in sorted(args.sizes):
        n_steps = max(1, round(n / stations.size))
        values = window(args.target, args.start, n_steps, grid)
        steps = np.arange(n_steps)
        tr_cells, tr_steps = np.tile(stations, n_steps), np.repeat(steps, stations.size)
        rng = np.random.default_rng(args.seed + 1)
        te_cells = rng.choice(held_out, args.n_test, p=grid.area[held_out] / grid.area[held_out].sum())
        te_steps = rng.integers(0, n_steps, args.n_test)
        x_tr, y_tr = space_time_inputs(grid, tr_cells, tr_steps), values[tr_steps, tr_cells]
        x_te, y_te = space_time_inputs(grid, te_cells, te_steps), values[te_steps, te_cells]
        span = np.array([0.0, n_steps * era5.STEP_HOURS / 24.0])
        sphere = sphere_centres(20000, grid, covariates=True)
        centres = np.column_stack([sphere, np.random.default_rng(args.seed).uniform(*span, len(sphere))])
        for name, (kind, cfg) in models_for(len(y_tr), args, centres).items():
            try:
                pred, seconds = fit_predict(build_model(kind, cfg, args.seed), x_tr, y_tr, x_te)
            except RuntimeError as err:
                print(f"N={len(y_tr)} {name}: failed ({str(err).splitlines()[0]})", flush=True)
                continue
            s = weighted_scores(y_te, pred.mean, pred.std)
            rows.append(
                dict(
                    target=args.target,
                    n_train=len(y_tr),
                    n_steps=n_steps,
                    model=name,
                    num_features=cfg.get("num_features"),
                    seconds=seconds,
                    **s,
                )
            )
            scores = " ".join(f"{k} {s[k]:.3f}" for k in ("rmse", "crps", "nll", "cov95"))
            print(f"N={len(y_tr):8d} {name:13s} {scores} {seconds:.1f}s", flush=True)
            torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(args.out / f"{args.tag}_{args.target}.csv", index=False)
    (args.out / f"{args.tag}_{args.target}.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
