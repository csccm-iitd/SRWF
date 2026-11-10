"""Regression of the irregular function on S^2 of Wyrwal, Krause and Borovitskiy (ICLR 2025).

Protocol of Section 4.1 of "Residual Deep Gaussian Processes on Manifolds" (arXiv:2411.00161):
``N`` training inputs on a Fibonacci lattice, noise variance 1e-4, a 5000-point lattice test set
with noisy targets, 5 seeds, test NLPD and MSE. RWF works on the R^3 embedding with wavelet
centres on a Fibonacci sphere and exact conditioning. Its settings are chosen per training set
on a random 20% validation split of that training set; the test set is read only once, by the
chosen model refitted on the whole training set.

    python experiments/sphere/irregular.py --out results/sphere
    python experiments/sphere/irregular.py --out results/sphere --n-train 100 1600 --seeds 2

Writes ``irregular_selection.csv`` (validation scores of every candidate), ``irregular_rwf.json``
(test scores of the chosen models) and, when ``--baselines`` points to a JSON of per-run
baseline scores, ``irregular_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from srwf import RWFGP
from srwf.data.sphere import regression_dataset
from srwf.geometry import fibonacci_sphere
from srwf.metrics import gaussian_nll

N_TRAIN = (100, 200, 400, 800, 1600)
CENTRE_POOL = 20000
BASE = dict(solver="dual", steps=300, lr=0.05)


def candidates() -> dict[str, dict]:
    out = {}
    for d in (4096, 16384):
        for levels in (2, 3, 4):
            for anchors in (0, 32, 64, 128):
                out[f"mh-D{d}-J{levels}-a{anchors}"] = dict(num_features=d, levels=levels, anchors=anchors)
    for levels in (2, 3):
        for anchors in (0, 64):
            out[f"morlet-D16384-J{levels}-a{anchors}"] = dict(
                num_features=16384, levels=levels, anchors=anchors, wavelet="morlet"
            )
    return out


def build(cfg: dict, seed: int, device: str) -> RWFGP:
    return RWFGP(centres=fibonacci_sphere(CENTRE_POOL), seed=seed, device=device, **BASE, **cfg)


def scores(model: RWFGP, x, y) -> dict[str, float]:
    p = model.predict(x)
    return {"nlpd": gaussian_nll(y, p.mean, p.std), "mse": float(np.mean((y - p.mean) ** 2))}


def split(n: int, seed: int, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(10_000 + seed).permutation(n)
    k = max(int(round(frac * n)), 1)
    return perm[k:], perm[:k]


def select(n_train: int, seed: int, cache: Path, device: str) -> str:
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["n_train", "seed", "candidate"])
    have = set(done.loc[(done.n_train == n_train) & (done.seed == seed), "candidate"])
    x, y, *_ = regression_dataset(n_train, seed)
    fit, val = split(n_train, seed)
    rows = []
    for name, cfg in candidates().items():
        if name in have:
            continue
        t0 = time.perf_counter()
        model = build(cfg, seed, device).fit(x[fit], y[fit])
        rows.append(
            {
                "n_train": n_train,
                "seed": seed,
                "candidate": name,
                **scores(model, x[val], y[val]),
                "seconds": time.perf_counter() - t0,
            }
        )
        print(n_train, seed, rows[-1], flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        done.to_csv(cache, index=False)
    sub = done[(done.n_train == n_train) & (done.seed == seed)]
    return str(sub.sort_values("nlpd").candidate.iloc[0])


def test(n_train: int, seed: int, name: str, device: str) -> dict:
    x, y, x_test, y_test, _ = regression_dataset(n_train, seed)
    model = build(candidates()[name], seed, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = scores(model, x_test, y_test)
    return {
        "model": "rwf",
        "n_train": n_train,
        "seed": seed,
        "candidate": name,
        **out,
        "fit_seconds": fit_seconds,
        "predict_seconds": time.perf_counter() - t1,
    }


def summarise(rwf: list[dict], baselines: list[dict]) -> dict:
    df = pd.DataFrame(rwf + baselines)
    g = df.groupby(["model", "n_train"])
    table = g.agg(
        nlpd_mean=("nlpd", "mean"),
        nlpd_std=("nlpd", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        fit_seconds=("fit_seconds", "mean"),
        runs=("seed", "count"),
    )
    return {"rows": table.reset_index().to_dict(orient="records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sphere")
    ap.add_argument("--n-train", type=int, nargs="+", default=list(N_TRAIN))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--baselines", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "irregular_rwf.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    have = {(r["n_train"], r["seed"]): r["candidate"] for r in runs}
    for n_train in args.n_train:
        for seed in range(args.seeds):
            name = select(n_train, seed, out / "irregular_selection.csv", args.device)
            if have.get((n_train, seed)) == name:
                continue
            runs = [r for r in runs if (r["n_train"], r["seed"]) != (n_train, seed)]
            runs.append(test(n_train, seed, name, args.device))
            print(runs[-1], flush=True)
            path.write_text(json.dumps(runs, indent=1))
    if args.baselines:
        base = json.loads(Path(args.baselines).read_text())
        (out / "irregular_summary.json").write_text(json.dumps(summarise(runs, base["runs"]), indent=1))


if __name__ == "__main__":
    main()
