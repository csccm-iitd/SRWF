"""Thompson-sampling BO benchmarks: RWF-TS vs RFF-TS, exact-GP TS / EI and random search.

    python experiments/bo/run.py --problems all --methods all --seeds 20 --workers 14

Writes one JSON per (problem, method, seed) to ``results/bo/runs/`` and a summary CSV of the
simple regret at the budget to ``results/bo/summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functions import PROBLEMS  # noqa: E402
from methods import METHODS, run_method  # noqa: E402


def run_one(problem: str, method: str, seed: int, out: str, options: dict) -> str:
    import torch

    torch.set_num_threads(1)
    path = Path(out) / "runs" / f"{problem}__{method}__{seed}.json"
    if path.exists():
        return str(path)
    p = PROBLEMS[problem]
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_init = 2 * p.dim + 1
    budget = options.pop("budget") or p.budget
    trace = run_method(method, p, p.bounds, budget, n_init, seed, **options)
    regret = np.maximum(p.optimum - trace.best, 0.0)
    record = {
        "problem": problem,
        "method": method,
        "seed": seed,
        "n_init": n_init,
        "y": trace.y.tolist(),
        "x": trace.x.tolist(),
        "regret": regret.tolist(),
        "fit_seconds": trace.fit_seconds,
        "acq_seconds": trace.acq_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return str(path)


def summarise(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((out / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append(
            {
                "problem": r["problem"],
                "method": r["method"],
                "seed": r["seed"],
                "final_regret": r["regret"][-1],
                "log10_regret": np.log10(max(r["regret"][-1], 1e-8)),
                "sec_per_iter": float(np.mean(np.add(r["fit_seconds"], r["acq_seconds"]))) if r["fit_seconds"] else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    g = df.groupby(["problem", "method"])
    table = g.agg(
        n=("seed", "count"),
        regret_mean=("final_regret", "mean"),
        regret_se=("final_regret", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan),
        log10_regret_mean=("log10_regret", "mean"),
        log10_regret_se=("log10_regret", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan),
        sec_per_iter=("sec_per_iter", "mean"),
    ).reset_index()
    table.to_csv(out / "summary.csv", index=False)
    curves = {}
    for (prob, meth), _ in g:
        runs = [json.loads(f.read_text()) for f in sorted((out / "runs").glob(f"{prob}__{meth}__*.json"))]
        curves[f"{prob}__{meth}"] = np.array([r["regret"] for r in runs])
    np.savez_compressed(out / "regret_curves.npz", **curves)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="all", help="comma-separated names or 'all': " + ",".join(PROBLEMS))
    ap.add_argument("--methods", default="all", help="comma-separated names or 'all': " + ",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--num-features", type=int, default=1024, help="total features of the RWF and RFF surrogates")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--data-fraction", type=float, default=0.5, help="share of RWF centres drawn from evaluated inputs")
    ap.add_argument("--lowpass", type=int, default=32)
    ap.add_argument("--fourier", type=int, default=0, help="random Fourier features inside the RWF feature budget")
    ap.add_argument("--anchors", type=int, default=0)
    ap.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm-start RWF/RFF hyperparameters across iterations",
    )
    ap.add_argument(
        "--weight-prior", type=float, default=1.0, help="std of the RWF octave log-weight prior; 0 for ML-II"
    )
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--budget", type=int, default=None, help="override the per-problem evaluation budget")
    ap.add_argument("--device", default="cpu", help="device of the TS surrogates (EI always runs on CPU)")
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    ap.add_argument("--out", default="results/bo")
    args = ap.parse_args()

    problems = list(PROBLEMS) if args.problems == "all" else args.problems.split(",")
    methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
    options = {
        "num_features": args.num_features,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "budget": args.budget,
        "device": args.device,
        "warm_start": args.warm_start,
    }
    rwf = {
        "levels": args.levels,
        "data_fraction": args.data_fraction,
        "lowpass": args.lowpass,
        "fourier": args.fourier,
        "anchors": args.anchors,
        "weight_prior": args.weight_prior,
    }
    jobs = [
        (p, m, s, args.out, {**options, **(rwf if m == "rwf_ts" else {})})
        for s in range(args.first_seed, args.first_seed + args.seeds)
        for p in problems
        for m in methods
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
    table = summarise(Path(args.out))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
