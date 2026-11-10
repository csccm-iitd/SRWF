"""Surrogate diagnostics on BO-shaped data: predictive quality of RWF variants, RFF and an exact GP.

Training sets are the first ``n`` evaluations of recorded BO runs (clustered, as during BO);
test sets are uniform over the box (global fit) and Gaussian perturbations of the optimum
(local fit, which decides whether Thompson samples can refine the incumbent).

    python experiments/bo/surrogate_quality.py --runs results/bo/runs --source gp_ts --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functions import PROBLEMS  # noqa: E402
from methods import _fit_gp, _Unit, rwf_sampler  # noqa: E402

from srwf import RFFGP  # noqa: E402
from srwf.bo import sobol  # noqa: E402

warnings.filterwarnings("ignore")

VARIANTS: dict[str, dict] = {
    "rwf": {},
    "rwf_ml2": {"weight_prior": 0.0},
    "rwf_steps300": {"steps": 300},
    "rwf_L6": {"levels": 6},
    "rwf_D4096": {"num_features": 4096},
    "rwf_data_centres": {"data_fraction": 1.0},
    "rwf_box_centres": {"data_fraction": 0.0},
    "rwf_lowpass128": {"lowpass": 128},
    "rwf_fourier256": {"fourier": 256},
    "rwf_anchors8": {"anchors": 8},
    "rwf_fourier256_steps300": {"fourier": 256, "steps": 300},
}
BASE = {"num_features": 1024, "levels": 4, "steps": 100}


def argmax_location(problem) -> np.ndarray:
    b = problem.bounds
    starts = sobol(256, b, 0)
    best = starts[np.argmax(problem(starts))]
    res = minimize(lambda z: -problem(z[None])[0], best, method="L-BFGS-B", bounds=list(map(tuple, b)))
    return res.x


def scores(mean: np.ndarray, std: np.ndarray, y: np.ndarray, scale: float) -> dict:
    z = (y - mean) / std
    return {
        "nrmse": float(np.sqrt(np.mean((y - mean) ** 2)) / scale),
        "nlpd": float(np.mean(0.5 * z**2 + np.log(std / scale) + 0.5 * np.log(2 * np.pi))),
        "cover95": float(np.mean(np.abs(z) < 1.96)),
    }


def evaluate(problem_name: str, run_file: str, n: int, model: str, seed: int) -> list[dict]:
    torch.set_num_threads(1)
    p = PROBLEMS[problem_name]
    r = json.loads(Path(run_file).read_text())
    x, y = np.asarray(r["x"])[:n], np.asarray(r["y"])[:n]
    rng = np.random.default_rng(seed)
    span = p.bounds[:, 1] - p.bounds[:, 0]
    x_glob = rng.uniform(p.bounds[:, 0], p.bounds[:, 1], size=(2000, p.dim))
    x_loc = np.clip(
        argmax_location(p) + 0.03 * span * rng.standard_normal((500, p.dim)), p.bounds[:, 0], p.bounds[:, 1]
    )
    scale = float(np.std(p(x_glob)))
    preds = {}
    if model == "exact_gp":
        u = _Unit(p.bounds, x, y)
        gp = _fit_gp(u)
        for name, xt in (("global", x_glob), ("local", x_loc)):
            with torch.no_grad():
                post = gp.posterior((torch.as_tensor(xt) - u.lo) / u.span)
            mu = post.mean.squeeze(-1).numpy() * u.sd + u.mu
            sd = np.sqrt(post.variance.squeeze(-1).numpy()) * u.sd
            preds[name] = (mu, sd)
    else:
        if model == "rff":
            m = RFFGP(1024, steps=100, device="cpu", seed=seed).fit(x, y)
        else:
            m = rwf_sampler(p.bounds, seed=seed, **{**BASE, **VARIANTS[model]}).make(x).fit(x, y)
        for name, xt in (("global", x_glob), ("local", x_loc)):
            pr = m.predict(xt, return_f_std=True)
            preds[name] = (pr.mean, np.maximum(pr.std, 1e-9))
    rows = []
    for name, xt in (("global", x_glob), ("local", x_loc)):
        mu, sd = preds[name]
        rows.append(
            {"problem": problem_name, "model": model, "seed": seed, "test": name, **scores(mu, sd, p(xt), scale)}
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/bo/runs")
    ap.add_argument("--source", default="gp_ts", help="method whose BO traces provide the training inputs")
    ap.add_argument("--problems", default="gramacy_lee,bumped_branin,branin,hartmann6,levy10")
    ap.add_argument(
        "--n", type=int, default=40, help="number of evaluations used for training (after the initial design)"
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", default="exact_gp,rff," + ",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/bo/surrogate_quality.csv")
    args = ap.parse_args()

    jobs = []
    for prob in args.problems.split(","):
        n = args.n + 2 * PROBLEMS[prob].dim + 1
        for s in range(args.seeds):
            f = Path(args.runs) / f"{prob}__{args.source}__{s}.json"
            for model in args.models.split(","):
                jobs.append((prob, str(f), n, model, s))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for res in pool.map(evaluate, *zip(*jobs, strict=True)) for row in res]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    table = df.groupby(["problem", "test", "model"])[["nrmse", "nlpd", "cover95"]].mean().unstack("test")
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(table.to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
