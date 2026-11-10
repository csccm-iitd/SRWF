"""Wall-clock of one batch Thompson-sampling iteration as the number of observations grows.

One iteration = refit hyperparameters, draw ``q`` posterior functions, maximise each draw
(Sobol + local candidates and gradient refinement, identical for all surrogates).

    python experiments/bo/iteration_cost.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functions import PROBLEMS  # noqa: E402

from srwf import RFFGP, RWFGP  # noqa: E402
from srwf.bo import maximise  # noqa: E402
from srwf.sampling import PosteriorFunction  # noqa: E402

warnings.filterwarnings("ignore")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def exact_gp_draws(x: np.ndarray, y: np.ndarray, bounds: np.ndarray, q: int, device: torch.device):
    """Exact GP (BoTorch defaults) with ``q`` pathwise draws, drawn one at a time for ``n >= 10^4``
    where the batched Matheron update fails on CUDA."""
    import gpytorch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.sampling.pathwise import draw_matheron_paths
    from gpytorch.mlls import ExactMarginalLogLikelihood

    lo = torch.as_tensor(bounds[:, 0], dtype=torch.float64, device=device)
    span = torch.as_tensor(bounds[:, 1] - bounds[:, 0], dtype=torch.float64, device=device)
    xt = (torch.as_tensor(x, dtype=torch.float64, device=device) - lo) / span
    mu, sd = float(y.mean()), float(y.std())
    yt = torch.as_tensor((y - mu) / sd, dtype=torch.float64, device=device).unsqueeze(-1)
    model = SingleTaskGP(xt, yt, outcome_transform=None)
    with gpytorch.settings.max_cholesky_size(10**6):
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        with torch.no_grad():
            if len(x) < 10_000:
                batched = draw_matheron_paths(model, sample_shape=torch.Size([q]))
            else:
                paths = [draw_matheron_paths(model, sample_shape=torch.Size([1])) for _ in range(q)]

    def f(z: torch.Tensor) -> torch.Tensor:
        u = (z.to(device).double() - lo) / span
        if len(x) < 10_000:
            return batched(u) * sd + mu
        if u.ndim == 2:
            return torch.cat([path(u) for path in paths]) * sd + mu
        return torch.cat([path(u[k]) for k, path in enumerate(paths)]) * sd + mu

    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="hartmann6")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--methods", default="rwf_ts,rff_ts,gp_ts")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--num-features", type=int, default=1024)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=600.0, help="skip larger sizes once a method exceeds this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/bo/iteration_cost.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    p = PROBLEMS[args.problem]
    rng = np.random.default_rng(0)
    rows, stopped = [], set()
    x0 = rng.uniform(p.bounds[:, 0], p.bounds[:, 1], size=(50, p.dim))
    RWFGP(64, 2, steps=2, device=str(device)).fit(x0, p(x0))
    for n in args.sizes:
        x = rng.uniform(p.bounds[:, 0], p.bounds[:, 1], size=(n, p.dim))
        y = p(x) + 0.01 * rng.standard_normal(n)
        for method in args.methods.split(","):
            if method in stopped:
                continue
            _sync(device)
            t0 = time.perf_counter()
            try:
                if method == "gp_ts":
                    f = exact_gp_draws(x, y, p.bounds, args.batch_size, device)
                else:
                    kw = dict(steps=args.steps, device=str(device), seed=0)
                    model = (
                        RWFGP(args.num_features, args.levels, lowpass=32, **kw)
                        if method == "rwf_ts"
                        else RFFGP(args.num_features, **kw)
                    )
                    f = PosteriorFunction(model.fit(x, y), args.batch_size, seed=0)
                _sync(device)
                t1 = time.perf_counter()
                maximise(f, p.bounds, x, y, seed=0)
                _sync(device)
                t2 = time.perf_counter()
                row = {
                    "method": method,
                    "n": n,
                    "fit_seconds": t1 - t0,
                    "acq_seconds": t2 - t1,
                    "total_seconds": t2 - t0,
                }
            except (torch.OutOfMemoryError, RuntimeError) as err:
                row = {
                    "method": method,
                    "n": n,
                    "fit_seconds": np.nan,
                    "acq_seconds": np.nan,
                    "total_seconds": np.nan,
                    "error": str(err)[:80],
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not np.isfinite(row["total_seconds"]) or row["total_seconds"] > args.max_seconds:
                stopped.add(method)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print(
        pd.DataFrame(rows)
        .pivot(index="n", columns="method", values="total_seconds")
        .to_string(float_format=lambda v: f"{v:.3g}")
    )


if __name__ == "__main__":
    main()
