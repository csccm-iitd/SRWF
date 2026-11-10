"""Cost and accuracy of posterior function samples: pathwise random features vs exact joint sampling.

    python experiments/bo/sampling_cost.py --device cuda

Cost: wall-clock to draw ``K`` joint posterior samples on ``n`` test points, for an exact GP
(Cholesky of the ``n x n`` posterior covariance, ``O(n^3)``) and for pathwise RWF / RFF samples
(``O(n D)``). Accuracy: 2-Wasserstein distance between the ``D``-feature posterior on a grid and
the exact posterior of the limiting kernel at the same hyperparameters (RWF: ``D_ref`` atoms;
RFF: the closed-form squared-exponential kernel), plus the moment error of ``K`` pathwise draws.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functions import PROBLEMS  # noqa: E402

from srwf import RFFGP, RWFGP, BayesianLinearRegression, RandomWaveletFeatures  # noqa: E402
from srwf.sampling import PosteriorFunction  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def matern52(a: torch.Tensor, b: torch.Tensor, ls: torch.Tensor, var: float) -> torch.Tensor:
    r = torch.cdist(a / ls, b / ls) * math.sqrt(5.0)
    return var * (1 + r + r.pow(2) / 3) * torch.exp(-r)


def exact_joint_samples(
    x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor, ls: torch.Tensor, var: float, noise: float, k: int
) -> torch.Tensor:
    """``k`` joint samples of an exact Matern-5/2 GP posterior on ``xs`` via an ``n x n`` Cholesky."""
    kxx = matern52(x, x, ls, var) + noise * torch.eye(len(x), dtype=x.dtype, device=x.device)
    lx = torch.linalg.cholesky(kxx)
    ksx = matern52(xs, x, ls, var)
    v = torch.linalg.solve_triangular(lx, ksx.t(), upper=False)
    mean = ksx @ torch.cholesky_solve(y.unsqueeze(-1), lx).squeeze(-1)
    cov = matern52(xs, xs, ls, var) - v.t() @ v
    del v, ksx
    cov.diagonal().add_(1e-6 * var)
    l = torch.linalg.cholesky(cov)
    return mean + (l @ torch.randn(len(xs), k, dtype=x.dtype, device=x.device)).t()


def timing(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    p = PROBLEMS["gramacy_2d"]
    rng = np.random.default_rng(0)
    x = rng.uniform(p.bounds[:, 0], p.bounds[:, 1], size=(args.n_train, 2))
    y = p(x) + 0.01 * rng.standard_normal(len(x))
    lo, hi = p.bounds[:, 0], p.bounds[:, 1]
    models = {
        "rwf": RWFGP(args.num_features, 6, steps=args.steps, device=str(device), seed=0).fit(x, y),
        "rff": RFFGP(args.num_features, steps=args.steps, device=str(device), seed=0).fit(x, y),
    }
    xt = torch.as_tensor((x - lo) / (hi - lo), dtype=torch.float64, device=device)
    yt = torch.as_tensor((y - y.mean()) / y.std(), dtype=torch.float64, device=device)
    ls, var, noise = torch.tensor([0.1, 0.1], dtype=torch.float64, device=device), 1.0, 1e-3
    rows = []
    for n in args.grid_sizes:
        side = int(math.ceil(math.sqrt(n)))
        g = np.stack(np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side)), -1).reshape(-1, 2)[:n]
        for name, model in models.items():
            ts = []
            for rep in range(args.repeats + 1):
                _sync(device)
                t0 = time.perf_counter()
                f = PosteriorFunction(model, args.samples, seed=rep)
                with torch.no_grad():
                    gx = torch.as_tensor(lo + g * (hi - lo), dtype=torch.float32, device=device)
                    out = torch.cat([f(c) for c in gx.split(65536)], dim=1)
                _sync(device)
                ts.append(time.perf_counter() - t0)
            rows.append(
                {"method": f"{name}_pathwise", "n": n, "seconds": float(np.median(ts[1:])), "shape": list(out.shape)}
            )
            print(rows[-1], flush=True)
        if n <= args.max_exact:
            gt = torch.as_tensor(g, dtype=torch.float64, device=device)
            ts = []
            try:
                for _ in range(args.repeats + 1):
                    _sync(device)
                    t0 = time.perf_counter()
                    out = exact_joint_samples(xt, yt, gt, ls, var, noise, args.samples)
                    _sync(device)
                    ts.append(time.perf_counter() - t0)
                    del out
                rows.append(
                    {
                        "method": "exact_cholesky",
                        "n": n,
                        "seconds": float(np.median(ts[1:])),
                        "shape": [args.samples, n],
                    }
                )
            except (torch.OutOfMemoryError, RuntimeError) as err:
                rows.append({"method": "exact_cholesky", "n": n, "seconds": float("nan"), "shape": str(err)[:60]})
            print(rows[-1], flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def gaussian_w2(m1: torch.Tensor, s1: torch.Tensor, m2: torch.Tensor, s2: torch.Tensor) -> float:
    """2-Wasserstein distance between ``N(m1, s1)`` and ``N(m2, s2)``."""

    def sqrtm(a: torch.Tensor) -> torch.Tensor:
        e, v = torch.linalg.eigh(0.5 * (a + a.t()))
        return (v * e.clamp_min(0).sqrt()) @ v.t()

    r2 = sqrtm(s2)
    cross = sqrtm(r2 @ s1 @ r2)
    w2 = (m1 - m2).pow(2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * torch.trace(cross)
    return float(w2.clamp_min(0).sqrt())


def _rwf_at(model: RWFGP, xs: torch.Tensor, num_features: int, seed: int) -> BayesianLinearRegression:
    """Fresh RWF feature map with ``num_features`` atoms and the hyperparameters of ``model``."""
    ref = model.blr.features
    f = RandomWaveletFeatures(
        1, num_features, model.levels, model.wavelet, model.growth, model.lowpass, model.ard, seed=seed
    )
    f.initialize(xs.float().cpu())
    with torch.no_grad():
        for name in ("raw_lengthscale", "log_base_scale", "level_log_scale", "level_log_weight"):
            getattr(f, name).copy_(getattr(ref, name))
    f = f.double().to(xs.device)
    blr = BayesianLinearRegression(f).double().to(xs.device)
    with torch.no_grad():
        blr.log_noise.copy_(model.blr.log_noise)
        blr.log_prior_var.copy_(model.blr.log_prior_var)
    return blr


def _posterior(
    z: torch.Tensor, zs: torch.Tensor, y: torch.Tensor, tau2: float, noise: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact function-space posterior on ``zs`` for the kernel ``tau2 * z z^T``."""
    k = tau2 * z @ z.t() + noise * torch.eye(len(z), dtype=z.dtype, device=z.device)
    l = torch.linalg.cholesky(k)
    ksx = tau2 * zs @ z.t()
    v = torch.linalg.solve_triangular(l, ksx.t(), upper=False)
    return ksx @ torch.cholesky_solve(y.unsqueeze(-1), l).squeeze(-1), tau2 * zs @ zs.t() - v.t() @ v


def accuracy(args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, dict]:
    p = PROBLEMS["gramacy_lee"]
    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 2.5, size=(args.n_acc, 1))
    y = p(x) + 0.02 * rng.standard_normal(len(x))
    grid = np.linspace(0.5, 2.5, args.acc_grid)[:, None]
    rwf = RWFGP(256, 5, steps=args.steps, device="cpu", seed=0, lowpass=0).fit(x, y)
    rff = RFFGP(256, steps=args.steps, device="cpu", seed=0).fit(x, y)
    rows, arrays = [], {"grid": grid[:, 0], "x": x[:, 0], "y": y}
    for name, model in (("rwf", rwf), ("rff", rff)):
        xs = model._x(x).double().to(device)
        gs = model._x(grid).double().to(device)
        ys = torch.as_tensor((y - model.y_mu) / model.y_sd, dtype=torch.float64, device=device)
        noise, tau2 = float(model.blr.noise), float(model.blr.prior_var)
        if name == "rwf":
            ref = _rwf_at(model, xs, args.d_ref, seed=10_000)
            with torch.no_grad():
                m_ref, s_ref = _posterior(ref.design(xs), ref.design(gs), ys, tau2, noise)
        else:
            f = model.blr.features
            ls, c = float(f.lengthscale), float(torch.exp(f.log_outputscale))

            def kfun(a, b, scale=tau2 * c, ls=ls):
                return scale * torch.exp(-0.5 * torch.cdist(a / ls, b / ls).pow(2))

            kxx = kfun(xs, xs) + noise * torch.eye(len(xs), dtype=xs.dtype, device=device)
            l = torch.linalg.cholesky(kxx)
            v = torch.linalg.solve_triangular(l, kfun(xs, gs), upper=False)
            m_ref = kfun(gs, xs) @ torch.cholesky_solve(ys.unsqueeze(-1), l).squeeze(-1)
            s_ref = kfun(gs, gs) - v.t() @ v
        arrays[f"{name}_ref_mean"] = (m_ref * model.y_sd + model.y_mu).cpu().numpy()
        arrays[f"{name}_ref_std"] = (s_ref.diagonal().clamp_min(0).sqrt() * model.y_sd).cpu().numpy()
        for d in args.feature_counts:
            for rep in range(args.acc_repeats):
                if name == "rwf":
                    blr = _rwf_at(model, xs, d, seed=rep)
                else:
                    blr = BayesianLinearRegression(_rff_like(model, d, rep, device)).double().to(device)
                    with torch.no_grad():
                        blr.log_noise.copy_(model.blr.log_noise)
                        blr.log_prior_var.copy_(model.blr.log_prior_var)
                with torch.no_grad():
                    blr.condition(xs, ys)
                    zg = blr.design(gs)
                    m = zg @ blr.mean_w + blr.bias
                    s = zg @ torch.cholesky_inverse(blr.chol) @ zg.t()
                    eps = torch.randn(
                        len(blr.mean_w),
                        args.acc_samples,
                        dtype=torch.float64,
                        generator=torch.Generator().manual_seed(rep),
                    )
                    w = blr.mean_w.unsqueeze(-1) + torch.linalg.solve_triangular(
                        blr.chol.t(), eps.to(device), upper=True
                    )
                    draws = (zg @ w).t() + blr.bias
                    emp_m, emp_s = draws.mean(0), torch.cov(draws.t())
                scale = model.y_sd
                rows.append(
                    {
                        "kernel": name,
                        "num_features": d,
                        "rep": rep,
                        "w2_vs_limit": gaussian_w2(m, s, m_ref, s_ref) * scale / math.sqrt(len(grid)),
                        "w2_samples_vs_feature_posterior": gaussian_w2(emp_m, emp_s, m, s)
                        * scale
                        / math.sqrt(len(grid)),
                        "mean_rmse_vs_limit": float((m - m_ref).pow(2).mean().sqrt()) * scale,
                        "std_rmse_vs_limit": float(
                            (s.diagonal().clamp_min(0).sqrt() - s_ref.diagonal().clamp_min(0).sqrt())
                            .pow(2)
                            .mean()
                            .sqrt()
                        )
                        * scale,
                    }
                )
            print(rows[-1], flush=True)
    return pd.DataFrame(rows), arrays


def _rff_like(model: RFFGP, d: int, seed: int, device: torch.device):
    from srwf import RandomFourierFeatures

    ref = model.blr.features
    f = RandomFourierFeatures(1, d, ard=model.ard, seed=seed).double().to(device)
    with torch.no_grad():
        f.raw_lengthscale.copy_(ref.raw_lengthscale)
        f.log_outputscale.copy_(ref.log_outputscale)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    ap.add_argument("--max-exact", type=int, default=30000)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--num-features", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-acc", type=int, default=60)
    ap.add_argument("--acc-grid", type=int, default=200)
    ap.add_argument("--feature-counts", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--d-ref", type=int, default=65536)
    ap.add_argument("--acc-repeats", type=int, default=5)
    ap.add_argument("--acc-samples", type=int, default=2000)
    ap.add_argument("--out", default="results/bo")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc, arrays = accuracy(args, device)
    acc.to_csv(out / "sampling_accuracy.csv", index=False)
    np.savez_compressed(out / "sampling_accuracy_reference.npz", **arrays)
    summary = acc.groupby(["kernel", "num_features"]).agg(["mean", "sem"]).drop(columns="rep")
    print(summary.to_string(float_format=lambda v: f"{v:.3g}"))
    cost = timing(args, device)
    cost.to_csv(out / "sampling_cost.csv", index=False)
    (out / "sampling_cost.json").write_text(json.dumps({"args": vars(args), "rows": cost.to_dict("records")}, indent=1))
    print(cost.pivot(index="n", columns="method", values="seconds").to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
