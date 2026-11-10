"""Large-scale UCI regression benchmark of the scalable-GP literature (Wang et al., 2019; Lin et al., 2023, 2024).

Protocol of Lin et al.: the fixed 90/10 splits of ``uci_datasets`` (five splits, three for
houseelectric), inputs and targets z-scored with training statistics, RMSE and Gaussian NLL in
standardised target units. Standardised inputs are clipped to +-1e4 (this only affects one test row of
keggdirected, split 2, which lies 1e17 training standard deviations out in a near-constant column).

The kernel hyperparameters (lengthscales, octave scales and weights, noise, prior variance) do not
depend on the number of features, so they are learned by exact marginal likelihood with
``--hyper-features`` features on a random subsample and then reused by a model with ``D`` features
that is conditioned exactly on all training rows; the noise and prior variances are then re-fitted
by exact marginal likelihood on all rows from the sufficient statistics (a scalar search over the
profile likelihood, one ``D x D`` Cholesky factorisation per evaluation). The number of octaves and ``D`` are selected per
dataset by validation NLL on 10% of the training rows of split 0. The RFF control uses the same
pipeline, ``D`` and solver. Reported time is wall-clock for hyperparameter learning, conditioning
and test prediction.

    python experiments/benchmarks/large_scale.py --datasets pol elevators --out results/large
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from srwf import RFFGP, RWFGP
from srwf.blr import cholesky
from srwf.data import large
from srwf.metrics import evaluate

LEVELS = (4, 6)
SIZES = (4096, 8192, 16384, 32768)
FOURIER_SHARE = 0.25
SPLITS = {"houseelectric": 3}


def fit_scales(blr, st, width: float = 6.0) -> None:
    """Maximise the exact marginal likelihood over ``(sigma^2, tau^2)`` on all rows.

    For a fixed ratio ``rho = sigma^2 / tau^2`` the optimal ``sigma^2`` is ``(||r||^2 - b^T (Z^T Z + rho I)^-1 b) / N``,
    so the profile likelihood is a function of ``rho`` alone and is minimised by a bounded scalar search.
    """
    eye = torch.eye(st.ztz.shape[0], dtype=st.ztz.dtype, device=st.ztz.device)
    d, n = st.ztz.shape[0], st.n

    def profile(log_rho: float) -> tuple[float, float]:
        rho = math.exp(log_rho)
        chol = cholesky(st.ztz + rho * eye)
        q = float(st.rss - st.ztr.dot(torch.cholesky_solve(st.ztr.unsqueeze(-1), chol).squeeze(-1)))
        s2 = max(q, 1e-300) / n
        logdet = 2.0 * float(torch.log(torch.diagonal(chol)).sum())
        return 0.5 * (n * math.log(s2) + logdet - d * log_rho), s2

    start = float(blr.log_noise - blr.log_prior_var)
    res = minimize_scalar(
        lambda t: profile(t)[0], bounds=(start - width, start + width), method="bounded", options=dict(xatol=1e-2)
    )
    s2 = profile(res.x)[1]
    with torch.no_grad():
        blr.log_noise.fill_(math.log(s2))
        blr.log_prior_var.fill_(math.log(s2) - res.x)


class _Pretrained:
    """Starts from given hyperparameters; ``refine`` re-fits the noise and prior variances on all rows."""

    def __init__(self, hyper: dict[str, torch.Tensor], refine: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hyper = hyper
        self.refine = refine

    def _optimise(self, x: torch.Tensor, y: torch.Tensor) -> list[float]:
        with torch.no_grad():
            for name, p in self.blr.named_parameters():
                p.copy_(self.hyper[name])
        return super()._optimise(x, y)

    def fit(self, x, y):
        super().fit(x, y)
        if self.refine:
            xs = self._x(x)
            ys = (torch.as_tensor(np.asarray(y), dtype=torch.float64, device=self.device) - self.y_mu) / self.y_sd
            with torch.no_grad():
                st = self.blr.stats(xs, ys, self.chunk)
            fit_scales(self.blr, st)
            with torch.no_grad():
                self.blr.chol, self.blr.mean_w = self.blr._posterior(st)
        return self


class PretrainedRWF(_Pretrained, RWFGP):
    pass


class PretrainedRFF(_Pretrained, RFFGP):
    pass


def model_kwargs(model: str, d: int, levels: int) -> dict:
    if model == "rff":
        return dict(num_features=d)
    n_rff = int(round(FOURIER_SHARE * d))
    return dict(num_features=d - n_rff, fourier=n_rff, levels=levels)


def standardise(x_train, y_train, x_test, y_test, clip: float = 1e4) -> tuple[np.ndarray, ...]:
    """Z-score with training statistics; inputs are clipped to ``+-clip`` standard deviations."""
    xm, xs = x_train.mean(0), x_train.std(0)
    xs = np.where(xs > 0, xs, 1.0)
    ym, ys = y_train.mean(), y_train.std()
    x_train, x_test = (np.clip((a - xm) / xs, -clip, clip) for a in (x_train, x_test))
    return x_train, (y_train - ym) / ys, x_test, (y_test - ym) / ys


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def learn_hyper(model: str, levels: int, x, y, args, seed: int) -> tuple[dict[str, torch.Tensor], float, float]:
    """Hyperparameters by exact marginal likelihood with ``args.hyper_features`` features; also seconds and peak GB."""
    cls = RFFGP if model == "rff" else RWFGP
    kw = model_kwargs(model, args.hyper_features, levels)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    m = cls(**kw, steps=args.steps, subsample=args.subsample, seed=seed).fit(x, y)
    hyper = {k: v.detach().clone() for k, v in m.blr.named_parameters()}
    _sync()
    return hyper, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def condition(model: str, levels: int, d: int, hyper: dict, x, y, x_test, y_test, seed: int, refine: bool) -> dict:
    """Exact posterior with ``d`` features on all rows; scores, time and peak memory of this stage."""
    cls = PretrainedRFF if model == "rff" else PretrainedRWF
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    m = cls(hyper, refine, **model_kwargs(model, d, levels), steps=0, seed=seed).fit(x, y)
    p = m.predict(x_test)
    _sync()
    out = dict(condition_seconds=time.perf_counter() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
    out.update(evaluate(y_test, p.mean, p.std))
    out["noise_var"] = m.noise_var
    return out


def select(dataset: str, args) -> dict[str, dict]:
    s = large.split(dataset, 0)
    perm = np.random.default_rng(0).permutation(len(s.y_train))
    va, tr = perm[: len(perm) // 10], perm[len(perm) // 10 :]
    x, y, xv, yv = standardise(s.x_train[tr], s.y_train[tr], s.x_train[va], s.y_train[va])
    scores = {}
    for levels in args.levels:
        hyper, t_hyper, _ = learn_hyper("rwf", levels, x, y, args, seed=0)
        for d in args.sizes:
            r = condition("rwf", levels, d, hyper, x, y, xv, yv, seed=0, refine=args.refine)
            scores[f"L{levels}-D{d}"] = dict(levels=levels, d=d, hyper_seconds=t_hyper, **r)
            print(f"  select L{levels} D{d:5d} " + fmt(r), flush=True)
    return scores


def fmt(r: dict) -> str:
    sec = r.get("seconds", r.get("condition_seconds", 0.0))
    return f"rmse {r['rmse']:.4f} nll {r['nll']:.4f} time {sec:.1f}s mem {r['peak_gb']:.1f}GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(large.DATASETS))
    ap.add_argument("--models", nargs="+", default=["rwf", "rff"])
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--levels", nargs="+", type=int, default=list(LEVELS))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--hyper-features", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--subsample", type=int, default=50000)
    ap.add_argument(
        "--no-refine", dest="refine", action="store_false", help="keep the subsample noise and prior variances"
    )
    ap.add_argument("--out", type=Path, default=Path("results/large"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        path = args.out / f"{dataset}.json"
        res = json.loads(path.read_text()) if path.exists() else {"runs": []}
        res.update(
            dataset=dataset,
            hyper_features=args.hyper_features,
            steps=args.steps,
            subsample=args.subsample,
            refine=args.refine,
        )
        print(dataset, flush=True)
        if "chosen" not in res:
            res["selection"] = select(dataset, args)
            res["chosen"] = min(res["selection"], key=lambda k: res["selection"][k]["nll"])
            path.write_text(json.dumps(res, indent=1))
        levels, d = res["selection"][res["chosen"]]["levels"], res["selection"][res["chosen"]]["d"]
        done = {(r["model"], r["split"]) for r in res["runs"]}
        for k in range(min(args.splits, SPLITS.get(dataset, args.splits))):
            todo = [m for m in args.models if (m, k) not in done]
            if not todo:
                continue
            s = large.split(dataset, k)
            data = standardise(s.x_train, s.y_train, s.x_test, s.y_test)
            for model in todo:
                hyper, t_hyper, gb_hyper = learn_hyper(model, levels, *data[:2], args, seed=k)
                r = dict(model=model, split=k, config=res["chosen"], n_train=len(s.y_train), hyper_seconds=t_hyper)
                r.update(condition(model, levels, d, hyper, *data, seed=k, refine=args.refine))
                r["seconds"] = r["hyper_seconds"] + r["condition_seconds"]
                r["peak_gb"] = max(r["peak_gb"], gb_hyper)
                res["runs"].append(r)
                path.write_text(json.dumps(res, indent=1))
                print(f"  {model} split {k} " + fmt(r), flush=True)


if __name__ == "__main__":
    main()
