"""BO methods compared in ``run.py``: random-feature TS, exact-GP TS / EI (BoTorch) and random search."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import numpy as np
import torch

from srwf import RFFGP, RWFGP
from srwf.bo import FeatureSampler, Trace, sobol, thompson_sampling

METHODS = ("rwf_ts", "rff_ts", "gp_ts", "gp_ei", "random")

warnings.filterwarnings("ignore", message=".*not standardized.*")


class _WarmStart:
    """Initialise every hyperparameter from ``self.warm`` (a previous fit) before optimising, so
    the marginal-likelihood optimisation accumulates across BO iterations."""

    warm: dict | None = None

    def _optimise(self, x: torch.Tensor, y: torch.Tensor) -> list[float]:
        if self.warm:
            with torch.no_grad():
                for name, p in self.blr.named_parameters():
                    if name in self.warm and self.warm[name].shape == p.shape:
                        p.copy_(self.warm[name])
        return super()._optimise(x, y)

    def parameters_dict(self) -> dict:
        return {k: v.detach().clone() for k, v in self.blr.named_parameters()}


class BORWFGP(_WarmStart, RWFGP):
    """RWF GP for BO with optional warm start and Gaussian priors (std ``weight_prior``) on the
    per-octave log weights and log scale offsets, which keep all octaves active when a few
    clustered observations would otherwise switch the fine octaves off."""

    def __init__(self, *args, weight_prior: float = 1.0, warm: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight_prior = float(weight_prior)
        self.warm = warm

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = super()._loss(x, y)
        if not self.weight_prior:
            return loss
        f = self.blr.features
        f = f.blocks[0] if hasattr(f, "blocks") else f
        pen = f.level_log_weight.pow(2).sum() + 4.0 * f.level_log_scale.pow(2).sum()
        return loss + 0.5 * pen / (self.weight_prior**2 * x.shape[0])


class BORFFGP(_WarmStart, RFFGP):
    def __init__(self, *args, warm: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warm = warm


def rwf_sampler(
    bounds: np.ndarray,
    num_features: int,
    levels: int,
    steps: int,
    seed: int,
    data_fraction: float = 0.5,
    lowpass: int = 32,
    fourier: int = 0,
    anchors: int = 0,
    weight_prior: float = 1.0,
    warm_start: bool = False,
    device: str = "cpu",
) -> FeatureSampler:
    """RWF surrogate with ``num_features`` features in total: wavelet atoms, ``lowpass`` Gaussian
    scaling functions and ``fourier`` random Fourier features (a stationary sum-kernel component).

    ``data_fraction`` of the wavelet centre pool are evaluated inputs (resolution where data are),
    the rest cover the box so the prior keeps fine-scale variance in unexplored regions.
    ``weight_prior = 0`` gives ML-II hyperparameters.
    """
    atoms = num_features - lowpass - fourier
    pool = 4 * atoms
    n_data = int(round(data_fraction * pool))
    cover = sobol(max(pool - n_data, 1), bounds, seed + 17)

    def make(x: np.ndarray) -> BORWFGP:
        reps = int(np.ceil(max(n_data, 1) / len(x)))
        centres = np.concatenate([np.tile(x, (reps, 1))[:n_data], cover]) if n_data > 0 else cover
        warm = sampler.model.parameters_dict() if warm_start and sampler.model is not None else None
        return BORWFGP(
            atoms,
            levels,
            centres=centres,
            lowpass=lowpass,
            fourier=fourier,
            anchors=anchors,
            weight_prior=weight_prior,
            warm=warm,
            steps=steps,
            device=device,
            seed=seed,
        )

    sampler = FeatureSampler(make)
    return sampler


def rff_sampler(
    num_features: int, steps: int, seed: int, device: str = "cpu", warm_start: bool = False
) -> FeatureSampler:
    def make(x: np.ndarray) -> BORFFGP:
        warm = sampler.model.parameters_dict() if warm_start and sampler.model is not None else None
        return BORFFGP(num_features, warm=warm, steps=steps, device=device, seed=seed)

    sampler = FeatureSampler(make)
    return sampler


class _Unit:
    def __init__(self, bounds: np.ndarray, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> None:
        kw = {"dtype": torch.float64, "device": device}
        self.lo = torch.as_tensor(bounds[:, 0], **kw)
        self.span = torch.as_tensor(bounds[:, 1] - bounds[:, 0], **kw)
        self.mu, self.sd = float(y.mean()), float(y.std() + 1e-12)
        self.x = (torch.as_tensor(x, **kw) - self.lo) / self.span
        self.y = torch.as_tensor((y - self.mu) / self.sd, **kw).unsqueeze(-1)


def _fit_gp(u: _Unit):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model = SingleTaskGP(u.x, u.y, outcome_transform=None)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def gp_ts_sampler(bounds: np.ndarray, device: str = "cpu") -> Callable:
    """Exact GP (BoTorch defaults) with pathwise (Matheron) posterior samples on an RFF prior."""
    from botorch.sampling.pathwise import draw_matheron_paths

    def sample(x: np.ndarray, y: np.ndarray, n_samples: int, seed: int):
        u = _Unit(bounds, x, y, device)
        model = _fit_gp(u)
        torch.manual_seed(seed)
        with torch.no_grad():
            paths = draw_matheron_paths(model, sample_shape=torch.Size([n_samples]))

        def f(z: torch.Tensor) -> torch.Tensor:
            return paths((z.to(u.lo.device).double() - u.lo) / u.span) * u.sd + u.mu

        return f

    return sample


def gp_ei(objective, bounds: np.ndarray, budget: int, n_init: int, seed: int, batch_size: int = 1) -> Trace:
    """Exact GP with (q-)log expected improvement, maximised by ``optimize_acqf``."""
    from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
    from botorch.optim import optimize_acqf

    x = sobol(n_init, bounds, seed)
    trace = Trace(x, objective(x), n_init)
    unit = torch.stack([torch.zeros(len(bounds)), torch.ones(len(bounds))]).double()
    while len(trace.y) < budget:
        q = min(batch_size, budget - len(trace.y))
        t0 = time.perf_counter()
        u = _Unit(bounds, trace.x, trace.y)
        model = _fit_gp(u)
        t1 = time.perf_counter()
        torch.manual_seed(seed * 100003 + len(trace.y))
        best = u.y.max()
        acq = LogExpectedImprovement(model, best_f=best) if q == 1 else qLogExpectedImprovement(model, best_f=best)
        cand, _ = optimize_acqf(acq, unit, q=q, num_restarts=8, raw_samples=512)
        x_next = (u.lo + cand.detach() * u.span).numpy()
        t2 = time.perf_counter()
        trace.x = np.concatenate([trace.x, x_next])
        trace.y = np.concatenate([trace.y, objective(x_next)])
        trace.fit_seconds.append(t1 - t0)
        trace.acq_seconds.append(t2 - t1)
    return trace


def random_search(objective, bounds: np.ndarray, budget: int, n_init: int, seed: int) -> Trace:
    x = sobol(n_init, bounds, seed)
    rest = np.random.default_rng(seed).uniform(bounds[:, 0], bounds[:, 1], size=(budget - n_init, len(bounds)))
    x = np.concatenate([x, rest])
    return Trace(x, objective(x), n_init, [0.0] * (budget - n_init), [0.0] * (budget - n_init))


def run_method(
    method: str,
    objective,
    bounds: np.ndarray,
    budget: int,
    n_init: int,
    seed: int,
    num_features: int = 1024,
    steps: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
    callback=None,
    **rwf_options,
) -> Trace:
    if method == "random":
        return random_search(objective, bounds, budget, n_init, seed)
    if method == "gp_ei":
        return gp_ei(objective, bounds, budget, n_init, seed, batch_size)
    if method == "rwf_ts":
        sampler = rwf_sampler(bounds, num_features, steps=steps, seed=seed, device=device, **rwf_options)
    elif method == "rff_ts":
        sampler = rff_sampler(num_features, steps, seed, device, rwf_options.get("warm_start", False))
    elif method == "gp_ts":
        sampler = gp_ts_sampler(bounds, device)
    else:
        raise ValueError(f"unknown method {method!r}")
    return thompson_sampling(objective, bounds, sampler, budget, n_init, batch_size, seed, callback)
