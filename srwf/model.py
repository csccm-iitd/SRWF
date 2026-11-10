"""Random-feature Gaussian process regressors with an exact posterior.

>>> model = RWFGP(num_features=1024, levels=5).fit(x_train, y_train)
>>> pred = model.predict(x_test)
>>> pred.mean, pred.std
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .blr import BayesianLinearRegression
from .features import RandomFourierFeatures, RandomWaveletFeatures, concat_features


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray
    f_std: np.ndarray


@dataclass
class FitInfo:
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0


def _as_tensor(a, device, dtype=torch.float32) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    return t.reshape(t.shape[0], -1) if t.ndim != 1 else t


class _FeatureGP:
    """Shared machinery: standardisation, hyperparameter learning, exact conditioning."""

    def __init__(
        self,
        steps: int = 300,
        lr: float = 0.05,
        objective: str = "mll",
        subsample: int | None = 20000,
        noise: float = 0.1,
        standardize: bool = True,
        device: str | None = None,
        seed: int = 0,
        chunk: int = 16384,
        solver: str = "primal",
        restarts: int = 1,
    ) -> None:
        self.steps = int(steps)
        self.lr = float(lr)
        self.objective = objective
        self.subsample = subsample
        self.init_noise = float(noise)
        self.standardize = standardize
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        self.chunk = int(chunk)
        self.solver = solver
        self.restarts = max(int(restarts), 1)
        self.info = FitInfo()

    def _make_features(self, d: int, seed: int) -> nn.Module:
        raise NotImplementedError

    def _init_kwargs(self) -> dict:
        return {}

    def _x(self, x) -> torch.Tensor:
        x = _as_tensor(x, self.device)
        x = x.unsqueeze(-1) if x.ndim == 1 else x
        return (x - self.x_mu) / self.x_sd

    def fit(self, x, y, warm_start: _FeatureGP | None = None) -> _FeatureGP:
        """Learn the hyperparameters and condition on all rows.

        ``warm_start`` initialises the hyperparameters from another fitted model of the same
        configuration (e.g. the previous iteration of a sequential design). With ``restarts > 1``
        the hyperparameters are learned for that many feature draws (seeds ``seed, seed + 1, ...``)
        and the draw with the highest training objective is kept.
        """
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        x = _as_tensor(x, self.device)
        x = x.unsqueeze(-1) if x.ndim == 1 else x
        y = _as_tensor(y, self.device, torch.float64).reshape(-1)
        if self.standardize:
            self.x_mu, self.x_sd = x.mean(0), x.std(0).clamp_min(1e-8)
            self.y_mu, self.y_sd = float(y.mean()), float(y.std().clamp_min(1e-12))
        else:
            self.x_mu, self.x_sd = (
                torch.zeros(x.shape[1], device=self.device),
                torch.ones(x.shape[1], device=self.device),
            )
            self.y_mu, self.y_sd = 0.0, 1.0
        xs = (x - self.x_mu) / self.x_sd
        ys = (y - self.y_mu) / self.y_sd

        g = torch.Generator().manual_seed(self.seed)
        if self.subsample and xs.shape[0] > self.subsample:
            idx = torch.randperm(xs.shape[0], generator=g)[: self.subsample].to(self.device)
            xf, yf = xs[idx], ys[idx]
        else:
            xf, yf = xs, ys
        best = None
        for k in range(self.restarts):
            self.blr = self._new_model(xs, self.seed + k, warm_start)
            losses = self._optimise(xf, yf)
            score = min(losses, default=float("inf"))
            if best is None or score < best[0]:
                best = (score, self.blr, losses)
        _, self.blr, self.info.losses = best
        del best
        self.blr.condition(xs, ys, chunk=self.chunk, solver=self.solver)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.info.seconds = time.perf_counter() - t0
        return self

    def _new_model(self, xs: torch.Tensor, seed: int, warm_start: _FeatureGP | None) -> BayesianLinearRegression:
        feats = self._make_features(xs.shape[1], seed).to(self.device)
        feats.initialize(xs, **self._init_kwargs())
        blr = BayesianLinearRegression(feats, noise=self.init_noise).to(self.device)
        if warm_start is not None:
            with torch.no_grad():
                source = dict(warm_start.blr.named_parameters())
                for name, param in blr.named_parameters():
                    if name in source and source[name].shape == param.shape:
                        param.copy_(source[name])
        return blr

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        pen = self.blr.features.penalty() if hasattr(self.blr.features, "penalty") else 0.0
        if self.objective == "mll":
            return (-self.blr.log_marginal_likelihood(x, y, chunk=self.chunk) + pen) / n
        if self.objective == "loo":
            mu, v = self.blr.loo(x, y, chunk=self.chunk)
            return (0.5 * torch.log(2 * torch.pi * v) + 0.5 * (y - mu).pow(2) / v).mean() + pen / n
        raise ValueError(f"unknown objective {self.objective!r}")

    def _optimise(self, x: torch.Tensor, y: torch.Tensor) -> list[float]:
        params = [p for p in self.blr.parameters() if p.requires_grad]
        if self.steps <= 0 or not params:
            return []
        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=self.lr * 0.05)
        losses: list[float] = []
        best = (float("inf"), None)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = self._loss(x, y)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            val = float(loss.detach())
            losses.append(val)
            if val < best[0]:
                best = (val, {k: v.detach().clone() for k, v in self.blr.state_dict().items()})
        if best[1] is not None:
            self.blr.load_state_dict(best[1])
        return losses

    @torch.no_grad()
    def predict(self, x, return_f_std: bool = False) -> Prediction:
        xs = self._x(x)
        mean, var = self.blr.predict(xs, chunk=self.chunk)
        _, fvar = self.blr.predict(xs, chunk=self.chunk, include_noise=False) if return_f_std else (None, None)
        m = (mean * self.y_sd + self.y_mu).cpu().numpy()
        s = (var.sqrt() * self.y_sd).cpu().numpy()
        fs = (
            (fvar.sqrt() * self.y_sd).cpu().numpy()
            if fvar is not None
            else np.sqrt(np.maximum(s**2 - self.noise_var, 0))
        )
        return Prediction(m, s, fs)

    @property
    def noise_var(self) -> float:
        return float(self.blr.noise) * self.y_sd**2

    def posterior_function(self, n_samples: int = 1, seed: int = 0):
        """``n_samples`` frozen posterior function draws, evaluable and differentiable anywhere."""
        from .sampling import PosteriorFunction

        return PosteriorFunction(self, n_samples=n_samples, seed=seed)

    @torch.no_grad()
    def sample_posterior(self, x, n_samples: int = 8, seed: int = 0) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        f = self.blr.sample_posterior(self._x(x), n_samples, generator=g)
        return (f * self.y_sd + self.y_mu).cpu().numpy()

    @torch.no_grad()
    def sample_prior(self, x, n_samples: int = 8, seed: int = 0) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        f = self.blr.sample_prior(self._x(x), n_samples, generator=g)
        return (f * self.y_sd + self.y_mu).cpu().numpy()


class RWFGP(_FeatureGP):
    """Gaussian process regression with Random Wavelet Features.

    Args:
        num_features: number of wavelet atoms ``D``.
        levels: number of dyadic octaves.
        wavelet: mother wavelet (``"mexican_hat"``, ``"morlet"``).
        growth: per-octave growth of the atom count.
        lowpass: extra Gaussian scaling functions at the coarsest scale.
        fourier: extra random Fourier features with their own lengthscales (sum kernel).
        ard: per-dimension lengthscales.
        centres: ``"data"``, ``"box"`` or an array of candidate centre locations.
        base_scale: coarsest scale in standardised units (default: half the mean input span).
        anchors: number of anchors of the location-dependent octave weights (0: the paper's model).
        anchor_prior: hyperprior variance of the location-dependent weights.
        **kwargs: options of the base class (``steps``, ``lr``, ``objective``, ``solver``, ...); ``solver="dual"``
            conditions through the ``N x N`` kernel matrix, the cheaper form when ``N < D``.
    """

    def __init__(
        self,
        num_features: int = 1024,
        levels: int = 5,
        wavelet: str = "mexican_hat",
        growth: float = 2.0,
        lowpass: int = 0,
        fourier: int = 0,
        ard: bool = True,
        centres="data",
        base_scale: float | None = None,
        omega: float = 5.0,
        anchors: int = 0,
        anchor_prior: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.levels = levels
        self.wavelet = wavelet
        self.growth = growth
        self.lowpass = lowpass
        self.fourier = fourier
        self.ard = ard
        self.centres = centres
        self.base_scale = base_scale
        self.omega = omega
        self.anchors = anchors
        self.anchor_prior = anchor_prior

    def _make_features(self, d: int, seed: int) -> nn.Module:
        rwf = RandomWaveletFeatures(
            d,
            self.num_features,
            self.levels,
            self.wavelet,
            self.growth,
            self.lowpass,
            self.ard,
            self.omega,
            self.anchors,
            self.anchor_prior,
            seed,
        )
        if self.fourier > 0:
            return concat_features([rwf, RandomFourierFeatures(d, self.fourier, self.ard, seed + 7)])
        return rwf

    def _init_kwargs(self) -> dict:
        centres = self.centres
        if not isinstance(centres, str):
            centres = self._x(centres)
        return {"centres": centres, "base_scale": self.base_scale}


class RFFGP(_FeatureGP):
    """Gaussian process regression with Random Fourier Features (squared-exponential kernel)."""

    def __init__(self, num_features: int = 1024, ard: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.ard = ard

    def _make_features(self, d: int, seed: int) -> nn.Module:
        return RandomFourierFeatures(d, self.num_features, self.ard, seed)


class MixtureGP:
    """Equal-weight mixture of feature GPs that differ only in their random feature draws.

    Each member is fitted independently; the predictive distribution is the Gaussian mixture of
    the members, summarised by its mean and variance. Averaging over feature draws removes most of
    the Monte Carlo variance of a single draw at ``K`` times the cost.

    >>> model = MixtureGP.of(RWFGP, members=4, num_features=1024).fit(x_train, y_train)

    Args:
        models: unfitted models of the same configuration with different seeds.
    """

    def __init__(self, models: list[_FeatureGP]) -> None:
        if not models:
            raise ValueError("a mixture needs at least one member")
        self.models = list(models)
        self.info = FitInfo()

    @classmethod
    def of(cls, model_cls: type[_FeatureGP], members: int = 4, seed: int = 0, **kwargs) -> MixtureGP:
        """``members`` copies of ``model_cls(**kwargs)`` with seeds ``seed, seed + 1, ...``."""
        return cls([model_cls(seed=seed + k, **kwargs) for k in range(int(members))])

    def fit(self, x, y) -> MixtureGP:
        t0 = time.perf_counter()
        for m in self.models:
            m.fit(x, y)
        self.info.seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, x, return_f_std: bool = False) -> Prediction:
        preds = [m.predict(x, return_f_std=return_f_std) for m in self.models]
        means = np.stack([p.mean for p in preds])
        mean = means.mean(0)
        spread = (means**2).mean(0) - mean**2
        std = np.sqrt(np.maximum(np.mean([p.std**2 for p in preds], 0) + spread, 0))
        f_std = np.sqrt(np.maximum(np.mean([p.f_std**2 for p in preds], 0) + spread, 0))
        return Prediction(mean, std, f_std)

    @property
    def noise_var(self) -> float:
        return float(np.mean([m.noise_var for m in self.models]))

    @torch.no_grad()
    def sample_posterior(self, x, n_samples: int = 8, seed: int = 0) -> np.ndarray:
        """Draws from the mixture: each sample comes from a uniformly chosen member."""
        rng = np.random.default_rng(seed)
        member = rng.integers(len(self.models), size=int(n_samples))
        out = None
        for k in np.unique(member):
            rows = np.flatnonzero(member == k)
            draws = self.models[k].sample_posterior(x, len(rows), seed=seed + int(k))
            if out is None:
                out = np.empty((int(n_samples), draws.shape[1]))
            out[rows] = draws
        return out
