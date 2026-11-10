"""Bayesian optimisation on S^2 with an RWF surrogate: the benchmark of Wyrwal, Krause and
Borovitskiy (ICLR 2025) and the validation objectives of ``objectives.py``.

Section 4.2 of "Residual Deep Gaussian Processes on Manifolds" (arXiv:2411.00161): minimise the
single-minimum irregular function from 5 uniform random initial points with 200 further noise-free
evaluations, 15 seeds, and report the log10 simple regret. The minimum is an infimum approached
towards the north pole along two meridians, where the function is discontinuous. The initial
designs are those of the baseline runs (``results/sphere/bo_initial_designs.json``).

The surrogate is an RWF GP on the R^3 embedding whose wavelet centres combine a Fibonacci cover of
the sphere (prior variance everywhere, so unexplored regions keep their uncertainty) with the
evaluated inputs (resolution where the data are); the two sets form separate feature blocks with
their own octave weights. The acquisition function (log expected improvement, upper confidence
bound, a pathwise Thompson draw, or a portfolio alternating the two former) is maximised over unit vectors: a randomly rotated Fibonacci
lattice plus perturbations of the best inputs at several angular scales are scored, and the top
candidates are refined with Adam on an ambient vector that is normalised inside the objective.

The configuration used on the benchmark is ``bo_final.json``, fixed from the validation study of
``bo_validation.py`` before the benchmark was run.

    python experiments/sphere/bo.py --out results/sphere --seeds 15 --workers 5

Writes one JSON per seed to ``<out>/bo_runs/rwf_v2__<seed>.json`` and the regret curves to
``<out>/bo_rwf_v2.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from srwf import RWFGP  # noqa: E402
from srwf.bo import PosteriorMoments, log_expected_improvement, upper_confidence_bound  # noqa: E402
from srwf.data.sphere import irregular_bo  # noqa: E402
from srwf.features import RandomFourierFeatures, RandomWaveletFeatures  # noqa: E402
from srwf.geometry import fibonacci_sphere  # noqa: E402
from srwf.model import MixtureGP  # noqa: E402
from srwf.sampling import PosteriorFunction  # noqa: E402

HERE = Path(__file__).resolve().parent
N_INIT = 5
ITERATIONS = 200
F_MIN = -2.0 * 3.0 * math.sqrt(35.0 / (96.0 * math.pi)) * 2.0 / (3.0 * math.sqrt(3.0))
METHOD = "rwf_v2"

DEFAULTS = dict(
    num_features=1024,
    levels=4,
    lowpass=32,
    data_fraction=0.5,
    fourier=0,
    base_scale=1.0,
    wavelet="mexican_hat",
    ard=False,
    fixed_noise=1e-4,
    amplitude_prior=None,
    members=1,
    acquisition="ei",
    beta=2.0,
    beta_schedule=None,
    steps=100,
    warm_start=False,
    n_lattice=4096,
    n_local=1000,
    n_starts=8,
    opt_steps=60,
    lr=0.02,
)


class _CoverAndData(nn.Module):
    """Wavelet blocks with separate centre pools (and octave weights), plus optional Fourier features."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.num_features = int(sum(b.num_features for b in blocks))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, centres: list[torch.Tensor], base_scale: float | None) -> _CoverAndData:
        for block, pool in zip(self.blocks, centres, strict=False):
            if isinstance(block, RandomWaveletFeatures):
                block.initialize(x, centres=pool, base_scale=base_scale)
            else:
                block.initialize(x)
        return self

    def penalty(self) -> torch.Tensor:
        return sum(b.penalty() for b in self.blocks if hasattr(b, "penalty"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.blocks], dim=1)


class SphereRWFGP(RWFGP):
    """RWF GP on the R^3 embedding of S^2 for Bayesian optimisation.

    Args:
        cover: unit vectors covering the sphere, the centre pool of the first wavelet block.
        data: evaluated inputs, the centre pool of the second wavelet block (``data_fraction`` of
            the atoms); ``None`` or ``data_fraction = 0`` uses the cover alone.
        fixed_noise: noise variance in standardised units held fixed during learning (noise-free
            objectives); ``None`` learns it.
        amplitude_prior: ``(mean, std)`` of a Gaussian prior on the log prior variance of the weights.
    """

    def __init__(
        self,
        cover: np.ndarray,
        data: np.ndarray | None = None,
        data_fraction: float = 0.5,
        fixed_noise: float | None = None,
        amplitude_prior: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cover = cover
        self.data = data if data is not None and data_fraction > 0 else None
        self.data_fraction = float(data_fraction) if self.data is not None else 0.0
        self.fixed_noise = fixed_noise
        self.amplitude_prior = amplitude_prior

    def _make_features(self, d: int, seed: int) -> nn.Module:
        n_data = int(round(self.data_fraction * self.num_features))
        kw = dict(levels=self.levels, wavelet=self.wavelet, growth=self.growth, ard=self.ard, omega=self.omega)
        blocks = [RandomWaveletFeatures(d, self.num_features - n_data, lowpass=self.lowpass, seed=seed, **kw)]
        if n_data:
            blocks.append(RandomWaveletFeatures(d, n_data, seed=seed + 3, **kw))
        if self.fourier > 0:
            blocks.append(RandomFourierFeatures(d, self.fourier, self.ard, seed + 7))
        return _CoverAndData(blocks)

    def _init_kwargs(self) -> dict:
        centres = [self._x(self.cover)]
        if self.data is not None:
            centres.append(self._x(self.data))
        return {"centres": centres, "base_scale": self.base_scale}

    def _new_model(self, xs, seed, warm_start):
        blr = super()._new_model(xs, seed, warm_start)
        if self.fixed_noise is not None:
            with torch.no_grad():
                blr.log_noise.fill_(math.log(self.fixed_noise))
            blr.log_noise.requires_grad_(False)
        return blr

    def _loss(self, x, y):
        loss = super()._loss(x, y)
        if self.amplitude_prior is None:
            return loss
        mu, sd = self.amplitude_prior
        return loss + 0.5 * ((self.blr.log_prior_var - mu) / sd) ** 2 / x.shape[0]


def build(cfg: dict, x: np.ndarray, seed: int, cover: np.ndarray) -> SphereRWFGP | MixtureGP:
    kw = dict(
        cover=cover,
        data=x,
        data_fraction=cfg["data_fraction"],
        fixed_noise=cfg["fixed_noise"],
        amplitude_prior=tuple(cfg["amplitude_prior"]) if cfg["amplitude_prior"] else None,
        num_features=cfg["num_features"],
        levels=cfg["levels"],
        lowpass=cfg["lowpass"],
        fourier=cfg["fourier"],
        base_scale=cfg["base_scale"],
        wavelet=cfg["wavelet"],
        ard=cfg["ard"],
        steps=cfg["steps"],
        solver="dual",
        device="cpu",
    )
    if cfg["members"] > 1:
        return MixtureGP.of(SphereRWFGP, members=cfg["members"], seed=seed, **kw)
    return SphereRWFGP(seed=seed, **kw)


def ucb_weight(cfg: dict, step: int) -> float:
    """Multiplier of the posterior standard deviation in the upper confidence bound.

    ``beta_schedule = c`` uses ``sqrt(c * beta_t)`` with the GP-UCB sequence of Srinivas et al.
    (2010), ``beta_t = 2 log(M t^2 pi^2 / (6 delta))`` for ``M`` lattice candidates and ``delta = 0.1``;
    otherwise the fixed ``beta``.
    """
    if cfg["beta_schedule"] is None:
        return float(cfg["beta"])
    beta_t = 2.0 * math.log(cfg["n_lattice"] * step**2 * math.pi**2 / 0.6)
    return math.sqrt(cfg["beta_schedule"] * beta_t)


def acquisition(
    cfg: dict, model: SphereRWFGP | MixtureGP, y_best: float, seed: int, step: int = 1
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A function of unit vectors (shape ``(N, 3)``) to be maximised; higher is better.

    ``"portfolio"`` alternates the upper confidence bound (odd steps) and expected improvement.
    """
    members = model.models if isinstance(model, MixtureGP) else [model]
    kind = cfg["acquisition"]
    if kind == "portfolio":
        kind = "ucb" if step % 2 else "ei"
    if kind == "ts":
        draws = [PosteriorFunction(m, 1, seed=seed + k) for k, m in enumerate(members)]
        return lambda x: torch.stack([f(x)[0] for f in draws]).mean(0)
    moments = [PosteriorMoments(m) for m in members]
    if kind == "ei":

        def ei(x: torch.Tensor) -> torch.Tensor:
            vals = torch.stack([log_expected_improvement(*mo(x), y_best) for mo in moments])
            return torch.logsumexp(vals, 0) - math.log(len(moments))

        return ei
    if kind == "ucb":

        def ucb(x: torch.Tensor) -> torch.Tensor:
            mean, std = zip(*[mo(x) for mo in moments], strict=True)
            mean, std = torch.stack(mean), torch.stack(std)
            total = (std.pow(2).mean(0) + mean.var(0, unbiased=False)).sqrt()
            return upper_confidence_bound(mean.mean(0), total, ucb_weight(cfg, step))

        return ucb
    raise ValueError(f"unknown acquisition {kind!r}")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def maximise_on_sphere(
    acq: Callable[[torch.Tensor], torch.Tensor],
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    seed: int,
    n_lattice: int,
    n_local: int,
    n_starts: int,
    opt_steps: int,
    lr: float,
) -> np.ndarray:
    """Approximate maximiser of ``acq`` over unit vectors."""
    rng = np.random.default_rng(seed)
    lattice = fibonacci_sphere(n_lattice) @ random_rotation(rng).T
    top = x_obs[np.argsort(-y_obs)[: min(5, len(y_obs))]]
    sd = np.repeat([0.3, 0.1, 0.03, 0.01, 0.003], n_local // 5)[:, None]
    local = top[rng.integers(0, len(top), len(sd))] + sd * rng.standard_normal((len(sd), 3))
    cand = np.concatenate([lattice, local / np.linalg.norm(local, axis=1, keepdims=True)])
    with torch.no_grad():
        vals = torch.cat([acq(torch.as_tensor(c, dtype=torch.float32)) for c in np.array_split(cand, 4)]).double()
    order = torch.argsort(vals, descending=True)[:n_starts].numpy()
    best_x, best_v = cand[order[0]], float(vals[order[0]])
    z = torch.as_tensor(cand[order], dtype=torch.float32).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(opt_steps):
        opt.zero_grad()
        loss = -acq(z / z.norm(dim=1, keepdim=True)).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        unit = z / z.norm(dim=1, keepdim=True)
        v = acq(unit).double()
    if torch.isfinite(v).any() and float(v[torch.isfinite(v)].max()) > best_v:
        best_x = unit[int(torch.where(torch.isfinite(v), v, -torch.inf).argmax())].double().numpy()
    return best_x / np.linalg.norm(best_x)


def optimise(
    objective: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    cfg: dict,
    seed: int,
    iterations: int = ITERATIONS,
) -> dict:
    """Minimise ``objective`` on S^2 from ``x_init`` with ``iterations`` further evaluations."""
    torch.manual_seed(seed)
    x = np.asarray(x_init, dtype=np.float64)
    y = -np.asarray(objective(x), dtype=np.float64)
    cover = fibonacci_sphere(4 * cfg["num_features"])
    model, fit_s, acq_s = None, [], []
    for it in range(iterations):
        t0 = time.perf_counter()
        previous = model if cfg["warm_start"] and isinstance(model, SphereRWFGP) else None
        model = build(cfg, x, seed, cover)
        model.fit(x, y, warm_start=previous) if previous is not None else model.fit(x, y)
        acq = acquisition(cfg, model, float(y.max()), seed * 100003 + it, it + 1)
        t1 = time.perf_counter()
        x_new = maximise_on_sphere(
            acq,
            x,
            y,
            seed * 100003 + it,
            cfg["n_lattice"],
            cfg["n_local"],
            cfg["n_starts"],
            cfg["opt_steps"],
            cfg["lr"],
        )
        acq_s.append(time.perf_counter() - t1)
        fit_s.append(t1 - t0)
        x = np.vstack([x, x_new])
        y = np.append(y, -float(objective(x_new[None])[0]))
    return {"x": x.tolist(), "f": (-y).tolist(), "fit_seconds": fit_s, "acq_seconds": acq_s}


def load_config(path: Path | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path is not None:
        cfg.update(json.loads(Path(path).read_text()))
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    return cfg


def initial_designs(path: Path) -> dict[int, np.ndarray]:
    d = json.loads(path.read_text())
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in d["x_init"].items()}


def run_seed(seed: int, out: str, config: str) -> str:
    torch.set_num_threads(1)
    path = Path(out) / "bo_runs" / f"{METHOD}__{seed}.json"
    if path.exists():
        return str(path)
    cfg = load_config(Path(config))
    x_init = initial_designs(Path(out) / "bo_initial_designs.json")[seed]
    rec = optimise(irregular_bo, x_init, cfg, seed)
    best = np.minimum.accumulate(np.array(rec["f"]))
    record = {"method": METHOD, "seed": seed, "settings": cfg, "f_min": F_MIN, "regret": (best - F_MIN).tolist(), **rec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return str(path)


def summarise(out: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted((out / "bo_runs").glob(f"{METHOD}__*.json"))]
    log_r = np.log10(np.maximum(np.array([r["regret"][N_INIT - 1 :] for r in runs]), 1e-12))
    summary = {
        "f_min": F_MIN,
        "step0": "after the 5 initial points",
        METHOD: {
            "seeds": [r["seed"] for r in runs],
            "settings": runs[0]["settings"],
            "log10_regret_median": np.median(log_r, 0).tolist(),
            "log10_regret_mean": log_r.mean(0).tolist(),
            "log10_regret_std": log_r.std(0).tolist(),
            "log10_regret_runs": log_r.tolist(),
            "seconds_per_iteration": float(
                np.mean([np.mean(r["fit_seconds"]) + np.mean(r["acq_seconds"]) for r in runs])
            ),
        },
    }
    (out / "bo_rwf_v2.json").write_text(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sphere")
    ap.add_argument("--config", default=str(HERE / "bo_final.json"))
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    with ProcessPoolExecutor(args.workers) as pool:
        for p in pool.map(run_seed, seeds, [args.out] * len(seeds), [args.config] * len(seeds)):
            print(p, flush=True)
    summary = summarise(Path(args.out))[METHOD]
    curves = np.array(summary["log10_regret_runs"])
    for step in (100, 180, 200):
        q = np.percentile(curves[:, step], [25, 50, 75])
        print(f"step {step}: median log10 regret {q[1]:.3f}, IQR [{q[0]:.3f}, {q[2]:.3f}]")
    print(f"{summary['seconds_per_iteration']:.2f} s per iteration")


if __name__ == "__main__":
    main()
