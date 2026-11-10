"""Validation study for the Bayesian-optimisation method of ``bo.py``.

Every candidate configuration is run on the validation objectives of ``objectives.py`` from
uniform random initial designs; the benchmark target is not used. Regret is normalised by the
standard deviation of each objective over the sphere so that runs on different objectives can be
pooled. ``exact_gp_ei``, an exact Matern-5/2 GP on the R^3 embedding with log expected improvement
and the same acquisition optimiser, is the stationary reference.

    python experiments/sphere/bo_validation.py --out results/sphere --stage 1 --workers 12
    python experiments/sphere/bo_validation.py --out results/sphere --stage 2 --workers 12
    python experiments/sphere/bo_validation.py --out results/sphere --summary

Writes one JSON per run to ``<out>/bo_validation/`` and the pooled table to
``<out>/bo_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bo import ITERATIONS, N_INIT, load_config, maximise_on_sphere, optimise  # noqa: E402
from objectives import validation_objectives  # noqa: E402

STAGES: dict[int, dict[str, dict]] = {
    1: {
        "ei": {},
        "ts": {"acquisition": "ts"},
        "ucb": {"acquisition": "ucb"},
        "ucb_gp": {"acquisition": "ucb", "beta_schedule": 1.0},
        "ucb_gp_fifth": {"acquisition": "ucb", "beta_schedule": 0.2},
        "ei_amplitude_10": {"amplitude_prior": [math.log(10.0), 0.5]},
        "ei_amplitude_100": {"amplitude_prior": [math.log(100.0), 0.5]},
        "ei_learned_noise": {"fixed_noise": None},
        "ei_cover_only": {"data_fraction": 0.0},
        "ei_levels6": {"levels": 6},
        "ei_fourier256": {"fourier": 256},
        "ei_members4": {"members": 4},
        "ei_amplitude_prior": {"amplitude_prior": [math.log(4.0), 1.0]},
        "exact_gp_ei": {"surrogate": "exact"},
    },
    2: {
        "ucb_gp": {"acquisition": "ucb", "beta_schedule": 1.0},
        "ucb_gp_levels6": {"acquisition": "ucb", "beta_schedule": 1.0, "levels": 6},
        "ucb_gp_fourier256": {"acquisition": "ucb", "beta_schedule": 1.0, "fourier": 256},
        "portfolio_gp": {"acquisition": "portfolio", "beta_schedule": 1.0},
    },
}
STAGE_OBJECTIVES = {
    1: ("fan-2-1", "fan-3-3", "fan-4-2", "fan-5-3", "kink-0", "smooth-0", "well-0", "well-1"),
    2: (
        "fan-2-1",
        "fan-3-1",
        "fan-3-3",
        "fan-4-2",
        "fan-4-3",
        "fan-5-3",
        "kink-0",
        "kink-1",
        "smooth-0",
        "smooth-1",
        "well-0",
        "well-1",
    ),
}


def exact_gp_ei(objective, x_init: np.ndarray, cfg: dict, seed: int, iterations: int = ITERATIONS) -> dict:
    """Exact GP (BoTorch ``SingleTaskGP``, Matern-5/2 on R^3) with log expected improvement."""
    from botorch.acquisition import LogExpectedImprovement
    from botorch.exceptions import InputDataWarning
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    warnings.filterwarnings("ignore", category=InputDataWarning)
    torch.manual_seed(seed)
    x = np.asarray(x_init, dtype=np.float64)
    y = -np.asarray(objective(x), dtype=np.float64)
    fit_s, acq_s = [], []
    for it in range(iterations):
        t0 = time.perf_counter()
        ys = torch.as_tensor((y - y.mean()) / (y.std(ddof=1) + 1e-12)).unsqueeze(-1)
        model = SingleTaskGP(torch.as_tensor(x), ys, outcome_transform=None)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        ei = LogExpectedImprovement(model, best_f=ys.max())

        def acq(z: torch.Tensor, ei=ei) -> torch.Tensor:
            return ei(z.double().unsqueeze(1))

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


def run_one(config: str, overrides: dict, name: str, seed: int, out: str) -> str:
    torch.set_num_threads(1)
    path = Path(out) / "bo_validation" / f"{config}__{name}__{seed}.json"
    if path.exists():
        return str(path)
    objective = validation_objectives(Path(out) / "bo_validation_objectives.json")[name]
    rng = np.random.default_rng(seed)
    x_init = rng.standard_normal((N_INIT, 3))
    x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    cfg = load_config()
    surrogate = overrides.pop("surrogate", "rwf")
    cfg.update(overrides)
    rec = (exact_gp_ei if surrogate == "exact" else optimise)(objective, x_init, cfg, seed)
    best = np.minimum.accumulate(np.array(rec["f"]))
    record = {
        "config": config,
        "objective": name,
        "seed": seed,
        "settings": cfg,
        "f_min": objective.f_min,
        "scale": objective.scale,
        "regret": (best - objective.f_min).tolist(),
        **rec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return str(path)


def summarise(out: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted((out / "bo_validation").glob("*.json"))]
    table: dict[str, dict] = {}
    for config in sorted({r["config"] for r in runs}):
        rows = [r for r in runs if r["config"] == config]
        log_r = np.log10(
            np.maximum(np.array([r["regret"][N_INIT - 1 :] for r in rows]), 1e-12)
            / np.array([[r["scale"]] for r in rows])
        )
        table[config] = {
            "runs": len(rows),
            "objectives": sorted({r["objective"] for r in rows}),
            "median_log10_normalised_regret": {s: float(np.median(log_r[:, s])) for s in (50, 100, 150, 200)},
            "mean_log10_normalised_regret": {s: float(log_r[:, s].mean()) for s in (50, 100, 150, 200)},
            "runs_within_0.1_scale_at_200": int((log_r[:, 200] < -1.0).sum()),
            "seconds_per_iteration": float(
                np.mean([np.mean(r["fit_seconds"]) + np.mean(r["acq_seconds"]) for r in rows])
            ),
            "per_objective_median_at_200": {
                name: float(np.median([c[200] for c, r in zip(log_r, rows, strict=True) if r["objective"] == name]))
                for name in sorted({r["objective"] for r in rows})
            },
        }
    (out / "bo_validation.json").write_text(json.dumps(table, indent=1))
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sphere")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--objectives", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        validation_objectives(Path(args.out) / "bo_validation_objectives.json")
        configs = STAGES[args.stage]
        names = args.configs or list(configs)
        objectives = args.objectives or STAGE_OBJECTIVES[args.stage]
        jobs = [(c, dict(configs[c]), o, s) for c in names for o in objectives for s in range(args.seeds)]
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run_one, c, ov, o, s, args.out) for c, ov, o, s in jobs]
            for f in as_completed(futures):
                print(f.result(), flush=True)
    table = summarise(Path(args.out))
    print(f"{'config':22s} {'runs':>4s} {'med@100':>8s} {'med@200':>8s} {'mean@200':>9s} {'hits':>5s} {'s/it':>6s}")
    for config, row in table.items():
        m = row["median_log10_normalised_regret"]
        print(
            f"{config:22s} {row['runs']:4d} {m[100]:8.3f} {m[200]:8.3f} {row['mean_log10_normalised_regret'][200]:9.3f} "
            f"{row['runs_within_0.1_scale_at_200']:5d} {row['seconds_per_iteration']:6.2f}"
        )


if __name__ == "__main__":
    main()
