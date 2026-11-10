"""Sparse-station reconstruction of global ERA5 fields on the sphere.

Each run observes one 6-hourly field at an area-weighted random 25% of the 4608 cells of the
T30 grid and predicts the other 75%. Inputs are the sphere embedding ``(x, y, z)`` plus
orography and a land flag; scores are area weighted. Snapshots: 8 per year, 45 days apart from
15 January, 2004 for validation (``--split val``) and 2005 for test (``--split test``).

    python experiments/climate/select.py                       # choose model settings on 2004
    python experiments/climate/reconstruct.py --split test     # frozen evaluation on 2005

The test run also writes the full-grid fields of snapshots ``--showcase-at`` (station seed 0)
to ``<out>/showcase/``.

Per-run scores are cached in ``<out>/<tag>_<split>/<model>__<target>.csv`` together with the model
setting; runs whose setting is unchanged are not repeated, so models, targets and snapshot
positions can be run by separate processes and summarised by a final call.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from common import build_model, fit_predict, sphere_centres, weighted_scores

from srwf.data import era5

SPLITS = {"val": "2004-01-15T00", "test": "2005-01-15T00"}
TARGETS = ("temperature", "precipitation", "surface_pressure", "humidity", "u_wind", "v_wind")
METRICS = ("rmse", "mae", "nll", "crps", "cov95", "seconds")

RWF = dict(num_features=8192, levels=4, centres="sphere", anchors=0, steps=300, lr=0.05)
DEFAULTS: dict[str, tuple[str, dict]] = {
    "rwf": ("rwf", RWF),
    "rwf-adaptive": ("rwf", {**RWF, "anchors": 16}),
    "rff": ("rff", dict(num_features=8192, steps=300, lr=0.05)),
    "svgp": ("svgp", dict(num_inducing=512, epochs=500, batch_size=1024, lr=0.01)),
    "exact": ("exact", dict(kernel="matern2.5", steps=200, lr=0.05)),
}


def resolve(cfg: dict, grid: era5.Grid, covariates: bool) -> dict:
    """Turn symbolic centre choices into arrays: ``sphere`` (Fibonacci lattice) or ``grid``."""
    cfg = dict(cfg)
    centres = cfg.get("centres")
    n = int(cfg.pop("sphere_points", 100000))
    if centres == "sphere":
        cfg["centres"] = sphere_centres(n, grid, covariates)
    elif centres == "grid":
        cfg["centres"] = grid.inputs(None, covariates)
    return cfg


def load_models(path: Path | None, names: list[str], targets: list[str]) -> dict[str, dict[str, tuple[str, dict]]]:
    """Per-target model settings: the validation-selected ones from ``path``, else the defaults."""
    chosen = json.loads(path.read_text()) if path is not None else {}
    out = {}
    for target in targets:
        models = dict(DEFAULTS)
        for fam, per_target in chosen.items():
            sel = per_target.get(target) or per_target.get("temperature")
            models[fam] = (sel["kind"], sel["config"])
        out[target] = {k: models[k] for k in names}
    return out


@contextmanager
def locked(path: Path):
    """Exclusive lock on the directory of ``path``, shared by concurrent runs writing there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def save_showcase(
    root: Path,
    grid: era5.Grid,
    target: str,
    step: int,
    truth: np.ndarray,
    stations: np.ndarray,
    seed: int,
    fields: dict,
) -> None:
    """Full-grid truth, station mask and per-model predictive mean / std; merges into an existing file."""
    shape = (era5.N_LAT, era5.N_LON)
    stamp = str(era5.step_time(step))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[stations] = True
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{target}_{stamp[:13]}.npz"
    with locked(path):
        arrays = dict(np.load(path)) if path.exists() else {}
        arrays.update(
            lat=grid.lat,
            lon=grid.lon,
            truth=truth.reshape(shape),
            stations=stations,
            station_mask=mask.reshape(shape),
            time=stamp,
            units=era5.UNITS[target],
            seed=seed,
            **{f"{k}_mean": p.mean.reshape(shape) for k, p in fields.items()},
            **{f"{k}_std": p.std.reshape(shape) for k, p in fields.items()},
        )
        np.savez_compressed(path, **arrays)


def run(
    split: str,
    targets: list[str],
    models: dict[str, dict[str, tuple[str, dict]]],
    seeds: list[int],
    count: int = 8,
    fraction: float = 0.25,
    covariates: bool = True,
    showcase: Path | None = None,
    showcase_at: tuple[int, ...] = (0, 4),
    positions: list[int] | None = None,
    log: bool = True,
) -> pd.DataFrame:
    grid = era5.load_grid()
    steps = era5.snapshot_steps(SPLITS[split], count, 45)
    rows = []
    for target in targets:
        resolved = {k: (kind, resolve(cfg, grid, covariates)) for k, (kind, cfg) in models[target].items()}
        for i, step in enumerate(steps):
            if positions is not None and i not in positions:
                continue
            truth = era5.field(target, step)
            for seed in seeds:
                stations, held_out = era5.sample_stations(grid, fraction, seed)
                x_tr, x_te = grid.inputs(stations, covariates), grid.inputs(None, covariates)
                fields = {}
                for name, (kind, cfg) in resolved.items():
                    pred, seconds = fit_predict(build_model(kind, cfg, seed), x_tr, truth[stations], x_te)
                    s = weighted_scores(truth[held_out], pred.mean[held_out], pred.std[held_out], grid.area[held_out])
                    rows.append(
                        dict(
                            target=target,
                            step=int(step),
                            time=str(era5.step_time(step)),
                            seed=seed,
                            model=name,
                            seconds=seconds,
                            **s,
                        )
                    )
                    fields[name] = pred
                    if log:
                        print(
                            f"{target:14s} {str(era5.step_time(step))} seed {seed} {name:13s} "
                            + " ".join(f"{k} {s[k]:.3f}" for k in METRICS[:-1])
                            + f" {seconds:.1f}s",
                            flush=True,
                        )
                if showcase is not None and i in showcase_at and seed == seeds[0]:
                    save_showcase(showcase, grid, target, step, truth, stations, seed, fields)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["target", "model"], sort=False)[list(METRICS)]
    out = g.mean().round(4).join(g.std().round(4), rsuffix="_std")
    return out[[c for m in METRICS for c in (m, f"{m}_std")]]


def cached_run(
    cache: Path,
    split: str,
    target: str,
    name: str,
    spec: tuple[str, dict],
    seeds: list[int],
    count: int,
    positions: list[int] | None,
    **kw,
) -> pd.DataFrame:
    """Scores of ``name`` on ``target``, running only the (snapshot, seed) pairs not yet cached."""
    path = cache / f"{name}__{target}.csv"
    setting = json.dumps({"kind": spec[0], "config": spec[1]}, sort_keys=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["target", "step", "seed", "setting"])
    valid = old[(old["target"] == target) & (old["setting"] == setting)]
    steps = era5.snapshot_steps(SPLITS[split], count, 45)
    done = {(int(a), int(b)) for a, b in zip(valid["step"], valid["seed"], strict=True)}
    wanted = range(count) if positions is None else positions
    missing = sorted({i for i in wanted for sd in seeds if (int(steps[i]), sd) not in done})
    for i in missing:
        df = run(split, [target], {target: {name: spec}}, seeds, count, positions=[i], **kw).assign(setting=setting)
        with locked(path):
            old = pd.read_csv(path) if path.exists() else old
            key = set(zip(df["step"], df["seed"], strict=True))
            stale = np.array([(a, b) in key for a, b in zip(old["step"], old["seed"], strict=True)], bool)
            old = pd.concat([old[~stale], df], ignore_index=True)
            old.to_csv(path, index=False)
    keep = old[(old["target"] == target) & (old["setting"] == setting)]
    return keep[keep["step"].isin(steps[list(wanted)]) & keep["seed"].isin(seeds)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULTS))
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("selected.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--positions", nargs="+", type=int, default=None, help="run only these snapshot positions")
    ap.add_argument("--showcase-at", nargs="+", type=int, default=[0, 4], help="positions saved as full fields")
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--no-covariates", action="store_true")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--out", type=Path, default=Path("results/climate"))
    args = ap.parse_args()

    config = args.config if args.config.exists() else None
    models = load_models(config, args.models, args.targets)
    print(f"model settings from {config or 'defaults'}", flush=True)
    stem = f"{args.tag}_{args.split}"
    cache = args.out / stem
    cache.mkdir(parents=True, exist_ok=True)
    frames = [
        cached_run(
            cache,
            args.split,
            target,
            name,
            spec,
            args.seeds,
            args.count,
            args.positions,
            fraction=args.fraction,
            covariates=not args.no_covariates,
            showcase=args.out / "showcase",
            showcase_at=tuple(args.showcase_at),
        )
        for target in args.targets
        for name, spec in models[target].items()
    ]
    if args.positions is not None:
        return
    df = pd.concat(frames, ignore_index=True).drop(columns="setting")
    df.to_csv(args.out / f"{stem}_runs.csv", index=False)
    summary = summarise(df)
    summary.to_csv(args.out / f"{stem}_summary.csv")
    (args.out / f"{stem}_models.json").write_text(
        json.dumps(
            {t: {k: {"kind": kind, "config": cfg} for k, (kind, cfg) in m.items()} for t, m in models.items()}, indent=1
        )
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
