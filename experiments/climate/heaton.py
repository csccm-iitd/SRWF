"""Heaton et al. (2019) MODIS land-surface-temperature benchmark.

One satellite image on a 500 x 300 lattice: 105,569 observed training pixels and 42,740 test
pixels hidden by a cloud mask (``AllSatelliteTemps.RData`` of the competition repository).
Inputs are (lon, lat); the scores are those of the paper's Table 3 (MAE, RMSE, CRPS, 95%
interval score, 95% coverage). Settings were chosen with ``--validate``, which holds out
20 x 20-pixel tiles of the training image; the test pixels are used only for the final scores.

    python experiments/climate/heaton.py --validate --models rwf --out results/heaton/val
    python experiments/climate/heaton.py --out results/heaton
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import build_model, fit_predict, weighted_scores

URL = "https://raw.githubusercontent.com/finnlindgren/heatoncomparison/master/Data/AllSatelliteTemps.RData"
ROOT = Path(__file__).resolve().parents[2] / "data" / "heaton"

PUBLISHED = {
    "FRK": (1.96, 2.44, 1.44, 14.08, 0.79, 2.32),
    "Gapfill": (1.33, 1.86, 1.17, 34.78, 0.36, 1.39),
    "LatticeKrig": (1.22, 1.68, 0.87, 7.55, 0.96, 27.92),
    "LAGP": (1.65, 2.08, 1.17, 10.81, 0.83, 2.27),
    "Metakriging": (2.08, 2.50, 1.44, 10.77, 0.89, 2888.52),
    "MRA": (1.33, 1.85, 0.94, 8.00, 0.92, 15.61),
    "NNGP": (1.21, 1.64, 0.85, 7.57, 0.95, 2.06),
    "Partition": (1.41, 1.80, 1.02, 10.49, 0.86, 79.98),
    "Pred. Proc.": (2.15, 2.64, 1.55, 15.51, 0.83, 160.24),
    "SPDE": (1.10, 1.53, 0.83, 8.85, 0.97, 120.33),
    "Tapering": (1.87, 2.45, 1.32, 10.31, 0.93, 133.26),
    "Periodic Embedding": (1.29, 1.79, 0.91, 7.44, 0.93, 9.81),
}


def load(root: Path = ROOT) -> dict[str, np.ndarray]:
    """``lonlat`` of all 150,000 pixels and the train / test rows of the official split."""
    cache = root / "heaton_satellite.npz"
    if not cache.exists():
        raw = root / "AllSatelliteTemps.RData"
        if not raw.exists():
            root.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(URL, timeout=120) as r:
                raw.write_bytes(r.read())
        import pyreadr

        df = pyreadr.read_r(str(raw))["all.sat.temps"]
        np.savez(
            cache,
            lonlat=df[["Lon", "Lat"]].to_numpy(np.float64),
            masked=df["MaskTemp"].to_numpy(np.float64),
            truth=df["TrueTemp"].to_numpy(np.float64),
        )
    d = np.load(cache)
    train = ~np.isnan(d["masked"])
    test = np.isnan(d["masked"]) & ~np.isnan(d["truth"])
    return {
        "lonlat": d["lonlat"],
        "x_train": d["lonlat"][train],
        "y_train": d["masked"][train],
        "x_test": d["lonlat"][test],
        "y_test": d["truth"][test],
    }


def validation_split(
    d: dict[str, np.ndarray], tile: int = 20, fraction: float = 0.15, seed: int = 0
) -> dict[str, np.ndarray]:
    """Hold out the training pixels of a random ``fraction`` of ``tile x tile`` lattice tiles,
    mimicking the contiguous cloud gaps of the test set with training data only."""
    lon, lat = np.unique(d["lonlat"][:, 0]), np.unique(d["lonlat"][:, 1])
    col = np.searchsorted(lon, d["x_train"][:, 0]) // tile
    row = np.searchsorted(lat, d["x_train"][:, 1]) // tile
    tiles = row * (len(lon) // tile + 1) + col
    ids = np.unique(tiles)
    held = np.random.default_rng(seed).choice(ids, int(fraction * len(ids)), replace=False)
    va = np.isin(tiles, held)
    x, y = d["x_train"], d["y_train"]
    return {**d, "x_train": x[~va], "y_train": y[~va], "x_test": x[va], "y_test": y[va]}


def configs(args: argparse.Namespace, lattice: np.ndarray) -> dict[str, tuple[str, dict]]:
    rwf = dict(
        num_features=args.features,
        levels=args.levels,
        growth=args.growth,
        fourier=args.fourier,
        centres=lattice if args.centres == "lattice" else args.centres,
        anchors=args.anchors,
        steps=args.steps,
        lr=args.lr,
    )
    return {
        "rwf": ("rwf", {**rwf, "subsample": args.subsample}),
        "rff": ("rff", dict(num_features=args.features, subsample=args.subsample, steps=args.steps, lr=args.lr)),
        "svgp": ("svgp", dict(num_inducing=args.inducing, epochs=args.epochs, batch_size=4096, lr=0.01)),
    }


def ensemble(out: Path, stem: str, seeds: list[int], y: np.ndarray) -> dict:
    """Equal-weight mixture of the per-seed predictive distributions (independent feature draws)."""
    preds = [np.load(out / f"{stem}_s{k}_pred.npz") for k in seeds]
    rows = [json.loads((out / f"{stem}_s{k}.json").read_text()) for k in seeds]
    m = np.mean([p["mean"] for p in preds], axis=0)
    v = np.mean([p["std"] ** 2 + p["mean"] ** 2 for p in preds], axis=0) - m**2
    seconds = float(sum(r["seconds"] for r in rows))
    name = f"{rows[0]['model']}-ens{len(seeds)}"
    row = {**rows[0], "model": name, "seed": seeds, "seconds": seconds, "minutes": seconds / 60}
    row.update(weighted_scores(y, m, np.sqrt(np.maximum(v, 1e-12))))
    (out / f"{stem}_ens.json").write_text(json.dumps(row, indent=1))
    return row


def summarise(out: Path) -> pd.DataFrame:
    """Mean and std over seeds of every test run in ``out``, followed by the published Table 3."""
    rows = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    df = pd.DataFrame([r for r in rows if r.get("split") == "test"])
    cols = ["mae", "rmse", "crps", "int95", "cov95", "minutes"]
    g = df.groupby("model")[cols]
    ours = g.mean().join(g.std().add_suffix("_std")).assign(n_seeds=g.size())
    pub = pd.DataFrame.from_dict(PUBLISHED, orient="index", columns=cols)
    table = pd.concat([ours, pub]).sort_values("rmse")
    table.to_csv(out / "summary.csv")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rwf", "svgp", "rff"])
    ap.add_argument("--features", type=int, default=49152)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--growth", type=float, default=4.0)
    ap.add_argument("--fourier", type=int, default=0)
    ap.add_argument("--centres", default="lattice", help="lattice (all 150k pixel locations), data or box")
    ap.add_argument("--anchors", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=8192)
    ap.add_argument("--validate", action="store_true", help="score on held-out training tiles instead of the test set")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--inducing", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ensemble", nargs="*", default=["rwf"], help="models whose seeds are also pooled into a mixture")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=Path("results/heaton"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load()
    if args.validate:
        d = validation_split(d)
    cfgs = configs(args, d["lonlat"])
    for name in args.models:
        for seed in args.seeds:
            kind, cfg = cfgs[name]
            torch.manual_seed(seed)
            model = build_model(kind, cfg, seed)
            pred, seconds = fit_predict(model, d["x_train"], d["y_train"], d["x_test"])
            s = weighted_scores(d["y_test"], pred.mean, pred.std)
            row = dict(
                model=name,
                tag=args.tag,
                seed=seed,
                split="val" if args.validate else "test",
                seconds=seconds,
                minutes=seconds / 60,
                **s,
            )
            row["config"] = {k: (v if not isinstance(v, np.ndarray) else "lattice") for k, v in cfg.items()}
            if kind != "svgp":
                row["noise_var"] = model.noise_var
            stem = f"{'val_' if args.validate else ''}{name}{'_' + args.tag if args.tag else ''}_s{seed}"
            (args.out / f"{stem}.json").write_text(json.dumps(row, indent=1))
            np.savez_compressed(args.out / f"{stem}_pred.npz", mean=pred.mean, std=pred.std)
            print(
                f"{stem:28s} MAE {s['mae']:.3f} RMSE {s['rmse']:.3f} CRPS {s['crps']:.3f} INT {s['int95']:.2f} "
                f"CVG {s['cov95']:.3f} time {seconds / 60:.2f} min",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        if name in args.ensemble and len(args.seeds) > 1:
            stem = f"{'val_' if args.validate else ''}{name}{'_' + args.tag if args.tag else ''}"
            r = ensemble(args.out, stem, args.seeds, d["y_test"])
            print(
                f"{stem + '_ens':28s} MAE {r['mae']:.3f} RMSE {r['rmse']:.3f} CRPS {r['crps']:.3f} "
                f"INT {r['int95']:.2f} CVG {r['cov95']:.3f} time {r['minutes']:.2f} min",
                flush=True,
            )
    if not args.validate:
        print(summarise(args.out).round(3).to_string())


if __name__ == "__main__":
    main()
