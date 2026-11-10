"""Choose the model settings of the reconstruction experiment on 2004 validation snapshots.

Every candidate is run on the eight 2004 validation snapshots (station draw ``--seeds``) and its
per-run scores are cached in ``<out>/selection/<family>__<candidate>.csv``; cached runs are not
repeated. For every target, each family keeps the candidate with the lowest mean validation
CRPS. The 2005 test snapshots are never read. The chosen settings are written to
``<out>/selected.json``; the released ``experiments/climate/selected.json`` is the result, which
``reconstruct.py`` reads by default.

    python experiments/climate/select.py                                    # all candidates
    python experiments/climate/select.py --candidates 'rwf/D48k*' --targets v_wind
"""

from __future__ import annotations

import argparse
import json
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
from reconstruct import SPLITS, TARGETS, locked, run

from srwf.data import era5

RWF = dict(steps=300, lr=0.05, centres="sphere", solver="dual")
RFF = dict(steps=300, lr=0.05)
SVGP = dict(epochs=1000, batch_size=1024, lr=0.01)


def rwf(d: int, levels: int, **kw) -> tuple[str, dict]:
    return "rwf", {**RWF, "num_features": d, "levels": levels, **kw}


CANDIDATES: dict[str, dict[str, tuple[str, dict]]] = {
    "rwf": {
        "D16k-J4-data": rwf(16384, 4, centres="data"),
        "D24k-J4-sphere": rwf(24576, 4),
        "D24k-J5-sphere": rwf(24576, 5),
        "D32k-J4-sphere": rwf(32768, 4),
        "D48k-J4-sphere": rwf(49152, 4),
        "D24k-J4-sphere-x4": rwf(24576, 4, members=4),
        "D24k-J4-sphere-g1": rwf(24576, 4, growth=1),
        "D24k-J4-sphere-g1.5": rwf(24576, 4, growth=1.5),
        "D48k-J4-sphere-g1": rwf(49152, 4, growth=1),
        "D24k-J4-sphere-r3": rwf(24576, 4, restarts=3),
        "D24k-J4-sphere-g1-r3": rwf(24576, 4, growth=1, restarts=3),
    },
    "rwf-adaptive": {
        "D16k-J4-sphere-a16": rwf(16384, 4, anchors=16),
        "D24k-J4-sphere-a16": rwf(24576, 4, anchors=16),
        "D24k-J4-sphere-a32": rwf(24576, 4, anchors=32),
        "D24k-J5-sphere-a16": rwf(24576, 5, anchors=16),
        "D24k-J5-sphere-a16-g3": rwf(24576, 5, anchors=16, growth=3),
        "D24k-J4-sphere-a64": rwf(24576, 4, anchors=64),
        "D24k-J4-sphere-a32-p0.3": rwf(24576, 4, anchors=32, anchor_prior=0.3),
        "D24k-J4-sphere-a32-p3": rwf(24576, 4, anchors=32, anchor_prior=3.0),
        "D48k-J4-sphere-a32": rwf(49152, 4, anchors=32),
        "D24k-J4-sphere-a128": rwf(24576, 4, anchors=128),
        "D48k-J4-sphere-a64": rwf(49152, 4, anchors=64),
        "D24k-J4-sphere-a32-r3": rwf(24576, 4, anchors=32, restarts=3),
    },
    "rff": {
        "D8k": ("rff", {**RFF, "num_features": 8192}),
        "D16k": ("rff", {**RFF, "num_features": 16384}),
        "D24k": ("rff", {**RFF, "num_features": 24576}),
    },
    "svgp": {
        "M512": ("svgp", {**SVGP, "num_inducing": 512}),
        "M1024": ("svgp", {**SVGP, "num_inducing": 1024}),
    },
    "exact": {
        "matern2.5": ("exact", dict(kernel="matern2.5", steps=200, lr=0.05)),
    },
}


def cache_path(out: Path, key: str) -> Path:
    return out / "selection" / (key.replace("/", "__") + ".csv")


def evaluate(
    key: str, spec: tuple[str, dict], targets: list[str], seeds: list[int], count: int, out: Path, force: bool
) -> None:
    """Run ``key`` on the snapshots and seeds missing from its cache."""
    path = cache_path(out, key)
    old = pd.read_csv(path) if path.exists() and not force else pd.DataFrame(columns=["target", "step", "seed"])
    steps = era5.snapshot_steps(SPLITS["val"], count, 45)
    for target in targets:
        rows = old[old["target"] == target]
        done = {(int(a), int(b)) for a, b in zip(rows["step"], rows["seed"], strict=True)}
        missing = sorted({i for i, st in enumerate(steps) for sd in seeds if (int(st), sd) not in done})
        if not missing:
            continue
        df = run("val", [target], {target: {key: spec}}, seeds, count, positions=missing)
        with locked(path):
            old = pd.read_csv(path) if path.exists() else old
            fresh = set(zip(df["step"], df["seed"], strict=True))
            pairs = zip(old["step"], old["seed"], strict=True)
            stale = (old["target"] == target) & np.array([(a, b) in fresh for a, b in pairs], bool)
            old = pd.concat([old[~stale], df], ignore_index=True)
            old.to_csv(path, index=False)


def collect(out: Path, models: dict[str, tuple[str, dict]], seeds: list[int], count: int) -> pd.DataFrame:
    """Cached runs on the requested snapshots and seeds, keeping only complete (target, model) cells."""
    frames = [pd.read_csv(cache_path(out, k)) for k in models if cache_path(out, k).exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["seed"].isin(seeds) & df["step"].isin(era5.snapshot_steps(SPLITS["val"], count, 45))]
    size = df.groupby(["target", "model"])["rmse"].transform("size")
    return df[size == count * len(seeds)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--families", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--candidates", nargs="+", default=["*"], help="glob patterns on 'family/candidate'")
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-run cached candidates")
    ap.add_argument("--out", type=Path, default=Path("results/climate"))
    args = ap.parse_args()

    models = {f"{fam}/{name}": spec for fam in CANDIDATES for name, spec in CANDIDATES[fam].items()}
    chosen = {
        k: v
        for k, v in models.items()
        if k.split("/")[0] in args.families and any(fnmatch(k, p) for p in args.candidates)
    }
    for key, spec in chosen.items():
        evaluate(key, spec, args.targets, args.seeds, args.count, args.out, args.force)

    df = collect(args.out, models, args.seeds, args.count)
    df.to_csv(args.out / "selection_runs.csv", index=False)
    table = df.groupby(["target", "model"])[["rmse", "mae", "nll", "crps", "cov95", "seconds"]].mean()
    table.to_csv(args.out / "selection.csv")
    print(table.round(4).to_string())

    selected = {}
    for fam in CANDIDATES:
        for target in TARGETS:
            if target not in table.index.get_level_values("target"):
                continue
            crps = table.loc[target, "crps"]
            names = [k for k in crps.index if k.split("/")[0] == fam]
            if not names:
                continue
            best = crps[names].idxmin()
            kind, cfg = models[best]
            selected.setdefault(fam, {})[target] = {
                "kind": kind,
                "config": cfg,
                "candidate": best.split("/", 1)[1],
                "val_crps": float(crps[best]),
            }
            print(f"{target} {fam}: {best} (CRPS {crps[best]:.4f})")
    (args.out / "selected.json").write_text(json.dumps(selected, indent=1))


if __name__ == "__main__":
    main()
