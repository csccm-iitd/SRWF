"""Multi-step function benchmark: all models on the same seeded data sets.

python experiments/synthetic/multi_step.py --seeds 0 1 2 3 4 --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from srwf import RFFGP, RWFGP
from srwf.baselines import SVGP, DeepGP, ExactGP
from srwf.data.synthetic import multi_step_dataset
from srwf.metrics import evaluate

MODELS = {
    "exact": lambda s: ExactGP(kernel="rbf", steps=300),
    "svgp": lambda s: SVGP(num_inducing=256, kernel="rbf", epochs=200, batch_size=512, seed=s),
    "dgp": lambda s: DeepGP(hidden=4, num_inducing=128, epochs=200, seed=s),
    "rff": lambda s: RFFGP(num_features=2048, steps=500, seed=s),
    "rwf": lambda s: RWFGP(num_features=2048, levels=11, steps=500, seed=s),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    RWFGP(num_features=32, levels=2, steps=2).fit(np.zeros((4, 1)), np.arange(4.0))

    rows, grid = [], np.linspace(0, 1, 2000)[:, None]
    curves = {}
    for seed in args.seeds:
        xtr, ytr, xte, yte = multi_step_dataset(seed=seed)
        for name in args.models:
            m = MODELS[name](seed).fit(xtr, ytr)
            p = m.predict(xte)
            sec = m.info.seconds if hasattr(m, "info") else m.seconds
            rows.append(dict(model=name, seed=seed, seconds=sec, **evaluate(yte, p.mean, p.std)))
            if seed == args.seeds[0]:
                g = m.predict(grid)
                curves[name] = np.stack([g.mean, g.std])
            print(rows[-1], flush=True)
    (args.out / "multi_step.json").write_text(json.dumps(rows, indent=1))
    xtr, ytr, _, _ = multi_step_dataset(seed=args.seeds[0])
    np.savez(args.out / "multi_step_curves.npz", grid=grid[:, 0], x_train=xtr[:, 0], y_train=ytr, **curves)
    for name in args.models:
        r = [x for x in rows if x["model"] == name]
        print(
            name,
            "  ".join(
                f"{k} {np.mean([x[k] for x in r]):.4f}±{np.std([x[k] for x in r]):.4f}"
                for k in ("rmse", "crps", "nll", "cov95", "seconds")
            ),
        )


if __name__ == "__main__":
    main()
