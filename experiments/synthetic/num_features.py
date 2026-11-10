"""Test RMSE versus the number of random features on the multi-step function.

python experiments/synthetic/num_features.py --seeds 0 1 2 3 4 --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from srwf import RFFGP, RWFGP
from srwf.data.synthetic import multi_step_dataset
from srwf.metrics import evaluate

MODELS = {
    "rwf": lambda d, s: RWFGP(num_features=d, levels=11, wavelet="mexican_hat", steps=500, seed=s),
    "rff": lambda d, s: RFFGP(num_features=d, steps=500, seed=s),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    RWFGP(num_features=32, levels=2, steps=2).fit(np.zeros((4, 1)), np.arange(4.0))

    rows = []
    for seed in args.seeds:
        xtr, ytr, xte, yte = multi_step_dataset(seed=seed)
        for d in args.features:
            for name, make in MODELS.items():
                m = make(d, seed).fit(xtr, ytr)
                p = m.predict(xte)
                rows.append(
                    dict(
                        model=name,
                        num_features=d,
                        atoms=int(m.blr.features.num_features),
                        seed=seed,
                        seconds=m.info.seconds,
                        **evaluate(yte, p.mean, p.std),
                    )
                )
                print(rows[-1], flush=True)
            (args.out / "num_features.json").write_text(json.dumps(rows, indent=1))
    for name in MODELS:
        for d in args.features:
            r = [x["rmse"] for x in rows if x["model"] == name and x["num_features"] == d]
            print(f"{name:4s} D={d:5d}  rmse {np.mean(r):.4f} ± {np.std(r):.4f}")


if __name__ == "__main__":
    main()
