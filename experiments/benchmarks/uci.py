"""UCI regression benchmarks.

RWF configurations are selected per dataset on an inner validation split (10% of the training
rows of the first five splits); the selected configuration is then evaluated on all splits.

    python experiments/benchmarks/uci.py --datasets energy concrete --out results/uci
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from srwf import RFFGP, RWFGP
from srwf.data import uci
from srwf.metrics import evaluate

CANDIDATES: dict[str, dict] = {
    "rwf": dict(num_features=2048, levels=4),
    "rwf-deep": dict(num_features=2048, levels=6),
    "rwf+rff": dict(num_features=1536, levels=4, fourier=512),
    "rwf+rff-wide": dict(num_features=1536, levels=4, fourier=512, base_scale=4.0),
    "rwf+rff-4k": dict(num_features=3072, levels=4, fourier=1024),
    "rwf+rff-8k": dict(num_features=6144, levels=4, fourier=2048),
    "rwf+rff-loo": dict(num_features=1536, levels=4, fourier=512, base_scale=4.0, objective="loo"),
    "rwf-loo": dict(num_features=2048, levels=4, objective="loo"),
}


def make(name: str, cfg: dict, seed: int, steps: int):
    if name == "rff":
        return RFFGP(num_features=2048, steps=steps, seed=seed)
    return RWFGP(steps=steps, seed=seed, **cfg)


def select(dataset: str, steps: int, n_splits: int = 5) -> tuple[str, dict]:
    scores: dict[str, list[float]] = {k: [] for k in CANDIDATES}
    for k in range(n_splits):
        s = uci.split(dataset, k)
        rng = np.random.default_rng(1000 + k)
        perm = rng.permutation(len(s.y_train))
        n_val = max(1, len(perm) // 10)
        va, tr = perm[:n_val], perm[n_val:]
        for name, cfg in CANDIDATES.items():
            m = RWFGP(steps=steps, seed=k, **cfg).fit(s.x_train[tr], s.y_train[tr])
            p = m.predict(s.x_train[va])
            scores[name].append(evaluate(s.y_train[va], p.mean, p.std)["nll"])
    summary = {k: float(np.mean(v)) for k, v in scores.items()}
    return min(summary, key=summary.get), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(uci.DATASETS))
    ap.add_argument("--models", nargs="+", default=["rwf", "rff"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-splits", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/uci"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    RWFGP(num_features=32, levels=2, steps=2).fit(np.zeros((4, 1)), np.arange(4.0))

    for dataset in args.datasets:
        chosen, val = select(dataset, args.steps)
        print(f"{dataset}: selected {chosen} {json.dumps({k: round(v, 4) for k, v in val.items()})}", flush=True)
        rows = []
        for k in range(min(uci.n_splits(dataset), args.max_splits)):
            s = uci.split(dataset, k)
            for model in args.models:
                m = make(model, CANDIDATES[chosen], seed=k, steps=args.steps)
                t0 = time.perf_counter()
                m.fit(s.x_train, s.y_train)
                p = m.predict(s.x_test)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                row = dict(dataset=dataset, model=model, split=k, seconds=time.perf_counter() - t0)
                row.update(evaluate(s.y_test, p.mean, p.std))
                if model == "rwf":
                    row["config"] = chosen
                rows.append(row)
        (args.out / f"{dataset}.json").write_text(
            json.dumps({"selection": val, "chosen": chosen, "runs": rows}, indent=1)
        )
        for model in args.models:
            r = [x for x in rows if x["model"] == model]
            agg = {
                k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]))
                for k in ("rmse", "nll", "crps", "cov95", "seconds")
            }
            print(f"  {model:4s} " + "  ".join(f"{k} {m:.4f}±{sd:.4f}" for k, (m, sd) in agg.items()), flush=True)


if __name__ == "__main__":
    main()
