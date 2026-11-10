"""Wall-clock fit time and peak GPU memory versus training-set size on the multi-step function.

Every fit runs in a fresh process after a warm-up fit. Minibatch methods (SVGP, deep GP) get a fixed
budget of STEPS optimiser steps at every size, as RWF and RFF do; a method is dropped from larger sizes once a
fit exceeds the time budget or runs out of memory.

python experiments/synthetic/scalability.py --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

SIZES = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
STEPS, BATCH = 2000, 1024


def _make(name: str, seed: int, n: int):
    from srwf import RFFGP, RWFGP
    from srwf.baselines import SVGP, DeepGP, ExactGP

    epochs = max(1, math.ceil(STEPS * BATCH / n))
    return {
        "rwf": lambda: RWFGP(num_features=2048, levels=11, steps=500, seed=seed),
        "rff": lambda: RFFGP(num_features=2048, steps=500, seed=seed),
        "exact": lambda: ExactGP(kernel="rbf", steps=300),
        "svgp": lambda: SVGP(num_inducing=256, kernel="rbf", epochs=epochs, batch_size=BATCH, seed=seed),
        "dgp": lambda: DeepGP(hidden=4, num_inducing=128, epochs=epochs, batch_size=BATCH, seed=seed),
    }[name]()


def _run(name: str, n: int, seed: int, queue) -> None:
    import torch

    from srwf.data.synthetic import multi_step_dataset
    from srwf.metrics import rmse

    try:
        xw, yw, _, _ = multi_step_dataset(n_train=256, n_test=1, seed=seed + 1)
        _make(name, seed, STEPS * BATCH).fit(xw, yw)
        xtr, ytr, xte, yte = multi_step_dataset(n_train=n, n_test=2000, seed=seed)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = _make(name, seed, n).fit(xtr, ytr)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        err = rmse(yte, model.predict(xte).mean)
        queue.put(dict(status="ok", seconds=seconds, memory_gb=peak, rmse=float(err)))
    except (torch.OutOfMemoryError, RuntimeError) as e:
        queue.put(dict(status="oom" if "out of memory" in str(e).lower() else "failed", error=str(e)[:200]))


def measure(name: str, n: int, seed: int, budget: float) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run, args=(name, n, seed, queue))
    proc.start()
    proc.join(budget + 120.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return dict(status="timeout")
    if queue.empty():
        return dict(status="failed")
    out = queue.get()
    if out["status"] == "ok" and out["seconds"] > budget:
        out["status"] = "timeout"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rwf", "rff", "exact", "svgp", "dgp"])
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--budget", type=float, default=600.0, help="seconds per fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "scalability.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["model"], r["n"]) for r in rows}

    for name in args.models:
        for n in args.sizes:
            if (name, n) in done:
                if next(r for r in rows if (r["model"], r["n"]) == (name, n))["status"] != "ok":
                    break
                continue
            row = dict(model=name, n=n, **measure(name, n, args.seed, args.budget))
            rows.append(row)
            path.write_text(json.dumps(rows, indent=1))
            print(row, flush=True)
            if row["status"] != "ok":
                break

    for name in args.models:
        line = "  ".join(
            f"{r['n']:>7d}: {r['seconds']:7.1f}s {r['memory_gb']:6.2f}GB"
            if r["status"] == "ok"
            else f"{r['n']:>7d}: {r['status']}"
            for r in rows
            if r["model"] == name
        )
        print(f"{name:5s} {line}")


if __name__ == "__main__":
    main()
