"""Non-stationary kernel baselines (Gibbs and deep kernel, exact and sparse variational) against RWF.

Learning rate and optimiser steps of every baseline are chosen by held-out NLL on an 80/20 split of an
independent draw (seed 100); the selected settings are refit on the evaluation seeds of multi_step.py
and sample_size.py. Finished tuning trials stored in the output file are reused. The scaling task fits
each size in a fresh process under a GPU memory cap.

python experiments/synthetic/nonstationary_baselines.py --tasks tune multi_step burst scaling --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASELINES = ["exact-gibbs", "svgp-gibbs", "exact-dkl", "svgp-dkl"]
GRID = {
    "exact": [dict(lr=lr, steps=s) for lr in (0.01, 0.03, 0.1, 0.3) for s in (300, 1000, 3000)],
    "svgp": [dict(lr=lr, steps=s) for lr in (0.001, 0.003, 0.01, 0.03) for s in (2000, 6000, 15000)],
}
RWF = {
    "multi_step": {"rwf": dict(num_features=2048, levels=11, steps=500)},
    "burst": {
        "rwf": dict(num_features=1024, levels=9, steps=500),
        "rwf-adaptive": dict(num_features=1024, levels=9, anchors=16, steps=500),
    },
}
TASKS = ["multi_step", "burst-256", "burst-1024"]
INDUCING, BATCH, TUNE_SEED = 256, 512, 100
SCALING_SIZES = [1000, 4000, 16000, 64000]


def make(name: str, cfg: dict, n: int, seed: int):
    from srwf import RWFGP
    from srwf.baselines import SVGP, ExactGP

    if name.startswith("rwf"):
        return RWFGP(**cfg, seed=seed)
    kind, kernel = name.split("-")
    if kind == "exact":
        return ExactGP(kernel=kernel, steps=cfg["steps"], lr=cfg["lr"], seed=seed)
    epochs = max(1, round(cfg["steps"] / math.ceil(n / BATCH)))
    return SVGP(num_inducing=INDUCING, kernel=kernel, epochs=epochs, batch_size=BATCH, lr=cfg["lr"], seed=seed)


def dataset(task: str, seed: int, n: int | None = None):
    from srwf.data.synthetic import multi_step_dataset, sample

    if task == "multi_step":
        return multi_step_dataset(seed=seed) if n is None else multi_step_dataset(n_train=n, n_test=2000, seed=seed)
    return sample("burst", int(task.split("-")[1]), seed=seed)


def score(model, x, y, f=None) -> dict:
    from srwf.metrics import evaluate, rmse

    p = model.predict(x)
    out = evaluate(y, p.mean, p.std)
    if f is not None:
        out["rmse_f"] = rmse(f(x[:, 0]), p.mean)
    out["seconds"] = model.info.seconds if hasattr(model, "info") else model.seconds
    return out


def fit_and_score(name, cfg, xtr, ytr, xte, yte, seed, f=None) -> dict:
    try:
        return score(make(name, cfg, len(xtr), seed).fit(xtr, ytr), xte, yte, f)
    except Exception as e:
        return dict(nll=float("nan"), error=f"{type(e).__name__}: {str(e)[:160]}")


def tune(results: dict, save) -> None:
    rows, selected = results.setdefault("tuning", []), results.setdefault("selected", {})
    for task in TASKS:
        x, y, _, _ = dataset(task, TUNE_SEED)
        perm = np.random.default_rng(TUNE_SEED).permutation(len(x))
        tr, va = perm[: int(0.8 * len(x))], perm[int(0.8 * len(x)) :]
        for name in BASELINES:
            trials = []
            for cfg in GRID[name.split("-")[0]]:
                key = dict(task=task, model=name, **cfg)
                done = [r for r in rows if key.items() <= r.items()]
                if done:
                    trials.append(done[0])
                    continue
                row = dict(**key, **fit_and_score(name, cfg, x[tr], y[tr], x[va], y[va], 0))
                rows.append(row)
                trials.append(row)
                save()
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
            best = min(trials, key=lambda r: r["nll"] if np.isfinite(r["nll"]) else np.inf)
            selected.setdefault(task, {})[name] = dict(lr=best["lr"], steps=best["steps"])
            save()


def evaluate_task(results: dict, save, task: str, seeds: list[int]) -> None:
    from srwf.data.synthetic import burst

    rows = results.setdefault(task.split("-")[0], [])
    rows[:] = [r for r in rows if r.get("task", task) != task]
    configs = {**results["selected"][task], **RWF[task.split("-")[0]]}
    for seed in seeds:
        xtr, ytr, xte, yte = dataset(task, seed)
        for name, cfg in configs.items():
            out = fit_and_score(name, cfg, xtr, ytr, xte, yte, seed, None if task == "multi_step" else burst)
            rows.append(dict(task=task, model=name, seed=seed, **out))
            save()
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)


def _scaling_run(name: str, cfg: dict, n: int, cap_gb: float, queue) -> None:
    import torch

    from srwf.metrics import rmse

    try:
        torch.cuda.set_per_process_memory_fraction(cap_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        xw, yw, _, _ = dataset("multi_step", 1, 256)
        make(name, cfg, 256, 0).fit(xw, yw)
        xtr, ytr, xte, yte = dataset("multi_step", 0, n)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = make(name, cfg, n, 0).fit(xtr, ytr)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        queue.put(dict(status="ok", seconds=seconds, memory_gb=peak, rmse=rmse(yte, model.predict(xte).mean)))
    except Exception as e:
        oom = "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError)
        queue.put(dict(status="oom" if oom else "failed", error=f"{type(e).__name__}: {str(e)[:160]}"))


def scaling(results: dict, save, cap_gb: float, budget: float) -> None:
    rows = results["scaling"] = []
    configs = {
        "exact-gibbs": results["selected"]["multi_step"]["exact-gibbs"],
        "svgp-gibbs": results["selected"]["multi_step"]["svgp-gibbs"],
        "rwf": RWF["multi_step"]["rwf"],
    }
    ctx = mp.get_context("spawn")
    for name, cfg in configs.items():
        for n in SCALING_SIZES:
            queue = ctx.Queue()
            proc = ctx.Process(target=_scaling_run, args=(name, cfg, n, cap_gb, queue))
            proc.start()
            proc.join(budget + 300.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            out = queue.get() if not queue.empty() else dict(status="timeout" if proc.exitcode is None else "failed")
            if out["status"] == "ok" and out["seconds"] > budget:
                out["status"] = "timeout"
            rows.append(dict(model=name, n=n, memory_cap_gb=cap_gb, **out))
            save()
            print(rows[-1], flush=True)
            if out["status"] != "ok":
                break


def summarise(results: dict) -> None:
    for task, keys in [("multi_step", ("rmse", "nll", "crps", "seconds")), ("burst", ("rmse_f", "nll", "seconds"))]:
        rows = [r for r in results.get(task, []) if "error" not in r]
        for t in sorted({r["task"] for r in rows}, key=TASKS.index):
            for name in dict.fromkeys(r["model"] for r in rows if r["task"] == t):
                r = [x for x in rows if (x["task"], x["model"]) == (t, name)]
                cells = "  ".join(f"{k} {np.mean([x[k] for x in r]):.4f}±{np.std([x[k] for x in r]):.4f}" for k in keys)
                print(f"{t:10s} {name:12s} {cells}")
    for r in results.get("scaling", []):
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["tune", "multi_step", "burst", "scaling"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--memory-cap", type=float, default=32.0, help="GiB available to each scaling fit")
    ap.add_argument("--budget", type=float, default=7200.0, help="seconds per scaling fit")
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "nonstationary_baselines.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    def save() -> None:
        path.write_text(json.dumps(results, indent=1))

    from srwf import RWFGP

    RWFGP(num_features=32, levels=2, steps=2).fit(np.zeros((4, 1)), np.arange(4.0))
    if "tune" in args.tasks:
        results["selected"] = {}
        tune(results, save)
    if "multi_step" in args.tasks:
        evaluate_task(results, save, "multi_step", args.seeds)
    if "burst" in args.tasks:
        for task in ("burst-256", "burst-1024"):
            evaluate_task(results, save, task, args.seeds)
    if "scaling" in args.tasks:
        scaling(results, save, args.memory_cap, args.budget)
    summarise(results)


if __name__ == "__main__":
    main()
