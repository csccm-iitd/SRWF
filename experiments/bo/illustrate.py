"""Record a 1-D Thompson-sampling run for figures: prior/posterior samples, the drawn acquisition
function and its argmax at every iteration.

    python experiments/bo/illustrate.py --problem gramacy_lee --method rwf_ts

Writes ``results/bo/illustration_<problem>_<method>_seed<seed>.npz`` and a sanity PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functions import PROBLEMS  # noqa: E402
from methods import rff_sampler, rwf_sampler  # noqa: E402

from srwf.bo import Trace, thompson_sampling  # noqa: E402
from srwf.sampling import PosteriorFunction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="gramacy_lee")
    ap.add_argument("--method", default="rwf_ts", choices=["rwf_ts", "rff_ts"])
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--grid", type=int, default=1000)
    ap.add_argument("--paths", type=int, default=8, help="extra posterior draws stored per iteration")
    ap.add_argument("--num-features", type=int, default=1024)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bo")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = PROBLEMS[args.problem]
    if p.dim != 1:
        raise ValueError("illustration needs a 1-D problem")
    grid = np.linspace(p.bounds[0, 0], p.bounds[0, 1], args.grid)[:, None]
    if args.method == "rwf_ts":
        sampler = rwf_sampler(p.bounds, args.num_features, args.levels, args.steps, args.seed, warm_start=True)
    else:
        sampler = rff_sampler(args.num_features, args.steps, args.seed, warm_start=True)

    rec: dict[str, list] = {k: [] for k in ("acq", "argmax", "mean", "std", "paths", "prior", "n_obs")}

    def callback(it: int, f, x_next: np.ndarray, trace: Trace) -> None:
        model = sampler.model
        pred = model.predict(grid, return_f_std=True)
        with torch.no_grad():
            rec["acq"].append(f(torch.as_tensor(grid, dtype=torch.float32))[0].cpu().numpy())
        rec["argmax"].append(x_next[0, 0])
        rec["mean"].append(pred.mean)
        rec["std"].append(pred.f_std)
        rec["paths"].append(PosteriorFunction(model, args.paths, seed=10_000 + it).numpy(grid))
        rec["prior"].append(PosteriorFunction(model, args.paths, prior=True, seed=10_000 + it).numpy(grid))
        rec["n_obs"].append(len(trace.y))

    trace = thompson_sampling(p, p.bounds, sampler, args.budget, args.n_init, 1, args.seed, callback)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"illustration_{args.problem}_{args.method}_seed{args.seed}.npz"
    np.savez_compressed(
        path,
        grid=grid[:, 0],
        truth=p(grid),
        x=trace.x[:, 0],
        y=trace.y,
        optimum=p.optimum,
        **{k: np.array(v) for k, v in rec.items()},
    )
    print(f"wrote {path}; final regret {p.optimum - trace.best[-1]:.3g}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = np.unique(np.linspace(0, len(rec["acq"]) - 1, 4).astype(int))
    fig, axes = plt.subplots(len(its), 1, figsize=(7, 2.2 * len(its)), sharex=True)
    for ax, it in zip(np.atleast_1d(axes), its, strict=True):
        n = rec["n_obs"][it]
        m, s = rec["mean"][it], rec["std"][it]
        ax.plot(grid, p(grid), "k", lw=1)
        ax.fill_between(grid[:, 0], m - 2 * s, m + 2 * s, alpha=0.2)
        ax.plot(grid, rec["paths"][it].T, lw=0.4, alpha=0.6)
        ax.plot(grid, rec["acq"][it], "C3", lw=1.2)
        ax.axvline(rec["argmax"][it], color="C3", ls="--", lw=0.8)
        ax.plot(trace.x[:n, 0], trace.y[:n], "ko", ms=3)
        ax.set_title(f"iteration {it}, n = {n}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"illustration_{args.problem}_{args.method}_seed{args.seed}.png", dpi=90)


if __name__ == "__main__":
    main()
