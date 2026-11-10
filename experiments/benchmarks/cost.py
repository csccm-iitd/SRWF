"""Training cost versus data size: floating-point operations and wall-clock time.

FLOPs count the dominant dense operations of each method (matrix products measured with
``torch.utils.flop_counter``; Cholesky factorisations and triangular solves at their standard
counts n^3/3 and n^2 per right-hand side); backward passes are counted as twice the forward.
Wall-clock times are measured on one GPU after a warm-up fit.

    python experiments/benchmarks/cost.py --out results/cost --time
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from srwf import RFFGP, RWFGP
from srwf.baselines import SVGP, ExactGP
from srwf.data.synthetic import sample


def chol(n: int) -> float:
    return n**3 / 3


def feature_flops(n: int, d: int, model: RWFGP | RFFGP, steps: int, subsample: int) -> float:
    m = min(n, subsample)
    x = torch.randn(512, d)
    model.steps = 0
    model.device = torch.device("cpu")
    model.fit(x.numpy(), x[:, 0].numpy())
    feats = model.blr.features
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            z = feats(x).double()
            z.t() @ z
    per_row = fc.get_total_flops() / 512
    dim = feats.num_features
    step = m * per_row + (min(m, dim) ** 3 / 3 if m < dim else chol(dim)) + 2 * m * dim
    return 3 * steps * step + n * per_row + chol(dim) + 2 * dim * dim


def exact_flops(n: int, d: int, steps: int) -> float:
    return 3 * steps * (chol(n) + 3 * n * n * d + 2 * n * n)


def svgp_flops(n: int, d: int, m: int, epochs: int, batch: int) -> float:
    iters = epochs * int(np.ceil(n / batch))
    return 3 * iters * (chol(m) + 3 * batch * m * d + 2 * batch * m * m + 2 * m**3)


def timed(make, x, y) -> float:
    m = make()
    t0 = time.perf_counter()
    m.fit(x, y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    ap.add_argument("--num-features", type=int, default=2048)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--exact-max", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("results/cost"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    rwf = partial(RWFGP, num_features=args.num_features, levels=6, steps=300)
    rff = partial(RFFGP, num_features=args.num_features, steps=300)
    exact = partial(ExactGP, kernel="rbf", steps=300)
    svgp = partial(SVGP, num_inducing=512, kernel="rbf", epochs=20, batch_size=1024)
    if args.time:
        x, y, _, _ = sample("multi_step", 2000)
        for make in (rwf, rff, exact, svgp):
            timed(make, x, y)
    for n in args.sizes:
        row = dict(
            n=n,
            rwf_flops=feature_flops(n, 1, RWFGP(num_features=args.num_features, levels=6), 300, 20000),
            rff_flops=feature_flops(n, 1, RFFGP(num_features=args.num_features), 300, 20000),
            exact_flops=exact_flops(n, 1, 300),
            svgp_flops=svgp_flops(n, 1, 512, 20, 1024),
        )
        if args.time:
            x, y, _, _ = sample("multi_step", n)
            row["rwf_seconds"] = timed(rwf, x, y)
            row["rff_seconds"] = timed(rff, x, y)
            row["svgp_seconds"] = timed(svgp, x, y)
            if n <= args.exact_max:
                row["exact_seconds"] = timed(exact, x, y)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.out / "cost.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
