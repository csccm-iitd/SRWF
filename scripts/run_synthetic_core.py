#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srwf.models import run_exact_gp, run_rff_gp, run_rwf_gp, run_svgp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train core SRWF synthetic models (Exact, SVGP, RFF, RWF).")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/synthetic/multi_step_shift_.npz"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/synthetic/core_models"),
    )
    parser.add_argument("--exact-iters", type=int, default=180)
    parser.add_argument("--svgp-iters", type=int, default=400)
    parser.add_argument("--rff-iters", type=int, default=300)
    parser.add_argument("--rwf-iters", type=int, default=400)
    parser.add_argument("--rff-features", type=int, default=100)
    parser.add_argument("--rwf-levels", type=int, default=4)
    parser.add_argument("--rwf-base-features", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _env_or_default(path: Path, env_name: str) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        return path
    return Path(raw)


def _to_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data).float()


def main() -> None:
    args = parse_args()
    data_path = _env_or_default(Path(str(args.data_path)), "SRWF_DATA_PATH")
    results_dir = _env_or_default(Path(str(args.results_dir)), "SRWF_RESULTS_DIR")
    results_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing synthetic dataset: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    x_train = _to_tensor(data["x_train"])
    y_train = _to_tensor(data["y_train"])
    x_test = _to_tensor(data["x_test"])
    y_test = _to_tensor(data["y_test"])

    pred_exact, loss_exact, _, _ = run_exact_gp(
        x_train, y_train, x_test, iterations=args.exact_iters, device=args.device
    )
    pred_svgp, loss_svgp, _, _ = run_svgp(
        x_train, y_train, x_test, iterations=args.svgp_iters, device=args.device
    )
    pred_rff, loss_rff, _ = run_rff_gp(
        x_train,
        y_train,
        x_test,
        iterations=args.rff_iters,
        num_features=args.rff_features,
        device=args.device,
    )
    pred_rwf, loss_rwf, _ = run_rwf_gp(
        x_train,
        y_train,
        x_test,
        iterations=args.rwf_iters,
        max_level=args.rwf_levels,
        base_features=args.rwf_base_features,
        device=args.device,
    )

    out_path = results_dir / "core_synthetic_results.npz"
    np.savez_compressed(
        out_path,
        x_train=x_train.cpu().numpy(),
        y_train=y_train.cpu().numpy(),
        x_test=x_test.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        exact_mean=pred_exact.mean.detach().cpu().numpy(),
        exact_std=pred_exact.std.detach().cpu().numpy(),
        exact_losses=np.asarray(loss_exact, dtype=np.float32),
        svgp_mean=pred_svgp.mean.detach().cpu().numpy(),
        svgp_std=pred_svgp.std.detach().cpu().numpy(),
        svgp_losses=np.asarray(loss_svgp, dtype=np.float32),
        rff_mean=pred_rff.mean.detach().cpu().numpy(),
        rff_std=pred_rff.std.detach().cpu().numpy(),
        rff_losses=np.asarray(loss_rff, dtype=np.float32),
        rwf_mean=pred_rwf.mean.detach().cpu().numpy(),
        rwf_std=pred_rwf.std.detach().cpu().numpy(),
        rwf_losses=np.asarray(loss_rwf, dtype=np.float32),
    )
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
