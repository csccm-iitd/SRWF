#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srwf.plotting import (
    create_extended_comparison_plot,
    create_feature_wrf_rff_plot,
    create_main_comparison_plot,
    create_scalability_combined_linear_plot,
    load_npz_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate camera-ready SRWF paper plots.")
    parser.add_argument(
        "--main-results",
        type=Path,
        default=Path("main/results_gp/non_st_complete_results.npz"),
        help="NPZ file for synthetic model comparison plots.",
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("/DATA/Sawan_projects/proj_1/WRF_01/results/feature_sweep.csv"),
        help="CSV path for feature RMSE plot.",
    )
    parser.add_argument(
        "--scalability-csv",
        type=Path,
        default=Path("/DATA/Sawan_projects/proj_1/WRF_01/results/scalability_results_with_std (2).csv"),
        help="CSV path for scalability plot.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/figures/paper"),
        help="Directory for primary plot outputs.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="PNG resolution (PDF is vector).",
    )
    parser.add_argument(
        "--only",
        choices=("all", "comparison", "benchmarks"),
        default="all",
        help="Subset of figures to generate.",
    )
    parser.add_argument(
        "--write-legacy-main",
        action="store_true",
        help="Also overwrite main/plots and main/plots_extended comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created: List[Path] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "comparison"):
        results = load_npz_results(args.main_results)
        out_main = args.out_dir / "main_comparison"
        out_ext = args.out_dir / "extended_comparison_6models"
        created.extend(create_main_comparison_plot(results, out_main, png_dpi=args.png_dpi))
        created.extend(create_extended_comparison_plot(results, out_ext, png_dpi=args.png_dpi))
        if args.write_legacy_main:
            created.extend(
                create_main_comparison_plot(
                    results,
                    Path("main/plots/main_comparison"),
                    png_dpi=args.png_dpi,
                )
            )
            created.extend(
                create_extended_comparison_plot(
                    results,
                    Path("main/plots_extended/extended_comparison_6models"),
                    png_dpi=args.png_dpi,
                )
            )

    if args.only in ("all", "benchmarks"):
        created.extend(
            create_feature_wrf_rff_plot(
                args.feature_csv,
                args.out_dir / "feature_wrf_rff",
                png_dpi=args.png_dpi,
            )
        )
        created.extend(
            create_scalability_combined_linear_plot(
                args.scalability_csv,
                args.out_dir / "scalability_combined_linear",
                png_dpi=args.png_dpi,
            )
        )

    print("Generated files:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()
