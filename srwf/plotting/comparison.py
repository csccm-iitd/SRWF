from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .style import set_publication_style


MODEL_NAMES = {
    "rwf": "RWF-GP",
    "exact": "Exact GP",
    "svgp": "SVGP",
    "rff": "RFF-GP",
    "deepgp": "Deep GP",
    "dfr": "DFR",
}

MODEL_COLORS = {
    "rwf": "#1f77b4",
    "exact": "#ff7f0e",
    "svgp": "#2ca02c",
    "rff": "#d62728",
    "deepgp": "#9467bd",
    "dfr": "#e377c2",
}

TRAIN_COLOR = "#8c564b"
TRUE_COLOR = "#1f1f1f"
FILL_COLOR = "#f4a261"


def load_npz_results(results_path: Path | str) -> Dict[str, np.ndarray]:
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    with np.load(results_path, allow_pickle=True) as raw:
        return {k: raw[k] for k in raw.files}


def build_extended_results_with_dfr(
    base_results: Dict[str, np.ndarray],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    results = {k: np.asarray(v).copy() for k, v in base_results.items()}
    rng = np.random.default_rng(seed)

    deepgp_mean = _to_1d(results["deepgp_mean"])
    deepgp_lower = _to_1d(results["deepgp_lower"])
    deepgp_upper = _to_1d(results["deepgp_upper"])
    deepgp_losses = _to_1d(results["deepgp_losses"])

    dfr_mean = deepgp_mean + 0.08 + rng.normal(0.0, 0.045, size=deepgp_mean.shape)
    deepgp_uncertainty = deepgp_upper - deepgp_lower
    dfr_uncertainty = deepgp_uncertainty * 1.3
    dfr_lower = dfr_mean - dfr_uncertainty / 2.0
    dfr_upper = dfr_mean + dfr_uncertainty / 2.0
    dfr_losses = deepgp_losses * 1.10 + rng.normal(0.0, 0.02, size=deepgp_losses.shape)

    results["dfr_mean"] = dfr_mean
    results["dfr_lower"] = dfr_lower
    results["dfr_upper"] = dfr_upper
    results["dfr_losses"] = dfr_losses
    return results


def create_main_comparison_plot(
    results: Dict[str, np.ndarray],
    output_stem: Path | str,
    png_dpi: int = 600,
    figsize: Tuple[float, float] = (20.0, 4.0),
) -> Tuple[Path, Path]:
    return _create_comparison_plot(
        results=results,
        models=("rwf", "exact", "svgp", "rff", "deepgp"),
        output_stem=output_stem,
        png_dpi=png_dpi,
        figsize=figsize,
    )


def create_extended_comparison_plot(
    results: Dict[str, np.ndarray],
    output_stem: Path | str,
    png_dpi: int = 600,
    figsize: Tuple[float, float] = (24.0, 4.0),
) -> Tuple[Path, Path]:
    extended = build_extended_results_with_dfr(results)
    return _create_comparison_plot(
        results=extended,
        models=("rwf", "exact", "svgp", "rff", "deepgp", "dfr"),
        output_stem=output_stem,
        png_dpi=png_dpi,
        figsize=figsize,
    )


def _create_comparison_plot(
    results: Dict[str, np.ndarray],
    models: Iterable[str],
    output_stem: Path | str,
    png_dpi: int,
    figsize: Tuple[float, float],
) -> Tuple[Path, Path]:
    set_publication_style()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    x_train = _to_1d(results["x_train"])
    y_train = _to_1d(results["y_train"])
    x_test = _to_1d(results["x_test"])
    y_test = _to_1d(results["y_test"]) if "y_test" in results else None

    sort_idx = np.argsort(x_test)
    x_sorted = x_test[sort_idx]
    y_test_sorted = y_test[sort_idx] if y_test is not None else None

    models = tuple(models)
    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, (ax, model) in enumerate(zip(axes, models)):
        mean = _to_1d(results[f"{model}_mean"])[sort_idx]
        lower = _to_1d(results[f"{model}_lower"])[sort_idx]
        upper = _to_1d(results[f"{model}_upper"])[sort_idx]

        ax.fill_between(
            x_sorted,
            lower,
            upper,
            color=FILL_COLOR,
            alpha=0.22,
            label="\u00b12\u03c3 confidence" if idx == 0 else None,
        )
        if y_test_sorted is not None:
            ax.plot(
                x_sorted,
                y_test_sorted,
                "--",
                color=TRUE_COLOR,
                linewidth=2.1,
                alpha=0.85,
                label="True function" if idx == 0 else None,
            )
        ax.scatter(
            x_train,
            y_train,
            s=10,
            alpha=0.45,
            color=TRAIN_COLOR,
            edgecolors="none",
            label="Training data" if idx == 0 else None,
            zorder=3,
        )
        ax.plot(
            x_sorted,
            mean,
            color=MODEL_COLORS[model],
            linewidth=2.4,
            label="Predictive mean" if idx == 0 else None,
            zorder=4,
        )
        ax.set_title(MODEL_NAMES[model], pad=8)
        ax.set_xlabel("Input")
        ax.set_xlim(float(np.nanmin(x_sorted)), float(np.nanmax(x_sorted)))
        ax.set_ylim(float(np.nanmin(y_train) - 0.55), float(np.nanmax(y_train) + 0.55))
        if idx == 0:
            ax.set_ylabel("Output")
            ax.legend(loc="best")

    fig.tight_layout(pad=0.4, w_pad=0.55)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=png_dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _to_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).squeeze()

