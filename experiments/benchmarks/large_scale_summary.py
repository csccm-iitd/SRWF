"""Markdown table of the large-scale benchmark next to published scalable-GP results.

python experiments/benchmarks/large_scale_summary.py --results results/large
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PUBLISHED = Path(__file__).with_name("published") / "large_scale.json"
NAMES = {"rwf": "RWF (ours)", "rff": "RFF (ours, same D)"}


def _ours(results: Path, datasets: list[str]) -> dict[str, dict[str, list[str]]]:
    table: dict[str, dict[str, list[str]]] = {}
    for model, label in NAMES.items():
        rows = {"rmse": [], "nll": [], "minutes": [], "peak_gb": [], "info": []}
        for ds in datasets:
            path = results / f"{ds}.json"
            runs = [r for r in json.loads(path.read_text())["runs"] if r["model"] == model] if path.exists() else []
            for key, scale in (("rmse", 1.0), ("nll", 1.0), ("minutes", 1 / 60), ("peak_gb", 1.0)):
                src = "seconds" if key == "minutes" else key
                v = np.array([r[src] for r in runs]) * scale
                if len(v) == 0:
                    rows[key].append("--")
                    continue
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                digits = (4 if abs(v.mean()) < 0.01 else 3) if key in ("rmse", "nll") else (1 if v.mean() >= 10 else 2)
                rows[key].append(f"{v.mean():.{digits}f}±{se:.{digits}f}" if key != "peak_gb" else f"{v.max():.1f}")
            rows["info"].append(f"{runs[0]['config']}, {len(runs)}" if runs else "--")
        table[label] = rows
    return table


def _published(pub: dict, metric: str) -> dict[str, list[str]]:
    return {k: ["--" if v is None else v for v in vals] for k, vals in pub.get(metric, {}).items()}


def render(results: Path) -> str:
    pub = json.loads(PUBLISHED.read_text())
    ds = pub["datasets"]
    ours = _ours(results, ds)
    head = "| Method | " + " | ".join(d.upper() for d in ds) + " |\n|---|" + "---|" * len(ds) + "\n"

    def block(metric: str) -> str:
        lines = [f"| **{k}** | " + " | ".join(v[metric]) + " |" for k, v in ours.items()]
        lines += [f"| {k} | " + " | ".join(v) + " |" for k, v in _published(pub, metric).items()]
        return head + "\n".join(lines) + "\n"

    meta = "| Method | " + " | ".join(d.upper() for d in ds) + " |\n|---|" + "---|" * len(ds) + "\n"
    meta += "\n".join(f"| {k} config, splits | " + " | ".join(v["info"]) + " |" for k, v in ours.items())
    meta += "\n" + "\n".join(f"| {k} peak GPU GB | " + " | ".join(v["peak_gb"]) + " |" for k, v in ours.items())
    src = "\n".join(f"- **{k}**: {v}" for k, v in pub["sources"].items())
    return (
        "# Large-scale UCI regression\n\n"
        "Ours: mean ± standard error over the fixed `uci_datasets` 90/10 splits, standardised targets. "
        "Published rows: as printed in the sources below (rows marked 2024 are from Lin et al. 2024; "
        "2019 rows are from Wang et al. 2019 and use different splits with 64% of rows for training).\n\n"
        f"## Test RMSE (lower is better)\n\n{block('rmse')}\n"
        f"## Test NLL (lower is better)\n\n{block('nll')}\n"
        "## Wall-clock minutes\n\n"
        "Ours: hyperparameter learning + exact conditioning on all training rows + test prediction, one A100 80GB. "
        "SDD/SGD/CG/SVGP (2024): posterior-mean solve on one A100 with hyperparameters fixed in advance "
        "(learning them is not included). Exact GP (2019): hyperparameter training on up to 8 V100s (published in seconds, converted).\n\n"
        f"{block('minutes')}\n"
        f"## Configuration and memory\n\n{meta}\n\n## Sources\n\n{src}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/large"))
    args = ap.parse_args()
    (args.results / "summary.md").write_text(render(args.results))
    print((args.results / "summary.md").read_text())


if __name__ == "__main__":
    main()
