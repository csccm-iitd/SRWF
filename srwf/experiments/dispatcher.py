from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from .benchmarks import prepare_benchmarks_run
from .synthetic import prepare_synthetic_run
from .timit import prepare_timit_run
from .types import ExperimentRunConfig, PreparedRun
from .uci import prepare_uci_run


RunBuilder = Callable[[ExperimentRunConfig, Path, str], PreparedRun]

BUILDERS: Dict[str, RunBuilder] = {
    "synthetic": prepare_synthetic_run,
    "benchmarks": prepare_benchmarks_run,
    "uci": prepare_uci_run,
    "timit": prepare_timit_run,
}


def prepare_run(
    config: ExperimentRunConfig,
    repo_root: Path,
    python_executable: str,
) -> PreparedRun:
    group = config.group.lower()
    if group not in BUILDERS:
        known = ", ".join(sorted(BUILDERS))
        raise ValueError(f"Unknown experiment group '{config.group}'. Expected one of: {known}")
    return BUILDERS[group](config, repo_root, python_executable)

