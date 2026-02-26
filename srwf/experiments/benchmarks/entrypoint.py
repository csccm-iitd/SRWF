from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..types import ExperimentRunConfig, PreparedRun


def prepare_benchmarks_run(
    config: ExperimentRunConfig,
    repo_root: Path,
    python_executable: str,
) -> PreparedRun:
    if config.mode != "python_script":
        message = config.notes or "Benchmarks runner expects mode=python_script."
        return PreparedRun(command=[], cwd=repo_root, manual=True, manual_message=message)
    if not config.entrypoint:
        raise ValueError("Benchmarks config requires 'entrypoint'.")

    script_path = _resolve(repo_root, config.entrypoint)
    if not script_path.exists():
        raise FileNotFoundError(f"Missing benchmark entrypoint: {script_path}")

    env_overrides: Dict[str, str] = dict(config.env)
    if config.data_path:
        env_overrides["SRWF_DATA_PATH"] = str(_resolve(repo_root, config.data_path))
    if config.results_dir:
        env_overrides["SRWF_RESULTS_DIR"] = str(_resolve(repo_root, config.results_dir))
    if config.dataset_name:
        env_overrides["SRWF_DATASET_NAME"] = config.dataset_name

    return PreparedRun(
        command=[python_executable, str(script_path), *config.args],
        cwd=repo_root,
        env_overrides=env_overrides,
    )


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path)

