from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .types import ExperimentRunConfig


def default_config_path(repo_root: Path, group: str) -> Path:
    return repo_root / "configs" / group / "default.yaml"


def load_experiment_config(path: Path | str) -> ExperimentRunConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    known: Dict[str, Any] = {
        "group": raw.get("group", ""),
        "name": raw.get("name", config_path.stem),
        "mode": raw.get("mode", "python_script"),
        "entrypoint": raw.get("entrypoint"),
        "args": list(raw.get("args", [])),
        "data_path": raw.get("data_path"),
        "dataset_name": raw.get("dataset_name"),
        "results_dir": raw.get("results_dir"),
        "notes": raw.get("notes"),
        "env": dict(raw.get("env", {})),
    }
    extras = {k: v for k, v in raw.items() if k not in known}
    cfg = ExperimentRunConfig(**known, extras=extras)
    if not cfg.group:
        raise ValueError(f"Config field 'group' is required: {config_path}")
    if not cfg.name:
        raise ValueError(f"Config field 'name' is required: {config_path}")
    return cfg
