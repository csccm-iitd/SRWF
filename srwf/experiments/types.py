from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRunConfig:
    group: str
    name: str
    mode: str
    entrypoint: Optional[str] = None
    args: List[str] = field(default_factory=list)
    data_path: Optional[str] = None
    dataset_name: Optional[str] = None
    results_dir: Optional[str] = None
    notes: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedRun:
    command: List[str]
    cwd: Path
    env_overrides: Dict[str, str] = field(default_factory=dict)
    manual: bool = False
    manual_message: Optional[str] = None

