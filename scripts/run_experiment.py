#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srwf.experiments import default_config_path, load_experiment_config, prepare_run


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRWF experiment groups with clean wrappers.")
    parser.add_argument(
        "--group",
        required=True,
        choices=("synthetic", "uci", "timit", "benchmarks"),
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config path. Defaults to configs/<group>/default.yaml",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for launched scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command/environment without executing.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print environment overrides in addition to command.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_path = args.config or default_config_path(REPO_ROOT, args.group)
    config = load_experiment_config(config_path)
    run = prepare_run(config, REPO_ROOT, args.python)

    print(f"Group: {config.group}")
    print(f"Config: {config_path}")
    print(f"Name: {config.name}")

    if run.manual:
        print("Mode: manual")
        print(run.manual_message or "No command configured.")
        return 0

    print("Mode: python_script")
    print("Command:")
    print("  " + " ".join(run.command))

    if args.print_env and run.env_overrides:
        print("Environment overrides:")
        for key in sorted(run.env_overrides):
            print(f"  {key}={run.env_overrides[key]}")

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.update(run.env_overrides)
    completed = subprocess.run(run.command, cwd=run.cwd, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

