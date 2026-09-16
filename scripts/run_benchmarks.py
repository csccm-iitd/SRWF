#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--group", "benchmarks", *sys.argv[1:]]))

