#!/usr/bin/env python3
"""Run the pinned GR00T model-core benchmark from the Robo checkout.

The ApxInf submodule owns the CUDA Graph runner and official-processor fixture
format. Running that maintained runner preserves its exact timing boundary.
"""

import os
import sys
from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for flag in (
        "--checkpoint", "--backbone", "--fixture", "--calibration",
        "--tactics", "--output", "--binary", "--reference",
    ):
        if flag in sys.argv:
            index = sys.argv.index(flag) + 1
            value = Path(sys.argv[index])
            if not value.is_absolute():
                sys.argv[index] = str(root / value)
    os.chdir(root / "apxinf")
    source = root / "apxinf/scripts/bench_gr00t.py"
    run_path(str(source), run_name="__main__")
