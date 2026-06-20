#!/usr/bin/env python3
"""
DEPRECATED. This script previously generated manuscript figures from numbers typed
into the source ("expected system behavior"), which did not match the actual model
output. It has been replaced by ``evaluation/generate_figures_from_results.py``,
which draws every figure from the committed ``results/*.csv`` produced by
``scripts/run_all.py`` (seed 42). It is kept only as a redirect.

    python scripts/run_all.py
    python evaluation/generate_figures_from_results.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    print("[DEPRECATED] create_manuscript_figures.py no longer emits figures.")
    print("             Running evaluation/generate_figures_from_results.py instead")
    print("             (all numbers read from results/*.csv, seed 42).")
    runpy.run_path(str(Path(__file__).with_name("generate_figures_from_results.py")),
                   run_name="__main__")
