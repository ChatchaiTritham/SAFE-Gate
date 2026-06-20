#!/usr/bin/env python3
"""
DEPRECATED. This script previously generated performance figures using hardcoded
baseline metrics (Random Forest / XGBoost / Neural Network straw-men) and other
typed-in numbers that did not match the actual evaluation. It has been replaced by
``evaluation/generate_figures_from_results.py``, which draws every figure from the
committed ``results/*.csv`` produced by ``scripts/run_all.py`` (seed 42). The real
baseline comparison lives in ``results/baseline_comparison.csv``. Kept only as a
redirect.

    python scripts/run_all.py
    python evaluation/generate_figures_from_results.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    print("[DEPRECATED] generate_performance_metrics.py no longer emits figures.")
    print("             Running evaluation/generate_figures_from_results.py instead")
    print("             (all numbers read from results/*.csv, seed 42).")
    runpy.run_path(str(Path(__file__).with_name("generate_figures_from_results.py")),
                   run_name="__main__")
