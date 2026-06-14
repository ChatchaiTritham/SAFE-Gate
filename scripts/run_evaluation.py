#!/usr/bin/env python3
"""
Lightweight safety-only evaluation driver for SAFE-Gate.

Loads the held-out synthetic test split, runs every case through SAFEGate, and
writes a single independent artifact:

  results/safety_check.json   confusion matrix + tier metrics + safety summary

The full reproducible pipeline (data regeneration, baselines, formal property
verification) lives in run_all.py at the repository root, which owns the canonical
results/*.csv and results/summary.json. This driver is a quick safety cross-check
and does not overwrite those files. Output is deterministic on a fixed test split.

Usage:
    pip install -e .
    python scripts/run_evaluation.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from safegate import SAFEGate

TIERS = ["R1", "R2", "R3", "R4", "R5"]  # R1 = highest acuity, R5 = discharge
CRITICAL = {"R1", "R2"}
DISCHARGE = "R5"

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "synthetic" / "test"
RESULTS = ROOT / "results"


def main():
    test_file = sorted(TEST_DIR.glob("synthetic_test_*.json"))[0]
    cases = json.loads(test_file.read_text(encoding="utf-8"))
    sg = SAFEGate()

    conf = {t: Counter() for t in TIERS}
    for p in cases:
        true = p["ground_truth_tier"]
        pred = sg.classify(p)["final_tier"]
        conf[true][pred] += 1

    true_dist = {t: sum(conf[t].values()) for t in TIERS}
    pred_dist = {t: sum(conf[r][t] for r in TIERS) for t in TIERS}
    n = sum(true_dist.values())

    # tier-level metrics
    tier_rows = []
    for t in TIERS:
        tp = conf[t][t]
        fp = sum(conf[r][t] for r in TIERS if r != t)
        fn = sum(conf[t][p] for p in TIERS if p != t)
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        tier_rows.append({"tier": t, "support": true_dist[t], "predicted": pred_dist[t],
                          "precision": round(prec, 4), "recall": round(rec, 4),
                          "specificity": round(spec, 4)})

    crit_true = sum(true_dist[t] for t in CRITICAL)
    crit_kept = sum(conf[t][p] for t in CRITICAL for p in CRITICAL)
    false_disch = sum(conf[t][DISCHARGE] for t in CRITICAL)
    r5_true = true_dist[DISCHARGE]
    r5_correct = conf[DISCHARGE][DISCHARGE]
    over_triage = sum(conf[t][p] for t in TIERS for p in TIERS
                      if TIERS.index(p) < TIERS.index(t))  # predicted more acute than truth

    summary = {
        "n_cases": n,
        "true_distribution": true_dist,
        "predicted_distribution": pred_dist,
        "tiers_never_predicted": [t for t in TIERS if pred_dist[t] == 0],
        "critical_sensitivity": {"caught": crit_kept, "total": crit_true,
                                 "pct": round(100 * crit_kept / crit_true, 2) if crit_true else 0.0},
        "false_discharges": false_disch,
        "discharge_specificity_R5": {"correct": r5_correct, "total": r5_true,
                                     "pct": round(100 * r5_correct / r5_true, 2) if r5_true else 0.0},
        "over_triage_rate_pct": round(100 * over_triage / n, 2),
    }

    # Lightweight safety-only check. The canonical artifacts (confusion_matrix.csv,
    # tier_metrics.csv, summary.json) are owned by run_all.py; this driver writes a
    # single independent file so the two never clobber each other.
    RESULTS.mkdir(exist_ok=True)
    summary["tier_metrics"] = tier_rows
    summary["confusion_matrix"] = {t: {p: conf[t][p] for p in TIERS} for t in TIERS}
    (RESULTS / "safety_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
