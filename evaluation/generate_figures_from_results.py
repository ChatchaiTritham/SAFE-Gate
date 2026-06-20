#!/usr/bin/env python3
"""
Honest manuscript-figure generator for SAFE-Gate.

Every number plotted here is READ from the committed CSV/JSON artifacts produced by
``scripts/run_all.py`` (seed 42) -- nothing is typed into this file. Run the driver
first if ``results/`` is empty:

    python scripts/run_all.py
    python evaluation/generate_figures_from_results.py

Inputs (results/):
    confusion_matrix.csv   true x pred tier counts
    tier_metrics.csv       per-tier precision / recall / specificity / f1
    baseline_comparison.csv  method x {critical_sensitivity, accuracy, R5 specificity, ...}
    summary.json           headline ACWCM metrics + ablation + safety properties

Outputs: evaluation/manuscript_figures/*.png|.pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTDIR = ROOT / "evaluation" / "manuscript_figures"
TIERS = ["R1", "R2", "R3", "R4", "R5"]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


def _rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python scripts/run_all.py` first.")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def _save(fig, name: str):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [OK] {name}.png/.pdf")


def fig_confusion_matrix():
    rows = {r["true\\pred"]: r for r in _rows("confusion_matrix.csv")}
    mat = np.array([[int(rows[t][p]) for p in TIERS] for t in TIERS], dtype=int)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(TIERS)
    ax.set_yticks(range(5)); ax.set_yticklabels(TIERS)
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("True tier")
    ax.set_title("SAFE-Gate (ACWCM) confusion matrix (test fold, seed 42)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cases")
    _save(fig, "confusion_matrix")


def fig_tier_metrics():
    rows = _rows("tier_metrics.csv")
    tiers = [r["tier"] for r in rows]
    prec = [float(r["precision"]) for r in rows]
    rec = [float(r["recall"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    x = np.arange(len(tiers)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(x - w, prec, w, label="Precision", color="#4e79a7")
    ax.bar(x, rec, w, label="Recall", color="#59a14f")
    ax.bar(x + w, f1, w, label="F1", color="#e15759")
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-tier metrics (test fold, seed 42)")
    ax.legend(fontsize=7, framealpha=0.95)
    _save(fig, "per_class_metrics")


def fig_baseline_comparison():
    rows = _rows("baseline_comparison.csv")
    methods = [r["method"] for r in rows]
    crit = [float(r["critical_sensitivity"]) for r in rows]
    r5 = [float(r["discharge_specificity_R5"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    x = np.arange(len(methods)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - w, crit, w, label="Critical sensitivity (%)", color="#e15759")
    ax.bar(x, r5, w, label="R5 discharge specificity (%)", color="#4e79a7")
    ax.bar(x + w, acc, w, label="Accuracy (%)", color="#bab0ac")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0, 105); ax.set_ylabel("Percent")
    ax.set_title("Baselines vs SAFE-Gate (test fold, seed 42)")
    ax.legend(fontsize=7, framealpha=0.95)
    _save(fig, "baseline_comparison")


def fig_safety_and_ablation():
    s = _summary()
    a = s["acwcm"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # Safety headline (computed values + Wilson CI on critical sensitivity)
    lo, hi = a["critical_sensitivity_ci95"]
    labels = ["Critical\nsensitivity", "R5 discharge\nspecificity", "Over-triage\nrate", "Macro-F1"]
    vals = [a["critical_sensitivity_pct"], a["discharge_specificity_R5_pct"],
            a["over_triage_rate_pct"], a["macro_f1_pct"]]
    bars = ax1.bar(labels, vals, color=["#59a14f", "#e15759", "#f28e2b", "#4e79a7"])
    ax1.errorbar(0, a["critical_sensitivity_pct"],
                 yerr=[[a["critical_sensitivity_pct"] - lo], [hi - a["critical_sensitivity_pct"]]],
                 fmt="none", ecolor="black", capsize=3, lw=0.8)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    ax1.set_ylim(0, 110); ax1.set_ylabel("Percent")
    ax1.set_title(f"ACWCM safety (n={s['safety_properties']['n_cases']}, "
                  f"{a['false_discharges']} false discharges)", fontsize=8)
    ax1.tick_params(axis="x", labelsize=6.5)
    # Ablation: over-triage & macro-F1 per removed gate
    abl = s["ablation"]
    cfg = [r["configuration"].replace("Full ACWCM (6 gates)", "Full") for r in abl]
    ot = [r["over_triage_pct"] for r in abl]
    f1 = [r["macro_f1_pct"] for r in abl]
    x = np.arange(len(cfg))
    ax2.plot(x, ot, "o-", color="#f28e2b", lw=1.2, ms=4, label="Over-triage (%)")
    ax2.plot(x, f1, "s--", color="#4e79a7", lw=1.2, ms=4, label="Macro-F1 (%)")
    ax2.set_xticks(x); ax2.set_xticklabels(cfg, rotation=30, ha="right", fontsize=6.5)
    ax2.set_ylabel("Percent"); ax2.set_title("Gate ablation", fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.95)
    plt.tight_layout()
    _save(fig, "safety_and_ablation")


def main():
    print(f"Reading committed results from {RESULTS}")
    fig_confusion_matrix()
    fig_tier_metrics()
    fig_baseline_comparison()
    fig_safety_and_ablation()
    print(f"Done. Figures written to {OUTDIR} (all values from results/*.csv, seed 42).")


if __name__ == "__main__":
    main()
