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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTDIR = ROOT / "evaluation" / "manuscript_figures"
TIERS = ["R1", "R2", "R3", "R4", "R5"]

# --- Canonical Top-Tier figure style (Okabe-Ito, serif, vector + 300 dpi PNG) ---
# Shared across all ChatchaiTritham PhD repos; see _management/FIGURE_STYLE.md.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


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
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.grid(False)
    im = ax.imshow(mat, cmap="Blues")
    thresh = mat.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(TIERS)
    ax.set_yticks(range(5)); ax.set_yticklabels(TIERS)
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("True tier")
    ax.set_title("SAFE-Gate (ACWCM) confusion matrix\n(test fold, seed 42)")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Cases (count)")
    _save(fig, "confusion_matrix")


def fig_tier_metrics():
    rows = _rows("tier_metrics.csv")
    tiers = [r["tier"] for r in rows]
    prec = [float(r["precision"]) for r in rows]
    rec = [float(r["recall"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    x = np.arange(len(tiers)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x - w, prec, w, label="Precision", color=PALETTE[0])
    ax.bar(x, rec, w, label="Recall", color=PALETTE[2])
    ax.bar(x + w, f1, w, label="F1", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_xlabel("Acuity tier")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (0-1)")
    ax.set_title("Per-tier classification metrics (test fold, seed 42)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26))
    _save(fig, "per_class_metrics")


def fig_baseline_comparison():
    rows = _rows("baseline_comparison.csv")
    methods = [r["method"] for r in rows]
    crit = [float(r["critical_sensitivity"]) for r in rows]
    r5 = [float(r["discharge_specificity_R5"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    x = np.arange(len(methods)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, crit, w, label="Critical sensitivity", color=PALETTE[1])
    ax.bar(x, r5, w, label="R5 discharge specificity", color=PALETTE[0])
    ax.bar(x + w, acc, w, label="Accuracy", color=PALETTE[4])
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 105); ax.set_ylabel("Score (%)")
    ax.set_title("Baselines vs SAFE-Gate (test fold, seed 42)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.42))
    _save(fig, "baseline_comparison")


def fig_safety_and_ablation():
    s = _summary()
    a = s["acwcm"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # Safety headline (computed values + Wilson CI on critical sensitivity)
    lo, hi = a["critical_sensitivity_ci95"]
    labels = ["Critical\nsensitivity", "R5 discharge\nspecificity", "Over-triage\nrate", "Macro-F1"]
    vals = [a["critical_sensitivity_pct"], a["discharge_specificity_R5_pct"],
            a["over_triage_rate_pct"], a["macro_f1_pct"]]
    bars = ax1.bar(labels, vals, color=[PALETTE[2], PALETTE[1], PALETTE[4], PALETTE[0]])
    ax1.errorbar(0, a["critical_sensitivity_pct"],
                 yerr=[[a["critical_sensitivity_pct"] - lo], [hi - a["critical_sensitivity_pct"]]],
                 fmt="none", ecolor="black", capsize=3, lw=0.8)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_ylim(0, 110); ax1.set_ylabel("Score (%)")
    ax1.set_title(f"ACWCM safety (n={s['safety_properties']['n_cases']}, "
                  f"{a['false_discharges']} false discharges)")
    ax1.tick_params(axis="x", labelsize=8)
    # Ablation: over-triage & macro-F1 per removed gate
    abl = s["ablation"]
    cfg = [r["configuration"].replace("Full ACWCM (6 gates)", "Full") for r in abl]
    ot = [r["over_triage_pct"] for r in abl]
    f1 = [r["macro_f1_pct"] for r in abl]
    x = np.arange(len(cfg))
    ax2.plot(x, ot, "o-", color=PALETTE[4], label="Over-triage")
    ax2.plot(x, f1, "s--", color=PALETTE[0], label="Macro-F1")
    ax2.set_xticks(x); ax2.set_xticklabels(cfg, rotation=30, ha="right", fontsize=8)
    ax2.set_xlabel("Configuration (removed gate)")
    ax2.set_ylabel("Score (%)"); ax2.set_title("Gate ablation")
    ax2.legend()
    _save(fig, "safety_and_ablation")


def main():
    apply_pub_style()
    print(f"Reading committed results from {RESULTS}")
    fig_confusion_matrix()
    fig_tier_metrics()
    fig_baseline_comparison()
    fig_safety_and_ablation()
    print(f"Done. Figures written to {OUTDIR} (all values from results/*.csv, seed 42).")


if __name__ == "__main__":
    main()
