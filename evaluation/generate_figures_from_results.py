#!/usr/bin/env python3
"""
Honest manuscript-figure generator for SAFE-Gate.

Every number plotted here is READ from the committed CSV/JSON artifacts produced by
``scripts/run_all.py`` (seed 42) -- nothing is typed into this file. Run the driver
first if ``results/`` is empty:

    python scripts/run_all.py
    python evaluation/generate_figures_from_results.py

Inputs (results/):
    confusion_matrix.csv     true x pred tier counts
    tier_metrics.csv         per-tier precision / recall / specificity / f1
    baseline_comparison.csv  method x {critical_sensitivity, accuracy, R5 specificity, ...}
    summary.json             headline ACWCM metrics + ablation + safety properties

Outputs: evaluation/manuscript_figures/*.png|.pdf

INTEGRITY NOTE (honest reporting): the reproduced ACWCM model collapses to three
active tiers. R2, R3 and R5 receive recall = 0 (R3 is never predicted at all); the
merging concentrates predicted mass on the critical tier (R1) and the low-acuity
tier (R4). The figures below therefore DO NOT present R2/R3/R5 as functional tiers:
the confusion matrix and per-tier panel carry an explicit zero-recall annotation,
and the headline sensitivity claim is shown with a Wilson binomial CI because it is
a zero-event (0 missed of 175) estimate. No safety number is altered.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pubviz import apply_pub_style, save_fig, PALETTE, load_results, results_dir

ROOT = Path(__file__).resolve().parent.parent
RESULTS = results_dir(ROOT)
OUTDIR = ROOT / "evaluation" / "manuscript_figures"
TIERS = ["R1", "R2", "R3", "R4", "R5"]


def _rows(name: str) -> list[dict]:
    # load_results(.csv) -> list[dict]; pin to this repo root so cwd does not matter.
    return load_results(name, start=ROOT)


def _summary() -> dict:
    return load_results("summary.json", start=ROOT)


def _save(fig, name: str):
    save_fig(fig, name, out_dir=OUTDIR)
    plt.close(fig)


def _zero_recall_tiers(tier_rows: list[dict]) -> list[str]:
    """Tiers the model fails to recover (recall == 0) -- honest tier-collapse flag."""
    return [r["tier"] for r in tier_rows if float(r["recall"]) == 0.0]


def fig_confusion_matrix():
    rows = {r["true\\pred"]: r for r in _rows("confusion_matrix.csv")}
    mat = np.array([[int(rows[t][p]) for p in TIERS] for t in TIERS], dtype=int)
    tier_rows = _rows("tier_metrics.csv")
    dead = _zero_recall_tiers(tier_rows)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.grid(False)
    im = ax.imshow(mat, cmap="Blues")
    thresh = mat.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(TIERS)
    ax.set_yticks(range(5)); ax.set_yticklabels(TIERS)
    # Mark the collapsed (zero-recall) true tiers so the diagonal gaps are not read
    # as a working five-tier classifier.
    for k, t in enumerate(TIERS):
        if t in dead:
            ax.get_yticklabels()[k].set_color(PALETTE[1])
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("True tier")
    ax.set_title("SAFE-Gate (ACWCM) confusion matrix\n(test fold, seed 42)")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Cases (count)")
    note = ("Zero-recall tiers (no correct recovery): " + ", ".join(dead) +
            ".\nACWCM concentrates predictions on the critical (R1) and low-acuity (R4) tiers;"
            "\nintermediate tiers are not functionally separated.")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7, color=PALETTE[1])
    _save(fig, "confusion_matrix")


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval (%) for a binomial proportion -- robust at k==n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return 100 * (centre - half), 100 * min(1.0, centre + half)


def fig_per_class_metrics():
    rows = _rows("tier_metrics.csv")
    tiers = [r["tier"] for r in rows]
    support = [int(r["support"]) for r in rows]
    prec = [float(r["precision"]) for r in rows]
    rec = [float(r["recall"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    dead = _zero_recall_tiers(rows)

    # Recall is a binomial proportion over each tier's support -> Wilson CI.
    rec_lo, rec_hi = [], []
    for r, n in zip(rec, support):
        lo, hi = _wilson_ci(int(round(r * n)), n)
        rec_lo.append(max(0.0, r - lo / 100.0))
        rec_hi.append(max(0.0, hi / 100.0 - r))

    x = np.arange(len(tiers)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w, prec, w, label="Precision", color=PALETTE[0])
    ax.bar(x, rec, w, label="Recall", color=PALETTE[2],
           yerr=[rec_lo, rec_hi], capsize=2, ecolor="black",
           error_kw={"lw": 0.7})
    ax.bar(x + w, f1, w, label="F1", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    for k, t in enumerate(tiers):
        if t in dead:
            ax.get_xticklabels()[k].set_color(PALETTE[1])
            ax.text(x[k], 0.02, "no recall", rotation=90, ha="center", va="bottom",
                    fontsize=6.5, color=PALETTE[1])
    ax.set_xlabel("Acuity tier")
    ax.set_ylim(0, 1.10); ax.set_ylabel("Score (0-1)")
    ax.set_title("Per-tier metrics with Wilson 95% CI on recall (test fold, seed 42)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.text(0.5, -0.10,
             "Tiers in colour collapse to recall = 0 (R3 is never predicted); the model "
             "is not a working 5-tier classifier.",
             ha="center", va="top", fontsize=7, color=PALETTE[1])
    _save(fig, "per_class_metrics")


def fig_baseline_comparison():
    """Cleveland-McGill dot plot: safety (critical sensitivity) vs accuracy trade-off."""
    rows = _rows("baseline_comparison.csv")
    methods = [r["method"] for r in rows]
    crit = [float(r["critical_sensitivity"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]

    order = sorted(range(len(methods)), key=lambda i: crit[i])
    methods = [methods[i] for i in order]
    crit = [crit[i] for i in order]
    acc = [acc[i] for i in order]
    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for yi, c, a in zip(y, crit, acc):
        ax.plot([min(c, a), max(c, a)], [yi, yi], "-", color="#bbbbbb", lw=1.1, zorder=1)
    ax.scatter(crit, y, s=46, color=PALETTE[1], zorder=3, label="Critical sensitivity")
    ax.scatter(acc, y, s=46, color=PALETTE[0], zorder=3, label="Overall accuracy")
    ax.set_yticks(y); ax.set_yticklabels(methods)
    ax.set_xlim(-2, 108); ax.set_xlabel("Score (%)")
    ax.set_title("Safety vs accuracy trade-off across methods (test fold, seed 42)")
    ax.legend(loc="lower right")
    # Annotate the SAFE-Gate trade-off explicitly.
    if "SAFE-Gate (ACWCM)" in methods:
        i = methods.index("SAFE-Gate (ACWCM)")
        ax.annotate("max safety,\nlowest accuracy",
                    xy=(acc[i], i), xytext=(acc[i] + 12, i - 0.45),
                    fontsize=7, color=PALETTE[0], va="top",
                    arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8))
    _save(fig, "baseline_comparison")


def fig_safety_and_ablation():
    s = _summary()
    a = s["acwcm"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # --- Safety headline as BARS with a Wilson CI on the zero-event sensitivity ---
    lo, hi = a["critical_sensitivity_ci95"]
    labels = ["Critical\nsensitivity", "R5 discharge\nspecificity", "Over-triage\nrate", "Macro-F1"]
    vals = [a["critical_sensitivity_pct"], a["discharge_specificity_R5_pct"],
            a["over_triage_rate_pct"], a["macro_f1_pct"]]
    bars = ax1.bar(labels, vals, color=[PALETTE[2], PALETTE[1], PALETTE[4], PALETTE[0]])
    ax1.errorbar(0, a["critical_sensitivity_pct"],
                 yerr=[[a["critical_sensitivity_pct"] - lo], [hi - a["critical_sensitivity_pct"]]],
                 fmt="none", ecolor="black", capsize=4, lw=0.9)
    ax1.text(0, hi + 2.5, f"95% CI\n[{lo:.1f}, {hi:.1f}]", ha="center", va="bottom",
             fontsize=6.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_ylim(0, 118); ax1.set_ylabel("Score (%)")
    ax1.set_title(f"ACWCM safety (n={s['safety_properties']['n_cases']}, "
                  f"{a['critical_caught']}/{a['critical_total']} critical, "
                  f"{a['false_discharges']} false discharges)")
    ax1.tick_params(axis="x", labelsize=8)

    # --- Ablation: over-triage & macro-F1 per removed gate ---
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


def fig_calibration():
    """Reliability diagram -- ONLY if Monte-Carlo-dropout uncertainty data exists.

    SAFE-Gate's reproduced results/ contains no per-case predicted probabilities or
    MC-dropout samples (the merging is rule/lattice-based, not a softmax classifier),
    so a calibration curve cannot be drawn from real data. We refuse to fabricate one.
    """
    candidates = ["mc_dropout.json", "calibration.csv", "calibration.json",
                  "predicted_probabilities.csv", "uncertainty.json"]
    found = [c for c in candidates if (RESULTS / c).exists()]
    if not found:
        print("  [skip] calibration/reliability diagram: no MC-dropout / predicted-"
              "probability artifact in results/ -- not fabricated.")
        return
    print(f"  [info] calibration source(s) present: {found} -- (no plotter wired; add "
          "one only against the real schema).")


def main():
    apply_pub_style()
    print(f"Reading committed results from {RESULTS}")
    fig_safety_and_ablation()
    fig_baseline_comparison()
    fig_confusion_matrix()
    fig_per_class_metrics()
    fig_calibration()
    print(f"Done. Figures written to {OUTDIR} (all values from results/*, seed 42).")


if __name__ == "__main__":
    main()
