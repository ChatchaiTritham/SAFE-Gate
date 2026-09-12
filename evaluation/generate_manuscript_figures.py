"""Draw the four manuscript figures at the journal text-block width.

The figures that shipped with the manuscript were 800-1000 pt wide and were
squeezed into a 390 pt column, which left their smallest labels at 2.4-4.9 pt.
Worse, they had no generator in this repository at all, so nothing tied them to
the computed results. Both problems are fixed here: every figure is authored at
390 pt (the measured \\the\\textwidth of the elsarticle review layout) with an
8 pt floor, and the two data figures read results/*.csv rather than literals.

    python evaluation/generate_manuscript_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUTDIR = ROOT / "evaluation" / "manuscript_figures"

TEXT_PT = 390.0
W_IN = TEXT_PT / 72.0
BODY_PT = 8.0

# Okabe-Ito
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
YELLOW = "#E69F00"
SKY = "#56B4E9"
PINK = "#CC79A7"
GREY = "#4D4D4D"
INK = "#1A1A1A"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": BODY_PT,
        "axes.labelsize": BODY_PT + 1,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "axes.linewidth": 0.7,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, stem):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  wrote", (OUTDIR / f"{stem}.pdf").relative_to(ROOT))


# --------------------------------------------------------------------------
def fig1_architecture():
    """Four-stage pipeline: features -> six gates -> ACWCM -> outputs."""
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.72))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 74)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def block(x, y, w, h, title, sub, colour, tint):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=tint, edgecolor=colour, linewidth=0.9, zorder=3))
        ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="center",
                fontsize=BODY_PT, fontweight="bold", color=colour, zorder=4)
        for i, s in enumerate(sub):
            ax.text(x + w / 2, y + h - 8.4 - i * 4.0, s, ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=4)

    def down(x, y0, y1, colour=GREY, label=None):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=colour, zorder=2))
        if label:
            ax.text(x + 1.6, (y0 + y1) / 2, label, ha="left", va="center",
                    fontsize=BODY_PT, style="italic", color=GREY, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    # stage 1
    block(18, 62, 64, 11, "Patient presentation", ["52 clinical features"], BLUE, "#DCE9F5")
    # stage 2 -- the six knowledge modules
    gates = [("G1", "Red flags"), ("G2", "Cardiac"), ("G3", "Neuro"),
             ("G4", "Syndrome"), ("G5", "Temporal"), ("G6", "Uncertainty")]
    gw, ggap = 12.0, 2.4
    x0 = (100 - (6 * gw + 5 * ggap)) / 2
    for i, (tag, name) in enumerate(gates):
        x = x0 + i * (gw + ggap)
        down(x + gw / 2, 62, 52.4, BLUE)
        ax.add_patch(FancyBboxPatch((x, 40), gw, 12.0, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor="#EAF3EF", edgecolor=GREEN, linewidth=0.9, zorder=3))
        ax.text(x + gw / 2, 48.6, tag, ha="center", va="center",
                fontsize=BODY_PT, fontweight="bold", color=GREEN, zorder=4)
        ax.text(x + gw / 2, 43.4, name, ha="center", va="center",
                fontsize=BODY_PT, color=INK, zorder=4)
        ax.add_patch(FancyArrowPatch((x + gw / 2, 40), (50, 31.4), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9, color=GREEN, zorder=2))
    ax.text(50, 57.2, "six knowledge modules evaluated in parallel", ha="center", va="center",
            fontsize=BODY_PT, style="italic", color=GREY,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    ax.text(50, 35.0, "each emits a tier $r_i$ and a confidence $c_i$", ha="center", va="center",
            fontsize=BODY_PT, style="italic", color=GREY,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    # stage 3
    block(18, 18, 64, 13.0, "ACWCM", ["confidence-weighted conservative merging",
                                      "relaxes by at most one tier"], ORANGE, "#F7E2D5")
    down(50, 18, 13.0, ORANGE)
    # stage 4
    block(10, 1.5, 80, 11.0, "Outputs", ["final risk tier  ·  safety certificate  ·  audit trail"],
          PINK, "#F6E3EE")
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------
def fig2_risk_lattice():
    """The totally ordered lattice, drawn vertically so six tiers fit at 8 pt."""
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.62))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    tiers = [("R*", "Bottom (abstain)", "no decision", GREY, "#ECECEC"),
             ("R1", "Critical emergency", "5 min", ORANGE, "#F7DAD5"),
             ("R2", "High risk, urgent", "15 min", ORANGE, "#F9E6DA"),
             ("R3", "Moderate risk", "30–120 min", YELLOW, "#FBF0DA"),
             ("R4", "Low risk, monitor", "1–4 h", SKY, "#DFF0FB"),
             ("R5", "Minimal risk, discharge", "outpatient", GREEN, "#D9EFE8")]
    h, gap = 7.4, 2.1
    for i, (tag, name, sla, col, tint) in enumerate(tiers):
        y = 62 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((16, y), 60, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=tint, edgecolor=col, linewidth=0.9, zorder=3))
        ax.text(20, y + h / 2, tag, ha="left", va="center",
                fontsize=BODY_PT, fontweight="bold", color=col, zorder=4)
        ax.text(30, y + h / 2, name, ha="left", va="center", fontsize=BODY_PT, color=INK, zorder=4)
        ax.text(74, y + h / 2, sla, ha="right", va="center", fontsize=BODY_PT, color=GREY, zorder=4)
        if i:
            ax.text(14, y + h + gap / 2, r"$\sqsubseteq$", ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=4)

    ax.add_patch(FancyArrowPatch((11, 62 - h), (11, 1.0), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=INK, zorder=2))
    ax.text(8.2, 46, "more conservative", rotation=90, ha="center", va="center",
            fontsize=BODY_PT, color=GREY)
    ax.text(8.2, 14, "less cautious", rotation=90, ha="center", va="center",
            fontsize=BODY_PT, color=GREY)
    ax.text(79.0, 31, "merging selects the\nmost cautious tier;\nACWCM may relax it\nby one step",
            ha="left", va="center", fontsize=BODY_PT, color=GREY)
    save(fig, "fig2_risk_lattice")


# --------------------------------------------------------------------------
def fig3_baseline_sensitivity():
    data = rows("baseline_comparison.csv")
    labels = [r["method"].replace("Arithmetic ensemble averaging", "Arithmetic\naveraging")
              .replace("Dempster-Shafer", "Dempster–\nShafer")
              .replace("Bayesian model averaging", "Bayesian model\naveraging")
              .replace("Single XGBoost", "Single\nXGBoost")
              .replace("SAFE-Gate (ACWCM)", "SAFE-Gate\n(ACWCM)") for r in data]
    vals = [float(r["critical_sensitivity"]) for r in data]
    colours = [BLUE if "SAFE-Gate" in r["method"] else "#9AA5AD" for r in data]

    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.50))
    bars = ax.bar(range(len(vals)), vals, color=colours, edgecolor=INK, linewidth=0.6, width=0.62)
    ax.axhline(100, color=ORANGE, linestyle="--", linewidth=1.0,
               label="100% deployment threshold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}", ha="center",
                va="bottom", fontsize=BODY_PT)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=BODY_PT)
    ax.set_ylabel("Critical-tier sensitivity, R1–R2 (%)")
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig3_sensitivity")


# --------------------------------------------------------------------------
def fig4_ablation():
    data = rows("ablation.csv")
    cfg = [r["configuration"].replace("Full ACWCM (6 gates)", "Full\n(6 gates)") for r in data]
    sens = [float(r["critical_sensitivity_pct"]) for r in data]
    spec = [float(r["discharge_specificity_R5_pct"]) for r in data]
    f1 = [float(r["macro_f1_pct"]) for r in data]

    x = np.arange(len(cfg))
    w = 0.26
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.52))
    ax.bar(x - w, sens, w, label="Critical sensitivity", color=BLUE, edgecolor=INK, linewidth=0.5)
    ax.bar(x, spec, w, label="Discharge specificity (R5)", color=YELLOW, edgecolor=INK, linewidth=0.5)
    ax.bar(x + w, f1, w, label="Macro F1", color=GREEN, edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg, fontsize=BODY_PT)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig4_ablation")


def main():
    style()
    fig1_architecture()
    fig2_risk_lattice()
    fig3_baseline_sensitivity()
    fig4_ablation()
    print("four manuscript figures at %.0f pt, 8 pt floor" % TEXT_PT)


if __name__ == "__main__":
    main()
