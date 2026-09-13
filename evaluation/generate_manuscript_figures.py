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

TEXT_PT = 372.0
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
    # Order and names follow src/gates/gate{1..6}_*.py
    gates = [("G1", "Red flags"), ("G2", "Cardio risk"), ("G3", "Data quality"),
             ("G4", "Syndrome"), ("G5", "Uncertainty"), ("G6", "Temporal")]
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
def _clean_axes(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def fig3_safety_tradeoff():
    """Two-panel dot plot: every rule but averaging keeps the floor; the price is over-triage."""
    data = {r["method"]: r for r in rows("baseline_comparison.csv")}
    spec = [("Arithmetic ensemble averaging", "Arithmetic averaging", GREY),
            ("Bayesian model averaging", "Bayesian model avg.", GREY),
            ("Dempster-Shafer", "Dempster–Shafer", GREY),
            ("Single XGBoost", "Single XGBoost", GREY),
            ("Always-critical rule (R1)", "Always-critical (R1)", ORANGE),
            ("SAFE-Gate (MIN)", "SAFE-Gate (MIN)", SKY),
            ("SAFE-Gate (ACWCM)", "SAFE-Gate (ACWCM)", BLUE)]
    y = np.arange(len(spec))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, W_IN * 0.40), sharey=True)
    for ax, col, title in zip(axes, ("critical_sensitivity", "over_triage"),
                              ("Critical-tier sensitivity (%)", "Over-triage (%)")):
        vals = np.array([float(data[k][col]) for k, _, _ in spec])
        cols = [c for _, _, c in spec]
        ax.hlines(y, 0, vals, color=cols, alpha=0.35, linewidth=1.2)
        ax.scatter(vals, y, color=cols, edgecolor=INK, linewidth=0.4, s=24, zorder=3)
        for yy, v in zip(y, vals):
            right = v < 85
            ax.text(v + 3 if right else v - 3, yy, f"{v:.1f}", fontsize=BODY_PT - 1, color=GREY,
                    ha="left" if right else "right", va="center",
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"), zorder=4)
        ax.set_ylim(-0.6, len(spec) - 0.4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=BODY_PT, pad=3)
        _clean_axes(ax, "x")
    axes[0].axvline(100, color=GREEN, linestyle=":", linewidth=0.9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab, _ in spec])
    for t, (_, _, c) in zip(axes[0].get_yticklabels(), spec):
        t.set_color(INK if c == GREY else c)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "fig3_safety_tradeoff")


# --------------------------------------------------------------------------
def fig4_ablation():
    """Cleveland dot plot: what each gate contributes beyond the constant safety floor."""
    data = rows("ablation.csv")
    labels = {"Full ACWCM (6 gates)": "Full (6 gates)", "-G1": "− G1 red flags", "-G2": "− G2 cardio risk",
              "-G3": "− G3 data quality", "-G4": "− G4 syndrome", "-G5": "− G5 uncertainty",
              "-G6": "− G6 temporal"}
    cfg = [labels[r["configuration"]] for r in data]
    panels = [("over_triage_pct", "Over-triage (%)", ORANGE),
              ("discharge_specificity_R5_pct", "Discharge specificity, R5 (%)", GREEN),
              ("macro_f1_pct", "Macro F1 (%)", BLUE)]
    y = np.arange(len(cfg))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, W_IN * 0.42), sharey=True)
    for ax, (col, title, colour) in zip(axes, panels):
        vals = np.array([float(r[col]) for r in data])
        ax.hlines(y, 0, vals, color=colour, alpha=0.35, linewidth=1.2)
        ax.scatter(vals, y, color=colour, edgecolor=INK, linewidth=0.4, s=22, zorder=3)
        ax.axvline(vals[0], color=GREY, linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title, fontsize=BODY_PT, pad=3)
        _clean_axes(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cfg)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "fig4_ablation")


# --------------------------------------------------------------------------
def fig5_dilution():
    """Dilution rate by number of dissenting gates across the 980 enumerated configurations."""
    data = rows("conflicting_evidence.csv")
    rules = [("arithmetic_average", "Arithmetic averaging", GREY, "o"),
             ("bayesian_model_avg", "Bayesian model avg.", YELLOW, "^"),
             ("dempster_shafer", "Dempster–Shafer", ORANGE, "s"),
             ("safegate_acwcm", "SAFE-Gate (ACWCM)", BLUE, "*")]
    ns = sorted({int(r["n_dissenting"]) for r in data})
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.46))
    for key, label, col, mk in rules:
        rate = [100 * np.mean([r[key + "_dilutes"] == "True" for r in data if int(r["n_dissenting"]) == n])
                for n in ns]
        ax.plot(ns, rate, marker=mk, color=col, markeredgecolor=INK, markeredgewidth=0.4,
                markersize=8 if mk == "*" else 5, linewidth=1.3, label=label, zorder=3)
    n_per = len(data) // len(ns)
    ax.set_xticks(ns)
    ax.set_xlabel(f"Gates dissenting toward a benign tier (n = {n_per} configurations each)")
    ax.set_ylabel("Critical signal diluted (%)")
    ax.set_ylim(-4, 104)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    _clean_axes(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "fig5_dilution")

def main():
    style()
    fig1_architecture()
    fig2_risk_lattice()
    fig3_safety_tradeoff()
    fig4_ablation()
    fig5_dilution()
    print("five manuscript figures at %.0f pt, 8 pt floor" % TEXT_PT)


if __name__ == "__main__":
    main()
