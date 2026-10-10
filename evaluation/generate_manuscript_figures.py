"""Draw the five manuscript figures at the journal text-block width.

The figures that shipped with the manuscript were 800-1000 pt wide and were
squeezed into a 390 pt column, which left their smallest labels at 2.4-4.9 pt.
Worse, they had no generator in this repository at all, so nothing tied them to
the computed results. Both problems are fixed here: every figure is authored at
the measured widths of the sn-jnl [iicol] layout recommended by the journal
(455 pt text block for two-column figures, 216 pt column for Fig. 2) with an
8 pt floor,
and the data figures read results/* rather than literals. Proportions estimated
on the 805-case split carry exact 95% Clopper-Pearson intervals; the conflicting-
evidence figure is an exhaustive enumeration and therefore carries none.

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
from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
OUTDIR = ROOT / "evaluation" / "manuscript_figures"

TEXT_PT = 455.24   # \the\textwidth, sn-jnl [iicol]
COL_PT = 216.24    # \the\columnwidth
W_IN = TEXT_PT / 72.0
C_IN = COL_PT / 72.0
BODY_PT = 8.0
N_TEST = 805

# Okabe-Ito
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
YELLOW = "#E69F00"
SKY = "#56B4E9"
PINK = "#CC79A7"
GREY = "#4D4D4D"
LIGHT = "#9A9A9A"
INK = "#1A1A1A"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BODY_PT,
        "axes.labelsize": BODY_PT,
        "axes.titlesize": BODY_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, stem):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=600 if ext == "png" else 300, facecolor="white")
    plt.close(fig)
    print("  wrote", (OUTDIR / f"{stem}.pdf").relative_to(ROOT))


def cp_interval(x, n, alpha=0.05):
    """Exact Clopper-Pearson interval, in percent."""
    lo = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return 100 * lo, 100 * hi


def panel_label(ax, tag, x=-0.02, y=1.0):
    ax.text(x, y, f"({tag})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=BODY_PT + 1, fontweight="bold", color=INK)


def _clean_axes(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def fig1_architecture():
    """Four-stage pipeline: features -> six gates -> ACWCM (three phases) -> outputs."""
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.62))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def box(x, y, w, h, colour, tint, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=tint, edgecolor=colour, linewidth=lw, zorder=3))

    def arrow(p0, p1, colour, ms=8, lw=0.9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                     linewidth=lw, color=colour, zorder=2, shrinkA=0, shrinkB=0))

    def note(x, y, s, ha="left"):
        ax.text(x, y, s, ha=ha, va="center", fontsize=BODY_PT, style="italic", color=GREY, zorder=5)

    # stage 1: input
    box(22, 70, 56, 9, BLUE, "#DCE9F5")
    ax.text(50, 76.4, "Patient presentation", ha="center", va="center",
            fontsize=BODY_PT, fontweight="bold", color=BLUE, zorder=4)
    ax.text(50, 72.6, "52 clinical features", ha="center", va="center", fontsize=BODY_PT, zorder=4)

    # stage 2: six knowledge modules (order and names follow src/gates/gate{1..6}_*.py)
    gates = [("G1", "Red flags", "rules"), ("G2", "Cardio risk", "statistical"),
             ("G3", "Data quality", "abstention"), ("G4", "Syndrome", "rules"),
             ("G5", "Uncertainty", "Bayesian"), ("G6", "Temporal", "state machine")]
    gw, ggap = 13.4, 1.9
    x0 = (100 - (6 * gw + 5 * ggap)) / 2
    bus_top, bus_bot = 64.5, 41.5
    ax.plot([x0 + gw / 2, 100 - x0 - gw / 2], [bus_top, bus_top], color=BLUE, lw=0.9, zorder=2)
    ax.plot([50, 50], [70, bus_top], color=BLUE, lw=0.9, zorder=2)
    ax.plot([x0 + gw / 2, 100 - x0 - gw / 2], [bus_bot, bus_bot], color=GREEN, lw=0.9, zorder=2)
    for i, (tag, name, kind) in enumerate(gates):
        x = x0 + i * (gw + ggap)
        cx = x + gw / 2
        arrow((cx, bus_top), (cx, 59.0), BLUE)
        box(x, 46, gw, 13, GREEN, "#EAF3EF")
        ax.text(cx, 55.6, tag, ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", color=GREEN, zorder=4)
        ax.text(cx, 51.9, name, ha="center", va="center", fontsize=BODY_PT, zorder=4)
        ax.text(cx, 48.6, kind, ha="center", va="center", fontsize=BODY_PT - 1,
                style="italic", color=GREY, zorder=4)
        ax.plot([cx, cx], [46, bus_bot], color=GREEN, lw=0.9, zorder=2)
    note(51.5, 67.2, "evaluated in parallel")
    arrow((50, bus_bot), (50, 35.0), GREEN)
    note(51.5, 38.4, r"each gate emits $(r_i, c_i)$: a tier on $\mathcal{R}$ and a confidence")

    # stage 3: ACWCM with its three phases
    box(8, 14, 84, 21, ORANGE, "#F7E2D5")
    ax.text(50, 32.0, "ACWCM: adaptive confidence-weighted conservative merging",
            ha="center", va="center", fontsize=BODY_PT, fontweight="bold", color=ORANGE, zorder=4)
    phases = [("Phase 1", r"any $r_i \in \{R^*, R1\}$", r"return it ($R^*$ first)"),
              ("Phase 2", r"$r_{\min}$ = most cautious $r_i$", "safety floor (MIN)"),
              ("Phase 3", "weighted-majority consensus", r"relax $\leq$1 tier; R2 held")]
    pw, pgap = 24.5, 2.5
    px0 = (100 - (3 * pw + 2 * pgap)) / 2
    for j, (head, l1, l2) in enumerate(phases):
        x = px0 + j * (pw + pgap)
        box(x, 16, pw, 12.5, ORANGE, "white", lw=0.6)
        ax.text(x + pw / 2, 25.6, head, ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + pw / 2, 21.9, l1, ha="center", va="center", fontsize=BODY_PT, zorder=4)
        ax.text(x + pw / 2, 18.6, l2, ha="center", va="center", fontsize=BODY_PT,
                color=GREY, zorder=4)
        if j:
            arrow((x - pgap + 0.2, 22.25), (x - 0.2, 22.25), ORANGE, ms=6, lw=0.7)
    arrow((50, 14), (50, 10.0), ORANGE)

    # stage 4: outputs
    box(14, 1.0, 72, 9, PINK, "#F6E3EE")
    ax.text(50, 7.3, "Outputs", ha="center", va="center", fontsize=BODY_PT,
            fontweight="bold", color=PINK, zorder=4)
    ax.text(50, 3.6, r"final tier $\hat r$  ·  counterfactual safety certificate  ·  audit trail",
            ha="center", va="center", fontsize=BODY_PT, zorder=4)
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------
def fig2_risk_lattice():
    """The totally ordered lattice at single-column width; annotations live in the caption."""
    fig, ax = plt.subplots(figsize=(C_IN, C_IN * 0.92))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    tiers = [(r"$R^{*}$", "Abstain (defer)", "", GREY, "#ECECEC"),
             ("R1", "Critical", "5 min", ORANGE, "#F7DAD5"),
             ("R2", "High risk", "15 min", ORANGE, "#F9E6DA"),
             ("R3", "Moderate", "30–120 min", YELLOW, "#FBF0DA"),
             ("R4", "Low risk", "1–4 h", SKY, "#DFF0FB"),
             ("R5", "Discharge", "outpatient", GREEN, "#D9EFE8")]
    h, gap = 7.6, 2.3
    ys = []
    for i, (tag, name, sla, col, tint) in enumerate(tiers):
        y = 62 - (i + 1) * h - i * gap
        ys.append(y)
        ax.add_patch(FancyBboxPatch((15, y), 72, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=tint, edgecolor=col, linewidth=0.8, zorder=3))
        ax.text(18, y + h / 2, tag, ha="left", va="center",
                fontsize=BODY_PT, fontweight="bold", color=col, zorder=4)
        ax.text(30, y + h / 2, name, ha="left", va="center", fontsize=BODY_PT, color=INK, zorder=4)
        ax.text(85, y + h / 2, sla, ha="right", va="center", fontsize=BODY_PT, color=GREY, zorder=4)
        if i:
            ax.text(12.2, y + h + gap / 2, r"$\sqsubseteq$", ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=4)

    ax.add_patch(FancyArrowPatch((7.5, 62 - h), (7.5, 1.0), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.8, color=INK, zorder=2))
    ax.text(3.2, 31, "more conservative  →  less cautious", rotation=-90, ha="center", va="center",
            fontsize=BODY_PT - 1, color=GREY)
    yc0, yc1 = ys[2], ys[1] + h
    ax.plot([89, 91, 91, 89], [yc1, yc1, yc0, yc0], color=ORANGE, lw=0.9)
    ax.text(95.5, (yc0 + yc1) / 2, r"$\mathcal{C}$", ha="center", va="center",
            fontsize=BODY_PT + 1, color=ORANGE)
    save(fig, "fig2_risk_lattice")


# --------------------------------------------------------------------------
def fig3_safety_tradeoff():
    """Three-panel dot plot with 95% CIs: the safety floor, its over-triage price and
    the under-triage each rule incurs (hatched share = errors of two or more tiers)."""
    data = {r["method"]: r for r in rows("baseline_comparison.csv")}
    spec = [("Arithmetic ensemble averaging", "Arithmetic averaging", GREY),
            ("Bayesian model averaging", "Bayesian model avg.", GREY),
            ("Dempster-Shafer", "Dempster–Shafer", GREY),
            ("Single XGBoost", "Single XGBoost", GREY),
            ("Always-critical rule (R1)", "Always-critical (R1)", ORANGE),
            ("SAFE-Gate (MIN)", "SAFE-Gate (MIN)", SKY),
            ("SAFE-Gate (ACWCM, any-dissent trigger)", "ACWCM, any-dissent", SKY),
            ("SAFE-Gate (ACWCM)", "SAFE-Gate (ACWCM)", BLUE)]
    y = np.arange(len(spec))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, W_IN * 0.42), sharey=True)
    titles = ("Critical-tier sensitivity (%)", "Over-triage (%)", "Under-triage (%)")
    for p, (ax, title) in enumerate(zip(axes, titles)):
        for (key, _, colour), yy in zip(spec, y):
            r = data[key]
            if p == 0:
                x, n = int(r["critical_caught"]), int(r["critical_total"])
            elif p == 1:
                x, n = round(float(r["over_triage"]) * N_TEST / 100), N_TEST
            else:
                x, n = round(float(r["under_triage"]) * N_TEST / 100), N_TEST
            v = 100 * x / n
            shown = float(r[("critical_sensitivity", "over_triage", "under_triage")[p]])
            lo, hi = cp_interval(x, n)
            ax.hlines(yy, 0, v, color=colour, alpha=0.30, linewidth=1.1)
            if p == 2 and int(r["under_triage_2plus_tiers"]):
                v2 = 100 * int(r["under_triage_2plus_tiers"]) / n
                ax.barh(yy, v2, height=0.42, color="white", edgecolor=ORANGE,
                        hatch="/////", linewidth=0.5, zorder=2)
            ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=1.6, zorder=3)
            ax.scatter(v, yy, color=colour, edgecolor=INK, linewidth=0.4, s=20, zorder=4)
            right = v < 72
            ax.text(hi + 3 if right else lo - 3, yy, f"{round(shown, 1):.1f}", fontsize=BODY_PT - 1, color=GREY,
                    ha="left" if right else "right", va="center", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"))
        ax.set_ylim(-0.6, len(spec) - 0.4)
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_title(f"({'abc'[p]}) {title}", pad=3)
        _clean_axes(ax, "x")
    axes[0].axvline(100, color=GREEN, linestyle=":", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab, _ in spec])
    for t, (_, _, c) in zip(axes[0].get_yticklabels(), spec):
        t.set_color(INK if c == GREY else c)
    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "fig3_safety_tradeoff")


# --------------------------------------------------------------------------
def fig4_ablation():
    """Cleveland dot plot: what each gate contributes beyond the constant safety floor."""
    data = rows("ablation.csv")
    labels = {"Full ACWCM (6 gates)": "Full (6 gates)", "-G1": "− G1 red flags", "-G2": "− G2 cardio risk",
              "-G3": "− G3 data quality", "-G4": "− G4 syndrome", "-G5": "− G5 uncertainty",
              "-G6": "− G6 temporal", "-G1-G4": "− G1, − G4"}
    cfg = [labels[r["configuration"]] for r in data]
    panels = [("over_triage_pct", "Over-triage (%)", ORANGE),
              ("discharge_specificity_R5_pct", "Discharge recall, R5 (%)", GREEN),
              ("under_triage_pct", "Under-triage (%)", PINK),
              ("macro_f1_pct", "Macro F1 (%)", BLUE)]
    y = np.arange(len(cfg))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(W_IN, W_IN * 0.40), sharey=True)
    for p, (ax, (col, title, colour)) in enumerate(zip(axes, panels)):
        vals = np.array([float(r[col]) for r in data])
        ax.axhspan(y[0] - 0.45, y[0] + 0.45, color="#EFEFEF", zorder=0)
        ax.hlines(y, 0, vals, color=colour, alpha=0.35, linewidth=1.1)
        ax.scatter(vals, y, color=colour, edgecolor=INK, linewidth=0.4, s=20, zorder=3)
        ax.axvline(vals[0], color=GREY, linestyle=":", linewidth=0.8)
        for yy, v, base in zip(y[1:], vals[1:], [vals[0]] * (len(vals) - 1)):
            if abs(v - base) >= 5:
                left = (v < base and v >= 20) or v >= 80
                ax.text(v - 5 if left else v + 5, yy, f"{v:.1f}", fontsize=BODY_PT - 1,
                        color=GREY, ha="right" if left else "left", va="center")
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_title(f"({'abcd'[p]}) {title}", pad=3)
        _clean_axes(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cfg)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig4_ablation")


# --------------------------------------------------------------------------
def fig5_dilution():
    """(a) dilution rate and (b) R1-to-discharge configurations, by dissenting gates."""
    data = rows("conflicting_evidence.csv")
    rules = [("arithmetic_average", "Arithmetic averaging", GREY, "o"),
             ("bayesian_model_avg", "Bayesian model avg.", YELLOW, "^"),
             ("dempster_shafer", "Dempster–Shafer", ORANGE, "s"),
             ("safegate_acwcm", "SAFE-Gate (ACWCM)", BLUE, "*")]
    ns = sorted({int(r["n_dissenting"]) for r in data})
    n_r1 = sum(1 for r in data if r["critical_tier"] == "R1" and int(r["n_dissenting"]) == ns[0])
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, W_IN * 0.40))
    for key, label, col, mk in rules:
        rate = [100 * np.mean([r[key + "_dilutes"] == "True" for r in data
                               if int(r["n_dissenting"]) == n]) for n in ns]
        r1_dis = [sum(1 for r in data if int(r["n_dissenting"]) == n and r["critical_tier"] == "R1"
                      and r[key] == "R5") for n in ns]
        kw = dict(marker=mk, color=col, markeredgecolor=INK, markeredgewidth=0.4,
                  markersize=8 if mk == "*" else 4.5, linewidth=1.2, label=label, zorder=3)
        axes[0].plot(ns, rate, **kw)
        axes[1].plot(ns, r1_dis, **kw)
    axes[0].set_ylabel("Configurations diluted (%)")
    axes[0].set_ylim(-4, 104)
    axes[1].set_ylabel(f"R1 signal sent to R5 (of {n_r1})")
    axes[1].set_ylim(-2, max(n_r1 * 0.5, 10))
    for p, ax in enumerate(axes):
        ax.set_xticks(ns)
        ax.set_xlabel("Gates dissenting toward a benign tier")
        _clean_axes(ax)
        ax.set_title(("(a) Share of configurations diluted", "(b) Life-threatening signal discharged")[p], pad=3)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0, 1, 0.88))
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
