"""Trade-off figure for the IJAR revision: over-triage vs R3 under-triage of the
ACWCM quota family (theta in {0, 1/2, 1}, mu in 0.05..1), the minimum and the
any-dissent extreme, read from results/ijar_r3_trigger_sweep.csv.
Run: python scripts/ijar_r3_figure.py OUT.pdf"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results" / "ijar_r3_trigger_sweep.csv", encoding="utf-8")))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "ijar" / "fig_tradeoff.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
titles = {"fresh_independent_generator": "(a) Independent generator",
          "fresh_diagnosis_generator": "(b) Diagnosis-conditioned generator"}
style = {"0.0": ("#9ecae1", "o"), "0.5": ("#3182bd", "o"), "1.0": ("#08519c", "s")}
for ax, (coh, ttl) in zip(axes, titles.items()):
    pts = [r for r in rows if r["cohort"] == coh]
    for th, (col, mk) in style.items():
        s = sorted([r for r in pts if r["theta"] == th and r["mu"] != "0.6667"], key=lambda r: float(r["mu"]))
        ax.plot([float(r["over_triage_pct"]) for r in s], [float(r["r3_under_triage_pct"]) for r in s],
                marker=mk, ms=2.5, lw=0.8, color=col, label=fr"quota, $\theta={th}$")
    for name, lab, mk, col in [("MIN", "minimum", "D", "black"),
                               ("Any-dissent (band-bounded extreme)", "any-dissent", "^", "grey")]:
        r = next(r for r in pts if r["theta"] == name)
        ax.plot(float(r["over_triage_pct"]), float(r["r3_under_triage_pct"]), mk, color=col, ms=5, label=lab)
    v = next(r for r in pts if r["theta"] == "0.5" and r["mu"] == "0.6667")
    ax.plot(float(v["over_triage_pct"]), float(v["r3_under_triage_pct"]), "*", color="#d62728", ms=9,
            label=r"ACWCM v2 ($\theta=1/2,\mu=2/3$)")
    ax.set_title(ttl, fontsize=8)
    ax.set_xlabel("Over-triage (%)")
    ax.grid(alpha=0.3, lw=0.4)
axes[0].set_ylabel("R3 under-triage (%)")
axes[0].legend(fontsize=6, loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(out)
print("written", out)
