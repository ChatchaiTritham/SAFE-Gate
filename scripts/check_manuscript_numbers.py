#!/usr/bin/env python3
"""Check that headline numbers quoted in the IJFS manuscript match results/.

Each check recomputes a value from the committed result files, formats it as
the manuscript does and asserts that the formatted string occurs in the .tex
file. Usage: python scripts/check_manuscript_numbers.py path/to/sn-article.tex
"""
import csv
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
tex = Path(sys.argv[1]).read_text(encoding="utf-8")
S = json.loads((RES / "round4_summary.json").read_text(encoding="utf-8"))
prim = {(r["cohort"], r["method"]): r for r in csv.DictReader(open(RES / "round4_primary.csv", encoding="utf-8"))}
dev = {r["method"]: r for r in csv.DictReader(open(RES / "round4_development.csv", encoding="utf-8"))}
conf = json.loads((RES / "conflicting_evidence.json").read_text(encoding="utf-8"))
I, D, M = "fresh_independent_generator", "fresh_diagnosis_generator", "fresh_main_generator"


def f1(x):
    """One decimal, half-up on the decimal string (as typeset)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def ci(r, key="critical_ci95"):
    lo, hi = json.loads(r[key])
    return f"{f1(lo)}--{f1(hi)}"


TABLE_METHODS = {"SAFE-Gate ACWCM (v2)", "ACWCM fuzzy-quantifier trigger", "SAFE-Gate ACWCM (v1)", "SAFE-Gate MIN",
                 "Weighted max (normalised possibility)", "DS upper-set (Bel median)", "BMA min expected cost",
                 "Ordinal OWA (k-th order statistic), k=2", "Band-clamped Sugeno integral",
                 "Sugeno (quantised additive capacity)", "Quantifier-guided OWA (Yager 1996, 'most')",
                 "TSK fuzzy rule base (zero order)", "Dempster-Shafer", "DS cautious rule (Denoeux)",
                 "DS min expected cost", "Bayesian model averaging", "XGBoost",
                 "XGBoost (safety threshold, val >=99%)", "Ordinal logistic regression"}
checks = []
a_i, a_d = prim[(I, "SAFE-Gate ACWCM (v2)")], prim[(D, "SAFE-Gate ACWCM (v2)")]
checks += [f1(a_i["critical_sensitivity_pct"]) + "\\%", ci(a_i), f1(a_d["critical_sensitivity_pct"]) + "\\%", ci(a_d),
           f1(a_i["over_triage_pct"]), f1(a_d["over_triage_pct"]), f1(a_i["r3_under_triage_pct"]),
           f1(a_d["r3_under_triage_pct"]), ci(a_d, "r3_under_ci95"),
           f1(prim[(I, "Dempster-Shafer")]["critical_sensitivity_pct"]),
           f1(prim[(D, "Dempster-Shafer")]["critical_sensitivity_pct"])]
for coh in (I, D):
    for m, r in ((k[1], v) for k, v in prim.items() if k[0] == coh and k[1] in TABLE_METHODS):
        checks.append(f"{f1(r['critical_sensitivity_pct'])} ({ci(r)}) & {r['false_discharges']} & {f1(r['accuracy_pct'])}")
m = prim[(M, "SAFE-Gate ACWCM (v2)")]
checks += [f"{int(m['n_critical']):,}".replace(",", "{,}"), f1(m["accuracy_pct"]) + "\\%", f1(m["r3_under_triage_pct"]) + "\\%"]
ms = S["multiseed"]
checks += [f"{f1(ms['accuracy_pct']['mean'])}\\pm{f1(ms['accuracy_pct']['sd'])}",
           f"{f1(ms['r3_under_triage_pct']['mean'])}\\pm{f1(ms['r3_under_triage_pct']['sd'])}"]
d = conf["dilution_counts"]
checks += [f"dilutes in {d['dempster_shafer']} configurations", f"and {d['dempster_shafer_tie_benign']} with benign",
           f"to R5 in {conf['r1_to_r5_counts']['dempster_shafer']} ({conf['r1_to_r5_counts']['dempster_shafer_tie_benign']})"]
for k, v in S["conflict"]["rules"].items():
    if k == "SAFE-Gate ACWCM (v2)":
        checks.append(f"ACWCM & {v['dilutions_980']} ({v['r1_to_r5_980']}) & {v['dilutions_hetero']} & {v['escalations_false_alarm']}")
ni = S["noninferiority_vs_DS_diagnosis"]
checks += [f"{ni['ci95'][0]:.1f}--{ni['ci95'][1]:.1f}"]
dv = dev["SAFE-Gate ACWCM (v2)"]
checks += [ci(dv, "r3_under_ci95"), f1(dv["r3_under_triage_pct"]) + "\\%"]
gr = S["gate_recall_union"]
checks += [f1(gr[I]["some_gate_fires_pct"]) + "\\%", f1(gr[D]["some_gate_fires_pct"]) + "\\%"]
comp = S["diagnosis_composition"]
checks += [f"{sum(comp['acwcm_missed_critical_by_diagnosis'].values())} of the 731",
           f"vestibular neuritis ({comp['acwcm_under_triaged_R3_by_diagnosis']['vestibular_neuritis']})"]
pa = S["profile_analysis"]
checks += [f1(pa["robust_row"][D]["critical_sensitivity_pct"]) + "\\%", f1(pa["robust_row"]["train"]["accuracy_pct"]) + "\\%"]
r1 = S["r1_only_via_g6"][M]
checks += [f"Among the {r1['predicted_R1']} cases", f"{r1['R1_only_via_G6']} were R1 only"]

bad = [c for c in checks if c not in tex]
print(f"{len(checks)} checks, {len(bad)} mismatches")
for c in bad:
    print("  MISSING:", c)
sys.exit(1 if bad else 0)
