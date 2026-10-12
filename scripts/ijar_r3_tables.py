"""LaTeX rows of Supplementary Table S11 (IJAR revision): all comparators not in
the main-text table, read from results/round4_primary.csv and
results/ijar_r3_cohort_rules.csv. Run: python scripts/ijar_r3_tables.py > OUT.tex"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
P = list(csv.DictReader(open(RES / "round4_primary.csv", encoding="utf-8")))
N = list(csv.DictReader(open(RES / "ijar_r3_cohort_rules.csv", encoding="utf-8")))
ROWS = [(P, "SAFE-Gate ACWCM (v1)", r"ACWCM (v1)"),
        (N, "Any-dissent (band-bounded extreme)", r"Any-dissent"),
        (P, "Weighted max (forced top, original)", r"W.\ max, forced top"),
        (P, "Sugeno (quantised additive capacity)", r"Sugeno, $\nu_+$"),
        (P, "Band-clamped Sugeno integral", r"Band-clamped Sugeno"),
        (P, "Ordinal OWA (k-th order statistic), k=2", r"$k$-th order stat."),
        (P, "Quantifier-guided OWA (Yager 1996, 'most')", r"Quantifier OWA"),
        (P, "Prioritized aggregation (Yager 2008)", r"Prioritized"),
        (P, "TSK fuzzy rule base (zero order)", r"TSK rule base"),
        (P, "Dempster-Shafer", r"DS Dempster (sequential)"),
        (N, "DS Dempster exact, conf<=0.999", r"DS Dempster, $c\le0.999$"),
        (N, "DS Dempster exact (upper expected cost)", r"DS Dempster, upper cost"),
        (P, "DS cautious rule (Denoeux)", r"DS cautious"),
        (N, "DS cautious, conf<=0.999", r"DS cautious, $c\le0.999$"),
        (P, "DS min expected cost", r"DS min.\ exp.\ cost"),
        (N, "DS disjunctive (BetP, critical ties)", r"DS disjunctive, BetP"),
        (N, "DS disjunctive (BetP, benign ties)", r"DS disj., BetP, benign ties"),
        (N, "DS disjunctive (upper expected cost)", r"DS disjunctive, upper cost"),
        (P, "Bayesian model averaging", r"BMA"),
        (P, "XGBoost", r"XGBoost"),
        (P, "XGBoost (safety threshold, val >=99%)", r"XGBoost, safety thr.")]
BS2 = "\\" + "\\"


def f1(x):
    return f"{float(x):.1f}"


for coh, lab in [("fresh_independent_generator", "(a) Independent generator"),
                 ("fresh_diagnosis_generator", "(b) Diagnosis-conditioned generator")]:
    print(r"\multicolumn{8}{@{}l}{\textit{" + lab + "}} " + BS2)
    for src, key, name in ROWS:
        r = next(x for x in src if x["cohort"] == coh and x["method"] == key)
        lo, hi = [float(v) for v in r["critical_ci95"].strip("[]").split(",")]
        print(f"{name} & {f1(r['critical_sensitivity_pct'])} ({lo:.1f}--{hi:.1f}) & {r['false_discharges']} & "
              f"{f1(r['accuracy_pct'])} & {f1(r['over_triage_pct'])} & {f1(r['under_triage_pct'])} & "
              f"{r['under_2plus']} & {f1(r['r3_under_triage_pct'])} " + BS2)
    if coh.startswith("fresh_indep"):
        print(r"\midrule")
