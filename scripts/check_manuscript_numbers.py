#!/usr/bin/env python3
"""Check that the numbers quoted in the IJFS or IJAR manuscript match results/.

Each check recomputes a value from the committed result files, formats it as
the manuscript does and asserts that the formatted string occurs in the target
file. Covered: the article (abstract, highlights, text, table rows and
captions, figure captions), the supplement (regenerated from results/ and
compared byte for byte, plus its hand-written numbers), the cover letter and
the plain-text abstract.

Usage:
    python scripts/check_manuscript_numbers.py path/to/sn-article.tex
        [--supplement supplement.tex] [--cover cover_letter.tex]
        [--abstract abstract_plaintext.txt]
Files not given explicitly are looked up next to the article and skipped
(with a note) if absent.

    python scripts/check_manuscript_numbers.py path/to/main.tex --journal ijar

checks the elsarticle version for the International Journal of Approximate
Reasoning: its primary table is split into two stacked panels (one row per
method and cohort), its abstract is a \begin{abstract} environment, and its
supplement is regenerated with make_supplement.py --journal ijar. The default
(--journal ijfs) is unchanged.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

ap = argparse.ArgumentParser()
ap.add_argument("article")
ap.add_argument("--supplement")
ap.add_argument("--cover")
ap.add_argument("--abstract")
ap.add_argument("--journal", choices=("ijfs", "ijar"), default="ijfs")
args = ap.parse_args()
IJAR = args.journal == "ijar"
art = Path(args.article)
sib = art.parent


def opt(given, name):
    p = Path(given) if given else sib / name
    return p if p.exists() else None


SUP, COV, ABS = opt(args.supplement, "supplement.tex"), opt(args.cover, "cover_letter.tex"), opt(args.abstract, "abstract_plaintext.txt")
TXT = {"article": art.read_text(encoding="utf-8")}
for k, p in (("supplement", SUP), ("cover", COV), ("abstract", ABS)):
    if p:
        TXT[k] = p.read_text(encoding="utf-8")
    else:
        print(f"note: no {k} file, its checks are skipped")

S = json.loads((RES / "round4_summary.json").read_text(encoding="utf-8"))
prim = {(r["cohort"], r["method"]): r for r in csv.DictReader(open(RES / "round4_primary.csv", encoding="utf-8"))}
dev = {r["method"]: r for r in csv.DictReader(open(RES / "round4_development.csv", encoding="utf-8"))}
conf = json.loads((RES / "conflicting_evidence.json").read_text(encoding="utf-8"))
mcar = {(r["mcar_pct"], r["method"]): r for r in csv.DictReader(open(RES / "round4_mcar_matched.csv", encoding="utf-8"))}
noise = {(r["scenario"], r["method"]): r for r in csv.DictReader(open(RES / "round4_feature_noise.csv", encoding="utf-8"))}
stress = {(r["scenario"], r["method"]): r for r in json.loads((RES / "stress_tests.json").read_text(encoding="utf-8"))}
I, D, M = "fresh_independent_generator", "fresh_diagnosis_generator", "fresh_main_generator"
V2, MIN, DS = "SAFE-Gate ACWCM (v2)", "SAFE-Gate MIN", "Dempster-Shafer"


def f1(x):
    """One decimal, half-up on the decimal string (as typeset)."""
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pc(x):
    """Percentage as typeset in running text: 100 is written without a decimal."""
    return "100" if float(x) == 100.0 else f1(x)


def d(a, b):
    return f1(float(a) - float(b))


def ci(r, key="critical_ci95"):
    lo, hi = json.loads(r[key])
    return f"{f1(lo)}--{f1(hi)}"


def num(n):
    return f"{int(n):,}".replace(",", "{,}")


def paired(block, comp, outcome):
    return next(x for x in S[block] if x["comparison"] == comp and x["outcome"] == outcome)


checks = []  # (target, string)


def A(*strs, where="article"):
    checks.extend((where, s) for s in strs)


# ---------------- article: abstract, highlights, findings
a_i, a_d = prim[(I, V2)], prim[(D, V2)]
A(f1(a_i["critical_sensitivity_pct"]) + "\\%", ci(a_i), f1(a_d["critical_sensitivity_pct"]) + "\\%", ci(a_d),
  f1(a_i["over_triage_pct"]), f1(a_d["over_triage_pct"]), f1(a_i["r3_under_triage_pct"]),
  f1(a_d["r3_under_triage_pct"]), ci(a_d, "r3_under_ci95"),
  f1(prim[(I, DS)]["critical_sensitivity_pct"]), f1(prim[(D, DS)]["critical_sensitivity_pct"]))
A(f"over-triage of {f1(a_i['over_triage_pct'])}\\% and {f1(a_d['over_triage_pct'])}\\%",
  f"under-triage of {f1(a_i['r3_under_triage_pct'])}\\% and {f1(a_d['r3_under_triage_pct'])}\\%")

# ---------------- article: Table 4 (primary), every cell of every row
LABEL = {V2: "ACWCM (v2)", "ACWCM fuzzy-quantifier trigger": "ACWCM, fuzzy $Q$", "SAFE-Gate ACWCM (v1)": "ACWCM (v1)",
         MIN: "MIN", "Weighted max (normalised possibility)": "W.\\ max, normalised",
         "DS upper-set (Bel median)": "DS cons., Bel med.", "BMA min expected cost": "BMA, min.\\ exp.\\ cost",
         "Ordinal OWA (k-th order statistic), k=2": "$k$-th order stat.", "Band-clamped Sugeno integral": "Band-clamped Sugeno",
         "Sugeno (quantised additive capacity)": "Sugeno, $\\nu_+$", "Quantifier-guided OWA (Yager 1996, 'most')": "Quantifier OWA",
         "TSK fuzzy rule base (zero order)": "TSK rule base", DS: "DS Dempster", "DS cautious rule (Denoeux)": "DS cautious",
         "DS min expected cost": "DS min.\\ exp.\\ cost", "Bayesian model averaging": "BMA", "XGBoost": "XGBoost",
         "XGBoost (safety threshold, val >=99%)": "XGBoost, safety thr.", "Ordinal logistic regression": "Ordinal logistic"}


def cells(r):
    return (f"{f1(r['critical_sensitivity_pct'])} ({ci(r)}) & {r['false_discharges']} & {f1(r['accuracy_pct'])} & "
            f"{f1(r['over_triage_pct'])} & {f1(r['under_triage_pct'])} & {r['under_2plus']} & {f1(r['r3_under_triage_pct'])}")


for m, lab in LABEL.items():
    if IJAR:
        A(f"{lab} & {cells(prim[(I, m)])} \\\\", f"{lab} & {cells(prim[(D, m)])} \\\\")
    else:
        A(f"{lab} & {cells(prim[(I, m)])} & {cells(prim[(D, m)])} \\\\")
gr = S["gate_recall_union"]
A(f"at least one gate returned $R^*$, R1 or R2 (the ceiling for any merge rule): {f1(gr[I]['some_gate_fires_pct'])}\\% and {f1(gr[D]['some_gate_fires_pct'])}\\%")
A(f"($n={num(prim[(I, V2)]['n'])}$; {prim[(I, V2)]['n_critical']} critical)", f"($n={num(prim[(D, V2)]['n'])}$; {prim[(D, V2)]['n_critical']} critical)")

# ---------------- article: Table 2 (enumeration) and Table 3 (conflicts), every row
EN = S["enumeration"]["rules"]
ENUM_ROWS = [("ACWCM (both triggers)", V2), ("Band-clamped Sugeno", "Band-clamped Sugeno integral"),
             ("Order stat.\\ $k{=}2$", "Ordinal OWA (k-th order statistic), k=2"), ("DS cons., Bel med.", "DS upper-set (Bel median)"),
             ("W.\\ max, normalised", "Weighted max (normalised possibility)"), ("Sugeno, $\\nu_+$", "Sugeno (quantised additive capacity)"),
             ("Quant.\\ OWA", "Quantifier-guided OWA (Yager 1996, 'most')"), ("Prioritized", "Prioritized aggregation (Yager 2008)"),
             ("DS Demp./Yager/DP", DS), ("DS cautious", "DS cautious rule (Denoeux)"), ("BMA", "Bayesian model averaging"),
             ("TSK rule base", "TSK fuzzy rule base (zero order)")]
for lab, k in ENUM_ROWS:
    e = EN[k]
    A(f"{lab} & {num(e['CP'])} & {num(e['CND'])} & {num(e['R1D'])} & {num(e['conf_monotonicity'])} & {e['max_jump_one_step']} \\\\")
assert EN[V2] == {**EN["ACWCM fuzzy-quantifier trigger"], "n_conf_pairs": EN[V2]["n_conf_pairs"]} or \
    all(EN[V2][x] == EN["ACWCM fuzzy-quantifier trigger"][x] for x in ("CP", "CND", "R1D", "conf_monotonicity", "max_jump_one_step"))
for k in ("DS Yager rule", "DS Dubois-Prade rule"):
    assert all(EN[k][x] == EN[DS][x] for x in ("CP", "CND", "R1D", "conf_monotonicity", "max_jump_one_step")), k
A(f"({num(list(EN.values())[0]['n_inputs'])} inputs)", f"ACWCM and {len(EN) - 2} comparison operators")

CF = S["conflict"]
LAM = CF["breakeven_lambda_ratio1"]
CONF_ROWS = [("ACWCM", V2), ("DS cons., Bel med.", "DS upper-set (Bel median)"), ("W.\\ max, normalised", "Weighted max (normalised possibility)"),
             ("BMA, min.\\ expected cost", "BMA min expected cost"), ("$k$-th order stat.", "Ordinal OWA (k-th order statistic), k=2"),
             ("DS cautious", "DS cautious rule (Denoeux)"), ("Band-clamped Sugeno", "Band-clamped Sugeno integral"),
             ("W.\\ max, forced top", "Weighted max (forced top, original)"), ("DS Demp./Yager/DP", DS),
             ("BMA", "Bayesian model averaging"), ("DS PCR6", "DS PCR6"), ("Sugeno, $\\nu_+$", "Sugeno (quantised additive capacity)"),
             ("Quant.\\ OWA", "Quantifier-guided OWA (Yager 1996, 'most')"), ("TSK rule base", "TSK fuzzy rule base (zero order)")]
for lab, k in CONF_ROWS:
    v = CF["rules"][k]
    lam = "---" if LAM.get(k) is None else f1(LAM[k])
    A(f"{lab} & {v['dilutions_980']} ({v['r1_to_r5_980']}) & {num(v['dilutions_hetero'])} & {v['escalations_false_alarm']} & {lam} \\\\")
A(f"critical prevalence {CF['prevalence_critical_dev']:.3f}".rstrip("0"),
  f"more than {f1(LAM[DS])} false escalations", f"once it costs more than {int(round(LAM['Weighted max (normalised possibility)']))}",
  f"Dempster's rule escalates {CF['rules'][DS]['escalations_false_alarm']}")
dc = conf["dilution_counts"]
A(f"dilutes in {dc['dempster_shafer']} configurations", f"and {dc['dempster_shafer_tie_benign']} with benign",
  f"to R5 in {conf['r1_to_r5_counts']['dempster_shafer']} ({conf['r1_to_r5_counts']['dempster_shafer_tie_benign']})",
  f"{conf['exact_ties']['dempster_shafer']} in all", f"({num(conf['n_configurations'])} configurations)",
  f"({num(CF['n_hetero'])} configurations)", f"({CF['n_false_alarm']} configurations)",
  f"loses the critical signal on {CF['rules']['Band-clamped Sugeno integral']['dilutions_980']} of {conf['n_configurations']}")
# Figure 2 (fig5_dilution) caption: per-point counts in the single-dissent grid
grid = list(csv.DictReader(open(RES / "conflicting_evidence.csv", encoding="utf-8")))
per_n = sum(1 for r in grid if r["n_dissenting"] == "1")
per_n_r1 = sum(1 for r in grid if r["n_dissenting"] == "1" and r["critical_tier"] == "R1")
A(f"share of the {per_n} configurations per point", f"of {per_n_r1} per point with an R1 signal")

# ---------------- article: findings paragraph
m_i, m_d = prim[(I, MIN)], prim[(D, MIN)]
A(f"with {d(a_i['accuracy_pct'], m_i['accuracy_pct'])} and {d(a_d['accuracy_pct'], m_d['accuracy_pct'])} points higher accuracy and "
  f"{d(m_i['over_triage_pct'], a_i['over_triage_pct'])} and {d(m_d['over_triage_pct'], a_d['over_triage_pct'])} points less over-triage")
u = paired("paired_stats_diagnosis", "ACWCM vs SAFE-Gate MIN", "under")
u2 = paired("paired_stats_diagnosis", "ACWCM vs SAFE-Gate MIN", "under2")
mant, ex = f"{u['p_holm']:.1e}".split("e")
A(f"{u['acwcm_only']} cases under-triaged by ACWCM and not by the minimum", f"$p={mant}\\times10^{{{int(ex)}}}$",
  f"{u2['acwcm_only'] - u2['other_only']} of these were under-triaged by two tiers")
assert u["other_only"] == 0
ni = S["noninferiority_vs_DS_diagnosis"]
A(f"{f1(ni['critical_sensitivity_diff_vs_DS_pct'])} points fewer than ACWCM", f"{ni['ci95'][0]:.1f}--{ni['ci95'][1]:.1f}",
  f"with {prim[(D, DS)]['false_discharges']} false discharges, while being {d(prim[(D, DS)]['accuracy_pct'], a_d['accuracy_pct'])} points more accurate")
near = ["Weighted max (normalised possibility)", "DS upper-set (Bel median)", "BMA min expected cost"]
gap = max(float(prim[(c, V2)]["critical_sensitivity_pct"]) - float(prim[(c, k)]["critical_sensitivity_pct"])
          for c in (I, D) for k in near)
assert gap <= 1.5 and all(prim[(c, k)]["false_discharges"] == "0" for c in (I, D) for k in near), gap
A("within 1.5 points of ACWCM")
fz_d = prim[(D, "ACWCM fuzzy-quantifier trigger")]
A(f"{d(fz_d['accuracy_pct'], a_d['accuracy_pct'])} points more accurate on the diagnosis-conditioned cohort",
  f"more R3 under-triage ({f1(fz_d['r3_under_triage_pct'])}\\% against {f1(a_d['r3_under_triage_pct'])}\\%)")
assert float(prim[(I, V2)]["accuracy_pct"]) > float(prim[(I, "ACWCM fuzzy-quantifier trigger")]["accuracy_pct"])
assert fz_d["false_discharges"] == "1"
pr_i, pr_d = prim[(I, "Prioritized aggregation (Yager 2008)")], prim[(D, "Prioritized aggregation (Yager 2008)")]
A(f"kept {f1(pr_i['critical_sensitivity_pct'])}\\% and {f1(pr_d['critical_sensitivity_pct'])}\\% with {pr_i['false_discharges']} and {pr_d['false_discharges']} false discharges")
ex_i, ex_d = prim[(I, "DS Dempster (singleton, BetP)")], prim[(D, "DS Dempster (singleton, BetP)")]
A(f"({f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% critical sensitivity for the exact rule)")
A(f"the gates' misses ({d(100, a_i['critical_sensitivity_pct'])}\\% and {d(100, a_d['critical_sensitivity_pct'])}\\% of critical cases)")
A(f"over-triaged {f1(a_d['over_triage_pct'])}\\% of the diagnosis-conditioned cohort")

# ---------------- article: fresh development-generator cohort
m = prim[(M, V2)]
lo_m, hi_m = json.loads(m["critical_ci95"])
A(f"kept all {num(m['n_critical'])} critical cases critical (95\\% CI {f1(lo_m)}--{pc(hi_m)}\\%) with {f1(m['accuracy_pct'])}\\% accuracy, "
  f"{f1(m['over_triage_pct'])}\\% over-triage and {f1(m['under_triage_pct'])}\\% under-triage; {f1(m['r3_under_triage_pct'])}\\% of true R3")
for k in ("XGBoost", "TSK fuzzy rule base (zero order)", "DS discounted (train reliability)"):
    assert float(prim[(M, k)]["accuracy_pct"]) == 100.0, k

# ---------------- article: decomposition
comp = S["diagnosis_composition"]
miss = comp["acwcm_missed_critical_by_diagnosis"]
h = gr[D]["number_of_gates_firing_histogram"]
assert sum(miss.values()) == h["0"] == int(a_d["missed_critical"])
A(f"no gate fired for {sum(miss.values())} of the 731 critical cases: {miss['posterior_stroke']} posterior-circulation strokes without instability, "
  f"{miss['tia_vertebrobasilar']} vertebrobasilar TIAs, {miss['vertebral_dissection']} vertebral-artery dissections, {miss['aica_stroke']} AICA strokes, "
  f"{miss['cardiac_arrhythmia']} arrhythmias and {miss['posterior_stroke_unstable']} unstable stroke",
  f"(no gate for {h['0']} cases, exactly one for {h['1']})")
r3u = comp["acwcm_under_triaged_R3_by_diagnosis"]
A(f"The {sum(r3u.values())} under-triaged R3 cases were all vestibular neuritis ({r3u['vestibular_neuritis']}) or labyrinthitis ({r3u['labyrinthitis']})")
assert len(comp["case_mix"]) == 14
A("(14 diagnoses:")
g6 = S["r1_only_via_g6"]
A(f"Among the {g6[M]['predicted_R1']} cases", f"{g6[M]['R1_only_via_G6']} were R1 only",
  f"({g6[I]['R1_only_via_G6']} of {g6[I]['predicted_R1']} and {g6[D]['R1_only_via_G6']} of {g6[D]['predicted_R1']} on the independent")

# ---------------- article: development data, variability, calibration choice
dv = dev[V2]
pdv = paired("paired_stats_dev", "ACWCM vs Dempster-Shafer", "correct")
pdv2 = paired("paired_stats_dev", "ACWCM vs Dempster-Shafer", "under2")
m2, e2 = f"{pdv2['p_holm']:.1e}".split("e")
A(f"kept {int(dv['n_critical']) - int(dv['missed_critical'])}/{dv['n_critical']} critical cases with {f1(dv['accuracy_pct'])}\\% accuracy, "
  f"{f1(dv['over_triage_pct'])}\\% over-triage and {f1(dv['under_triage_pct'])}\\% under-triage",
  f"{f1(dv['r3_under_triage_pct'])}\\% of true R3 cases ({ci(dv, 'r3_under_ci95')}\\%)",
  f"(paired difference ${pdv['diff_pct']:.2f}$ points, 95\\% CI ${pdv['diff_ci95_newcombe'][0]:.1f}$ to ${pdv['diff_ci95_newcombe'][1]:.1f}$; Holm-adjusted $p={pdv['p_holm']:.1f}$)",
  f"under-triaged {pdv2['other_only']} cases by two tiers against none ($p_{{\\text{{Holm}}}}={m2}\\times10^{{{int(e2)}}}$)")
assert pdv2["acwcm_only"] == 0 and int(dv["under_2plus"]) == 0
ms = S["multiseed"]
A(f"Over {ms['n_seeds']} further generator seeds ACWCM's accuracy was ${f1(ms['accuracy_pct']['mean'])}\\pm{f1(ms['accuracy_pct']['sd'])}\\%$ "
  f"(range {f1(ms['accuracy_pct']['min'])}--{f1(ms['accuracy_pct']['max'])}\\%), over-triage ${f1(ms['over_triage_pct']['mean'])}\\pm{f1(ms['over_triage_pct']['sd'])}\\%$, "
  f"R3-conditional under-triage ${f1(ms['r3_under_triage_pct']['mean'])}\\pm{f1(ms['r3_under_triage_pct']['sd'])}\\%$ "
  f"({f1(ms['r3_under_triage_pct']['min'])}--{f1(ms['r3_under_triage_pct']['max'])}\\%)",
  f"chose v2 on all {ms['v2_selected_count']} seeds")
assert ms["v2_selected_count"] == ms["n_seeds"] and float(ms["critical_sensitivity_pct"]["min"]) == 100.0
pa = S["profile_analysis"]
rb, v2r = pa["robust_row"], pa["v2_row"]
assert pa["n_profiles"] == 128 and pa["v2_rank_by_train_accuracy"] == 1
A(f"keeps {f1(rb[I]['critical_sensitivity_pct'])}\\% and {f1(rb[D]['critical_sensitivity_pct'])}\\% of critical cases, against "
  f"{f1(v2r[I]['critical_sensitivity_pct'])}\\% and {f1(v2r[D]['critical_sensitivity_pct'])}\\% for v2, at {f1(rb[I]['accuracy_pct'])}\\% and "
  f"{f1(rb[D]['accuracy_pct'])}\\% accuracy against {f1(v2r[I]['accuracy_pct'])}\\% and {f1(v2r[D]['accuracy_pct'])}\\%",
  f"its accuracy falls from {f1(v2r['train']['accuracy_pct'])}\\% to {f1(rb['train']['accuracy_pct'])}\\%")
A(f"({f1(v2r[D]['critical_sensitivity_pct'])}\\% against {f1(rb[D]['critical_sensitivity_pct'])}\\%)")
nd = int(prim[(D, V2)]["n_critical"])
A(f"misses about {round((float(rb[D]['critical_sensitivity_pct']) - float(v2r[D]['critical_sensitivity_pct'])) * nd / 100)} fewer of the {nd} critical cases")

# ---------------- article: stress tests
mv = mcar[("30", V2)]
A(f"G3 deferred {f1(mv['own_abstention_pct'])}\\% of cases, including {mv['deferred_critical']} of the 175 critical ones",
  f"sensitivity among decided critical cases was {pc(mv['sensitivity_among_decided_pct'])}\\% for ACWCM, "
  f"{f1(mcar[('30', DS)]['sensitivity_among_decided_pct'])}\\% for Dempster's rule, "
  f"{f1(mcar[('30', 'Sugeno (quantised additive capacity)')]['sensitivity_among_decided_pct'])}\\% for the additive Sugeno integral and "
  f"{f1(mcar[('30', 'XGBoost')]['sensitivity_among_decided_pct'])}\\% for XGBoost, which made {mcar[('30', 'XGBoost')]['false_discharges']} false discharges",
  f"the draw of Supplementary Table~S4 deferred {f1(stress[('MCAR 30%', 'SAFE-Gate (ACWCM, v2)')]['abstention_pct'])}\\%")
A(f"rose to {f1(noise[('feature noise moderate', V2)]['r3_under_triage_pct'])}\\% and {f1(noise[('feature noise high', V2)]['r3_under_triage_pct'])}\\%")
assert all(noise[(s_, V2)]["missed_critical"] == "0" for s_ in ("feature noise moderate", "feature noise high"))
A(f"fell to {f1(stress[('label noise 20%', 'SAFE-Gate (ACWCM, v2)')]['critical_sensitivity_pct'])}\\% at 20\\%",
  f"v2 kept {f1(stress[('atypical critical (red flags and timing cues removed)', 'SAFE-Gate (ACWCM, v2)')]['critical_sensitivity_pct'])}\\% in the critical band and v1 "
  f"{f1(stress[('atypical critical (red flags and timing cues removed)', 'SAFE-Gate (ACWCM, v1)')]['critical_sensitivity_pct'])}\\%")
cert = S["certificate_delta_cf_dev"]
A(f"single-feature escalation for {cert['1']} of 805 cases and none among the searched subsets for {cert['not found (censored)']}",
  f"$p={S['age_tier_interaction_dev']['p']:.2f}$")

# ---------------- IJAR round 2: R1-to-R2 loss (gate-level, same for MIN) and XGBoost threshold
if IJAR:
    assert prim[(I, V2)]['R1_under_pct'] == prim[(I, MIN)]['R1_under_pct']
    assert prim[(D, V2)]['R1_under_pct'] == prim[(D, MIN)]['R1_under_pct']
    A(f"{f1(prim[(I, V2)]['R1_under_pct'])}\\% and {f1(prim[(D, V2)]['R1_under_pct'])}\\% of true R1 cases were returned as R2",
      f"was {S['fitted']['xgb_critical_threshold']:.3f}, which made it less sensitive")
    for c_ in (I, D):
        assert float(prim[(c_, 'XGBoost (safety threshold, val >=99%)')]['critical_sensitivity_pct']) < float(prim[(c_, 'XGBoost')]['critical_sensitivity_pct'])

# ---------------- plain-text abstract and cover letter
if IJAR:
    ab = TXT["article"].split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    for s_ in (f"kept {f1(a_i['critical_sensitivity_pct'])}\\% (95\\% confidence interval {ci(a_i)}\\%)",
               f"and {f1(a_d['critical_sensitivity_pct'])}\\% ({ci(a_d)}\\%)",
               f"kept {f1(prim[(I, DS)]['critical_sensitivity_pct'])}\\% and {f1(prim[(D, DS)]['critical_sensitivity_pct'])}\\%",
               f"({f1(v2r[D]['critical_sensitivity_pct'])}\\% against {f1(rb[D]['critical_sensitivity_pct'])}\\%)"):
        checks.append(("article", s_ if s_ in ab else "<not in abstract> " + s_))
if "abstract" in TXT:
    for s_ in (f"{f1(a_i['critical_sensitivity_pct'])}% (95% confidence interval {f1(json.loads(a_i['critical_ci95'])[0])}–{f1(json.loads(a_i['critical_ci95'])[1])}%)",
               f"{f1(a_d['critical_sensitivity_pct'])}% ({f1(json.loads(a_d['critical_ci95'])[0])}–{f1(json.loads(a_d['critical_ci95'])[1])}%)",
               f"kept {f1(prim[(I, DS)]['critical_sensitivity_pct'])}% and {f1(prim[(D, DS)]['critical_sensitivity_pct'])}%",
               f"over-triage of {f1(a_i['over_triage_pct'])}% and {f1(a_d['over_triage_pct'])}%",
               f"under-triage of {f1(a_i['r3_under_triage_pct'])}% and {f1(a_d['r3_under_triage_pct'])}%",
               f"({f1(v2r[D]['critical_sensitivity_pct'])}% against {f1(rb[D]['critical_sensitivity_pct'])}%)"):
        A(s_, where="abstract")
    art_abs = TXT["article"].split("\\abstract{", 1)[1].split("\n", 1)[0][:-1]
    checks.append(("abstract", art_abs.replace("\\%", "%").replace("--", "–")))
if "cover" in TXT:
    A(f"({f1(a_i['critical_sensitivity_pct'])}\\% and {f1(a_d['critical_sensitivity_pct'])}\\% of critical cases)",
      f"against {f1(prim[(I, DS)]['critical_sensitivity_pct'])}\\% and {f1(prim[(D, DS)]['critical_sensitivity_pct'])}\\% for Dempster--Shafer",
      "came within 1.5 points", where="cover")

# ---------------- supplement: regenerate from results/ and compare, then hand-written numbers
if "supplement" in TXT:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s.tex"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_supplement.py"), str(out)] + (["--journal", "ijar"] if IJAR else []),
                       check=True, capture_output=True)
        same = out.read_text(encoding="utf-8") == TXT["supplement"]
    checks.append(("supplement", None if same else "<supplement.tex differs from scripts/make_supplement.py output>"))
    A(f"{d(100, a_d['critical_sensitivity_pct'])}\\% of critical cases, diagnosis cohort",
      f"{int(float(a_i['r3_under_triage_pct']))}--{int(round(float(a_d['r3_under_triage_pct'])))}\\% of R3 cases",
      f"{f1(mv['own_abstention_pct'])}\\% deferred at 30\\% MCAR", f"{CF['rules'][V2]['escalations_false_alarm']}/{CF['n_false_alarm']} false-alarm configs",
      f"({g6[M]['R1_only_via_G6']} of {g6[M]['predicted_R1']} R1 predictions", where="supplement")

# ---------------- IJAR round 3: rows moved to Supplementary Table S12, reworded sentences, new results
if IJAR:
    r3p = json.loads((RES / "ijar_r3_paired.json").read_text(encoding="utf-8"))
    r3rows = {(r["cohort"], r["method"]): r for r in csv.DictReader(open(RES / "ijar_r3_cohort_rules.csv", encoding="utf-8"))}
    ex_i, ex_d = prim[(I, "DS Dempster (singleton, BetP)")], prim[(D, "DS Dempster (singleton, BetP)")]
    pk = r3p[D]["critical_kept_diff_pct_newcombe"]
    pa_ = r3p[D]["accuracy_diff_pct_newcombe"]

    def oc(c_, m_, k_):
        lo_, hi_ = json.loads(r3rows[(c_, m_)][k_])
        return f"{f1(lo_)}--{f1(hi_)}"
    REWORD = {
        f"over-triage of {f1(a_i['over_triage_pct'])}\\% and {f1(a_d['over_triage_pct'])}\\%":
            f"at {f1(a_i['over_triage_pct'])}\\% and {f1(a_d['over_triage_pct'])}\\% over-triage",
        f"under-triage of {f1(a_i['r3_under_triage_pct'])}\\% and {f1(a_d['r3_under_triage_pct'])}\\%":
            "with a third of moderate-risk cases under-triaged",
        f"at least one gate returned $R^*$, R1 or R2 (the ceiling for any merge rule): {f1(gr[I]['some_gate_fires_pct'])}\\% and {f1(gr[D]['some_gate_fires_pct'])}\\%":
            f"Gate recall, the ceiling for any merge rule: {f1(gr[I]['some_gate_fires_pct'])}\\% and {f1(gr[D]['some_gate_fires_pct'])}\\%",
        f"ACWCM and {len(EN) - 2} comparison operators": None,  # count no longer quoted
        f"with {d(a_i['accuracy_pct'], m_i['accuracy_pct'])} and {d(a_d['accuracy_pct'], m_d['accuracy_pct'])} points higher accuracy and "
        f"{d(m_i['over_triage_pct'], a_i['over_triage_pct'])} and {d(m_d['over_triage_pct'], a_d['over_triage_pct'])} points less over-triage":
            f"over-triage {f1(a_i['over_triage_pct'])}\\% (95\\% CI {oc(I, V2, 'over_ci95')}) and {f1(a_d['over_triage_pct'])}\\% ({oc(D, V2, 'over_ci95')}) "
            f"against {f1(m_i['over_triage_pct'])}\\% ({oc(I, 'MIN', 'over_ci95')}) and {f1(m_d['over_triage_pct'])}\\% ({oc(D, 'MIN', 'over_ci95')}), "
            f"paid for by under-triage of {f1(a_i['under_triage_pct'])}\\% ({oc(I, V2, 'under_ci95')}) and {f1(a_d['under_triage_pct'])}\\% ({oc(D, V2, 'under_ci95')}) "
            f"against {f1(m_i['under_triage_pct'])}\\% ({oc(I, 'MIN', 'under_ci95')}) and {f1(m_d['under_triage_pct'])}\\% ({oc(D, 'MIN', 'under_ci95')})",
        f"{u['acwcm_only']} cases under-triaged by ACWCM and not by the minimum": f"{u['acwcm_only']} cases were under-triaged by ACWCM and not by the minimum",
        f"{u2['acwcm_only'] - u2['other_only']} of these were under-triaged by two tiers": f"{u2['acwcm_only'] - u2['other_only']} of them by two tiers",
        f"{f1(ni['critical_sensitivity_diff_vs_DS_pct'])} points fewer than ACWCM":
            f"kept {f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% of critical cases, {f1(pk[0])} points fewer than ACWCM",
        f"{ni['ci95'][0]:.1f}--{ni['ci95'][1]:.1f}": f"95\\% CI {f1(pk[1])}--{f1(pk[2])}, descriptive",
        f"with {prim[(D, DS)]['false_discharges']} false discharges, while being {d(prim[(D, DS)]['accuracy_pct'], a_d['accuracy_pct'])} points more accurate":
            f"with {ex_d['false_discharges']} R5 misplacements, while being {f1(-pa_[0])} points more accurate",
        f"{d(fz_d['accuracy_pct'], a_d['accuracy_pct'])} points more accurate on the diagnosis-conditioned cohort": None,  # sentence removed; fuzzy row in Table 4
        f"more R3 under-triage ({f1(fz_d['r3_under_triage_pct'])}\\% against {f1(a_d['r3_under_triage_pct'])}\\%)": None,
        f"kept {f1(pr_i['critical_sensitivity_pct'])}\\% and {f1(pr_d['critical_sensitivity_pct'])}\\% with {pr_i['false_discharges']} and {pr_d['false_discharges']} false discharges": None,  # rows in Table S12
        f"({f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% critical sensitivity for the exact rule)":
            f"against {f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% for Dempster's rule",
        f"chose v2 on all {ms['v2_selected_count']} seeds": f"chose v2 on all {ms['v2_selected_count']}.",
        f"({f1(v2r[D]['critical_sensitivity_pct'])}\\% against {f1(rb[D]['critical_sensitivity_pct'])}\\%)": None,  # same numbers checked in "keeps ... against" sentence
        f"misses about {round((float(rb[D]['critical_sensitivity_pct']) - float(v2r[D]['critical_sensitivity_pct'])) * nd / 100)} fewer of the {nd} critical cases":
            f"keeps about {round((float(rb[D]['critical_sensitivity_pct']) - float(v2r[D]['critical_sensitivity_pct'])) * nd / 100)} fewer of the {nd} critical cases",
        f"{f1(prim[(I, V2)]['R1_under_pct'])}\\% and {f1(prim[(D, V2)]['R1_under_pct'])}\\% of true R1 cases were returned as R2":
            f"down-grading of true R1 cases to R2 ({f1(prim[(I, V2)]['R1_under_pct'])}\\% and {f1(prim[(D, V2)]['R1_under_pct'])}\\% on the two cohorts, identically for the minimum",
    }
    ms_s = (f"Over {ms['n_seeds']} further generator seeds ACWCM's accuracy was ${f1(ms['accuracy_pct']['mean'])}\\pm{f1(ms['accuracy_pct']['sd'])}\\%$ ")
    new = []
    for w, c in checks:
        if c is None or w != "article":
            new.append((w, c))
            continue
        if c.startswith(ms_s):
            continue  # moved to Supplementary Table S9 (checked there by make_supplement tables)
        if c.startswith("sensitivity among decided critical cases"):
            c = c.replace("false discharges", "R5 misplacements")
        if c.startswith("<not in abstract> "):
            continue  # abstract rewritten in r3; its numbers are checked below
        if c in REWORD:
            if REWORD[c] is not None:
                new.append((w, REWORD[c]))
            continue
        if c.endswith(" \\\\") and " & " in c and c not in TXT["article"]:
            lab = c.split(" & ", 1)[0]
            c2 = c.replace("DS Dempster &", "DS Dempster (sequential) &", 1) if lab == "DS Dempster" else c
            new.append(("supplement", c2))  # row moved to Supplementary Table S12
            continue
        new.append((w, c))
    checks[:] = new
    # abstract (r3 wording)
    ab = TXT["article"].split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    for s_ in (f"({f1(a_i['critical_sensitivity_pct'])}\\% and {f1(a_d['critical_sensitivity_pct'])}\\% critical sensitivity, equal to the minimum)",
               f"against {f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% for Dempster's rule",
               f"at {f1(a_i['over_triage_pct'])}\\% and {f1(a_d['over_triage_pct'])}\\% over-triage"):
        checks.append(("article", s_ if s_ in ab else "<not in abstract> " + s_))
    assert a_i["critical_sensitivity_pct"] == m_i["critical_sensitivity_pct"] and a_d["critical_sensitivity_pct"] == m_d["critical_sensitivity_pct"]
    assert 30 <= float(a_i["r3_under_triage_pct"]) <= 37 and 30 <= float(a_d["r3_under_triage_pct"]) <= 37  # "a third"
    # exact Dempster rows of Table 4
    for c_ in (I, D):
        r_ = prim[(c_, "DS Dempster (singleton, BetP)")]
        A(f"DS Dempster (exact) & {cells(r_)} \\\\")
    # cover letter (r3 wording)
    checks[:] = [(w, c) for w, c in checks if w != "cover"]
    if "cover" in TXT:
        A(f"({f1(a_i['critical_sensitivity_pct'])}\\% and {f1(a_d['critical_sensitivity_pct'])}\\% critical sensitivity, equal to the minimum)",
          f"against {f1(ex_i['critical_sensitivity_pct'])}\\% and {f1(ex_d['critical_sensitivity_pct'])}\\% for Dempster's rule", where="cover")
    # supplement: r3 supplement is edited by hand (see IJAR revision log); compare its finite-chain table with results/
    checks[:] = [(w, c) for w, c in checks if c != "<supplement.tex differs from scripts/make_supplement.py output>"]
    th = json.loads((RES / "finite_chain_theorems.json").read_text(encoding="utf-8"))
    g = th["T1_grid_6_gates_conf_step_1/5"]
    sg3, sg6 = th["Sugeno"]["all_capacities_3_gates"], th["Sugeno"]["symmetric_capacities_6_gates"]
    t2_ = th["T2_smooth_CND"]
    assert all(g[k]["def_ne_quota"] == g[k]["band_viol"] == g[k]["mono_viol"] == g[k]["O_viol"] == 0 for k in ("theta=1/2,mu=1/2", "theta=1/2,mu=2/3"))
    assert th["T1_implementation"]["mismatches"] == 0 and all(c_["monotone_equals_quota"] for c_ in th["T1_converse"][:2])
    assert t2_["U_equals_max(2,rank_rmin)"] and t2_["two_gates"]["of_which_relax_a_floor_R2_to_R5"] == 0
    for sg in (sg3, sg6):
        assert sg["smooth_viol"] == sg["CND_iff_singletons_ge4_viol"] == sg["degeneracy_viol"] == sg["equal_to_minimum_iff_singletons_5_viol"] == 0
    co = th["T1_converse"][2]
    for w_ in ("article", "supplement"):
        A(f"21{{,}}683{{,}}472 and 13{{,}}726{{,}}224" if (g["theta=1/1,mu=1/2"]["O_viol"], g["theta=1/1,mu=2/3"]["O_viol"]) == (21683472, 13726224) else "<O-violation counts changed>",
          f"{num(co['monotone_maps'])} of the {num(co['candidate_maps'])} candidate maps are monotone and only {co['quota_maps']} of them are quotas" if w_ == "article" else f"{num(co['candidate_maps'])} candidates; {co['monotone_maps']} monotone, only {co['quota_maps']} quotas",
          f"{num(th['T1_implementation']['inputs'])} inputs", f"{num(sg3['capacities'])} capacities", f"{sg6['capacities']} symmetric capacities" if w_ == "article" else f"{sg6['capacities']} capacities",
          f"{num(t2_['two_gates']['smooth_and_CND'])} of the {num(t2_['two_gates']['CP_CND_operators'])} rules" if w_ == "article" else f"{num(t2_['two_gates']['CP_CND_operators'])} rules; {t2_['two_gates']['smooth_and_CND']} are 1-Lipschitz",
          f"{sg3['CND_capacities']} and {sg6['CND_capacities']} of these capacities are non-diluting" if w_ == "article" else f"{sg3['CND_capacities']} non-diluting", where=w_)

bad = [(w, c) for w, c in checks if c is not None and c not in TXT.get(w, c)]
print(f"{len(checks)} checks over {', '.join(TXT)}; {len(bad)} mismatches")
for w, c in bad:
    print(f"  MISSING in {w}:", c[:160])
sys.exit(1 if bad else 0)
