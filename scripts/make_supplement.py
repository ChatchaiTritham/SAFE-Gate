"""Build the manuscript supplement; every table number is read from results/.

    python scripts/make_supplement.py OUT.tex                  # IJFS submission
    python scripts/make_supplement.py OUT.tex --journal ijar   # IJAR submission

Fills scripts/supplement_template.tex (placeholders @@NAME@@) and writes OUT.tex.
--journal ijar applies the IJAR_EDITS below (journal name, journal-neutral
release tag, and the theta <= 1 condition of the confidence statement in
Proposition 1, with a counterexample for theta > 1); the IJFS output is
unchanged.
"""
import csv
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
JOURNAL = sys.argv[sys.argv.index("--journal") + 1] if "--journal" in sys.argv else "ijfs"
if JOURNAL != "ijfs":
    _args = [a for a in _args if a != JOURNAL]
OUT = Path(_args[0]) if _args else ROOT / "supplement.tex"
S = json.loads((RES / "round4_summary.json").read_text(encoding="utf-8"))
stress = json.loads((RES / "stress_tests.json").read_text(encoding="utf-8"))
BS = "\\"


def f1(x):
    return "---" if x in ("", None) else f"{float(x):.1f}"


def num(n):
    return f"{int(n):,}".replace(",", "{,}")


# ---------------- Table S2: original 6^6 enumeration
enum_rows = list(csv.DictReader(open(RES / "invariant_enumeration.csv", encoding="utf-8")))
names = {"SAFE-Gate ACWCM (v2)": "ACWCM (v2)", "SAFE-Gate ACWCM (v1)": "ACWCM (v1)", "SAFE-Gate MIN": "MIN",
         "Sugeno integral (additive capacity)": r"Sugeno, $\nu_+$",
         "Ordinal weighted maximum": "W.\\ max, forced top", "Dempster-Shafer": "Dempster--Shafer",
         "Bayesian model averaging": "BMA", "Weighted arithmetic mean": "Weighted mean"}
t_enum = "\n".join(
    f"{names[r['rule']]} & {num(r['CP_violations'])} & {num(r['AC_violations'])} & {num(r['CND_violations'])} & "
    f"{num(r['R1D_violations'])} & {num(r['MON_violations'])} & {num(r['IDEM_violations'])} {BS}{BS}"
    for r in enum_rows)

# ---------------- Table S3: quantiser sensitivity
q = S["quantiser_sensitivity"]
t_q = "\n".join(f"{k} & " + " & ".join(str(v) for v in q[k].values()) + f" {BS}{BS}" for k in q)
q_cols = list(next(iter(q.values())).keys())

# ---------------- Table S4: stress tests (development split)
meth = [("SAFE-Gate (ACWCM, v2)", "ACWCM v2"), ("SAFE-Gate (ACWCM, v1)", "ACWCM v1"),
        ("Dempster-Shafer", "DS"), ("Sugeno integral (additive capacity)", "Sugeno"),
        ("XGBoost (trained on clean train)", "XGB")]
scen = []
for r in stress:
    if r["scenario"] not in scen:
        scen.append(r["scenario"])
short = {"clean test split": "Clean", "label noise 5%": "Ref.\\ label noise 5\\%",
         "label noise 10%": "Ref.\\ label noise 10\\%", "label noise 20%": "Ref.\\ label noise 20\\%",
         "atypical critical (red flags removed)": "No red flags",
         "atypical critical (red flags and timing cues removed)": "No flags, no timing",
         "spurious red flag in R4/R5": "Spurious flag (R4/R5)", "MCAR 10%": "MCAR 10\\%",
         "MCAR 20%": "MCAR 20\\%", "MCAR 30%": "MCAR 30\\%",
         "independent generator variant": "Indep.\\ generator (2{,}000)"}
lines = []
for sc in scen:
    cells = [short[sc]]
    for m, _ in meth:
        r = next(x for x in stress if x["scenario"] == sc and x["method"] == m)
        cells.append(f"{f1(r['critical_sensitivity_pct'])}/{r['false_discharges']}")
    a = next(x for x in stress if x["scenario"] == sc and x["method"] == meth[0][0])
    cells += [f1(a["abstention_pct"]), f1(a["over_triage_pct_decided"]), f1(a["under_triage_pct_decided"])]
    lines.append(" & ".join(cells) + f" {BS}{BS}")
t_stress = "\n".join(lines)

# ---------------- Table S5: confusion matrix + subgroups
cm = list(csv.reader(open(RES / "confusion_matrix.csv", encoding="utf-8")))
t_cm = "\n".join(" & ".join([row[0].replace("total_predicted", "Total")] + row[1:]) + f" {BS}{BS}" for row in cm[1:])
sub = [r for r in csv.DictReader(open(RES / "subgroup_under_triage.csv", encoding="utf-8")) if r["calibration"] == "v2"]
LE = "$" + BS + "le$"
t_sub = "\n".join(
    f"{r['subgroup'].replace('<=', LE).replace('>', '$>$')} & {r['n']} & {f1(r['under_triage_pct'])} ({f1(r['ci95_low_pct'])}--{f1(r['ci95_high_pct'])}) & "
    f"{f1(r['over_triage_pct'])} ({f1(r['ot_ci95_low_pct'])}--{f1(r['ot_ci95_high_pct'])}) {BS}{BS}" for r in sub)

# ---------------- Table S6: diagnosis case mix
comp = S["diagnosis_composition"]
miss = comp["acwcm_missed_critical_by_diagnosis"]
r3u = comp["acwcm_under_triaged_R3_by_diagnosis"]
mix = sorted(((k.split("|")[0], k.split("|")[1], v) for k, v in comp["case_mix"].items()),
             key=lambda x: (x[0], -x[2]))
t_mix = "\n".join(
    f"{t} & {d.replace('_', ' ')} & {num(n)} & {miss.get(d, 0) if t in ('R1', 'R2') else '---'} & "
    f"{r3u.get(d, 0) if t == 'R3' else '---'} {BS}{BS}" for t, d, n in mix)

# ---------------- Table S10: multiseed + frontier
ms = S["multiseed"]
lab = {"critical_sensitivity_pct": "Critical sensitivity (\\%)", "false_discharges": "False discharges",
       "accuracy_pct": "Accuracy (\\%)", "over_triage_pct": "Over-triage (\\%)",
       "under_triage_pct": "Under-triage (\\%)", "r3_under_triage_pct": "R3 under-triage (\\%)",
       "under_2plus": "Under-triage $\\ge$2 tiers"}
if JOURNAL == "ijar":
    lab["false_discharges"] = "R5 misplacements"
t_ms = "\n".join(f"{lab[k]} & {ms[k]['mean']} & {ms[k]['sd']} & {ms[k]['min']}--{ms[k]['max']} {BS}{BS}" for k in lab)
pa = S["profile_analysis"]
cols = [("train", "Dev.\\ train"), ("val", "Dev.\\ val."), ("val_independent", "Sel.\\ indep."),
        ("val_diagnosis", "Sel.\\ diag."), ("fresh_independent_generator", "Fresh indep."),
        ("fresh_diagnosis_generator", "Fresh diag.")]
t_front = "\n".join(
    f"{c[1]} & {pa['v2_row'][c[0]]['critical_sensitivity_pct']} & {pa['v2_row'][c[0]]['accuracy_pct']} & "
    f"{pa['robust_row'][c[0]]['critical_sensitivity_pct']} & {pa['robust_row'][c[0]]['accuracy_pct']} {BS}{BS}"
    for c in cols)

# ---------------- Table S11: ablation on external cohorts
abl_i, abl_d = S["ablation_independent"], S["ablation_diagnosis"]
t_abl = "\n".join(
    f"{a['configuration'].replace('-', '$-$')} & {f1(a['critical_sensitivity_pct'])} & {a['false_discharges']} & {f1(a['accuracy_pct'])} & "
    f"{f1(b['critical_sensitivity_pct'])} & {b['false_discharges']} & {f1(b['accuracy_pct'])} {BS}{BS}"
    for a, b in zip(abl_i, abl_d))

def h1(x):
    """One decimal, half-up on the decimal string (as in check_manuscript_numbers.py)."""
    return str(Decimal(str(x).strip()).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def ijar_tables():
    """IJAR-only tables (post-trim supplement): finite-chain checks, the 5^6 enumeration
    moved from the main text, and all comparators not in the main-text results table."""
    th = json.loads((RES / "finite_chain_theorems.json").read_text(encoding="utf-8"))
    g = th["T1_grid_6_gates_conf_step_1/5"]
    ok = [g[k] for k in ("theta=1/2,mu=1/2", "theta=1/2,mu=2/3")]
    o1, o2 = g["theta=1/1,mu=1/2"], g["theta=1/1,mu=2/3"]
    sh = g["quota_mu=2/3_function_of_share_only"]
    im = th["T1_implementation"]
    c3, c4, cc = th["T1_converse"]
    t2 = th["T2_smooth_CND"]
    sg3, sg6 = th["Sugeno"]["all_capacities_3_gates"], th["Sugeno"]["symmetric_capacities_6_gates"]
    assert ok[0]["inputs"] == ok[1]["inputs"] == 729000000 and c3["candidate_maps"] == c4["candidate_maps"]
    assert c3["monotone_maps"] == c4["monotone_maps"] == c3["quota_maps"] == c4["quota_maps"]
    assert t2["U_equals_max(2,rank_rmin)"] and t2["bound_attained_by_smooth_CND_operator"]
    z = lambda key: sum(x[key] for x in ok)  # noqa: E731
    for sg in (sg3, sg6):
        assert sg["CND_iff_singletons_ge4_viol"] == sg["degeneracy_viol"] == 0 and sg["equal_to_minimum_iff_singletons_5_viol"] == 0
    rows = [
        r"Thm 4 ($\Leftarrow$): Definition 1 with $(\theta,\mu)=(1/2,1/2)$ and $(1/2,2/3)$ vs.\ quota rule; band, (M), (O); $5^6$ profiles $\times$ $6^6$ confidence vectors in $\{0,1/5,\ldots,1\}^6$ & "
        rf"$7.29\times10^8$ inputs each; {z('def_ne_quota')} differences, {z('band_viol')} band, {z('mono_viol')} monotonicity, {z('O_viol')} (O) violations",
        rf"Thm 4, (W) and (N) for $Q_{{2/3}}$ on the same grid & output a function of the share alone: {sh['distinct_shares']} distinct shares, {sh['clashes']} clashes",
        rf"Thm 4, outside the theorem: $\theta=1$, $\mu=1/2$ and $\mu=2/3$ & (O) fails on {num(o1['O_viol'])} and {num(o2['O_viol'])} inputs; differs from the quota on {num(o1['def_ne_quota'])} and {num(o2['def_ne_quota'])}",
        rf"Released implementation vs.\ Definition 1 ($\theta=1/2$, $\mu=2/3$), {im['confidence_vectors']} confidence vectors $\times$ $5^6$ profiles & {num(im['inputs'])} inputs, {im['mismatches']} mismatches",
        rf"Thm 4 ($\Rightarrow$): all maps $g$ on the attainable shares with $g(0)=0$; normalised confidences with step $1/6$, 3 and 4 gates ({c3['confidence_vectors']} and {c4['confidence_vectors']} vectors) & {c3['candidate_maps']} candidate maps; the {c3['monotone_maps']} tier-monotone ones are exactly the {c3['quota_maps']} quotas, for both",
        rf"Same, coarse domain $\{{0,1/2,1\}}^3$ (proof construction unavailable) & {num(cc['candidate_maps'])} candidates; {cc['monotone_maps']} monotone, only {cc['quota_maps']} quotas (the theorem needs a rich confidence domain)",
        rf"Thm 5: largest rank compatible with 1-Lipschitz and non-dilution, fixed point on $5^6$ profiles & equals $\max\{{2,\text{{rank}}(r_{{\min}})\}}$ on every profile ({t2['iterations']} iterations); attained by a 1-Lipschitz non-diluting rule",
        rf"Cor 6, two gates: every rule with CP and CND & {num(t2['two_gates']['CP_CND_operators'])} rules; {t2['two_gates']['smooth_and_CND']} are 1-Lipschitz; {t2['two_gates']['of_which_relax_a_floor_R2_to_R5']} of them relax a floor",
        rf"Lemma 7 and Prop 8: all capacities on 3 gates ($5^3$ profiles) & {num(sg3['capacities'])} capacities; {sg3['smooth_viol']} smoothness violations; criterion $\nu(\{{i\}})\ge4$ exact; {sg3['CND_capacities']} non-diluting, all return $r_{{\min}}$ when no input is R1; {sg3['equal_to_minimum']} (all singletons 5) equals the minimum everywhere",
        rf"Lemma 7 and Prop 8: all symmetric capacities on 6 gates ($5^6$ profiles) & {sg6['capacities']} capacities; {sg6['smooth_viol']} smoothness violations; criterion exact; {sg6['CND_capacities']} non-diluting, all return $r_{{\min}}$ when no input is R1; {sg6['equal_to_minimum']} equals the minimum everywhere",
    ]
    t_thm = "\n".join(r + f" {BS}{BS}" for r in rows)

    EN = S["enumeration"]["rules"]
    AE = json.loads((RES / "ijar_r3_analyses.json").read_text(encoding="utf-8"))["enumeration"]
    ER = [("ACWCM (both triggers)", EN, "SAFE-Gate ACWCM (v2)"), ("Minimum", AE, "MIN"),
          ("Any-dissent (Lemma~3)", AE, "Any-dissent (band-bounded extreme)"),
          ("DS disjunctive, upper cost", AE, "DS disjunctive (upper expected cost)"),
          ("DS disjunctive, BetP", AE, "DS disjunctive (BetP, critical ties)"),
          ("Band-clamped Sugeno", EN, "Band-clamped Sugeno integral"),
          ("Order stat.\\ $k{=}2$", EN, "Ordinal OWA (k-th order statistic), k=2"),
          ("DS cons., Bel med.", EN, "DS upper-set (Bel median)"),
          ("W.\\ max, normalised", EN, "Weighted max (normalised possibility)"),
          ("Sugeno, $\\nu_+$", EN, "Sugeno (quantised additive capacity)"),
          ("Quant.\\ OWA", EN, "Quantifier-guided OWA (Yager 1996, 'most')"),
          ("Prioritized", EN, "Prioritized aggregation (Yager 2008)"),
          ("DS Demp./Yager/DP", EN, "Dempster-Shafer"),
          ("DS Dempster, upper cost", AE, "DS Dempster exact (upper expected cost)"),
          ("DS cautious", EN, "DS cautious rule (Denoeux)"),
          ("BMA", EN, "Bayesian model averaging"),
          ("TSK rule base", EN, "TSK fuzzy rule base (zero order)")]
    for x in ("CP", "CND", "R1D", "conf_monotonicity", "max_jump_one_step"):
        assert EN["SAFE-Gate ACWCM (v2)"][x] == EN["ACWCM fuzzy-quantifier trigger"][x]
    t_en = "\n".join(f"{lab} & {num(src[k]['CP'])} & {num(src[k]['CND'])} & {num(src[k]['R1D'])} & "
                     f"{num(src[k]['conf_monotonicity'])} & {src[k]['max_jump_one_step']} {BS}{BS}" for lab, src, k in ER)

    P = list(csv.DictReader(open(RES / "round4_primary.csv", encoding="utf-8")))
    N = list(csv.DictReader(open(RES / "ijar_r3_cohort_rules.csv", encoding="utf-8")))
    AR = [(P, "SAFE-Gate ACWCM (v1)", "ACWCM (v1)"), (N, "Any-dissent (band-bounded extreme)", "Any-dissent"),
          (P, "Weighted max (forced top, original)", "W.\\ max, forced top"),
          (P, "Sugeno (quantised additive capacity)", "Sugeno, $\\nu_+$"),
          (P, "Band-clamped Sugeno integral", "Band-clamped Sugeno"),
          (P, "Ordinal OWA (k-th order statistic), k=2", "$k$-th order stat."),
          (P, "Quantifier-guided OWA (Yager 1996, 'most')", "Quantifier OWA"),
          (P, "Prioritized aggregation (Yager 2008)", "Prioritized"),
          (P, "TSK fuzzy rule base (zero order)", "TSK rule base"),
          (P, "Dempster-Shafer", "DS Dempster (sequential)"),
          (N, "DS Dempster exact, conf<=0.999", "DS Dempster, $c\\le0.999$"),
          (N, "DS Dempster exact (upper expected cost)", "DS Dempster, upper cost"),
          (P, "DS cautious rule (Denoeux)", "DS cautious"),
          (N, "DS cautious, conf<=0.999", "DS cautious, $c\\le0.999$"),
          (P, "DS min expected cost", "DS min.\\ exp.\\ cost"),
          (N, "DS disjunctive (BetP, critical ties)", "DS disjunctive, BetP"),
          (N, "DS disjunctive (BetP, benign ties)", "DS disj., BetP, benign ties"),
          (N, "DS disjunctive (upper expected cost)", "DS disjunctive, upper cost"),
          (P, "Bayesian model averaging", "BMA"), (P, "XGBoost", "XGBoost"),
          (P, "XGBoost (safety threshold, val >=99%)", "XGBoost, safety thr.")]
    out = []
    for coh, lab in (("fresh_independent_generator", "(a) Independent generator"),
                     ("fresh_diagnosis_generator", "(b) Diagnosis-conditioned generator")):
        out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{lab}}}}} {BS}{BS}")
        for src, key, name in AR:
            r = next(x for x in src if x["cohort"] == coh and x["method"] == key)
            lo, hi = [h1(v) for v in r["critical_ci95"].strip("[]").split(",")]
            out.append(f"{name} & {h1(r['critical_sensitivity_pct'])} ({lo}--{hi}) & {r['false_discharges']} & "
                       f"{h1(r['accuracy_pct'])} & {h1(r['over_triage_pct'])} & {h1(r['under_triage_pct'])} & "
                       f"{r['under_2plus']} & {h1(r['r3_under_triage_pct'])} {BS}{BS}")
        if coh.startswith("fresh_indep"):
            out.append("\\midrule")
    return {"@@THM@@": t_thm, "@@ENUMMAIN@@": t_en, "@@ALL@@": "\n".join(out)}


cert = S["certificate_delta_cf_dev"]
gru = S["gate_recall_union"]
fit = S["fitted"]

TEMPLATE = "supplement_template_ijar.tex" if JOURNAL == "ijar" else "supplement_template.tex"
tex = (Path(__file__).resolve().parent / TEMPLATE).read_text(encoding="utf-8")
repl = {"@@ENUM@@": t_enum, "@@Q@@": t_q, "@@QCOLS@@": r"Sugeno $\nu_+$ & W.\ max forced & W.\ max norm. & Band-clamped", "@@STRESS@@": t_stress,
        "@@CM@@": t_cm, "@@SUB@@": t_sub, "@@MIX@@": t_mix, "@@MS@@": t_ms, "@@FRONT@@": t_front,
        "@@ABL@@": t_abl, "@@CERT1@@": str(cert.get("1", 0)), "@@CERTNONE@@": str(cert.get("not found (censored)", 0)),
        "@@RELIAB@@": ", ".join(f"{g} {v}" for g, v in fit["reliability"].items()),
        "@@XGBT@@": str(fit["xgb_critical_threshold"]), "@@MSN@@": str(ms["n_seeds"]),
        "@@V2SEL@@": str(ms["v2_selected_count"]),
        "@@GRU_I@@": str(gru["fresh_independent_generator"]["some_gate_fires_pct"]),
        "@@GRU_D@@": str(gru["fresh_diagnosis_generator"]["some_gate_fires_pct"]),
        "@@PERGATE_D@@": ", ".join(f"{g} {v}\\%" for g, v in gru["fresh_diagnosis_generator"]["per_gate_fire_pct"].items()),
        "@@AGEINT@@": f"likelihood-ratio statistic {S['age_tier_interaction_dev']['lr_stat']}, df {S['age_tier_interaction_dev']['df']}, $p={S['age_tier_interaction_dev']['p']}$"}
if JOURNAL == "ijar":
    repl.update(ijar_tables())
for k, v in repl.items():
    assert tex.count(k) >= 1, k
    tex = tex.replace(k, v)
assert "@@" not in tex
IJAR_EDITS = [
    ("%% Supplementary Material for \"SAFE-Gate\" (International Journal of  %%\n"
     "%% Fuzzy Systems). Tables are generated from the repository results/ %%",
     "%% Supplementary Material for \"SAFE-Gate\" (International Journal of  %%\n"
     "%% Approximate Reasoning). Tables are generated from results/        %%"),
    ("(tag \\texttt{v2.0.1-ijfs}; the script", "(tag \\texttt{v2.0.3}; the script"),
    ("code tag \\texttt{v2.0.1-ijfs}", "code tag \\texttt{v2.0.3}"),
    ("for the fusion rules of the earlier version of this work;", "for the fusion rules of the v1 calibration;"),
    (", called a ``safety certificate'' in earlier versions; it certifies nothing and has been renamed.",
     "; it is an audit record, not a certificate."),
    ("both increase when a dissenting confidence rises or a floor confidence falls.",
     ". $m_>$ increases when a dissenting confidence rises or a floor confidence falls. The margin $\\tau_w-\\text{rank}(r)=S/(A+B)$, "
     "with $S=\\sum_{\\text{above}}c_id_i$ and $d_i=\\text{rank}(r_i)-\\text{rank}(r)\\ge1$, can decrease, but if $S\\ge\\theta(A+B)$ and "
     "$\\theta\\le1$ then $S+\\delta d_j\\ge\\theta(A+B+\\delta)$ for any added dissenting confidence $\\delta$, and lowering a floor confidence "
     "only lowers $A+B$; so for $\\theta\\le1$ the trigger cannot switch off. For $\\theta>1$ the confidence statement fails: with "
     "$\\theta=1.5$, $\\mu=1/2$, gates at (R3, 0.2), (R4, $c$), (R5, 1) and three at R5 with confidence 0, the output is R4 for $c=0$ "
     "($\\tau_w-3=1.67$) and R3 for $c=1$ ($\\tau_w-3=1.36$), as the implementation confirms. Every calibration profile has $\\theta\\le1$."),
    ("(A+B)$ . $m_>$", "(A+B)$. $m_>$"),
]
if JOURNAL not in ("ijfs", "ijar"):
    sys.exit(f"unknown --journal {JOURNAL}")
OUT.write_text(tex, encoding="utf-8", newline="\n")
print("written", len(tex))
