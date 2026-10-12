"""Build the IJFS supplement; every table number is read from results/.

    python scripts/make_supplement.py OUT.tex

Fills scripts/supplement_template.tex (placeholders @@NAME@@) and writes OUT.tex.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "supplement.tex"
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

cert = S["certificate_delta_cf_dev"]
gru = S["gate_recall_union"]
fit = S["fitted"]

tex = (Path(__file__).resolve().parent / "supplement_template.tex").read_text(encoding="utf-8")
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
for k, v in repl.items():
    assert tex.count(k) >= 1, k
    tex = tex.replace(k, v)
assert "@@" not in tex
OUT.write_text(tex, encoding="utf-8", newline="\n")
print("written", len(tex))
