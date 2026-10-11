"""
Robustness / stress evaluation that loosens the generator-gate circularity.

Scenarios (all seeded; applied to the held-out test split unless noted):
  clean               the test split as generated
  label noise p       p = 5/10/20 % of labels flipped to a uniformly drawn
                      different tier (features unchanged)
  atypical critical   every truly critical (R1/R2) case stripped of its red-flag
                      features: vital signs normalised, focal and headache
                      flags cleared, HINTS rewritten to a peripheral pattern;
                      a harder version also replaces the hyperacute/worsening
                      timing with an acute, stable, gradual 10 h course
  spurious flag       every truly benign (R4/R5) case given one spurious red
                      flag (SBP 185, SpO2 91 %, new diplopia or a positive test
                      of skew, drawn at random)
  MCAR p              each of the 52 features removed independently with
                      probability p = 10/20/30 % (missing completely at random)
  variant generator   2,000 cases from data/generation/generate_variant.py,
                      whose parameters are not shared with the gate thresholds

Methods: SAFE-Gate ACWCM (v2 and v1 calibration), Dempster-Shafer, Bayesian
model averaging, the two Sugeno-family rules (all on the v2 gate outputs),
arithmetic ensemble averaging, and XGBoost trained on the clean training split.

Two different safety notions are reported and must not be confused:
  * invariant violations: the rule-level properties (AC, CP, CND) checked
    against the gate outputs the rule actually received. They hold by
    construction for ACWCM, whatever the data.
  * true-tier critical sensitivity: share of truly critical cases kept at
    R*, R1 or R2 (an abstention R* hands the case to a clinician and is
    counted as kept). It depends on whether some gate recognises the case,
    so it can fail when the labels or features stop matching the gate
    knowledge.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generation"))

from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination  # noqa: E402
from baselines.ensemble_average import EnsembleAverage  # noqa: E402
from baselines.sugeno import SugenoAdditive, WeightedMax  # noqa: E402
from generate_data import FEATURE_SCHEMA  # noqa: E402
from generate_variant import VariantGenerator  # noqa: E402
from safegate import SAFEGate  # noqa: E402

TIERS = ["R1", "R2", "R3", "R4", "R5"]
IDX = {t: i for i, t in enumerate(TIERS)}
CRIT = {"R1", "R2"}
RANK = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
SEED = 7


def cp_interval(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #
def flip_labels(cases, p, rng):
    out = []
    for c in cases:
        c2 = dict(c)
        if rng.random() < p:
            c2["ground_truth_tier"] = str(rng.choice([t for t in TIERS if t != c["ground_truth_tier"]]))
        out.append(c2)
    return out


def atypical_critical(cases, strip_timing=False):
    out = []
    for c in cases:
        if c["ground_truth_tier"] not in CRIT:
            continue
        c2 = dict(c)
        c2.update(systolic_bp=130, diastolic_bp=82, heart_rate=84, spo2=97, gcs=15,
                  hints_head_impulse="abnormal", hints_nystagmus="horizontal",
                  hints_test_of_skew="negative", vertical_skew_deviation=False)
        for f in ("dysarthria", "ataxia", "diplopia", "new_onset_diplopia", "crossed_sensory_loss",
                  "facial_weakness", "limb_weakness", "focal_weakness", "thunderclap_headache",
                  "severe_headache", "worst_headache_ever", "nystagmus_with_diplopia"):
            c2[f] = False
        if strip_timing:
            # Also remove the "time is brain" cues: an acute, stable, gradual
            # presentation (10 h), which is what a stroke mimicking a benign
            # vestibular syndrome can look like.
            c2.update(symptom_onset_hours=10.0, symptom_duration_days=round(10.0 / 24, 3),
                      progression_pattern="stable", sudden_onset=False)
        out.append(c2)
    return out


def spurious_flag(cases, rng):
    out = []
    for c in cases:
        if c["ground_truth_tier"] not in ("R4", "R5"):
            continue
        c2 = dict(c)
        k = int(rng.randint(4))
        if k == 0:
            c2["systolic_bp"] = 185
        elif k == 1:
            c2["spo2"] = 91
        elif k == 2:
            c2["diplopia"] = c2["new_onset_diplopia"] = True
        else:
            c2["hints_test_of_skew"] = "positive"
        out.append(c2)
    return out


def mcar(cases, p, rng):
    out = []
    for c in cases:
        c2 = {k: v for k, v in c.items() if k not in FEATURE_SCHEMA or rng.random() >= p}
        out.append(c2)
    return out


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def predictions(cases, xgb_predict):
    sg2, sg1 = SAFEGate(calibration="v2"), SAFEGate(calibration="v1")
    ds, bma, sa, wm, ea = (DempsterShaferCombination(), BayesianModelAveraging(),
                           SugenoAdditive(), WeightedMax(), EnsembleAverage())
    preds = {k: [] for k in ("SAFE-Gate (ACWCM, v2)", "SAFE-Gate (ACWCM, v1)", "Dempster-Shafer",
                             "Bayesian model averaging", "Sugeno integral (additive capacity)",
                             "Ordinal weighted maximum", "Arithmetic ensemble averaging")}
    gate_outs = []
    for c in cases:
        r = sg2.classify(c, return_audit_trail=False, return_certificate=False)
        go, gc = r["gate_outputs"], r["gate_confidences"]
        gate_outs.append(go)
        clean = {g: (t if t in TIERS else "R1") for g, t in go.items()}
        preds["SAFE-Gate (ACWCM, v2)"].append(r["final_tier"])
        preds["SAFE-Gate (ACWCM, v1)"].append(
            sg1.classify(c, return_audit_trail=False, return_certificate=False)["final_tier"])
        preds["Dempster-Shafer"].append(ds.classify(clean, gc)["final_tier"])
        preds["Bayesian model averaging"].append(bma.classify(clean, gc)["final_tier"])
        preds["Sugeno integral (additive capacity)"].append(sa.classify(go, gc)["final_tier"])
        preds["Ordinal weighted maximum"].append(wm.classify(go, gc)["final_tier"])
        preds["Arithmetic ensemble averaging"].append(ea.classify(c)[0])
    if xgb_predict is not None:
        preds["XGBoost (trained on clean train)"] = xgb_predict(cases)
    return preds, gate_outs


def metrics(scenario, method, cases, preds, gate_outs, check_invariants):
    n = len(cases)
    crit = [i for i, c in enumerate(cases) if c["ground_truth_tier"] in CRIT]
    kept = sum(1 for i in crit if preds[i] in ("R*", "R1", "R2"))
    lo, hi = cp_interval(kept, len(crit)) if crit else (0.0, 0.0)
    fd = sum(1 for i in crit if preds[i] == "R5")
    decided = [i for i in range(n) if preds[i] in TIERS]
    ot = sum(1 for i in decided if IDX[preds[i]] < IDX[cases[i]["ground_truth_tier"]])
    ut = sum(1 for i in decided if IDX[preds[i]] > IDX[cases[i]["ground_truth_tier"]])
    acc = sum(1 for i in decided if preds[i] == cases[i]["ground_truth_tier"])
    abst = n - len(decided)
    viol = ""
    ac_ex = ""
    if check_invariants:
        viol = 0
        ac_ex = 0
        for p, go in zip(preds, gate_outs):
            ranks = [RANK[t] for t in go.values()]
            rmin = min(ranks)
            if rmin == 0:
                ac_ex += 1
                viol += p != "R*"
            else:
                viol += RANK[p] > rmin + 1 if method.startswith("SAFE-Gate") else 0
                viol += rmin in (1, 2) and p not in ("R*", "R1", "R2")
    nd = max(len(decided), 1)
    return {
        "scenario": scenario, "method": method, "n": n, "n_critical": len(crit),
        "critical_sensitivity_pct": round(100 * kept / len(crit), 2) if crit else "",
        "critical_ci95_low_pct": round(100 * lo, 2) if crit else "",
        "critical_ci95_high_pct": round(100 * hi, 2) if crit else "",
        "false_discharges": fd,
        "abstention_pct": round(100 * abst / n, 2),
        "accuracy_pct_decided": round(100 * acc / nd, 2),
        "over_triage_pct_decided": round(100 * ot / nd, 2),
        "under_triage_pct_decided": round(100 * ut / nd, 2),
        "cases_with_abstaining_gate": ac_ex,
        "invariant_violations": viol,
    }


def run(test, xgb_predict):
    rng = np.random.RandomState(SEED)
    rows = []
    gate_rules = {"SAFE-Gate (ACWCM, v2)", "Dempster-Shafer", "Bayesian model averaging",
                  "Sugeno integral (additive capacity)", "Ordinal weighted maximum"}

    def add(name, cases, cache=None):
        preds, gouts = cache or predictions(cases, xgb_predict)
        for m, p in preds.items():
            rows.append(metrics(name, m, cases, p, gouts, m in gate_rules))
        return preds, gouts

    clean = add("clean test split", test)
    for p in (0.05, 0.10, 0.20):
        add(f"label noise {int(p * 100)}%", flip_labels(test, p, rng), clean)
    add("atypical critical (red flags removed)", atypical_critical(test))
    add("atypical critical (red flags and timing cues removed)", atypical_critical(test, True))
    add("spurious red flag in R4/R5", spurious_flag(test, rng))
    for p in (0.10, 0.20, 0.30):
        add(f"MCAR {int(p * 100)}%", mcar(test, p, rng))
    add("independent generator variant", VariantGenerator(n_total=2000, random_state=2026).generate())

    out = ROOT / "results"
    with (out / "stress_tests.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "stress_tests.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
