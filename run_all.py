#!/usr/bin/env python3
"""
SAFE-Gate end-to-end reproducible evaluation driver.

Single entry point that:
  1. Regenerates the seeded synthetic cohort (seed=42, 6,400 cases, 52 features).
  2. Classifies the held-out test split with SAFE-Gate (ACWCM and basic MIN).
  3. Runs the four comparison fusion baselines (arithmetic ensemble averaging,
     Dempster-Shafer, Bayesian model averaging, single XGBoost).
  4. Verifies the formal safety properties on the test split.
  5. Writes every headline artifact to results/ as committed CSV / JSON.

The pipeline is fully deterministic: data generation is seeded, gate inference is
seeded per case, and no metric is hardcoded. Re-running reproduces byte-identical
results/ files.

Usage:
    pip install -e .
    python run_all.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generation"))

from generate_data import FEATURE_SCHEMA, SAFEGateDataGenerator  # noqa: E402
from safegate import SAFEGate  # noqa: E402
from baselines.ensemble_average import EnsembleAverage  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination  # noqa: E402
from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402

SEED = 42
N_TOTAL = 6400
TIERS = ["R1", "R2", "R3", "R4", "R5"]
TIER_IDX = {t: i for i, t in enumerate(TIERS)}
CRITICAL = {"R1", "R2"}
DISCHARGE = "R5"

RESULTS = ROOT / "results"
DATA = ROOT / "data" / "synthetic"


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def confusion(cases, predictions):
    """Build a true x predicted confusion dict over the five tiers (R* excluded)."""
    conf = {t: Counter() for t in TIERS}
    abstain = 0
    for case, pred in zip(cases, predictions):
        true = case["ground_truth_tier"]
        if pred not in TIERS:
            abstain += 1
            continue
        conf[true][pred] += 1
    return conf, abstain


def tier_metrics(conf):
    n = sum(sum(conf[t].values()) for t in TIERS)
    rows = []
    for t in TIERS:
        tp = conf[t][t]
        fp = sum(conf[r][t] for r in TIERS if r != t)
        fn = sum(conf[t][p] for p in TIERS if p != t)
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "tier": t,
            "support": sum(conf[t].values()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
        })
    return rows


def accuracy(conf):
    n = sum(sum(conf[t].values()) for t in TIERS)
    correct = sum(conf[t][t] for t in TIERS)
    return correct / n if n else 0.0


def critical_sensitivity(cases, predictions):
    """Fraction of truly-critical (R1/R2) cases kept at a critical tier."""
    total = sum(1 for c in cases if c["ground_truth_tier"] in CRITICAL)
    kept = sum(
        1 for c, p in zip(cases, predictions)
        if c["ground_truth_tier"] in CRITICAL and p in CRITICAL
    )
    return kept, total


def false_discharges(cases, predictions):
    """Truly-critical cases routed to the discharge tier (R5) -- must be zero."""
    return sum(
        1 for c, p in zip(cases, predictions)
        if c["ground_truth_tier"] in CRITICAL and p == DISCHARGE
    )


def over_triage_rate(conf):
    n = sum(sum(conf[t].values()) for t in TIERS)
    over = sum(conf[t][p] for t in TIERS for p in TIERS if TIER_IDX[p] < TIER_IDX[t])
    return over / n if n else 0.0


def under_triage(conf):
    """Count cases assigned a less urgent tier than their true tier, and the
    deepest such step (in tiers). Under-triage is the error the safety floor
    must keep out of the critical band; outside it, ACWCM bounds its depth."""
    under = sum(conf[t][p] for t in TIERS for p in TIERS if TIER_IDX[p] > TIER_IDX[t])
    two_plus = sum(conf[t][p] for t in TIERS for p in TIERS if TIER_IDX[p] - TIER_IDX[t] >= 2)
    depth = max([TIER_IDX[p] - TIER_IDX[t] for t in TIERS for p in TIERS if conf[t][p]] + [0])
    return under, two_plus, depth


def clopper_pearson_upper(x, n, alpha=0.05):
    """Upper bound of the Clopper-Pearson exact CI (for the zero-event case)."""
    from scipy.stats import beta
    if x == n:
        return 1.0
    return float(beta.ppf(1 - alpha / 2, x + 1, n - x))


def clopper_pearson_lower(x, n, alpha=0.05):
    from scipy.stats import beta
    if x == 0:
        return 0.0
    return float(beta.ppf(alpha / 2, x, n - x + 1))


# --------------------------------------------------------------------------- #
# SAFE-Gate classification
# --------------------------------------------------------------------------- #
def classify_safegate(cases, mode, trigger="consensus", calibration="v2"):
    sg = SAFEGate(mode=mode, trigger=trigger, calibration=calibration)
    return [sg.classify(c, return_audit_trail=False, return_certificate=False)["final_tier"]
            for c in cases]


def gate_outputs_for(cases):
    """Collect raw per-gate tier strings + confidences for the fusion baselines."""
    sg = SAFEGate()
    out = []
    for c in cases:
        res = sg.classify(c, return_audit_trail=False, return_certificate=False)
        out.append((res["gate_outputs"], res["gate_confidences"]))
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def run_ensemble(cases):
    ea = EnsembleAverage()
    return [ea.classify(c)[0] for c in cases]


def run_ds(gate_io):
    ds = DempsterShaferCombination()
    preds = []
    for outputs, confs in gate_io:
        # R* outputs are abstentions; map to the most conservative non-abstain
        # tier the gate carried (R1) so DS still receives critical mass.
        clean = {g: (t if t in TIERS else "R1") for g, t in outputs.items()}
        preds.append(ds.classify(clean, confs)["final_tier"])
    return preds


def run_bma(gate_io):
    bma = BayesianModelAveraging()
    preds = []
    for outputs, confs in gate_io:
        clean = {g: (t if t in TIERS else "R1") for g, t in outputs.items()}
        preds.append(bma.classify(clean, confs)["final_tier"])
    return preds


def run_xgboost(train, test):
    """Single gradient-boosted tree on the encoded feature matrix."""
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    def encode(rows):
        X = []
        for r in rows:
            vec = []
            for f in FEATURE_SCHEMA:
                v = r[f]
                if isinstance(v, bool):
                    vec.append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    vec.append(float(v))
                else:
                    # Stable categorical hashing into a small numeric code.
                    vec.append(float(int(__import__("hashlib").sha256(str(v).encode()).hexdigest()[:6], 16) % 997))
            X.append(vec)
        return np.asarray(X, dtype=float)

    y_train = np.array([TIER_IDX[r["ground_truth_tier"]] for r in train])
    clf = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=SEED, tree_method="hist", verbosity=0,
        num_class=len(TIERS), objective="multi:softmax",
    )
    clf.fit(encode(train), y_train)
    pred_idx = clf.predict(encode(test))
    return [TIERS[int(i)] for i in pred_idx]


def baseline_row(name, cases, preds):
    kept, total = critical_sensitivity(cases, preds)
    conf, _ = confusion(cases, preds)
    return {
        "method": name,
        "critical_sensitivity": round(100 * kept / total, 2) if total else 0.0,
        "critical_caught": kept,
        "critical_total": total,
        "missed_critical": total - kept,
        "discharge_specificity_R5": round(100 * conf[DISCHARGE][DISCHARGE] / sum(conf[DISCHARGE].values()), 2)
            if sum(conf[DISCHARGE].values()) else 0.0,
        "accuracy": round(100 * accuracy(conf), 2),
        "false_discharges": false_discharges(cases, preds),
        "over_triage": round(100 * over_triage_rate(conf), 2),
        "under_triage": round(100 * under_triage(conf)[0] / len(cases), 2),
        "under_triage_2plus_tiers": under_triage(conf)[1],
        "max_under_triage_depth": under_triage(conf)[2],
    }


# --------------------------------------------------------------------------- #
# Ablation: ACWCM with each gate disabled
# --------------------------------------------------------------------------- #
def ablation(cases):
    """Re-run ACWCM with each gate removed in turn and record the metrics."""
    def metrics_for(disabled):
        sg = SAFEGate()
        for g in disabled:
            del sg.gates[g]
        preds = [sg.classify(c, return_audit_trail=False, return_certificate=False)["final_tier"]
                 for c in cases]
        conf, _ = confusion(cases, preds)
        kept, total = critical_sensitivity(cases, preds)
        rows = tier_metrics(conf)
        r5 = next(r for r in rows if r["tier"] == "R5")
        return {
            "critical_sensitivity_pct": round(100 * kept / total, 1) if total else 0.0,
            "missed_critical": total - kept,
            "discharge_specificity_R5_pct": round(100 * r5["recall"], 1),
            "over_triage_pct": round(100 * over_triage_rate(conf), 1),
            "macro_f1_pct": round(100 * float(np.mean([r["f1"] for r in rows])), 1),
            "accuracy_pct": round(100 * accuracy(conf), 1),
            "under_triage_pct": round(100 * under_triage(conf)[0] / len(cases), 1),
            "under_triage_2plus_tiers": under_triage(conf)[1],
        }

    full = metrics_for(())
    rows = [{"configuration": "Full ACWCM (6 gates)", **full, "delta_f1": 0.0}]
    configs = [(g,) for g in ["G1", "G2", "G3", "G4", "G5", "G6"]] + [("G1", "G4")]
    for gs in configs:
        m = metrics_for(gs)
        m["delta_f1"] = round(m["macro_f1_pct"] - full["macro_f1_pct"], 1)
        rows.append({"configuration": "".join(f"-{g}" for g in gs), **m})
    return rows


# --------------------------------------------------------------------------- #
# Paired comparisons on the same 805 cases (exact McNemar)
# --------------------------------------------------------------------------- #
def _indicators(cases, preds):
    out = {"correct": [], "over_triage": [], "under_triage": []}
    for c, p in zip(cases, preds):
        t = TIER_IDX[c["ground_truth_tier"]]
        q = TIER_IDX.get(p, -1)  # R* (never observed) would count as over-triage
        out["correct"].append(q == t)
        out["over_triage"].append(q < t)
        out["under_triage"].append(q > t)
    return out


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test on paired binary outcomes a, b."""
    from scipy.stats import binomtest
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    p = binomtest(n10, n, 0.5).pvalue if n else 1.0
    return n10, n01, float(p)


def paired_tests(cases, reference, others):
    rows = []
    ia = _indicators(cases, reference)
    for name, preds in others:
        ib = _indicators(cases, preds)
        for k in ("correct", "over_triage", "under_triage"):
            n10, n01, p = mcnemar_exact(ia[k], ib[k])
            rows.append({
                "comparison": f"ACWCM vs {name}",
                "outcome": k,
                "acwcm_rate_pct": round(100 * sum(ia[k]) / len(cases), 2),
                "other_rate_pct": round(100 * sum(ib[k]) / len(cases), 2),
                "acwcm_only": n10,
                "other_only": n01,
                "mcnemar_exact_p": float(f"{p:.3g}"),
            })
    return rows


# --------------------------------------------------------------------------- #
# Under-triage by subgroup (age, sex) and by true tier
# --------------------------------------------------------------------------- #
def subgroup_under_triage(cases, preds):
    def ut(c, p):
        return TIER_IDX.get(p, -1) > TIER_IDX[c["ground_truth_tier"]]
    groups = [
        ("age > 60", lambda c: c["age"] > 60),
        ("age <= 60", lambda c: c["age"] <= 60),
        ("female", lambda c: c["gender"] == "female"),
        ("male", lambda c: c["gender"] == "male"),
    ] + [(f"true {t}", (lambda t: lambda c: c["ground_truth_tier"] == t)(t)) for t in TIERS]
    total_ut = sum(ut(c, p) for c, p in zip(cases, preds))
    rows = []
    for name, f in groups:
        idx = [i for i, c in enumerate(cases) if f(c)]
        k = sum(ut(cases[i], preds[i]) for i in idx)
        n = len(idx)
        rows.append({
            "subgroup": name,
            "n": n,
            "under_triaged": k,
            "under_triage_pct": round(100 * k / n, 2) if n else 0.0,
            "ci95_low_pct": round(100 * clopper_pearson_lower(k, n), 2) if n else 0.0,
            "ci95_high_pct": round(100 * clopper_pearson_upper(k, n), 2) if n else 0.0,
            "share_of_all_under_triage_pct": round(100 * k / total_ut, 2) if total_ut else 0.0,
        })
    return rows


# --------------------------------------------------------------------------- #
# Formal safety property verification (as stated in the manuscript)
# --------------------------------------------------------------------------- #
def verify_properties(cases, calibration="v2"):
    """
    Verify the three formally proven invariants plus the no-false-discharge
    guarantee, evaluating each property exactly as defined in the manuscript.

      CP  Conservative preservation: r_final <= r_min + 1 (lattice rank).
      AC  Abstention correctness:    any gate = R*  =>  r_final = R*.
      CND Critical non-dilution:     r_min in {R1,R2}  =>  r_final in {R*,R1,R2}.
      ND  No false discharge:        true critical (R1/R2)  =>  r_final != R5.
    """
    sg = SAFEGate(calibration=calibration)
    counts = {"CP": 0, "AC": 0, "CND": 0, "ND": 0}
    n_abstain_inputs = 0
    examples = {"CP": [], "AC": [], "CND": [], "ND": []}
    rank = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}

    for c in cases:
        res = sg.classify(c, return_audit_trail=False, return_certificate=False)
        final = res["final_tier"]
        gate_tiers = list(res["gate_outputs"].values())
        ranks = [rank[t] for t in gate_tiers]
        r_min = min(ranks)
        fr = rank[final]
        abstain_present = any(t == "R*" for t in gate_tiers)
        n_abstain_inputs += abstain_present

        # AC
        if abstain_present and final != "R*":
            counts["AC"] += 1
            if len(examples["AC"]) < 5:
                examples["AC"].append({"patient_id": c.get("patient_id"), "final": final})
        # CP (only when no abstention drives the output)
        if not abstain_present and fr > r_min + 1:
            counts["CP"] += 1
            if len(examples["CP"]) < 5:
                examples["CP"].append({"patient_id": c.get("patient_id"), "final": final, "r_min": r_min})
        # CND
        if r_min in (1, 2) and final not in ("R*", "R1", "R2"):
            counts["CND"] += 1
            if len(examples["CND"]) < 5:
                examples["CND"].append({"patient_id": c.get("patient_id"), "final": final})
        # ND
        if c["ground_truth_tier"] in CRITICAL and final == DISCHARGE:
            counts["ND"] += 1
            if len(examples["ND"]) < 5:
                examples["ND"].append({"patient_id": c.get("patient_id")})

    total = sum(counts.values())
    return {
        "n_cases": len(cases),
        "cases_with_an_abstaining_gate": n_abstain_inputs,
        "note": "implementation conformance test of invariants that hold by construction; "
                "AC is exercised only if some gate abstains",
        "properties": {
            "conservative_preservation": {"violations": counts["CP"], "holds": counts["CP"] == 0, "examples": examples["CP"]},
            "abstention_correctness": {"violations": counts["AC"], "holds": counts["AC"] == 0, "examples": examples["AC"]},
            "critical_non_dilution": {"violations": counts["CND"], "holds": counts["CND"] == 0, "examples": examples["CND"]},
            "no_false_discharge": {"violations": counts["ND"], "holds": counts["ND"] == 0, "examples": examples["ND"]},
        },
        "total_violations": total,
        "all_pass": total == 0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("[1/5] Regenerating seeded synthetic cohort ...")
    gen = SAFEGateDataGenerator(n_total=N_TOTAL, random_state=SEED)
    all_cases, train, val, test, written = gen.save(DATA)
    print(f"      cohort={len(all_cases)}  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"      features per case = {len(FEATURE_SCHEMA)}")

    print("[2/5] Classifying test split with SAFE-Gate (ACWCM + MIN) ...")
    acwcm = classify_safegate(test, "acwcm")
    acwcm_any = classify_safegate(test, "acwcm", trigger="any_dissent")
    basic = classify_safegate(test, "min")
    # Validation-split check of the a-priori trigger thresholds (reported, not tuned).
    val_rows = [baseline_row("SAFE-Gate (ACWCM)", val, classify_safegate(val, "acwcm")),
                baseline_row("SAFE-Gate (ACWCM, any-dissent trigger)", val,
                             classify_safegate(val, "acwcm", trigger="any_dissent"))]

    conf_acwcm, abstain_acwcm = confusion(test, acwcm)
    conf_basic, abstain_basic = confusion(test, basic)
    rows_acwcm = tier_metrics(conf_acwcm)
    rows_basic = tier_metrics(conf_basic)

    kept, crit_total = critical_sensitivity(test, acwcm)
    cs_lo = clopper_pearson_lower(kept, crit_total)
    macro_f1 = float(np.mean([r["f1"] for r in rows_acwcm]))

    print("[3/5] Running fusion baselines (ensemble / DS / BMA / XGBoost) ...")
    gate_io = gate_outputs_for(test)
    ens = run_ensemble(test)
    ds = run_ds(gate_io)
    bma = run_bma(gate_io)
    xgb = run_xgboost(train, test)

    baselines = [
        baseline_row("Arithmetic ensemble averaging", test, ens),
        baseline_row("Dempster-Shafer", test, ds),
        baseline_row("Bayesian model averaging", test, bma),
    ]
    if xgb is not None:
        baselines.append(baseline_row("Single XGBoost", test, xgb))
    # Trivial reference: escalate every case to the most critical tier. It attains
    # perfect critical sensitivity by construction, so any safety claim must beat
    # it on over-triage, not on sensitivity.
    baselines.append(baseline_row("Always-critical rule (R1)", test, ["R1"] * len(test)))
    baselines.append(baseline_row("SAFE-Gate (MIN)", test, basic))
    baselines.append(baseline_row("SAFE-Gate (ACWCM, any-dissent trigger)", test, acwcm_any))
    baselines.append(baseline_row("SAFE-Gate (ACWCM)", test, acwcm))

    print("[4/5] Verifying formal safety properties + ablation ...")
    props = verify_properties(test)
    props_all = verify_properties(all_cases)
    abl = ablation(test)
    paired = paired_tests(test, acwcm, [("MIN", basic), ("Dempster-Shafer", ds),
                                         ("any-dissent trigger", acwcm_any)])
    subgroups = subgroup_under_triage(test, acwcm)

    # ---- assemble summary ----
    n_test = sum(sum(conf_acwcm[t].values()) for t in TIERS) + abstain_acwcm
    summary = {
        "seed": SEED,
        "cohort": {
            "n_total": len(all_cases),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_features": len(FEATURE_SCHEMA),
            "tier_distribution_test": {t: sum(conf_acwcm[t].values()) for t in TIERS},
        },
        "acwcm": {
            "accuracy_pct": round(100 * accuracy(conf_acwcm), 2),
            "critical_sensitivity_pct": round(100 * kept / crit_total, 2),
            "critical_caught": kept,
            "critical_total": crit_total,
            "critical_sensitivity_ci95": [round(100 * cs_lo, 2), 100.0],
            "false_discharges": false_discharges(test, acwcm),
            "false_negative_rate_pct": round(100 * (crit_total - kept) / crit_total, 4),
            "discharge_specificity_R5_pct": next(r["recall"] for r in rows_acwcm if r["tier"] == "R5") * 100,
            "macro_f1_pct": round(100 * macro_f1, 2),
            "over_triage_rate_pct": round(100 * over_triage_rate(conf_acwcm), 2),
            "under_triage_count": under_triage(conf_acwcm)[0],
            "under_triage_rate_pct": round(100 * under_triage(conf_acwcm)[0] / len(test), 2),
            "max_under_triage_depth": under_triage(conf_acwcm)[2],
            "abstention_count": abstain_acwcm,
            "abstention_rate_pct": round(100 * abstain_acwcm / n_test, 2),
        },
        "basic_min": {
            "accuracy_pct": round(100 * accuracy(conf_basic), 2),
            "macro_f1_pct": round(100 * float(np.mean([r["f1"] for r in rows_basic])), 2),
            "discharge_specificity_R5_pct": next(r["recall"] for r in rows_basic if r["tier"] == "R5") * 100,
            "over_triage_rate_pct": round(100 * over_triage_rate(conf_basic), 2),
            "under_triage_rate_pct": round(100 * under_triage(conf_basic)[0] / len(test), 2),
        },
        "baselines": baselines,
        "ablation": abl,
        "paired_tests": paired,
        "subgroup_under_triage": subgroups,
        "validation_check": val_rows,
        "acwcm_trigger": {"rule": "consensus", "theta": 0.5, "min_mass": 0.5,
                          "provenance": "fixed a priori; validation split reported as a check, not tuned"},
        "safety_properties": props,
        "safety_properties_all_cases": props_all,
    }

    print("[5/5] Writing committed artifacts to results/ ...")
    RESULTS.mkdir(exist_ok=True)

    # confusion matrix (ACWCM)
    with (RESULTS / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + TIERS + ["support"])
        for t in TIERS:
            w.writerow([t] + [conf_acwcm[t][p] for p in TIERS] + [sum(conf_acwcm[t].values())])
        w.writerow(["total_predicted"] + [sum(conf_acwcm[r][p] for r in TIERS) for p in TIERS]
                   + [sum(sum(conf_acwcm[t].values()) for t in TIERS)])

    # per-tier metrics (ACWCM)
    with (RESULTS / "tier_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_acwcm[0]))
        w.writeheader()
        w.writerows(rows_acwcm)

    # baseline comparison
    with (RESULTS / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baselines[0]))
        w.writeheader()
        w.writerows(baselines)

    # ablation
    with (RESULTS / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(abl[0]))
        w.writeheader()
        w.writerows(abl)

    for name, rows_ in (("paired_tests.csv", paired), ("subgroup_under_triage.csv", subgroups)):
        with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)

    # theorem / property verification
    (RESULTS / "theorem_verification.json").write_text(
        json.dumps({"test_split": props, "all_cases": props_all}, indent=2), encoding="utf-8")

    # master summary
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # overwrite the previously broken figure summary with the real numbers
    fig_dir = ROOT / "evaluation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
