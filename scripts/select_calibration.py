"""
Select the gate calibration on the TRAINING split, confirm on VALIDATION.

The test split is never scored here. Every candidate in src/calibration.py
(128 combinations of discrete, clinically motivated switches and two
conventional majority rules for the ACWCM trigger) is scored on the training
split; the validation split is used only to confirm that the selected profile
also satisfies the hard constraints.

Selection criterion (fixed before the full grid was scored):
  1. Hard constraints on train AND validation: critical-tier (R1/R2)
     sensitivity 100 %, zero false discharges, zero invariant violations
     (conservative preservation, abstention correctness, critical
     non-dilution).
  2. Among feasible profiles, minimise the training mistriage rate
     (over-triage + under-triage = 1 - accuracy).
  3. Ties (within 0.1 percentage point): higher training macro F1, then fewer
     switches changed from v1.

Writes results/calibration_protocol.json. A UTC timestamp is added only with
--stamp (the committed protocol carries it).

    python scripts/select_calibration.py --stamp
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generation"))

from generate_data import SAFEGateDataGenerator  # noqa: E402
from calibration import CANDIDATES, V1  # noqa: E402
from gates.gate1_critical_flags import Gate1CriticalFlags  # noqa: E402
from gates.gate2_moderate_risk import Gate2ModerateRisk  # noqa: E402
from gates.gate3_data_quality import Gate3DataQuality  # noqa: E402
from gates.gate4_titrate_logic import Gate4TiTrATELogic  # noqa: E402
from gates.gate5_uncertainty import Gate5Uncertainty  # noqa: E402
from gates.gate6_temporal_risk import Gate6TemporalRisk  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402

SEED, N_TOTAL = 42, 6400
TIERS = ["R1", "R2", "R3", "R4", "R5"]
IDX = {t: i for i, t in enumerate(TIERS)}
PROTOCOL = ROOT / "results" / "calibration_protocol.json"


def gate_table(cases):
    """Evaluate every gate variant once per case (outputs are cached)."""
    variants = {
        "G1": {(a, e): Gate1CriticalFlags(a, e) for a, e in product((False, True), repeat=2)},
        "G2": {None: Gate2ModerateRisk()},
        "G3": {e: Gate3DataQuality(e) for e in (False, True)},
        "G4": {b: Gate4TiTrATELogic(b) for b in (False, True)},
        "G5": {(c, d): Gate5Uncertainty(age_heuristic=c, escalation_cap=d)
               for c in (True, False) for d in (None, 3)},
        "G6": {None: Gate6TemporalRisk()},
    }
    return [{g: {k: gate.evaluate(c)[:2] for k, gate in vs.items()} for g, vs in variants.items()}
            for c in cases]


def outputs_for(row, cal):
    keys = {"G1": (cal["g1_graded"], cal["vacuous_negatives"]), "G2": None,
            "G3": cal["vacuous_negatives"], "G4": cal["g4_central_only"],
            "G5": (cal["g5_age"], cal["g5_cap"]), "G6": None}
    return {g: row[g][k] for g, k in keys.items()}


def score(cases, table, cal):
    merger = ConservativeMerging("acwcm", "consensus", cal["theta"], cal["min_mass"])
    conf = [[0] * 5 for _ in TIERS]
    miss = fd = viol = 0
    for c, row in zip(cases, table):
        o = outputs_for(row, cal)
        tiers = {g: v[0] for g, v in o.items()}
        final, _, _ = merger.merge(tiers, {g: v[1] for g, v in o.items()})
        ranks = [t.value for t in tiers.values()]
        r_min, fr = min(ranks), final.value
        if any(r == 0 for r in ranks) and fr != 0:
            viol += 1                                    # abstention correctness
        elif fr > r_min + 1 or (r_min in (1, 2) and fr not in (0, 1, 2)):
            viol += 1                                    # CP / critical non-dilution
        if final == RiskTier.R_STAR:
            continue
        t, p = IDX[c["ground_truth_tier"]], IDX[str(final)]
        conf[t][p] += 1
        miss += t < 2 and p >= 2
        fd += t < 2 and p == 4
    n = sum(map(sum, conf))
    acc = sum(conf[i][i] for i in range(5)) / n
    ot = sum(conf[i][j] for i in range(5) for j in range(i)) / n
    f1 = []
    for k in range(5):
        tp = conf[k][k]
        fp = sum(conf[i][k] for i in range(5)) - tp
        fn = sum(conf[k]) - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    crit = sum(conf[0]) + sum(conf[1])
    return {
        "critical_sensitivity_pct": round(100 * (crit - miss) / crit, 2),
        "false_discharges": fd,
        "invariant_violations": viol,
        "accuracy_pct": round(100 * acc, 2),
        "over_triage_pct": round(100 * ot, 2),
        "under_triage_pct": round(100 * (1 - acc - ot), 2),
        "discharge_recall_R5_pct": round(100 * conf[4][4] / sum(conf[4]), 2),
        "macro_f1_pct": round(100 * sum(f1) / 5, 2),
    }


def feasible(m):
    return (m["critical_sensitivity_pct"] == 100.0 and m["false_discharges"] == 0
            and m["invariant_violations"] == 0)


def select():
    gen = SAFEGateDataGenerator(n_total=N_TOTAL, random_state=SEED)
    train, val, _test = gen.split(gen.generate_all_cases())
    del _test  # never scored
    t_train, t_val = gate_table(train), gate_table(val)
    table = []
    for cal in CANDIDATES:
        tr, va = score(train, t_train, cal), score(val, t_val, cal)
        table.append({"calibration": cal,
                      "n_switches_changed": sum(cal[k] != V1[k] for k in V1),
                      "train": tr, "validation": va,
                      "feasible": feasible(tr) and feasible(va)})
    ok = [r for r in table if r["feasible"]]
    top = max(r["train"]["accuracy_pct"] for r in ok)
    near = [r for r in ok if r["train"]["accuracy_pct"] >= top - 0.1]
    best = min(near, key=lambda r: (-r["train"]["macro_f1_pct"], r["n_switches_changed"]))
    v1row = next(r for r in table if r["calibration"] == V1)
    return train, val, table, best, v1row


def main(stamp: bool):
    train, val, table, best, v1row = select()
    out = {
        "selected": best["calibration"],
        "selected_train": best["train"],
        "selected_validation": best["validation"],
        "v1": V1,
        "v1_train": v1row["train"],
        "v1_validation": v1row["validation"],
        "selection_split": (f"train (n={len(train)}); hard constraints confirmed on validation "
                            f"(n={len(val)}); test split not scored"),
        "n_candidates": len(table),
        "n_feasible": sum(r["feasible"] for r in table),
        "criterion": [
            "hard constraints on train and validation: critical sensitivity 100%, "
            "0 false discharges, 0 invariant violations",
            "minimise training mistriage (1 - accuracy)",
            "ties within 0.1 point: higher training macro F1, then fewer switches changed from v1",
        ],
        "provenance": {
            "g1_graded": "ESI handbook: level 1 = immediate life-saving intervention "
                         "(hypotension, extreme heart rate, hypoxaemia, depressed consciousness); "
                         "stroke symptoms and hypertensive crisis = level 2",
            "g4_central_only": "TiTrATE/HINTS: central signs or focal deficits warrant emergent "
                               "work-up; a peripheral HINTS pattern argues against stroke",
            "g5_age": "the surrogate's age cut-offs (>70, <50) have no guideline source; age "
                      "enters through G2 (ABCD2-style age > 60 point)",
            "g5_cap": "design choice: uncertainty may escalate a prediction but not, by itself, "
                      "into the critical band; critical uncertainty is expressed by abstention",
            "vacuous_negatives": "absence of a red flag (G1) or a complete record (G3) is not "
                                 "evidence of a benign course: vacuous belief (confidence 0)",
            "theta_min_mass": "simple (1/2) or qualified (2/3) confidence-weighted majority; "
                              "margin of half or one tier",
        },
        "candidates": table,
    }
    if stamp:
        out["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PROTOCOL.parent.mkdir(exist_ok=True)
    PROTOCOL.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("feasible:", out["n_feasible"], "of", out["n_candidates"])
    print("selected:", best["calibration"])
    print("train:", best["train"])
    print("validation:", best["validation"])


if __name__ == "__main__":
    main("--stamp" in sys.argv)
