"""Conflicting-evidence stress test for the fusion baselines.

The paper argues that expected-performance fusion rules can let a minority safety
signal be outvoted, and that lattice-based conservative merging cannot. On the main
805-case cohort that argument is not visible: Dempster-Shafer matches SAFE-Gate on every
safety metric there, because the synthetic tiers are cleanly separated and the gates
rarely disagree.

This script builds the disagreement case explicitly. For n = 1..5 dissenting gates,
n gates report a benign tier with confidence c_benign and the remaining 6 - n gates
report the same critical tier with confidence c_crit (so n = 5 is one critical gate
against five benign ones). Every combination of a fixed grid is enumerated -- no
sampling, no seed, nothing tuned -- and each fusion rule is asked what it outputs.

Exact belief ties (e.g. three critical gates at 0.99 against three benign gates at
0.99) are resolved by the baselines' argmax toward the first, i.e. more critical,
tier. Because that choice can matter, the counts are also reported with ties broken
toward the LESS critical tier (tolerance 1e-9), as a tie-break sensitivity.

A configuration is a DILUTION when a rule returns a tier outside the critical band
{R1, R2} while at least one gate reported inside it. That is exactly the failure the
lattice invariants forbid.

Run:  python scripts/conflicting_evidence.py
Writes: results/conflicting_evidence.csv
        results/conflicting_evidence.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination  # noqa: E402
from baselines.sugeno import SugenoAdditive, WeightedMax  # noqa: E402
from calibration import V2  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402

GATES = ["G1", "G2", "G3", "G4", "G5", "G6"]
CRITICAL = {"R1", "R2"}

# Grid. Confidences span the range a calibrated gate can plausibly report; the
# dissent count spans one dissenting gate through all five.
C_CRIT = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
C_BENIGN = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
N_DISSENT = [1, 2, 3, 4, 5]
CRIT_TIERS = ["R1", "R2"]
BENIGN_TIERS = ["R4", "R5"]
METHODS = ["dempster_shafer", "bayesian_model_avg",
           "dempster_shafer_tie_benign", "bayesian_model_avg_tie_benign",
           "arithmetic_average", "sugeno_additive", "weighted_max", "safegate_acwcm"]


def arithmetic_average(outputs: dict[str, str], confs: dict[str, float]) -> str:
    """Confidence-weighted mean of tier ranks, rounded to the nearest tier.

    This is the ensemble-averaging baseline expressed directly on the tier scale,
    which is what the fusion comparison in the paper contrasts against.
    """
    rank = {f"R{i}": i for i in range(1, 6)}
    num = sum(confs[g] * rank[t] for g, t in outputs.items())
    den = sum(confs[g] for g in outputs) or 1.0
    return f"R{max(1, min(5, round(num / den)))}"


def tie_benign(scores: dict[str, float], tol: float = 1e-9) -> str:
    """Argmax over tiers with exact ties (within tol) broken toward the LESS
    critical tier -- the tie-break sensitivity counterpart of the default."""
    best = max(scores.values())
    return [t for t in ["R1", "R2", "R3", "R4", "R5"] if scores[t] >= best - tol][-1]


def acwcm(outputs: dict[str, str], confs: dict[str, float]) -> str:
    tiers = {g: RiskTier[t] for g, t in outputs.items()}
    merger = ConservativeMerging(mode="acwcm", theta=V2["theta"], min_mass=V2["min_mass"])
    final, _enforcing, _audit = merger.merge(tiers, confs)
    return str(final)


def main() -> int:
    ds = DempsterShaferCombination()
    bma = BayesianModelAveraging()
    sug, wmax = SugenoAdditive(), WeightedMax()

    rows = []
    for crit_tier, benign_tier, n_dis, c_crit, c_ben in product(
        CRIT_TIERS, BENIGN_TIERS, N_DISSENT, C_CRIT, C_BENIGN
    ):
        # G1 carries the critical signal; n_dis gates (G2..G{n_dis+1}) dissent
        # toward the benign tier at c_benign. The non-dissenting gates report the
        # SAME critical tier at c_crit, so there are 6 - n_dis critical votes
        # against n_dis benign ones.
        outputs = {"G1": crit_tier}
        confs = {"G1": c_crit}
        for i, g in enumerate(GATES[1:], start=1):
            if i <= n_dis:
                outputs[g], confs[g] = benign_tier, c_ben
            else:
                outputs[g], confs[g] = crit_tier, c_crit
        ds_res = ds.classify(outputs, confs)
        bma_res = bma.classify(outputs, confs)
        result = {
            "critical_tier": crit_tier,
            "benign_tier": benign_tier,
            "n_dissenting": n_dis,
            "c_critical": c_crit,
            "c_benign": c_ben,
            "dempster_shafer": ds_res["final_tier"],
            "bayesian_model_avg": bma_res["final_tier"],
            "dempster_shafer_tie_benign": tie_benign(ds_res["beliefs"]),
            "bayesian_model_avg_tie_benign": tie_benign(bma_res["probabilities"]),
            "arithmetic_average": arithmetic_average(outputs, confs),
            "sugeno_additive": sug.classify(outputs, confs)["final_tier"],
            "weighted_max": wmax.classify(outputs, confs)["final_tier"],
            "safegate_acwcm": acwcm(outputs, confs),
        }
        for method in METHODS:
            result[f"{method}_dilutes"] = result[method] not in CRITICAL
        rows.append(result)

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "conflicting_evidence.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    methods = METHODS
    summary = {
        "n_configurations": len(rows),
        "grid": {
            "critical_tiers": CRIT_TIERS, "benign_tiers": BENIGN_TIERS,
            "n_dissenting": N_DISSENT, "c_critical": C_CRIT, "c_benign": C_BENIGN,
        },
        "dilution_counts": {m: sum(r[f"{m}_dilutes"] for r in rows) for m in methods},
        "r1_to_r5_counts": {
            m: sum(1 for r in rows if r["critical_tier"] == "R1" and r[m] == "R5")
            for m in methods
        },
        "exact_ties": {
            "dempster_shafer": sum(r["dempster_shafer"] != r["dempster_shafer_tie_benign"] for r in rows),
            "bayesian_model_avg": sum(r["bayesian_model_avg"] != r["bayesian_model_avg_tie_benign"] for r in rows),
        },
        "dilution_rate_pct": {
            m: round(100.0 * sum(r[f"{m}_dilutes"] for r in rows) / len(rows), 2)
            for m in methods
        },
    }

    # The smallest dissent that already breaks each rule, and a worked example.
    for m in methods:
        bad = [r for r in rows if r[f"{m}_dilutes"]]
        if bad:
            first = min(bad, key=lambda r: (r["n_dissenting"], -r["c_critical"]))
            summary.setdefault("first_failure", {})[m] = {
                "n_dissenting": first["n_dissenting"],
                "c_critical": first["c_critical"],
                "c_benign": first["c_benign"],
                "critical_tier": first["critical_tier"],
                "benign_tier": first["benign_tier"],
                "output": first[m],
            }
        else:
            summary.setdefault("first_failure", {})[m] = None

    with open(ROOT / "results" / "conflicting_evidence.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"configurations: {summary['n_configurations']}")
    for m in methods:
        print(f"  {m:22s} dilutes {summary['dilution_counts'][m]:5d}  "
              f"({summary['dilution_rate_pct'][m]:5.2f}%)")
    print("\nfirst failure (fewest dissenting gates, then highest critical confidence):")
    for m in methods:
        print(f"  {m:22s} {summary['first_failure'][m]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())