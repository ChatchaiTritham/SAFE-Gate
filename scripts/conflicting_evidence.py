"""Conflicting-evidence stress test for the fusion baselines.

The paper argues that expected-performance fusion rules can let a minority safety
signal be outvoted, and that lattice-based conservative merging cannot. On the main
805-case cohort that argument is not visible: Dempster-Shafer matches SAFE-Gate on every
safety metric there, because the synthetic tiers are cleanly separated and the gates
rarely disagree.

This script builds the disagreement case explicitly. One gate reports a critical tier
with confidence c_crit; the remaining gates report a benign tier with confidence
c_benign. Every combination is enumerated over a fixed grid -- no sampling, no seed, and
nothing to tune -- and each fusion rule is asked what it outputs.

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


def arithmetic_average(outputs: dict[str, str], confs: dict[str, float]) -> str:
    """Confidence-weighted mean of tier ranks, rounded to the nearest tier.

    This is the ensemble-averaging baseline expressed directly on the tier scale,
    which is what the fusion comparison in the paper contrasts against.
    """
    rank = {f"R{i}": i for i in range(1, 6)}
    num = sum(confs[g] * rank[t] for g, t in outputs.items())
    den = sum(confs[g] for g in outputs) or 1.0
    return f"R{max(1, min(5, round(num / den)))}"


def acwcm(outputs: dict[str, str], confs: dict[str, float]) -> str:
    tiers = {g: RiskTier[t] for g, t in outputs.items()}
    final, _enforcing, _audit = ConservativeMerging(mode="acwcm").merge(tiers, confs)
    return str(final)


def main() -> int:
    ds = DempsterShaferCombination()
    bma = BayesianModelAveraging()

    rows = []
    for crit_tier, benign_tier, n_dis, c_crit, c_ben in product(
        CRIT_TIERS, BENIGN_TIERS, N_DISSENT, C_CRIT, C_BENIGN
    ):
        # G1 carries the critical signal; n_dis gates dissent toward benign.
        # Any remaining gates abstain from the disagreement by reporting the
        # same benign tier at the same confidence, so the only varying factor
        # is how many voices oppose the critical one.
        outputs = {"G1": crit_tier}
        confs = {"G1": c_crit}
        for i, g in enumerate(GATES[1:], start=1):
            if i <= n_dis:
                outputs[g], confs[g] = benign_tier, c_ben
            else:
                outputs[g], confs[g] = crit_tier, c_crit
        result = {
            "critical_tier": crit_tier,
            "benign_tier": benign_tier,
            "n_dissenting": n_dis,
            "c_critical": c_crit,
            "c_benign": c_ben,
            "dempster_shafer": ds.classify(outputs, confs)["final_tier"],
            "bayesian_model_avg": bma.classify(outputs, confs)["final_tier"],
            "arithmetic_average": arithmetic_average(outputs, confs),
            "safegate_acwcm": acwcm(outputs, confs),
        }
        for method in ("dempster_shafer", "bayesian_model_avg",
                       "arithmetic_average", "safegate_acwcm"):
            result[f"{method}_dilutes"] = result[method] not in CRITICAL
        rows.append(result)

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "conflicting_evidence.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    methods = ["dempster_shafer", "bayesian_model_avg",
               "arithmetic_average", "safegate_acwcm"]
    summary = {
        "n_configurations": len(rows),
        "grid": {
            "critical_tiers": CRIT_TIERS, "benign_tiers": BENIGN_TIERS,
            "n_dissenting": N_DISSENT, "c_critical": C_CRIT, "c_benign": C_BENIGN,
        },
        "dilution_counts": {m: sum(r[f"{m}_dilutes"] for r in rows) for m in methods},
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