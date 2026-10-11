"""
Exhaustive check of aggregation properties for every fusion rule.

All 6^6 = 46,656 assignments of the six gate outputs over
{R*, R1, R2, R3, R4, R5} are enumerated under two fixed confidence profiles
(homogeneous 0.8 for every gate; heterogeneous 0.9/0.5/0.7/0.6/0.8/0.4), so
93,312 inputs per rule, with no sampling. For each rule we count violations of

  CP   conservative preservation: no R* input  =>  rank(out) <= rank(r_min) + 1
  AC   abstention correctness:    some input R* =>  out = R*
  CND  critical non-dilution:     r_min in {R1, R2} => out in {R*, R1, R2}
  R1D  R1 never discharged:       some input R1 => out != R5
  MON  monotonicity: making one gate one tier more urgent never makes the
       output less urgent (checked on the 5^6 abstention-free inputs)
  IDEM idempotence: all six gates equal t in R1..R5  =>  out = t

The Sugeno-family rules are known to be monotone and idempotent (Marichal
2001); the enumeration confirms it for the implementation and shows which
safety properties they lack.

    python scripts/invariant_enumeration.py
Writes results/invariant_enumeration.csv and .json
"""
from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination  # noqa: E402
from baselines.sugeno import SugenoAdditive, WeightedMax  # noqa: E402
from calibration import V1, V2  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402

GATES = ["G1", "G2", "G3", "G4", "G5", "G6"]
ALL = ["R*", "R1", "R2", "R3", "R4", "R5"]
RANK = {t: i for i, t in enumerate(ALL)}
PROFILES = {
    "homogeneous": dict(zip(GATES, [0.8] * 6)),
    "heterogeneous": dict(zip(GATES, [0.9, 0.5, 0.7, 0.6, 0.8, 0.4])),
}


def _acwcm(theta, mm):
    m = ConservativeMerging("acwcm", "consensus", theta, mm)

    def f(out, conf):
        return str(m.merge({g: RiskTier[("R_STAR" if t == "R*" else t)] for g, t in out.items()},
                           conf)[0])
    return f


def _min(out, conf):
    return min(out.values(), key=RANK.get)


def _clean(out):
    return {g: (t if t != "R*" else "R1") for g, t in out.items()}


def _avg(out, conf):
    c = _clean(out)
    num = sum(conf[g] * RANK[t] for g, t in c.items())
    return f"R{max(1, min(5, round(num / sum(conf.values()))))}"


_ds, _bma, _sa, _wm = DempsterShaferCombination(), BayesianModelAveraging(), SugenoAdditive(), WeightedMax()
RULES = {
    "SAFE-Gate ACWCM (v2)": _acwcm(V2["theta"], V2["min_mass"]),
    "SAFE-Gate ACWCM (v1)": _acwcm(V1["theta"], V1["min_mass"]),
    "SAFE-Gate MIN": _min,
    "Sugeno integral (additive capacity)": lambda o, c: _sa.classify(o, c)["final_tier"],
    "Ordinal weighted maximum": lambda o, c: _wm.classify(o, c)["final_tier"],
    "Dempster-Shafer": lambda o, c: _ds.classify(_clean(o), c)["final_tier"],
    "Bayesian model averaging": lambda o, c: _bma.classify(_clean(o), c)["final_tier"],
    "Weighted arithmetic mean": _avg,
}


def enumerate_rule(f):
    v = dict.fromkeys(["CP", "AC", "CND", "R1D", "MON", "IDEM"], 0)
    n = n_mon = 0
    for conf in PROFILES.values():
        cache = {}
        for tup in product(ALL, repeat=6):
            out = dict(zip(GATES, tup))
            r = f(out, conf)
            cache[tup] = r
            n += 1
            ranks = [RANK[t] for t in tup]
            rmin = min(ranks)
            has_star = rmin == 0
            if has_star and r != "R*":
                v["AC"] += 1
            if not has_star and RANK[r] > rmin + 1:
                v["CP"] += 1
            if rmin in (1, 2) and r not in ("R*", "R1", "R2"):
                v["CND"] += 1
            if "R1" in tup and r == "R5":
                v["R1D"] += 1
            if len(set(tup)) == 1 and tup[0] != "R*" and r != tup[0]:
                v["IDEM"] += 1
        for tup in product(ALL[1:], repeat=6):
            base = RANK[cache[tup]]
            for i, t in enumerate(tup):
                if t == "R1":
                    continue
                up = tup[:i] + (ALL[RANK[t] - 1],) + tup[i + 1:]
                n_mon += 1
                r_up = RANK[cache[up]]
                if base != 0 and r_up != 0 and r_up > base:
                    v["MON"] += 1
    return n, n_mon, v


def main():
    rows = []
    for name, f in RULES.items():
        n, n_mon, v = enumerate_rule(f)
        rows.append({"rule": name, "n_inputs": n, "n_monotonicity_pairs": n_mon,
                     **{f"{k}_violations": x for k, x in v.items()}})
        print(name, v, flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with (out / "invariant_enumeration.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "invariant_enumeration.json").write_text(
        json.dumps({"profiles": PROFILES, "rows": rows}, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
