"""
Secondary analyses added for the IJAR revision (post-hoc; the frozen round-4
protocol, its cohorts and results/round4_primary.csv are not changed).

Reuses the frozen cohorts (digests checked by round4.load_frozen) and the
gate outputs of the released SAFE-Gate v2 calibration, and adds:

  1. disjunctive (Dubois-Prade / Smets DRC) combination of the singleton gate
     supports, with pignistic decision (critical or benign tie-breaking) and a
     credal (upper expected cost) decision; Dempster's rule with the same
     credal decision; Dempster and cautious rules with confidences capped at
     0.999 (non-dogmatic masses);
  2. the minimum and the any-dissent extreme of the band-bounded class in the
     enumeration and the conflict grids;
  3. a (theta, mu) sweep of the ACWCM trigger on the two fresh cohorts that do
     not share the development generator (over-triage vs R3 under-triage);
  4. Clopper-Pearson CIs for over- and under-triage;
  5. the break-even cost ratio over several miss / false-alarm probability ratios;
  6. a non-expert HINTS scenario (central-sign sensitivity 0.80 instead of 0.98)
     on the diagnosis-conditioned generator with the frozen seed, and a
     re-labelling scenario with vertebrobasilar TIA in R1.

Output: results/ijar_r3_analyses.json, results/ijar_r3_cohort_rules.csv,
results/ijar_r3_trigger_sweep.csv.
Run: python scripts/ijar_r3_analyses.py
"""
from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round4 as r4  # noqa: E402
from baselines import belief_variants as bv  # noqa: E402
from baselines.dempster_shafer import argmax_tol  # noqa: E402
import generate_diagnosis as gd  # noqa: E402

TIERS, IDX, RANK, CRIT, GATES = r4.TIERS, r4.IDX, r4.RANK, r4.CRIT, r4.GATES
THETA = bv.THETA
CAP = 0.999
LAM = bv.LAMBDA


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def disjunctive(srcs):
    """Disjunctive rule on simple support functions: each choice of focal
    elements contributes the product of masses to the union of the chosen sets."""
    m = {}
    for choice in product((0, 1), repeat=len(srcs)):
        p, u = 1.0, frozenset()
        for (a, c), ch in zip(srcs, choice):
            s, w = (a, c) if ch == 0 else (THETA, 1.0 - c)
            p *= w
            u = u | s
        if p > 0.0:
            m[u] = m.get(u, 0.0) + p
    return m or {THETA: 1.0}


def upper_cost_decision(m, lam=LAM):
    """Credal (Gamma-minimax) decision: minimise the upper expected cost over the
    credal set of m; cost d for over-triage by d tiers, lam*d for under-triage."""
    def cost(pred, t):
        d = IDX[pred] - t
        return d * lam if d > 0 else -d
    z = sum(m.values()) or 1.0
    up = {p: -sum(v / z * max(cost(p, t) for t in a) for a, v in m.items()) for p in TIERS}
    return argmax_tol(up, TIERS)


def srcs_of(o, c, cap=None, drop_vacuous=False):
    out = []
    for a, cc in bv.sources(o, c, "singleton", "critical"):
        if drop_vacuous and cc <= 0.0:
            continue
        out.append((a, min(cc, cap) if cap else cc))
    return out


def make_rules():
    acw = r4.acwcm_fn()

    def mn(o, c):
        if any(t == "R*" for t in o.values()):
            return "R*"
        return min(o.values(), key=lambda t: RANK[t])

    def any_dissent(o, c):
        # least urgent band-bounded operator (Proposition 3 extreme)
        if any(t == "R*" for t in o.values()):
            return "R*"
        r = min(o.values(), key=lambda t: RANK[t])
        if r in ("R3", "R4") and any(RANK[t] > RANK[r] for t in o.values()):
            return TIERS[IDX[r] + 1]
        return r

    def dempster(o, c, cap=None, decision="pignistic"):
        m = bv.combine(srcs_of(o, c, cap), "dempster")
        return upper_cost_decision(m) if decision == "upper" else argmax_tol(bv.betp(m), TIERS)

    def cautious(o, c, cap=None):
        m = bv.combine(srcs_of(o, c, cap), "cautious")
        return argmax_tol(bv.betp(m), TIERS)

    def disj(o, c, decision="pignistic"):
        # Silent gates (confidence 0) carry no source: under the disjunctive rule a
        # vacuous source would make the whole combination vacuous.
        m = disjunctive(srcs_of(o, c, drop_vacuous=True))
        if decision == "upper":
            return upper_cost_decision(m)
        return argmax_tol(bv.betp(m), TIERS, "benign" if decision == "benign_ties" else "critical")

    return {
        "SAFE-Gate ACWCM (v2)": acw,
        "MIN": mn,
        "Any-dissent (band-bounded extreme)": any_dissent,
        "DS Dempster exact (BetP)": lambda o, c: dempster(o, c),
        "DS Dempster exact, conf<=0.999": lambda o, c: dempster(o, c, CAP),
        "DS cautious, conf<=0.999": lambda o, c: cautious(o, c, CAP),
        "DS Dempster exact (upper expected cost)": lambda o, c: dempster(o, c, decision="upper"),
        "DS disjunctive (BetP, critical ties)": lambda o, c: disj(o, c),
        "DS disjunctive (BetP, benign ties)": lambda o, c: disj(o, c, "benign_ties"),
        "DS disjunctive (upper expected cost)": lambda o, c: disj(o, c, "upper"),
    }


# --------------------------------------------------------------------------- #
# Trigger sweep
# --------------------------------------------------------------------------- #
def acwcm_param(o, c, theta, mu):
    if any(t == "R*" for t in o.values()):
        return "R*"
    if any(t == "R1" for t in o.values()):
        return "R1"
    r = min(o.values(), key=lambda t: RANK[t])
    C = sum(c[g] for g in o)
    if r in ("R2", "R5") or C <= 0:
        return r
    tau = sum(c[g] * RANK[o[g]] for g in o) / C
    above = sum(c[g] for g in o if RANK[o[g]] > RANK[r]) / C
    if tau >= RANK[r] + theta and above >= mu:
        return TIERS[IDX[r] + 1]
    return r


def cp_pct(k, n):
    lo, hi = r4.cp(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


def rate_cis(cases, preds):
    dec = [i for i, p in enumerate(preds) if p in IDX]
    d = [IDX[preds[i]] - IDX[cases[i]["ground_truth_tier"]] for i in dec]
    return {"over_ci95": cp_pct(sum(x < 0 for x in d), len(dec)),
            "under_ci95": cp_pct(sum(x > 0 for x in d), len(dec))}


def main():
    proto, cohorts = r4.load_frozen()
    rules = make_rules()
    out = {"note": "post-hoc secondary analyses; frozen protocol unchanged",
           "protocol_sha256": __import__("hashlib").sha256(r4.PROTOCOL.read_bytes()).hexdigest()}
    rows, sweep = [], []
    names = ("fresh_independent_generator", "fresh_diagnosis_generator")
    for name in ("fresh_main_generator",) + names:
        cs = cohorts[name]
        io = r4.gate_io(cs)
        P = {k: [f(o, c) for o, c, _ in io] for k, f in rules.items()}
        # the pure re-implementation of the trigger must reproduce the released code
        assert [acwcm_param(o, c, 0.5, 2 / 3) for o, c, _ in io] == [fin for _, _, fin in io]
        assert P["SAFE-Gate ACWCM (v2)"] == [fin for _, _, fin in io]
        for k, p in P.items():
            rows.append({"cohort": name, "method": k, **r4.metrics(cs, p), **rate_cis(cs, p)})
        if name in names:
            grid = [(th, mu) for th in (0.0, 0.5, 1.0) for mu in [round(0.05 * i, 2) for i in range(1, 21)]]
            for th, mu in grid + [(0.5, 2 / 3)]:
                p = [acwcm_param(o, c, th, mu) for o, c, _ in io]
                m = r4.metrics(cs, p)
                sweep.append({"cohort": name, "theta": th, "mu": round(mu, 4), **{
                    k: m[k] for k in ("critical_sensitivity_pct", "accuracy_pct", "over_triage_pct",
                                      "under_triage_pct", "r3_under_triage_pct", "under_2plus")}})
            for k in ("MIN", "Any-dissent (band-bounded extreme)"):
                m = r4.metrics(cs, P[k])
                sweep.append({"cohort": name, "theta": k, "mu": "", **{
                    x: m[x] for x in ("critical_sensitivity_pct", "accuracy_pct", "over_triage_pct",
                                      "under_triage_pct", "r3_under_triage_pct", "under_2plus")}})
        if name == "fresh_diagnosis_generator":
            # Re-labelling scenario: vertebrobasilar TIA as R1 instead of R2.
            rl = [dict(c, ground_truth_tier="R1") if c.get("diagnosis", "").lower().find("tia") >= 0
                  and c["ground_truth_tier"] == "R2" else c for c in cs]
            n_tia = sum(1 for a, b in zip(cs, rl) if a is not b)
            out["vbtia_as_R1"] = {"n_relabelled": n_tia, **{
                k: {"critical_sensitivity_pct": r4.metrics(rl, P[k])["critical_sensitivity_pct"],
                    "R1_under_pct": r4.metrics(rl, P[k]).get("R1_under_pct"),
                    "relabelled_returned_R1": sum(1 for a, b, p in zip(cs, rl, P[k]) if a is not b and p == "R1")}
                for k in ("SAFE-Gate ACWCM (v2)", "MIN", "DS Dempster exact (BetP)")}}
    # Pareto check on (over-triage, R3 under-triage) for each cohort.
    par = {}
    for name in names:
        pts = [s for s in sweep if s["cohort"] == name]
        v2 = next(s for s in pts if s["theta"] == 0.5 and s["mu"] == round(2 / 3, 4))
        dom = [s for s in pts if s["over_triage_pct"] <= v2["over_triage_pct"]
               and s["r3_under_triage_pct"] <= v2["r3_under_triage_pct"]
               and (s["over_triage_pct"], s["r3_under_triage_pct"]) != (v2["over_triage_pct"], v2["r3_under_triage_pct"])]
        par[name] = {"v2": v2, "n_points": len(pts), "points_dominating_v2": dom}
    out["pareto_v2"] = par

    # Enumeration, conflict grids, break-even over miss/false-alarm ratios.
    res, n_het, n_fa = r4.conflict_grid(rules)
    prev = json.loads((r4.RES / "round4_summary.json").read_text())["conflict"]["prevalence_critical_dev"]
    out["conflict"] = {"n_hetero": n_het, "n_false_alarm": n_fa, "rules": res,
                       "prevalence_critical_dev": prev,
                       "breakeven_by_ratio_qf_over_qd": {str(q): r4.breakeven(res, prev, ratio=q)
                                                        for q in (0.25, 0.5, 1.0, 2.0, 4.0)}}
    out["enumeration"] = r4.extended_enumeration(rules)["rules"]

    # Non-expert HINTS scenario (not frozen; same generator seed).
    meta = proto["cohorts"]["fresh_diagnosis_generator"]
    orig = gd.SENS_CENTRAL
    gd.SENS_CENTRAL = 0.80
    try:
        cs = r4.make(meta["generator"], meta["seed"], meta["n"])
    finally:
        gd.SENS_CENTRAL = orig
    io = r4.gate_io(cs)
    ne = {}
    for k in ("SAFE-Gate ACWCM (v2)", "MIN", "DS Dempster exact (BetP)", "DS disjunctive (BetP, critical ties)"):
        p = [rules[k](o, c) for o, c, _ in io]
        ne[k] = r4.metrics(cs, p)
    ne["gate_recall_union"] = r4.gate_recall_union(cs, io)
    out["nonexpert_hints_sens080"] = ne

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(r4.RES / "ijar_r3_cohort_rules.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    with open(r4.RES / "ijar_r3_trigger_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    (r4.RES / "ijar_r3_analyses.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
