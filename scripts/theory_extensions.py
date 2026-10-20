"""Exhaustive checks for the three theory extensions (IJAR v2.1.0).

E1  lambda-Lipschitz bound on relaxation (Theorem "lambda-smooth relaxation").
E2  Weak Pareto optimality of the quota chain and a dominance counterexample
    for the full band-bounded class.
E3  Non-dilution as an aggregation-function property: relation table.

Everything is exhaustive and deterministic (no random numbers).
Tiers are ranks 1..K (1 = most urgent); band = ranks <= c.
Output: results/theory_extensions.json, results/theory_relations.csv
"""
import itertools
import json
import heapq
from fractions import Fraction as F
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"


def neighbours(r, K):
    for i in range(len(r)):
        for d in (-1, 1):
            v = r[i] + d
            if 1 <= v <= K:
                yield r[:i] + (v,) + r[i + 1:]


# ---------------------------------------------------------------- E1
def envelope(K, c, n, lam, internal=True):
    """Pointwise largest rank of a lambda-Lipschitz f with rank f <= c on
    profiles whose floor is in the band and, if internal, rank f <= max rank.
    Computed as min_s h(s) + lam * d(r, s) by multi-source Dijkstra."""
    prof = list(itertools.product(range(1, K + 1), repeat=n))
    dist = {}
    pq = []
    for r in prof:
        top = max(r) if internal else K
        h = min(c, top) if min(r) <= c else top
        dist[r] = h
        heapq.heappush(pq, (h, r))
    while pq:
        d, r = heapq.heappop(pq)
        if d > dist[r]:
            continue
        for s in neighbours(r, K):
            if d + lam < dist[s]:
                dist[s] = d + lam
                heapq.heappush(pq, (d + lam, s))
    return dist


def formula(r, c, lam, K=None):
    j = min(r)
    top = max(r) if K is None else K
    return min(top, c + lam * max(0, j - c))


def is_feasible(f, K, c, lam, internal=True):
    for r, v in f.items():
        if not (min(r) <= v <= (max(r) if internal else K)):
            return False
        if min(r) <= c and v > c:
            return False
        for s in neighbours(r, K):
            if abs(f[s] - v) > lam:
                return False
    return True


def e1():
    rows = []
    for K, n in ((5, 2), (5, 3), (5, 4), (5, 6), (7, 3)):
        for c in range(1, K):
            for lam in (1, 2, 3, 4):
                env = envelope(K, c, n, lam)
                fm = {r: formula(r, c, lam) for r in env}
                env0 = envelope(K, c, n, lam, internal=False)
                fm0 = {r: formula(r, c, lam, K) for r in env0}
                rows.append(dict(K=K, n=n, c=c, lam=lam,
                                 envelope_equals_formula=all(env[r] == fm[r] for r in env),
                                 formula_feasible=is_feasible(fm, K, c, lam),
                                 noninternal_envelope_equals_formula=all(env0[r] == fm0[r] for r in env0),
                                 noninternal_formula_feasible=is_feasible(fm0, K, c, lam, internal=False),
                                 max_relax=max(fm[r] - min(r) for r in fm)))
    # brute force over ALL maps L^2 -> L for K=3 and K=4 (c=1): the pointwise
    # maximum over feasible maps equals the formula
    brute = []
    for K in (3, 4):
        c = 1
        prof = list(itertools.product(range(1, K + 1), repeat=2))
        choices = [list(range(min(r), max(r) + 1)) if min(r) > c else
                   list(range(min(r), min(c, max(r)) + 1)) for r in prof]
        for lam in (1, 2, 3):
            best = {r: 0 for r in prof}
            nfeas = 0
            for vals in itertools.product(*choices):
                f = dict(zip(prof, vals))
                if is_feasible(f, K, c, lam):
                    nfeas += 1
                    for r in prof:
                        best[r] = max(best[r], f[r])
            brute.append(dict(K=K, n=2, c=c, lam=lam, feasible_maps=nfeas,
                              max_equals_formula=all(best[r] == formula(r, c, lam) for r in prof)))
    # every CP map (f in {floor, floor+1}) is 2-Lipschitz: all CP maps, K=4, n=2
    K = 4
    prof = list(itertools.product(range(1, K + 1), repeat=2))
    worst = 0
    for vals in itertools.product((0, 1), repeat=len(prof)):
        f = {r: min(K, min(r) + v) for r, v in zip(prof, vals)}
        for r in prof:
            for s in neighbours(r, K):
                worst = max(worst, abs(f[r] - f[s]))
    return dict(envelope=rows, brute_force=brute, cp_maps_max_jump_K4_n2=worst)


# ---------------------------------------------------------------- ACWCM (tiers only)
def acwcm(r, conf, theta=F(1, 2), mu=F(2, 3)):
    if 1 in r:
        return 1
    j = min(r)
    C = sum(conf)
    if C == 0 or j in (2, 5):
        return j
    tau = sum(ci * ri for ci, ri in zip(conf, r)) / C
    m = sum(ci for ci, ri in zip(conf, r) if ri > j) / C
    return j + 1 if (tau >= j + theta and m >= mu) else j


def quota(r, conf, mu, strict=False):
    j = min(r)
    if j not in (3, 4):
        return j
    C = sum(conf)
    if C == 0:
        return j
    m = sum(ci for ci, ri in zip(conf, r) if ri > j) / C
    return j + 1 if ((m > mu) if strict else (m >= mu)) else j


# ---------------------------------------------------------------- E2
CONFS = [tuple(F(x, 10) for x in p) for p in (
    (8, 8, 8, 8, 8, 8), (9, 7, 5, 3, 2, 1), (10, 0, 4, 6, 2, 8),
    (2, 4, 6, 8, 10, 0), (5, 5, 10, 1, 3, 7), (6, 2, 2, 9, 9, 4),
    (1, 1, 1, 10, 10, 10), (3, 9, 6, 0, 8, 5))]


def e2():
    mus = [F(k, 20) for k in range(1, 21)]
    prof = list(itertools.product(range(1, 6), repeat=6))
    nested = True
    acwcm_is_q23 = True
    for conf in CONFS:
        for r in prof:
            outs = [quota(r, conf, mu) for mu in mus]
            # relaxation sets are nested: relaxed at mu implies relaxed at every smaller mu
            if any(outs[a] < outs[a + 1] for a in range(len(mus) - 1)):
                nested = False
            if 1 not in r and acwcm(r, conf) != quota(r, conf, F(2, 3)):
                acwcm_is_q23 = False
    # dominance counterexample for the full band-bounded class (n = 3, conf 0.8)
    conf = (F(8, 10),) * 3
    A = (3, 4, 4)   # share above floor 2/3 -> ACWCM relaxes to R4; true tier R3
    B = (3, 3, 4)   # share 1/3 -> ACWCM keeps R3; true tier R4
    dist = [(A, 3, F(1, 2)), (B, 4, F(1, 2))]

    def ot_ut(f):
        ot = sum(p for r, t, p in dist if f(r) < t)
        ut = sum(p for r, t, p in dist if f(r) > t)
        return ot, ut

    def g(r):  # band-bounded: relax B, keep A
        return 4 if r == B else min(r)
    a_ot, a_ut = ot_ut(lambda r: acwcm(r, conf))
    g_ot, g_ut = ot_ut(g)
    return dict(quota_relaxation_sets_nested=nested, acwcm_equals_Q23_on_grid=acwcm_is_q23,
                mu_grid=[str(m) for m in mus], confidence_profiles=len(CONFS),
                counterexample=dict(A=A, B=B, acwcm_out=[acwcm(A, conf), acwcm(B, conf)],
                                    acwcm_OT=str(a_ot), acwcm_UT=str(a_ut),
                                    g_OT=str(g_ot), g_UT=str(g_ut),
                                    g_strictly_dominates=(g_ot < a_ot and g_ut < a_ut)))


# ---------------------------------------------------------------- E3
K, N = 5, 3
P3 = list(itertools.product(range(1, K + 1), repeat=N))
CONF3 = {"homog": (F(8, 10),) * 3, "heterog": (F(9, 10), F(5, 10), F(2, 10))}


def u(rank):
    return K + 1 - rank  # urgency code 1..5 (R1 -> 5)


def sugeno(r, nu):
    us = [u(x) for x in r]
    best = 0
    for lev in range(1, K + 1):
        A = frozenset(i for i in range(N) if us[i] >= lev)
        best = max(best, min(lev, nu(A)))
    return K + 1 - max(best, 1)


def nu_plus(A):
    return -(-K * len(A) // N)  # ceil(5|A|/3), additive homogeneous capacity


def nu_nd(A):  # singletons 4 = code of R2, larger sets 5
    return 0 if not A else (4 if len(A) == 1 else 5)


OPS = {
    "Minimum (rank)": lambda r: min(r),
    "Maximum (rank)": lambda r: max(r),
    "Median": lambda r: sorted(r)[1],
    "Any-dissent": lambda r: min(r) + (1 if min(r) in (3, 4) and max(r) > min(r) else 0),
    "ACWCM v2, homogeneous": lambda r: acwcm(r, CONF3["homog"]),
    "ACWCM v2, heterogeneous": lambda r: acwcm(r, CONF3["heterog"]),
    "Sugeno nu_plus": lambda r: sugeno(r, nu_plus),
    "Sugeno nu({i})=4": lambda r: sugeno(r, nu_nd),
    "Escalate-all": lambda r: 1 if min(r) < K else K,
    "R2-clamp": lambda r: (min(r) if min(r) > 2 else (1 if max(r) == 1 else 2)),
    "R1-hold median": lambda r: 1 if 1 in r else sorted(r)[1],
}


def props(f):
    vals = {r: f(r) for r in P3}
    nd = lambda c: all(vals[r] <= c for r in P3 if min(r) <= c)
    mono = all(vals[r] >= vals[s] for r in P3 for s in neighbours(r, K) if all(a >= b for a, b in zip(r, s)))
    phis = [p for p in itertools.product(range(1, K + 1), repeat=K) if all(p[i] <= p[i + 1] for i in range(K - 1))]
    stable = all(f(tuple(p[x - 1] for x in r)) == p[vals[r] - 1] for p in phis for r in P3)
    lip = max(abs(vals[r] - vals[s]) for r in P3 for s in neighbours(r, K))
    return {
        "monotone": mono,
        "idempotent": all(vals[(a,) * N] == a for a in range(1, K + 1)),
        "internal": all(min(r) <= vals[r] <= max(r) for r in P3),
        "conjunctive_rank": all(vals[r] <= min(r) for r in P3),
        "R1_absorbing": all(vals[r] == 1 for r in P3 if 1 in r),
        "ND_c1": nd(1), "ND_c2": nd(2), "ND_all": all(nd(c) for c in range(1, K)),
        "ordinally_stable": stable,
        "lipschitz": lip,
        "relaxes_floor_rank_ge2": any(vals[r] > min(r) for r in P3 if min(r) >= 2),
    }


def e3():
    table = {name: props(f) for name, f in OPS.items()}
    # general equivalences checked over ALL maps L^2 -> L, K = 3
    Kt = 3
    prof = list(itertools.product(range(1, Kt + 1), repeat=2))
    eq_all, eq_c1, cor = True, True, True
    for vals in itertools.product(range(1, Kt + 1), repeat=len(prof)):
        f = dict(zip(prof, vals))
        ndc = {c: all(f[r] <= c for r in prof if min(r) <= c) for c in range(1, Kt)}
        conj = all(f[r] <= min(r) for r in prof)
        absorb = all(f[r] == 1 for r in prof if 1 in r)
        eq_all &= (all(ndc.values()) == conj)
        eq_c1 &= (ndc[1] == absorb)
        internal = all(min(r) <= f[r] <= max(r) for r in prof)
        if internal and conj and any(f[r] != min(r) for r in prof):
            cor = False
    return dict(table=table, ND_all_iff_conjunctive=eq_all, ND_c1_iff_R1_absorbing=eq_c1,
                internal_and_conjunctive_implies_minimum=cor)


def main():
    res = {"E1": e1(), "E2": e2(), "E3": e3()}
    OUT.mkdir(exist_ok=True)
    with open(OUT / "theory_extensions.json", "w", encoding="utf8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=str, sort_keys=True)
        fh.write("\n")
    keys = list(next(iter(res["E3"]["table"].values())).keys())
    with open(OUT / "theory_relations.csv", "w", encoding="utf8", newline="\n") as fh:
        fh.write("operator," + ",".join(keys) + "\n")
        for name, p in res["E3"]["table"].items():
            fh.write(name + "," + ",".join(str(p[k]) for k in keys) + "\n")
    e1r = res["E1"]
    print("E1 envelope==formula:", all(x["envelope_equals_formula"] for x in e1r["envelope"]),
          "feasible:", all(x["formula_feasible"] for x in e1r["envelope"]),
          "non-internal:", all(x["noninternal_envelope_equals_formula"] and x["noninternal_formula_feasible"]
                               for x in e1r["envelope"]),
          "brute:", all(x["max_equals_formula"] for x in e1r["brute_force"]),
          "CP max jump:", e1r["cp_maps_max_jump_K4_n2"])
    print("E2", {k: v for k, v in res["E2"].items() if k != "mu_grid"})
    print("E3", {k: v for k, v in res["E3"].items() if k != "table"})
    for name, p in res["E3"]["table"].items():
        print(f"  {name:26s}", p)


if __name__ == "__main__":
    main()
