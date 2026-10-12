"""
Exhaustive computational check of the finite-chain results of the IJAR paper.

Every statement is checked by enumeration on a finite instance; no sampling.
Tiers are ranks 1..5 (R1 most urgent); confidences are integers 0..Q read as
c = k/Q, so every share comparison is exact integer arithmetic.

  T1  characterisation of ACWCM (quota form, theta <= mu)
      (a) Definition 1 (tau_w and m_> with theta) equals the quota operator
          relax(r_min, [r_min in {R3,R4} and share(D) >= mu]) on every
          abstention-free tier profile of 6 gates and every confidence vector
          of the grid {0, 1/5, ..., 1}^6 (quantiser-exact), for each
          calibration pair with theta <= mu; for theta > mu we count the
          profiles on which the output depends on how far dissenters lie
          (axiom O fails there, so those operators are outside the theorem).
      (b) the axioms hold for the quota operator (band, tier monotonicity,
          ordinal dissent, weighted anonymity, floor neutrality) on that grid;
      (c) the released implementation (ConservativeMerging) agrees with
          Definition 1 on all 5^6 profiles for 12 confidence vectors;
      (d) converse on reduced instances: for n = 3 and n = 4 gates and the
          confidence grid {0, 1/2, 1}^n, every operator relax(r_min, g(share))
          (g one map on the attainable shares, g(0) = 0) that is tier-monotone
          is a closed or strict quota or the minimum, and every quota is
          monotone.
  T2  smooth + CND => rank f(r) <= max(2, rank r_min): least upper bound over
      all smooth CND operators on 6 gates, computed as a shortest-path fixed
      point on the 5^6 profiles; plus brute force over every CP operator on
      2 gates.
  L   every discrete Sugeno integral is smooth (all capacities, 3 gates;
      all symmetric capacities, 6 gates).
  P   a Sugeno integral satisfies CND iff nu({i}) >= 4 for every gate, and
      then it returns r_min whenever r_min != R1 (same instances).

    python scripts/verify_theorems.py
Writes results/finite_chain_theorems.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N = 6
Q = 5  # confidence grid step 1/Q (the paper's quantiser q(c) = ceil(5c))

PROFILES = np.array(list(product(range(1, 6), repeat=N)), dtype=np.int64)  # 15625 x 6
_CONF = np.array(list(product(range(Q + 1), repeat=N)), dtype=np.int64)    # 46656 x 6
CONF_CHUNKS = [_CONF[i:i + 32] for i in range(0, len(_CONF), 32)]


def acwcm_def(tiers, conf, theta_num, theta_den, mu_num, mu_den):
    """Definition 1 on abstention-free profiles, exact integer arithmetic.

    tiers: (P, n) ranks 1..5; conf: (K, n) integers 0..Q. theta = theta_num/theta_den,
    mu = mu_num/mu_den. Returns (K, P) output ranks.
    """
    conf = np.atleast_2d(conf)
    rmin = tiers.min(axis=1)
    C = conf.sum(axis=1)[:, None]
    above = (tiers > rmin[:, None]).astype(np.int64)
    A = conf @ above.T                       # confidence above the floor
    S = conf @ (tiers - rmin[:, None]).T     # C * (tau_w - rank r_min)
    relax = (S * theta_den >= theta_num * C) & (A * mu_den >= mu_num * C) & (C > 0)
    relax &= ((rmin != 1) & (rmin != 2) & (rmin != 5))[None, :]
    return rmin[None, :] + relax


def quota(tiers, conf, mu_num, mu_den, strict=False):
    """relax(r_min, [r_min in {3,4} and share >= mu]) (or > mu); (K, P)."""
    conf = np.atleast_2d(conf)
    rmin = tiers.min(axis=1)
    C = conf.sum(axis=1)[:, None]
    A = conf @ (tiers > rmin[:, None]).astype(np.int64).T
    ok = (A * mu_den > mu_num * C) if strict else (A * mu_den >= mu_num * C)
    ok &= (C > 0) & ((rmin == 3) | (rmin == 4))[None, :]
    return rmin[None, :] + ok


def index_of(tiers):
    """Row index of each profile in PROFILES (base-5 code)."""
    idx = np.zeros(len(tiers), dtype=np.int64)
    for j in range(tiers.shape[1]):
        idx = idx * 5 + (tiers[:, j] - 1)
    return idx


def t1_grid():
    pairs = [(1, 2, 1, 2), (1, 2, 2, 3), (1, 1, 1, 2), (1, 1, 2, 3)]  # (theta, mu) calibration grid
    rmin = PROFILES.min(axis=1)
    # neighbours for monotonicity: raise gate j by one tier
    nb = []
    for j in range(N):
        up = PROFILES.copy()
        ok = up[:, j] < 5
        up[ok, j] += 1
        nb.append((ok, index_of(up)))
    # projections for axiom O: dissenters to floor+1 and to R5
    diss = PROFILES > rmin[:, None]
    p1 = np.where(diss, np.minimum(rmin[:, None] + 1, 5), PROFILES)
    p5 = np.where(diss, 5, PROFILES)
    i1, i5 = index_of(p1), index_of(p5)
    res = {}
    for (tn, td, mn, md) in pairs:
        key = f"theta={tn}/{td},mu={mn}/{md}"
        r = {"inputs": 0, "def_ne_quota": 0, "band_viol": 0, "mono_viol": 0, "O_viol": 0}
        for conf in CONF_CHUNKS:
            f = acwcm_def(PROFILES, conf, tn, td, mn, md)
            g = quota(PROFILES, conf, mn, md)
            r["inputs"] += f.size
            r["def_ne_quota"] += int((f != g).sum())
            band = (f >= rmin) & (f <= np.minimum(rmin + 1, 5))
            band &= ~(np.isin(rmin, [1, 2, 5]) & (f != rmin))
            r["band_viol"] += int((~band).sum())
            for ok, ix in nb:
                r["mono_viol"] += int((ok & (f > f[:, ix])).sum())
            r["O_viol"] += int(((f != f[:, i1]) | (f != f[:, i5])).sum())
        res[key] = r
        print(key, r, flush=True)
    # weighted anonymity + floor neutrality for the quota operator are
    # immediate from its formula; confirm that over the grid the quota output
    # is a function of (floor class, share) only, with one g for floors 3 and 4.
    from math import gcd
    table = {}
    clash = 0
    sel = (rmin == 3) | (rmin == 4)
    abv = (PROFILES[sel] > rmin[sel][:, None]).astype(np.int64)
    for conf in CONF_CHUNKS:
        f = quota(PROFILES, conf, 2, 3)[:, sel] - rmin[sel][None, :]
        A = conf @ abv.T
        C = conf.sum(axis=1)
        for k in range(len(conf)):
            if C[k] == 0:
                continue
            pairs = np.unique(np.stack([A[k], f[k]], axis=1), axis=0)
            if len(np.unique(pairs[:, 0])) != len(pairs):
                clash += 1
            for a_, v in pairs.tolist():
                gg = gcd(a_, int(C[k]))
                key2 = (a_ // gg, int(C[k]) // gg)
                if table.setdefault(key2, v) != v:
                    clash += 1
    res["quota_mu=2/3_function_of_share_only"] = {"distinct_shares": len(table), "clashes": clash}
    print("W/N", res["quota_mu=2/3_function_of_share_only"], flush=True)
    return res


def t1_implementation():
    from merging.conservative_merging import ConservativeMerging
    from merging.risk_lattice import RiskTier
    tiers_of = {t.value: t for t in RiskTier}
    gates = [f"G{i}" for i in range(1, 7)]
    rng = np.random.default_rng(20261012)
    confs = [np.full(N, 4), np.full(N, 5), np.zeros(N, dtype=np.int64),
             np.array([5, 0, 0, 0, 0, 0]), np.array([1, 2, 3, 4, 5, 0])]
    confs += [rng.integers(0, Q + 1, N) for _ in range(7)]
    cm = ConservativeMerging(mode="acwcm", trigger="consensus", theta=0.5, min_mass=2 / 3)
    mism = 0
    n = 0
    for conf in confs:
        f = acwcm_def(PROFILES, conf, 1, 2, 2, 3)[0]
        for p, fo in zip(PROFILES, f):
            out, _, _ = cm.merge({g: tiers_of[int(t)] for g, t in zip(gates, p)},
                                 {g: float(c) / Q for g, c in zip(gates, conf)})
            n += 1
            mism += int(out.value != fo)
    r = {"inputs": n, "confidence_vectors": len(confs), "mismatches": mism}
    print("impl", r, flush=True)
    return r


def t1_converse(n, simplex_m=None):
    """All g: shares -> {0,1}, g(0)=0, one g for floors 3 and 4; keep the
    tier-monotone ones and check they are exactly the quotas."""
    from fractions import Fraction
    profs = np.array(list(product(range(1, 6), repeat=n)), dtype=np.int64)
    rmin = profs.min(axis=1)
    if simplex_m is None:   # coarse grid {0,1/2,1}^n (too coarse for the proof's construction)
        confs = [np.array(c, dtype=np.int64) for c in product(range(3), repeat=n) if sum(c) > 0]
    else:                   # normalised confidences on the simplex grid with step 1/m
        confs = [np.array(c, dtype=np.int64) for c in product(range(simplex_m + 1), repeat=n)
                 if sum(c) == simplex_m]
    shares = set()
    data = []
    for conf in confs:
        C = int(conf.sum())
        A = ((profs > rmin[:, None]) * conf).sum(axis=1)
        sh = [Fraction(int(a), C) for a in A]
        shares.update(sh)
        data.append(sh)
    W = sorted(shares)
    assert W[0] == 0
    pos = {s: i for i, s in enumerate(W)}
    sidx = [np.array([pos[s] for s in sh]) for sh in data]
    nbs = []
    for j in range(n):
        up = profs.copy()
        ok = up[:, j] < 5
        up[ok, j] += 1
        code = np.zeros(len(up), dtype=np.int64)
        for k in range(n):
            code = code * 5 + (up[:, k] - 1)
        nbs.append((ok, code))
    floor34 = (rmin == 3) | (rmin == 4)
    monotone = []
    for bits in range(2 ** (len(W) - 1)):
        g = np.array([0] + [(bits >> k) & 1 for k in range(len(W) - 1)], dtype=np.int64)
        good = True
        for si in sidx:
            f = rmin + (floor34 & (g[si] == 1))
            for ok, code in nbs:
                if np.any(ok & (f > f[code])):
                    good = False
                    break
            if not good:
                break
        if good:
            monotone.append(g)
    thresholds = {tuple(int(s >= W[k]) for s in W) for k in range(1, len(W))} | {tuple([0] * len(W))}
    got = {tuple(g.tolist()) for g in monotone}
    grid = "{0,1/2,1}^n" if simplex_m is None else f"simplex, step 1/{simplex_m}"
    r = {"gates": n, "confidence_grid": grid, "confidence_vectors": len(confs), "attainable_shares": len(W),
         "candidate_maps": 2 ** (len(W) - 1), "monotone_maps": len(monotone),
         "quota_maps": len(thresholds), "monotone_equals_quota": got == thresholds}
    print("converse", r, flush=True)
    return r


def t2():
    """Least upper bound U(r) on rank f(r) over all smooth operators with CND
    (output in {R*,R1,R2} = ranks <= 2 when r_min <= 2). Smoothness: adjacent
    profiles have outputs at most one rank apart. U is the largest function
    with U <= 5, U <= 2 on critical floors and U(r) <= U(r') + 1 for adjacent
    r, r' (a shortest-path fixed point)."""
    rmin = PROFILES.min(axis=1)
    U = np.where(rmin <= 2, 2, 5)
    adj = []
    for j in range(N):
        for d in (-1, 1):
            q = PROFILES.copy()
            ok = (q[:, j] + d >= 1) & (q[:, j] + d <= 5)
            q[ok, j] += d
            adj.append((ok, index_of(q)))
    it = 0
    while True:
        it += 1
        new = U.copy()
        for ok, ix in adj:
            new = np.where(ok, np.minimum(new, U[ix] + 1), new)
        if np.array_equal(new, U):
            break
        U = new
    bound = np.maximum(2, rmin)
    # the bound is attained by f = max(R2, r_min) in rank terms, which is smooth and CND
    f = bound
    smooth = all(int(np.abs(f[ok] - f[ix][ok]).max()) <= 1 for ok, ix in adj)
    r = {"profiles": len(PROFILES), "iterations": it,
         "U_equals_max(2,rank_rmin)": bool(np.array_equal(U, bound)),
         "bound_attained_by_smooth_CND_operator": smooth}
    # brute force, 2 gates: every operator with CP (r_min <= f <= r_min+1, f <= 5)
    p2 = list(product(range(1, 6), repeat=2))
    # CP and CND fix the output on floors R2 and R5; floors R1, R3, R4 have two options
    choices = [[m] if m in (2, 5) else [m, m + 1] for m in (min(p) for p in p2)]
    free = [i for i, c in enumerate(choices) if len(c) == 2]
    surv = 0
    surv_relax = 0
    for bits in range(2 ** len(free)):
        f2 = {p: min(p) for p in p2}
        for k, i in enumerate(free):
            if (bits >> k) & 1:
                f2[p2[i]] = min(p2[i]) + 1
        if any(min(p) <= 2 and f2[p] > 2 for p in p2):
            continue
        if any(abs(f2[p] - f2[(p[0] + 1, p[1])]) > 1 for p in p2 if p[0] < 5):
            continue
        if any(abs(f2[p] - f2[(p[0], p[1] + 1)]) > 1 for p in p2 if p[1] < 5):
            continue
        surv += 1
        if any(f2[p] != min(p) for p in p2 if min(p) >= 2):
            surv_relax += 1
    r["two_gates"] = {"CP_CND_operators": 2 ** len(free), "smooth_and_CND": surv,
                      "of_which_relax_a_floor_R2_to_R5": surv_relax}
    print("T2", r, flush=True)
    return r


def sugeno(u, nu_of_mask, n):
    """Discrete Sugeno integral on L={0..5}; u (P,n) in 0..5; nu_of_mask array 2^n."""
    best = np.zeros(len(u), dtype=np.int64)
    w = 1 << np.arange(n)
    for lvl in range(1, 6):
        mask = ((u >= lvl) * w).sum(axis=1)
        best = np.maximum(best, np.minimum(lvl, nu_of_mask[mask]))
    return best


def capacities(n):
    """All capacities 2^G -> {0..5}, nu(empty)=0, nu(G)=5, monotone."""
    masks = sorted(range(1 << n), key=lambda m: bin(m).count("1"))
    full = (1 << n) - 1
    inner = [m for m in masks if m not in (0, full)]
    nu = np.zeros(1 << n, dtype=np.int64)
    nu[full] = 5

    def rec(k):
        if k == len(inner):
            yield nu.copy()
            return
        m = inner[k]
        lo = max([nu[s] for s in range(1 << n) if s != m and (s & m) == s and s in inner[:k]] + [0])
        for v in range(lo, 6):
            nu[m] = v
            yield from rec(k + 1)
    yield from rec(0)


def sugeno_checks():
    out = {}
    # all capacities, 3 gates; all symmetric capacities, 6 gates
    for label, n, gen in [("all_capacities_3_gates", 3, None), ("symmetric_capacities_6_gates", 6, "sym")]:
        profs = np.array(list(product(range(1, 6), repeat=n)), dtype=np.int64)
        u = 6 - profs  # urgency 5..1
        rmin = profs.min(axis=1)
        adj = []
        for j in range(n):
            q = u.copy()
            ok = q[:, j] < 5
            q[ok, j] += 1
            adj.append((ok, q))
        if gen is None:
            caps = list(capacities(n))
        else:
            caps = []
            for seq in combinations(range(n + 5 - 1), n - 1):  # non-decreasing nu_1..nu_{n-1} in 0..5
                vals = [s - i for i, s in enumerate(seq)]
                if all(0 <= v <= 5 for v in vals):
                    sizes = [0] + vals + [5]
                    caps.append(np.array([sizes[bin(m).count("1")] for m in range(1 << n)], dtype=np.int64))
        cnt = {"capacities": len(caps), "profiles": len(profs), "smooth_viol": 0,
               "CND_iff_singletons_ge4_viol": 0, "CND_capacities": 0, "degeneracy_viol": 0,
               "equal_to_minimum": 0, "equal_to_minimum_iff_singletons_5_viol": 0}
        sing = [1 << i for i in range(n)]
        for nu in caps:
            s = sugeno(u, nu, n)
            for ok, q in adj:
                s2 = sugeno(q, nu, n)
                if np.any(ok & (np.abs(s2 - s) > 1)):
                    cnt["smooth_viol"] += 1
                    break
            tier = np.where(s == 0, 5, 6 - s)
            cnd = bool(np.all(~(rmin <= 2) | (tier <= 2)))
            pred = min(nu[m] for m in sing) >= 4
            cnt["CND_iff_singletons_ge4_viol"] += int(cnd != pred)
            if cnd:
                cnt["CND_capacities"] += 1
                bad = ((rmin >= 2) & (tier != rmin)) | ((rmin == 1) & (tier > 2))
                cnt["degeneracy_viol"] += int(np.any(bad))
                eq_min = bool(np.all(tier == rmin))
                cnt["equal_to_minimum"] += int(eq_min)
                cnt["equal_to_minimum_iff_singletons_5_viol"] += int(eq_min != (min(nu[m] for m in sing) == 5))
        out[label] = cnt
        print(label, cnt, flush=True)
    return out


def main():
    res = {"T1_grid_6_gates_conf_step_1/5": t1_grid(),
           "T1_implementation": t1_implementation(),
           "T1_converse": [t1_converse(3, 6), t1_converse(4, 6), t1_converse(3)],
           "T2_smooth_CND": t2(),
           "Sugeno": sugeno_checks()}
    path = ROOT / "results" / "finite_chain_theorems.json"
    path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("written", path)
    return res


if __name__ == "__main__":
    main()
