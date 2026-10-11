#!/usr/bin/env python3
"""
Round-4 evaluation (IJFS revision).

Two stages, run in this order and committed in between:

  python scripts/round4.py freeze
      Generates the never-scored evaluation cohorts from seeds fixed in this
      file, writes them to data/fresh/ and records seeds, sizes and SHA-256
      digests in results/round4_protocol.json. Scores nothing.

  python scripts/round4.py evaluate
      Regenerates the cohorts, checks every digest against the frozen
      protocol (aborts on mismatch), fits the trainable baselines on the
      DEVELOPMENT training split (seed 42) and scores every method once on
      each fresh cohort. Also runs the development-set analyses (multi-seed,
      128-profile selection analysis, conflict characterisation, extended
      enumeration, matched MCAR deferral, feature noise, statistics).
      Writes results/round4_*.json|csv.

`python run_all.py --round4` runs `evaluate` after the main pipeline.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generation"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_data import FEATURE_SCHEMA, SAFEGateDataGenerator  # noqa: E402
from generate_variant import VariantGenerator  # noqa: E402
from generate_diagnosis import DANGEROUS, DiagnosisGenerator  # noqa: E402
from safegate import SAFEGate  # noqa: E402
from baselines import sugeno as sugeno_mod  # noqa: E402
from baselines.sugeno import SugenoAdditive, WeightedMax  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination  # noqa: E402
from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402
from baselines.ensemble_average import EnsembleAverage  # noqa: E402
from baselines import ordinal_extra as ox  # noqa: E402
from baselines import belief_variants as bv  # noqa: E402
from calibration import CANDIDATES, V1, V2  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402

TIERS = ["R1", "R2", "R3", "R4", "R5"]
IDX = {t: i for i, t in enumerate(TIERS)}
RANK = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
CRIT = {"R1", "R2"}
RES = ROOT / "results"
FRESH = ROOT / "data" / "fresh"
PROTOCOL = RES / "round4_protocol.json"

# Fresh evaluation cohorts (seeds fixed before any scoring).
FRESH_COHORTS = {
    "fresh_main_generator": ("main", 20261012, 6400),
    "fresh_independent_generator": ("variant", 20261013, 3000),
    "fresh_diagnosis_generator": ("diagnosis", 20261014, 5000),
}
# Validation cohorts for the robustness-aware selection (never used as test).
SELECTION_COHORTS = {"val_independent": ("variant", 101, 2000),
                     "val_diagnosis": ("diagnosis", 102, 2000)}
MULTISEED = list(range(1001, 1021))  # 20 generator seeds, each with a fresh split


def make(kind, seed, n):
    if kind == "main":
        return SAFEGateDataGenerator(n_total=n, random_state=seed).generate_all_cases()
    if kind == "variant":
        return VariantGenerator(n_total=n, random_state=seed).generate()
    return DiagnosisGenerator(n_total=n, random_state=seed).generate()


def digest(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
def freeze():
    FRESH.mkdir(parents=True, exist_ok=True)
    proto = {"purpose": "round-4 frozen evaluation protocol: cohorts generated and hashed "
                        "before any method is scored on them",
             "primary_endpoints": ["critical (R1/R2) sensitivity with exact 95% CI",
                                   "false discharges (critical -> R5)",
                                   "R3-conditional under-triage", "accuracy",
                                   "over- and under-triage", ">=2-tier under-triage"],
             "methods_fitted_on": "development training split (generator seed 42)",
             "calibration": V2, "cohorts": {}}
    for name, (kind, seed, n) in {**FRESH_COHORTS, **SELECTION_COHORTS}.items():
        cases = make(kind, seed, n)
        path = FRESH / f"{name}.json"
        path.write_text(json.dumps(cases, sort_keys=True), encoding="utf-8")
        proto["cohorts"][name] = {"generator": kind, "seed": seed, "n": n,
                                  "sha256": digest(cases), "file": f"data/fresh/{name}.json",
                                  "role": "evaluation" if name in FRESH_COHORTS else "selection"}
        print(name, proto["cohorts"][name]["sha256"])
    proto["multiseed_seeds"] = MULTISEED
    PROTOCOL.write_text(json.dumps(proto, indent=2), encoding="utf-8")


def load_frozen():
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    out = {}
    for name, meta in proto["cohorts"].items():
        cases = make(meta["generator"], meta["seed"], meta["n"])
        assert digest(cases) == meta["sha256"], f"cohort {name} differs from frozen protocol"
        out[name] = cases
    return proto, out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cp(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def metrics(cases, preds):
    n = len(cases)
    crit = [i for i, c in enumerate(cases) if c["ground_truth_tier"] in CRIT]
    kept = sum(preds[i] in ("R*", "R1", "R2") for i in crit)
    lo, hi = cp(kept, len(crit))
    dec = [i for i in range(n) if preds[i] in IDX]
    nd = max(1, len(dec))
    d = {i: IDX[preds[i]] - IDX[cases[i]["ground_truth_tier"]] for i in dec}
    r3 = [i for i in dec if cases[i]["ground_truth_tier"] == "R3"]
    r3u = sum(d[i] > 0 for i in r3)
    r3lo, r3hi = cp(r3u, len(r3))
    row = {
        "n": n, "n_critical": len(crit),
        "critical_sensitivity_pct": round(100 * kept / len(crit), 2) if crit else None,
        "critical_ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "missed_critical": len(crit) - kept,
        "false_discharges": sum(preds[i] == "R5" for i in crit),
        "abstention_pct": round(100 * (n - len(dec)) / n, 2),
        "accuracy_pct": round(100 * sum(d[i] == 0 for i in dec) / nd, 2),
        "over_triage_pct": round(100 * sum(d[i] < 0 for i in dec) / nd, 2),
        "under_triage_pct": round(100 * sum(d[i] > 0 for i in dec) / nd, 2),
        "under_2plus": sum(d[i] >= 2 for i in dec),
        "r3_under_triage_pct": round(100 * r3u / len(r3), 2) if r3 else None,
        "r3_under_ci95": [round(100 * r3lo, 2), round(100 * r3hi, 2)],
    }
    for t in TIERS:
        idx = [i for i in dec if cases[i]["ground_truth_tier"] == t]
        if idx:
            row[f"{t}_under_pct"] = round(100 * sum(d[i] > 0 for i in idx) / len(idx), 2)
            row[f"{t}_over_pct"] = round(100 * sum(d[i] < 0 for i in idx) / len(idx), 2)
    return row


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def gate_io(cases, sg=None):
    sg = sg or SAFEGate()
    out = []
    for c in cases:
        r = sg.classify(c, return_audit_trail=False, return_certificate=False)
        out.append((r["gate_outputs"], r["gate_confidences"], r["final_tier"]))
    return out


def clean(o):
    return {g: (t if t in IDX else "R1") for g, t in o.items()}


def encode(rows):
    X = []
    for r in rows:
        vec = []
        for f in FEATURE_SCHEMA:
            v = r.get(f)
            if v is None:
                vec.append(np.nan)
            elif isinstance(v, bool):
                vec.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vec.append(float(v))
            else:
                vec.append(float(int(hashlib.sha256(str(v).encode()).hexdigest()[:6], 16) % 997))
        X.append(vec)
    return np.asarray(X, dtype=float)


class Fitted:
    """Trainable baselines, fitted once on the development training split."""

    def __init__(self, train, val):
        io_tr = gate_io(train)
        ytr = [c["ground_truth_tier"] for c in train]
        pairs = [(o, c) for o, c, _ in io_tr]
        self.kth = ox.KthOrderStatistic.fit([(clean(o), c) for o, c in pairs], ytr)
        self.sym = ox.SymmetricSugeno.fit([(clean(o), c) for o, c in pairs], ytr)
        self.tsk = ox.TSKFuzzy.fit([(clean(o), c) for o, c in pairs], ytr)
        # Per-gate reliability for discounting: exact-tier agreement on train,
        # counted over the cases where the gate is not vacuous (confidence > 0).
        rel = {}
        for g in pairs[0][0]:
            idx = [i for i, (o, c) in enumerate(pairs) if c[g] > 0]
            rel[g] = (sum(pairs[i][0][g] == ytr[i] for i in idx) / len(idx)) if idx else 1.0
        self.reliability = {g: round(v, 4) for g, v in rel.items()}
        from xgboost import XGBClassifier
        y = np.array([IDX[t] for t in ytr])
        self.xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                                 tree_method="hist", verbosity=0, num_class=5,
                                 objective="multi:softprob")
        self.xgb.fit(encode(train), y)
        pv = self.xgb.predict_proba(encode(val))
        crit_v = np.array([c["ground_truth_tier"] in CRIT for c in val])
        pc = pv[:, 0] + pv[:, 1]
        # Largest threshold that keeps >= 99% validation critical sensitivity.
        ts = sorted(set(np.round(pc[crit_v], 6)))
        self.xgb_t = max([t for t in ts if (pc[crit_v] >= t).mean() >= 0.99] or [0.0])
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Xtr = np.nan_to_num(encode(train))
        keep = Xtr.std(axis=0) > 0
        self.ol_keep, self.ol_mu, self.ol_sd = keep, Xtr[:, keep].mean(0), Xtr[:, keep].std(0)
        Z = (Xtr[:, keep] - self.ol_mu) / self.ol_sd
        try:
            self.ol = OrderedModel(y, Z, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
        except Exception as exc:  # recorded, not hidden
            self.ol, self.ol_error = None, repr(exc)

    def xgb_pred(self, cases):
        p = self.xgb.predict_proba(encode(cases))
        return [TIERS[int(np.argmax(r))] for r in p]

    def xgb_safe(self, cases):
        p = self.xgb.predict_proba(encode(cases))
        out = []
        for r in p:
            if r[0] + r[1] >= self.xgb_t:
                out.append("R1" if r[0] >= r[1] else "R2")
            else:
                out.append(TIERS[2 + int(np.argmax(r[2:]))])
        return out

    def ordlogit(self, cases):
        if self.ol is None:
            return None
        Z = (np.nan_to_num(encode(cases))[:, self.ol_keep] - self.ol_mu) / self.ol_sd
        return [TIERS[int(np.argmax(r))] for r in np.asarray(self.ol.predict(Z))]


def run_methods(cases, fit, io=None, extended=True):
    io = io or gate_io(cases)
    v1 = SAFEGate(calibration="v1")
    mn = SAFEGate(mode="min")
    fz = SAFEGate(trigger="fuzzy_quantifier")
    ds, bma = DempsterShaferCombination(), BayesianModelAveraging()
    sa, wm = SugenoAdditive(), WeightedMax()
    P = {k: [] for k in ["SAFE-Gate ACWCM (v2)", "SAFE-Gate ACWCM (v1)", "SAFE-Gate MIN",
                         "ACWCM fuzzy-quantifier trigger", "Dempster-Shafer", "Bayesian model averaging",
                         "BMA min expected cost", "Sugeno (quantised additive capacity)",
                         "Weighted max (forced top, original)", ox.WeightedMaxNormalised.name,
                         f"{ox.KthOrderStatistic.name}, k={fit.kth.k}", ox.QuantifierOWA.name,
                         ox.BandClampedSugeno.name, ox.SymmetricSugeno.name,
                         ox.PrioritizedYager.name, ox.TSKFuzzy.name]}
    variants = bv.variants(fit.reliability) if extended else []
    for v in variants:
        P[v.name] = []
    fixed = [ox.WeightedMaxNormalised(), ox.QuantifierOWA(), ox.BandClampedSugeno(),
             ox.PrioritizedYager()]
    for c, (o, cf, fin) in zip(cases, io):
        cl = clean(o)
        P["SAFE-Gate ACWCM (v2)"].append(fin)
        P["SAFE-Gate ACWCM (v1)"].append(v1.classify(c, False, False)["final_tier"])
        P["SAFE-Gate MIN"].append(mn.classify(c, False, False)["final_tier"])
        P["ACWCM fuzzy-quantifier trigger"].append(fz.classify(c, False, False)["final_tier"])
        P["Dempster-Shafer"].append(ds.classify(cl, cf)["final_tier"])
        P["Bayesian model averaging"].append(bma.classify(cl, cf)["final_tier"])
        P["BMA min expected cost"].append(bv.bma_min_cost(bma, cl, cf))
        P["Sugeno (quantised additive capacity)"].append(sa.classify(o, cf)["final_tier"])
        P["Weighted max (forced top, original)"].append(wm.classify(o, cf)["final_tier"])
        for r in fixed:
            P[r.name].append(r.classify(cl, cf)["final_tier"])
        P[f"{ox.KthOrderStatistic.name}, k={fit.kth.k}"].append(fit.kth.classify(cl)["final_tier"])
        P[ox.SymmetricSugeno.name].append(fit.sym.classify(cl)["final_tier"])
        P[ox.TSKFuzzy.name].append(fit.tsk.classify(cl)["final_tier"])
        for v in variants:
            P[v.name].append(v.classify(o, cf)["final_tier"])
    P["XGBoost"] = fit.xgb_pred(cases)
    P["XGBoost (safety threshold, val >=99%)"] = fit.xgb_safe(cases)
    ol = fit.ordlogit(cases)
    if ol is not None:
        P["Ordinal logistic regression"] = ol
    return P, io


# --------------------------------------------------------------------------- #
# Development-set analyses
# --------------------------------------------------------------------------- #
def gate_recall_union(cases, io):
    crit = [i for i, c in enumerate(cases) if c["ground_truth_tier"] in CRIT]
    any_fire = sum(any(t in ("R*", "R1", "R2") for t in io[i][0].values()) for i in crit)
    per_gate = {g: round(100 * sum(io[i][0][g] in ("R*", "R1", "R2") for i in crit) / len(crit), 2)
                for g in io[0][0]} if crit else {}
    n_fire = Counter(sum(t in ("R*", "R1", "R2") for t in io[i][0].values()) for i in crit)
    return {"n_critical": len(crit),
            "some_gate_fires_pct": round(100 * any_fire / len(crit), 2) if crit else None,
            "per_gate_fire_pct": per_gate,
            "number_of_gates_firing_histogram": {str(k): v for k, v in sorted(n_fire.items())}}


def r1_only_via_g6(cases, io):
    r1 = [i for i, c in enumerate(cases) if io[i][2] == "R1"]
    only = sum(1 for i in r1 if io[i][0].get("G6") == "R1"
               and all(t != "R1" for g, t in io[i][0].items() if g != "G6"))
    return {"predicted_R1": len(r1), "R1_only_via_G6": only}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def newcombe_paired(a, b, z=1.959964):
    """Newcombe (1998) method 10 CI for the difference of paired proportions."""
    n = len(a)
    a_, b_ = sum(a), sum(b)
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n11 = sum(x and y for x, y in zip(a, b))
    n00 = n - n10 - n01 - n11

    def wilson(k, m):
        p = k / m
        d = 1 + z * z / m
        c = (p + z * z / (2 * m)) / d
        h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
        return c - h, c + h
    p1, p2 = a_ / n, b_ / n
    l1, u1 = wilson(a_, n)
    l2, u2 = wilson(b_, n)
    num = (n11 * n00 - n10 * n01)
    den = math.sqrt(max((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 0))
    phi = 0.0 if den == 0 else num / den
    if num > 0:
        phi = max(0.0, (num - n / 2) / den) if den else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    hi = d + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)


def paired_stats(cases, P, ref="SAFE-Gate ACWCM (v2)"):
    from scipy.stats import binomtest
    rows = []
    a = P[ref]
    crit = [i for i, c in enumerate(cases) if c["ground_truth_tier"] in CRIT]

    def ind(p, kind):
        out = []
        for c, x in zip(cases, p):
            t, q = IDX[c["ground_truth_tier"]], RANK[x] - 1
            out.append({"correct": q == t, "under": q > t, "under2": q - t >= 2,
                        "over": 0 <= q < t}[kind])
        return out
    for name in ("Dempster-Shafer", "SAFE-Gate MIN", "Sugeno (quantised additive capacity)",
                 ox.BandClampedSugeno.name, "SAFE-Gate ACWCM (v1)"):
        for kind in ("correct", "over", "under", "under2"):
            x, y = ind(a, kind), ind(P[name], kind)
            n10 = sum(p and not q for p, q in zip(x, y))
            n01 = sum(q and not p for p, q in zip(x, y))
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            d, lo, hi = newcombe_paired(x, y)
            rows.append({"comparison": f"ACWCM vs {name}", "outcome": kind, "acwcm_only": n10,
                         "other_only": n01, "p_exact": float(f"{p:.4g}"),
                         "diff_pct": d, "diff_ci95_newcombe": [lo, hi]})
    adj = holm([r["p_exact"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(f"{q:.4g}")
    # Non-inferiority of critical sensitivity vs DS (paired, among critical cases).
    ka = [a[i] in ("R*", "R1", "R2") for i in crit]
    kd = [P["Dempster-Shafer"][i] in ("R*", "R1", "R2") for i in crit]
    ni = newcombe_paired(ka, kd)
    return rows, {"critical_sensitivity_diff_vs_DS_pct": ni[0], "ci95": [ni[1], ni[2]]}


def age_tier_interaction(cases, preds):
    import statsmodels.api as sm
    y = np.array([RANK[p] - 1 > IDX[c["ground_truth_tier"]] for c, p in zip(cases, preds)], float)
    old = np.array([c["age"] > 60 for c in cases], float)
    T = np.array([[c["ground_truth_tier"] == t for t in TIERS[1:]] for c in cases], float)
    keep = [j for j in range(T.shape[1]) if 0 < y[T[:, j] == 1].sum() < (T[:, j] == 1).sum()]
    base = np.column_stack([np.ones(len(y)), old, T])
    inter = np.column_stack([base] + [old * T[:, j] for j in keep])
    try:
        m0 = sm.Logit(y, base).fit(disp=False, method="bfgs", maxiter=1000)
        m1 = sm.Logit(y, inter).fit(disp=False, method="bfgs", maxiter=1000)
        lr = 2 * (m1.llf - m0.llf)
        from scipy.stats import chi2
        return {"lr_stat": round(float(lr), 3), "df": len(keep),
                "p": float(f"{chi2.sf(lr, len(keep)):.4g}"), "label": "exploratory"}
    except Exception as exc:
        return {"error": repr(exc)}


def matched_mcar(test, fit, rng_seed=7):
    """MCAR with the same G3 deferral for every method (deferred = R*)."""
    rng = np.random.RandomState(rng_seed)
    out = []
    for p in (0.10, 0.20, 0.30):
        cases = [{k: v for k, v in c.items() if k not in FEATURE_SCHEMA or rng.random() >= p}
                 for c in test]
        P, io = run_methods(cases, fit, extended=False)
        deferred = [io[i][0].get("G3") == "R*" for i in range(len(cases))]
        crit = [i for i, c in enumerate(cases) if c["ground_truth_tier"] in CRIT]
        for name, preds in P.items():
            mp = ["R*" if deferred[i] else preds[i] for i in range(len(cases))]
            dec_crit = [i for i in crit if mp[i] != "R*"]
            out.append({
                "mcar_pct": int(p * 100), "method": name,
                "coverage_pct": round(100 * sum(x != "R*" for x in mp) / len(mp), 2),
                "deferred_critical": len(crit) - len(dec_crit),
                "sensitivity_among_decided_pct": round(100 * sum(mp[i] in CRIT for i in dec_crit)
                                                       / len(dec_crit), 2) if dec_crit else None,
                "false_discharges": sum(mp[i] == "R5" for i in crit),
                "own_abstention_pct": round(100 * sum(x == "R*" for x in preds) / len(preds), 2),
            })
    return out


def feature_noise(test, fit, seed=11):
    """Measurement noise on continuous features, scored against the true labels."""
    rng = np.random.RandomState(seed)
    rows = []
    for level, (sd_bp, sd_hr, sd_o2, onset_sd) in {"moderate": (8, 6, 1.0, 0.3),
                                                  "high": (15, 12, 2.0, 0.6)}.items():
        cases = []
        for c in test:
            c2 = dict(c)
            c2["systolic_bp"] = int(round(c["systolic_bp"] + rng.normal(0, sd_bp)))
            c2["heart_rate"] = int(round(c["heart_rate"] + rng.normal(0, sd_hr)))
            c2["spo2"] = int(min(100, round(c["spo2"] + rng.normal(0, sd_o2))))
            c2["symptom_onset_hours"] = round(c["symptom_onset_hours"] * math.exp(rng.normal(0, onset_sd)), 2)
            c2["symptom_duration_days"] = round(c2["symptom_onset_hours"] / 24, 3)
            cases.append(c2)
        P, _ = run_methods(cases, fit, extended=False)
        for name, p in P.items():
            rows.append({"scenario": f"feature noise {level}", "method": name, **metrics(cases, p)})
    return rows


def multiseed(fit_kind="v2"):
    import select_calibration as sc
    rows = []
    for s in MULTISEED:
        gen = SAFEGateDataGenerator(n_total=6400, random_state=s)
        train, val, test = gen.split(gen.generate_all_cases())
        ttr, tva = sc.gate_table(train), sc.gate_table(val)
        table = []
        for cal in CANDIDATES:
            tr, va = sc.score(train, ttr, cal), sc.score(val, tva, cal)
            table.append((cal, tr, va, sc.feasible(tr) and sc.feasible(va)))
        ok = [r for r in table if r[3]] or table
        top = max(r[1]["accuracy_pct"] for r in ok)
        near = [r for r in ok if r[1]["accuracy_pct"] >= top - 0.1]
        best = min(near, key=lambda r: (-r[1]["macro_f1_pct"], sum(r[0][k] != V1[k] for k in V1)))
        sg = SAFEGate()
        preds = [sg.classify(c, False, False)["final_tier"] for c in test]
        m = metrics(test, preds)
        rows.append({"seed": s, "v2_selected": best[0] == V2, "n_feasible": sum(r[3] for r in table),
                     **{k: m[k] for k in ("critical_sensitivity_pct", "false_discharges", "accuracy_pct",
                                          "over_triage_pct", "under_triage_pct", "r3_under_triage_pct",
                                          "under_2plus")}})
        print("seed", s, rows[-1]["v2_selected"], m["accuracy_pct"], flush=True)
    keys = ["critical_sensitivity_pct", "false_discharges", "accuracy_pct", "over_triage_pct",
            "under_triage_pct", "r3_under_triage_pct", "under_2plus"]
    summ = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 2),
                "sd": round(float(np.std([r[k] for r in rows], ddof=1)), 2),
                "min": min(r[k] for r in rows), "max": max(r[k] for r in rows)} for k in keys}
    summ["v2_selected_count"] = sum(r["v2_selected"] for r in rows)
    summ["n_seeds"] = len(rows)
    summ["label"] = "within-generator sampling variability"
    return rows, summ


def profile_analysis(dev_train, dev_val, sel_cohorts, eval_cohorts):
    """All 128 profiles: train/val metrics, v2 rank, scores on independent and
    diagnosis-conditioned cohorts, and a robustness-aware selection."""
    import select_calibration as sc
    tabs = {"train": (dev_train, sc.gate_table(dev_train)), "val": (dev_val, sc.gate_table(dev_val))}
    for k, v in {**sel_cohorts, **eval_cohorts}.items():
        tabs[k] = (v, sc.gate_table(v))
    rows = []
    for cal in CANDIDATES:
        r = {"calibration": cal, "is_v2": cal == V2, "is_v1": cal == V1}
        for k, (cs, tb) in tabs.items():
            m = sc.score(cs, tb, cal)
            r[k] = {x: m[x] for x in ("critical_sensitivity_pct", "false_discharges", "accuracy_pct",
                                      "under_triage_pct", "over_triage_pct")}
        rows.append(r)
    by_train = sorted(rows, key=lambda r: -r["train"]["accuracy_pct"])
    v2_rank = next(i for i, r in enumerate(by_train) if r["is_v2"]) + 1
    # Robustness-aware criterion (selection cohorts only): maximise the worst
    # critical sensitivity over train/val/val_independent/val_diagnosis, then
    # train accuracy.
    def robust_key(r):
        return (min(r[k]["critical_sensitivity_pct"] for k in ("train", "val") + tuple(sel_cohorts)),
                r["train"]["accuracy_pct"])
    robust = max(rows, key=robust_key)
    return {"n_profiles": len(rows), "v2_rank_by_train_accuracy": v2_rank,
            "best_train_accuracy": by_train[0]["train"]["accuracy_pct"],
            "v2_train_accuracy": next(r for r in rows if r["is_v2"])["train"]["accuracy_pct"],
            "robust_selected": robust["calibration"], "robust_equals_v2": robust["is_v2"],
            "robust_row": robust, "v2_row": next(r for r in rows if r["is_v2"]),
            "rows": rows}


# --------------------------------------------------------------------------- #
# Aggregation-level analyses (no data)
# --------------------------------------------------------------------------- #
GATES = ["G1", "G2", "G3", "G4", "G5", "G6"]


def acwcm_fn(trigger="consensus"):
    m = ConservativeMerging("acwcm", trigger, V2["theta"], V2["min_mass"])

    def f(o, c):
        return str(m.merge({g: RiskTier["R_STAR" if t == "R*" else t] for g, t in o.items()}, c)[0])
    return f


def rule_set(fit):
    ds, bma, sa, wm = DempsterShaferCombination(), BayesianModelAveraging(), SugenoAdditive(), WeightedMax()
    R = {"SAFE-Gate ACWCM (v2)": acwcm_fn(), "ACWCM fuzzy-quantifier trigger": acwcm_fn("fuzzy_quantifier"),
         "Dempster-Shafer": lambda o, c: ds.classify(clean(o), c)["final_tier"],
         "Bayesian model averaging": lambda o, c: bma.classify(clean(o), c)["final_tier"],
         "BMA min expected cost": lambda o, c: bv.bma_min_cost(bma, clean(o), c),
         "Sugeno (quantised additive capacity)": lambda o, c: sa.classify(o, c)["final_tier"],
         "Weighted max (forced top, original)": lambda o, c: wm.classify(o, c)["final_tier"]}
    for r in ox.all_fixed():
        R[r.name] = (lambda r: lambda o, c: r.classify(clean(o), c)["final_tier"])(r)
    R[f"{ox.KthOrderStatistic.name}, k={fit.kth.k}"] = lambda o, c: fit.kth.classify(clean(o))["final_tier"]
    R[ox.SymmetricSugeno.name] = lambda o, c: fit.sym.classify(clean(o))["final_tier"]
    R[ox.TSKFuzzy.name] = lambda o, c: fit.tsk.classify(clean(o))["final_tier"]
    for v in bv.variants(fit.reliability):
        R[v.name] = (lambda v: lambda o, c: v.classify(o, c)["final_tier"])(v)
    return R


def conflict_grid(R):
    """980-config single-tier dissent grid (as scripts/conflicting_evidence.py),
    the heterogeneous-dissent grid and the false-alarm grid."""
    from conflicting_evidence import C_BENIGN, C_CRIT, N_DISSENT
    res = {k: {"dilutions_980": 0, "r1_to_r5_980": 0, "dilutions_hetero": 0, "escalations_false_alarm": 0}
           for k in R}
    n_het = n_fa = 0
    for ct, bt, nd, cc, cb in product(["R1", "R2"], ["R4", "R5"], N_DISSENT, C_CRIT, C_BENIGN):
        o = {g: (bt if 1 <= i <= nd else ct) for i, g in enumerate(GATES)}
        c = {g: (cb if 1 <= i <= nd else cc) for i, g in enumerate(GATES)}
        for k, f in R.items():
            r = f(o, c)
            res[k]["dilutions_980"] += r not in CRIT
            res[k]["r1_to_r5_980"] += ct == "R1" and r == "R5"
    # Heterogeneous dissent: G1 critical, n dissenters spread over R3..R5 with
    # confidences in {0.4, 0.9}; remaining gates agree with G1.
    for ct, cc in product(["R1", "R2"], [0.5, 0.7, 0.9]):
        for nd in range(1, 6):
            for tiers in product(["R3", "R4", "R5"], repeat=nd):
                for confs in product([0.4, 0.9], repeat=nd):
                    o, c = {"G1": ct}, {"G1": cc}
                    for i, g in enumerate(GATES[1:]):
                        o[g], c[g] = (tiers[i], confs[i]) if i < nd else (ct, cc)
                    n_het += 1
                    for k, f in R.items():
                        res[k]["dilutions_hetero"] += f(o, c) not in CRIT
    # False alarm: one gate falsely critical, five gates agree on the true benign tier.
    for ft, fc, bt, bc in product(["R1", "R2"], [0.5, 0.7, 0.9, 0.99], ["R4", "R5"], [0.3, 0.5, 0.7, 0.9]):
        o = {g: (ft if i == 0 else bt) for i, g in enumerate(GATES)}
        c = {g: (fc if i == 0 else bc) for i, g in enumerate(GATES)}
        n_fa += 1
        for k, f in R.items():
            res[k]["escalations_false_alarm"] += f(o, c) in CRIT
    for k in res:
        res[k]["dilution_rate_980"] = res[k]["dilutions_980"] / 980
        res[k]["escalation_rate_false_alarm"] = res[k]["escalations_false_alarm"] / n_fa
    return res, n_het, n_fa


def breakeven(res, prevalence, ref="SAFE-Gate ACWCM (v2)", ratio=1.0):
    """Cost ratio lambda* (under- vs over-triage) at which a rule's expected harm
    equals ACWCM's, under the single-gate error model: a critical case is seen by
    exactly one firing gate with probability q_d, a benign case by one false
    critical gate with probability q_f = ratio * q_d."""
    out = {}
    a = res[ref]
    for k, r in res.items():
        dd = r["dilution_rate_980"] - a["dilution_rate_980"]
        de = a["escalation_rate_false_alarm"] - r["escalation_rate_false_alarm"]
        out[k] = (round((1 - prevalence) / prevalence * ratio * de / dd, 3) if dd > 0 else None)
    return out


def extended_enumeration(R):
    ALL = ["R1", "R2", "R3", "R4", "R5"]
    profiles = {"homogeneous": dict(zip(GATES, [0.8] * 6)),
                "heterogeneous": dict(zip(GATES, [0.9, 0.5, 0.7, 0.6, 0.8, 0.4]))}
    rng = np.random.RandomState(5)
    grid = [round(0.2 * i, 1) for i in range(6)]
    sampled = [dict(zip(GATES, rng.choice(grid, 6))) for _ in range(6)]
    out = {}
    for k, f in R.items():
        v = {"max_jump_one_step": 0, "CP": 0, "CND": 0, "R1D": 0, "conf_monotonicity": 0,
             "n_inputs": 0, "n_conf_pairs": 0}
        for conf in list(profiles.values()) + sampled:
            cache = {}
            for tup in product(ALL, repeat=6):
                o = dict(zip(GATES, tup))
                r = RANK[f(o, conf)]
                cache[tup] = r
                v["n_inputs"] += 1
                rmin = min(RANK[t] for t in tup)
                v["CP"] += r > rmin + 1
                v["CND"] += rmin <= 2 and r > 2
                v["R1D"] += "R1" in tup and r == 5
            for tup, r in cache.items():
                for i, t in enumerate(tup):
                    if t != "R5":
                        nb = tup[:i] + (ALL[IDX[t] + 1],) + tup[i + 1:]
                        v["max_jump_one_step"] = max(v["max_jump_one_step"], abs(cache[nb] - r))
        # Confidence monotonicity (homogeneous base): raising the confidence of a
        # gate that is more urgent than the output must not make the output less
        # urgent.
        base = profiles["homogeneous"]
        for tup in product(ALL, repeat=6):
            o = dict(zip(GATES, tup))
            r = RANK[f(o, base)]
            for g in GATES:
                if RANK[o[g]] < r:
                    c2 = dict(base)
                    c2[g] = 1.0
                    v["n_conf_pairs"] += 1
                    v["conf_monotonicity"] += RANK[f(o, c2)] > r
        out[k] = v
        print("enum", k, v, flush=True)
    return {"confidence_profiles": {"fixed": profiles, "sampled_grid": sampled}, "rules": out}


def quantiser_sensitivity():
    orig = sugeno_mod.quantise
    modes = {"ceil (default)": orig,
             "round": lambda c: max(0, min(5, int(math.floor(5 * float(c) + 0.5)))),
             "floor": lambda c: max(0, min(5, int(math.floor(5 * float(c) + 1e-12))))}
    from conflicting_evidence import C_BENIGN, C_CRIT, N_DISSENT
    out = {}
    for name, q in modes.items():
        sugeno_mod.quantise = q
        ox.quantise = q
        rules = {"Sugeno additive": SugenoAdditive(), "Weighted max forced": WeightedMax(),
                 "Weighted max normalised": ox.WeightedMaxNormalised(),
                 "Band-clamped Sugeno": ox.BandClampedSugeno()}
        cnt = dict.fromkeys(rules, 0)
        for ct, bt, nd, cc, cb in product(["R1", "R2"], ["R4", "R5"], N_DISSENT, C_CRIT, C_BENIGN):
            o = {g: (bt if 1 <= i <= nd else ct) for i, g in enumerate(GATES)}
            c = {g: (cb if 1 <= i <= nd else cc) for i, g in enumerate(GATES)}
            for k, r in rules.items():
                cnt[k] += r.classify(o, c)["final_tier"] not in CRIT
        out[name] = cnt
    sugeno_mod.quantise = orig
    ox.quantise = orig
    return out


def certificate_distribution(test):
    sg = SAFEGate()
    vals = Counter()
    for c in test:
        r = sg.classify(c, return_audit_trail=False, return_certificate=True)
        cert = r.get("safety_certificate") or r.get("certificate") or {}
        v = cert.get("delta_cf") if isinstance(cert, dict) else getattr(cert, "delta_cf", None)
        vals["not found (censored)" if v is None else str(v)] += 1
    return dict(sorted(vals.items()))


# --------------------------------------------------------------------------- #
def write_csv(name, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (RES / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def evaluate(skip_multiseed=False):
    proto, cohorts = load_frozen()
    gen = SAFEGateDataGenerator(n_total=6400, random_state=42)
    train, val, test = gen.split(gen.generate_all_cases())
    fit = Fitted(train, val)
    out = {"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "fitted": {"kth_k": fit.kth.k, "symmetric_phi": list(fit.sym.phi),
                      "reliability": fit.reliability, "xgb_critical_threshold": round(float(fit.xgb_t), 6),
                      "ordinal_logistic": "fitted" if fit.ol is not None else getattr(fit, "ol_error", "")}}

    # PRIMARY: fresh, never-scored cohorts (scored exactly once here).
    prim = []
    comp = {}
    for name in FRESH_COHORTS:
        cs = cohorts[name]
        P, io = run_methods(cs, fit)
        for m, p in P.items():
            prim.append({"cohort": name, "method": m, **metrics(cs, p)})
        out.setdefault("gate_recall_union", {})[name] = gate_recall_union(cs, io)
        out.setdefault("r1_only_via_g6", {})[name] = r1_only_via_g6(cs, io)
        if name == "fresh_diagnosis_generator":
            acw = P["SAFE-Gate ACWCM (v2)"]
            under_r3 = Counter(c["diagnosis"] for c, p in zip(cs, acw)
                               if c["ground_truth_tier"] == "R3" and RANK[p] > 3)
            missed = Counter(c["diagnosis"] for c, p in zip(cs, acw)
                             if c["ground_truth_tier"] in CRIT and p not in ("R*", "R1", "R2"))
            comp = {"case_mix": dict(Counter((c["ground_truth_tier"], c["diagnosis"]) for c in cs).most_common()),
                    "acwcm_under_triaged_R3_by_diagnosis": dict(under_r3),
                    "acwcm_missed_critical_by_diagnosis": dict(missed),
                    "dangerous_diagnoses": sorted(DANGEROUS)}
            comp["case_mix"] = {f"{k[0]}|{k[1]}": v for k, v in comp["case_mix"].items()}
            out["paired_stats_diagnosis"], out["noninferiority_vs_DS_diagnosis"] = paired_stats(cs, P)
            out["ablation_diagnosis"] = ablation(cs)
        if name == "fresh_independent_generator":
            out["ablation_independent"] = ablation(cs)
    write_csv("round4_primary.csv", prim)
    out["diagnosis_composition"] = comp

    # DEVELOPMENT split (seed 42 test; scored in earlier rounds).
    P, io = run_methods(test, fit)
    dev = [{"cohort": "development_test_seed42", "method": m, **metrics(test, p)} for m, p in P.items()]
    write_csv("round4_development.csv", dev)
    out["paired_stats_dev"], out["noninferiority_vs_DS_dev"] = paired_stats(test, P)
    out["age_tier_interaction_dev"] = age_tier_interaction(test, P["SAFE-Gate ACWCM (v2)"])
    out["gate_recall_union"]["development_test"] = gate_recall_union(test, io)
    out["r1_only_via_g6"]["development_test"] = r1_only_via_g6(test, io)
    out["certificate_delta_cf_dev"] = certificate_distribution(test)
    write_csv("round4_mcar_matched.csv", matched_mcar(test, fit))
    write_csv("round4_feature_noise.csv", feature_noise(test, fit))

    print("conflict grids ...", flush=True)
    R = rule_set(fit)
    res, n_het, n_fa = conflict_grid(R)
    prev = sum(c["ground_truth_tier"] in CRIT for c in test) / len(test)
    out["conflict"] = {"n_hetero": n_het, "n_false_alarm": n_fa, "rules": res,
                       "breakeven_lambda_ratio1": breakeven(res, prev),
                       "prevalence_critical_dev": round(prev, 4)}
    out["quantiser_sensitivity"] = quantiser_sensitivity()
    print("enumeration ...", flush=True)
    out["enumeration"] = extended_enumeration(R)
    print("profiles ...", flush=True)
    pa = profile_analysis(train, val, {k: cohorts[k] for k in SELECTION_COHORTS},
                          {k: cohorts[k] for k in ("fresh_independent_generator", "fresh_diagnosis_generator")})
    write_csv("round4_profiles.csv", [{"calibration": r["calibration"], "is_v2": r["is_v2"],
                                       **{f"{k}_{x}": r[k][x] for k in r if isinstance(r[k], dict) and k != "calibration"
                                          for x in r[k]}} for r in pa["rows"]])
    out["profile_analysis"] = {k: v for k, v in pa.items() if k != "rows"}
    if not skip_multiseed:
        print("multi-seed ...", flush=True)
        rows, summ = multiseed()
        write_csv("round4_multiseed.csv", rows)
        out["multiseed"] = summ
    (RES / "round4_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("done")


def ablation(cases):
    rows = []
    for dis in [()] + [(g,) for g in GATES]:
        sg = SAFEGate()
        for g in dis:
            del sg.gates[g]
        p = [sg.classify(c, False, False)["final_tier"] for c in cases]
        m = metrics(cases, p)
        rows.append({"configuration": "full" if not dis else f"-{dis[0]}",
                     **{k: m[k] for k in ("critical_sensitivity_pct", "false_discharges", "accuracy_pct",
                                          "under_triage_pct", "over_triage_pct")}})
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["freeze"]:
        freeze()
    elif sys.argv[1:2] == ["evaluate"]:
        evaluate("--skip-multiseed" in sys.argv)
    else:
        print(__doc__)
