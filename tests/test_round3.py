"""
Tests for the v2 calibration, the Sugeno-family baselines and the invariants
under missing data (abstention correctness is exercised here).
"""

import json
import os
import sys
from itertools import product

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "data", "generation"))

from baselines.sugeno import SugenoAdditive, WeightedMax, quantise, sugeno_integral  # noqa: E402
from calibration import V1, V2, resolve  # noqa: E402
from gates.gate1_critical_flags import Gate1CriticalFlags  # noqa: E402
from gates.gate3_data_quality import Gate3DataQuality  # noqa: E402
from gates.gate4_titrate_logic import Gate4TiTrATELogic  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402
from safegate import SAFEGate  # noqa: E402

GATES = ["G1", "G2", "G3", "G4", "G5", "G6"]
URG = {"R1": 5, "R2": 4, "R3": 3, "R4": 2, "R5": 1}


def test_v2_matches_committed_protocol():
    proto = json.load(open(os.path.join(_ROOT, "results", "calibration_protocol.json")))
    sel = proto["selected"]
    for k, v in V2.items():
        assert abs(sel[k] - v) < 1e-12 if isinstance(v, float) else sel[k] == v
    assert "timestamp_utc" in proto
    assert resolve("v1") == V1 and resolve("v2") == V2


def test_g1_graded_separates_life_threat_from_other_red_flags():
    g = Gate1CriticalFlags(graded=True)
    assert g.evaluate({"systolic_bp": 82})[0] == RiskTier.R1
    assert g.evaluate({"spo2": 88})[0] == RiskTier.R1
    assert g.evaluate({"systolic_bp": 190})[0] == RiskTier.R2      # hypertensive crisis
    assert g.evaluate({"dysarthria": True})[0] == RiskTier.R2      # stroke symptom
    assert g.evaluate({"hints_test_of_skew": "positive"})[0] == RiskTier.R2
    # The v1 behaviour (every red flag -> R1) is preserved when not graded.
    assert Gate1CriticalFlags().evaluate({"dysarthria": True})[0] == RiskTier.R1


def test_vacuous_negatives_carry_zero_confidence():
    t, c, _ = Gate1CriticalFlags(vacuous_negative=True).evaluate({})
    assert t == RiskTier.R5 and c == 0.0
    full = {f: 1 for f in Gate3DataQuality().essential_fields}
    t, c, _ = Gate3DataQuality(vacuous_complete=True).evaluate(full)
    assert t == RiskTier.R5 and c == 0.0


def test_g4_central_only_keeps_r2_for_central_signs():
    g = Gate4TiTrATELogic(central_only_escalation=True)
    assert g.evaluate({"hints_nystagmus": "vertical"})[0] == RiskTier.R2
    assert g.evaluate({"dysarthria": True})[0] == RiskTier.R2


def test_sugeno_integral_is_idempotent_and_monotone():
    caps = [SugenoAdditive.capacity, WeightedMax.capacity]
    confs = dict(zip(GATES, [0.9, 0.5, 0.7, 0.6, 0.8, 0.4]))
    for capf in caps:
        mu = capf(confs, GATES)
        for t in URG:
            assert sugeno_integral({g: URG[t] for g in GATES}, mu) == URG[t]
        rng = np.random.RandomState(0)
        for _ in range(300):
            x = {g: int(rng.randint(1, 6)) for g in GATES}
            g = GATES[rng.randint(6)]
            y = dict(x)
            y[g] = min(5, y[g] + 1)
            assert sugeno_integral(y, mu) >= sugeno_integral(x, mu)


def test_weighted_max_with_full_confidence_equals_min_selection():
    confs = dict.fromkeys(GATES, 1.0)
    for tiers in product(["R1", "R3", "R5"], repeat=6):
        out = dict(zip(GATES, tiers))
        assert WeightedMax().classify(out, confs)["final_tier"] == min(tiers, key=lambda t: int(t[1]))


def test_sugeno_additive_can_dilute_a_lone_critical_signal():
    # One R1 gate against five R5 gates of equal confidence: the additive
    # Sugeno integral is a weighted median and discards the critical signal.
    out = {"G1": "R1", **{g: "R5" for g in GATES[1:]}}
    res = SugenoAdditive().classify(out, dict.fromkeys(GATES, 0.8))
    assert res["final_tier"] not in ("R1", "R2")
    assert quantise(0.0) == 0 and quantise(1.0) == 5


def test_invariants_hold_under_missing_data():
    """Abstention correctness was never exercised on complete synthetic records;
    drop features at random so that G3 abstains, and check every invariant."""
    from generate_data import SAFEGateDataGenerator
    gen = SAFEGateDataGenerator(n_total=400, random_state=3)
    cases = gen.generate_all_cases()
    rng = np.random.RandomState(1)
    sg = SAFEGate()
    rank = {"R*": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
    n_abstain = 0
    for c in cases:
        c2 = {k: v for k, v in c.items() if k in ("ground_truth_tier", "patient_id") or rng.random() > 0.3}
        res = sg.classify(c2, return_audit_trail=False, return_certificate=False)
        ranks = [rank[t] for t in res["gate_outputs"].values()]
        fin = rank[res["final_tier"]]
        if min(ranks) == 0:
            n_abstain += 1
            assert fin == 0                       # abstention correctness
        else:
            assert fin <= min(ranks) + 1          # conservative preservation
            if min(ranks) <= 2:
                assert fin <= 2                   # critical non-dilution
    assert n_abstain > 0
