"""Tests for the round-4 changes: computed audit flags, certificate without
fabricated fallback, tie tolerance, new baselines and belief variants."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "data", "generation"))

from baselines import belief_variants as bv  # noqa: E402
from baselines import ordinal_extra as ox  # noqa: E402
from baselines.bayesian_model_avg import BayesianModelAveraging  # noqa: E402
from baselines.dempster_shafer import DempsterShaferCombination, argmax_tol  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from merging.risk_lattice import RiskTier  # noqa: E402
from merging.safety_certificate import SafetyCertificateGenerator  # noqa: E402

G = ["G1", "G2", "G3", "G4", "G5", "G6"]


def test_audit_flags_are_computed():
    m = ConservativeMerging("acwcm", "consensus", 0.5, 2 / 3)
    out = dict(zip(G, [RiskTier.R2, RiskTier.R4, RiskTier.R4, RiskTier.R4, RiskTier.R4, RiskTier.R4]))
    _, _, audit = m.merge(out, dict.fromkeys(G, 0.9))
    assert audit["safety_properties"] == {"CP": True, "AC": True, "CND": True}
    # A violating pair is reported as False, not asserted True.
    flags = ConservativeMerging.check_properties(out, RiskTier.R5)
    assert flags["CND"] is False and flags["CP"] is False
    star = dict(out, G3=RiskTier.R_STAR)
    assert ConservativeMerging.check_properties(star, RiskTier.R2)["AC"] is False


def test_certificate_has_no_fabricated_fallback():
    gen = SafetyCertificateGenerator()
    out = dict.fromkeys(G, RiskTier.R4)
    cert = gen.generate(patient_data={"age": 50}, r_final=RiskTier.R4, gate_outputs=out,
                        gate_confidences=dict.fromkeys(G, 0.8),
                        gate_reasonings=dict.fromkeys(G, ""), classify_fn=None, feature_names=None)
    assert cert.delta_cf is None


def test_exact_ties_resolved_with_tolerance():
    assert argmax_tol({"R1": 0.3, "R4": 0.3 * (1 + 1e-12)}, ["R1", "R4"]) == "R1"
    assert argmax_tol({"R1": 0.3, "R4": 0.3 * (1 + 1e-12)}, ["R1", "R4"], "benign") == "R4"
    # Three critical vs three benign gates at equal confidence: an exact tie.
    out = {g: ("R1" if i < 3 else "R5") for i, g in enumerate(G)}
    c = dict.fromkeys(G, 0.7)
    assert DempsterShaferCombination().classify(out, c)["final_tier"] == "R1"
    assert BayesianModelAveraging().classify(out, c)["final_tier"] in ("R1", "R5")


def test_dempster_variants_agree_on_argmax():
    out = {g: ("R1" if i < 2 else "R4") for i, g in enumerate(G)}
    c = dict.fromkeys(G, 0.6)
    ds = DempsterShaferCombination().classify(out, c)["final_tier"]
    for rule in ("dempster", "yager"):
        assert bv.BeliefRule(rule).classify(out, c)["final_tier"] == ds


def test_upper_set_betp_is_degenerate():
    out = dict.fromkeys(G, "R5")
    assert bv.BeliefRule("dempster", coding="upper").classify(out, dict.fromkeys(G, 0.9))["final_tier"] == "R1"


def test_new_ordinal_rules_are_idempotent():
    for t in ("R1", "R2", "R3", "R4", "R5"):
        out, c = dict.fromkeys(G, t), dict.fromkeys(G, 0.8)
        for r in (ox.WeightedMaxNormalised(), ox.QuantifierOWA(), ox.BandClampedSugeno(),
                  ox.KthOrderStatistic(3)):
            assert r.classify(out, c)["final_tier"] == t, (r.name, t)


def test_band_clamp_within_one_tier_of_floor():
    out = {g: ("R2" if i == 0 else "R5") for i, g in enumerate(G)}
    r = ox.BandClampedSugeno().classify(out, dict.fromkeys(G, 0.9))["final_tier"]
    assert r in ("R2", "R3")


def test_fuzzy_quantifier_trigger_records_degree():
    m = ConservativeMerging("acwcm", "fuzzy_quantifier", 0.5, 2 / 3)
    out = dict(zip(G, [RiskTier.R3] + [RiskTier.R5] * 5))
    final, _, audit = m.merge(out, dict.fromkeys(G, 0.8))
    assert 0.0 <= audit["quantifier_degree"] <= 1.0 and final == RiskTier.R4


def test_acwcm_one_step_jump_is_two_at_r2_hold():
    m = ConservativeMerging("acwcm", "consensus", 0.5, 2 / 3)
    c = dict.fromkeys(G, 0.9)
    a = dict(zip(G, [RiskTier.R2] + [RiskTier.R5] * 5))
    b = dict(zip(G, [RiskTier.R3] + [RiskTier.R5] * 5))
    assert m.merge(a, c)[0] == RiskTier.R2 and m.merge(b, c)[0] == RiskTier.R4
