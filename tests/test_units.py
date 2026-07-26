"""
Focused unit tests for SAFE-Gate pure / deterministic logic.

Targets the most testable, network-free, training-free functions:
  - RiskLattice ordering + minimum (conservative merging core)
  - Gate1 critical-flag rule firing (rule-based, deterministic)
  - Gate3 data-quality completeness thresholds (Equation 4)
  - Conservative minimum-merge + conservative-property guarantee

All inputs are tiny hand-made dicts; asserts are deterministic value/bound
checks. No randomness, no IO, no model training.
"""

import os
import sys

# Repo layout exposes packages under src/ (pytest.ini sets pythonpath=src too,
# but we add it explicitly so the file is runnable standalone).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from merging.risk_lattice import RiskTier, RiskLattice  # noqa: E402
from merging.conservative_merging import ConservativeMerging  # noqa: E402
from gates.gate1_critical_flags import Gate1CriticalFlags  # noqa: E402
from gates.gate3_data_quality import Gate3DataQuality  # noqa: E402


# --------------------------------------------------------------------------- #
# RiskLattice                                                                  #
# --------------------------------------------------------------------------- #
def test_risk_tier_ordering_and_str():
    """R* is most conservative (value 0); ordering is strict by value."""
    assert RiskTier.R_STAR.value == 0
    assert RiskTier.R1.value < RiskTier.R5.value
    assert RiskTier.R_STAR.is_more_conservative_than(RiskTier.R5)
    assert not RiskTier.R5.is_more_conservative_than(RiskTier.R1)
    assert str(RiskTier.R_STAR) == "R*"
    assert str(RiskTier.R3) == "R3"
    assert RiskTier.R_STAR.is_abstention()
    assert RiskTier.R1.requires_immediate_intervention()


def test_lattice_minimum_picks_most_conservative():
    """minimum() returns the lowest-value tier; R* always dominates."""
    assert RiskLattice.minimum([RiskTier.R5, RiskTier.R3, RiskTier.R4]) == RiskTier.R3
    # R_STAR present -> always R_STAR regardless of others
    assert (
        RiskLattice.minimum([RiskTier.R1, RiskTier.R_STAR, RiskTier.R2])
        == RiskTier.R_STAR
    )
    # empty list is an error
    try:
        RiskLattice.minimum([])
        assert False, "empty list should raise ValueError"
    except ValueError:
        pass


def test_lattice_from_string_roundtrip_and_invalid():
    """from_string parses valid tiers (case-insensitive) and rejects junk."""
    assert RiskLattice.from_string("R*") == RiskTier.R_STAR
    assert RiskLattice.from_string("r3") == RiskTier.R3
    try:
        RiskLattice.from_string("R9")
        assert False, "invalid tier should raise ValueError"
    except ValueError:
        pass


def test_conservative_property_check():
    """final tier must be <= every gate tier to satisfy Theorem 2."""
    gates = [RiskTier.R5, RiskTier.R3, RiskTier.R4]
    assert RiskLattice.verify_conservative_property(RiskTier.R3, gates) is True
    # R4 is NOT <= R3, so property must fail
    assert RiskLattice.verify_conservative_property(RiskTier.R4, gates) is False


# --------------------------------------------------------------------------- #
# Gate 1 — critical red flags (rule-based, deterministic)                      #
# --------------------------------------------------------------------------- #
def test_gate1_fires_r1_on_single_critical_flag():
    g1 = Gate1CriticalFlags()
    # Hypotension SBP < 90 -> R1 at full confidence
    tier, conf, reason = g1.evaluate({"systolic_bp": 80})
    assert tier == RiskTier.R1
    assert conf == 1.0
    assert len(reason["triggers"]) >= 1


def test_gate1_all_normal_is_r5():
    g1 = Gate1CriticalFlags()
    tier, conf, reason = g1.evaluate(
        {
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "heart_rate": 75,
            "spo2": 98,
            "gcs": 15,
        }
    )
    assert tier == RiskTier.R5
    assert conf == 1.0
    assert reason["triggers"] == []


def test_gate1_central_hints_counts_as_focal_deficit():
    g1 = Gate1CriticalFlags()
    tier, _, reason = g1.evaluate({"hints_nystagmus": "central"})
    assert tier == RiskTier.R1
    assert any("central" in t for t in reason["triggers"])


# --------------------------------------------------------------------------- #
# Gate 3 — data-quality completeness (Equation 4 thresholds)                   #
# --------------------------------------------------------------------------- #
def test_gate3_full_data_r5_conf_one():
    g3 = Gate3DataQuality()
    full = {f: 1 for f in g3.essential_fields}
    tier, conf, reason = g3.evaluate(full)
    assert tier == RiskTier.R5
    assert abs(conf - 1.0) < 1e-9
    assert reason["completeness"] == 1.0


def test_gate3_threshold_bands():
    """rho<0.70 -> R*, 0.70<=rho<0.85 -> R4, rho>=0.85 -> R5; conf == rho."""
    g3 = Gate3DataQuality()
    fields = g3.essential_fields
    n = len(fields)  # 22

    # ~63.6% complete (14/22) -> abstain R*
    sparse = {f: 1 for f in fields[:14]}
    tier, conf, _ = g3.evaluate(sparse)
    assert tier == RiskTier.R_STAR
    assert abs(conf - 14 / n) < 1e-9

    # ~72.7% complete (16/22) -> escalate R4
    partial = {f: 1 for f in fields[:16]}
    tier, conf, _ = g3.evaluate(partial)
    assert tier == RiskTier.R4
    assert abs(conf - 16 / n) < 1e-9


# --------------------------------------------------------------------------- #
# Conservative merging                                                         #
# --------------------------------------------------------------------------- #
def test_min_merge_abstention_and_minimum():
    merger = ConservativeMerging(mode="min")

    # R* present -> propagates to final
    outs = {"G1": RiskTier.R5, "G3": RiskTier.R_STAR, "G4": RiskTier.R4}
    confs = {"G1": 1.0, "G3": 0.5, "G4": 0.9}
    tier, enforcing, _ = merger.merge(outs, confs)
    assert tier == RiskTier.R_STAR
    assert enforcing == "G3"

    # No R*: minimum-tier wins and final satisfies conservative property
    outs = {"G1": RiskTier.R5, "G2": RiskTier.R2, "G3": RiskTier.R4}
    confs = {"G1": 0.9, "G2": 0.8, "G3": 0.7}
    tier, _, _ = merger.merge(outs, confs)
    assert tier == RiskTier.R2
    assert RiskLattice.verify_conservative_property(tier, list(outs.values()))


def test_relax_is_bounded_at_r5():
    """_relax never exceeds R5 and moves k positions less conservative."""
    assert ConservativeMerging._relax(RiskTier.R2, 1) == RiskTier.R3
    assert ConservativeMerging._relax(RiskTier.R5, 3) == RiskTier.R5
    assert ConservativeMerging._relax(RiskTier.R4, 5) == RiskTier.R5
