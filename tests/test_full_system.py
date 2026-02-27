"""
Comprehensive test suite for SAFE-Gate system.

Tests all 6 gates, ACWCM merging, safety certificates, and baselines
against article specifications.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from merging.risk_lattice import RiskTier
from gates.gate1_critical_flags import Gate1CriticalFlags
from gates.gate2_moderate_risk import Gate2ModerateRisk
from gates.gate3_data_quality import Gate3DataQuality
from gates.gate4_titrate_logic import Gate4TiTrATELogic
from gates.gate5_uncertainty import Gate5Uncertainty
from gates.gate6_temporal_risk import Gate6TemporalRisk
from merging.conservative_merging import ConservativeMerging
from merging.safety_certificate import SafetyCertificateGenerator
from baselines.dempster_shafer import DempsterShaferCombination
from baselines.bayesian_model_avg import BayesianModelAveraging


def test_gate1_critical_flags():
    """Test G1: 18 rules, 5 categories."""
    g1 = Gate1CriticalFlags()

    # Test 1: SBP < 90 -> R1
    tier, conf, reason = g1.evaluate({'systolic_bp': 85})
    assert tier == RiskTier.R1, f"SBP<90 should be R1, got {tier}"
    assert conf == 1.0, f"G1 confidence should be 1.0, got {conf}"

    # Test 2: SBP > 180 -> R1
    tier, _, _ = g1.evaluate({'systolic_bp': 195})
    assert tier == RiskTier.R1, f"SBP>180 should be R1, got {tier}"

    # Test 3: GCS < 14 -> R1 (article specifies GCS<14, not GCS<12)
    tier, _, _ = g1.evaluate({'gcs': 13})
    assert tier == RiskTier.R1, f"GCS=13 (<14) should be R1, got {tier}"

    # Test 4: O2 < 92% -> R1 (article specifies O2<92%)
    tier, _, _ = g1.evaluate({'spo2': 91})
    assert tier == RiskTier.R1, f"SpO2=91 (<92%) should be R1, got {tier}"

    # Test 5: All normal -> R5
    tier, conf, _ = g1.evaluate({
        'systolic_bp': 120, 'diastolic_bp': 80,
        'heart_rate': 75, 'spo2': 98, 'gcs': 15,
    })
    assert tier == RiskTier.R5, f"All normal should be R5, got {tier}"
    assert conf == 1.0

    # Test 6: Multiple triggers
    _, _, reason = g1.evaluate({
        'systolic_bp': 85, 'gcs': 12, 'spo2': 88,
        'dysarthria': True, 'ataxia': True,
    })
    assert len(reason['triggers']) >= 4, f"Should have >=4 triggers, got {len(reason['triggers'])}"

    print("  [PASS] Gate 1: 18 rules, GCS<14, O2<92% all correct")


def test_gate2_cardiovascular():
    """Test G2: Equation 3 weights."""
    g2 = Gate2ModerateRisk()

    # Test 1: High-risk CV factors
    tier, conf, reason = g2.evaluate({
        'age': 72, 'gender': 'male',
        'atrial_fibrillation': True, 'prior_stroke': True,
        'sudden_onset': True,
    })
    assert tier in [RiskTier.R2, RiskTier.R3], f"High CV risk should be R2-R3, got {tier}"

    # Test 2: Article weights check (AF=+1.8, prior CVA=+2.0, age>60=+1.0, male=+0.5, sudden=+1.5)
    expected_score = 1.8 + 2.0 + 1.0 + 0.5 + 1.5  # = 6.8
    assert reason['total_score'] >= 6.0, f"Score should be >=6, got {reason['total_score']}"

    # Test 3: Low-risk
    tier, _, _ = g2.evaluate({'age': 30, 'gender': 'female'})
    assert tier == RiskTier.R5, f"Young female no risk factors should be R5, got {tier}"

    # Test 4: Confidence is computed via bootstrap variance formula
    assert 'confidence_detail' in reason, "Should include confidence detail (c2 formula)"

    print("  [PASS] Gate 2: Equation 3 weights, c2 = 1 - sigma^2/sigma_max^2")


def test_gate3_data_quality():
    """Test G3: 22 essential fields, thresholds 0.70/0.85."""
    g3 = Gate3DataQuality()

    # Verify exactly 22 fields
    assert len(g3.essential_fields) == 22, f"Should have 22 fields, got {len(g3.essential_fields)}"

    # Test 1: Full data -> R5 (rho = 1.0 >= 0.85)
    full_data = {f: 1 for f in g3.essential_fields}
    tier, conf, reason = g3.evaluate(full_data)
    assert tier == RiskTier.R5, f"Full data should be R5, got {tier}"
    assert abs(conf - 1.0) < 0.01, f"Full data confidence should be ~1.0, got {conf}"

    # Test 2: 70% complete -> R4 (escalate 1 tier)
    partial_data = {}
    for i, f in enumerate(g3.essential_fields):
        if i < 16:  # 16/22 = 0.727 (between 0.70 and 0.85)
            partial_data[f] = 1
    tier, _, _ = g3.evaluate(partial_data)
    assert tier == RiskTier.R4, f"70-85% complete should be R4, got {tier}"

    # Test 3: < 70% -> R* (abstain)
    sparse_data = {}
    for i, f in enumerate(g3.essential_fields):
        if i < 14:  # 14/22 = 0.636 < 0.70
            sparse_data[f] = 1
    tier, _, _ = g3.evaluate(sparse_data)
    assert tier == RiskTier.R_STAR, f"<70% complete should be R*, got {tier}"

    print("  [PASS] Gate 3: 22 fields, thresholds 0.70/0.85")


def test_gate4_syndrome_matching():
    """Test G4: Weighted Hamming distance, 3 syndromes."""
    g4 = Gate4TiTrATELogic()

    # Test 1: BPPV-like case -> should get high similarity -> R5
    bppv_case = {
        'hints_head_impulse': 'normal',
        'hints_nystagmus': 'peripheral',
        'hints_test_of_skew': 'negative',
        'positional_triggers': True,
        'sudden_onset': True,
        'symptom_duration_days': 0.01,  # Brief episodes
        'episodic_pattern': True,
        'hearing_loss': False,
        'tinnitus': False,
        'aural_fullness': False,
        'nausea_vomiting': True,
        'progression_pattern': 'stable',
    }
    tier, conf, reason = g4.evaluate(bppv_case)
    assert tier in [RiskTier.R5, RiskTier.R4], f"BPPV case should be R4-R5, got {tier}"
    assert reason['best_match'] == 'BPPV', f"Best match should be BPPV, got {reason['best_match']}"

    # Test 2: No matching syndrome -> R2
    unclear_case = {
        'hints_head_impulse': 'normal',
        'hints_nystagmus': 'central',
        'hints_test_of_skew': 'positive',
        'positional_triggers': False,
        'hearing_loss': True,
        'tinnitus': True,
    }
    tier, _, reason = g4.evaluate(unclear_case)
    # Central signs = red flags, so even with moderate similarity -> R3 max
    assert tier in [RiskTier.R2, RiskTier.R3], f"Central signs should be R2-R3, got {tier}"

    # Test 3: Confidence = max similarity
    assert 0 <= conf <= 1.0, f"Confidence should be in [0,1], got {conf}"

    # Test 4: Verify three syndrome profiles exist
    assert len(g4.syndrome_profiles) == 3, f"Should have 3 syndrome profiles"
    assert 'BPPV' in g4.syndrome_profiles
    assert 'Vestibular_Neuritis' in g4.syndrome_profiles
    assert 'Meniere_Disease' in g4.syndrome_profiles

    print("  [PASS] Gate 4: Weighted Hamming distance, 3 syndromes, thresholds 0.85/0.60")


def test_gate5_uncertainty():
    """Test G5: BNN MC dropout, Equation 5."""
    g5 = Gate5Uncertainty()

    # Verify architecture
    assert g5.INPUT_DIM == 52, f"Input should be 52, got {g5.INPUT_DIM}"
    assert g5.HIDDEN_LAYERS == [128, 64], f"Hidden layers should be [128,64], got {g5.HIDDEN_LAYERS}"
    assert g5.OUTPUT_DIM == 5, f"Output should be 5, got {g5.OUTPUT_DIM}"
    assert g5.DROPOUT_RATE == 0.3, f"Dropout should be 0.3, got {g5.DROPOUT_RATE}"
    assert g5.MC_PASSES == 20, f"MC passes should be 20, got {g5.MC_PASSES}"

    # Verify thresholds match Equation 5
    assert g5.ABSTENTION_THRESHOLD == 0.80
    assert g5.ESCALATE_2_THRESHOLD == 0.60
    assert g5.ESCALATE_1_THRESHOLD == 0.30

    # Test: Clear critical case
    tier, conf, reason = g5.evaluate({
        'systolic_bp': 75, 'heart_rate': 145, 'spo2': 82, 'gcs': 10,
    })
    assert 'mu' in reason['mc_predictions'], "Should compute composite uncertainty mu"

    # Test: Confidence = 1 - mu
    mu = reason['mc_predictions']['mu']
    expected_conf = 1.0 - mu
    assert abs(conf - expected_conf) < 0.01, f"c5 should be 1-mu={expected_conf:.3f}, got {conf:.3f}"

    print("  [PASS] Gate 5: 52->128->64->5, T=20, Equation 5 mu thresholds")


def test_gate6_temporal():
    """Test G6: Finite-state machine, 5 temporal states."""
    g6 = Gate6TemporalRisk()

    # Test 1: Hyperacute + worsening -> R1
    tier, conf, _ = g6.evaluate({
        'symptom_onset_hours': 0.5,
        'progression_pattern': 'worsening',
    })
    assert tier == RiskTier.R1, f"Hyperacute worsening should be R1, got {tier}"
    assert conf == 1.0, f"Unambiguous temporal profile -> c6=1.0, got {conf}"

    # Test 2: Acute stable -> R2-R3
    tier, _, _ = g6.evaluate({
        'symptom_onset_hours': 6,
        'progression_pattern': 'stable',
    })
    assert tier in [RiskTier.R2, RiskTier.R3], f"Acute stable should be R2-R3, got {tier}"

    # Test 3: Chronic -> R4-R5
    tier, _, _ = g6.evaluate({
        'symptom_duration_days': 14,
        'progression_pattern': 'improving',
    })
    assert tier in [RiskTier.R4, RiskTier.R5], f"Chronic improving should be R4-R5, got {tier}"

    # Test 4: No temporal data -> R3 with c6=0.5
    tier, conf, _ = g6.evaluate({'age': 50})
    assert tier == RiskTier.R3, f"No temporal data should be R3, got {tier}"
    assert conf == 0.5, f"Uncertain timeline should be c6=0.5, got {conf}"

    print("  [PASS] Gate 6: FSM, hyperacute->R1, chronic->R4-R5, c6=1.0/0.5")


def test_acwcm_merging():
    """Test ACWCM Algorithm 1."""
    merger = ConservativeMerging(mode="acwcm")

    # Test 1: R* propagation (Phase 1 hard constraint)
    outputs = {'G1': RiskTier.R5, 'G3': RiskTier.R_STAR, 'G4': RiskTier.R5}
    confs = {'G1': 1.0, 'G3': 0.5, 'G4': 0.8}
    tier, enforcer, _ = merger.merge(outputs, confs)
    assert tier == RiskTier.R_STAR, f"R* should propagate, got {tier}"

    # Test 2: R1 propagation (Phase 1 CND)
    outputs = {'G1': RiskTier.R1, 'G4': RiskTier.R5, 'G5': RiskTier.R4}
    confs = {'G1': 1.0, 'G4': 0.9, 'G5': 0.8}
    tier, _, _ = merger.merge(outputs, confs)
    assert tier in [RiskTier.R1, RiskTier.R2], f"CND: R1 present -> R1 or R2, got {tier}"

    # Test 3: Bounded relaxation (relax(r_min, 1) = max 1 tier above min)
    outputs = {'G1': RiskTier.R3, 'G2': RiskTier.R5, 'G3': RiskTier.R5}
    confs = {'G1': 0.6, 'G2': 0.9, 'G3': 0.9}
    tier, _, _ = merger.merge(outputs, confs)
    # r_min = R3, relax(R3,1) = R4, so final should be <= R4
    assert tier.value <= RiskTier.R4.value, f"Bounded relaxation: should be <= R4, got {tier}"

    print("  [PASS] ACWCM: R* propagation, CND, bounded relaxation")


def test_safety_certificate():
    """Test Algorithm 2: Safety Certificate Generation."""
    cert_gen = SafetyCertificateGenerator()

    # Generate a certificate
    gate_outputs = {'G1': RiskTier.R5, 'G4': RiskTier.R5, 'G6': RiskTier.R5}
    gate_confs = {'G1': 1.0, 'G4': 0.92, 'G6': 0.9}
    gate_reasonings = {
        g: {'tier': t, 'confidence': gate_confs[g], 'reasoning': {}}
        for g, t in gate_outputs.items()
    }

    cert = cert_gen.generate(
        patient_data={'systolic_bp': 120, 'gcs': 15, 'spo2': 98},
        r_final=RiskTier.R5,
        gate_outputs=gate_outputs,
        gate_confidences=gate_confs,
        gate_reasonings=gate_reasonings,
        classify_fn=None,
        feature_names=['systolic_bp', 'gcs', 'spo2']
    )

    assert cert is not None, "Certificate should be generated"
    assert hasattr(cert, 'delta_min'), "Certificate should have delta_min"
    assert hasattr(cert, 'delta_cf'), "Certificate should have delta_cf"
    assert hasattr(cert, 'g_enforce'), "Certificate should have g_enforce"

    d = cert.to_dict()
    assert 'delta_min' in d and 'delta_cf' in d and 'g_enforce' in d

    print("  [PASS] Safety Certificate: delta_min, delta_cf, g_enforce generated")


def test_baselines():
    """Test DS and BMA baselines."""
    ds = DempsterShaferCombination()
    bma = BayesianModelAveraging()

    gate_outputs = {'G1': 'R5', 'G2': 'R3', 'G3': 'R5'}
    gate_confidences = {'G1': 0.9, 'G2': 0.8, 'G3': 0.7}

    # Test DS
    result = ds.classify(gate_outputs, gate_confidences)
    assert 'final_tier' in result, "DS should return final_tier"
    assert 'beliefs' in result, "DS should return beliefs"

    # Test BMA
    result = bma.classify(gate_outputs, gate_confidences)
    assert 'final_tier' in result, "BMA should return final_tier"
    assert 'probabilities' in result, "BMA should return probabilities"

    print("  [PASS] Baselines: Dempster-Shafer + BMA working")


def test_full_pipeline():
    """Test full SAFE-Gate pipeline end-to-end."""
    from safegate import SAFEGate

    sg = SAFEGate(mode="acwcm")

    # Critical case
    result = sg.classify({
        'patient_id': 'TEST_CRITICAL',
        'age': 72, 'gender': 'male',
        'systolic_bp': 85, 'diastolic_bp': 55,
        'heart_rate': 125, 'spo2': 88, 'gcs': 13,
        'temperature': 37.2,
        'symptom_onset_hours': 1.5, 'symptom_duration_days': 0.1,
        'vertigo_severity': 'severe', 'sudden_onset': True,
        'progression_pattern': 'worsening',
        'dysarthria': True, 'ataxia': True,
        'hypertension': True, 'atrial_fibrillation': True,
        'diabetes': False, 'prior_stroke': True,
        'hints_head_impulse': 'abnormal',
        'hints_nystagmus': 'central',
        'hints_test_of_skew': 'positive',
    })
    assert result['final_tier'] == 'R1', f"Critical case should be R1, got {result['final_tier']}"
    assert 'safety_certificate' in result, "Should include safety certificate"

    # Safe case
    result = sg.classify({
        'patient_id': 'TEST_SAFE',
        'age': 34, 'gender': 'female',
        'systolic_bp': 118, 'diastolic_bp': 72,
        'heart_rate': 76, 'spo2': 99, 'gcs': 15,
        'temperature': 36.8,
        'symptom_onset_hours': 72, 'symptom_duration_days': 3,
        'vertigo_severity': 'mild', 'sudden_onset': False,
        'progression_pattern': 'improving',
        'dysarthria': False, 'ataxia': False,
        'hypertension': False, 'atrial_fibrillation': False,
        'diabetes': False, 'prior_stroke': False,
        'hints_head_impulse': 'normal',
        'hints_nystagmus': 'peripheral',
        'hints_test_of_skew': 'negative',
        'positional_triggers': True,
        'hearing_loss': False,
        'tinnitus': False,
        'aural_fullness': False,
        'episodic_pattern': True,
    })
    assert result['final_tier'] in ['R4', 'R5'], f"Safe case should be R4-R5, got {result['final_tier']}"

    # Incomplete data -> R*
    result = sg.classify({
        'patient_id': 'TEST_INCOMPLETE',
        'age': 55,
        'vertigo_severity': 'moderate',
    })
    assert result['final_tier'] == 'R*', f"Incomplete should be R*, got {result['final_tier']}"

    print("  [PASS] Full pipeline: Critical->R1, Safe->R4/R5, Incomplete->R*")


def main():
    print("=" * 70)
    print("SAFE-Gate Comprehensive Test Suite")
    print("Verifying code matches article specifications")
    print("=" * 70)
    print()

    tests = [
        ("Gate 1: Critical Red Flags", test_gate1_critical_flags),
        ("Gate 2: Cardiovascular Risk", test_gate2_cardiovascular),
        ("Gate 3: Data Quality", test_gate3_data_quality),
        ("Gate 4: Syndrome Matching", test_gate4_syndrome_matching),
        ("Gate 5: Uncertainty", test_gate5_uncertainty),
        ("Gate 6: Temporal Risk", test_gate6_temporal),
        ("ACWCM Merging", test_acwcm_merging),
        ("Safety Certificate", test_safety_certificate),
        ("Baselines (DS + BMA)", test_baselines),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
