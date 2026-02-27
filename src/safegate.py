"""
SAFE-Gate: Safety-Assured Fusion Engine with Gated Expert Triage

Main system implementation coordinating:
  Stage 1: Patient data input (52 clinical features)
  Stage 2: Parallel gate evaluation (G1-G6)
  Stage 3: ACWCM conservative merging
  Stage 4: Output (risk tier, safety certificate, audit trail)
"""

import time
from typing import Dict, Tuple, Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from merging.risk_lattice import RiskTier
from merging.conservative_merging import ConservativeMerging
from merging.safety_certificate import SafetyCertificateGenerator, SafetyCertificate
from gates.gate1_critical_flags import Gate1CriticalFlags
from gates.gate2_moderate_risk import Gate2ModerateRisk
from gates.gate3_data_quality import Gate3DataQuality
from gates.gate4_titrate_logic import Gate4TiTrATELogic
from gates.gate5_uncertainty import Gate5Uncertainty
from gates.gate6_temporal_risk import Gate6TemporalRisk
from utils.audit_trail import AuditTrailGenerator


class SAFEGate:
    """
    SAFE-Gate: Formally verified clinical triage architecture.

    Implements a four-stage pipeline:
      1. Patient data input (52 clinical features)
      2. Parallel gate evaluation (G1-G6)
      3. ACWCM conservative merging (Algorithm 1)
      4. Output: risk tier, safety certificate (Algorithm 2), audit trail (Algorithm 3)

    Safety properties (Definition 2, Theorem 1):
      - Conservative Preservation (CP)
      - Abstention Correctness (AC)
      - Critical Non-Dilution (CND)
    """

    def __init__(self, mode: str = "acwcm"):
        """
        Initialize SAFE-Gate system.

        Args:
            mode: "acwcm" for Adaptive Confidence-Weighted Conservative Merging,
                  "min" for basic minimum selection (Equation 2)
        """
        self.gates = {
            'G1': Gate1CriticalFlags(),
            'G2': Gate2ModerateRisk(),
            'G3': Gate3DataQuality(),
            'G4': Gate4TiTrATELogic(),
            'G5': Gate5Uncertainty(),
            'G6': Gate6TemporalRisk()
        }

        self.merger = ConservativeMerging(mode=mode)
        self.cert_generator = SafetyCertificateGenerator()
        self.audit_generator = AuditTrailGenerator()
        self.mode = mode
        self.decision_count = 0

    def classify(
        self,
        patient_data: Dict,
        patient_id: Optional[str] = None,
        return_audit_trail: bool = True,
        return_certificate: bool = True
    ) -> Dict:
        """
        Perform triage classification for a patient.

        Args:
            patient_data: Dictionary with 52 clinical features
            patient_id: Optional identifier
            return_audit_trail: Include full audit trail
            return_certificate: Include safety certificate

        Returns:
            Dictionary with:
              - 'final_tier': Risk tier string (R*, R1-R5)
              - 'confidence': Overall confidence
              - 'enforcing_gate': Gate that determined final tier
              - 'safety_certificate': SafetyCertificate dict (if requested)
              - 'audit_trail': Full audit trail (if requested)
              - 'latency_ms': Processing time
        """
        start = time.time()

        # --- Stage 2: Parallel gate evaluation ---
        gate_outputs = {}
        gate_confidences = {}
        gate_reasonings = {}

        for name, gate in self.gates.items():
            tier, conf, reasoning = gate.evaluate(patient_data)
            gate_outputs[name] = tier
            gate_confidences[name] = conf
            gate_reasonings[name] = {
                'tier': tier,
                'confidence': conf,
                'reasoning': reasoning
            }

        # --- Stage 3: ACWCM merging (Algorithm 1) ---
        final_tier, enforcing_gate, merge_audit = self.merger.merge(
            gate_outputs, gate_confidences, patient_id
        )

        # --- Gate conflict resolution (Algorithm 3) ---
        conflict_audit = self.merger.resolve_conflicts(
            gate_outputs, gate_confidences, final_tier
        )

        # --- Safety certificate generation (Algorithm 2) ---
        certificate = None
        if return_certificate:
            certificate = self.cert_generator.generate(
                patient_data=patient_data,
                r_final=final_tier,
                gate_outputs=gate_outputs,
                gate_confidences=gate_confidences,
                gate_reasonings=gate_reasonings,
                classify_fn=lambda pd: self._quick_classify(pd),
                feature_names=list(patient_data.keys())
            )

        # --- Audit trail ---
        audit_trail = None
        if return_audit_trail:
            audit_trail = self.audit_generator.generate(
                patient_id=patient_id,
                patient_data=patient_data,
                gate_outputs=gate_reasonings,
                final_tier=str(final_tier),
                enforcing_gate=enforcing_gate,
                merging_audit=merge_audit,
                theorem_verification=merge_audit.get('safety_properties', {})
            )
            audit_trail['conflict_resolution'] = conflict_audit

        latency_ms = (time.time() - start) * 1000
        self.decision_count += 1

        result = {
            'final_tier': str(final_tier),
            'confidence': gate_confidences.get(enforcing_gate, 0.0),
            'enforcing_gate': enforcing_gate,
            'latency_ms': round(latency_ms, 2),
            'decision_number': self.decision_count,
            'mode': self.mode,
            'gate_outputs': {g: str(t) for g, t in gate_outputs.items()},
            'gate_confidences': gate_confidences,
        }

        if certificate:
            result['safety_certificate'] = certificate.to_dict()
        if audit_trail:
            result['audit_trail'] = audit_trail

        return result

    def _quick_classify(self, patient_data: Dict) -> Dict:
        """Quick classification without certificates (for counterfactual analysis)."""
        gate_outputs = {}
        gate_confidences = {}
        for name, gate in self.gates.items():
            tier, conf, _ = gate.evaluate(patient_data)
            gate_outputs[name] = tier
            gate_confidences[name] = conf

        final_tier, _, _ = self.merger.merge(gate_outputs, gate_confidences)
        return {'final_tier': str(final_tier)}

    def batch_classify(
        self,
        patients: list,
        show_progress: bool = True
    ) -> list:
        """Classify multiple patients."""
        results = []
        iterator = patients

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(patients, desc="Classifying patients")
            except ImportError:
                pass

        for patient in iterator:
            pid = patient.get('patient_id')
            result = self.classify(
                patient, pid,
                return_audit_trail=False,
                return_certificate=False
            )
            results.append(result)
        return results

    def get_statistics(self) -> Dict:
        return {
            'total_decisions': self.decision_count,
            'gates_implemented': len(self.gates),
            'mode': self.mode,
        }

    def print_audit_trail(self, audit_trail: Dict):
        if audit_trail:
            report = self.audit_generator.to_clinical_report(audit_trail)
            print(report)
        else:
            print("No audit trail available")


def main():
    """Demonstration of SAFE-Gate system."""
    print("SAFE-Gate: Safety-Assured Fusion Engine with Gated Expert Triage")
    print("=" * 70)
    print()

    safegate = SAFEGate(mode="acwcm")
    print(f"System initialised with {len(safegate.gates)} gates (mode: {safegate.mode})")
    print()

    # Example 1: Critical case
    print("Example 1: Critical Patient (expected R1)")
    print("-" * 70)
    critical = {
        'patient_id': 'P001',
        'age': 72, 'gender': 'male',
        'systolic_bp': 85, 'diastolic_bp': 55,
        'heart_rate': 125, 'spo2': 88, 'gcs': 13, 'temperature': 37.2,
        'symptom_onset_hours': 1.5, 'symptom_duration_days': 0.1,
        'vertigo_severity': 'severe', 'sudden_onset': True,
        'progression_pattern': 'worsening',
        'dysarthria': True, 'ataxia': True,
        'hypertension': True, 'atrial_fibrillation': True,
        'diabetes': False, 'prior_stroke': True,
        'hints_head_impulse': 'abnormal',
        'hints_nystagmus': 'central',
        'hints_test_of_skew': 'positive',
    }

    r1 = safegate.classify(critical)
    print(f"  Final Tier:      {r1['final_tier']}")
    print(f"  Enforcing Gate:  {r1['enforcing_gate']}")
    print(f"  Confidence:      {r1['confidence']:.2f}")
    print(f"  Latency:         {r1['latency_ms']:.2f} ms")
    if 'safety_certificate' in r1:
        sc = r1['safety_certificate']
        print(f"  Certificate:     delta_min={sc['delta_min']}, "
              f"delta_cf={sc['delta_cf']}, enforcer={sc['g_enforce']}")
    print()

    # Example 2: Safe discharge case
    print("Example 2: Safe Discharge (expected R4-R5)")
    print("-" * 70)
    safe = {
        'patient_id': 'P247',
        'age': 34, 'gender': 'female',
        'systolic_bp': 118, 'diastolic_bp': 72,
        'heart_rate': 76, 'spo2': 99, 'gcs': 15, 'temperature': 36.8,
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
    }

    r2 = safegate.classify(safe)
    print(f"  Final Tier:      {r2['final_tier']}")
    print(f"  Enforcing Gate:  {r2['enforcing_gate']}")
    print(f"  Confidence:      {r2['confidence']:.2f}")
    print(f"  Latency:         {r2['latency_ms']:.2f} ms")
    if 'safety_certificate' in r2:
        sc = r2['safety_certificate']
        print(f"  Certificate:     delta_min={sc['delta_min']}, "
              f"delta_cf={sc['delta_cf']}, enforcer={sc['g_enforce']}")
    print()

    # Example 3: Incomplete data (expected R*)
    print("Example 3: Incomplete Data (expected R*)")
    print("-" * 70)
    incomplete = {
        'patient_id': 'P003',
        'age': 55,
        'vertigo_severity': 'moderate',
    }

    r3 = safegate.classify(incomplete)
    print(f"  Final Tier:      {r3['final_tier']}")
    print(f"  Enforcing Gate:  {r3['enforcing_gate']}")
    print(f"  Latency:         {r3['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
