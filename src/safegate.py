"""
SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine with parallel GATEs

Main system implementation coordinating parallel gate evaluation, conservative merging,
and audit trail generation.
"""

from typing import Dict, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from merging.risk_lattice import RiskTier
from merging.conservative_merging import ConservativeMerging
from gates.gate1_critical_flags import Gate1CriticalFlags
from gates.gate2_moderate_risk import Gate2ModerateRisk
from gates.gate3_data_quality import Gate3DataQuality
from gates.gate4_titrate_logic import Gate4TiTrATELogic
from gates.gate5_uncertainty import Gate5Uncertainty
from gates.gate6_temporal_risk import Gate6TemporalRisk
from utils.audit_trail import AuditTrailGenerator


class SAFEGate:
    """
    SAFE-Gate: Formally verified clinical triage architecture

    Implements three-phase pipeline:
    1. Parallel Gate Evaluation (O(1) complexity)
    2. Conservative Merging (O(1) complexity)
    3. Audit Trail Generation (O(1) complexity)

    Mathematical guarantees:
    - Theorem 1: Bounded false negative rates approaching zero
    - Theorem 2: Conservative bias preservation (T_final ⊑ Ti for all gates)
    - Theorem 3: Abstention correctness
    - Theorem 4: Monotonicity
    - Theorem 5: Data quality gate
    - Theorem 6: Temporal consistency
    """

    def __init__(self):
        """
        Initialize SAFE-Gate system with six parallel gates.

        All 6 gates operational:
        - G1: Critical Flags Detection (rule-based)
        - G2: Moderate Risk Scoring (XGBoost-based weighted scoring)
        - G3: Data Quality Assessment (completeness checking)
        - G4: TiTrATE Clinical Logic (Timing, Triggers, Targeted Exam)
        - G5: Uncertainty Quantification (Monte Carlo dropout)
        - G6: Temporal Risk Analysis (symptom evolution)
        """
        # Initialize all six parallel gates
        self.gates = {
            'G1': Gate1CriticalFlags(),
            'G2': Gate2ModerateRisk(),
            'G3': Gate3DataQuality(),
            'G4': Gate4TiTrATELogic(),
            'G5': Gate5Uncertainty(),
            'G6': Gate6TemporalRisk()
        }

        # Initialize conservative merging
        self.merger = ConservativeMerging()

        # Initialize audit trail generator
        self.audit_generator = AuditTrailGenerator()

        # Performance tracking
        self.decision_count = 0

    def classify(
        self,
        patient_data: Dict,
        patient_id: Optional[str] = None,
        return_audit_trail: bool = True
    ) -> Dict:
        """
        Perform triage classification for a patient presentation.

        Args:
            patient_data: Dictionary containing patient features (52 features)
            patient_id: Optional patient identifier
            return_audit_trail: Whether to include full audit trail in output

        Returns:
            Dictionary containing:
            - 'final_tier': Final risk tier (R*, R1-R5)
            - 'confidence': Overall confidence score
            - 'enforcing_gate': Which gate determined final tier
            - 'audit_trail': Complete audit trail (if requested)
            - 'latency_ms': Decision latency in milliseconds

        Example:
            >>> safegate = SAFEGate()
            >>> patient = {
            ...     'age': 68,
            ...     'systolic_bp': 145,
            ...     'heart_rate': 88,
            ...     'spo2': 97,
            ...     'symptom_onset_hours': 2.5,
            ...     'vertigo_severity': 'severe',
            ...     'dysarthria': True
            ... }
            >>> result = safegate.classify(patient)
            >>> print(f"Risk Tier: {result['final_tier']}")
        """
        import time
        start_time = time.time()

        # Phase 1: Parallel Gate Evaluation
        gate_outputs = {}
        gate_confidences = {}
        gate_reasonings = {}

        # Evaluate all six gates in parallel
        for gate_name, gate in self.gates.items():
            tier, confidence, reasoning = gate.evaluate(patient_data)
            gate_outputs[gate_name] = tier
            gate_confidences[gate_name] = confidence
            gate_reasonings[gate_name] = {
                'tier': tier,
                'confidence': confidence,
                'reasoning': reasoning
            }

        # Phase 2: Conservative Merging
        final_tier, enforcing_gate, merging_audit = self.merger.merge(
            gate_outputs,
            gate_confidences,
            patient_id
        )

        # Phase 3: Audit Trail Generation
        audit_trail = None
        if return_audit_trail:
            audit_trail = self.audit_generator.generate(
                patient_id=patient_id,
                patient_data=patient_data,
                gate_outputs=gate_reasonings,
                final_tier=str(final_tier),
                enforcing_gate=enforcing_gate,
                merging_audit=merging_audit,
                theorem_verification=merging_audit.get('theorem_verification', {})
            )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Increment decision counter
        self.decision_count += 1

        # Compile results
        result = {
            'final_tier': str(final_tier),
            'confidence': gate_confidences.get(enforcing_gate, 0.0),
            'enforcing_gate': enforcing_gate,
            'latency_ms': round(latency_ms, 2),
            'decision_number': self.decision_count
        }

        if return_audit_trail:
            result['audit_trail'] = audit_trail

        return result

    def batch_classify(
        self,
        patients: list,
        show_progress: bool = True
    ) -> list:
        """
        Classify multiple patients in batch.

        Args:
            patients: List of patient data dictionaries
            show_progress: Whether to show progress bar

        Returns:
            List of classification results
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(patients, desc="Classifying patients")
            except ImportError:
                iterator = patients
        else:
            iterator = patients

        for patient in iterator:
            patient_id = patient.get('patient_id')
            result = self.classify(patient, patient_id, return_audit_trail=False)
            results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """
        Get system performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_decisions': self.decision_count,
            'gates_implemented': len(self.gates),
            'gates_total': 6
        }

    def print_audit_trail(self, audit_trail: Dict):
        """Print human-readable audit trail."""
        if audit_trail:
            clinical_report = self.audit_generator.to_clinical_report(audit_trail)
            print(clinical_report)
        else:
            print("No audit trail available")


def main():
    """
    Demonstration of SAFE-Gate system.
    """
    print("SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine")
    print("=" * 70)
    print()

    # Initialize system
    safegate = SAFEGate()
    print(f"System initialized with {len(safegate.gates)} gates")
    print()

    # Example patient 1: Critical case
    print("Example 1: Critical Patient")
    print("-" * 70)
    critical_patient = {
        'patient_id': 'P001',
        'age': 72,
        'systolic_bp': 85,  # Hypotension (critical flag)
        'heart_rate': 125,  # Tachycardia (critical flag)
        'spo2': 88,  # Hypoxemia (critical flag)
        'gcs': 13,  # Altered consciousness (critical flag)
        'temperature': 37.2,
        'symptom_onset_hours': 1.5,
        'vertigo_severity': 'severe',
        'dysarthria': True,  # Neurological red flag
        'symptom_duration_days': 0.1,
        'hypertension': True,
        'atrial_fibrillation': True,
        'diabetes': False,
        'hints_head_impulse': 'abnormal',
        'hints_nystagmus': 'central',
        'hints_test_of_skew': 'positive'
    }

    result1 = safegate.classify(critical_patient)
    print(f"Final Tier: {result1['final_tier']}")
    print(f"Enforcing Gate: {result1['enforcing_gate']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Latency: {result1['latency_ms']:.2f} ms")
    print()

    # Example patient 2: Incomplete data (should trigger R*)
    print("Example 2: Incomplete Data Patient")
    print("-" * 70)
    incomplete_patient = {
        'patient_id': 'P002',
        'age': 55,
        'vertigo_severity': 'moderate',
        # Missing many critical fields → should trigger G3 abstention
    }

    result2 = safegate.classify(incomplete_patient)
    print(f"Final Tier: {result2['final_tier']}")
    print(f"Enforcing Gate: {result2['enforcing_gate']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Latency: {result2['latency_ms']:.2f} ms")
    print()

    # Print audit trail for incomplete data case
    print("Audit Trail for Incomplete Data Case:")
    print("=" * 70)
    safegate.print_audit_trail(result2['audit_trail'])


if __name__ == "__main__":
    main()
