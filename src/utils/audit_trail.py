"""
Audit Trail Generation for SAFE-Gate

Generates complete, faithful audit trails documenting decision logic for
clinical review, quality assurance, and medicolegal documentation.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json


class AuditTrailGenerator:
    """
    Generates comprehensive audit trails for SAFE-Gate decisions.

    Unlike post-hoc explainability (SHAP, LIME), audit trails emerge directly
    from gate logic, ensuring explanations remain faithful to actual decision mechanisms.
    """

    def __init__(self):
        """Initialize audit trail generator."""
        pass

    def generate(
        self,
        patient_id: Optional[str],
        patient_data: Dict,
        gate_outputs: Dict,
        final_tier: str,
        enforcing_gate: str,
        merging_audit: Dict,
        theorem_verification: Optional[Dict] = None
    ) -> Dict:
        """
        Generate complete audit trail for a single patient decision.

        Args:
            patient_id: Patient identifier
            patient_data: Input patient features
            gate_outputs: Outputs from all six gates
            final_tier: Final risk tier after merging
            enforcing_gate: Which gate enforced the final tier
            merging_audit: Audit trail from conservative merging
            theorem_verification: Theorem verification results

        Returns:
            Complete audit trail dictionary
        """
        audit_trail = {
            'metadata': {
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'system': 'SAFE-Gate v1.0.0'
            },
            'input_data': {
                'num_features': len(patient_data),
                'features': patient_data
            },
            'gate_evaluations': {},
            'merging': merging_audit,
            'final_decision': {
                'tier': final_tier,
                'enforcing_gate': enforcing_gate,
                'description': self._get_tier_description(final_tier)
            },
            'theorem_verification': theorem_verification or {}
        }

        # Document individual gate outputs
        for gate_name, gate_result in gate_outputs.items():
            audit_trail['gate_evaluations'][gate_name] = {
                'tier': str(gate_result.get('tier', 'Unknown')),
                'confidence': gate_result.get('confidence', 0.0),
                'reasoning': gate_result.get('reasoning', {})
            }

        return audit_trail

    def _get_tier_description(self, tier_str: str) -> str:
        """Get clinical description of risk tier."""
        descriptions = {
            'R*': 'Abstention: Uncertainty/incomplete data, requires human review',
            'R1': 'Critical: Life-threatening, immediate care required (<5 min)',
            'R2': 'High Risk: Suspected stroke, urgent evaluation (<15 min)',
            'R3': 'Moderate: Acute vertigo, standard evaluation (30-120 min)',
            'R4': 'Low Risk: Positional dizziness, delayed evaluation OK (1-4 hours)',
            'R5': 'Minimal: Chronic dizziness, safe discharge to outpatient'
        }
        return descriptions.get(tier_str, 'Unknown tier')

    def to_json(self, audit_trail: Dict, indent: int = 2) -> str:
        """Convert audit trail to JSON string."""
        return json.dumps(audit_trail, indent=indent)

    def to_clinical_report(self, audit_trail: Dict) -> str:
        """
        Generate human-readable clinical report from audit trail.

        Returns multi-line string suitable for clinical review.
        """
        report_lines = []

        # Header
        report_lines.append("=" * 70)
        report_lines.append("SAFE-GATE CLINICAL TRIAGE REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Patient & timestamp
        metadata = audit_trail.get('metadata', {})
        report_lines.append(f"Patient ID: {metadata.get('patient_id', 'Unknown')}")
        report_lines.append(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
        report_lines.append("")

        # Final decision
        final = audit_trail.get('final_decision', {})
        report_lines.append("FINAL TRIAGE DECISION:")
        report_lines.append(f"  Risk Tier: {final.get('tier', 'Unknown')}")
        report_lines.append(f"  Description: {final.get('description', 'Unknown')}")
        report_lines.append(f"  Enforced by: {final.get('enforcing_gate', 'Unknown')}")
        report_lines.append("")

        # Gate evaluations
        report_lines.append("INDIVIDUAL GATE EVALUATIONS:")
        gate_evals = audit_trail.get('gate_evaluations', {})
        for gate_name in sorted(gate_evals.keys()):
            gate_data = gate_evals[gate_name]
            report_lines.append(f"  {gate_name}:")
            report_lines.append(f"    Tier: {gate_data.get('tier', 'Unknown')}")
            report_lines.append(f"    Confidence: {gate_data.get('confidence', 0.0):.2f}")

            reasoning = gate_data.get('reasoning', {})
            if 'decision' in reasoning:
                report_lines.append(f"    Reasoning: {reasoning['decision']}")

            if 'triggers' in reasoning and reasoning['triggers']:
                report_lines.append(f"    Triggers: {', '.join(reasoning['triggers'])}")
            report_lines.append("")

        # Theorem verification
        theorem_ver = audit_trail.get('theorem_verification', {})
        if theorem_ver:
            report_lines.append("THEOREM VERIFICATION:")
            for theorem_name, result in theorem_ver.items():
                if isinstance(result, dict):
                    holds = result.get('holds', False)
                    status = "[PASS]" if holds else "[FAIL]"
                    report_lines.append(f"  {theorem_name}: {status}")
            report_lines.append("")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)
