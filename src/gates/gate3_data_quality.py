"""
Gate 3: Data Quality Assessment

Monitors whether the clinical record contains sufficient information to
support automated triage. A completeness ratio is calculated across 22
essential clinical fields (Equation 4 in paper).

Output mapping:
  rho < 0.70 --> R* (abstain)
  0.70 <= rho < 0.85 --> escalate 1 tier
  rho >= 0.85 --> R5

Confidence: c3 = rho_comp(x)
"""

from typing import Dict, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate3DataQuality:
    """
    Gate 3: Data Quality Assessment (22 essential fields).

    Equation 4:
      rho_comp(x) = |{f in F_ess : f(x) != missing}| / |F_ess|

    Tier mapping:
      rho < 0.70        --> R* (abstain: insufficient data)
      0.70 <= rho < 0.85 --> escalate 1 tier from baseline
      rho >= 0.85        --> R5 (data quality satisfactory)

    Confidence: c3 = rho_comp(x)
    """

    def __init__(self):
        """Initialize the 22 essential clinical fields."""
        self.essential_fields = [
            # Vital signs (6)
            'systolic_bp', 'diastolic_bp', 'heart_rate',
            'spo2', 'temperature', 'gcs',
            # Demographics (2)
            'age', 'gender',
            # Symptom characteristics (5)
            'symptom_onset_hours', 'symptom_duration_days',
            'vertigo_severity', 'sudden_onset', 'progression_pattern',
            # HINTS protocol (3)
            'hints_head_impulse', 'hints_nystagmus', 'hints_test_of_skew',
            # Cardiovascular history (4)
            'hypertension', 'atrial_fibrillation', 'diabetes', 'prior_stroke',
            # Neurological exam (2)
            'dysarthria', 'ataxia',
        ]
        assert len(self.essential_fields) == 22, \
            f"Expected 22 essential fields, got {len(self.essential_fields)}"

        # Thresholds from Equation 4
        self.abstention_threshold = 0.70
        self.escalation_threshold = 0.85

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate data completeness across 22 essential fields.

        Returns:
            (tier, confidence, reasoning)
            - confidence = rho_comp(x)
        """
        reasoning = {
            'gate': 'G3_Data_Quality',
            'mechanism': 'completeness_checking',
            'essential_fields_total': len(self.essential_fields),
            'missing_fields': []
        }

        present = 0
        missing = []
        for field in self.essential_fields:
            if field in patient_data and patient_data[field] is not None:
                present += 1
            else:
                missing.append(field)

        rho = present / len(self.essential_fields)
        reasoning['present_fields'] = present
        reasoning['missing_fields'] = missing
        reasoning['completeness'] = round(rho, 3)

        # Tier mapping (Equation 4)
        if rho < self.abstention_threshold:
            tier = RiskTier.R_STAR
            reasoning['decision'] = (
                f'Abstention: completeness {rho:.1%} < {self.abstention_threshold:.0%} '
                f'({len(missing)} of 22 fields missing)'
            )
        elif rho < self.escalation_threshold:
            # Escalate 1 tier from R5 baseline → R4
            tier = RiskTier.R4
            reasoning['decision'] = (
                f'Marginal completeness: {rho:.1%} in range '
                f'[{self.abstention_threshold:.0%}, {self.escalation_threshold:.0%}). '
                f'Escalating 1 tier. ({len(missing)} of 22 fields missing)'
            )
        else:
            tier = RiskTier.R5
            reasoning['decision'] = (
                f'Data quality satisfactory: completeness {rho:.1%} >= '
                f'{self.escalation_threshold:.0%}'
            )

        confidence = rho  # c3 = rho_comp(x)
        return tier, confidence, reasoning

    def get_name(self) -> str:
        return "G3"

    def get_description(self) -> str:
        return "Data Quality Assessment (Completeness, 22 essential fields)"
