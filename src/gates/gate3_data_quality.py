"""
Gate 3: Data Quality Assessment

Evaluates whether available data suffices for reliable classification.
Implements Theorem 5 (Data Quality Gate): Missing critical data forces R* abstention.
"""

from typing import Dict, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate3DataQuality:
    """
    Gate 3: Data Quality Assessment

    Implements explicit data completeness checking:
    - Completeness C = (# present critical fields) / (# required critical fields)
    - C < 85% → R* (abstention)
    - 85% ≤ C < 95% → downgrade confidence but permit classification
    - C ≥ 95% → proceed normally

    Theorem 5: Data Quality Gate
    If >15% of critical fields missing → T_final = R*
    """

    def __init__(self):
        """Initialize critical feature requirements."""
        # Critical features required for safe classification
        self.critical_fields = {
            # Vital signs
            'systolic_bp',
            'heart_rate',
            'spo2',
            'temperature',

            # Neurological examination
            'gcs',
            'hints_head_impulse',
            'hints_nystagmus',
            'hints_test_of_skew',

            # Symptom characteristics
            'symptom_onset_hours',
            'vertigo_severity',
            'symptom_duration_days',

            # Clinical history
            'age',
            'hypertension',
            'atrial_fibrillation',
            'diabetes'
        }

        self.completeness_thresholds = {
            'abstention_threshold': 0.85,    # C < 0.85 → R*
            'warning_threshold': 0.95        # C < 0.95 → reduce confidence
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate data quality and completeness.

        Args:
            patient_data: Dictionary containing patient features

        Returns:
            Tuple of (risk_tier, confidence, reasoning)

        Implementation of Theorem 5:
            completeness < 0.85 → R* (abstention)
        """
        reasoning = {
            'gate': 'G3_Data_Quality',
            'mechanism': 'completeness_checking',
            'critical_fields_required': len(self.critical_fields),
            'missing_fields': []
        }

        # Check which critical fields are present
        present_fields = 0
        missing_fields = []

        for field in self.critical_fields:
            if field in patient_data and patient_data[field] is not None:
                present_fields += 1
            else:
                missing_fields.append(field)

        # Calculate completeness
        completeness = present_fields / len(self.critical_fields)

        reasoning['present_fields'] = present_fields
        reasoning['missing_fields'] = missing_fields
        reasoning['completeness'] = round(completeness, 3)

        # Decision logic based on completeness thresholds
        if completeness < self.completeness_thresholds['abstention_threshold']:
            # Insufficient data → R* (abstention)
            tier = RiskTier.R_STAR
            confidence = 0.0

            reasoning['decision'] = (
                f'Insufficient data quality: completeness {completeness:.1%} '
                f'< threshold {self.completeness_thresholds["abstention_threshold"]:.0%}'
            )
            reasoning['theorem'] = 'Theorem 5 (Data Quality Gate) triggered'

        elif completeness < self.completeness_thresholds['warning_threshold']:
            # Marginal data quality → Proceed but with reduced confidence
            tier = RiskTier.R3  # Conservative default
            confidence = 0.6    # Reduced confidence

            reasoning['decision'] = (
                f'Marginal data quality: completeness {completeness:.1%} '
                f'in warning range [{self.completeness_thresholds["abstention_threshold"]:.0%}, '
                f'{self.completeness_thresholds["warning_threshold"]:.0%})'
            )

        else:
            # Sufficient data quality → Proceed normally
            tier = RiskTier.R5  # No data quality concerns
            confidence = 1.0

            reasoning['decision'] = (
                f'Sufficient data quality: completeness {completeness:.1%} '
                f'≥ threshold {self.completeness_thresholds["warning_threshold"]:.0%}'
            )

        return tier, confidence, reasoning

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G3"

    def get_description(self) -> str:
        """Return gate description."""
        return "Data Quality Assessment (Completeness Checking)"

    def check_theorem5(self, patient_data: Dict) -> bool:
        """
        Verify Theorem 5 (Data Quality Gate).

        Theorem 5: If >15% of critical fields missing → T_final = R*

        Returns:
            True if theorem conditions met (should output R*), False otherwise
        """
        tier, _, _ = self.evaluate(patient_data)
        present_count = sum(
            1 for field in self.critical_fields
            if field in patient_data and patient_data[field] is not None
        )
        completeness = present_count / len(self.critical_fields)

        # Theorem 5: completeness < 0.85 (i.e., >15% missing) should yield R*
        if completeness < 0.85:
            return tier == RiskTier.R_STAR
        return True  # Theorem doesn't apply
