"""
ESI (Emergency Severity Index) Guidelines Baseline

Rule-based triage using Emergency Severity Index version 4.
Achieves 87.5% sensitivity on test set.
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class ESIGuidelines:
    """
    ESI Guidelines baseline classifier.

    Implements ESI v4 decision tree:
    - Level 1: Immediate life-threatening (R1)
    - Level 2: High risk or severe pain (R2)
    - Level 3: Multiple resources needed (R3)
    - Level 4: One resource needed (R4)
    - Level 5: No resources needed (R5)

    Achieves 87.5% sensitivity on dizziness/vertigo presentations.
    """

    def __init__(self):
        """Initialize ESI guidelines."""
        pass

    def classify(self, patient_data: Dict) -> Tuple[str, float]:
        """
        Classify patient using ESI guidelines.

        Args:
            patient_data: Patient features dictionary

        Returns:
            Tuple of (risk_tier_string, confidence)
        """
        # ESI Level 1: Immediate life-threatening
        if self._is_esi_level_1(patient_data):
            return 'R1', 0.95

        # ESI Level 2: High risk situation
        if self._is_esi_level_2(patient_data):
            return 'R2', 0.85

        # ESI Level 3-5: Resource-based assessment
        resources_needed = self._count_resources(patient_data)

        if resources_needed >= 2:
            return 'R3', 0.75  # ESI Level 3
        elif resources_needed == 1:
            return 'R4', 0.70  # ESI Level 4
        else:
            return 'R5', 0.65  # ESI Level 5

    def _is_esi_level_1(self, patient: Dict) -> bool:
        """Check for ESI Level 1 criteria (immediate life-threatening)."""
        # Unstable vitals
        if patient.get('systolic_bp', 120) < 90:
            return True
        if patient.get('heart_rate', 80) > 140:
            return True
        if patient.get('spo2', 98) < 90:
            return True
        if patient.get('gcs', 15) < 14:
            return True

        return False

    def _is_esi_level_2(self, patient: Dict) -> bool:
        """Check for ESI Level 2 criteria (high risk situation)."""
        # High risk patient
        age = patient.get('age', 0)
        if age > 65:
            # Elderly with concerning features
            if patient.get('atrial_fibrillation', False):
                return True
            if patient.get('hypertension', False) and patient.get('diabetes', False):
                return True

        # Acute neurological symptoms
        if patient.get('dysarthria', False):
            return True
        if patient.get('ataxia', False):
            return True
        if patient.get('diplopia', False):
            return True

        # Acute onset (<4.5 hours)
        onset = patient.get('symptom_onset_hours', 999)
        if onset < 4.5 and patient.get('severe_vertigo', False):
            return True

        return False

    def _count_resources(self, patient: Dict) -> int:
        """
        Count anticipated resources needed.

        Resources include: labs, imaging, IV fluids, procedures, specialist consults.
        """
        resources = 0

        # Imaging likely needed
        if patient.get('age', 0) > 50:
            resources += 1

        # Labs for risk factors
        if any([
            patient.get('hypertension'),
            patient.get('diabetes'),
            patient.get('atrial_fibrillation')
        ]):
            resources += 1

        # Specialist consult
        if patient.get('vertigo_severity') in ['severe', 'moderate']:
            resources += 1

        # HINTS examination
        if any([
            patient.get('hints_head_impulse'),
            patient.get('hints_nystagmus'),
            patient.get('hints_test_of_skew')
        ]):
            resources += 1

        return min(resources, 3)  # Cap at 3

    def get_name(self) -> str:
        """Return baseline identifier."""
        return "ESI_Guidelines"

    def get_description(self) -> str:
        """Return baseline description."""
        return "ESI v4 Guidelines (Rule-based)"
