"""
Gate 1: Critical Red Flag Detection (Rule-Based)

Deterministic screening via 20 Boolean conditions derived from established
emergency medicine guidelines (AHA/ASA 2019). Rules are grouped into 5
clinical categories. Any single positive rule immediately produces R1 at
maximal confidence; all 20 conditions negative yields R5 with c1=1.0.
"""

import os
import sys
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate1CriticalFlags:
    """
    Gate 1: Critical Red Flag Detection

    20 Boolean conditions in 5 categories (Table 1 in paper):
      1. Hemodynamic instability: SBP < 90 or SBP > 180
      2. Altered mental status:   GCS < 14
      3. Acute focal deficits:    diplopia, dysarthria, ataxia, etc.
      4. Severe headache:         thunderclap onset
      5. Respiratory compromise:  O2 < 92%

    Output: R1 (c=1.0) if ANY rule fires; R5 (c=1.0) if ALL negative.
    """

    # Conditions that need an immediate life-saving intervention (ESI level 1:
    # hypotension, extreme heart rate, hypoxaemia, depressed consciousness).
    # Under the graded calibration every other red flag (hypertensive crisis,
    # focal deficit, central HINTS sign, headache red flag) maps to R2, the
    # high-risk/emergent tier (ESI level 2, which includes stroke symptoms).
    LIFE_THREAT_PREFIXES = ('Hemodynamic: SBP', 'Hemodynamic: HR', 'Altered mental status',
                            'Respiratory compromise')

    def __init__(self, graded: bool = False, vacuous_negative: bool = False):
        """Initialize red flag rules matching article Table 1.

        Args:
            graded: False (calibration v1) maps every red flag to R1. True
                (calibration v2) maps life-threatening instability to R1 and the
                remaining red flags to R2.
        """
        self.graded = graded
        # vacuous_negative (calibration v2): finding no red flag is absence of
        # evidence, not evidence of a benign course, so the negative output R5
        # carries confidence 0 (a vacuous belief): it sets no floor and has no
        # weight in the consensus of the merging rule.
        self.vacuous_negative = vacuous_negative
        # Category 1: Hemodynamic instability
        self.sbp_low = 90  # SBP < 90 mmHg
        self.sbp_high = 180  # SBP > 180 mmHg
        self.hr_low = 50  # HR < 50 bpm (severe bradycardia)
        self.hr_high = 150  # HR > 150 bpm (severe tachycardia)

        # Category 2: Altered mental status
        self.gcs_threshold = 14  # GCS < 14

        # Category 3: Acute focal neurological deficits
        self.focal_deficit_fields = [
            'diplopia',
            'dysarthria',
            'ataxia',
            'crossed_sensory_loss',
            'vertical_skew_deviation',
            'facial_weakness',
            'limb_weakness',
            'new_onset_diplopia',
        ]

        # Category 4: Severe headache
        self.headache_fields = [
            'thunderclap_headache',
            'severe_headache',
            'worst_headache_ever',
        ]

        # Category 5: Respiratory compromise
        self.spo2_threshold = 92  # O2 < 92%

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate patient for critical red flags.

        Returns:
            (RiskTier, confidence, reasoning_dict)
            - R1, 1.0 if any red flag detected
            - R5, 1.0 if all 20 conditions negative
        """
        reasoning = {
            'gate': 'G1_Critical_Red_Flags',
            'triggers': [],
            'mechanism': 'rule-based',
            'n_rules': 18,
            'categories': 5,
        }

        # --- Category 1: Hemodynamic instability ---
        sbp = patient_data.get('systolic_bp', 120)
        if sbp < self.sbp_low:
            reasoning['triggers'].append(
                f'Hemodynamic: SBP {sbp} < {self.sbp_low} mmHg'
            )
        if sbp > self.sbp_high:
            reasoning['triggers'].append(
                f'Hemodynamic: SBP {sbp} > {self.sbp_high} mmHg (hypertensive emergency)'
            )

        hr = patient_data.get('heart_rate', 80)
        if hr < self.hr_low:
            reasoning['triggers'].append(
                f'Hemodynamic: HR {hr} < {self.hr_low} bpm (severe bradycardia)'
            )
        if hr > self.hr_high:
            reasoning['triggers'].append(
                f'Hemodynamic: HR {hr} > {self.hr_high} bpm (severe tachycardia)'
            )

        dbp = patient_data.get('diastolic_bp', 80)
        if dbp > 120:
            reasoning['triggers'].append(f'Hemodynamic: DBP {dbp} > 120 mmHg')

        # --- Category 2: Altered mental status ---
        gcs = patient_data.get('gcs', 15)
        if gcs < self.gcs_threshold:
            reasoning['triggers'].append(
                f'Altered mental status: GCS {gcs} < {self.gcs_threshold}'
            )

        # --- Category 3: Acute focal neurological deficits ---
        for field in self.focal_deficit_fields:
            if patient_data.get(field, False):
                reasoning['triggers'].append(
                    f'Focal deficit: {field.replace("_", " ")}'
                )

        # HINTS signs of central pathology
        if patient_data.get('hints_nystagmus') == 'central':
            reasoning['triggers'].append('Focal deficit: central nystagmus pattern')
        if patient_data.get('hints_test_of_skew') == 'positive':
            reasoning['triggers'].append('Focal deficit: positive test of skew')

        # --- Category 4: Severe headache ---
        for field in self.headache_fields:
            if patient_data.get(field, False):
                reasoning['triggers'].append(
                    f'Severe headache: {field.replace("_", " ")}'
                )

        # --- Category 5: Respiratory compromise ---
        spo2 = patient_data.get('spo2', 98)
        if spo2 < self.spo2_threshold:
            reasoning['triggers'].append(
                f'Respiratory compromise: O2 {spo2}% < {self.spo2_threshold}%'
            )

        # --- Decision: ANY trigger → R1 (c=1.0); ALL negative → R5 (c=1.0) ---
        life_threat = [t for t in reasoning['triggers']
                       if t.startswith(self.LIFE_THREAT_PREFIXES) and 'hypertensive' not in t]
        if reasoning['triggers'] and self.graded and not life_threat:
            tier = RiskTier.R2
            confidence = 1.0
            reasoning['decision'] = (
                f'RED FLAG DETECTED without life-threatening instability: '
                f'{len(reasoning["triggers"])} finding(s). Emergent R2 classification.'
            )
        elif reasoning['triggers']:
            tier = RiskTier.R1
            confidence = 1.0
            reasoning['decision'] = (
                f'RED FLAG DETECTED: {len(reasoning["triggers"])} critical finding(s). '
                f'Immediate R1 classification.'
            )
        else:
            tier = RiskTier.R5
            confidence = 0.0 if self.vacuous_negative else 1.0
            reasoning['decision'] = (
                'No critical red flags detected across all 20 conditions. '
                'R5 assigned with full confidence.'
            )

        return tier, confidence, reasoning

    def get_name(self) -> str:
        return "G1"

    def get_description(self) -> str:
        return "Critical Red Flag Detection (Rule-based, 20 conditions, 5 categories)"
