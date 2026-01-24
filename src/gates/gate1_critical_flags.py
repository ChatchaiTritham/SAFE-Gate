"""
Gate 1: Critical Flags Detection

Rule-based binary safety screening for life-threatening red flags requiring
immediate intervention regardless of symptom complexity.
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate1CriticalFlags:
    """
    Gate 1: Critical Flags Detection

    Detects life-threatening red flags through rule-based logic:
    - Hemodynamic instability (BP < 90 mmHg, HR > 120, SpO2 < 90%)
    - Altered consciousness (GCS < 14)
    - Acute focal neurological deficits (brainstem ischemia signs)

    Deterministic behavior ensures verifiable and stable performance.
    """

    def __init__(self):
        """Initialize critical flag thresholds."""
        self.thresholds = {
            'systolic_bp_min': 90,      # mmHg
            'heart_rate_max': 120,      # bpm
            'spo2_min': 90,             # %
            'gcs_min': 14,              # Glasgow Coma Scale
            'temperature_max': 39.0     # °C
        }

        self.neurological_flags = {
            'vertical_skew_deviation',
            'new_onset_diplopia',
            'crossed_sensory_loss',
            'severe_ataxia',
            'dysarthria',
            'nystagmus_with_diplopia'
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate patient for critical flags.

        Args:
            patient_data: Dictionary containing patient features

        Returns:
            Tuple of (risk_tier, confidence, reasoning)

        Algorithm:
            IF any critical flag detected:
                RETURN R1 (Critical) with confidence 1.0
            ELSE:
                RETURN None (allow other gates to determine tier)
        """
        reasoning = {
            'gate': 'G1_Critical_Flags',
            'triggers': [],
            'mechanism': 'rule-based'
        }

        # Check hemodynamic stability
        systolic_bp = patient_data.get('systolic_bp', 120)
        if systolic_bp < self.thresholds['systolic_bp_min']:
            reasoning['triggers'].append(
                f'Hypotension: BP {systolic_bp} < {self.thresholds["systolic_bp_min"]} mmHg'
            )

        heart_rate = patient_data.get('heart_rate', 80)
        if heart_rate > self.thresholds['heart_rate_max']:
            reasoning['triggers'].append(
                f'Tachycardia: HR {heart_rate} > {self.thresholds["heart_rate_max"]} bpm'
            )

        spo2 = patient_data.get('spo2', 98)
        if spo2 < self.thresholds['spo2_min']:
            reasoning['triggers'].append(
                f'Hypoxemia: SpO2 {spo2} < {self.thresholds["spo2_min"]}%'
            )

        # Check consciousness level
        gcs = patient_data.get('gcs', 15)
        if gcs < self.thresholds['gcs_min']:
            reasoning['triggers'].append(
                f'Altered consciousness: GCS {gcs} < {self.thresholds["gcs_min"]}'
            )

        # Check for acute focal neurological deficits
        for flag in self.neurological_flags:
            if patient_data.get(flag, False):
                reasoning['triggers'].append(
                    f'Neurological red flag: {flag.replace("_", " ")}'
                )

        # Decision logic
        if reasoning['triggers']:
            # Critical flag detected -> R1 (Critical) with confidence 1.0
            tier = RiskTier.R1
            confidence = 1.0
            reasoning['decision'] = f'Critical flags detected: {len(reasoning["triggers"])} triggers'
        else:
            # No critical flags -> R5 (safe) to avoid being conservative without reason
            tier = RiskTier.R5
            confidence = 0.8
            reasoning['decision'] = 'No critical flags detected'

        return tier, confidence, reasoning

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G1"

    def get_description(self) -> str:
        """Return gate description."""
        return "Critical Flags Detection (Rule-based)"
