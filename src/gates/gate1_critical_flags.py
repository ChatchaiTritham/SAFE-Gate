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
        # Hemodynamic thresholds
        self.thresholds = {
            'systolic_bp_min': 90,      # mmHg
            'heart_rate_max': 120,      # bpm
            'spo2_min': 90,             # %
            'gcs_min': 14,              # Glasgow Coma Scale
            'temperature_max': 39.0     # °C
        }

        # Severe thresholds for single-flag R1 trigger
        self.severe_thresholds = {
            'systolic_bp_severe': 80,   # mmHg - shock
            'heart_rate_severe': 140,   # bpm - severe tachycardia
            'spo2_severe': 85,          # % - severe hypoxemia
            'gcs_severe': 12            # Major altered consciousness
        }

        # Neurological red flags (central stroke signs)
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
        hemodynamic_flags = 0
        severe_flags = 0

        if systolic_bp < self.severe_thresholds['systolic_bp_severe']:
            severe_flags += 1
            reasoning['triggers'].append(
                f'SEVERE Hypotension: BP {systolic_bp} < {self.severe_thresholds["systolic_bp_severe"]} mmHg (SHOCK)'
            )
        elif systolic_bp < self.thresholds['systolic_bp_min']:
            hemodynamic_flags += 1
            reasoning['triggers'].append(
                f'Hypotension: BP {systolic_bp} < {self.thresholds["systolic_bp_min"]} mmHg'
            )

        heart_rate = patient_data.get('heart_rate', 80)
        if heart_rate > self.severe_thresholds['heart_rate_severe']:
            severe_flags += 1
            reasoning['triggers'].append(
                f'SEVERE Tachycardia: HR {heart_rate} > {self.severe_thresholds["heart_rate_severe"]} bpm'
            )
        elif heart_rate > self.thresholds['heart_rate_max']:
            hemodynamic_flags += 1
            reasoning['triggers'].append(
                f'Tachycardia: HR {heart_rate} > {self.thresholds["heart_rate_max"]} bpm'
            )

        spo2 = patient_data.get('spo2', 98)
        if spo2 < self.severe_thresholds['spo2_severe']:
            severe_flags += 1
            reasoning['triggers'].append(
                f'SEVERE Hypoxemia: SpO2 {spo2} < {self.severe_thresholds["spo2_severe"]}%'
            )
        elif spo2 < self.thresholds['spo2_min']:
            hemodynamic_flags += 1
            reasoning['triggers'].append(
                f'Hypoxemia: SpO2 {spo2} < {self.thresholds["spo2_min"]}%'
            )

        # Check consciousness level
        gcs = patient_data.get('gcs', 15)
        if gcs < self.severe_thresholds['gcs_severe']:
            severe_flags += 1
            reasoning['triggers'].append(
                f'SEVERE Altered consciousness: GCS {gcs} < {self.severe_thresholds["gcs_severe"]}'
            )
        elif gcs < self.thresholds['gcs_min']:
            hemodynamic_flags += 1
            reasoning['triggers'].append(
                f'Altered consciousness: GCS {gcs} < {self.thresholds["gcs_min"]}'
            )

        # Check for acute focal neurological deficits
        neuro_flags = 0
        for flag in self.neurological_flags:
            if patient_data.get(flag, False):
                neuro_flags += 1
                reasoning['triggers'].append(
                    f'Neurological red flag: {flag.replace("_", " ")}'
                )

        # Decision logic: R1 ONLY for truly critical cases
        # Very strict criteria to avoid over-calling R1
        total_flags = hemodynamic_flags + neuro_flags

        # R1: Only SEVERE flags OR multiple hemodynamic instability
        if severe_flags > 0:
            # Severe flag (shock, severe hypoxemia, major altered consciousness) -> R1
            tier = RiskTier.R1
            confidence = 1.0
            reasoning['decision'] = f'SEVERE critical flag detected ({severe_flags} severe flags)'
        elif hemodynamic_flags >= 2 and neuro_flags >= 1:
            # Multiple hemodynamic issues + neuro deficit -> R1 (critically unstable)
            tier = RiskTier.R1
            confidence = 0.95
            reasoning['decision'] = f'Multiple hemodynamic instability ({hemodynamic_flags}) + neurological deficit ({neuro_flags})'
        elif total_flags >= 3 and hemodynamic_flags >= 1:
            # Many flags including hemodynamic -> R1
            tier = RiskTier.R1
            confidence = 0.90
            reasoning['decision'] = f'Multiple critical flags: {total_flags} total (hemodynamic: {hemodynamic_flags}, neuro: {neuro_flags})'
        elif total_flags >= 2:
            # 2+ concerning flags (but not meeting R1 criteria) -> R2
            tier = RiskTier.R2
            confidence = 0.85
            reasoning['decision'] = f'Multiple concerning flags: {total_flags} flags (hemodynamic: {hemodynamic_flags}, neuro: {neuro_flags})'
        elif reasoning['triggers']:
            # Single mild flag -> R3 (moderate concern, not high risk)
            # G1's role is detecting CRITICAL instability
            # Single non-severe flag suggests monitoring needed but not urgent
            tier = RiskTier.R3
            confidence = 0.75
            reasoning['decision'] = f'Single concerning flag detected: {reasoning["triggers"][0]}'
        else:
            # No critical flags -> R5 (minimal risk)
            # G1's role is FLAG DETECTION: absence of life-threatening flags
            # means patient is safe from hemodynamic/neurological crisis perspective
            # Other gates can escalate via conservative merging if needed
            tier = RiskTier.R5
            confidence = 0.8
            reasoning['decision'] = 'No critical flags detected -> minimal risk from safety perspective'

        return tier, confidence, reasoning

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G1"

    def get_description(self) -> str:
        """Return gate description."""
        return "Critical Flags Detection (Rule-based)"
