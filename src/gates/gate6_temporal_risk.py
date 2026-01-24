"""
Gate 6: Temporal Risk Analysis

Analyzes symptom evolution patterns over time to differentiate vascular etiology
(acute/rapid progression) from peripheral vestibular disorders (gradual onset).
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate6TemporalRisk:
    """
    Gate 6: Temporal Risk Analysis

    Implements temporal state machine tracking symptom duration and progression:

    Time Course Categories:
    - Hyperacute (<1 hour): Suggests vascular event -> R1
    - Acute (1-24 hours): Stroke concern vs acute vestibular neuritis -> R2-R3
    - Subacute (1-7 days): Vestibular neuritis likely -> R3-R4
    - Chronic (>7 days): Peripheral disorder, safe for outpatient -> R4-R5

    Implements Theorem 6 (Temporal Consistency):
    Δt < 4.5 hours + neurological signs -> T ∈ {R1, R2}
    Δt > 7 days absent critical indicators -> T ∈ {R3, R4, R5}
    """

    def __init__(self):
        """Initialize temporal risk analysis thresholds."""
        # Temporal thresholds (hours)
        self.thresholds = {
            'hyperacute': 1.0,      # <1 hour
            'acute': 24.0,          # 1-24 hours
            'subacute': 168.0,      # 1-7 days (168 hours)
            'chronic': float('inf') # >7 days
        }

        # Duration thresholds in days (for symptom_duration_days field)
        self.duration_days_thresholds = {
            'hyperacute': 0.042,  # <1 hour (1/24 day)
            'acute': 1.0,         # <1 day
            'subacute': 7.0,      # <7 days
            'chronic': 7.0        # >=7 days
        }

        # Progression patterns
        self.progression_patterns = {
            'sudden_worsening': -1,     # Concerning (increases risk)
            'rapidly_progressive': -1,  # Concerning
            'stable': 0,                # Neutral
            'gradually_improving': +1,  # Reassuring
            'resolved': +2              # Reassuring
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate temporal patterns for risk stratification.

        Args:
            patient_data: Dictionary containing patient features

        Returns:
            Tuple of (risk_tier, confidence, reasoning)

        Algorithm:
            1. Determine time category (hyperacute/acute/subacute/chronic)
            2. Assess progression pattern
            3. Check for rapid worsening (red flag)
            4. Map to risk tier based on temporal state machine
        """
        reasoning = {
            'gate': 'G6_Temporal_Risk',
            'mechanism': 'temporal_state_machine',
            'triggers': []
        }

        # Extract temporal information
        onset_hours = patient_data.get('symptom_onset_hours')
        duration_days = patient_data.get('symptom_duration_days')

        # Use whichever field is available
        if onset_hours is not None:
            time_hours = onset_hours
            time_days = onset_hours / 24.0
        elif duration_days is not None:
            time_days = duration_days
            time_hours = duration_days * 24.0
        else:
            # No temporal information -> Cannot assess
            return RiskTier.R3, 0.5, {
                **reasoning,
                'decision': 'No temporal information available -> default R3',
                'time_category': 'unknown'
            }

        # Determine time category
        time_category = self._categorize_time_course(time_hours)
        reasoning['time_category'] = time_category
        reasoning['time_hours'] = round(time_hours, 2)
        reasoning['time_days'] = round(time_days, 2)

        # Assess progression pattern
        progression = patient_data.get('progression_pattern', 'stable')
        progression_modifier = self.progression_patterns.get(progression, 0)
        reasoning['progression_pattern'] = progression
        reasoning['progression_modifier'] = progression_modifier

        # Check for concerning features
        has_neurological_signs = any([
            patient_data.get('dysarthria', False),
            patient_data.get('ataxia', False),
            patient_data.get('diplopia', False),
            patient_data.get('focal_weakness', False)
        ])

        # Temporal state machine logic
        tier, confidence = self._temporal_state_machine(
            time_category,
            progression_modifier,
            has_neurological_signs
        )

        # Document decision
        decision_parts = [f"Time course: {time_category} ({time_hours:.1f} hours)"]

        if progression != 'stable':
            decision_parts.append(f"Progression: {progression}")

        if has_neurological_signs:
            decision_parts.append("+ Neurological signs")
            reasoning['triggers'].append("Neurological deficits present")

        reasoning['decision'] = ', '.join(decision_parts) + f" -> {tier}"

        # Check Theorem 6 compliance
        if time_hours < 4.5 and has_neurological_signs:
            reasoning['theorem6'] = (
                f"Within 4.5h window + neuro signs -> T ∈ {{R1, R2}} "
                f"(actual: {tier})"
            )

        if time_days > 7 and not has_neurological_signs:
            reasoning['theorem6'] = (
                f"Chronic (>7d) without critical signs -> T ∈ {{R3, R4, R5}} "
                f"(actual: {tier})"
            )

        return tier, confidence, reasoning

    def _categorize_time_course(self, time_hours: float) -> str:
        """
        Categorize time course into clinical categories.

        Args:
            time_hours: Time since symptom onset in hours

        Returns:
            Category: 'hyperacute', 'acute', 'subacute', or 'chronic'
        """
        if time_hours < self.thresholds['hyperacute']:
            return 'hyperacute'
        elif time_hours < self.thresholds['acute']:
            return 'acute'
        elif time_hours < self.thresholds['subacute']:
            return 'subacute'
        else:
            return 'chronic'

    def _temporal_state_machine(
        self,
        time_category: str,
        progression_modifier: int,
        has_neuro_signs: bool
    ) -> Tuple[RiskTier, float]:
        """
        Temporal state machine mapping time course to risk tier.

        State transitions:
        Hyperacute -> R1 (vascular event likely)
        Acute + neuro -> R2 (stroke vs acute vestibular)
        Acute alone -> R3
        Subacute -> R3-R4
        Chronic -> R4-R5

        Modifiers:
        - Neurological signs: Escalate risk
        - Rapid worsening: Escalate risk
        - Improving: De-escalate risk
        """
        base_tier_map = {
            'hyperacute': RiskTier.R1,
            'acute': RiskTier.R2,
            'subacute': RiskTier.R3,
            'chronic': RiskTier.R4
        }

        # Start with base tier from time category
        base_tier = base_tier_map.get(time_category, RiskTier.R3)
        base_confidence = 0.8

        # Apply modifiers
        final_value = base_tier.value

        # Neurological signs escalate risk (more conservative)
        if has_neuro_signs and final_value > 1:
            final_value -= 1  # Move toward R1 (more conservative)
            base_confidence += 0.1

        # Progression modifiers
        if progression_modifier < 0:  # Worsening
            if final_value > 1:
                final_value -= 1
        elif progression_modifier > 0:  # Improving
            if final_value < 5:
                final_value += 1

        # Ensure valid range
        final_value = max(1, min(5, final_value))

        final_tier = RiskTier(final_value)
        final_confidence = min(0.95, base_confidence)

        return final_tier, final_confidence

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G6"

    def get_description(self) -> str:
        """Return gate description."""
        return "Temporal Risk Analysis (Symptom Evolution)"

    def check_theorem6(self, patient_data: Dict) -> bool:
        """
        Verify Theorem 6 (Temporal Consistency).

        Theorem 6:
        - Δt < 4.5h + neuro signs -> T ∈ {R1, R2}
        - Δt > 7d without critical signs -> T ∈ {R3, R4, R5}

        Returns:
            True if theorem holds, False otherwise
        """
        tier, _, reasoning = self.evaluate(patient_data)

        onset_hours = patient_data.get('symptom_onset_hours')
        duration_days = patient_data.get('symptom_duration_days')

        time_hours = onset_hours if onset_hours is not None else (duration_days * 24 if duration_days else 999)

        has_neuro = any([
            patient_data.get('dysarthria', False),
            patient_data.get('ataxia', False),
            patient_data.get('diplopia', False)
        ])

        # Check acute with neuro signs
        if time_hours < 4.5 and has_neuro:
            return tier in [RiskTier.R_STAR, RiskTier.R1, RiskTier.R2]

        # Check chronic without critical signs
        if time_hours > 168 and not has_neuro:  # >7 days
            return tier in [RiskTier.R3, RiskTier.R4, RiskTier.R5]

        return True  # Theorem doesn't apply
