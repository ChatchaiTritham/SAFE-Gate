"""
Gate 6: Temporal Risk Analysis (State Machine)

Captures time-dependent risk trajectories through a finite-state machine
representation. State transitions follow established temporal patterns
from acute neurovascular management ("time is brain" doctrine).

State transitions (from article Section 3.8):
  Hyperacute (<1h) + worsening  --> R1
  Acute stable (1-24h)          --> R2-R3
  Acute improving (1-24h)       --> R3-R4
  Subacute (1-7 days)           --> R3-R4
  Chronic (>7 days)             --> R4-R5

Confidence:
  c6 = 1.0 when temporal profile is unambiguous
  c6 = 0.5 when timeline cannot be established with certainty
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate6TemporalRisk:
    """
    Gate 6: Temporal Risk Analysis (Finite-State Machine).

    Five temporal states with progression-modified transitions:
      HYPERACUTE (<1h) + worsening       -> R1
      ACUTE_STABLE (1-24h, stable)       -> R2 or R3
      ACUTE_IMPROVING (1-24h, improving) -> R3 or R4
      SUBACUTE (1-7d)                    -> R3 or R4
      CHRONIC (>7d)                      -> R4 or R5

    Confidence:
      c6 = 1.0 for unambiguous temporal profile
      c6 = 0.5 when timeline uncertain
    """

    # Temporal thresholds (hours)
    HYPERACUTE_LIMIT = 1.0       # <1 hour
    ACUTE_LIMIT = 24.0           # 1-24 hours
    SUBACUTE_LIMIT = 168.0       # 1-7 days (168 hours)

    def __init__(self):
        """Initialise temporal risk analysis thresholds."""
        # Progression pattern modifiers
        self.progression_effects = {
            'sudden_worsening': 'worsening',
            'rapidly_progressive': 'worsening',
            'worsening': 'worsening',
            'stable': 'stable',
            'gradually_improving': 'improving',
            'improving': 'improving',
            'resolved': 'improving',
            'episodic': 'stable',  # Recurrent episodes treated as stable trajectory
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate temporal patterns via finite-state machine.

        Returns:
            (tier, confidence, reasoning)
            - c6 = 1.0 if temporal profile unambiguous, 0.5 otherwise
        """
        reasoning = {
            'gate': 'G6_Temporal_Risk',
            'mechanism': 'finite_state_machine',
            'triggers': [],
        }

        # Extract temporal information
        onset_hours = patient_data.get('symptom_onset_hours')
        duration_days = patient_data.get('symptom_duration_days')

        # Determine time in hours from available data
        if onset_hours is not None:
            time_hours = float(onset_hours)
            time_days = time_hours / 24.0
            temporal_certainty = True
        elif duration_days is not None:
            time_days = float(duration_days)
            time_hours = time_days * 24.0
            temporal_certainty = True
        else:
            # No temporal information -> cannot assess
            reasoning['decision'] = (
                'No temporal information available. '
                'Default R3 with low confidence (c6=0.5).'
            )
            reasoning['time_category'] = 'unknown'
            return RiskTier.R3, 0.5, reasoning

        reasoning['time_hours'] = round(time_hours, 2)
        reasoning['time_days'] = round(time_days, 2)

        # Determine progression trajectory
        raw_progression = patient_data.get('progression_pattern', 'stable')
        trajectory = self.progression_effects.get(
            str(raw_progression).lower(), 'stable'
        )
        reasoning['progression_raw'] = raw_progression
        reasoning['trajectory'] = trajectory

        # Check for neurological signs (affects acute state transitions)
        has_neuro_signs = any([
            patient_data.get('dysarthria', False),
            patient_data.get('ataxia', False),
            patient_data.get('diplopia', False),
            patient_data.get('focal_weakness', False),
            patient_data.get('limb_weakness', False),
        ])

        # ---- Finite-State Machine Transitions ----
        if time_hours < self.HYPERACUTE_LIMIT:
            # HYPERACUTE (<1h)
            time_category = 'hyperacute'
            if trajectory == 'worsening':
                tier = RiskTier.R1
                reasoning['decision'] = (
                    f'Hyperacute onset ({time_hours:.1f}h) with worsening '
                    f'trajectory -> R1 (vascular event likely)'
                )
            elif has_neuro_signs:
                tier = RiskTier.R1
                reasoning['decision'] = (
                    f'Hyperacute onset ({time_hours:.1f}h) with neurological '
                    f'signs -> R1'
                )
                reasoning['triggers'].append('Hyperacute + neurological signs')
            else:
                tier = RiskTier.R2
                reasoning['decision'] = (
                    f'Hyperacute onset ({time_hours:.1f}h) without worsening '
                    f'-> R2 (concerning timing, needs monitoring)'
                )

        elif time_hours < self.ACUTE_LIMIT:
            # ACUTE (1-24h)
            time_category = 'acute'
            if trajectory == 'improving':
                # Acute improving -> R3-R4
                tier = RiskTier.R4 if not has_neuro_signs else RiskTier.R3
                reasoning['decision'] = (
                    f'Acute ({time_hours:.1f}h) with improving trajectory '
                    f'-> {tier.name}'
                )
            elif trajectory == 'worsening':
                # Acute worsening -> R2
                tier = RiskTier.R2
                reasoning['decision'] = (
                    f'Acute ({time_hours:.1f}h) with worsening trajectory '
                    f'-> R2 (escalation concern)'
                )
                reasoning['triggers'].append('Acute worsening trajectory')
            else:
                # Acute stable -> R2-R3
                if has_neuro_signs:
                    tier = RiskTier.R2
                    reasoning['decision'] = (
                        f'Acute stable ({time_hours:.1f}h) with neurological '
                        f'signs -> R2'
                    )
                else:
                    tier = RiskTier.R3
                    reasoning['decision'] = (
                        f'Acute stable ({time_hours:.1f}h) -> R3'
                    )

        elif time_hours < self.SUBACUTE_LIMIT:
            # SUBACUTE (1-7 days)
            time_category = 'subacute'
            if trajectory == 'worsening':
                tier = RiskTier.R3
                reasoning['decision'] = (
                    f'Subacute ({time_days:.1f}d) with worsening -> R3'
                )
                reasoning['triggers'].append('Subacute worsening')
            elif has_neuro_signs:
                tier = RiskTier.R3
                reasoning['decision'] = (
                    f'Subacute ({time_days:.1f}d) with neurological signs -> R3'
                )
            else:
                tier = RiskTier.R4
                reasoning['decision'] = (
                    f'Subacute ({time_days:.1f}d) without concerning features '
                    f'-> R4'
                )

        else:
            # CHRONIC (>7 days)
            time_category = 'chronic'
            if trajectory == 'worsening':
                tier = RiskTier.R4
                reasoning['decision'] = (
                    f'Chronic ({time_days:.1f}d) but worsening -> R4 '
                    f'(unusual trajectory, warrants attention)'
                )
            elif has_neuro_signs:
                tier = RiskTier.R4
                reasoning['decision'] = (
                    f'Chronic ({time_days:.1f}d) with neurological signs -> R4'
                )
            else:
                tier = RiskTier.R5
                reasoning['decision'] = (
                    f'Chronic ({time_days:.1f}d) stable/improving -> R5 '
                    f'(safe for outpatient management)'
                )

        reasoning['time_category'] = time_category

        # Confidence: c6 = 1.0 if temporal profile unambiguous, 0.5 if uncertain
        if temporal_certainty and time_category != 'unknown':
            # Unambiguous: clear time category AND clear trajectory
            if trajectory in ('worsening', 'improving'):
                confidence = 1.0
            else:
                confidence = 0.8  # Stable trajectory is less discriminating
        else:
            confidence = 0.5

        reasoning['confidence_rationale'] = (
            f'c6={confidence:.1f} '
            f'({"unambiguous" if confidence >= 0.8 else "uncertain"} temporal profile)'
        )

        return tier, confidence, reasoning

    def get_name(self) -> str:
        return "G6"

    def get_description(self) -> str:
        return "Temporal Risk Analysis (Finite-State Machine)"
