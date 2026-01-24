"""
Gate 4: TiTrATE Clinical Logic

Implements validated clinical decision rules from the TiTrATE framework
(Timing, Triggers, Targeted Examination) specific to dizziness and vertigo
differential diagnosis.
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate4TiTrATELogic:
    """
    Gate 4: TiTrATE Clinical Logic

    Embedded expert system encoding neurologist knowledge through
    validated clinical decision rules:

    1. Timing: Symptom onset within vs beyond 4.5-hour thrombolysis window
    2. Triggers: Spontaneous onset vs positional triggers
    3. Targeted Examination: HINTS protocol (Head Impulse, Nystagmus, Test of Skew)

    Reference: Newman-Toker & Edlow (2008), Academic Emergency Medicine
    """

    def __init__(self):
        """Initialize TiTrATE clinical decision rules."""
        # AHA/ASA stroke guidelines: 4.5-hour thrombolysis window
        self.thrombolysis_window_hours = 4.5

        # HINTS protocol components
        self.hints_components = [
            'hints_head_impulse',
            'hints_nystagmus',
            'hints_test_of_skew'
        ]

        # Trigger patterns
        self.positional_triggers = {
            'rolling_over_in_bed',
            'looking_up',
            'bending_forward',
            'turning_head_quickly'
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate patient using TiTrATE framework.

        Args:
            patient_data: Dictionary containing patient features

        Returns:
            Tuple of (risk_tier, confidence, reasoning)

        Algorithm:
            1. Timing: Check onset vs thrombolysis window
            2. Triggers: Identify spontaneous vs positional
            3. Targeted Exam: Apply HINTS protocol
            4. Integrate findings to determine tier
        """
        reasoning = {
            'gate': 'G4_TiTrATE_Logic',
            'mechanism': 'clinical_decision_rules',
            'titrate_components': {},
            'triggers': []
        }

        # Component 1: Timing (T)
        timing_assessment = self._assess_timing(patient_data)
        reasoning['titrate_components']['timing'] = timing_assessment

        # Component 2: Triggers (Tr)
        trigger_assessment = self._assess_triggers(patient_data)
        reasoning['titrate_components']['triggers'] = trigger_assessment

        # Component 3: Targeted Examination (ATE)
        hints_assessment = self._assess_hints(patient_data)
        reasoning['titrate_components']['targeted_exam'] = hints_assessment

        # Decision logic integrating all TiTrATE components
        tier, confidence = self._integrate_findings(
            timing_assessment,
            trigger_assessment,
            hints_assessment
        )

        # Document decision rationale
        decision_factors = []

        if timing_assessment['within_window']:
            decision_factors.append("Within thrombolysis window")

        if trigger_assessment['pattern'] == 'spontaneous':
            decision_factors.append("Spontaneous onset (concerning)")
        elif trigger_assessment['pattern'] == 'positional':
            decision_factors.append("Positional triggers (suggests BPPV)")

        if hints_assessment['interpretation'] == 'central':
            decision_factors.append("HINTS suggests central etiology")
        elif hints_assessment['interpretation'] == 'peripheral':
            decision_factors.append("HINTS suggests peripheral etiology")

        reasoning['triggers'] = decision_factors
        reasoning['decision'] = f"TiTrATE assessment -> {tier}"

        return tier, confidence, reasoning

    def _assess_timing(self, patient_data: Dict) -> Dict:
        """
        Assess symptom timing relative to thrombolysis window.

        Critical distinction: Within 4.5 hours (thrombolysis candidate)
        vs beyond window (different management).
        """
        onset_hours = patient_data.get('symptom_onset_hours', 999)

        within_window = onset_hours <= self.thrombolysis_window_hours

        assessment = {
            'onset_hours': onset_hours,
            'within_window': within_window,
            'window_threshold': self.thrombolysis_window_hours
        }

        if within_window:
            assessment['interpretation'] = 'Thrombolysis candidate - urgent imaging required'
            assessment['urgency'] = 'high'
        else:
            assessment['interpretation'] = 'Beyond thrombolysis window - standard workup'
            assessment['urgency'] = 'moderate'

        return assessment

    def _assess_triggers(self, patient_data: Dict) -> Dict:
        """
        Identify trigger pattern: spontaneous vs positional.

        Key differential:
        - Spontaneous onset -> Vascular etiology (stroke) more likely
        - Positional triggers -> BPPV (benign) more likely
        """
        # Check for positional triggers
        positional_count = 0
        identified_triggers = []

        for trigger in self.positional_triggers:
            if patient_data.get(trigger, False):
                positional_count += 1
                identified_triggers.append(trigger.replace('_', ' '))

        # Determine pattern
        if positional_count >= 1:
            pattern = 'positional'
            interpretation = 'Positional triggers suggest benign peripheral vestibular disorder (BPPV)'
        else:
            pattern = 'spontaneous'
            interpretation = 'Spontaneous onset concerning for vascular etiology'

        assessment = {
            'pattern': pattern,
            'positional_triggers_count': positional_count,
            'identified_triggers': identified_triggers,
            'interpretation': interpretation
        }

        return assessment

    def _assess_hints(self, patient_data: Dict) -> Dict:
        """
        Apply HINTS protocol (Head Impulse, Nystagmus, Test of Skew).

        HINTS Protocol (Kattah et al., 2009):
        - More sensitive than early MRI for stroke detection
        - Central pattern -> Stroke likely
        - Peripheral pattern -> Vestibular neuritis likely

        Findings:
        - Head Impulse: Normal (central) vs Abnormal (peripheral)
        - Nystagmus: Direction-changing/vertical (central) vs Horizontal (peripheral)
        - Test of Skew: Positive (central) vs Negative (peripheral)
        """
        hints_findings = {}

        # Collect HINTS findings
        for component in self.hints_components:
            hints_findings[component] = patient_data.get(component, 'not_assessed')

        # Interpret findings
        central_signs = 0
        peripheral_signs = 0
        findings_list = []

        # Head Impulse Test
        head_impulse = hints_findings.get('hints_head_impulse', '')
        if head_impulse == 'normal':
            central_signs += 1
            findings_list.append("Normal head impulse (central sign)")
        elif head_impulse == 'abnormal':
            peripheral_signs += 1
            findings_list.append("Abnormal head impulse (peripheral sign)")

        # Nystagmus Pattern
        nystagmus = hints_findings.get('hints_nystagmus', '')
        if nystagmus in ['vertical', 'direction_changing', 'central']:
            central_signs += 1
            findings_list.append(f"{nystagmus} nystagmus (central sign)")
        elif nystagmus in ['horizontal', 'unidirectional', 'peripheral']:
            peripheral_signs += 1
            findings_list.append(f"{nystagmus} nystagmus (peripheral sign)")

        # Test of Skew
        test_of_skew = hints_findings.get('hints_test_of_skew', '')
        if test_of_skew == 'positive':
            central_signs += 1
            findings_list.append("Positive skew deviation (central sign)")
        elif test_of_skew == 'negative':
            peripheral_signs += 1
            findings_list.append("Negative skew deviation (peripheral sign)")

        # Overall interpretation
        if central_signs >= 1:  # Any central sign is concerning
            interpretation = 'central'
            explanation = 'HINTS protocol suggests central (stroke) etiology'
        elif peripheral_signs >= 2:
            interpretation = 'peripheral'
            explanation = 'HINTS protocol suggests peripheral vestibular disorder'
        else:
            interpretation = 'inconclusive'
            explanation = 'HINTS findings inconclusive - clinical correlation required'

        assessment = {
            'findings': hints_findings,
            'central_signs': central_signs,
            'peripheral_signs': peripheral_signs,
            'interpretation': interpretation,
            'explanation': explanation,
            'findings_list': findings_list
        }

        return assessment

    def _integrate_findings(
        self,
        timing: Dict,
        triggers: Dict,
        hints: Dict
    ) -> Tuple[RiskTier, float]:
        """
        Integrate TiTrATE components to determine final tier.

        Decision logic:
        1. Within window + central HINTS -> R1 (Critical)
        2. Within window + spontaneous -> R2 (High Risk)
        3. Beyond window + central HINTS -> R2 (High Risk)
        4. Positional + peripheral HINTS -> R4 (Low Risk - BPPV)
        5. Default -> R3 (Moderate)
        """
        within_window = timing['within_window']
        spontaneous = triggers['pattern'] == 'spontaneous'
        central_hints = hints['interpretation'] == 'central'
        peripheral_hints = hints['interpretation'] == 'peripheral'
        positional = triggers['pattern'] == 'positional'

        # Critical: Within window AND central findings
        if within_window and central_hints:
            return RiskTier.R1, 0.95

        # High Risk: Within window OR central findings
        if within_window and spontaneous:
            return RiskTier.R2, 0.90

        if central_hints:
            return RiskTier.R2, 0.85

        # Low Risk: Positional triggers AND peripheral findings (BPPV)
        if positional and peripheral_hints:
            return RiskTier.R4, 0.80

        # Moderate: Uncertain cases
        return RiskTier.R3, 0.70

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G4"

    def get_description(self) -> str:
        """Return gate description."""
        return "TiTrATE Clinical Logic (Timing, Triggers, Targeted Exam)"
