"""
Gate 4: Clinical Syndrome Pattern Matching (Rule-Based)

Operationalises the TiTrATE diagnostic framework (Newman-Toker & Edlow 2008)
by matching incoming presentations against characterised benign vestibular
syndromes -- BPPV, vestibular neuritis, and Meniere disease -- through
weighted Hamming distance.

Tier mapping (from article Section 3.6):
  similarity >= 0.85 AND no red flags --> R5 (safe discharge)
  0.60 <= similarity < 0.85           --> R3 (moderate risk)
  similarity < 0.60                   --> R2 (high-risk escalation)

Confidence: c4 = max_s similarity(x, s)
"""

import numpy as np
from typing import Dict, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate4TiTrATELogic:
    """
    Gate 4: Clinical Syndrome Pattern Matching (Rule-Based).

    Implements weighted Hamming distance between patient presentation
    and three reference benign vestibular syndrome profiles:
      - BPPV (Benign Paroxysmal Positional Vertigo)
      - Vestibular Neuritis
      - Meniere Disease

    similarity(x, s) = 1 - d_w(x, s) / sum(w)
    d_w(x, s) = sum_j w_j * I(x_j != s_j)

    Tier assignment:
      similarity >= 0.85 AND no red flags --> R5
      0.60 <= similarity < 0.85           --> R3
      similarity < 0.60                   --> R2

    Confidence: c4 = max_s similarity(x, s)
    """

    # Thresholds from article Section 3.6
    HIGH_SIMILARITY = 0.85
    MODERATE_SIMILARITY = 0.60

    def __init__(self):
        """Initialise syndrome reference profiles and feature weights."""
        # ---- Feature weights (clinical discriminative value) ----
        # Higher weight = stronger discriminator for syndrome matching
        self.feature_weights = {
            # HINTS protocol (highest discriminative power)
            'hints_head_impulse': 3.0,     # abnormal=peripheral, normal=central
            'hints_nystagmus': 3.0,        # horizontal=peripheral, vertical/direction-changing=central
            'hints_test_of_skew': 3.0,     # negative=peripheral, positive=central
            # Trigger pattern (strong BPPV discriminator)
            'positional_triggers': 2.5,    # present=BPPV, absent=other
            'sudden_onset': 2.0,           # True for neuritis, False for Meniere gradual
            # Duration / timing
            'symptom_duration_days': 2.0,  # seconds-minutes=BPPV, hours-days=neuritis, episodic=Meniere
            'symptom_onset_hours': 1.5,    # recent vs chronic
            'episodic_pattern': 2.0,       # recurrent episodes = Meniere
            # Auditory symptoms (Meniere discriminator)
            'hearing_loss': 2.5,           # present=Meniere, absent=BPPV/neuritis
            'tinnitus': 2.0,              # present=Meniere
            'aural_fullness': 2.0,        # present=Meniere
            # Associated features
            'nausea_vomiting': 1.0,        # common in all, low discriminant
            'vertigo_severity': 1.0,       # severity (1-10 scale)
            'progression_pattern': 1.5,    # improving/stable/worsening
        }

        # ---- Reference syndrome profiles ----
        # Each profile defines expected values for clinical features.
        # For continuous features, we define expected ranges.

        self.syndrome_profiles = {
            'BPPV': {
                'hints_head_impulse': 'normal',       # or not assessable (not triggered)
                'hints_nystagmus': 'peripheral',      # torsional, positional
                'hints_test_of_skew': 'negative',
                'positional_triggers': True,
                'sudden_onset': True,
                'symptom_duration_days': (0.0, 0.08), # seconds to minutes (<2 hours)
                'symptom_onset_hours': (0.0, 999),    # any onset time
                'episodic_pattern': True,              # recurrent brief episodes
                'hearing_loss': False,
                'tinnitus': False,
                'aural_fullness': False,
                'nausea_vomiting': True,               # common with positional
                'vertigo_severity': (3, 8),            # moderate-severe but brief
                'progression_pattern': 'stable',       # self-limiting episodes
            },
            'Vestibular_Neuritis': {
                'hints_head_impulse': 'abnormal',     # positive head thrust
                'hints_nystagmus': 'peripheral',      # horizontal, unidirectional
                'hints_test_of_skew': 'negative',
                'positional_triggers': False,
                'sudden_onset': True,                  # acute onset
                'symptom_duration_days': (1.0, 14.0), # days to 2 weeks
                'symptom_onset_hours': (0.0, 72.0),   # recent onset
                'episodic_pattern': False,             # single prolonged episode
                'hearing_loss': False,                 # preserved hearing
                'tinnitus': False,
                'aural_fullness': False,
                'nausea_vomiting': True,               # prominent
                'vertigo_severity': (5, 10),           # severe
                'progression_pattern': 'improving',    # gradual improvement
            },
            'Meniere_Disease': {
                'hints_head_impulse': 'normal',       # between attacks
                'hints_nystagmus': 'peripheral',      # during attack
                'hints_test_of_skew': 'negative',
                'positional_triggers': False,
                'sudden_onset': False,                 # gradual build-up
                'symptom_duration_days': (0.08, 1.0), # 20 minutes to hours
                'symptom_onset_hours': (0.0, 999),    # variable
                'episodic_pattern': True,              # hallmark recurrent episodes
                'hearing_loss': True,                  # fluctuating, low-frequency
                'tinnitus': True,                      # characteristic
                'aural_fullness': True,                # characteristic triad
                'nausea_vomiting': True,
                'vertigo_severity': (4, 9),
                'progression_pattern': 'episodic',     # waxing/waning
            }
        }

        # Red flag features that override benign classification
        self.red_flag_features = [
            'dysarthria', 'ataxia', 'diplopia',
            'facial_weakness', 'limb_weakness',
            'crossed_sensory_loss', 'vertical_skew_deviation',
            'thunderclap_headache', 'worst_headache_ever',
        ]

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate patient using TiTrATE syndrome pattern matching.

        Computes weighted Hamming distance to each reference syndrome,
        selects the best match, and applies threshold-based tier assignment.

        Returns:
            (tier, confidence, reasoning)
            - confidence = max_s similarity(x, s)
        """
        reasoning = {
            'gate': 'G4_TiTrATE_Logic',
            'mechanism': 'weighted_hamming_distance',
            'syndrome_similarities': {},
            'best_match': None,
            'red_flags_present': [],
        }

        # Step 1: Compute similarity to each syndrome profile
        similarities = {}
        for syndrome_name, profile in self.syndrome_profiles.items():
            sim, detail = self._compute_similarity(patient_data, profile)
            similarities[syndrome_name] = sim
            reasoning['syndrome_similarities'][syndrome_name] = {
                'similarity': round(sim, 4),
                'detail': detail,
            }

        # Step 2: Find best matching syndrome
        best_syndrome = max(similarities, key=similarities.get)
        best_similarity = similarities[best_syndrome]
        reasoning['best_match'] = best_syndrome
        reasoning['best_similarity'] = round(best_similarity, 4)

        # Step 3: Check for red flags
        red_flags = []
        for flag in self.red_flag_features:
            if patient_data.get(flag, False):
                red_flags.append(flag.replace('_', ' '))
        # Also check HINTS central signs
        if patient_data.get('hints_nystagmus') in ['central', 'vertical', 'direction_changing']:
            red_flags.append('central nystagmus pattern')
        if patient_data.get('hints_test_of_skew') == 'positive':
            red_flags.append('positive test of skew')
        if patient_data.get('hints_head_impulse') == 'normal' and \
           patient_data.get('hints_nystagmus') in ['central', 'vertical', 'direction_changing']:
            red_flags.append('normal HIT with central nystagmus (HINTS central)')

        reasoning['red_flags_present'] = red_flags
        has_red_flags = len(red_flags) > 0

        # Step 4: Tier assignment per article thresholds
        if best_similarity >= self.HIGH_SIMILARITY and not has_red_flags:
            tier = RiskTier.R5
            reasoning['decision'] = (
                f'High similarity to {best_syndrome} ({best_similarity:.2f} >= '
                f'{self.HIGH_SIMILARITY}), no red flags. Safe discharge tier.'
            )
        elif best_similarity >= self.MODERATE_SIMILARITY:
            tier = RiskTier.R3
            if has_red_flags:
                reasoning['decision'] = (
                    f'Moderate similarity to {best_syndrome} ({best_similarity:.2f}), '
                    f'but red flags present: {", ".join(red_flags)}. '
                    f'Assigned R3 despite benign pattern match.'
                )
            else:
                reasoning['decision'] = (
                    f'Moderate similarity to {best_syndrome} ({best_similarity:.2f}) '
                    f'in [{self.MODERATE_SIMILARITY}, {self.HIGH_SIMILARITY}). '
                    f'Moderate risk tier.'
                )
        else:
            tier = RiskTier.R2
            reasoning['decision'] = (
                f'Low similarity to all syndromes (best: {best_syndrome} = '
                f'{best_similarity:.2f} < {self.MODERATE_SIMILARITY}). '
                f'High-risk escalation.'
            )

        # Confidence = max_s similarity(x, s)
        confidence = best_similarity

        return tier, confidence, reasoning

    def _compute_similarity(
        self,
        patient_data: Dict,
        profile: Dict
    ) -> Tuple[float, Dict]:
        """
        Compute weighted Hamming similarity between patient and syndrome profile.

        similarity(x, s) = 1 - d_w(x, s) / sum(w_active)
        d_w(x, s) = sum_j w_j * I(x_j != s_j)

        Only features present in both patient data and profile contribute.

        Returns:
            (similarity_score, feature_detail)
        """
        total_weight = 0.0
        weighted_distance = 0.0
        detail = {}

        for feature, expected in profile.items():
            weight = self.feature_weights.get(feature, 1.0)
            patient_val = patient_data.get(feature, None)

            # Skip features not present in patient data
            if patient_val is None:
                detail[feature] = {'status': 'missing', 'match': None, 'weight': weight}
                continue

            total_weight += weight
            match = self._feature_match(patient_val, expected)

            if not match:
                weighted_distance += weight
                detail[feature] = {'status': 'mismatch', 'match': False,
                                   'patient': str(patient_val), 'expected': str(expected),
                                   'weight': weight}
            else:
                detail[feature] = {'status': 'match', 'match': True,
                                   'patient': str(patient_val), 'expected': str(expected),
                                   'weight': weight}

        # Compute similarity
        if total_weight > 0:
            similarity = 1.0 - (weighted_distance / total_weight)
        else:
            similarity = 0.0  # No features to compare

        return similarity, detail

    def _feature_match(self, patient_val, expected) -> bool:
        """
        Check if a patient feature value matches the expected syndrome value.

        Handles:
        - Boolean features: exact match
        - Categorical features: exact match or compatible values
        - Continuous features (tuple range): check if patient value falls in range
        - String features: case-insensitive match with synonym handling
        """
        # Tuple = expected range for continuous feature
        if isinstance(expected, tuple) and len(expected) == 2:
            try:
                val = float(patient_val)
                return expected[0] <= val <= expected[1]
            except (ValueError, TypeError):
                return False

        # Boolean comparison
        if isinstance(expected, bool):
            if isinstance(patient_val, bool):
                return patient_val == expected
            # Handle string representations
            if isinstance(patient_val, str):
                truthy = patient_val.lower() in ('true', 'yes', '1', 'present', 'positive')
                return truthy == expected
            # Handle numeric (1/0)
            try:
                return bool(int(patient_val)) == expected
            except (ValueError, TypeError):
                return False

        # String comparison (categorical)
        if isinstance(expected, str) and isinstance(patient_val, str):
            # Direct match
            if patient_val.lower() == expected.lower():
                return True
            # Synonym handling for nystagmus
            peripheral_synonyms = {'peripheral', 'horizontal', 'unidirectional'}
            central_synonyms = {'central', 'vertical', 'direction_changing'}
            if expected.lower() in peripheral_synonyms and patient_val.lower() in peripheral_synonyms:
                return True
            if expected.lower() in central_synonyms and patient_val.lower() in central_synonyms:
                return True
            # Progression pattern synonyms
            if expected.lower() in ('stable', 'episodic') and patient_val.lower() in ('stable', 'episodic'):
                return True
            return False

        # Fallback: equality
        return patient_val == expected

    def get_name(self) -> str:
        return "G4"

    def get_description(self) -> str:
        return "Clinical Syndrome Pattern Matching (TiTrATE, weighted Hamming distance)"
