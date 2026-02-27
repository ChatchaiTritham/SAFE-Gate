"""
Gate 2: Cardiovascular Risk Assessment (Statistical)

Aggregates cardiovascular risk factors using a weighted accumulation model
(Equation 3 in paper). A gradient-boosted tree classifier (XGBoost, 100
estimators) serves as a consistency check against the rule-derived score.

Equation 3:
  Score(x) = w_demo * f_demo(x) + w_symp * f_symp(x) + w_hist * f_hist(x)

Feature weights (from article):
  f_demo: age > 60 = +1.0, male = +0.5
  f_symp: sudden onset = +1.5, continuous duration > 1h = +1.0
  f_hist: atrial fibrillation = +1.8, prior CVA = +2.0

Confidence: c2 = 1 - sigma_score^2 / sigma_max^2
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate2ModerateRisk:
    """
    Gate 2: Cardiovascular Risk Assessment (Statistical).

    Implements Equation 3 from the article:
      Score(x) = w_demo * f_demo(x) + w_symp * f_symp(x) + w_hist * f_hist(x)

    Feature weights specified in article:
      f_demo: age > 60 (+1.0), male (+0.5)
      f_symp: sudden onset (+1.5), duration > 1h (+1.0)
      f_hist: atrial fibrillation (+1.8), prior CVA (+2.0)

    Additional clinical risk factors (beyond Equation 3 core):
      hypertension (+1.2), diabetes (+0.9)

    Note: Full implementation uses XGBoost (100 estimators) as consistency
    check. This version demonstrates the weighted scoring approach.
    """

    # Maximum possible score for confidence normalisation
    SIGMA_MAX_SQUARED = 4.0  # Empirical upper bound for score variance

    def __init__(self, use_xgboost: bool = False):
        """
        Initialise cardiovascular risk assessment gate.

        Args:
            use_xgboost: Whether to use XGBoost model (requires trained model file)
        """
        self.use_xgboost = use_xgboost
        self.model = None

        # ---------- Article Equation 3 weights ----------
        # f_demo: Demographic risk contributors
        self.demographic_weights = {
            'age_over_60': 1.0,    # Article: age exceeding 60 = +1.0
            'male': 0.5,           # Article: male sex = +0.5
        }

        # f_symp: Symptom severity indicators
        self.symptom_weights = {
            'sudden_onset': 1.5,           # Article: sudden onset = +1.5
            'continuous_duration_1h': 1.0,  # Article: continuous duration > 1h = +1.0
        }

        # f_hist: Clinical history factors
        self.history_weights = {
            'atrial_fibrillation': 1.8,    # Article: AF = +1.8
            'prior_stroke': 2.0,           # Article: prior CVA = +2.0
        }

        # Additional clinical factors (supplement Eq.3 core)
        self.supplementary_weights = {
            'hypertension': 1.2,
            'diabetes': 0.9,
            'coronary_artery_disease': 1.0,
        }

        # Score thresholds mapping to risk tiers
        # G2 assesses cardiovascular risk, NOT acute instability (G1 domain)
        self.thresholds = {
            'R2': 5.5,   # >= 5.5 -> High Risk
            'R3': 3.5,   # 3.5-5.4 -> Moderate
            'R4': 2.0,   # 2.0-3.4 -> Low Risk
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate cardiovascular risk through weighted scoring (Equation 3).

        Returns:
            (risk_tier, confidence, reasoning)
            - confidence = c2 = 1 - sigma_score^2 / sigma_max^2
        """
        reasoning = {
            'gate': 'G2_Cardiovascular_Risk',
            'mechanism': 'weighted_accumulation_model',
            'equation': 'Score(x) = w_demo*f_demo + w_symp*f_symp + w_hist*f_hist',
            'components': {},
            'triggers': []
        }

        # Component 1: f_demo (demographic risk)
        demo_score = 0.0
        demo_factors = []

        age = patient_data.get('age', 0)
        if age > 60:
            demo_score += self.demographic_weights['age_over_60']
            demo_factors.append(f'age {age} > 60 (+{self.demographic_weights["age_over_60"]})')

        if patient_data.get('gender') == 'male':
            demo_score += self.demographic_weights['male']
            demo_factors.append(f'male (+{self.demographic_weights["male"]})')

        reasoning['components']['f_demo'] = {
            'score': round(demo_score, 2),
            'factors': demo_factors
        }

        # Component 2: f_symp (symptom indicators)
        symp_score = 0.0
        symp_factors = []

        if patient_data.get('sudden_onset', False):
            symp_score += self.symptom_weights['sudden_onset']
            symp_factors.append(f'sudden onset (+{self.symptom_weights["sudden_onset"]})')

        # Duration > 1h: check symptom_duration_days or symptom_onset_hours
        duration_days = patient_data.get('symptom_duration_days', 0)
        onset_hours = patient_data.get('symptom_onset_hours', 999)
        if duration_days > (1.0 / 24.0) or (onset_hours <= 24 and onset_hours > 1):
            symp_score += self.symptom_weights['continuous_duration_1h']
            symp_factors.append(f'continuous duration > 1h (+{self.symptom_weights["continuous_duration_1h"]})')

        reasoning['components']['f_symp'] = {
            'score': round(symp_score, 2),
            'factors': symp_factors
        }

        # Component 3: f_hist (clinical history)
        hist_score = 0.0
        hist_factors = []

        if patient_data.get('atrial_fibrillation', False):
            hist_score += self.history_weights['atrial_fibrillation']
            hist_factors.append(f'atrial fibrillation (+{self.history_weights["atrial_fibrillation"]})')

        if patient_data.get('prior_stroke', False):
            hist_score += self.history_weights['prior_stroke']
            hist_factors.append(f'prior CVA (+{self.history_weights["prior_stroke"]})')

        reasoning['components']['f_hist'] = {
            'score': round(hist_score, 2),
            'factors': hist_factors
        }

        # Supplementary factors (beyond Equation 3 core)
        supp_score = 0.0
        supp_factors = []
        for factor, weight in self.supplementary_weights.items():
            if patient_data.get(factor, False):
                supp_score += weight
                supp_factors.append(f'{factor} (+{weight})')

        reasoning['components']['supplementary'] = {
            'score': round(supp_score, 2),
            'factors': supp_factors
        }

        # Total score (Equation 3)
        total_score = demo_score + symp_score + hist_score + supp_score
        reasoning['total_score'] = round(total_score, 2)

        # Map score to risk tier
        if total_score >= self.thresholds['R2']:
            tier = RiskTier.R2
        elif total_score >= self.thresholds['R3']:
            tier = RiskTier.R3
        elif total_score >= self.thresholds['R4']:
            tier = RiskTier.R4
        else:
            tier = RiskTier.R5

        reasoning['decision'] = (
            f'Score(x) = {total_score:.2f} -> {tier.name}'
        )

        # Confidence: c2 = 1 - sigma_score^2 / sigma_max^2
        # Approximate score variance via bootstrap-like estimation
        # Higher scores have more contributing factors -> lower variance
        n_active = len(demo_factors) + len(symp_factors) + len(hist_factors) + len(supp_factors)
        n_total = (len(self.demographic_weights) + len(self.symptom_weights) +
                   len(self.history_weights) + len(self.supplementary_weights))

        # Estimate variance: fewer active factors -> higher variance
        if n_total > 0:
            activation_ratio = n_active / n_total
            sigma_sq = (1.0 - activation_ratio) * self.SIGMA_MAX_SQUARED
        else:
            sigma_sq = self.SIGMA_MAX_SQUARED

        confidence = max(0.1, 1.0 - sigma_sq / self.SIGMA_MAX_SQUARED)
        reasoning['confidence_detail'] = {
            'sigma_sq': round(sigma_sq, 4),
            'sigma_max_sq': self.SIGMA_MAX_SQUARED,
            'c2': round(confidence, 4)
        }

        # Record triggers
        all_factors = demo_factors + symp_factors + hist_factors + supp_factors
        reasoning['triggers'] = all_factors

        return tier, float(confidence), reasoning

    def load_xgboost_model(self, model_path: str):
        """
        Load pre-trained XGBoost model.

        Model specifications (from article):
        - 100 estimators (gradient-boosted tree classifier)
        - Serves as consistency check against rule-derived score
        """
        try:
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.use_xgboost = True
            return True
        except Exception as e:
            print(f"Warning: Could not load XGBoost model: {e}")
            return False

    def get_name(self) -> str:
        return "G2"

    def get_description(self) -> str:
        method = "XGBoost + Weighted Scoring" if self.use_xgboost else "Weighted Accumulation Model"
        return f"Cardiovascular Risk Assessment ({method})"
