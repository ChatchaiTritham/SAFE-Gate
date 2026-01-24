"""
Gate 2: Moderate Risk Scoring

XGBoost-based weighted risk scoring for features that individually remain
insufficient for R1 classification but collectively elevate stroke probability.
"""

from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging.risk_lattice import RiskTier


class Gate2ModerateRisk:
    """
    Gate 2: Moderate Risk Scoring

    Evidence-weighted scoring of concerning features:
    - Cardiovascular risk factors (hypertension, AF, diabetes)
    - Symptom characteristics (sudden onset, severe imbalance)
    - Demographic factors (age > 65)

    Scoring thresholds:
    - Score >= 6.0 -> R1 (Critical)
    - Score 4.0-5.9 -> R2 (High Risk)
    - Score 2.0-3.9 -> R3 (Moderate)
    - Score < 2.0 -> R4-R5 (Low-Minimal)

    Note: Full implementation uses XGBoost (learning rate 0.03, max depth 5, 400 estimators).
    This version demonstrates the weighted scoring approach.
    """

    def __init__(self, use_xgboost: bool = False):
        """
        Initialize moderate risk scoring.

        Args:
            use_xgboost: Whether to use XGBoost model (requires trained model file)
        """
        self.use_xgboost = use_xgboost
        self.model = None

        # Feature weights (clinically motivated)
        # These approximate XGBoost feature importances
        self.cardiovascular_weights = {
            'hypertension': 1.2,
            'atrial_fibrillation': 1.8,  # Highest risk for stroke
            'diabetes': 0.9,
            'prior_stroke': 2.0,
            'coronary_artery_disease': 1.3
        }

        self.symptom_weights = {
            'sudden_onset': 1.5,
            'severe_imbalance': 1.1,
            'severe_vertigo': 0.8,
            'vomiting': 0.6,
            'headache': 0.7
        }

        self.demographic_weights = {
            'age_over_65': 0.8,
            'age_over_75': 1.2,
            'male_gender': 0.4
        }

        # Score thresholds mapping to risk tiers
        self.thresholds = {
            'R1': 6.0,   # >= 6.0 -> Critical
            'R2': 4.0,   # 4.0-5.9 -> High Risk
            'R3': 2.0,   # 2.0-3.9 -> Moderate
            'R4': 1.0,   # 1.0-1.9 -> Low Risk
            'R5': 0.0    # < 1.0 -> Minimal
        }

    def evaluate(self, patient_data: Dict) -> Tuple[RiskTier, float, Dict]:
        """
        Evaluate moderate risk through weighted scoring.

        Args:
            patient_data: Dictionary containing patient features

        Returns:
            Tuple of (risk_tier, confidence, reasoning)

        Algorithm:
            1. Calculate cardiovascular risk score
            2. Calculate symptom severity score
            3. Calculate demographic risk score
            4. Total score = sum of all components
            5. Map score to risk tier via thresholds
        """
        reasoning = {
            'gate': 'G2_Moderate_Risk',
            'mechanism': 'weighted_scoring',
            'components': {},
            'triggers': []
        }

        # Component 1: Cardiovascular risk factors
        cv_score = 0.0
        cv_factors = []

        for factor, weight in self.cardiovascular_weights.items():
            if patient_data.get(factor, False):
                cv_score += weight
                cv_factors.append(f"{factor} (+{weight})")

        reasoning['components']['cardiovascular'] = {
            'score': round(cv_score, 2),
            'factors': cv_factors
        }

        # Component 2: Symptom characteristics
        symptom_score = 0.0
        symptom_factors = []

        for symptom, weight in self.symptom_weights.items():
            if patient_data.get(symptom, False):
                symptom_score += weight
                symptom_factors.append(f"{symptom} (+{weight})")

        reasoning['components']['symptoms'] = {
            'score': round(symptom_score, 2),
            'factors': symptom_factors
        }

        # Component 3: Demographics
        demo_score = 0.0
        demo_factors = []

        age = patient_data.get('age', 0)
        if age >= 75:
            demo_score += self.demographic_weights['age_over_75']
            demo_factors.append(f"age {age} >= 75 (+{self.demographic_weights['age_over_75']})")
        elif age >= 65:
            demo_score += self.demographic_weights['age_over_65']
            demo_factors.append(f"age {age} >= 65 (+{self.demographic_weights['age_over_65']})")

        if patient_data.get('gender') == 'male':
            demo_score += self.demographic_weights['male_gender']
            demo_factors.append(f"male (+{self.demographic_weights['male_gender']})")

        reasoning['components']['demographics'] = {
            'score': round(demo_score, 2),
            'factors': demo_factors
        }

        # Total score
        total_score = cv_score + symptom_score + demo_score
        reasoning['total_score'] = round(total_score, 2)

        # Map score to risk tier
        if total_score >= self.thresholds['R1']:
            tier = RiskTier.R1
            confidence = 0.9
            reasoning['decision'] = f"Score {total_score:.1f} >= {self.thresholds['R1']} -> R1 (Critical)"
        elif total_score >= self.thresholds['R2']:
            tier = RiskTier.R2
            confidence = 0.85
            reasoning['decision'] = f"Score {total_score:.1f} in [{self.thresholds['R2']}, {self.thresholds['R1']}) -> R2 (High Risk)"
        elif total_score >= self.thresholds['R3']:
            tier = RiskTier.R3
            confidence = 0.8
            reasoning['decision'] = f"Score {total_score:.1f} in [{self.thresholds['R3']}, {self.thresholds['R2']}) -> R3 (Moderate)"
        elif total_score >= self.thresholds['R4']:
            tier = RiskTier.R4
            confidence = 0.75
            reasoning['decision'] = f"Score {total_score:.1f} in [{self.thresholds['R4']}, {self.thresholds['R3']}) -> R4 (Low Risk)"
        else:
            tier = RiskTier.R5
            confidence = 0.7
            reasoning['decision'] = f"Score {total_score:.1f} < {self.thresholds['R4']} -> R5 (Minimal)"

        # Record significant risk factors
        all_factors = cv_factors + symptom_factors + demo_factors
        if all_factors:
            reasoning['triggers'] = all_factors

        return tier, confidence, reasoning

    def load_xgboost_model(self, model_path: str):
        """
        Load pre-trained XGBoost model.

        Args:
            model_path: Path to saved XGBoost model (.pkl file)

        Model specifications (from paper):
        - Learning rate: 0.03
        - Max depth: 5
        - Number of estimators: 400
        - Trained on 4,800 synthetic cases
        """
        try:
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.use_xgboost = True
            return True
        except Exception as e:
            print(f"Warning: Could not load XGBoost model: {e}")
            print("Falling back to weighted scoring approach")
            return False

    def get_name(self) -> str:
        """Return gate identifier."""
        return "G2"

    def get_description(self) -> str:
        """Return gate description."""
        method = "XGBoost" if self.use_xgboost else "Weighted Scoring"
        return f"Moderate Risk Scoring ({method})"
