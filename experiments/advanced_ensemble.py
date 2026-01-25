#!/usr/bin/env python3
"""
Advanced Ensemble Methods for SAFE-Gate
Combines multiple models to improve performance beyond single models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class AdvancedEnsemble:
    """
    Advanced ensemble methods for SAFE-Gate.

    Strategies:
    1. Stacking - Meta-learner on top of base models
    2. Weighted Voting - Optimized weights for each model
    3. Cascade Ensemble - Sequential refinement
    """

    def __init__(self):
        self.models = {}
        self.ensemble = None

    def build_stacking_ensemble(self):
        """
        Stacking ensemble with diverse base learners.

        Base Models:
        - XGBoost (gradient boosting)
        - LightGBM (fast gradient boosting)
        - Random Forest (bagging)
        - SVM (kernel-based)

        Meta-learner:
        - Logistic Regression
        """
        base_learners = [
            ('xgb', XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )),
            ('lgbm', LGBMClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                verbose=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ]

        # Meta-learner
        meta_learner = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )

        # Stacking
        self.ensemble = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

        return self.ensemble

    def build_weighted_voting(self, weights=None):
        """
        Weighted voting ensemble.

        Args:
            weights: Custom weights for each model (auto-optimized if None)
        """
        if weights is None:
            # Default: equal weights
            weights = [1.0, 1.0, 1.0, 1.0]

        estimators = [
            ('xgb', XGBClassifier(max_depth=6, n_estimators=100, random_state=42)),
            ('lgbm', LGBMClassifier(max_depth=6, n_estimators=100, random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]

        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability predictions
            weights=weights,
            n_jobs=-1
        )

        return self.ensemble

    def optimize_voting_weights(self, X_train, y_train, X_val, y_val):
        """
        Optimize voting weights using grid search.

        Returns:
            Best weights and corresponding F1-score
        """
        from itertools import product

        # Train individual models first
        models = [
            XGBClassifier(max_depth=6, n_estimators=100, random_state=42),
            LGBMClassifier(max_depth=6, n_estimators=100, random_state=42, verbose=-1),
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        ]

        # Get predictions
        predictions = []
        for model in models:
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_val)
            predictions.append(pred_proba)

        # Grid search for optimal weights
        weight_options = [0.5, 1.0, 1.5, 2.0]
        best_f1 = 0
        best_weights = None

        print("Optimizing voting weights...")
        for weights in product(weight_options, repeat=len(models)):
            # Weighted average of probabilities
            weighted_proba = np.average(predictions, axis=0, weights=weights)
            y_pred = np.argmax(weighted_proba, axis=1)

            # Calculate F1
            f1 = f1_score(y_val, y_pred, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights

        print(f"Best weights: {best_weights}")
        print(f"Best F1-Score: {best_f1:.4f}")

        return best_weights, best_f1

    def build_cascade_ensemble(self):
        """
        Cascade ensemble for sequential refinement.

        Stage 1: Fast models (RF, XGB) - Initial classification
        Stage 2: Deep models (GB, SVM) - Refinement on uncertain cases
        """
        class CascadeEnsemble:
            def __init__(self):
                # Stage 1: Fast models
                self.stage1 = VotingClassifier([
                    ('xgb', XGBClassifier(max_depth=4, n_estimators=50, random_state=42)),
                    ('rf', RandomForestClassifier(max_depth=8, n_estimators=50, random_state=42))
                ], voting='soft')

                # Stage 2: Deep models (for uncertain cases)
                self.stage2 = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42
                )

                self.confidence_threshold = 0.7

            def fit(self, X, y):
                self.stage1.fit(X, y)
                self.stage2.fit(X, y)
                return self

            def predict(self, X):
                # Stage 1: Get predictions and confidence
                proba = self.stage1.predict_proba(X)
                max_proba = np.max(proba, axis=1)
                pred_stage1 = self.stage1.predict(X)

                # Stage 2: Refine uncertain predictions
                uncertain_mask = max_proba < self.confidence_threshold
                if uncertain_mask.any():
                    pred_stage1[uncertain_mask] = self.stage2.predict(X[uncertain_mask])

                return pred_stage1

        return CascadeEnsemble()

    def evaluate_ensemble(self, ensemble, X_train, y_train, X_test, y_test):
        """Evaluate ensemble performance."""
        print("\nTraining ensemble...")
        ensemble.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = ensemble.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy, f1


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    ensemble_builder = AdvancedEnsemble()

    # 1. Stacking Ensemble
    print("="*60)
    print("1. Stacking Ensemble")
    print("="*60)
    stacking = ensemble_builder.build_stacking_ensemble()
    acc1, f1_1 = ensemble_builder.evaluate_ensemble(
        stacking, X_train, y_train, X_test, y_test
    )

    # 2. Optimized Weighted Voting
    print("\n" + "="*60)
    print("2. Weighted Voting Ensemble (Optimized)")
    print("="*60)
    best_weights, val_f1 = ensemble_builder.optimize_voting_weights(
        X_train, y_train, X_val, y_val
    )
    voting = ensemble_builder.build_weighted_voting(weights=best_weights)
    acc2, f1_2 = ensemble_builder.evaluate_ensemble(
        voting, X_train, y_train, X_test, y_test
    )

    # 3. Cascade Ensemble
    print("\n" + "="*60)
    print("3. Cascade Ensemble")
    print("="*60)
    cascade = ensemble_builder.build_cascade_ensemble()
    acc3, f1_3 = ensemble_builder.evaluate_ensemble(
        cascade, X_train, y_train, X_test, y_test
    )

    # Summary
    print("\n" + "="*60)
    print("ENSEMBLE COMPARISON")
    print("="*60)
    print(f"Stacking:         F1={f1_1:.4f}, Acc={acc1:.4f}")
    print(f"Weighted Voting:  F1={f1_2:.4f}, Acc={acc2:.4f}")
    print(f"Cascade:          F1={f1_3:.4f}, Acc={acc3:.4f}")
    print(f"\nBest Ensemble: ", end="")
    best_idx = np.argmax([f1_1, f1_2, f1_3])
    best_names = ['Stacking', 'Weighted Voting', 'Cascade']
    print(f"{best_names[best_idx]} (F1={max(f1_1, f1_2, f1_3):.4f})")
