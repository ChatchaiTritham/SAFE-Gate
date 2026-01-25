#!/usr/bin/env python3
"""
Hyperparameter Optimization for SAFE-Gate using Optuna
Improves model performance through automated parameter search
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class HyperparameterTuner:
    """Automated hyperparameter tuning for SAFE-Gate Gate2 (XGBoost)."""

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = None
        self.best_score = 0

    def objective(self, trial):
        """Optuna objective function."""
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }

        # Train model
        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )

        # Cross-validation
        f1_scorer = make_scorer(f1_score, average='macro')
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring=f1_scorer, n_jobs=-1
        )

        return scores.mean()

    def optimize(self, n_trials=100):
        """Run optimization."""
        print(f"Starting hyperparameter optimization ({n_trials} trials)...")

        study = optuna.create_study(
            direction='maximize',
            study_name='safegate_gate2',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[self._callback]
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"\nOptimization complete!")
        print(f"Best F1-Score: {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")

        return study

    def _callback(self, study, trial):
        """Callback for progress tracking."""
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: F1={trial.value:.4f}")

    def train_best_model(self):
        """Train final model with best parameters."""
        model = xgb.XGBClassifier(
            **self.best_params,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        model.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred = model.predict(self.X_val)
        val_f1 = f1_score(self.y_val, y_pred, average='macro')

        print(f"\nValidation F1-Score: {val_f1:.4f}")
        return model


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=5,
        random_state=42
    )

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optimize
    tuner = HyperparameterTuner(X_train, y_train, X_val, y_val)
    study = tuner.optimize(n_trials=50)

    # Train best model
    best_model = tuner.train_best_model()

    # Save results
    import json
    with open('experiments/best_hyperparameters.json', 'w') as f:
        json.dump({
            'best_params': tuner.best_params,
            'best_score': float(tuner.best_score)
        }, f, indent=2)

    print("\n✓ Best hyperparameters saved to experiments/best_hyperparameters.json")
