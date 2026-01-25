#!/usr/bin/env python3
"""
Automated Feature Engineering for SAFE-Gate
Improves model performance through intelligent feature transformation
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class AutomatedFeatureEngineering:
    """
    Automated feature engineering pipeline.

    Techniques:
    1. Feature selection (remove irrelevant features)
    2. Feature interaction (create new features)
    3. Dimensionality reduction (PCA)
    4. Feature importance analysis
    """

    def __init__(self):
        self.selected_features = None
        self.feature_importance = None
        self.pca = None
        self.poly = None

    def select_features_univariate(self, X, y, k=10):
        """
        Univariate feature selection using ANOVA F-statistic.

        Args:
            X: Features
            y: Labels
            k: Number of top features to select

        Returns:
            Selected features
        """
        print(f"\n[1/4] Univariate Feature Selection (k={k})...")

        # F-statistic
        selector_f = SelectKBest(f_classif, k=k)
        selector_f.fit(X, y)

        # Mutual information
        selector_mi = SelectKBest(mutual_info_classif, k=k)
        selector_mi.fit(X, y)

        # Get scores
        f_scores = selector_f.scores_
        mi_scores = selector_mi.scores_

        # Create DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        df_scores = pd.DataFrame({
            'feature': feature_names,
            'f_score': f_scores,
            'mi_score': mi_scores
        })

        df_scores = df_scores.sort_values('f_score', ascending=False)

        print("\nTop 10 Features (F-statistic):")
        print(df_scores.head(10).to_string(index=False))

        # Select top k features
        self.selected_features = df_scores.head(k)['feature'].tolist()

        return self.selected_features, df_scores

    def select_features_rfe(self, X, y, n_features=10, step=1):
        """
        Recursive Feature Elimination with Cross-Validation.

        Args:
            X: Features
            y: Labels
            n_features: Target number of features
            step: Number of features to remove at each iteration

        Returns:
            Selected features
        """
        print(f"\n[2/4] Recursive Feature Elimination (target={n_features})...")

        # Base estimator
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )

        # RFE with CV
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        selector.fit(X, y)

        # Get selected features
        if isinstance(X, pd.DataFrame):
            selected = X.columns[selector.support_].tolist()
        else:
            selected = [f'feature_{i}' for i in np.where(selector.support_)[0]]

        print(f"Optimal number of features: {selector.n_features_}")
        print(f"Selected features: {selected[:10]}...")  # Show first 10

        return selected, selector

    def create_interaction_features(self, X, degree=2, interaction_only=True):
        """
        Create polynomial and interaction features.

        Args:
            X: Original features
            degree: Polynomial degree (2 = quadratic interactions)
            interaction_only: If True, only a*b (not a^2)

        Returns:
            Augmented feature matrix
        """
        print(f"\n[3/4] Creating Interaction Features (degree={degree})...")

        self.poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )

        X_poly = self.poly.fit_transform(X)

        print(f"Original features: {X.shape[1]}")
        print(f"After interaction: {X_poly.shape[1]}")
        print(f"New features created: {X_poly.shape[1] - X.shape[1]}")

        return X_poly

    def reduce_dimensions_pca(self, X, n_components=0.95):
        """
        Dimensionality reduction using PCA.

        Args:
            X: Features
            n_components: Number of components or variance to preserve

        Returns:
            Transformed features
        """
        print(f"\n[4/4] PCA Dimensionality Reduction...")

        # Standardize first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        print(f"Original dimensions: {X.shape[1]}")
        print(f"Reduced dimensions: {X_pca.shape[1]}")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")

        return X_pca

    def analyze_feature_importance(self, X, y, top_k=20):
        """
        Feature importance analysis using Random Forest.

        Args:
            X: Features
            y: Labels
            top_k: Number of top features to display

        Returns:
            Feature importance DataFrame
        """
        print(f"\n[BONUS] Feature Importance Analysis...")

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Get importance
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop {top_k} Most Important Features:")
        print(importance_df.head(top_k).to_string(index=False))

        # Visualize
        self._plot_feature_importance(importance_df.head(top_k))

        self.feature_importance = importance_df
        return importance_df

    def _plot_feature_importance(self, df, save_path='experiments/feature_importance.png'):
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='importance', y='feature', palette='viridis')
        plt.xlabel('Importance', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title('Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✓ Plot saved to {save_path}")
        plt.close()

    def build_optimal_pipeline(self, X_train, y_train):
        """
        Build complete feature engineering pipeline.

        Steps:
        1. Feature selection (top features)
        2. Feature interaction (polynomial)
        3. Dimensionality reduction (PCA)

        Returns:
            Transformed training data
        """
        print("\n" + "="*60)
        print("AUTOMATED FEATURE ENGINEERING PIPELINE")
        print("="*60)

        # Step 1: Feature importance analysis
        importance_df = self.analyze_feature_importance(X_train, y_train)

        # Step 2: Select top features (50%)
        n_features = max(10, int(X_train.shape[1] * 0.5))
        top_features = importance_df.head(n_features)['feature'].tolist()

        if isinstance(X_train, pd.DataFrame):
            X_selected = X_train[top_features]
        else:
            X_selected = X_train[:, :n_features]

        print(f"\n→ Selected {len(top_features)} features")

        # Step 3: Create interactions (selective)
        # Only create interactions for top 20 features to avoid explosion
        top_20 = top_features[:min(20, len(top_features))]
        if isinstance(X_train, pd.DataFrame):
            X_top = X_train[top_20].values
        else:
            X_top = X_train[:, :20]

        X_inter = self.create_interaction_features(X_top, degree=2)

        # Step 4: PCA for dimension reduction
        X_final = self.reduce_dimensions_pca(X_inter, n_components=0.95)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Original features: {X_train.shape[1]}")
        print(f"Selected features: {len(top_features)}")
        print(f"After interactions: {X_inter.shape[1]}")
        print(f"Final (PCA): {X_final.shape[1]}")

        return X_final


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=15,
        n_classes=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize
    fe = AutomatedFeatureEngineering()

    # Build pipeline
    X_train_transformed = fe.build_optimal_pipeline(X_train, y_train)

    # Test baseline vs optimized
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    # Baseline: Original features
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    f1_baseline = f1_score(y_test, y_pred_baseline, average='macro')

    print(f"Baseline (original features):  F1 = {f1_baseline:.4f}")

    # Optimized: Engineered features
    # Transform test data
    importance_df = fe.feature_importance
    top_features = importance_df.head(25)['feature'].tolist()
    X_test_selected = X_test[:, :25]
    X_test_inter = fe.poly.transform(X_test_selected[:, :20])
    X_test_transformed = fe.pca.transform(X_test_inter)

    clf_optimized = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_optimized.fit(X_train_transformed, y_train)
    y_pred_optimized = clf_optimized.predict(X_test_transformed)
    f1_optimized = f1_score(y_test, y_pred_optimized, average='macro')

    print(f"Optimized (engineered features): F1 = {f1_optimized:.4f}")

    improvement = (f1_optimized - f1_baseline) / f1_baseline * 100
    print(f"\nImprovement: {improvement:+.2f}%")
