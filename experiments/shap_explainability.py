#!/usr/bin/env python3
"""
SHAP-based Explainability for SAFE-Gate with Game Theory Foundation

SHAP (SHapley Additive exPlanations) เชื่อมโยงกับ Game Theory:
- Shapley values มาจาก cooperative game theory (Lloyd Shapley, 1953)
- ในบริบททางการแพทย์:
  * อาการสำคัญ (important symptoms) = ผู้เล่นที่มีค่า Shapley สูง (high contribution)
  * อาการไม่แน่นอน (uncertain symptoms) = ผู้เล่นที่มีค่า Shapley ต่ำ (low contribution)
  * Output = ผลรวมของ contribution จากทุกอาการ (coalition value)

Mathematical Foundation:
  φᵢ = Σ [|S|!(|N|-|S|-1)! / |N|!] × [v(S∪{i}) - v(S)]

  where:
  - φᵢ = Shapley value for feature i (symptom i's contribution)
  - S = subset of features (coalition of symptoms)
  - v(S) = model prediction with features S (outcome with symptoms S)
  - N = all features (all possible symptoms)

Clinical Interpretation:
  SHAP value บอก: "ถ้าไม่มีอาการนี้ prediction จะเปลี่ยนไปเท่าไร?"
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP Explainer with Game Theory interpretation for medical AI.

    Provides comprehensive explanations:
    1. Global Feature Importance - อาการไหนสำคัญที่สุดโดยรวม
    2. Local Explanations - ทำไมผู้ป่วยคนนี้ได้ prediction นี้
    3. Feature Interactions - อาการไหนทำงานร่วมกัน
    4. Decision Plots - แสดงการตัดสินใจแบบ step-by-step
    5. Force Plots - visualize แรงผลักดัน prediction
    """

    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model
            X_train: Training data (background distribution)
            feature_names: Feature names
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.feature_names = feature_names

        if feature_names is None and isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        elif feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]

        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        print(f"Initializing SHAP TreeExplainer for {model_type}...")

        try:
            # Try TreeExplainer first (fastest, exact for tree models)
            self.explainer = shap.TreeExplainer(self.model)
            print("✓ TreeExplainer initialized")
        except:
            # Fallback to KernelExplainer (slower, model-agnostic)
            print("TreeExplainer failed, using KernelExplainer...")
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
            print("✓ KernelExplainer initialized")

    def compute_shap_values(self, X):
        """Compute SHAP values for dataset."""
        print(f"Computing SHAP values for {len(X)} samples...")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.shap_values = self.explainer.shap_values(X_array)

        print("✓ SHAP values computed")
        return self.shap_values

    def plot_global_importance(self, X, top_n=20,
                               save_path='experiments/charts/shap_01_global_importance.png'):
        """
        Global Feature Importance using SHAP values.

        Game Theory Interpretation:
        - ค่า mean |SHAP| สูง = ผู้เล่นสำคัญ (important player)
        - ค่า mean |SHAP| ต่ำ = ผู้เล่นไม่สำคัญ (minor player)
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Handle multi-class
        if isinstance(self.shap_values, list):
            shap_abs = np.abs(self.shap_values[0])
            for i in range(1, len(self.shap_values)):
                shap_abs += np.abs(self.shap_values[i])
            shap_abs /= len(self.shap_values)
        else:
            shap_abs = np.abs(self.shap_values)

        # Mean absolute SHAP value per feature
        importance = np.mean(shap_abs, axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance (Mean |SHAP|)': importance,
            'Game Theory Role': ['Important Player' if imp > importance.mean()
                                 else 'Minor Player' for imp in importance]
        }).sort_values('Importance (Mean |SHAP|)', ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        top_features = importance_df.head(top_n)
        colors = ['#e74c3c' if role == 'Important Player' else '#95a5a6'
                  for role in top_features['Game Theory Role']]

        bars = ax.barh(range(len(top_features)), top_features['Importance (Mean |SHAP|)'],
                      color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Mean |SHAP Value| (Feature Contribution)', fontsize=12, fontweight='bold')
        ax.set_title('Global Feature Importance (Game Theory: Shapley Values)\n' +
                     'Red = Important Players, Gray = Minor Players',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['Importance (Mean |SHAP|)'])):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Global importance saved to {save_path}")
        plt.close()

        return importance_df

    def plot_summary(self, X, save_path='experiments/charts/shap_02_summary_plot.png'):
        """
        SHAP Summary Plot - shows feature impact distribution.

        - Each dot = one patient
        - Color: red (high feature value) to blue (low feature value)
        - X-axis: SHAP value (impact on prediction)
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # For multi-class, show class 0
        shap_plot = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_plot,
            X_array,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title('SHAP Summary Plot\n(Feature Value Impact on Predictions)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Summary plot saved to {save_path}")
        plt.close()

    def plot_waterfall(self, X, sample_idx=0,
                      save_path='experiments/charts/shap_03_waterfall.png'):
        """
        Waterfall plot - shows how features push prediction from base value.

        Game Theory Interpretation:
        - Base value = E[f(x)] = average prediction (no players)
        - Each bar = contribution from one feature (player joins coalition)
        - Final value = actual prediction (all players joined)
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get SHAP values for sample
        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_idx]
            base_value = self.explainer.expected_value[0]
        else:
            sample_shap = self.shap_values[sample_idx]
            base_value = self.explainer.expected_value

        # Create explanation object
        explanation = shap.Explanation(
            values=sample_shap,
            base_values=base_value,
            data=X_array[sample_idx],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f'SHAP Waterfall Plot (Sample #{sample_idx})\n' +
                  'Game Theory: Sequential Coalition Formation',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Waterfall plot saved to {save_path}")
        plt.close()

    def plot_force(self, X, sample_idx=0,
                   save_path='experiments/charts/shap_04_force_plot.png'):
        """
        Force plot - visualize forces pushing prediction higher or lower.

        - Red arrows = push toward higher risk
        - Blue arrows = push toward lower risk
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get values
        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_idx]
            base_value = self.explainer.expected_value[0]
        else:
            sample_shap = self.shap_values[sample_idx]
            base_value = self.explainer.expected_value

        # Create force plot
        shap.force_plot(
            base_value,
            sample_shap,
            X_array[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        plt.title(f'SHAP Force Plot (Sample #{sample_idx})\n' +
                  'Red = Increase Risk, Blue = Decrease Risk',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Force plot saved to {save_path}")
        plt.close()

    def plot_decision(self, X, sample_indices=None,
                     save_path='experiments/charts/shap_05_decision_plot.png'):
        """
        Decision plot - trace decision path for multiple patients.

        Shows how cumulative feature effects lead to final prediction.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        if sample_indices is None:
            sample_indices = range(min(20, len(X_array)))

        # Get values
        if isinstance(self.shap_values, list):
            shap_plot = self.shap_values[0][sample_indices]
            base_value = self.explainer.expected_value[0]
        else:
            shap_plot = self.shap_values[sample_indices]
            base_value = self.explainer.expected_value

        plt.figure(figsize=(12, 10))
        shap.decision_plot(
            base_value,
            shap_plot,
            X_array[sample_indices],
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Decision Plot\n(Cumulative Feature Effects)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Decision plot saved to {save_path}")
        plt.close()

    def plot_dependence(self, X, feature_name, interaction_feature=None,
                       save_path='experiments/charts/shap_06_dependence.png'):
        """
        Dependence plot - shows how one feature affects predictions.

        If interaction_feature specified, color by interaction.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get feature index
        if isinstance(feature_name, str):
            feature_idx = self.feature_names.index(feature_name)
        else:
            feature_idx = feature_name
            feature_name = self.feature_names[feature_idx]

        # Get interaction index
        if interaction_feature is not None:
            if isinstance(interaction_feature, str):
                interaction_idx = self.feature_names.index(interaction_feature)
            else:
                interaction_idx = interaction_feature
        else:
            interaction_idx = 'auto'

        # Plot
        shap_plot = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values

        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            feature_idx,
            shap_plot,
            X_array,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        plt.title(f'SHAP Dependence Plot: {feature_name}\n' +
                  '(How feature value affects SHAP value)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dependence plot saved to {save_path}")
        plt.close()

    def plot_interaction_heatmap(self, X, top_n=15,
                                 save_path='experiments/charts/shap_07_interaction_heatmap.png'):
        """
        Feature interaction heatmap.

        Shows which features work together (synergy in Game Theory).
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get top features by importance
        shap_abs = np.abs(self.shap_values[0] if isinstance(self.shap_values, list)
                          else self.shap_values)
        importance = np.mean(shap_abs, axis=0)
        top_indices = importance.argsort()[-top_n:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]

        # Compute correlation of SHAP values (proxy for interactions)
        shap_top = shap_abs[:, top_indices]
        correlation = np.corrcoef(shap_top.T)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            correlation,
            xticklabels=top_features,
            yticklabels=top_features,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'SHAP Correlation'}
        )
        ax.set_title('Feature Interaction Heatmap (SHAP Correlation)\n' +
                     'Game Theory: Coalition Synergies',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Interaction heatmap saved to {save_path}")
        plt.close()

    def plot_beeswarm(self, X, save_path='experiments/charts/shap_08_beeswarm.png'):
        """
        Beeswarm plot - enhanced summary plot with density.

        Shows distribution of feature impacts across all samples.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        shap_plot = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values

        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_plot,
                base_values=self.explainer.expected_value[0] if isinstance(self.shap_values, list)
                           else self.explainer.expected_value,
                data=X_array,
                feature_names=self.feature_names
            ),
            show=False,
            max_display=20
        )
        plt.title('SHAP Beeswarm Plot\n(Feature Impact Distribution)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Beeswarm plot saved to {save_path}")
        plt.close()

    def generate_clinical_report(self, X, sample_idx=0, predicted_class=None):
        """
        Generate clinical report with Game Theory interpretation.

        Returns human-readable explanation for physicians.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get SHAP values
        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_idx]
            base_value = self.explainer.expected_value[0]
        else:
            sample_shap = self.shap_values[sample_idx]
            base_value = self.explainer.expected_value

        # Rank features
        feature_contributions = [
            (self.feature_names[i], sample_shap[i], X_array[sample_idx, i])
            for i in range(len(sample_shap))
        ]
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Build report
        report = f"""
{'=' * 70}
CLINICAL DECISION SUPPORT - SHAP EXPLANATION
{'=' * 70}

PATIENT: Sample #{sample_idx}
PREDICTED RISK TIER: {predicted_class if predicted_class else 'N/A'}

GAME THEORY INTERPRETATION:
  Base Value (E[f(x)]): {base_value:.3f}
  - This is the average prediction across all patients
  - Represents outcome with no features (no players)

  Final Prediction: {base_value + sample_shap.sum():.3f}
  - Sum of base value + all feature contributions
  - Represents outcome with all features (all players joined)

TOP 5 RISK FACTORS (Positive SHAP = Increase Risk):
"""

        risk_factors = [(f, s, v) for f, s, v in feature_contributions if s > 0]
        for i, (feat, shap_val, feat_val) in enumerate(risk_factors[:5], 1):
            report += f"\n  {i}. {feat} = {feat_val:.2f}"
            report += f"\n     Shapley Value: +{shap_val:.3f}"
            report += f"\n     Interpretation: Important player increasing risk\n"

        report += f"\nTOP 5 PROTECTIVE FACTORS (Negative SHAP = Decrease Risk):\n"

        protective = [(f, s, v) for f, s, v in feature_contributions if s < 0]
        for i, (feat, shap_val, feat_val) in enumerate(protective[:5], 1):
            report += f"\n  {i}. {feat} = {feat_val:.2f}"
            report += f"\n     Shapley Value: {shap_val:.3f}"
            report += f"\n     Interpretation: Important player decreasing risk\n"

        report += f"""
GAME THEORY SUMMARY:
  - Important Players (|SHAP| > 0.05): {sum(1 for _, s, _ in feature_contributions if abs(s) > 0.05)}
  - Minor Players (|SHAP| ≤ 0.05): {sum(1 for _, s, _ in feature_contributions if abs(s) <= 0.05)}

  Coalition Value = Σ(Shapley Values) = {sample_shap.sum():.3f}

CLINICAL RECOMMENDATION:
  Focus on top 3-5 features with highest |SHAP| values for intervention.
  These are the "important players" most affecting patient outcome.

{'=' * 70}
"""

        return report


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    print("=" * 70)
    print("SHAP Explainability with Game Theory Foundation")
    print("=" * 70)

    # Generate data
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_classes=5,
        random_state=42
    )

    feature_names = [
        'Age', 'BMI', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol',
        'Glucose', 'Smoking', 'Exercise', 'Family_History', 'Stress_Level',
        'Sleep_Hours', 'Alcohol', 'Diet_Score', 'Medication', 'Previous_Events'
    ]

    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Train model
    print("\nTraining model...")
    model = XGBClassifier(max_depth=6, n_estimators=100, random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # Initialize SHAP explainer
    explainer = SHAPExplainer(model, X_train, feature_names)

    print("\n" + "=" * 70)
    print("Generating SHAP Visualizations (8 Charts)")
    print("=" * 70)

    # 1. Global importance
    print("\n1. Global Feature Importance...")
    importance_df = explainer.plot_global_importance(X_test)

    # 2. Summary plot
    print("\n2. Summary Plot...")
    explainer.plot_summary(X_test)

    # 3. Waterfall
    print("\n3. Waterfall Plot...")
    explainer.plot_waterfall(X_test, sample_idx=0)

    # 4. Force plot
    print("\n4. Force Plot...")
    explainer.plot_force(X_test, sample_idx=0)

    # 5. Decision plot
    print("\n5. Decision Plot...")
    explainer.plot_decision(X_test)

    # 6. Dependence plot
    print("\n6. Dependence Plot...")
    top_feature = importance_df.iloc[0]['Feature']
    explainer.plot_dependence(X_test, top_feature)

    # 7. Interaction heatmap
    print("\n7. Interaction Heatmap...")
    explainer.plot_interaction_heatmap(X_test)

    # 8. Beeswarm
    print("\n8. Beeswarm Plot...")
    explainer.plot_beeswarm(X_test)

    # Clinical report
    print("\n" + "=" * 70)
    print("Clinical Report")
    print("=" * 70)
    y_pred = model.predict(X_test)
    report = explainer.generate_clinical_report(X_test, sample_idx=0,
                                                predicted_class=f"R{y_pred[0]+1}")
    print(report)

    print("\n" + "=" * 70)
    print("SHAP Analysis Complete!")
    print("=" * 70)
    print("\nGenerated 8 charts in experiments/charts/:")
    print("  1. shap_01_global_importance.png")
    print("  2. shap_02_summary_plot.png")
    print("  3. shap_03_waterfall.png")
    print("  4. shap_04_force_plot.png")
    print("  5. shap_05_decision_plot.png")
    print("  6. shap_06_dependence.png")
    print("  7. shap_07_interaction_heatmap.png")
    print("  8. shap_08_beeswarm.png")
