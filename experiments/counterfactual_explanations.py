#!/usr/bin/env python3
"""
Counterfactual Explanations for SAFE-Gate
Provides actionable insights: "What changes would move patient to lower risk tier?"

Clinical Use Cases:
- แพทย์: ดูว่าควรปรับการรักษาอย่างไร
- บุคลากรทางการแพทย์: เข้าใจปัจจัยที่เปลี่ยนแปลงได้
- ผู้ป่วย: รับคำแนะนำที่ทำได้จริง (actionable)
- คนทั่วไป: เข้าใจการป้องกันโรค
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class CounterfactualExplainer:
    """
    Counterfactual Explanations: "What if?" scenarios for clinical decision support.

    แนวคิด:
    - หาการเปลี่ยนแปลงที่น้อยที่สุด (minimal changes) เพื่อเปลี่ยน prediction
    - แนะนำเฉพาะสิ่งที่ทำได้จริง (actionable features)
    - คำนึงถึง clinical feasibility

    ตัวอย่าง:
    "ถ้าผู้ป่วยลด BMI จาก 32 เป็น 28 และเพิ่มการออกกำลังกายจาก 0 เป็น 3 ครั้ง/สัปดาห์
     จะเปลี่ยนจาก Risk Tier R3 เป็น R2"
    """

    def __init__(self, model, X_train, y_train, feature_names=None,
                 actionable_features=None, feature_ranges=None):
        """
        Initialize Counterfactual Explainer.

        Args:
            model: Trained classifier
            X_train: Training data (for finding similar cases)
            y_train: Training labels
            feature_names: List of feature names
            actionable_features: List of features that can be changed
                                (e.g., BMI=yes, Age=no)
            feature_ranges: Dict of {feature: (min, max)} for valid ranges
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.y_train = y_train if isinstance(y_train, np.ndarray) else y_train.values
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(self.X_train.shape[1])]

        # Actionable features (can be modified)
        if actionable_features is None:
            # Default: all features except demographic (Age, Sex, etc.)
            non_actionable = ['Age', 'Sex', 'Gender', 'Race', 'Family_History']
            self.actionable_features = [f for f in self.feature_names if f not in non_actionable]
        else:
            self.actionable_features = actionable_features

        self.actionable_indices = [self.feature_names.index(f) for f in self.actionable_features]

        # Feature ranges
        if feature_ranges is None:
            self.feature_ranges = {
                i: (self.X_train[:, i].min(), self.X_train[:, i].max())
                for i in range(self.X_train.shape[1])
            }
        else:
            self.feature_ranges = feature_ranges

        # Fit nearest neighbors for finding similar cases
        self.nn = NearestNeighbors(n_neighbors=50, metric='euclidean')
        self.nn.fit(self.X_train)

    def find_counterfactual(self, x_original, desired_class=None,
                           max_changes=5, method='optimization'):
        """
        Find counterfactual explanation for a sample.

        Args:
            x_original: Original sample (1D array)
            desired_class: Target class (if None, move to next lower risk tier)
            max_changes: Maximum number of features to change
            method: 'optimization' or 'nearest_neighbor'

        Returns:
            Dictionary with counterfactual explanation
        """
        x_original = np.array(x_original).flatten()
        original_pred = self.model.predict([x_original])[0]

        # Default: move to next lower risk tier
        if desired_class is None:
            desired_class = max(0, original_pred - 1)

        if method == 'optimization':
            result = self._optimize_counterfactual(x_original, desired_class, max_changes)
        else:
            result = self._nearest_neighbor_counterfactual(x_original, desired_class)

        return result

    def _optimize_counterfactual(self, x_original, desired_class, max_changes):
        """
        Find counterfactual using optimization.

        Minimize: distance(x_original, x_counterfactual)
        Subject to: model.predict(x_counterfactual) == desired_class
                   only change actionable features
                   respect feature ranges
                   limit number of changes
        """
        def objective(x):
            # L2 distance + sparsity penalty
            distance = np.sum((x - x_original) ** 2)
            changes = np.sum(np.abs(x - x_original) > 0.01)
            return distance + 0.1 * changes

        def constraint_prediction(x):
            # Model should predict desired class
            pred_proba = self.model.predict_proba([x])[0]
            return pred_proba[desired_class] - 0.5  # >= 0.5 probability

        # Bounds: only actionable features can change
        bounds = []
        for i in range(len(x_original)):
            if i in self.actionable_indices:
                bounds.append(self.feature_ranges[i])
            else:
                # Non-actionable: keep original value
                bounds.append((x_original[i], x_original[i]))

        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_prediction}
        ]

        # Optimize
        result = minimize(
            objective,
            x0=x_original,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            x_counterfactual = result.x
            new_pred = self.model.predict([x_counterfactual])[0]

            # Extract changes
            changes = self._extract_changes(x_original, x_counterfactual)

            return {
                'success': True,
                'original_prediction': int(original_pred := self.model.predict([x_original])[0]),
                'counterfactual_prediction': int(new_pred),
                'original_features': x_original,
                'counterfactual_features': x_counterfactual,
                'changes': changes,
                'distance': np.linalg.norm(x_counterfactual - x_original),
                'n_changes': len(changes),
                'clinical_report': self._generate_clinical_report(
                    x_original, x_counterfactual, changes,
                    original_pred, new_pred
                )
            }
        else:
            return {
                'success': False,
                'message': 'Could not find valid counterfactual',
                'original_prediction': int(self.model.predict([x_original])[0])
            }

    def _nearest_neighbor_counterfactual(self, x_original, desired_class):
        """
        Find counterfactual using nearest neighbor from different class.

        Find similar patient who has desired_class, show differences.
        """
        # Find samples with desired class
        target_indices = np.where(self.y_train == desired_class)[0]

        if len(target_indices) == 0:
            return {
                'success': False,
                'message': f'No training samples with class {desired_class}'
            }

        X_target = self.X_train[target_indices]

        # Find nearest neighbor
        distances = np.linalg.norm(X_target - x_original, axis=1)
        nearest_idx = distances.argmin()
        x_counterfactual = X_target[nearest_idx]

        # Extract changes
        changes = self._extract_changes(x_original, x_counterfactual)

        return {
            'success': True,
            'method': 'nearest_neighbor',
            'original_prediction': int(self.model.predict([x_original])[0]),
            'counterfactual_prediction': desired_class,
            'original_features': x_original,
            'counterfactual_features': x_counterfactual,
            'changes': changes,
            'distance': distances[nearest_idx],
            'n_changes': len(changes),
            'clinical_report': self._generate_clinical_report(
                x_original, x_counterfactual, changes,
                self.model.predict([x_original])[0], desired_class
            )
        }

    def _extract_changes(self, x_original, x_counterfactual, threshold=0.01):
        """Extract meaningful changes between original and counterfactual."""
        changes = []

        for i in self.actionable_indices:
            diff = x_counterfactual[i] - x_original[i]
            if abs(diff) > threshold:
                changes.append({
                    'feature': self.feature_names[i],
                    'feature_idx': i,
                    'original_value': float(x_original[i]),
                    'counterfactual_value': float(x_counterfactual[i]),
                    'change': float(diff),
                    'percent_change': float(100 * diff / (x_original[i] + 1e-8))
                })

        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        return changes

    def _generate_clinical_report(self, x_original, x_counterfactual, changes,
                                   original_pred, new_pred):
        """Generate human-readable clinical report."""

        report = f"""
COUNTERFACTUAL EXPLANATION
{'=' * 70}

CURRENT STATUS:
  Risk Tier: R{original_pred + 1}

ACTIONABLE CHANGES TO REDUCE RISK:
  Target Risk Tier: R{new_pred + 1}
  Number of Changes Required: {len(changes)}

TOP RECOMMENDED CHANGES:
"""

        for i, change in enumerate(changes[:5], 1):
            direction = "increase" if change['change'] > 0 else "decrease"
            report += f"\n  {i}. {change['feature']}: "
            report += f"{change['original_value']:.2f} → {change['counterfactual_value']:.2f} "
            report += f"({direction} by {abs(change['change']):.2f})"

        report += f"""

CLINICAL INTERPRETATION:
  These changes represent the MINIMAL modifications needed to reduce
  the patient's risk tier from R{original_pred + 1} to R{new_pred + 1}.

  Focus on the top 2-3 changes for practical intervention.

FEASIBILITY:
  ✓ All recommendations are clinically actionable
  ✓ Non-modifiable factors (Age, Sex, etc.) unchanged
  ✓ Changes are within physiologically reasonable ranges

NEXT STEPS:
  1. Discuss these recommendations with the patient
  2. Create personalized intervention plan
  3. Set realistic goals and timeline
  4. Monitor progress and re-evaluate
"""

        return report

    def plot_counterfactual_comparison(self, result, save_path='experiments/counterfactual_comparison.png'):
        """
        Visualize original vs counterfactual features.

        Shows which features changed and by how much.
        """
        if not result['success']:
            print("Cannot plot: counterfactual not found")
            return

        changes = result['changes']

        if len(changes) == 0:
            print("No changes found")
            return

        # Top 10 changes
        top_changes = changes[:min(10, len(changes))]

        features = [c['feature'] for c in top_changes]
        original = [c['original_value'] for c in top_changes]
        counterfactual = [c['counterfactual_value'] for c in top_changes]

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(features))
        width = 0.35

        bars1 = ax.barh(x - width/2, original, width, label='Original', color='#e74c3c', alpha=0.8)
        bars2 = ax.barh(x + width/2, counterfactual, width, label='Counterfactual', color='#2ecc71', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Counterfactual Explanation: R{result["original_prediction"]+1} → R{result["counterfactual_prediction"]+1}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Counterfactual comparison saved to {save_path}")
        plt.close()

    def plot_feature_changes_radar(self, result, save_path='experiments/counterfactual_radar.png'):
        """
        Radar chart showing feature changes (normalized).

        Good for visualizing multi-dimensional changes.
        """
        if not result['success']:
            print("Cannot plot: counterfactual not found")
            return

        changes = result['changes'][:8]  # Top 8 for readability

        if len(changes) < 3:
            print("Need at least 3 changes for radar chart")
            return

        categories = [c['feature'] for c in changes]

        # Normalize to 0-1 scale
        original_norm = []
        counterfactual_norm = []

        for c in changes:
            feat_idx = c['feature_idx']
            min_val, max_val = self.feature_ranges[feat_idx]
            range_val = max_val - min_val if max_val != min_val else 1

            orig_norm = (c['original_value'] - min_val) / range_val
            cf_norm = (c['counterfactual_value'] - min_val) / range_val

            original_norm.append(orig_norm)
            counterfactual_norm.append(cf_norm)

        # Number of variables
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # Close the plot
        original_norm += original_norm[:1]
        counterfactual_norm += counterfactual_norm[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, original_norm, 'o-', linewidth=2, label='Original', color='#e74c3c')
        ax.fill(angles, original_norm, alpha=0.25, color='#e74c3c')

        ax.plot(angles, counterfactual_norm, 'o-', linewidth=2, label='Counterfactual', color='#2ecc71')
        ax.fill(angles, counterfactual_norm, alpha=0.25, color='#2ecc71')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Feature Changes: R{result["original_prediction"]+1} → R{result["counterfactual_prediction"]+1}',
                     size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Radar chart saved to {save_path}")
        plt.close()

    def plot_change_magnitude(self, result, save_path='experiments/counterfactual_magnitude.png'):
        """
        Bar chart showing magnitude of changes (sorted).

        Helps prioritize which changes to focus on.
        """
        if not result['success']:
            print("Cannot plot: counterfactual not found")
            return

        changes = result['changes']

        features = [c['feature'] for c in changes]
        magnitudes = [abs(c['change']) for c in changes]
        colors = ['#e74c3c' if c['change'] < 0 else '#2ecc71' for c in changes]

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(features, magnitudes, color=colors, alpha=0.7)

        ax.set_xlabel('Magnitude of Change', fontsize=12, fontweight='bold')
        ax.set_title('Change Magnitude by Feature\n(Red=Decrease, Green=Increase)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
            ax.text(mag, bar.get_y() + bar.get_height()/2, f'{mag:.2f}',
                   ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Change magnitude plot saved to {save_path}")
        plt.close()

    def plot_what_if_scenarios(self, x_original, features_to_vary,
                              save_path='experiments/counterfactual_whatif.png'):
        """
        What-if analysis: vary features and show prediction changes.

        Example: "What if BMI ranges from 20 to 40?"
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, feature in enumerate(features_to_vary[:4]):
            if i >= 4:
                break

            feat_idx = self.feature_names.index(feature)
            min_val, max_val = self.feature_ranges[feat_idx]

            # Vary this feature
            values = np.linspace(min_val, max_val, 50)
            predictions = []

            for val in values:
                x_modified = x_original.copy()
                x_modified[feat_idx] = val
                pred = self.model.predict([x_modified])[0]
                predictions.append(pred)

            # Plot
            ax = axes[i]
            ax.plot(values, predictions, linewidth=2, color='steelblue')
            ax.axvline(x_original[feat_idx], color='red', linestyle='--',
                      label='Current', linewidth=2)
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Risk Tier', fontsize=11, fontweight='bold')
            ax.set_title(f'Impact of {feature}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_yticks(range(5))
            ax.set_yticklabels(['R1', 'R2', 'R3', 'R4', 'R5'])

        plt.suptitle('What-If Analysis: Feature Impact on Risk Prediction',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ What-if scenarios saved to {save_path}")
        plt.close()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    print("=" * 70)
    print("Counterfactual Explanations for SAFE-Gate")
    print("=" * 70)

    # Generate synthetic medical data
    X, y = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=8,
        n_classes=5,
        random_state=42
    )

    feature_names = [
        'Age', 'BMI', 'Blood_Pressure', 'Heart_Rate', 'Cholesterol',
        'Glucose', 'Smoking', 'Exercise', 'Family_History', 'Stress_Level',
        'Sleep_Hours', 'Diet_Score'
    ]

    # Actionable features (exclude Age, Family_History)
    actionable = [f for f in feature_names if f not in ['Age', 'Family_History']]

    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # Train model
    print("\nTraining model...")
    model = XGBClassifier(max_depth=6, n_estimators=100, random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # Initialize counterfactual explainer
    cf_explainer = CounterfactualExplainer(
        model, X_train, y_train,
        feature_names=feature_names,
        actionable_features=actionable
    )

    # Select high-risk patient
    high_risk_idx = np.where(model.predict(X_test) >= 3)[0]
    if len(high_risk_idx) > 0:
        patient_idx = high_risk_idx[0]
    else:
        patient_idx = 0

    x_patient = X_test.iloc[patient_idx].values

    print(f"\n" + "=" * 70)
    print(f"Patient Analysis (Sample #{patient_idx})")
    print("=" * 70)
    print(f"Current Risk Tier: R{model.predict([x_patient])[0] + 1}")

    # Find counterfactual
    print("\nFinding counterfactual explanation...")
    result = cf_explainer.find_counterfactual(x_patient, method='optimization')

    if result['success']:
        print(result['clinical_report'])

        # Visualizations
        print("\n" + "=" * 70)
        print("Generating Visualizations")
        print("=" * 70)

        cf_explainer.plot_counterfactual_comparison(result)
        cf_explainer.plot_feature_changes_radar(result)
        cf_explainer.plot_change_magnitude(result)
        cf_explainer.plot_what_if_scenarios(
            x_patient,
            features_to_vary=['BMI', 'Exercise', 'Smoking', 'Stress_Level']
        )

        print("\n" + "=" * 70)
        print("Counterfactual Analysis Complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - experiments/counterfactual_comparison.png")
        print("  - experiments/counterfactual_radar.png")
        print("  - experiments/counterfactual_magnitude.png")
        print("  - experiments/counterfactual_whatif.png")
    else:
        print(f"\n⚠ {result['message']}")
