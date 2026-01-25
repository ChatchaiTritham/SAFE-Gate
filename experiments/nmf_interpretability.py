#!/usr/bin/env python3
"""
NMF (Non-negative Matrix Factorization) Interpretability for SAFE-Gate

NMF discovers interpretable clinical syndromes (patterns) in patient data:
- Each component = Clinical syndrome (e.g., cardiovascular, neurological)
- Each patient = Weighted combination of syndromes
- Non-negativity constraint = Easy to interpret (no negative values)

Complements existing XAI methods:
1. SHAP          → "WHY?" (feature importance for predictions)
2. Counterfactual → "HOW?" (actionable changes to improve)
3. NMF           → "WHAT PATTERNS?" (clinical syndromes and patient archetypes)

Medical Interpretation:
- Components = Disease patterns/syndromes
- Feature loadings = Symptom importance in each syndrome
- Patient loadings = Which syndromes affect this patient
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NMFInterpreter:
    """
    NMF-based interpretability for clinical decision support.

    Discovers latent clinical syndromes that explain patient presentations.

    Key Features:
    1. Syndrome Discovery - Find meaningful clinical patterns
    2. Patient Archetypes - Identify typical patient profiles
    3. Feature Grouping - Group related symptoms
    4. Dimensionality Reduction - Simplify complex data
    """

    def __init__(self, n_components=5, random_state=42):
        """
        Initialize NMF interpreter.

        Args:
            n_components: Number of clinical syndromes to discover
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.nmf_model = None
        self.scaler = MinMaxScaler()  # NMF requires non-negative data
        self.W = None  # Patient-syndrome matrix (n_patients × n_components)
        self.H = None  # Syndrome-feature matrix (n_components × n_features)
        self.feature_names = None
        self.component_names = None

    def fit(self, X, feature_names=None):
        """
        Fit NMF model to discover clinical syndromes.

        Args:
            X: Patient data (n_patients × n_features)
            feature_names: List of feature names

        Returns:
            self
        """
        print(f"Fitting NMF with {self.n_components} components...")

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
            X_array = X

        # Scale to [0, 1] (NMF requires non-negative)
        X_scaled = self.scaler.fit_transform(X_array)

        # Fit NMF
        self.nmf_model = NMF(
            n_components=self.n_components,
            init='nndsvda',  # Non-negative double SVD initialization
            solver='cd',      # Coordinate descent
            max_iter=500,
            random_state=self.random_state,
            alpha_W=0.1,     # L1 regularization for sparsity
            alpha_H=0.1,
            l1_ratio=0.5
        )

        self.W = self.nmf_model.fit_transform(X_scaled)  # Patient loadings
        self.H = self.nmf_model.components_              # Feature loadings

        # Auto-name components based on top features
        self.component_names = self._auto_name_components()

        print(f"✓ NMF fitted (reconstruction error: {self.nmf_model.reconstruction_err_:.3f})")

        return self

    def _auto_name_components(self, top_n=3):
        """
        Automatically name components based on top features.

        Args:
            top_n: Number of top features to use for naming

        Returns:
            List of component names
        """
        component_names = []

        for i in range(self.n_components):
            # Get top features for this component
            top_indices = self.H[i].argsort()[-top_n:][::-1]
            top_features = [self.feature_names[idx] for idx in top_indices]

            # Create name from top features
            name = f"Syndrome_{i+1}: {', '.join(top_features[:2])}"
            component_names.append(name)

        return component_names

    def name_components(self, names):
        """
        Manually name components (e.g., based on clinical interpretation).

        Args:
            names: List of component names (length = n_components)
        """
        if len(names) != self.n_components:
            raise ValueError(f"Expected {self.n_components} names, got {len(names)}")

        self.component_names = names

    def plot_components_heatmap(self, top_features=15,
                                save_path='experiments/charts/nmf_01_components_heatmap.png'):
        """
        Heatmap showing feature loadings for each syndrome.

        Shows which features (symptoms) define each clinical syndrome.

        Interpretation:
        - Rows = Clinical syndromes
        - Columns = Features (symptoms)
        - Color intensity = How much this symptom contributes to this syndrome
        """
        if self.H is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get top features per component
        top_feature_indices = set()
        for i in range(self.n_components):
            top_indices = self.H[i].argsort()[-top_features:]
            top_feature_indices.update(top_indices)

        top_feature_indices = sorted(list(top_feature_indices))
        top_feature_names = [self.feature_names[i] for i in top_feature_indices]

        # Extract loadings for top features
        H_subset = self.H[:, top_feature_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            H_subset,
            xticklabels=top_feature_names,
            yticklabels=self.component_names,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Feature Loading'},
            linewidths=0.5
        )

        ax.set_xlabel('Features (Symptoms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clinical Syndromes', fontsize=12, fontweight='bold')
        ax.set_title('NMF Components: Clinical Syndrome Patterns\n' +
                     '(Feature loadings show which symptoms define each syndrome)',
                     fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Components heatmap saved to {save_path}")
        plt.close()

    def plot_component_loadings(self, component_idx=0, top_n=15,
                                save_path='experiments/charts/nmf_02_component_loadings.png'):
        """
        Bar chart showing top features for a specific syndrome.

        Interpretation:
        - Shows which symptoms are most important for this syndrome
        - Higher values = stronger association
        """
        if self.H is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get feature loadings for this component
        loadings = self.H[component_idx]
        top_indices = loadings.argsort()[-top_n:][::-1]

        top_features = [self.feature_names[i] for i in top_indices]
        top_values = loadings[top_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.YlOrRd(top_values / top_values.max())
        bars = ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Loading', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.component_names[component_idx]}\n' +
                     f'Top {top_n} Defining Features',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Component loadings saved to {save_path}")
        plt.close()

    def plot_patient_space(self, y=None, save_path='experiments/charts/nmf_03_patient_space.png'):
        """
        2D visualization of patients in syndrome space.

        Uses first 2 components to visualize patient distribution.

        Interpretation:
        - Each point = one patient
        - Position = combination of syndromes
        - Color = risk tier (if provided)
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        fig, ax = plt.subplots(figsize=(10, 8))

        if y is not None:
            # Color by risk tier
            scatter = ax.scatter(
                self.W[:, 0],
                self.W[:, 1],
                c=y,
                cmap='RdYlGn_r',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Risk Tier', fontsize=11, fontweight='bold')
        else:
            ax.scatter(
                self.W[:, 0],
                self.W[:, 1],
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )

        ax.set_xlabel(f'{self.component_names[0]} Loading',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{self.component_names[1]} Loading',
                     fontsize=12, fontweight='bold')
        ax.set_title('Patient Distribution in Syndrome Space\n' +
                     '(First 2 NMF Components)',
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Patient space plot saved to {save_path}")
        plt.close()

    def plot_syndrome_composition(self, save_path='experiments/charts/nmf_04_syndrome_composition.png'):
        """
        Stacked bar chart showing syndrome prevalence across patients.

        Interpretation:
        - Shows which syndromes are most common
        - Height = average loading across all patients
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Average loading per component
        avg_loadings = self.W.mean(axis=0)
        std_loadings = self.W.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = plt.cm.Set3(np.linspace(0, 1, self.n_components))
        bars = ax.bar(
            range(self.n_components),
            avg_loadings,
            yerr=std_loadings,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
            capsize=5
        )

        ax.set_xticks(range(self.n_components))
        ax.set_xticklabels(self.component_names, rotation=45, ha='right')
        ax.set_ylabel('Average Patient Loading', fontsize=12, fontweight='bold')
        ax.set_title('Clinical Syndrome Prevalence\n' +
                     '(Average across all patients)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val, std in zip(bars, avg_loadings, std_loadings):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Syndrome composition saved to {save_path}")
        plt.close()

    def plot_patient_profile(self, patient_idx=0,
                            save_path='experiments/charts/nmf_05_patient_profile.png'):
        """
        Radar chart showing syndrome composition for a specific patient.

        Interpretation:
        - Shows which syndromes affect this patient
        - Useful for personalized understanding
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get patient loadings
        patient_loadings = self.W[patient_idx]

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, self.n_components, endpoint=False).tolist()
        patient_loadings_plot = patient_loadings.tolist()

        # Close the plot
        angles += angles[:1]
        patient_loadings_plot += patient_loadings_plot[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, patient_loadings_plot, 'o-', linewidth=2, color='steelblue', label='Patient')
        ax.fill(angles, patient_loadings_plot, alpha=0.25, color='steelblue')

        # Add average for comparison
        avg_loadings = self.W.mean(axis=0).tolist()
        avg_loadings += avg_loadings[:1]
        ax.plot(angles, avg_loadings, 'o--', linewidth=2, color='red', label='Average', alpha=0.6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.component_names, size=9)
        ax.set_ylim(0, max(patient_loadings.max(), self.W.mean(axis=0).max()) * 1.2)
        ax.set_title(f'Patient #{patient_idx} Syndrome Profile\n' +
                     '(Comparison with population average)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Patient profile saved to {save_path}")
        plt.close()

    def plot_syndrome_correlation(self, save_path='experiments/charts/nmf_06_syndrome_correlation.png'):
        """
        Correlation heatmap between syndromes.

        Interpretation:
        - Shows which syndromes co-occur
        - High correlation = often appear together in patients
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute correlation between components
        correlation = np.corrcoef(self.W.T)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation,
            xticklabels=self.component_names,
            yticklabels=self.component_names,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'},
            linewidths=0.5
        )

        ax.set_title('Syndrome Co-occurrence Patterns\n' +
                     '(Correlation between clinical syndromes)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Syndrome correlation saved to {save_path}")
        plt.close()

    def identify_patient_archetypes(self, n_archetypes=5):
        """
        Identify patient archetypes (typical patient profiles).

        Uses k-means on syndrome loadings to find common patient types.

        Returns:
            Dictionary with archetype information
        """
        from sklearn.cluster import KMeans

        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Cluster patients in syndrome space
        kmeans = KMeans(n_clusters=n_archetypes, random_state=self.random_state)
        archetypes = kmeans.fit_predict(self.W)

        # Analyze each archetype
        archetype_info = []

        for i in range(n_archetypes):
            mask = archetypes == i
            n_patients = mask.sum()

            # Average syndrome profile for this archetype
            avg_profile = self.W[mask].mean(axis=0)

            # Dominant syndromes
            top_syndromes_idx = avg_profile.argsort()[-3:][::-1]
            top_syndromes = [self.component_names[idx] for idx in top_syndromes_idx]

            archetype_info.append({
                'archetype_id': i,
                'n_patients': n_patients,
                'percentage': 100 * n_patients / len(self.W),
                'avg_profile': avg_profile,
                'top_syndromes': top_syndromes,
                'description': f"Archetype {i+1}: {', '.join(top_syndromes[:2])}"
            })

        return archetype_info, archetypes

    def generate_clinical_report(self, patient_idx=0, y_pred=None):
        """
        Generate clinical report for a patient using NMF interpretation.

        Args:
            patient_idx: Patient index
            y_pred: Predicted risk tier (optional)

        Returns:
            Clinical report string
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        patient_loadings = self.W[patient_idx]

        # Rank syndromes by loading
        syndrome_ranking = [(self.component_names[i], patient_loadings[i])
                           for i in range(self.n_components)]
        syndrome_ranking.sort(key=lambda x: x[1], reverse=True)

        # Get top features for top syndrome
        top_syndrome_idx = patient_loadings.argmax()
        top_features_idx = self.H[top_syndrome_idx].argsort()[-5:][::-1]
        top_features = [self.feature_names[i] for i in top_features_idx]

        report = f"""
{'=' * 70}
NMF CLINICAL SYNDROME ANALYSIS
{'=' * 70}

PATIENT: Sample #{patient_idx}
PREDICTED RISK: {f'R{y_pred+1}' if y_pred is not None else 'N/A'}

SYNDROME COMPOSITION:
  This patient's clinical presentation can be decomposed into:
"""

        for i, (syndrome, loading) in enumerate(syndrome_ranking, 1):
            percentage = 100 * loading / patient_loadings.sum()
            report += f"\n  {i}. {syndrome}"
            report += f"\n     Loading: {loading:.3f} ({percentage:.1f}%)"

            if i == 1:
                report += " ← DOMINANT"

        report += f"""

PRIMARY SYNDROME: {syndrome_ranking[0][0]}
  Top associated features:
"""

        for i, feat in enumerate(top_features, 1):
            report += f"\n  {i}. {feat}"

        # Compare with population average
        avg_loadings = self.W.mean(axis=0)
        deviation = patient_loadings - avg_loadings

        report += f"""

COMPARISON WITH POPULATION AVERAGE:
"""

        for i in range(self.n_components):
            diff = deviation[i]
            if abs(diff) > 0.1:  # Significant deviation
                direction = "HIGHER" if diff > 0 else "LOWER"
                report += f"\n  - {self.component_names[i]}: {direction} than average ({diff:+.3f})"

        report += f"""

CLINICAL INTERPRETATION:
  The patient exhibits primarily {syndrome_ranking[0][0].lower()} characteristics,
  with {syndrome_ranking[1][0].lower()} as secondary pattern.

  This suggests a clinical presentation dominated by symptoms in the
  {syndrome_ranking[0][0].split(':')[1].strip() if ':' in syndrome_ranking[0][0] else 'primary'} domain.

RECOMMENDATIONS:
  1. Focus clinical attention on {syndrome_ranking[0][0].lower()}
  2. Monitor features: {', '.join(top_features[:3])}
  3. Consider comorbidities related to secondary syndromes

{'=' * 70}
"""

        return report


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    print("=" * 70)
    print("NMF Interpretability for SAFE-Gate")
    print("=" * 70)

    # Generate synthetic medical data
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=12,
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

    # Train model for predictions
    print("\nTraining model...")
    model = XGBClassifier(max_depth=6, n_estimators=100, random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("✓ Model trained")

    # Initialize NMF interpreter
    print("\n" + "=" * 70)
    print("Fitting NMF to discover clinical syndromes...")
    print("=" * 70)

    nmf_interpreter = NMFInterpreter(n_components=5, random_state=42)
    nmf_interpreter.fit(X_train)

    # Manually name components (based on domain knowledge)
    nmf_interpreter.name_components([
        'Syndrome 1: Cardiovascular',
        'Syndrome 2: Metabolic',
        'Syndrome 3: Neurological',
        'Syndrome 4: Lifestyle',
        'Syndrome 5: Stress-related'
    ])

    print("\n" + "=" * 70)
    print("Generating NMF Visualizations (6 Charts)")
    print("=" * 70)

    # 1. Components heatmap
    print("\n[1/6] Components Heatmap...")
    nmf_interpreter.plot_components_heatmap()

    # 2. Component loadings
    print("[2/6] Component Loadings...")
    nmf_interpreter.plot_component_loadings(component_idx=0)

    # 3. Patient space
    print("[3/6] Patient Space...")
    nmf_interpreter.plot_patient_space(y=y_test)

    # 4. Syndrome composition
    print("[4/6] Syndrome Composition...")
    nmf_interpreter.plot_syndrome_composition()

    # 5. Patient profile
    print("[5/6] Patient Profile...")
    nmf_interpreter.plot_patient_profile(patient_idx=0)

    # 6. Syndrome correlation
    print("[6/6] Syndrome Correlation...")
    nmf_interpreter.plot_syndrome_correlation()

    # Clinical report
    print("\n" + "=" * 70)
    print("Clinical Report")
    print("=" * 70)

    report = nmf_interpreter.generate_clinical_report(patient_idx=0, y_pred=y_pred[0])
    print(report)

    # Patient archetypes
    print("\n" + "=" * 70)
    print("Patient Archetypes")
    print("=" * 70)

    archetypes, assignments = nmf_interpreter.identify_patient_archetypes(n_archetypes=3)

    for arch in archetypes:
        print(f"\n{arch['description']}")
        print(f"  Patients: {arch['n_patients']} ({arch['percentage']:.1f}%)")
        print(f"  Top syndromes: {', '.join(arch['top_syndromes'])}")

    print("\n" + "=" * 70)
    print("NMF Analysis Complete!")
    print("=" * 70)
    print("\nGenerated 6 charts in experiments/charts/:")
    print("  1. nmf_01_components_heatmap.png")
    print("  2. nmf_02_component_loadings.png")
    print("  3. nmf_03_patient_space.png")
    print("  4. nmf_04_syndrome_composition.png")
    print("  5. nmf_05_patient_profile.png")
    print("  6. nmf_06_syndrome_correlation.png")
    print("\nNMF provides complementary insights to SHAP and Counterfactual:")
    print("  - SHAP: Feature importance (WHY?)")
    print("  - Counterfactual: Actionable changes (HOW?)")
    print("  - NMF: Clinical patterns (WHAT PATTERNS?)")
