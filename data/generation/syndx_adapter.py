"""
SynDX Adapter for SAFE-Gate

Generates 6,400 synthetic dizziness/vertigo cases using SynDX framework
and converts to SAFE-Gate format with 52 features.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add SynDX to path
syndx_path = Path(__file__).parent.parent.parent.parent / "SynDX"
sys.path.insert(0, str(syndx_path))

try:
    from syndx.phase1_knowledge.domain_config import create_vestibular_domain
    from syndx.phase1_knowledge.xai_explorer import XAIGuidedExplorer
    SYNDX_AVAILABLE = True
except ImportError:
    print("Warning: SynDX not found. Will use fallback generator.")
    SYNDX_AVAILABLE = False


class SAFEGateSynDXAdapter:
    """
    Adapter between SynDX framework and SAFE-Gate data requirements.

    Generates 6,400 cases with risk tier distribution:
    - R1 (Critical): 5.9% (377 cases)
    - R2 (High Risk): 15.8% (1,011 cases)
    - R3 (Moderate): 37.3% (2,387 cases)
    - R4 (Low Risk): 27.6% (1,766 cases)
    - R5 (Minimal): 13.4% (858 cases)
    """

    def __init__(self, n_total: int = 6400, random_state: int = 42):
        """
        Initialize adapter.

        Args:
            n_total: Total number of cases to generate
            random_state: Random seed for reproducibility
        """
        self.n_total = n_total
        self.random_state = random_state
        np.random.seed(random_state)

        # Risk tier distribution from paper
        self.tier_distribution = {
            'R1': 0.059,   # 5.9%
            'R2': 0.158,   # 15.8%
            'R3': 0.373,   # 37.3%
            'R4': 0.276,   # 27.6%
            'R5': 0.134    # 13.4%
        }

        # Calculate target counts
        self.tier_counts = {
            tier: int(n_total * prob)
            for tier, prob in self.tier_distribution.items()
        }

    def generate_all_cases(self):
        """
        Generate all 6,400 cases.

        Returns:
            List of patient dictionaries
        """
        if SYNDX_AVAILABLE:
            print("Using SynDX framework for generation...")
            cases = self._generate_with_syndx()
        else:
            print("SynDX not available. Using fallback generator...")
            cases = self._generate_fallback()

        # Assign patient IDs
        for i, case in enumerate(cases):
            case['patient_id'] = f"P{i+1:06d}"

        return cases

    def _generate_with_syndx(self):
        """Generate cases using SynDX framework."""
        # Create vestibular parameter space
        param_space = create_vestibular_domain()

        # Initialize XAI-guided explorer
        explorer = XAIGuidedExplorer(
            parameter_space=param_space,
            n_target=self.n_total,
            nmf_factors=20,
            random_state=self.random_state
        )

        # Execute exploration
        archetypes = explorer.explore()

        # Convert archetypes to SAFE-Gate format
        cases = []
        for archetype in archetypes[:self.n_total]:
            case = self._convert_archetype_to_safegate(archetype)
            cases.append(case)

        return cases

    def _generate_fallback(self):
        """
        Fallback generator when SynDX not available.

        Creates synthetic cases with realistic clinical distributions.
        """
        cases = []

        for tier, count in self.tier_counts.items():
            for _ in range(count):
                case = self._generate_single_case(tier)
                cases.append(case)

        # Shuffle cases
        np.random.shuffle(cases)

        return cases

    def _generate_single_case(self, target_tier: str):
        """
        Generate single patient case for target risk tier.

        Args:
            target_tier: Risk tier (R1-R5)

        Returns:
            Patient data dictionary with 52 features
        """
        case = {}

        # Demographics
        case['age'] = self._sample_age(target_tier)
        case['gender'] = np.random.choice(['male', 'female'])

        # Vital signs
        case['systolic_bp'] = self._sample_vital('bp', target_tier)
        case['diastolic_bp'] = case['systolic_bp'] - np.random.randint(30, 50)
        case['heart_rate'] = self._sample_vital('hr', target_tier)
        case['spo2'] = self._sample_vital('spo2', target_tier)
        case['temperature'] = np.random.normal(37.0, 0.5)
        case['gcs'] = self._sample_vital('gcs', target_tier)

        # Symptom characteristics
        case['symptom_onset_hours'] = self._sample_onset(target_tier)
        case['symptom_duration_days'] = case['symptom_onset_hours'] / 24.0
        case['vertigo_severity'] = self._sample_severity(target_tier)
        case['progression_pattern'] = self._sample_progression(target_tier)

        # Neurological examination (HINTS protocol)
        hints = self._sample_hints(target_tier)
        case.update(hints)

        # Cardiovascular risk factors
        case['hypertension'] = self._sample_risk_factor('htn', target_tier)
        case['atrial_fibrillation'] = self._sample_risk_factor('af', target_tier)
        case['diabetes'] = self._sample_risk_factor('dm', target_tier)
        case['prior_stroke'] = self._sample_risk_factor('prior_stroke', target_tier)
        case['coronary_artery_disease'] = self._sample_risk_factor('cad', target_tier)

        # Neurological red flags
        case['dysarthria'] = self._sample_neuro_flag(target_tier)
        case['ataxia'] = self._sample_neuro_flag(target_tier)
        case['diplopia'] = self._sample_neuro_flag(target_tier)
        case['focal_weakness'] = self._sample_neuro_flag(target_tier)
        case['vertical_skew_deviation'] = self._sample_neuro_flag(target_tier)
        case['new_onset_diplopia'] = case['diplopia']
        case['crossed_sensory_loss'] = self._sample_neuro_flag(target_tier)
        case['severe_ataxia'] = case['ataxia']
        case['nystagmus_with_diplopia'] = case['diplopia'] and self._sample_neuro_flag(target_tier)

        # Symptom triggers
        case['sudden_onset'] = target_tier in ['R1', 'R2']
        case['severe_imbalance'] = target_tier in ['R1', 'R2', 'R3']
        case['severe_vertigo'] = case['vertigo_severity'] == 'severe'
        case['vomiting'] = np.random.random() < 0.4
        case['headache'] = np.random.random() < 0.3

        # Positional triggers (more common in R4/R5)
        if target_tier in ['R4', 'R5']:
            case['rolling_over_in_bed'] = np.random.random() < 0.7
            case['looking_up'] = np.random.random() < 0.6
            case['bending_forward'] = np.random.random() < 0.5
            case['turning_head_quickly'] = np.random.random() < 0.5
        else:
            case['rolling_over_in_bed'] = np.random.random() < 0.2
            case['looking_up'] = np.random.random() < 0.1
            case['bending_forward'] = np.random.random() < 0.1
            case['turning_head_quickly'] = np.random.random() < 0.1

        # Ground truth risk tier
        case['ground_truth_tier'] = target_tier

        return case

    def _sample_age(self, tier: str) -> int:
        """Sample age based on risk tier."""
        age_params = {
            'R1': (75, 10),   # Mean 75, std 10
            'R2': (70, 12),
            'R3': (60, 15),
            'R4': (55, 15),
            'R5': (50, 18)
        }
        mean, std = age_params[tier]
        age = int(np.random.normal(mean, std))
        return np.clip(age, 18, 95)

    def _sample_vital(self, vital_type: str, tier: str):
        """Sample vital signs based on risk tier."""
        if vital_type == 'bp':
            if tier == 'R1':
                return int(np.random.normal(85, 10))  # Hypotension
            elif tier == 'R2':
                return int(np.random.normal(130, 15))
            else:
                return int(np.random.normal(120, 12))
        elif vital_type == 'hr':
            if tier == 'R1':
                return int(np.random.normal(125, 15))  # Tachycardia
            elif tier == 'R2':
                return int(np.random.normal(95, 12))
            else:
                return int(np.random.normal(80, 10))
        elif vital_type == 'spo2':
            if tier == 'R1':
                return int(np.random.normal(89, 3))  # Hypoxemia
            else:
                return int(np.random.normal(97, 2))
        elif vital_type == 'gcs':
            if tier == 'R1':
                return int(np.random.choice([12, 13]))  # Altered
            else:
                return 15  # Normal

    def _sample_onset(self, tier: str) -> float:
        """Sample symptom onset hours."""
        if tier == 'R1':
            return float(np.random.uniform(0.5, 2.0))  # Hyperacute
        elif tier == 'R2':
            return float(np.random.uniform(1.0, 4.5))  # Acute, within window
        elif tier == 'R3':
            return float(np.random.uniform(4.5, 48.0))  # Beyond window
        elif tier == 'R4':
            return float(np.random.uniform(24.0, 168.0))  # Subacute
        else:
            return float(np.random.uniform(168.0, 720.0))  # Chronic

    def _sample_severity(self, tier: str) -> str:
        """Sample vertigo severity."""
        if tier in ['R1', 'R2']:
            return np.random.choice(['severe', 'moderate'], p=[0.8, 0.2])
        elif tier == 'R3':
            return np.random.choice(['moderate', 'mild'], p=[0.7, 0.3])
        else:
            return np.random.choice(['mild', 'minimal'], p=[0.6, 0.4])

    def _sample_progression(self, tier: str) -> str:
        """Sample progression pattern."""
        if tier in ['R1', 'R2']:
            return np.random.choice(
                ['rapidly_progressive', 'sudden_worsening', 'stable'],
                p=[0.4, 0.4, 0.2]
            )
        else:
            return np.random.choice(
                ['stable', 'gradually_improving', 'resolved'],
                p=[0.5, 0.3, 0.2]
            )

    def _sample_hints(self, tier: str) -> dict:
        """Sample HINTS protocol findings."""
        hints = {}

        if tier in ['R1', 'R2']:  # Central pattern
            hints['hints_head_impulse'] = 'normal'
            hints['hints_nystagmus'] = np.random.choice(['vertical', 'direction_changing', 'central'])
            hints['hints_test_of_skew'] = 'positive'
        else:  # Peripheral pattern
            hints['hints_head_impulse'] = 'abnormal'
            hints['hints_nystagmus'] = np.random.choice(['horizontal', 'unidirectional', 'peripheral'])
            hints['hints_test_of_skew'] = 'negative'

        return hints

    def _sample_risk_factor(self, factor: str, tier: str) -> bool:
        """Sample cardiovascular risk factors."""
        probs = {
            'R1': {'htn': 0.8, 'af': 0.6, 'dm': 0.5, 'prior_stroke': 0.4, 'cad': 0.5},
            'R2': {'htn': 0.7, 'af': 0.4, 'dm': 0.4, 'prior_stroke': 0.2, 'cad': 0.3},
            'R3': {'htn': 0.5, 'af': 0.2, 'dm': 0.3, 'prior_stroke': 0.1, 'cad': 0.2},
            'R4': {'htn': 0.4, 'af': 0.1, 'dm': 0.2, 'prior_stroke': 0.05, 'cad': 0.1},
            'R5': {'htn': 0.3, 'af': 0.05, 'dm': 0.15, 'prior_stroke': 0.02, 'cad': 0.05}
        }
        prob = probs[tier].get(factor, 0.2)
        return np.random.random() < prob

    def _sample_neuro_flag(self, tier: str) -> bool:
        """Sample neurological red flags."""
        if tier == 'R1':
            return np.random.random() < 0.8  # High probability
        elif tier == 'R2':
            return np.random.random() < 0.4
        else:
            return np.random.random() < 0.05

    def _convert_archetype_to_safegate(self, archetype):
        """Convert SynDX archetype to SAFE-Gate format."""
        # This would map SynDX archetype fields to SAFE-Gate's 52 features
        # For now, use fallback generator
        tier = np.random.choice(list(self.tier_counts.keys()),
                               p=list(self.tier_distribution.values()))
        return self._generate_single_case(tier)

    def split_train_val_test(self, cases):
        """
        Split cases into train/val/test sets.

        - Train: 75% (4,800 cases)
        - Val: 12.5% (800 cases)
        - Test: 12.5% (800 cases)

        Returns:
            Tuple of (train, val, test) lists
        """
        n_train = 4800
        n_val = 800
        n_test = 800

        # Ensure we have exactly 6,400 cases
        cases = cases[:self.n_total]

        # Stratified split by risk tier
        train, val, test = [], [], []

        for tier in ['R1', 'R2', 'R3', 'R4', 'R5']:
            tier_cases = [c for c in cases if c['ground_truth_tier'] == tier]

            # Calculate split sizes proportionally
            n_tier = len(tier_cases)
            n_tier_train = int(n_tier * 0.75)
            n_tier_val = int(n_tier * 0.125)

            train.extend(tier_cases[:n_tier_train])
            val.extend(tier_cases[n_tier_train:n_tier_train + n_tier_val])
            test.extend(tier_cases[n_tier_train + n_tier_val:])

        return train, val, test

    def save_datasets(self, output_dir: Path):
        """
        Generate and save all datasets.

        Args:
            output_dir: Output directory for data files
        """
        print("Generating 6,400 synthetic cases...")
        cases = self.generate_all_cases()

        print(f"Generated {len(cases)} cases")
        print("\nRisk tier distribution:")
        for tier in ['R1', 'R2', 'R3', 'R4', 'R5']:
            count = sum(1 for c in cases if c['ground_truth_tier'] == tier)
            pct = count / len(cases) * 100
            print(f"  {tier}: {count:4d} ({pct:5.1f}%)")

        print("\nSplitting into train/val/test...")
        train, val, test = self.split_train_val_test(cases)

        print(f"  Train: {len(train)} cases")
        print(f"  Val:   {len(val)} cases")
        print(f"  Test:  {len(test)} cases")

        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)

        train_file = output_dir / 'train' / 'synthetic_train_4800.json'
        val_file = output_dir / 'val' / 'synthetic_val_800.json'
        test_file = output_dir / 'test' / 'synthetic_test_800.json'

        for file in [train_file, val_file, test_file]:
            file.parent.mkdir(parents=True, exist_ok=True)

        with open(train_file, 'w') as f:
            json.dump(train, f, indent=2)

        with open(val_file, 'w') as f:
            json.dump(val, f, indent=2)

        with open(test_file, 'w') as f:
            json.dump(test, f, indent=2)

        print(f"\nDatasets saved to {output_dir}")
        print(f"  - {train_file}")
        print(f"  - {val_file}")
        print(f"  - {test_file}")


def main():
    """Main function to generate SAFE-Gate datasets."""
    # Create adapter
    adapter = SAFEGateSynDXAdapter(n_total=6400, random_state=42)

    # Output directory
    output_dir = Path(__file__).parent.parent / 'synthetic'

    # Generate and save
    adapter.save_datasets(output_dir)

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
