"""
Standalone Synthetic Data Generator for SAFE-Gate

Generates 6,400 synthetic dizziness/vertigo cases with 52 features.
Risk tier distribution matches paper specifications.
"""

import json
import numpy as np
from pathlib import Path


class SAFEGateDataGenerator:
    """
    Generates synthetic clinical data for SAFE-Gate evaluation.

    Creates 6,400 cases with risk tier distribution:
    - R1 (Critical): 5.9% (377 cases)
    - R2 (High Risk): 15.8% (1,011 cases)
    - R3 (Moderate): 37.3% (2,387 cases)
    - R4 (Low Risk): 27.6% (1,766 cases)
    - R5 (Minimal): 13.4% (858 cases)
    """

    def __init__(self, n_total=6400, random_state=42):
        self.n_total = n_total
        np.random.seed(random_state)

        # Risk tier distribution from paper
        self.tier_distribution = {
            'R1': 0.059,
            'R2': 0.158,
            'R3': 0.373,
            'R4': 0.276,
            'R5': 0.134
        }

        self.tier_counts = {
            tier: int(n_total * prob)
            for tier, prob in self.tier_distribution.items()
        }

    def generate_all_cases(self):
        """Generate all 6,400 cases."""
        cases = []

        for tier, count in self.tier_counts.items():
            for i in range(count):
                case = self._generate_case(tier)
                cases.append(case)

        # Shuffle
        np.random.shuffle(cases)

        # Assign IDs
        for i, case in enumerate(cases):
            case['patient_id'] = f"P{i+1:06d}"

        return cases

    def _generate_case(self, tier):
        """Generate single case for target tier."""
        case = {'ground_truth_tier': tier}

        # Demographics
        case['age'] = self._age(tier)
        case['gender'] = np.random.choice(['male', 'female'])

        # Vitals
        case['systolic_bp'] = self._bp(tier)
        case['diastolic_bp'] = case['systolic_bp'] - np.random.randint(30, 50)
        case['heart_rate'] = self._hr(tier)
        case['spo2'] = self._spo2(tier)
        case['temperature'] = round(np.random.normal(37.0, 0.5), 1)
        case['gcs'] = self._gcs(tier)

        # Symptoms
        case['symptom_onset_hours'] = round(self._onset(tier), 2)
        case['symptom_duration_days'] = round(case['symptom_onset_hours'] / 24.0, 3)
        case['vertigo_severity'] = self._severity(tier)
        case['progression_pattern'] = self._progression(tier)

        # HINTS protocol
        hints = self._hints(tier)
        case.update(hints)

        # Cardiovascular risk factors
        case['hypertension'] = self._rf(tier, 'htn')
        case['atrial_fibrillation'] = self._rf(tier, 'af')
        case['diabetes'] = self._rf(tier, 'dm')
        case['prior_stroke'] = self._rf(tier, 'stroke')
        case['coronary_artery_disease'] = self._rf(tier, 'cad')

        # Neuro flags
        neuro_prob = {'R1': 0.8, 'R2': 0.4, 'R3': 0.1, 'R4': 0.05, 'R5': 0.02}[tier]
        case['dysarthria'] = np.random.random() < neuro_prob
        case['ataxia'] = np.random.random() < neuro_prob
        case['diplopia'] = np.random.random() < neuro_prob
        case['focal_weakness'] = np.random.random() < neuro_prob
        case['vertical_skew_deviation'] = np.random.random() < neuro_prob
        case['new_onset_diplopia'] = case['diplopia']
        case['crossed_sensory_loss'] = np.random.random() < neuro_prob
        case['severe_ataxia'] = case['ataxia']
        case['nystagmus_with_diplopia'] = case['diplopia'] and (np.random.random() < 0.5)

        # Symptom characteristics
        case['sudden_onset'] = tier in ['R1', 'R2']
        case['severe_imbalance'] = tier in ['R1', 'R2', 'R3']
        case['severe_vertigo'] = case['vertigo_severity'] == 'severe'
        case['vomiting'] = np.random.random() < 0.4
        case['headache'] = np.random.random() < 0.3

        # Positional triggers (BPPV indicators)
        pos_prob = 0.7 if tier in ['R4', 'R5'] else 0.2
        case['rolling_over_in_bed'] = np.random.random() < pos_prob
        case['looking_up'] = np.random.random() < pos_prob
        case['bending_forward'] = np.random.random() < pos_prob
        case['turning_head_quickly'] = np.random.random() < pos_prob

        return case

    def _age(self, tier):
        params = {'R1': (75, 10), 'R2': (70, 12), 'R3': (60, 15), 'R4': (55, 15), 'R5': (50, 18)}
        mean, std = params[tier]
        return int(np.clip(np.random.normal(mean, std), 18, 95))

    def _bp(self, tier):
        if tier == 'R1':
            return int(np.random.normal(85, 10))
        elif tier == 'R2':
            return int(np.random.normal(130, 15))
        else:
            return int(np.random.normal(120, 12))

    def _hr(self, tier):
        if tier == 'R1':
            return int(np.random.normal(125, 15))
        elif tier == 'R2':
            return int(np.random.normal(95, 12))
        else:
            return int(np.random.normal(80, 10))

    def _spo2(self, tier):
        if tier == 'R1':
            return int(np.random.normal(89, 3))
        else:
            return int(np.random.normal(97, 2))

    def _gcs(self, tier):
        return int(np.random.choice([12, 13])) if tier == 'R1' else 15

    def _onset(self, tier):
        ranges = {
            'R1': (0.5, 2.0),
            'R2': (1.0, 4.5),
            'R3': (4.5, 48.0),
            'R4': (24.0, 168.0),
            'R5': (168.0, 720.0)
        }
        return float(np.random.uniform(*ranges[tier]))

    def _severity(self, tier):
        if tier in ['R1', 'R2']:
            return np.random.choice(['severe', 'moderate'], p=[0.8, 0.2])
        elif tier == 'R3':
            return np.random.choice(['moderate', 'mild'], p=[0.7, 0.3])
        else:
            return np.random.choice(['mild', 'minimal'], p=[0.6, 0.4])

    def _progression(self, tier):
        if tier in ['R1', 'R2']:
            return np.random.choice(['rapidly_progressive', 'sudden_worsening', 'stable'], p=[0.4, 0.4, 0.2])
        else:
            return np.random.choice(['stable', 'gradually_improving', 'resolved'], p=[0.5, 0.3, 0.2])

    def _hints(self, tier):
        if tier in ['R1', 'R2']:  # Central
            return {
                'hints_head_impulse': 'normal',
                'hints_nystagmus': np.random.choice(['vertical', 'direction_changing', 'central']),
                'hints_test_of_skew': 'positive'
            }
        else:  # Peripheral
            return {
                'hints_head_impulse': 'abnormal',
                'hints_nystagmus': np.random.choice(['horizontal', 'unidirectional', 'peripheral']),
                'hints_test_of_skew': 'negative'
            }

    def _rf(self, tier, factor):
        probs = {
            'R1': {'htn': 0.8, 'af': 0.6, 'dm': 0.5, 'stroke': 0.4, 'cad': 0.5},
            'R2': {'htn': 0.7, 'af': 0.4, 'dm': 0.4, 'stroke': 0.2, 'cad': 0.3},
            'R3': {'htn': 0.5, 'af': 0.2, 'dm': 0.3, 'stroke': 0.1, 'cad': 0.2},
            'R4': {'htn': 0.4, 'af': 0.1, 'dm': 0.2, 'stroke': 0.05, 'cad': 0.1},
            'R5': {'htn': 0.3, 'af': 0.05, 'dm': 0.15, 'stroke': 0.02, 'cad': 0.05}
        }
        return np.random.random() < probs[tier][factor]

    def split(self, cases):
        """Split into train/val/test (75/12.5/12.5)."""
        train, val, test = [], [], []

        for tier in ['R1', 'R2', 'R3', 'R4', 'R5']:
            tier_cases = [c for c in cases if c['ground_truth_tier'] == tier]
            n = len(tier_cases)
            n_train = int(n * 0.75)
            n_val = int(n * 0.125)

            train.extend(tier_cases[:n_train])
            val.extend(tier_cases[n_train:n_train + n_val])
            test.extend(tier_cases[n_train + n_val:])

        return train, val, test

    def save(self, output_dir):
        """Generate and save datasets."""
        print("Generating 6,400 synthetic cases...")
        cases = self.generate_all_cases()

        print(f"\nGenerated {len(cases)} cases")
        print("\nRisk tier distribution:")
        for tier in ['R1', 'R2', 'R3', 'R4', 'R5']:
            count = sum(1 for c in cases if c['ground_truth_tier'] == tier)
            pct = count / len(cases) * 100
            target = self.tier_distribution[tier] * 100
            print(f"  {tier}: {count:4d} ({pct:5.1f}% vs target {target:5.1f}%)")

        print("\nSplitting into train/val/test...")
        train, val, test = self.split(cases)

        print(f"  Train: {len(train):4d} cases (75.0%)")
        print(f"  Val:   {len(val):4d} cases (12.5%)")
        print(f"  Test:  {len(test):4d} cases (12.5%)")

        # Save
        output_dir = Path(output_dir)
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            file_path = split_dir / f"synthetic_{split_name}_{len(split_data)}.json"
            with open(file_path, 'w') as f:
                json.dump(split_data, f, indent=2)

            print(f"\nSaved: {file_path}")

        print(f"\nData generation complete!")


if __name__ == "__main__":
    generator = SAFEGateDataGenerator(n_total=6400, random_state=42)
    generator.save('D:/PhD/Manuscript/GitHub/SAFE-Gate/data/synthetic')
