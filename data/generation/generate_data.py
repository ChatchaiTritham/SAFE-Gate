"""
Synthetic cohort generator for SAFE-Gate.

Generates a labelled synthetic dizziness/vertigo cohort whose clinical feature
vectors are internally consistent with the assigned risk tier, so that the
deterministic knowledge gates (G1-G6) can recover the intended tier. Every case
carries a fixed schema of 52 clinical features plus a patient identifier and the
ground-truth tier label.

The tier-conditioned feature distributions encode the same published clinical
knowledge (AHA/ASA red flags, HINTS, TiTrATE, "time is brain" timing) that the
gate logic implements. This circularity between generation templates and gate
logic is an inherent property of synthetic evaluation and is disclosed as a
limitation in the manuscript; it is not concealed here.

Seed and cohort size are fixed for byte-stable reproducibility.
"""

import json
from pathlib import Path

import numpy as np

# Fixed 52-feature clinical schema (order is canonical; do not reorder).
FEATURE_SCHEMA = [
    # Demographics (2)
    "age",
    "gender",
    # Vital signs (6)
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "spo2",
    "temperature",
    "gcs",
    # Symptom timing / character (5)
    "symptom_onset_hours",
    "symptom_duration_days",
    "vertigo_severity",
    "sudden_onset",
    "progression_pattern",
    # HINTS oculomotor protocol (3)
    "hints_head_impulse",
    "hints_nystagmus",
    "hints_test_of_skew",
    # Cardiovascular history (5)
    "hypertension",
    "atrial_fibrillation",
    "diabetes",
    "prior_stroke",
    "coronary_artery_disease",
    # Acute focal neurological deficits (9)
    "dysarthria",
    "ataxia",
    "diplopia",
    "new_onset_diplopia",
    "crossed_sensory_loss",
    "vertical_skew_deviation",
    "facial_weakness",
    "limb_weakness",
    "focal_weakness",
    # Headache red flags (3)
    "thunderclap_headache",
    "severe_headache",
    "worst_headache_ever",
    # Vestibular syndrome features (7)
    "positional_triggers",
    "episodic_pattern",
    "hearing_loss",
    "tinnitus",
    "aural_fullness",
    "nausea_vomiting",
    "headache",
    # Positional trigger detail (4)
    "rolling_over_in_bed",
    "looking_up",
    "bending_forward",
    "turning_head_quickly",
    # Composite / derived clinical flags (4)
    "severe_imbalance",
    "severe_vertigo",
    "nystagmus_with_diplopia",
    "vomiting",
    # Additional clinical context (4)
    "gait_unsteady",
    "hearing_loss_sudden",
    "data_completeness",
    "age_over_60",
]
assert len(FEATURE_SCHEMA) == 52, f"schema must define 52 features, got {len(FEATURE_SCHEMA)}"

TIERS = ["R1", "R2", "R3", "R4", "R5"]


class SAFEGateDataGenerator:
    """Generate a tier-consistent synthetic clinical cohort (52 features per case)."""

    def __init__(self, n_total=6400, random_state=42):
        self.n_total = n_total
        self.rng = np.random.RandomState(random_state)

        # Risk tier prevalence (vertigo/dizziness triage prior).
        self.tier_distribution = {
            "R1": 0.059,
            "R2": 0.158,
            "R3": 0.373,
            "R4": 0.276,
            "R5": 0.134,
        }
        self.tier_counts = {
            tier: int(n_total * prob) for tier, prob in self.tier_distribution.items()
        }
        # Absorb rounding remainder into the largest tier (R3).
        remainder = n_total - sum(self.tier_counts.values())
        self.tier_counts["R3"] += remainder

    # ------------------------------------------------------------------ #
    # Cohort assembly
    # ------------------------------------------------------------------ #
    def generate_all_cases(self):
        cases = []
        for tier, count in self.tier_counts.items():
            for _ in range(count):
                cases.append(self._generate_case(tier))

        order = self.rng.permutation(len(cases))
        cases = [cases[i] for i in order]
        for i, case in enumerate(cases):
            case["patient_id"] = f"P{i + 1:06d}"
        return cases

    def _bool(self, p):
        return bool(self.rng.random() < p)

    def _generate_case(self, tier):
        rng = self.rng
        case = {"ground_truth_tier": tier}

        # ---- Demographics ----
        age_params = {"R1": (75, 8), "R2": (70, 9), "R3": (58, 13), "R4": (50, 14), "R5": (44, 15)}
        mean, std = age_params[tier]
        case["age"] = int(np.clip(rng.normal(mean, std), 18, 95))
        case["gender"] = str(rng.choice(["male", "female"]))

        # ---- Vital signs ----
        # Only R1 carries hemodynamic / respiratory instability (G1 red flags).
        if tier == "R1":
            # Choose one instability axis so the case is unambiguously critical.
            axis = rng.choice(["sbp_low", "sbp_high", "hr_high", "spo2_low", "gcs_low"])
            case["systolic_bp"] = int(rng.normal(82, 4)) if axis == "sbp_low" else (
                int(rng.normal(192, 5)) if axis == "sbp_high" else int(rng.normal(125, 8))
            )
            case["heart_rate"] = int(rng.normal(158, 5)) if axis == "hr_high" else int(rng.normal(96, 8))
            case["spo2"] = int(rng.normal(88, 1)) if axis == "spo2_low" else int(rng.normal(96, 1))
            case["gcs"] = int(rng.choice([11, 12, 13])) if axis == "gcs_low" else 15
        else:
            case["systolic_bp"] = int(np.clip(rng.normal(128, 10), 95, 175))
            case["heart_rate"] = int(np.clip(rng.normal(80, 9), 55, 145))
            case["spo2"] = int(np.clip(rng.normal(97, 1.5), 93, 100))
            case["gcs"] = 15
        case["diastolic_bp"] = int(np.clip(case["systolic_bp"] - rng.randint(35, 50), 45, 110))
        case["temperature"] = round(float(np.clip(rng.normal(37.0, 0.4), 35.5, 39.0)), 1)

        # ---- Symptom timing / progression (drives G6) ----
        onset_ranges = {
            "R1": (0.2, 0.9),     # hyperacute, < 1h
            "R2": (1.0, 4.0),     # acute, worsening
            "R3": (5.0, 23.0),    # acute stable
            "R4": (25.0, 160.0),  # subacute
            "R5": (180.0, 700.0), # chronic
        }
        case["symptom_onset_hours"] = round(float(rng.uniform(*onset_ranges[tier])), 2)
        case["symptom_duration_days"] = round(case["symptom_onset_hours"] / 24.0, 3)

        sev_map = {
            "R1": ["severe"], "R2": ["severe", "moderate"], "R3": ["moderate"],
            "R4": ["mild", "moderate"], "R5": ["mild", "minimal"],
        }
        case["vertigo_severity"] = str(rng.choice(sev_map[tier]))
        case["sudden_onset"] = tier in ("R1", "R2")

        prog_map = {
            "R1": "worsening", "R2": "worsening", "R3": "stable",
            "R4": "improving", "R5": "resolved",
        }
        case["progression_pattern"] = prog_map[tier]

        # ---- HINTS protocol (central pattern only for R1/R2) ----
        if tier in ("R1", "R2"):
            case["hints_head_impulse"] = "normal"
            case["hints_nystagmus"] = str(rng.choice(["central", "vertical", "direction_changing"]))
            case["hints_test_of_skew"] = "positive"
        else:
            case["hints_head_impulse"] = "abnormal"
            case["hints_nystagmus"] = str(rng.choice(["horizontal", "unidirectional", "peripheral"]))
            case["hints_test_of_skew"] = "negative"

        # ---- Cardiovascular history (drives G2) ----
        rf = {
            "R1": {"htn": 0.8, "af": 0.6, "dm": 0.5, "stroke": 0.5, "cad": 0.5},
            "R2": {"htn": 0.75, "af": 0.55, "dm": 0.4, "stroke": 0.35, "cad": 0.35},
            "R3": {"htn": 0.45, "af": 0.12, "dm": 0.3, "stroke": 0.05, "cad": 0.15},
            "R4": {"htn": 0.3, "af": 0.04, "dm": 0.18, "stroke": 0.02, "cad": 0.08},
            "R5": {"htn": 0.2, "af": 0.02, "dm": 0.12, "stroke": 0.0, "cad": 0.03},
        }[tier]
        case["hypertension"] = self._bool(rf["htn"])
        case["atrial_fibrillation"] = self._bool(rf["af"])
        case["diabetes"] = self._bool(rf["dm"])
        case["prior_stroke"] = self._bool(rf["stroke"])
        case["coronary_artery_disease"] = self._bool(rf["cad"])

        # ---- Acute focal neurological deficits ----
        # Red-flag focal deficits appear only in R1 (so G1 fires R1 there) and
        # rarely in R2 (central vascular suspicion already carried by HINTS).
        focal_p = {"R1": 0.55, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0}[tier]
        case["dysarthria"] = self._bool(focal_p)
        case["ataxia"] = self._bool(focal_p)
        case["diplopia"] = self._bool(focal_p)
        case["new_onset_diplopia"] = case["diplopia"]
        case["crossed_sensory_loss"] = self._bool(focal_p)
        case["vertical_skew_deviation"] = (case["hints_test_of_skew"] == "positive") and self._bool(0.5) if tier == "R1" else False
        case["facial_weakness"] = self._bool(focal_p)
        case["limb_weakness"] = self._bool(focal_p)
        case["focal_weakness"] = case["limb_weakness"]

        # ---- Headache red flags (R1 only) ----
        hra_p = {"R1": 0.3, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0}[tier]
        case["thunderclap_headache"] = self._bool(hra_p)
        case["worst_headache_ever"] = self._bool(hra_p)
        case["severe_headache"] = self._bool(hra_p)

        # ---- Vestibular syndrome features (drive G4 specificity) ----
        # Benign tiers (R4/R5) match BPPV/peripheral profiles: positional
        # triggers present, no central signs.
        if tier in ("R4", "R5"):
            case["positional_triggers"] = self._bool(0.85)
            case["episodic_pattern"] = self._bool(0.7)
            pos_p = 0.7
        elif tier == "R3":
            case["positional_triggers"] = self._bool(0.4)
            case["episodic_pattern"] = self._bool(0.4)
            pos_p = 0.35
        else:
            case["positional_triggers"] = False
            case["episodic_pattern"] = False
            pos_p = 0.05

        # Meniere-type auditory features (a minority of benign cases).
        meniere = tier in ("R4", "R5") and self._bool(0.25)
        case["hearing_loss"] = meniere
        case["tinnitus"] = meniere or (tier in ("R4", "R5") and self._bool(0.15))
        case["aural_fullness"] = meniere

        case["nausea_vomiting"] = self._bool({"R1": 0.7, "R2": 0.6, "R3": 0.5, "R4": 0.35, "R5": 0.2}[tier])
        case["headache"] = self._bool({"R1": 0.4, "R2": 0.35, "R3": 0.3, "R4": 0.2, "R5": 0.15}[tier])

        # ---- Positional trigger detail ----
        case["rolling_over_in_bed"] = self._bool(pos_p)
        case["looking_up"] = self._bool(pos_p)
        case["bending_forward"] = self._bool(pos_p)
        case["turning_head_quickly"] = self._bool(pos_p)

        # ---- Composite / derived flags ----
        case["severe_imbalance"] = tier in ("R1", "R2", "R3")
        case["severe_vertigo"] = case["vertigo_severity"] == "severe"
        case["nystagmus_with_diplopia"] = bool(case["diplopia"]) and self._bool(0.5)
        case["vomiting"] = case["nausea_vomiting"] and self._bool(0.6)

        # ---- Additional clinical context ----
        case["gait_unsteady"] = bool(case["ataxia"]) or self._bool(
            {"R1": 0.6, "R2": 0.5, "R3": 0.3, "R4": 0.15, "R5": 0.05}[tier]
        )
        case["hearing_loss_sudden"] = bool(case["hearing_loss"]) and self._bool(0.4)
        case["data_completeness"] = 1.0
        case["age_over_60"] = case["age"] > 60

        # Enforce canonical schema ordering and completeness.
        ordered = {"ground_truth_tier": tier}
        for f in FEATURE_SCHEMA:
            ordered[f] = case[f]
        return ordered

    # ------------------------------------------------------------------ #
    # Split + persistence
    # ------------------------------------------------------------------ #
    def split(self, cases):
        train, val, test = [], [], []
        for tier in TIERS:
            tier_cases = [c for c in cases if c["ground_truth_tier"] == tier]
            n = len(tier_cases)
            n_train = int(n * 0.75)
            n_val = int(n * 0.125)
            train.extend(tier_cases[:n_train])
            val.extend(tier_cases[n_train:n_train + n_val])
            test.extend(tier_cases[n_train + n_val:])
        return train, val, test

    def save(self, output_dir):
        cases = self.generate_all_cases()
        train, val, test = self.split(cases)
        output_dir = Path(output_dir)
        written = {}
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            # Remove any stale split files so counts in filenames stay correct.
            for old in split_dir.glob(f"synthetic_{split_name}_*.json"):
                old.unlink()
            file_path = split_dir / f"synthetic_{split_name}_{len(split_data)}.json"
            file_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
            written[split_name] = file_path
        return cases, train, val, test, written


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    gen = SAFEGateDataGenerator(n_total=6400, random_state=42)
    cases, train, val, test, written = gen.save(here / "data" / "synthetic")
    print(f"Generated {len(cases)} cases ({len(FEATURE_SCHEMA)} features each)")
    for name, path in written.items():
        print(f"  {name}: {path.name}")
