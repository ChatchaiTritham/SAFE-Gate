"""
Independently parameterised synthetic cohort for stress testing SAFE-Gate.

The main generator (generate_data.py) draws features from tier templates that
mirror the gate thresholds (e.g. onset windows aligned with G6's 1 h / 24 h /
168 h cut-offs, central HINTS in every R1/R2 case, focal deficits only in R1).
This variant breaks that alignment on purpose. Its parameters were written
without reference to the gate thresholds and are NOT shared with them:

  * different tier prevalence (8 / 17 / 30 / 27 / 18 %);
  * symptom onset drawn from overlapping log-normal distributions, so tier
    and G6's time windows disagree for a sizeable share of cases;
  * vital signs drawn from continuous distributions that straddle the G1
    cut-offs, and only ~70 % of R1 cases have a measurable instability;
  * HINTS central pattern in 75 % of R1 and 60 % of R2 cases (imperfect
    sensitivity) and in 2-8 % of non-critical cases (imperfect specificity);
  * focal deficits in 45 % of R1, 20 % of R2 and a few non-critical cases;
  * probabilistic, overlapping progression patterns and risk factors.

The labels are therefore not recoverable by the gate rules alone, which is the
point: the variant measures what the safety floor does when the generator and
the gate logic no longer share their knowledge.
"""

import numpy as np

from generate_data import FEATURE_SCHEMA

TIERS = ["R1", "R2", "R3", "R4", "R5"]
PREVALENCE = {"R1": 0.08, "R2": 0.17, "R3": 0.30, "R4": 0.27, "R5": 0.18}
AGE = {"R1": (68, 12), "R2": (64, 13), "R3": (55, 15), "R4": (50, 16), "R5": (46, 16)}
ONSET_MEDIAN_H = {"R1": 2.0, "R2": 6.0, "R3": 18.0, "R4": 72.0, "R5": 240.0}
ONSET_SIGMA = 0.9
UNSTABLE = {"R1": 0.70, "R2": 0.10, "R3": 0.03, "R4": 0.01, "R5": 0.0}
CENTRAL = {"R1": 0.75, "R2": 0.60, "R3": 0.08, "R4": 0.03, "R5": 0.02}
FOCAL = {"R1": 0.45, "R2": 0.20, "R3": 0.03, "R4": 0.01, "R5": 0.0}
HEADACHE_RF = {"R1": 0.20, "R2": 0.10, "R3": 0.02, "R4": 0.0, "R5": 0.0}
PROGRESSION = {
    "R1": (["worsening", "stable"], [0.6, 0.4]),
    "R2": (["worsening", "stable", "improving"], [0.4, 0.5, 0.1]),
    "R3": (["stable", "improving", "worsening"], [0.5, 0.3, 0.2]),
    "R4": (["improving", "stable", "worsening"], [0.6, 0.3, 0.1]),
    "R5": (["resolved", "improving", "stable"], [0.5, 0.4, 0.1]),
}
SUDDEN = {"R1": 0.8, "R2": 0.7, "R3": 0.4, "R4": 0.2, "R5": 0.1}
RISK = {  # hypertension, AF, diabetes, prior stroke, CAD
    "R1": (0.65, 0.35, 0.35, 0.30, 0.30),
    "R2": (0.60, 0.30, 0.30, 0.20, 0.25),
    "R3": (0.45, 0.12, 0.22, 0.08, 0.12),
    "R4": (0.35, 0.06, 0.18, 0.04, 0.08),
    "R5": (0.30, 0.04, 0.15, 0.02, 0.05),
}
POSITIONAL = {"R1": 0.10, "R2": 0.10, "R3": 0.30, "R4": 0.60, "R5": 0.60}
SEVERITY = {
    "R1": (["severe", "moderate"], [0.7, 0.3]),
    "R2": (["severe", "moderate"], [0.5, 0.5]),
    "R3": (["moderate", "severe", "mild"], [0.6, 0.2, 0.2]),
    "R4": (["mild", "moderate"], [0.6, 0.4]),
    "R5": (["mild", "minimal", "moderate"], [0.5, 0.4, 0.1]),
}


class VariantGenerator:
    def __init__(self, n_total=2000, random_state=2026):
        self.n_total = n_total
        self.rng = np.random.RandomState(random_state)

    def _b(self, p):
        return bool(self.rng.random() < p)

    def generate(self):
        counts = {t: int(round(self.n_total * p)) for t, p in PREVALENCE.items()}
        counts["R3"] += self.n_total - sum(counts.values())
        cases = [self._case(t) for t, n in counts.items() for _ in range(n)]
        order = self.rng.permutation(len(cases))
        cases = [cases[i] for i in order]
        for i, c in enumerate(cases):
            c["patient_id"] = f"V{i + 1:06d}"
        return cases

    def _case(self, tier):
        rng = self.rng
        c = {}
        c["age"] = int(np.clip(rng.normal(*AGE[tier]), 18, 95))
        c["gender"] = str(rng.choice(["male", "female"]))
        unstable = self._b(UNSTABLE[tier])
        if unstable:
            axis = rng.choice(["sbp_low", "sbp_high", "hr_high", "spo2_low", "gcs_low"])
        else:
            axis = None
        c["systolic_bp"] = int(rng.normal(86, 7)) if axis == "sbp_low" else (
            int(rng.normal(188, 10)) if axis == "sbp_high" else int(rng.normal(132, 17)))
        c["heart_rate"] = int(rng.normal(150, 12)) if axis == "hr_high" else int(rng.normal(84, 13))
        c["spo2"] = int(rng.normal(90, 2.5)) if axis == "spo2_low" else int(np.clip(rng.normal(96.5, 1.8), 85, 100))
        c["gcs"] = int(rng.choice([10, 12, 13, 14])) if axis == "gcs_low" else (14 if self._b(0.02) else 15)
        c["diastolic_bp"] = int(np.clip(c["systolic_bp"] * 0.62 + rng.normal(0, 8), 40, 130))
        c["temperature"] = round(float(np.clip(rng.normal(37.0, 0.5), 35.0, 40.0)), 1)

        onset = float(np.exp(rng.normal(np.log(ONSET_MEDIAN_H[tier]), ONSET_SIGMA)))
        c["symptom_onset_hours"] = round(onset, 2)
        c["symptom_duration_days"] = round(onset / 24.0, 3)
        sev, p = SEVERITY[tier]
        c["vertigo_severity"] = str(rng.choice(sev, p=p))
        c["sudden_onset"] = self._b(SUDDEN[tier])
        prog, p = PROGRESSION[tier]
        c["progression_pattern"] = str(rng.choice(prog, p=p))

        if self._b(CENTRAL[tier]):
            c["hints_head_impulse"] = "normal"
            c["hints_nystagmus"] = str(rng.choice(["central", "vertical", "direction_changing"]))
            c["hints_test_of_skew"] = "positive" if self._b(0.6) else "negative"
        else:
            c["hints_head_impulse"] = "abnormal" if self._b(0.7) else "normal"
            c["hints_nystagmus"] = str(rng.choice(["horizontal", "unidirectional", "peripheral"]))
            c["hints_test_of_skew"] = "negative"

        htn, af, dm, st, cad = RISK[tier]
        c["hypertension"] = self._b(htn)
        c["atrial_fibrillation"] = self._b(af)
        c["diabetes"] = self._b(dm)
        c["prior_stroke"] = self._b(st)
        c["coronary_artery_disease"] = self._b(cad)

        focal = self._b(FOCAL[tier])
        k = rng.randint(1, 4) if focal else 0
        fields = ["dysarthria", "ataxia", "diplopia", "crossed_sensory_loss",
                  "facial_weakness", "limb_weakness"]
        chosen = set(rng.choice(fields, size=k, replace=False)) if k else set()
        for f in fields:
            c[f] = f in chosen
        c["new_onset_diplopia"] = c["diplopia"]
        c["vertical_skew_deviation"] = c["hints_test_of_skew"] == "positive" and self._b(0.4)
        c["focal_weakness"] = c["limb_weakness"]
        hrf = self._b(HEADACHE_RF[tier])
        c["thunderclap_headache"] = hrf and self._b(0.5)
        c["worst_headache_ever"] = hrf and self._b(0.5)
        c["severe_headache"] = hrf

        c["positional_triggers"] = self._b(POSITIONAL[tier])
        c["episodic_pattern"] = self._b(0.5 if tier in ("R4", "R5") else 0.25)
        men = tier in ("R3", "R4", "R5") and self._b(0.15)
        c["hearing_loss"] = men
        c["tinnitus"] = men or self._b(0.1)
        c["aural_fullness"] = men
        c["nausea_vomiting"] = self._b(0.55)
        c["headache"] = self._b(0.3)
        pp = 0.6 if c["positional_triggers"] else 0.1
        for f in ("rolling_over_in_bed", "looking_up", "bending_forward", "turning_head_quickly"):
            c[f] = self._b(pp)
        c["severe_imbalance"] = self._b({"R1": 0.7, "R2": 0.6, "R3": 0.4, "R4": 0.2, "R5": 0.1}[tier])
        c["severe_vertigo"] = c["vertigo_severity"] == "severe"
        c["nystagmus_with_diplopia"] = bool(c["diplopia"]) and self._b(0.5)
        c["vomiting"] = c["nausea_vomiting"] and self._b(0.5)
        c["gait_unsteady"] = bool(c["ataxia"]) or self._b(0.3)
        c["hearing_loss_sudden"] = bool(c["hearing_loss"]) and self._b(0.3)
        c["data_completeness"] = 1.0
        c["age_over_60"] = c["age"] > 60
        out = {"ground_truth_tier": tier}
        for f in FEATURE_SCHEMA:
            out[f] = c[f]
        return out
