"""
Diagnosis-conditioned synthetic cohort (round-4 evaluation generator).

Cases are generated diagnosis first; the triage tier is a property of the
diagnosis (and, for strokes, of the presentation), and every feature is drawn
from a distribution that depends on the DIAGNOSIS, not on the tier. Feature
distributions overlap across tiers by design:

  * onset times are log-normal per diagnosis and overlap (strokes include
    >24 h and wake-up presentations; vestibular neuritis presents at 2-72 h);
  * the HINTS examination is only meaningful in the acute vestibular syndrome
    (AVS: continuous vertigo with spontaneous nystagmus). In AVS a central
    pattern is produced with probability SENS_CENTRAL for strokes and
    1 - SPEC_CENTRAL for neuritis/labyrinthitis, and the reading is flipped
    with probability EXAMINER_ERROR (non-expert examiner);
  * BPPV has a normal head impulse (no spontaneous nystagmus) and positional
    triggers; AICA stroke carries new hearing loss (HINTS-plus); a share of
    AVS strokes has a normal-looking HINTS (false negative, "atypical");
  * vertebral-artery dissection occurs in younger patients with neck pain /
    headache.

All numerical parameters are the authors' modelling assumptions. Literature
anchors (stated in the manuscript supplement): Kattah et al. 2009 and
Tarnutzer et al. 2011 for the order of magnitude of HINTS sensitivity and
specificity when performed by trained examiners; the examiner-error term and
the case mix are assumptions, not estimates. The tier of each diagnosis is the
authors' mapping (see DIAGNOSES) and is NOT derived from the gate rules.
"""
from __future__ import annotations

import numpy as np

from generate_data import FEATURE_SCHEMA

SENS_CENTRAL = 0.98       # stroke in AVS -> central HINTS (Tarnutzer 2011 pooled sensitivity)
SPEC_CENTRAL = 0.85       # peripheral AVS -> peripheral HINTS (Tarnutzer 2011 pooled specificity)
EXAMINER_ERROR = 0.10     # probability that a HINTS reading is inverted

# name: (tier, share, age (mean, sd), onset median h, onset log-sd, AVS prob)
DIAGNOSES = {
    "posterior_stroke_unstable": ("R1", 0.020, (72, 11), 4.0, 1.2, 0.8),
    "subarachnoid_haemorrhage":  ("R1", 0.008, (55, 13), 2.0, 1.0, 0.0),
    "cardiac_arrhythmia":        ("R1", 0.020, (70, 12), 1.0, 1.0, 0.0),
    "posterior_stroke":          ("R2", 0.040, (68, 12), 8.0, 1.3, 0.8),
    "aica_stroke":               ("R2", 0.008, (66, 12), 8.0, 1.3, 1.0),
    "vertebral_dissection":      ("R2", 0.006, (44, 10), 12.0, 1.2, 0.6),
    "tia_vertebrobasilar":       ("R2", 0.030, (69, 11), 6.0, 1.2, 0.0),
    "vestibular_neuritis":       ("R3", 0.110, (50, 15), 14.0, 1.0, 1.0),
    "labyrinthitis":             ("R3", 0.030, (48, 15), 14.0, 1.0, 1.0),
    "vestibular_migraine":       ("R4", 0.100, (40, 12), 6.0, 1.2, 0.2),
    "menieres_attack":           ("R4", 0.070, (52, 13), 4.0, 0.9, 0.0),
    "orthostatic_dizziness":     ("R4", 0.080, (62, 16), 24.0, 1.4, 0.0),
    "bppv":                      ("R5", 0.270, (58, 14), 72.0, 1.2, 0.0),
    "persistent_postural":       ("R5", 0.108, (45, 13), 600.0, 0.8, 0.0),
}
STROKES = {"posterior_stroke_unstable", "posterior_stroke", "aica_stroke",
           "vertebral_dissection", "tia_vertebrobasilar"}
DANGEROUS = STROKES | {"subarachnoid_haemorrhage", "cardiac_arrhythmia"}
FOCAL_P = {"posterior_stroke_unstable": 0.6, "posterior_stroke": 0.35, "aica_stroke": 0.4,
           "vertebral_dissection": 0.3, "tia_vertebrobasilar": 0.25}


class DiagnosisGenerator:
    def __init__(self, n_total=2000, random_state=1):
        self.n_total = n_total
        self.rng = np.random.RandomState(random_state)

    def _b(self, p):
        return bool(self.rng.random() < p)

    def generate(self):
        names = list(DIAGNOSES)
        shares = np.array([DIAGNOSES[d][1] for d in names])
        counts = np.floor(shares / shares.sum() * self.n_total).astype(int)
        counts[names.index("bppv")] += self.n_total - counts.sum()
        cases = [self._case(d) for d, n in zip(names, counts) for _ in range(n)]
        order = self.rng.permutation(len(cases))
        cases = [cases[i] for i in order]
        for i, c in enumerate(cases):
            c["patient_id"] = f"D{i + 1:06d}"
        return cases

    def _hints(self, central):
        if self._b(EXAMINER_ERROR):
            central = not central
        if central:
            return {"hints_head_impulse": "normal",
                    "hints_nystagmus": str(self.rng.choice(["direction_changing", "vertical", "horizontal"])),
                    "hints_test_of_skew": "positive" if self._b(0.5) else "negative"}
        return {"hints_head_impulse": "abnormal",
                "hints_nystagmus": str(self.rng.choice(["horizontal", "unidirectional"])),
                "hints_test_of_skew": "negative"}

    def _case(self, dx):
        rng = self.rng
        tier, _, age, onset_med, onset_sd, p_avs = DIAGNOSES[dx]
        c = {}
        c["age"] = int(np.clip(rng.normal(*age), 18, 95))
        c["gender"] = str(rng.choice(["male", "female"]))
        # Vital signs: physiological variation for all; instability only where the
        # diagnosis implies it, and drawn so that it straddles the G1 cut-offs.
        sbp, hr, spo2, gcs = rng.normal(135, 18), rng.normal(82, 13), rng.normal(96.5, 1.7), 15
        if dx == "posterior_stroke_unstable":
            axis = rng.choice(["gcs", "sbp_high", "spo2"])
            gcs = int(rng.choice([9, 11, 12, 13, 14])) if axis == "gcs" else 15
            sbp = rng.normal(195, 15) if axis == "sbp_high" else sbp
            spo2 = rng.normal(90, 2.5) if axis == "spo2" else spo2
        elif dx == "cardiac_arrhythmia":
            hr = rng.normal(150, 25) if self._b(0.5) else rng.normal(42, 6)
            sbp = rng.normal(100, 18)
        elif dx == "subarachnoid_haemorrhage":
            sbp = rng.normal(165, 20)
            gcs = 14 if self._b(0.3) else 15
        elif dx == "orthostatic_dizziness":
            sbp = rng.normal(108, 14)
        c["systolic_bp"] = int(np.clip(sbp, 60, 240))
        c["heart_rate"] = int(np.clip(hr, 30, 200))
        c["spo2"] = int(np.clip(spo2, 80, 100))
        c["gcs"] = int(gcs)
        c["diastolic_bp"] = int(np.clip(c["systolic_bp"] * 0.62 + rng.normal(0, 8), 35, 140))
        c["temperature"] = round(float(np.clip(rng.normal(37.0 + (0.3 if dx == "labyrinthitis" else 0), 0.45), 35.0, 40.0)), 1)

        onset = float(np.exp(rng.normal(np.log(onset_med), onset_sd)))
        if dx in STROKES and self._b(0.15):
            onset = float(rng.uniform(24, 72))          # late or wake-up presentation
        c["symptom_onset_hours"] = round(onset, 2)
        c["symptom_duration_days"] = round(onset / 24.0, 3)
        avs = self._b(p_avs)
        severe = avs or dx in ("subarachnoid_haemorrhage", "menieres_attack")
        c["vertigo_severity"] = "severe" if severe and self._b(0.7) else str(rng.choice(["moderate", "mild"]))
        c["sudden_onset"] = self._b(0.8 if dx in DANGEROUS else 0.5 if avs else 0.3)
        c["progression_pattern"] = str(rng.choice(
            ["worsening", "stable", "improving"], p=[0.4, 0.45, 0.15] if dx in DANGEROUS
            else [0.1, 0.5, 0.4] if avs else [0.05, 0.35, 0.6]))

        if avs:
            c.update(self._hints(dx in STROKES and self._b(SENS_CENTRAL) or
                                 (dx not in STROKES and not self._b(SPEC_CENTRAL))))
        elif dx == "bppv":
            c.update(hints_head_impulse="normal", hints_nystagmus="peripheral", hints_test_of_skew="negative")
        else:
            c.update(hints_head_impulse="normal", hints_nystagmus="none", hints_test_of_skew="negative")

        vasc = dx in STROKES or dx == "cardiac_arrhythmia"
        older = c["age"] >= 60
        c["hypertension"] = self._b(0.65 if vasc else 0.45 if older else 0.2)
        c["atrial_fibrillation"] = self._b(0.25 if vasc else 0.08 if older else 0.02)
        c["diabetes"] = self._b(0.3 if vasc else 0.18)
        c["prior_stroke"] = self._b(0.15 if vasc else 0.03)
        c["coronary_artery_disease"] = self._b(0.25 if vasc else 0.08)

        fields = ["dysarthria", "ataxia", "diplopia", "crossed_sensory_loss",
                  "facial_weakness", "limb_weakness"]
        k = rng.randint(1, 4) if self._b(FOCAL_P.get(dx, 0.01)) else 0
        chosen = set(rng.choice(fields, size=k, replace=False)) if k else set()
        for f in fields:
            c[f] = f in chosen
        if dx in STROKES and not avs and self._b(0.4):
            c["ataxia"] = True                           # truncal ataxia, graded via gait below
        c["new_onset_diplopia"] = c["diplopia"]
        c["vertical_skew_deviation"] = c["hints_test_of_skew"] == "positive" and self._b(0.4)
        c["focal_weakness"] = c["limb_weakness"]
        sah = dx == "subarachnoid_haemorrhage"
        neck = dx == "vertebral_dissection"
        c["thunderclap_headache"] = sah and self._b(0.7)
        c["worst_headache_ever"] = sah and self._b(0.6)
        c["severe_headache"] = sah or (neck and self._b(0.6)) or (dx == "vestibular_migraine" and self._b(0.3))
        c["headache"] = c["severe_headache"] or self._b(0.6 if dx == "vestibular_migraine" else 0.15)

        c["positional_triggers"] = self._b(0.95 if dx == "bppv" else 0.35 if dx == "vestibular_migraine" else 0.15)
        c["episodic_pattern"] = self._b(0.9 if dx in ("bppv", "menieres_attack", "vestibular_migraine") else 0.25 if dx == "tia_vertebrobasilar" else 0.05)
        hearing = (dx in ("menieres_attack", "labyrinthitis")) or (dx == "aica_stroke" and self._b(0.8))
        c["hearing_loss"] = hearing
        c["hearing_loss_sudden"] = hearing and dx != "menieres_attack"
        c["tinnitus"] = hearing or self._b(0.08)
        c["aural_fullness"] = dx == "menieres_attack" and self._b(0.8)
        c["nausea_vomiting"] = self._b(0.8 if avs else 0.5)
        c["vomiting"] = c["nausea_vomiting"] and self._b(0.5)
        pp = 0.8 if c["positional_triggers"] else 0.1
        for f in ("rolling_over_in_bed", "looking_up", "bending_forward", "turning_head_quickly"):
            c[f] = self._b(pp)
        c["severe_imbalance"] = self._b(0.7 if dx in STROKES else 0.4 if avs else 0.1)
        c["severe_vertigo"] = c["vertigo_severity"] == "severe"
        c["nystagmus_with_diplopia"] = bool(c["diplopia"]) and self._b(0.5)
        c["gait_unsteady"] = bool(c["ataxia"]) or self._b(0.5 if avs else 0.15)
        c["data_completeness"] = 1.0
        c["age_over_60"] = c["age"] > 60
        out = {"ground_truth_tier": tier, "diagnosis": dx}
        for f in FEATURE_SCHEMA:
            out[f] = c[f]
        return out
