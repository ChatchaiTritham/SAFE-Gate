"""
Gate calibration profiles for SAFE-Gate.

Each profile is a set of discrete, clinically motivated switches; none is a
continuous threshold fitted to data.

  g1_graded          G1 maps only life-threatening instability (SBP < 90, HR < 50
                     or > 150, GCS < 14, SpO2 < 92 %) to R1 and every other red
                     flag to R2 (ESI level 1 vs level 2).
  g4_central_only    G4 reserves R2 for central signs / focal deficits and grades
                     peripheral presentations R5 / R4 / R3 by benign-syndrome
                     similarity (TiTrATE, HINTS).
  g5_age             G5 surrogate uses the age cut-offs (> 70 -> R3, < 50 -> R5)
                     that have no guideline source.
  g5_cap             Most urgent tier that uncertainty escalation alone may
                     reach (None = R1, 3 = R3).
  vacuous_negatives  A negative screen (G1: no red flag; G3: complete record)
                     returns R5 with confidence 0, a vacuous belief that sets no
                     floor and carries no consensus weight.
  theta, min_mass    ACWCM consensus trigger: margin of tau_w above the floor
                     (1/2 or 1 tier) and required dissent mass (simple majority
                     1/2 or qualified majority 2/3).

"v1" is the calibration of the earlier manuscript versions. "v2" is chosen on
the training split (hard constraints confirmed on validation) by
scripts/select_calibration.py; the choice and the criterion are recorded in
results/calibration_protocol.json, committed before any test evaluation.
"""

V1 = {"g1_graded": False, "g4_central_only": False, "g5_age": True, "g5_cap": None,
      "vacuous_negatives": False, "theta": 0.5, "min_mass": 0.5}

# Selected on the training split by scripts/select_calibration.py and
# recorded in results/calibration_protocol.json before any test evaluation.
V2 = {"g1_graded": True, "g4_central_only": True, "g5_age": False, "g5_cap": 3,
      "vacuous_negatives": True, "theta": 0.5, "min_mass": 2 / 3}

CANDIDATES = [
    {"g1_graded": a, "g4_central_only": b, "g5_age": c, "g5_cap": d,
     "vacuous_negatives": e, "theta": th, "min_mass": mm}
    for a in (False, True)
    for b in (False, True)
    for c in (True, False)
    for d in (None, 3)
    for e in (False, True)
    for th in (0.5, 1.0)
    for mm in (0.5, 2 / 3)
]


def resolve(calibration):
    """Return the switch dictionary for a profile name or an explicit dict."""
    if isinstance(calibration, dict):
        return {**V1, **calibration}
    if calibration == "v1":
        return dict(V1)
    if calibration == "v2":
        if V2 is None:
            raise ValueError("calibration v2 has not been selected yet")
        return dict(V2)
    raise ValueError(f"unknown calibration: {calibration}")
