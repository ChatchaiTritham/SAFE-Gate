# SAFE-Gate: Safety-Assured Fusion Engine with Gated Expert Triage

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Journal: Applied Intelligence](https://img.shields.io/badge/Journal-Applied%20Intelligence-orange.svg)
![Safety Violations](https://img.shields.io/badge/Safety%20Violations-0%2F6400-brightgreen.svg)

---

## Overview

SAFE-Gate is a safety-critical clinical decision support framework for emergency
department triage of patients presenting with dizziness and vertigo. The system
routes each patient case through six parallel expert gates -- each implementing a
distinct clinical or analytical perspective -- and fuses their outputs through an
Adaptive Confidence-Weighted Conservative Merging (ACWCM) mechanism that
provably preserves patient safety.

Unlike conventional ensemble methods that optimize for accuracy, SAFE-Gate is
designed around a **conservative-by-construction** principle: whenever expert
gates disagree, the system defaults to the most cautious risk tier. This
eliminates dangerous under-triage (false negatives) at the cost of controlled
over-triage, a clinically acceptable trade-off in emergency medicine. The
framework achieves 100% critical sensitivity (R1--R2) with zero safety
violations across 6,400 synthetic triage cases.

The architecture is grounded in a formal risk lattice
`R* <= R1 <= R2 <= R3 <= R4 <= R5`, where `R*` represents abstention (defer to
human clinician). Three safety theorems -- Conservative Preservation, Abstention
Correctness, and Critical Non-Dilution -- are verified at every inference step,
providing mathematical guarantees that no critical patient is misclassified to a
lower-acuity tier.

---

## Architecture

```
                         SAFE-Gate Architecture
  ========================================================================

  Stage 1              Stage 2              Stage 3         Stage 4
  PARALLEL GATES       RISK LATTICE         ACWCM FUSION    OUTPUT
  ----------------     ----------------     ------------    ----------

  [G1] Critical     -->|
       Red Flags       |
                       |
  [G2] Cardiovasc.  -->|                     +----------+
       Risk            |    R* <= R1 <=      |          |   Final Risk
                       |--> R2 <= R3 <= ---->|  ACWCM   |-->  Tier
  [G3] Data         -->|    R4 <= R5         |  Fusion  |   + Audit
       Quality         |                     +----------+     Trail
                       |
  [G4] TiTrATE      -->|    Conservative         ^
       Patterns        |    Minimum              |
                       |                    Confidence
  [G5] Bayesian     -->|                    Weights
       Uncertainty     |
                       |
  [G6] Temporal     -->|
       Risk

  ========================================================================
  Properties: Conservative Preservation | Abstention Correctness
              Critical Non-Dilution    | Zero Safety Violations
```

---

## Key Results

| Metric                        | Value         |
|-------------------------------|---------------|
| Critical sensitivity (R1--R2) | 100.0%        |
| Discharge specificity (R5)    | 66.7%         |
| False negative rate           | 0.0%          |
| Overall accuracy              | 65.2%         |
| Macro F1-score                | 68.9%         |
| Over-triage rate (ACWCM)      | 16.4%         |
| Over-triage rate (basic MIN)  | 21.3%         |
| Safety violations             | 0 / 6,400     |
| Inference latency             | 1.70 ms/case  |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ChatchaiTritham/SAFE-Gate.git
cd SAFE-Gate

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

### Basic Usage

```python
from src.safegate import SAFEGate

# Initialize the system (all 6 gates)
safegate = SAFEGate()

# Define a patient presentation (52 features)
patient = {
    'patient_id': 'P001',
    'age': 72,
    'systolic_bp': 85,
    'heart_rate': 125,
    'spo2': 88,
    'gcs': 13,
    'temperature': 37.2,
    'symptom_onset_hours': 1.5,
    'vertigo_severity': 'severe',
    'dysarthria': True,
    'hypertension': True,
    'atrial_fibrillation': True,
    'hints_head_impulse': 'abnormal',
    'hints_nystagmus': 'central',
    'hints_test_of_skew': 'positive'
}

# Classify the patient
result = safegate.classify(patient)

print(f"Risk Tier:      {result['final_tier']}")
print(f"Enforcing Gate: {result['enforcing_gate']}")
print(f"Confidence:     {result['confidence']:.2f}")
print(f"Latency:        {result['latency_ms']:.2f} ms")

# Print the full clinical audit trail
safegate.print_audit_trail(result['audit_trail'])
```

### Batch Classification

```python
import json

# Load synthetic dataset
with open('data/synthetic/test/synthetic_test_804.json', 'r') as f:
    test_data = json.load(f)

# Classify all patients
results = safegate.batch_classify(test_data)
```

---

## Repository Structure

```
SAFE-Gate/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- setup.py
|-- .gitignore
|
|-- src/
|   |-- safegate.py                  # Main SAFEGate class
|   |-- gates/
|   |   |-- gate1_critical_flags.py  # G1: Rule-based red flag detection
|   |   |-- gate2_moderate_risk.py   # G2: Cardiovascular risk scoring
|   |   |-- gate3_data_quality.py    # G3: Data completeness assessment
|   |   |-- gate4_titrate_logic.py   # G4: TiTrATE clinical patterns
|   |   |-- gate5_uncertainty.py     # G5: Bayesian uncertainty (MC dropout)
|   |   |-- gate6_temporal_risk.py   # G6: Temporal risk analysis (FSM)
|   |-- merging/
|   |   |-- conservative_merging.py  # ACWCM fusion mechanism
|   |   |-- risk_lattice.py          # Risk lattice (R*, R1--R5)
|   |-- theorems/
|   |   |-- theorem_verification.py  # Runtime safety theorem checking
|   |-- utils/
|       |-- audit_trail.py           # Clinical audit trail generator
|       |-- visualization.py         # Plotting utilities
|
|-- data/
|   |-- generation/
|   |   |-- generate_data.py         # Synthetic data generator
|   |   |-- syndx_adapter.py         # SynDX integration adapter
|   |-- synthetic/
|       |-- train/                   # 4,796 training cases
|       |-- val/                     # 798 validation cases
|       |-- test/                    # 804 test cases (+ 2 reserved)
|
|-- evaluation/
|   |-- generate_performance_metrics.py
|   |-- create_manuscript_figures.py
|   |-- figures/                     # Generated evaluation plots
|   |-- manuscript_figures/          # Publication-quality figures
|
|-- experiments/
|   |-- advanced_ensemble.py
|   |-- counterfactual_explanations.py
|   |-- feature_engineering.py
|   |-- hyperparameter_tuning.py
|   |-- interpretability_dashboard.py
|   |-- nmf_interpretability.py      # NMF rank-15 analysis
|   |-- shap_explainability.py
|
|-- notebooks/
|   |-- 00_quickstart.ipynb
|
|-- manuscript/                      # LaTeX source and figures
|-- models/                          # Trained model artifacts
|-- tests/                           # Unit and integration tests
|-- docs/                            # Additional documentation
```

---

## Gates Description

### G1: Critical Red Flag Detection

Rule-based gate implementing 18 clinical rules across 5 categories (neurological,
cardiovascular, consciousness, respiratory, trauma). Any triggered rule
immediately escalates to R1 or R2. Sensitivity-maximizing design ensures no
critical presentation is missed.

### G2: Cardiovascular Risk Assessment

Statistical gate combining weighted scoring of cardiovascular risk factors with
XGBoost validation. Assesses hemodynamic instability, arrhythmia risk, and
cerebrovascular risk profiles to assign risk tiers R2--R4.

### G3: Data Quality Assessment

Evaluates completeness ratio across 22 essential clinical fields. When data
completeness falls below the safety threshold, the gate outputs `R*`
(abstention), forcing the system to defer to a human clinician rather than risk
an unreliable classification.

### G4: Clinical Syndrome Pattern Matching

Implements the TiTrATE (Timing, Triggers, and Targeted Examination) diagnostic
framework using weighted Hamming distance against 320 expert-annotated syndrome
templates derived via NMF (rank-15) decomposition. Maps patient presentations to
known vertigo syndromes (BPPV, vestibular neuritis, Meniere's, central causes).

### G5: Epistemic Uncertainty Quantification

Bayesian neural network gate using Monte Carlo dropout (architecture:
52 -> 128 -> 64 -> 5, T=20 forward passes). Captures model epistemic uncertainty
and triggers abstention when predictive entropy exceeds calibrated thresholds,
preventing overconfident misclassification.

### G6: Temporal Risk Analysis

Finite-state machine modeling symptom evolution trajectories over time. Detects
deteriorating, stable, or improving temporal patterns and adjusts risk tier
accordingly. Captures dynamic risk that point-in-time assessments miss.

---

## Safety Properties

SAFE-Gate enforces three formally verified safety properties at every inference:

**Theorem 1 -- Conservative Preservation.**
The final merged tier is always at least as conservative as every individual gate
output: `T_final <= T_i` for all gates `i` in `{G1, ..., G6}`.

**Theorem 2 -- Abstention Correctness.**
If any gate outputs `R*` (abstention), the final system output is `R*`. The
system never overrides a gate's decision to defer to human judgment.

**Theorem 3 -- Critical Non-Dilution.**
A critical assessment (R1 or R2) from any gate cannot be diluted by non-critical
assessments from other gates. The conservative merging mechanism guarantees that
critical signals propagate to the final output.

These properties are verified at runtime by the theorem verification module
(`src/theorems/theorem_verification.py`) and are logged in the audit trail for
every patient classification.

---

## Evaluation

### Reproducing Results

```bash
# Generate evaluation metrics
python evaluation/generate_performance_metrics.py

# Generate manuscript figures (confusion matrix, per-class metrics, etc.)
python evaluation/create_manuscript_figures.py
```

### Baseline Comparisons

The evaluation suite includes comparisons against four baseline methods:

- **Single XGBoost** -- standard gradient boosting classifier
- **Ensemble Average** -- unweighted averaging of gate outputs
- **Confidence Threshold** -- thresholding on prediction confidence
- **ESI Guidelines** -- Emergency Severity Index rule-based triage

Results are reported in `evaluation/figures/` and
`evaluation/manuscript_figures/`.

### Explainability Experiments

```bash
# SHAP-based feature importance analysis
python experiments/shap_explainability.py

# NMF interpretability (rank-15 decomposition)
python experiments/nmf_interpretability.py

# Counterfactual explanations
python experiments/counterfactual_explanations.py
```

---

## Citation

If you use SAFE-Gate in your research, please cite:

```bibtex
@article{tritham2026safegate,
  title     = {{SAFE-Gate}: Safety-Assured Fusion Engine with Gated Expert
               Triage for Emergency Vertigo Assessment},
  author    = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal   = {Applied Intelligence},
  year      = {2026},
  publisher = {Springer},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

---

## Authors

- **Chatchai Tritham** -- Naresuan University, Thailand
- **Chakkrit Snae Namahoot** -- Naresuan University, Thailand
  ([chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th))
