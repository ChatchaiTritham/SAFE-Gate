# SAFE-Gate: Safety-Assured Fusion Engine with Gated Expert Triage

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)
![Safety Violations](https://img.shields.io/badge/Safety%20Violations-0%2F6398-brightgreen.svg)

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
over-triage, a clinically acceptable trade-off in emergency medicine.

The architecture is grounded in a formal risk lattice
`R* ≤ R1 ≤ R2 ≤ R3 ≤ R4 ≤ R5`, where `R*` represents abstention (defer to
human clinician). Three safety properties -- Conservative Preservation, Abstention
Correctness, and Critical Non-Dilution -- are verified at every inference step,
providing mathematical guarantees that no critical patient is misclassified to a
lower-acuity tier.

---

## Architecture

```text
                         SAFE-Gate Architecture
  ========================================================================

  Stage 1              Stage 2              Stage 3         Stage 4
  INPUT                PARALLEL GATES       ACWCM FUSION    OUTPUT
  ----------------     ----------------     ------------    ----------

  Patient Data  -----> [G1] Critical     -->|
  (52 features)             Red Flags       |
                                            |
                       [G2] Cardiovasc.  -->|                +----------+
                            Risk            |  Risk Lattice  |          |  Risk Tier
                                            |-->             |  ACWCM   |-> + Safety
                       [G3] Data         -->|  R* ≤ R1 ≤    |  Fusion  |  Certificate
                            Quality         |  R2 ≤ R3 ≤    +----------+  + Audit
                                            |  R4 ≤ R5          ^          Trail
                       [G4] TiTrATE      -->|                   |
                            Patterns        |              Confidence
                                            |              Weights
                       [G5] Bayesian     -->|
                            Uncertainty     |
                                            |
                       [G6] Temporal     -->|
                            Risk

  ========================================================================
  Safety Properties: Conservative Preservation | Abstention Correctness
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
| Safety violations             | 0 / 6,398     |
| Inference latency             | 1.70 ms/case  |

---

## Quick Start

### Installation

```bash
git clone https://github.com/ChatchaiTritham/SAFE-Gate.git
cd SAFE-Gate

pip install -r requirements.txt

# Optional: install in development mode
pip install -e .
```

### Basic Usage

```python
from src.safegate import SAFEGate

# Initialize (6 parallel gates with ACWCM fusion)
safegate = SAFEGate(mode="acwcm")

# Define patient presentation
patient = {
    'age': 72, 'gender': 'male',
    'systolic_bp': 85, 'heart_rate': 125,
    'spo2': 88, 'gcs': 13,
    'symptom_onset_hours': 1.5,
    'vertigo_severity': 'severe',
    'dysarthria': True,
    'atrial_fibrillation': True,
    'hints_head_impulse': 'abnormal',
    'hints_nystagmus': 'central',
    'hints_test_of_skew': 'positive',
}

result = safegate.classify(patient)

print(f"Risk Tier:      {result['final_tier']}")
print(f"Enforcing Gate: {result['enforcing_gate']}")
print(f"Confidence:     {result['confidence']:.2f}")
print(f"Latency:        {result['latency_ms']:.2f} ms")
```

### Running Tests

```bash
python tests/test_full_system.py
```

### Batch Classification

```python
import json

with open('data/synthetic/test/synthetic_test_804.json', 'r') as f:
    test_data = json.load(f)

results = safegate.batch_classify(test_data)
```

---

## Repository Structure

```text
SAFE-Gate/
├── src/
│   ├── safegate.py                  # Main SAFEGate orchestrator
│   ├── gates/
│   │   ├── gate1_critical_flags.py  # G1: Rule-based red flag detection (18 rules)
│   │   ├── gate2_moderate_risk.py   # G2: Cardiovascular risk scoring (Eq. 3)
│   │   ├── gate3_data_quality.py    # G3: Data completeness (22 fields, Eq. 4)
│   │   ├── gate4_titrate_logic.py   # G4: TiTrATE syndrome matching (Hamming)
│   │   ├── gate5_uncertainty.py     # G5: BNN MC dropout (52→128→64→5)
│   │   └── gate6_temporal_risk.py   # G6: Temporal FSM (5 states)
│   ├── merging/
│   │   ├── conservative_merging.py  # ACWCM fusion + conflict resolution
│   │   ├── risk_lattice.py          # Risk lattice (R*, R1--R5)
│   │   └── safety_certificate.py   # Safety certificate generation
│   ├── baselines/
│   │   ├── esi_guidelines.py        # ESI rule-based baseline
│   │   ├── single_xgboost.py        # XGBoost baseline
│   │   ├── ensemble_average.py      # Ensemble averaging baseline
│   │   ├── confidence_threshold.py  # Confidence thresholding baseline
│   │   ├── dempster_shafer.py       # Dempster-Shafer combination
│   │   └── bayesian_model_avg.py    # Bayesian Model Averaging
│   ├── theorems/
│   │   └── theorem_verification.py  # Runtime safety property checking
│   └── utils/
│       ├── audit_trail.py           # Clinical audit trail generator
│       └── visualization.py         # Plotting utilities
├── data/
│   ├── generation/                  # Synthetic data generators
│   └── synthetic/                   # Train (4,796) / Val (798) / Test (804)
├── evaluation/                      # Evaluation pipeline and figures
├── experiments/                     # XAI: SHAP, counterfactual, NMF
├── tests/                           # Test suite
├── docs/                            # Additional documentation
├── notebooks/                       # Jupyter quickstart
├── requirements.txt
├── setup.py
├── CITATION.cff
└── LICENSE
```

---

## Gates Description

### G1: Critical Red Flag Detection (Rule-Based)

Deterministic screening via 18 atomic Boolean rules across 5 clinical categories
(hemodynamic instability, altered mental status, acute focal deficits, severe
headache, respiratory compromise). Any single triggered rule immediately escalates
to R1 at maximal confidence.

### G2: Cardiovascular Risk Assessment (Statistical)

Weighted accumulation model combining demographic, symptom, and clinical history
risk factors with XGBoost consistency validation. Captures elevated stroke
probability from features that individually fall below critical thresholds but
collectively signal cardiovascular concern.

### G3: Data Quality Assessment

Evaluates completeness ratio across 22 essential clinical fields. When data
completeness falls below the safety threshold (ρ < 0.70), the gate outputs R*
(abstention), forcing the system to defer to a human clinician rather than risk
an unreliable classification.

### G4: Clinical Syndrome Pattern Matching (TiTrATE)

Implements weighted Hamming distance matching against three characterised benign
vestibular syndromes (BPPV, vestibular neuritis, Meniere disease). High similarity
to a known benign profile supports safe discharge; low similarity triggers
escalation.

### G5: Epistemic Uncertainty Quantification (Bayesian)

Bayesian neural network (52→128→64→5) with Monte Carlo dropout (T=20 forward
passes). Computes a composite uncertainty index from predictive entropy and
prediction variance. Triggers abstention or escalation when model uncertainty
exceeds calibrated thresholds.

### G6: Temporal Risk Analysis (State Machine)

Finite-state machine modelling symptom evolution trajectories. Five temporal
states (hyperacute, acute stable, acute improving, subacute, chronic) with
progression-modified transitions capture dynamic risk that point-in-time
assessments miss.

---

## Safety Properties

SAFE-Gate enforces three formally verified safety properties at every inference:

**Conservative Preservation (CP).**
The final merged tier is never more than one tier less conservative than the most
cautious gate output: `rank(T_final) ≤ rank(min(T_i)) + 1`. Under basic minimum
selection `T_final = min(T_i)`; ACWCM permits bounded one-tier relaxation only
when high-confidence gate consensus supports it.

**Abstention Correctness (AC).**
If any gate outputs R* (abstention), the final system output is R*. The system
never overrides a gate's decision to defer to human judgment.

**Critical Non-Dilution (CND).**
A critical assessment (R1 or R2) from any gate cannot be diluted by non-critical
assessments from other gates. Critical signals propagate to the final output.

These properties are verified at runtime and logged in the audit trail for every
patient classification.

---

## Evaluation

### Reproducing Results

```bash
# Run the comprehensive test suite
python tests/test_full_system.py

# Generate evaluation metrics
python evaluation/generate_performance_metrics.py

# Generate manuscript figures
python evaluation/create_manuscript_figures.py
```

### Baseline Comparisons

The evaluation suite includes comparisons against six baseline methods:

| Method                    | Type                |
|---------------------------|---------------------|
| ESI Guidelines            | Rule-based          |
| Single XGBoost            | Gradient boosting   |
| Ensemble Average          | Unweighted fusion   |
| Confidence Threshold      | Threshold-based     |
| Dempster-Shafer           | Evidence theory     |
| Bayesian Model Averaging  | Probabilistic       |

### Explainability

```bash
python experiments/shap_explainability.py
python experiments/nmf_interpretability.py
python experiments/counterfactual_explanations.py
```

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch from `main`.

2. **Write tests** for new functionality in `tests/`.
3. **Follow existing code style** (PEP 8, type hints, docstrings).
4. **Run the test suite** before submitting: `python tests/test_full_system.py`
5. **Submit a pull request** with a clear description of changes.

### Code Quality Standards

- All gate implementations must include `evaluate()`, `get_name()`, and
  `get_description()` methods.
- Safety properties (CP, AC, CND) must not be violated by any code change.
- New baselines should follow the interface pattern in `src/baselines/`.
- Clinical thresholds must be referenced to published guidelines.

### Reporting Issues

Please use [GitHub Issues](https://github.com/ChatchaiTritham/SAFE-Gate/issues)
for bug reports and feature requests. Include:
- Python version and OS
- Minimal reproducible example
- Expected vs. actual behaviour

---

## Citation

If you use SAFE-Gate in your research, please cite:

```bibtex
@article{tritham2026safegate,
  title     = {{SAFE-Gate}: An Adaptive Knowledge-Based Expert System for
               Emergency Triage Safety with Confidence-Weighted Conservative
               Merging and Formal Safety Guarantees},
  author    = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal   = {Soft Computing},
  publisher = {Springer},
  year      = {2026},
  url       = {https://github.com/ChatchaiTritham/SAFE-Gate},
  license   = {MIT}
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
