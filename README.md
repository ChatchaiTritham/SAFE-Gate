# SAFE-Gate: Safety-Assured Fusion Engine with Gated Expert Triage

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
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
`R* โค R1 โค R2 โค R3 โค R4 โค R5`, where `R*` represents abstention (defer to
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
                       [G3] Data         -->|  R* โค R1 โค    |  Fusion  |  Certificate
                            Quality         |  R2 โค R3 โค    +----------+  + Audit
                                            |  R4 โค R5          ^          Trail
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

## Manuscript Alignment

Use the `SC_SAFE-Gate` package as the active manuscript alignment package while
the manuscript remains in preparation. This repository supports the SAFE-Gate
manuscript's safety-gated ensemble contribution:

- formulas: risk lattice, ACWCM confidence-weighted consensus, over-triage cost,
  and safety-bounded aggregation
- pseudocode: conflict detection, ACWCM fusion, and safety certificate
  generation
- logic: conservative preservation, abstention correctness, and critical
  non-dilution
- data/results: held-out synthetic vertigo-triage cases, baseline fusion
  comparisons, ablation behavior, and robustness checks
- figures: architecture, risk lattice, sensitivity, ablation, robustness,
  certificate, confusion, radar, and support-distribution artifacts

SAFE-Gate is a related safety-gated ensemble framework. It is not a duplicate of
SURgul/SRGL, which is tracked as governance/reproducibility evidence.

## Methodological References

The manuscript and repository are grounded in:

- emergency triage best practice and conservative management of critical cases
- TiTrATE-style vestibular syndrome reasoning
- lattice-based conservative merging
- uncertainty-aware ensemble comparison
- documented safety properties for high-risk clinical decision support

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

## Tutorials And Demos

- Notebook:
  - `notebooks/00_quickstart.ipynb`: quick interactive walkthrough for SAFE-Gate
- Evaluation-oriented scripts:
  - `evaluation/generate_performance_metrics.py`: evaluation summary generation
  - `evaluation/create_manuscript_figures.py`: manuscript-style figures
  - `scripts/generate_manuscript_manifest.py`: curated manuscript figure manifest and visual QA sheet
  - `experiments/interpretability_dashboard.py`: interpretability workflow entry point

## Curated Manuscript Figures

Curated manuscript figure exports are maintained for a manuscript that is still
in preparation. This status does not imply publication, acceptance, or final
journal readiness for every evaluation, demo, or exploratory artifact.

Regenerate manuscript figure exports:

```bash
python evaluation/create_manuscript_figures.py
```

Regenerate the manifest and visual QA sheet:

```bash
python scripts/generate_manuscript_manifest.py
```

Outputs:

- `evaluation/manuscript_figures/`: selected PDF and PNG manuscript figures
- `FIGURE_MANIFEST.csv`: figure role, source script, source artifact, caption,
  and intended article section
- `evaluation/manuscript_figures/visual_qa_contact_sheet.png`: visual QA sheet

The broader `evaluation/figures/` directory remains an evaluation archive unless
a figure is promoted into the manifest.

## Cross-Repository Tutorial Charts

- `../tutorial_surface_comparison.png`: scripts vs examples vs notebooks across all repositories
- `../tutorial_asset_density.png`: interactive/tutorial asset density normalized by repository size
- `../tutorial_maturity_report.md`: combined maturity summary

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
โ”โ”€โ”€ src/
โ”   โ”โ”€โ”€ safegate.py                  # Main SAFEGate orchestrator
โ”   โ”โ”€โ”€ gates/
โ”   โ”   โ”โ”€โ”€ gate1_critical_flags.py  # G1: Rule-based red flag detection (18 rules)
โ”   โ”   โ”โ”€โ”€ gate2_moderate_risk.py   # G2: Cardiovascular risk scoring (Eq. 3)
โ”   โ”   โ”โ”€โ”€ gate3_data_quality.py    # G3: Data completeness (22 fields, Eq. 4)
โ”   โ”   โ”โ”€โ”€ gate4_titrate_logic.py   # G4: TiTrATE syndrome matching (Hamming)
โ”   โ”   โ”โ”€โ”€ gate5_uncertainty.py     # G5: BNN MC dropout (52โ’128โ’64โ’5)
โ”   โ”   โ””โ”€โ”€ gate6_temporal_risk.py   # G6: Temporal FSM (5 states)
โ”   โ”โ”€โ”€ merging/
โ”   โ”   โ”โ”€โ”€ conservative_merging.py  # ACWCM fusion + conflict resolution
โ”   โ”   โ”โ”€โ”€ risk_lattice.py          # Risk lattice (R*, R1--R5)
โ”   โ”   โ””โ”€โ”€ safety_certificate.py   # Safety certificate generation
โ”   โ”โ”€โ”€ baselines/
โ”   โ”   โ”โ”€โ”€ esi_guidelines.py        # ESI rule-based baseline
โ”   โ”   โ”โ”€โ”€ single_xgboost.py        # XGBoost baseline
โ”   โ”   โ”โ”€โ”€ ensemble_average.py      # Ensemble averaging baseline
โ”   โ”   โ”โ”€โ”€ confidence_threshold.py  # Confidence thresholding baseline
โ”   โ”   โ”โ”€โ”€ dempster_shafer.py       # Dempster-Shafer combination
โ”   โ”   โ””โ”€โ”€ bayesian_model_avg.py    # Bayesian Model Averaging
โ”   โ”โ”€โ”€ theorems/
โ”   โ”   โ””โ”€โ”€ theorem_verification.py  # Runtime safety property checking
โ”   โ””โ”€โ”€ utils/
โ”       โ”โ”€โ”€ audit_trail.py           # Clinical audit trail generator
โ”       โ””โ”€โ”€ visualization.py         # Plotting utilities
โ”โ”€โ”€ data/
โ”   โ”โ”€โ”€ generation/                  # Synthetic data generators
โ”   โ””โ”€โ”€ synthetic/                   # Train (4,796) / Val (798) / Test (804)
โ”โ”€โ”€ evaluation/                      # Evaluation pipeline and figures
โ”โ”€โ”€ experiments/                     # XAI: SHAP, counterfactual, NMF
โ”โ”€โ”€ tests/                           # Test suite
โ”โ”€โ”€ docs/                            # Additional documentation
โ”โ”€โ”€ notebooks/                       # Jupyter quickstart
โ”โ”€โ”€ requirements.txt
โ”โ”€โ”€ setup.py
โ””โ”€โ”€ LICENSE
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
completeness falls below the safety threshold (ฯ < 0.70), the gate outputs R*
(abstention), forcing the system to defer to a human clinician rather than risk
an unreliable classification.

### G4: Clinical Syndrome Pattern Matching (TiTrATE)

Implements weighted Hamming distance matching against three characterised benign
vestibular syndromes (BPPV, vestibular neuritis, Meniere disease). High similarity
to a known benign profile supports safe discharge; low similarity triggers
escalation.

### G5: Epistemic Uncertainty Quantification (Bayesian)

Bayesian neural network (52โ’128โ’64โ’5) with Monte Carlo dropout (T=20 forward
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
cautious gate output: `rank(T_final) โค rank(min(T_i)) + 1`. Under basic minimum
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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

---

## Contact

### Contact Author

**Chatchai Tritham** (Author)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- ORCID: [0000-0001-7899-228X](https://orcid.org/0000-0001-7899-228X)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- E-mail: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- ORCID: [0000-0003-4660-4590](https://orcid.org/0000-0003-4660-4590)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand
