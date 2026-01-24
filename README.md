# SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine with parallel GATEs

A formally verified clinical triage architecture for dizziness and vertigo presentations in emergency departments, achieving 95.3% sensitivity with provable safety guarantees.

## Overview

SAFE-Gate implements a six-gate parallel evaluation system with conservative merging to guarantee safe triage decisions for diagnostically challenging presentations. Unlike traditional machine learning approaches that rely solely on empirical accuracy, SAFE-Gate provides mathematical proofs of safety properties through six formal theorems.

**Key Contributions:**
- Parallel architecture with redundancy ensuring no single component failure produces unsafe outputs
- Conservative merging via minimum lattice selection (2.5% improvement over ensemble averaging)
- Explicit abstention mechanism acknowledging uncertainty rather than forcing unreliable predictions
- Formal verification with zero theorem violations across 6,400 test cases
- Real-time performance (1.23ms mean latency) supporting emergency deployment

## Performance Metrics

Evaluated on 800 held-out synthetic dizziness/vertigo presentations:

| Metric | Value | 95% CI |
|--------|-------|--------|
| Sensitivity (R1-R2 detection) | 95.3% | 92.1-97.8% |
| Specificity (R5 safe discharge) | 94.7% | 91.3-97.2% |
| Abstention Rate (R* tier) | 12.4% | 9.7-15.6% |
| Mean Decision Latency | 1.23 ms | 1.18-1.29 ms |
| Safety Theorem Violations | 0/800 | 0.0-1.8% |

**Baseline Comparison:**
- ESI Guidelines: 87.5% sensitivity
- Single XGBoost: 91.2% sensitivity
- Ensemble Averaging: 92.8% sensitivity (2.5% lower than SAFE-Gate)
- Confidence Thresholding: 88.9% sensitivity, 15.2% abstention

## Installation

### Requirements
- Python 3.8+
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0

### Setup

```bash
# Clone repository
git clone https://github.com/ChatchaiTritham/SAFE-Gate.git
cd SAFE-Gate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/safegate.py
```

## Quick Start

### Basic Usage

```python
from src.safegate import SAFEGate

# Initialize system
safegate = SAFEGate()

# Prepare patient data
patient = {
    'age': 68,
    'systolic_bp': 145,
    'heart_rate': 88,
    'spo2': 97,
    'gcs': 15,
    'temperature': 37.2,
    'symptom_onset_hours': 2.5,
    'vertigo_severity': 'severe',
    'dysarthria': True,
    'ataxia': True,
    'symptom_duration_days': 0.1,
    'hypertension': True,
    'atrial_fibrillation': False,
    'diabetes': True,
    'hints_head_impulse': 'abnormal',
    'hints_nystagmus': 'central',
    'hints_test_of_skew': 'positive'
}

# Perform classification
result = safegate.classify(patient, patient_id='P001')

print(f"Risk Tier: {result['final_tier']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Enforcing Gate: {result['enforcing_gate']}")
print(f"Latency: {result['latency_ms']:.2f} ms")

# Print audit trail
safegate.print_audit_trail(result['audit_trail'])
```

### Output

```
Risk Tier: R1
Confidence: 1.00
Enforcing Gate: G1
Latency: 1.15 ms

======================================================================
SAFE-GATE CLINICAL TRIAGE REPORT
======================================================================

Patient ID: P001
Timestamp: 2026-01-24T...

FINAL TRIAGE DECISION:
  Risk Tier: R1
  Description: Critical: Life-threatening, immediate care required (<5 min)
  Enforced by: G1

INDIVIDUAL GATE EVALUATIONS:
  G1:
    Tier: R1
    Confidence: 1.00
    Reasoning: Critical flags detected: 2 triggers
    Triggers: Tachycardia: HR 88 > 120 bpm, Neurological red flag: dysarthria

  G3:
    Tier: R5
    Confidence: 1.00
    Reasoning: Sufficient data quality: completeness 100.0% ≥ threshold 95%

THEOREM VERIFICATION:
  theorem2_conservative_bias: ✓ PASS

======================================================================
```

## Architecture

SAFE-Gate operates through a three-phase pipeline:

### Phase 1: Parallel Gate Evaluation (O(1) complexity)

Six independent gates assess distinct safety dimensions:

- **G1 (Critical Flags)**: Rule-based detection of life-threatening red flags
- **G2 (Moderate Risk)**: XGBoost-based weighted risk scoring
- **G3 (Data Quality)**: Completeness assessment with 85% threshold
- **G4 (TiTrATE Logic)**: Clinical decision rules (Timing, Triggers, Targeted Exam)
- **G5 (Uncertainty)**: Monte Carlo dropout uncertainty quantification
- **G6 (Temporal Risk)**: Symptom evolution analysis

### Phase 2: Conservative Merging (O(1) complexity)

Implements minimum lattice selection on risk lattice (R*, ⊑):

```
R* ⊑ R1 ⊑ R2 ⊑ R3 ⊑ R4 ⊑ R5
```

Where:
- R* = Abstention (most conservative)
- R1 = Critical (life-threatening)
- R2 = High Risk (suspected stroke)
- R3 = Moderate (acute vertigo)
- R4 = Low Risk (positional dizziness)
- R5 = Minimal (safe discharge)

**Algorithm:**
1. If any gate outputs R*, return R* (abstention-first priority)
2. Otherwise, return minimum tier: T_final = min_⊑{T1, ..., T6}

### Phase 3: Audit Trail Generation (O(1) complexity)

Documents:
1. Individual gate outputs with confidence scores
2. Features triggering each gate's assessment
3. Merging logic showing enforcing gate
4. Theorem verification status
5. Abstention justification (if R* selected)

## Formal Safety Theorems

SAFE-Gate establishes six mathematical safety properties:

### Theorem 1: No False Discharge
For critical presentations, probability of safe discharge recommendation approaches zero:
```
P(T_final ∈ {R4, R5} | critical(x)) ≤ ε where ε → 0
```

### Theorem 2: Conservative Bias Preservation
Final tier is at least as conservative as every gate output:
```
T_final ⊑ Ti for all gate outputs Ti
```

### Theorem 3: Abstention Correctness
High uncertainty or missing data triggers abstention:
```
(max_i u_i > τ) ∨ (C < C_min) ⇒ T_final = R*
```

### Theorem 4: Monotonicity
Increasing symptom severity produces equal or higher risk tier:
```
s(x') ≥ s(x) ⇒ T(x') ⊑ T(x)
```

### Theorem 5: Data Quality Gate
Missing critical data forces abstention:
```
completeness < 0.85 ⇒ T_final = R*
```

### Theorem 6: Temporal Consistency
Symptom duration incorporated into risk assessment per stroke guidelines.

**Empirical Validation:** Zero violations across all 6,400 synthetic cases.

## Risk Tier Definitions

| Tier | Clinical Presentation | Target Time | Disposition |
|------|----------------------|-------------|-------------|
| R1 (Critical) | Life-threatening, hemodynamic instability | <5 min | Immediate resuscitation |
| R2 (High Risk) | Suspected stroke, focal neurological signs | <15 min | Urgent neurology, imaging |
| R3 (Moderate) | Acute vertigo, concerning features | 30-120 min | HINTS exam, observation |
| R4 (Low Risk) | Episodic positional vertigo | 1-4 hours | Dix-Hallpike, repositioning |
| R5 (Minimal) | Chronic dizziness, stable | Outpatient | Safe discharge |
| R* (Abstain) | Uncertainty/incomplete data | N/A | Mandatory physician review |

## Dataset

Evaluated on 6,400 entirely synthetic dizziness/vertigo presentations:
- **Training:** 4,800 cases (75%)
- **Validation:** 800 cases (12.5%)
- **Test:** 800 cases (12.5%)

**Generation Method:** SynDX methodology combining:
1. Counterfactual reasoning across risk tier boundaries
2. Negative matrix factorization for symptom consistency
3. Formal validation against safety theorems

**Features:** 52 dimensions including:
- Demographics (age, sex)
- Vital signs (BP, HR, SpO2, temperature)
- Symptom characteristics (onset, duration, severity)
- Neurological examination (HINTS protocol, focal deficits)
- Cardiovascular risk factors (hypertension, AF, diabetes)
- Temporal patterns

## Repository Structure

```
SAFE-Gate/
├── src/
│   ├── safegate.py              # Main system
│   ├── gates/                   # Six parallel gates
│   │   ├── gate1_critical_flags.py
│   │   ├── gate2_moderate_risk.py
│   │   ├── gate3_data_quality.py
│   │   ├── gate4_titrate_logic.py
│   │   ├── gate5_uncertainty.py
│   │   └── gate6_temporal_risk.py
│   ├── merging/                 # Conservative merging
│   │   ├── risk_lattice.py
│   │   └── conservative_merging.py
│   ├── theorems/                # Theorem verification
│   └── utils/                   # Audit trail, metrics
├── data/                        # Synthetic datasets
├── notebooks/                   # Reproducibility notebooks
├── tests/                       # Test suite
├── experiments/                 # Results, figures
└── docs/                        # Documentation
```

## Reproducibility

A quickstart Jupyter notebook demonstrates the complete SAFE-Gate workflow:

- **Quickstart Demo:** `notebooks/00_quickstart.ipynb`
  - Loading synthetic test data (804 cases)
  - SAFE-Gate classification with audit trails
  - Baseline method comparison (ESI, XGBoost, Ensemble, Confidence)
  - Performance metrics calculation
  - Batch processing demonstration

This notebook reproduces the key results from the IEEE EMBC 2026 paper including the 95.3% sensitivity, 2.5% improvement over ensemble averaging, and zero theorem violations.

## Citation

```bibtex
@inproceedings{tritham2026safegate,
  title={SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine with parallel GATEs for Medical AI Systems with Provable Safety Guarantees},
  author={Tritham, Chatchai and Namahoot, Chakkrit Snae},
  booktitle={2026 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  year={2026},
  organization={IEEE}
}
```

## Paper Reference

Chatchai Tritham and Chakkrit Snae Namahoot. "SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine with parallel GATEs for Medical AI Systems with Provable Safety Guarantees." *IEEE EMBC 2026* (submitted).

**Preprint:** arXiv:XXXX.XXXXX (to be updated)

## Validation Pathway

### Current Status: Preliminary Computational Study

This repository contains the preliminary validation on synthetic data establishing architectural feasibility and theorem correctness.

### Planned Validation Phases:

**Phase 1: Retrospective Validation (Next Step)**
- Evaluate on de-identified emergency department records
- Compare against ground-truth clinical outcomes (neuroimaging, thrombolysis)
- Measure real-world sensitivity, specificity, abstention rates
- Verify theorems hold on real patient presentations

**Phase 2: Prospective Observational Study**
- Deploy as non-interventional decision support
- Measure clinician-system concordance
- Assess audit trail utility in clinical workflow
- Evaluate physician trust and perceived value

**Phase 3: Randomized Controlled Trial**
- Primary outcome: Time to stroke diagnosis, thrombolysis administration
- Secondary outcomes: 90-day mRankin scale, length of stay, costs
- Definitive evidence for patient outcome improvement

## Limitations

1. **Synthetic Data:** Trained on 6,400 entirely synthetic cases. Real-world validation with de-identified emergency department records required.

2. **Single Domain:** Evaluated only on dizziness/vertigo. Generalization to other presentations (chest pain, dyspnea, abdominal pain) requires domain-specific adaptation.

3. **Computational Feasibility Only:** Establishes architectural feasibility but does not measure real-world clinical impact on patient outcomes.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

**Corresponding Author:** Chakkrit Snae Namahoot (chakkrits@nu.ac.th)

**Institution:** Department of Computer Science and Information Technology
Faculty of Science, Naresuan University
Phitsanulok 65000, Thailand

**GitHub Issues:** https://github.com/ChatchaiTritham/SAFE-Gate/issues

## Acknowledgments

We thank the emergency medicine and neurology domain experts who contributed to risk tier guideline encoding, synthetic data validation, clinical protocol review, and formal theorem verification.

---

**Version:** 1.0.0
**Last Updated:** January 24, 2026
