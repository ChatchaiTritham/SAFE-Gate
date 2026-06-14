# Reproducibility

A single deterministic driver, `run_all.py`, reproduces every headline number in
the manuscript from source. It regenerates the seeded synthetic cohort, classifies
the held-out test split with SAFE-Gate, runs the comparison baselines, and verifies
the formal safety properties.

```bash
pip install -e .
python run_all.py
```

All randomness is seeded (cohort generation `seed=42`; per-case gate inference is
seeded from each case's clinical signature). Re-running produces **byte-identical**
files in `results/`. No metric is hardcoded; every value below is computed by the
driver and written to `results/`.

## Cohort

| Quantity | Value |
|---|---|
| Total cases | 6,400 |
| Train / Validation / Test | 4,797 / 798 / 805 |
| Features per case | 52 |
| Tier distribution (test) | R1 48, R2 127, R3 300, R4 222, R5 108 |

The synthetic generator (`data/generation/generate_data.py`) draws tier-conditioned
feature vectors that are internally consistent with the assigned tier. Because the
generation templates and the gate logic encode the same published clinical knowledge
(AHA/ASA red flags, HINTS, TiTrATE timing), the evaluation is subject to the
synthetic-data circularity disclosed as a limitation in the manuscript.

## What reproduces — the safety guarantee (the central result)

| Result | `run_all.py` output |
|---|---|
| Critical-tier sensitivity (R1/R2) | **100.0%** (175/175), 95% CI 97.9–100% |
| False discharges (true critical → R5) | **0** |
| False negative rate | **0.0%** |
| Formal safety-property violations | **0 / 805** (conservative preservation, abstention correctness, critical non-dilution, no false discharge) |

Every truly critical case is kept at a critical tier, no critical case is routed to
discharge, and all four formally specified invariants hold with zero violations.

## Measured performance (point estimates, ACWCM, 805-case test split)

| Metric | Basic MIN | ACWCM |
|---|---|---|
| Overall accuracy | 33.8% | 29.9% |
| Macro F1 | — | 19.2% |
| Discharge specificity (R5 recall) | 0.0% | 0.0% |
| Over-triage rate | 66.2% | 42.2% |
| Abstention rate | 0.0% | 0.0% |

Per-tier recall (ACWCM): R1 100%, R2 0%, R3 0%, R4 86.9%, R5 0%.

These figures are reported as the genuine output of the released implementation.
The conservative merging rule keeps every critical case critical (the design goal),
but at the cost of pervasive over-triage: R2, R3 and R5 are seldom or never assigned,
so discharge specificity is 0% and overall accuracy is low. The reproducible,
defensible contribution is the **safety floor**, not high aggregate accuracy.

## Comparison baselines (critical-tier sensitivity)

| Method | Sensitivity | Missed critical | Accuracy |
|---|---|---|---|
| Arithmetic ensemble averaging | 28.0% | 126 | 35.2% |
| Dempster–Shafer | 100.0% | 0 | 34.8% |
| Bayesian model averaging | 98.9% | 2 | 42.2% |
| Single XGBoost | 100.0% | 0 | 100.0% |
| SAFE-Gate (ACWCM) | 100.0% | 0 | 29.9% |

On this internally consistent synthetic cohort, Dempster–Shafer also attains 100%
critical sensitivity and the single XGBoost classifier separates the tiers almost
perfectly — consistent with the manuscript's caution that synthetic class boundaries
are sharper than real emergency-department records. The distinguishing property of
the lattice approach is therefore the *formally guaranteed* zero-violation safety
floor with per-case audit certificates, not empirical accuracy superiority.

## Artifacts

| File | Contents |
|---|---|
| `results/summary.json` | All headline metrics + cohort + safety properties |
| `results/confusion_matrix.csv` | ACWCM true × predicted confusion matrix |
| `results/tier_metrics.csv` | Per-tier precision / recall / specificity / F1 |
| `results/baseline_comparison.csv` | Baseline fusion-method comparison |
| `results/theorem_verification.json` | Formal safety-property verification |

`scripts/run_evaluation.py` remains as a lightweight safety-only driver (confusion
matrix + critical sensitivity); `run_all.py` supersedes it for the full pipeline.
