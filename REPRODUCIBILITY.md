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

## Calibration protocol

The gate calibration (v2) was selected on the training split by
`scripts/select_calibration.py` (128 discrete profiles; hard constraints
confirmed on validation) and recorded in `results/calibration_protocol.json`,
committed before the test split was scored. `run_all.py` refuses to run if
`src/calibration.py` disagrees with that file. The earlier calibration (v1) is
reported as a sensitivity row.

## Measured performance (805-case test split)

| Metric | Basic MIN | ACWCM (v2) | ACWCM (v1) |
|---|---|---|---|
| Overall accuracy | 79.0% | 82.5% | 36.8% |
| Macro F1 | 70.9% | 86.3% | 26.2% |
| Discharge recall (R5) | 0.0% | 66.7% | 0.0% |
| Over-triage | 21.0% | 6.2% | 42.2% |
| Under-triage (all one tier) | 0.0% | 11.3% | 21.0% |

## Comparison baselines

| Method | Sensitivity | Accuracy | Over-triage | Under-triage (>=2 tiers) |
|---|---|---|---|---|
| Arithmetic ensemble averaging | 28.0% | 35.2% | 13.7% | 51.2% (0) |
| Dempster-Shafer | 100.0% | 83.2% | 5.7% | 11.1% (83) |
| Bayesian model averaging | 100.0% | 87.7% | 0.0% | 12.3% (83) |
| Sugeno integral (additive capacity) | 100.0% | 69.2% | 13.3% | 17.5% (0) |
| Ordinal weighted maximum | 100.0% | 79.9% | 20.1% | 0.0% (0) |
| Single XGBoost | 100.0% | 100.0% | 0.0% | 0.0% (0) |
| SAFE-Gate (ACWCM) | 100.0% | 82.5% | 6.2% | 11.3% (0) |

Also regenerated: conflicting-evidence enumeration (`results/conflicting_evidence.*`),
exhaustive invariant enumeration (`results/invariant_enumeration.*`), stress tests
(`results/stress_tests.*`: label noise, atypical presentations, MCAR missingness,
independent generator), subgroup and age diagnostics. Runtime about one minute.

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
