# Changelog

## v2.0.0-ijfs (round-4 revision)

- Audit trail: CP/AC/CND are now computed per decision (`ConservativeMerging.check_properties`); they were previously written as constants.
- Counterfactual explanation record: removed the invented fallback value and the silent exception handler; unsearched or censored values are reported as `None`.
- Dempster-Shafer and BMA: ties within relative tolerance 1e-9 are treated as ties (24 exact ties in the conflict grid, previously 16 detected).
- New baselines: normalised weighted max, k-th order statistic, quantifier-guided OWA, band-clamped Sugeno, trained symmetric-capacity Sugeno, Yager prioritized aggregation, zero-order TSK rule base, belief-function variants (Yager, Dubois-Prade, PCR6, cautious, consonant coding, min expected cost, discounting, vacuous abstention), safety-thresholded XGBoost, ordinal logistic regression.
- ACWCM fuzzy-quantifier trigger (`trigger="fuzzy_quantifier"`).
- Diagnosis-conditioned generator `data/generation/generate_diagnosis.py`.
- `scripts/round4.py` (frozen protocol, fresh cohorts, multi-seed, profile analysis, extended enumeration, conflict grids); `run_all.py --round4`.
- Pinned environment `requirements.lock`; Dockerfile pinned to Python 3.11.15.
- G1 metadata: `n_rules` corrected to 20.

## Earlier

- Standardized root metadata, authorship, reference, testing, and container files.
