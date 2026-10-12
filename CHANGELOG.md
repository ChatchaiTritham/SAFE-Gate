# Changelog

## v2.0.1-ijfs (round-5 pre-submission audit)

No change to scoring code, generators or `results/`; the frozen round-4 cohorts were not re-scored.

- `scripts/make_supplement.py` and `scripts/supplement_template.tex`: the supplement generator is now in the repository.
- `scripts/check_manuscript_numbers.py`: 133 checks (was 69), covering every row of the main Tables 2-4, captions, the plain-text abstract, the cover letter and a byte-for-byte regeneration of the supplement.
- `evaluation/generate_manuscript_figures.py`: sans-serif 8 pt lettering (IJFS artwork rules), panel letters instead of in-figure titles, labels aligned with the manuscript (G2 point score, G5 surrogate, counterfactual explanation record, R1 routed to R5), arithmetic averaging dropped from the dilution figure.
- `src/merging/conservative_merging.py`: corrected a stale comment that called the class-default trigger parameters the untuned choice; the reported calibration v2 uses min_mass = 2/3.

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
