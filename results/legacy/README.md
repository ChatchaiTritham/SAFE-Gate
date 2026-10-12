# Legacy outputs (pre-round-4, development split only)

These files were written by the default (pre-round-4) mode of `run_all.py` on the
development split of generator seed 42. They predate the frozen round-4 protocol
(commit 03b0adb) and are NOT the source of any number in the IJAR manuscript;
some of them (e.g. `baseline_comparison.csv`) disagree with the published results
because they use earlier calibrations and baselines. They are kept for history only.

Canonical outputs: `results/round4_primary.csv` (primary, frozen cohorts),
`results/round4_summary.json`, `results/round4_*.csv`, and the post-hoc
secondary analyses `results/ijar_r3_*` (scripts/ijar_r3_analyses.py).
Re-running `python run_all.py` without `--round4` writes fresh copies of these
legacy files to `results/`.
