# SAFE-Gate Figure Generation Summary

**Repository:** SAFE-Gate (reproducibility companion)
**Purpose:** How the manuscript figures are produced from committed results.

---

## Overview

Every manuscript figure is rendered **from the committed result artifacts** produced
by the seed-42 reproducibility driver — no numbers are typed into the figure code.
This document does not imply manuscript review, acceptance, or publication.

> **History.** Two earlier scripts (`evaluation/create_manuscript_figures.py` and
> `evaluation/generate_performance_metrics.py`) plotted hardcoded/typed-in
> "expected-behaviour" numbers and straw-man baselines that did not match the model
> output. They were removed (commit `82a2a77`). The single honest generator below
> replaces both.

---

## Pipeline (2 steps)

```bash
# 1. Run the reproducibility driver (writes results/*.csv + results/summary.json, seed 42)
python scripts/run_all.py

# 2. Render every figure from those committed artifacts
python evaluation/generate_figures_from_results.py
```

### `evaluation/generate_figures_from_results.py`

**Inputs (read from `results/`):**
- `confusion_matrix.csv` — true × pred tier counts
- `tier_metrics.csv` — per-tier precision / recall / specificity / f1
- `baseline_comparison.csv` — method × {critical sensitivity, accuracy, R5 specificity, …}
- `summary.json` — headline ACWCM metrics + ablation + safety properties

**Output:** `evaluation/manuscript_figures/*.png|.pdf` (300 DPI + vector PDF)

**Integrity note (honest reporting):** the reproduced ACWCM model collapses to three
active tiers — R2, R3, R5 receive recall = 0 (R3 is never predicted). The figures do
**not** present R2/R3/R5 as functional tiers: the confusion matrix and per-tier panel
carry an explicit zero-recall annotation, and the headline critical-sensitivity claim
is shown with a Wilson binomial CI because it is a zero-event (0 missed of 175)
estimate. No safety number is altered.

---

## Provenance guarantee

Because figure values are loaded from `results/` (not embedded in the plotting code),
re-running `scripts/run_all.py` on seed 42 regenerates identical inputs, and the
figures track the data exactly. Any future change to the model is reflected in the
figures automatically on the next run.
