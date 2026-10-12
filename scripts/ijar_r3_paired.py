"""Descriptive paired comparisons of ACWCM (v2) with the exact Dempster rule
(the main-text DS row of the IJAR revision) on the two fresh cohorts that do not
share the development generator. Not part of the Holm family of the frozen
protocol. Output: results/ijar_r3_paired.json. Run: python scripts/ijar_r3_paired.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round4 as r4  # noqa: E402
import ijar_r3_analyses as a3  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

proto, cohorts = r4.load_frozen()
rules = a3.make_rules()
out = {}
for name in ("fresh_independent_generator", "fresh_diagnosis_generator"):
    cs = cohorts[name]
    io = r4.gate_io(cs)
    acw = [fin for _, _, fin in io]
    ds = [rules["DS Dempster exact (BetP)"](o, c) for o, c, _ in io]
    crit = [i for i, c in enumerate(cs) if c["ground_truth_tier"] in r4.CRIT]
    ka = [acw[i] in ("R*", "R1", "R2") for i in crit]
    kd = [ds[i] in ("R*", "R1", "R2") for i in crit]
    n10 = sum(x and not y for x, y in zip(ka, kd))
    n01 = sum(y and not x for x, y in zip(ka, kd))
    ca = [p == c["ground_truth_tier"] for p, c in zip(acw, cs)]
    cd = [p == c["ground_truth_tier"] for p, c in zip(ds, cs)]
    out[name] = {"critical_kept_diff_pct_newcombe": r4.newcombe_paired(ka, kd),
                 "critical_acwcm_only": n10, "critical_ds_only": n01,
                 "critical_p_exact_mcnemar": float(f"{binomtest(n10, n10 + n01, 0.5).pvalue:.4g}") if n10 + n01 else 1.0,
                 "accuracy_diff_pct_newcombe": r4.newcombe_paired(ca, cd)}
(r4.RES / "ijar_r3_paired.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
