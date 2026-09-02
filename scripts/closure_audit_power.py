# -*- coding: utf-8 -*-
"""Detection threshold of the ending audit's per-stratum test (SI S8).

Each stratum holds 12 items and a generated stratum is compared with its paired human
stratum by a one-sided Fisher exact test (generated > human). This script tabulates,
for a paired human sealing count of 0, 1 or 2 out of 12, the one-sided p value for
every possible generated sealing count k = 0..12, and the smallest k that reaches
alpha = 0.05. It is a property of the design (n = 12 per stratum), not of the data.

Run: python3 scripts/closure_audit_power.py
-> results/phase2/closure_audit_power.csv
"""
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
N = 12
ALPHA = 0.05

rows = []
for human_seal in (0, 1, 2):
    for k in range(N + 1):
        p = fisher_exact([[k, N - k], [human_seal, N - human_seal]], alternative="greater")[1]
        rows.append({"n_per_stratum": N, "human_seal": human_seal, "generated_seal": k,
                     "p_one_sided": round(float(p), 4), "significant_at_0.05": bool(p < ALPHA)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/phase2/closure_audit_power.csv", index=False)

for human_seal, g in df.groupby("human_seal"):
    kmin = g[g["significant_at_0.05"]].generated_seal.min()
    print(f"paired human {human_seal}/{N}: smallest significant generated count = {kmin}/{N} "
          f"(p = {g[g.generated_seal == kmin].p_one_sided.iloc[0]})")
