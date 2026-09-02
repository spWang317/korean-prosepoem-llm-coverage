"""company_profile.py — summary of the company profiles (paper §2.4.4, Table 4).

Turns company_weighted.json (25 anchors x 24 neighbours, weighted mean shares, human
and generated) into a company profile per anchor (the 24 neighbour shares renormalized
to sum to 1) and reports three parameter-free summaries:
  - tv              : total variation distance between the human and generated profile (0-1)
  - eff_h/eff_g     : effective number of neighbours, 1/sum(p^2) (maximum 24)
  - up_h/up_g       : the share of the increased side (retention > 1, the anchor excluded)
  - avoid_h/avoid_g : the share of the avoided four (wounds, old age, hospital, school)
The groups are only those fixed by the boundary (retention 1) and the qualification
judgment of §2.4.2. No anchor is selected; all 25 are reported.
Run: python3 scripts/company_profile.py -> results/phase2/company_profile.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
cw = json.load(open(ROOT / "results/phase2/company_weighted.json"))["anchors"]
lad = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
UP = set(lad.index[lad["retention"] > 1]); AVOID = {"T18", "T9", "T22", "T5"}
rows = []
for a in sorted(cw, key=lambda m: int(m[1:])):
    nb = cw[a]["neighbors"]; ms = sorted(nb, key=lambda m: int(m[1:]))
    h = np.array([nb[m]["human"] for m in ms], float); g = np.array([nb[m]["gen"] for m in ms], float)
    hp, gp = h / h.sum(), g / g.sum()
    up = [i for i, m in enumerate(ms) if m in UP and m != a]; av = [i for i, m in enumerate(ms) if m in AVOID and m != a]
    rows.append({"anchor": a, "name": lad.loc[a, "name"], "name_en": lad.loc[a, "name_en"], "anchor_share_h": round(lad.loc[a, "human_mean_pct"], 2),
                 "anchor_share_g": round(lad.loc[a, "gen_mean_pct"], 2), "retention": round(lad.loc[a, "retention"], 3),
                 "tv": round(0.5 * np.abs(hp - gp).sum(), 3),
                 "eff_h": round(1 / (hp ** 2).sum(), 2), "eff_g": round(1 / (gp ** 2).sum(), 2),
                 "up_h": round(hp[up].sum() * 100, 2), "up_g": round(gp[up].sum() * 100, 2),
                 "avoid_h": round(hp[av].sum() * 100, 2), "avoid_g": round(gp[av].sum() * 100, 2)})
df = pd.DataFrame(rows).sort_values("anchor_share_g", ascending=False)
df.to_csv(ROOT / "results/phase2/company_profile.csv", index=False)
print(df.to_string(index=False))
