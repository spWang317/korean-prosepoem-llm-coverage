"""phase2f_concentration.py — concentration of each corpus on the 25 instrument motifs (paper §3.3, Table 3).

For the human corpus and each system, takes the mean 25-motif share vector (results/phase2/
motif_full_distribution_balanced.csv for labels; shares are taken unrounded from topic_records.csv, human shares from the frozen instrument assets), renormalises it within the
25 motifs, and reports the largest motif's share, the top-3 share, and the effective number of
motifs 1/Σp² (inverse Simpson). Writes results/phase2/concentration_two_axes.csv.
Run: python3 scripts/phase2f_concentration.py
"""
import pandas as pd, numpy as np, json
from pathlib import Path
R = Path(__file__).resolve().parent.parent
sm = json.load(open(R / "results/phase2/summary.json"))
resid = {pm["model"]: pm["unexplained_mean"] * 100 for pm in sm["per_model"]}
hdist = R / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv"
resid["human"] = float(pd.read_csv(hdist)["unexplained_pct"].mean()) if hdist.exists() else float("nan")  # human residual needs the frozen instrument assets
f = pd.read_csv(R / "results/phase2/motif_full_distribution_balanced.csv")
kw = {r["motif"]: r["topwords"] for _, r in f.iterrows()}
mods = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0", "HCX-005"]
rec = pd.read_csv(R / "results/phase2/topic_records.csv")
T = [f"T{j}" for j in range(25)]
vecs = {m: rec[rec.model == m][T].mean().values for m in mods}
if hdist.exists():
    hd = pd.read_csv(hdist); vecs["human"] = hd[[t + "_pct" for t in T]].mean().values / 100
else:  # frozen instrument assets absent: fall back to the rounded balanced table
    vecs["human"] = f["human_%"].values / 100
rows = []
for m in ["human"] + mods:
    v = vecs[m]; p = v / v.sum(); o = np.argsort(p)[::-1]
    rows.append({"model": m,
                 "fx25_top1_motif": f.iloc[o[0]]["motif"] + " " + kw[f.iloc[o[0]]["motif"]].split(",")[0].strip(),
                 "fx25_top1_pct": round(p[o[0]] * 100, 2), "fx25_top3_pct": round(p[o[:3]].sum() * 100, 2),
                 "fx25_eff_ntopics": round(1 / np.sum(p ** 2), 2), "residual_pct": round(resid[m], 2)})
pd.DataFrame(rows).to_csv(R / "results/phase2/concentration_two_axes.csv", index=False)
print("saved results/phase2/concentration_two_axes.csv")
