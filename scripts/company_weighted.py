"""company_weighted.py — company analysis (paper §2.4.4): neighbour motif shares averaged with the anchor share as weight.

A weighted mean of the neighbour motif shares, the weight being the anchor motif
share. No truncation, no threshold, no selection of targets: all 25 anchors x (sum
over the avoided four, sum over the eight with retention > 1, the full per-neighbour
matrix) x (human, generated).
Run: python3 scripts/company_weighted.py -> results/phase2/company_weighted.json
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOC = ["T5", "T9", "T18", "T22"]  # the avoided four (set qualification, paper §3.2.2)
ATM = ["T7", "T15", "T6", "T17", "T13", "T16", "T23", "T12"]  # the eight with retention > 1 (paper §3.2.1; a continuum)

rec = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
gen = rec[rec["model"] != "HCX-005"]


def wmean(df, anchor, targets, pct=False):
    w = df[anchor + "_pct"] / 100 if pct else df[anchor]
    tgt = df[[t + "_pct" for t in targets]].sum(axis=1) if pct else df[targets].sum(axis=1) * 100
    return float((w * tgt).sum() / w.sum())


out = {"note": "Weighted company: neighbour shares averaged with the anchor share as weight (paper 2.4.4; source of Table 4).",
       "anchors": {}, "rain_family_by_system": {}}
for i in range(25):
    t = f"T{i}"
    soc = [s for s in SOC if s != t]
    atm = [a for a in ATM if a != t]
    out["anchors"][t] = {
        "soc_sum": {"human": round(wmean(h, t, soc, True), 2), "gen": round(wmean(gen, t, soc), 2)},
        "atm_sum": {"human": round(wmean(h, t, atm, True), 2), "gen": round(wmean(gen, t, atm), 2)},
        "neighbors": {f"T{j}": {"human": round(wmean(h, t, [f"T{j}"], True), 2),
                                "gen": round(wmean(gen, t, [f"T{j}"]), 2)}
                      for j in range(25) if j != i},
    }
out["rain_family_by_system"]["human"] = round(wmean(h, "T7", ["T2"], True), 2)
for m in ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]:
    out["rain_family_by_system"][m] = round(wmean(rec[rec["model"] == m], "T7", ["T2"]), 2)
json.dump(out, open(ROOT / "results/phase2/company_weighted.json", "w"), ensure_ascii=False, indent=1)
print("saved company_weighted.json")
