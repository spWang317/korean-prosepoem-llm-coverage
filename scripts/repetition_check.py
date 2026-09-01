"""repetition_check.py — literal repetition among protocol-valid outputs (paper §3.3, SI Table S1-4).

For each system, takes the balanced valid set (<=25 per prompt cell, same window as
phase2_topic_full.py) and counts distinct full texts, distinct 60-character openings and the
frequency of the most common opening. Writes results/phase2/repetition_check.csv.
Run: python3 scripts/repetition_check.py
"""
from pathlib import Path
from collections import Counter
import pandas as pd
from phase2_topic_full import load_valid_balanced, MODELS
R = Path(__file__).resolve().parent.parent
rows = []
for stem, label in MODELS.items():
    texts, *_ = load_valid_balanced(stem)
    heads = Counter(t[:60] for t in texts)
    rows.append({"system": label, "valid_n": len(texts), "distinct_full": len(set(texts)),
                 "distinct_head60": len(heads), "max_head_freq": max(heads.values())})
pd.DataFrame(rows).to_csv(R / "results/phase2/repetition_check.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
