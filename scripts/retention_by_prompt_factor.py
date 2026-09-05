# -*- coding: utf-8 -*-
"""Retention recomputed within each prompt factor (SI Table S1-8).

Twelve prompts were used so that no single prompt's wording fixes the result.
Equal weighting of the twelve cells makes every prompt count the same; it does
not by itself show that the prompts agree. This script recomputes retention on
the six cells whose prompt supplies no genre definition, the six that do, and
each of the three framings (instruction, completion, persona), so that the
direction result of Section 3.2 can be read condition by condition. Prompt ids
follow SI S5: odd ids supply the definition; P00-P03 instruction, P04-P07
completion, P08-P11 persona.

Run: python3 scripts/retention_by_prompt_factor.py
-> results/phase2/retention_by_prompt_factor.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIVE = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]

g = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
lad = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
g = g[g.model.isin(FIVE)].copy()
tcols = [c for c in g.columns if re.fullmatch(r"T\d+", c)]
pid = g.prompt_id.astype(str).str.extract(r"(\d+)")[0].astype(int)
g["definition"] = np.where(pid % 2 == 1, "definition", "no_definition")
g["framing"] = pd.cut(pid, [-1, 3, 7, 11], labels=["instruction", "completion", "persona"])

def retention(df):
    return df[tcols].mean() * 100 / lad.loc[tcols, "human_mean_pct"]

out = pd.DataFrame({"motif": tcols, "name_en": lad.loc[tcols, "name_en"].values,
                    "all_cells": retention(g).values})
for k, x in g.groupby("definition"):
    out[k] = retention(x).values
for k, x in g.groupby("framing", observed=True):
    out[k] = retention(x).values
out = out.sort_values("all_cells", ascending=False).round(3)
out.to_csv(ROOT / "results/phase2/retention_by_prompt_factor.csv", index=False)
r = out.set_index("motif")
from scipy.stats import spearmanr
print("Spearman(definition, no_definition) =", round(spearmanr(r.definition, r.no_definition).statistic, 3))
for c in ["no_definition", "definition", "instruction", "completion", "persona"]:
    s = r[c].sort_values()
    print(f"{c:14s} above 1: {int((s > 1).sum()):2d} | lowest four: {list(r.loc[s.index[:4], 'name_en'])}")
