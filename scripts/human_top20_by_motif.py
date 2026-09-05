# -*- coding: utf-8 -*-
"""Bibliography of the twenty human works loading most on each motif (SI Table S2-3).

The full text of the human works is not redistributed, so the 25 motif labels
cannot be checked against the poems inside this package. This table lets a
reader with library access do so: for each motif, the twenty human works with
the highest share of that motif, with poet, title, year, publisher and share.
Needs the bibliographic columns of data/atlas/metadata.csv (not included);
the output is released.

Run: python3 scripts/human_top20_by_motif.py
-> results/phase2/human_top20_by_motif.csv
"""
from pathlib import Path
import re
import pandas as pd
from common import ATLAS_META

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT = ROOT / "data/atlas/instrument_frozen"
TOP = 20

h = pd.read_csv(INSTRUMENT / "human_doc_motif_distribution.csv")
meta = pd.read_csv(ATLAS_META)[["work_id", "작가", "작품 제목", "출판년도", "출판사"]]
names = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
d = h.merge(meta, on="work_id")
cols = [c for c in h.columns if re.fullmatch(r"T\d+_pct", c)]

rows = []
for c in cols:
    t = c[:-4]
    for r, (_, x) in enumerate(d.sort_values(c, ascending=False).head(TOP).iterrows(), 1):
        rows.append({"motif": t, "name": names.loc[t, "name"], "name_en": names.loc[t, "name_en"], "rank": r,
                     "work_id": x.work_id, "poet": x["작가"], "title": x["작품 제목"],
                     "year": int(x["출판년도"]), "publisher": x["출판사"], "share_pct": round(float(x[c]), 2)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/phase2/human_top20_by_motif.csv", index=False)
print(len(out), "rows;", out.motif.nunique(), "motifs")
