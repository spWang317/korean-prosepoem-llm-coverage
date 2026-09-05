# -*- coding: utf-8 -*-
"""Centroid-keyword alignment of the frozen instrument (SI Table S2-4).

A reader without the human corpus can still check that the 25 frozen motif
centroids agree with the keywords that name them. For each motif the ten
c-TF-IDF keywords (SI Table S2-2) are joined into one string, embedded with
the same encoder as the instrument (ko-sroberta-multitask), and compared by
cosine with all 25 centroids. The table reports, per motif, the cosine with
its own centroid, the rank of its own centroid among the 25, and the nearest
other motif. Runs from the released package alone.

Run: python3 scripts/centroid_keyword_alignment.py
-> results/phase2/centroid_keyword_alignment.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_embedder, encode_texts, EMBED_MODEL

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT = ROOT / "data/atlas/instrument_frozen"

z = np.load(INSTRUMENT / "motif_centroids.npz", allow_pickle=True)
C = z["centroids"].astype(float)
C = C / np.linalg.norm(C, axis=1, keepdims=True)
ids = [int(t) for t in z["topic_ids"]]
names = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
topwords = pd.read_csv(ROOT / "results/phase2/motif_map.csv").set_index("motif")["topwords"]

texts = [" ".join(k.strip() for k in topwords[f"T{t}"].split(",")) for t in ids]
emb = load_embedder(EMBED_MODEL)
K = encode_texts(emb, texts, EMBED_MODEL)
K = K / np.linalg.norm(K, axis=1, keepdims=True)
S = K @ C.T                                   # rows: keyword strings, cols: centroids

rows = []
for i, t in enumerate(ids):
    order = np.argsort(-S[i])
    own = S[i, i]
    rank = int(np.where(order == i)[0][0]) + 1
    other = [j for j in order if j != i][0]
    rows.append({"motif": f"T{t}", "name": names.loc[f"T{t}", "name"], "name_en": names.loc[f"T{t}", "name_en"],
                 "cos_own_centroid": round(float(own), 3), "rank_of_own": rank,
                 "nearest_other": f"T{ids[other]} {names.loc[f'T{ids[other]}', 'name_en']}",
                 "cos_nearest_other": round(float(S[i, other]), 3),
                 "margin": round(float(own - S[i, other]), 3)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/phase2/centroid_keyword_alignment.csv", index=False)
print(df.to_string(index=False))
print(f"\nown centroid ranked first: {int((df.rank_of_own == 1).sum())}/25; "
      f"mean own cosine {df.cos_own_centroid.mean():.3f}, mean off-diagonal {float((S.sum() - np.trace(S)) / (25 * 24)):.3f}")
