"""closure_audit_sampling.py — draws the sample for the ending audit (paper §2.4.6, §3.6).

Strata: a rain-intensive stratum for each of the five form-compliant systems, Opus
winter and Opus family, Gemini winter, and human rain, family, winter, wounds, old age,
hospital and school -- 15 strata, 12 works each.

The same procedure in every population:
  population = among the protocol-valid works (topic_records.csv; 25 valid per cell),
               those intensive on the motif (share >= 0.10); for human works, T*_pct >= 10.
  order      = the row order of results/phase2/topic_records.csv
               (for human works, human_doc_motif_distribution.csv).
  draw       = random.Random(seed).sample(pop, k)  # a fixed per-stratum seed (42-53);
               strata drawn from the same corpus use different seeds.
The criteria and the judgments are in docs/closure_audit.md.
Run: python3 scripts/closure_audit_sampling.py -> results/phase2/closure_audit_samples.txt
"""
import json, re, random
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEM = {"GPT-5.6": "openai_gpt-5.6", "Opus-4.8": "anthropic_opus-4-8",
        "Gemini-2.5": "google_gemini-2.5-pro", "Solar-Pro3": "upstage_solar-pro3",
        "EXAONE-4.0": "lg_exaone"}
rec = pd.read_csv(ROOT / "results/phase2/topic_records.csv")




# English stratum labels (the row labels of Table 5 of the paper), written beside the Korean
STRATUM_EN = {"GPT 비": "GPT rain-intensive", "Solar 비": "Solar rain-intensive", "EXAONE 비": "EXAONE rain-intensive", "Opus 겨울": "Opus winter", "Opus 가족": "Opus family", "Opus 비": "Opus rain-intensive", "Gemini 비": "Gemini rain-intensive", "Gemini 겨울": "Gemini winter", "인간 비": "Human rain-intensive", "인간 겨울": "Human winter", "인간 가족": "Human family", "인간 상처": "Human wounds", "인간 노인": "Human old age", "인간 병원": "Human hospital", "인간 학교": "Human school"}


def gen_sample(model, tcol, thr, k, seed):
    sub = rec[rec["model"] == model].reset_index(drop=True)
    pop = []
    for _, row in sub[sub[tcol] >= thr].iterrows():
        t = row["text_full"]
        pop.append(t)
    return pop, random.Random(seed).sample(pop, min(k, len(pop)))


def main():
    lines = []
    # A different seed per stratum, so that strata drawn from the same corpus do not
    # re-draw the same positions. Stratum names are the CSV values; see docs/glossary.md.
    groups = [("GPT 비", "GPT-5.6", "T7", 12, 42),
              ("Gemini 비", "Gemini-2.5", "T7", 12, 42),
              ("Solar 비", "Solar-Pro3", "T7", 12, 42),
              ("EXAONE 비", "EXAONE-4.0", "T7", 12, 42),
              ("Opus 비", "Opus-4.8", "T7", 12, 43),
              ("Opus 겨울", "Opus-4.8", "T6", 12, 44),
              ("Opus 가족", "Opus-4.8", "T2", 12, 45),
              ("Gemini 겨울", "Gemini-2.5", "T6", 12, 53)]
    for name, model, tcol, k, seed in groups:
        pop, sample = gen_sample(model, tcol, 0.10, k, seed)
        lines.append(f"\n### {name} / {STRATUM_EN[name]} (population {len(pop)}, sample {len(sample)})\n")
        for i, t in enumerate(sample, 1):
            lines.append(f"[{name}-{i}] …{t[-170:]}\n")
    h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
    meta = pd.read_csv(ROOT / "data/atlas/metadata.csv")
    h["work_id"] = h["work_id"].astype(str); meta["work_id"] = meta["work_id"].astype(str)
    d = h.merge(meta, on="work_id")
    for tcol, k, name, seed in [("T7_pct", 12, "인간 비", 46), ("T2_pct", 12, "인간 가족", 47), ("T6_pct", 12, "인간 겨울", 48),
                                ("T18_pct", 12, "인간 상처", 49), ("T9_pct", 12, "인간 노인", 50), ("T22_pct", 12, "인간 병원", 51), ("T5_pct", 12, "인간 학교", 52)]:
        sub = d[d[tcol] >= 10]
        idx = random.Random(seed).sample(range(len(sub)), k)
        lines.append(f"\n### {name} / {STRATUM_EN[name]} (population {len(sub)}, sample {k})\n")
        for i, j in enumerate(idx, 1):
            row = sub.iloc[j]; t = str(row["원문"])
            lines.append(f"[{name}-{i}] {row['작가']} 「{row['작품 제목']}」 …{t[-170:]}\n")
    out = ROOT / "results/phase2/closure_audit_samples.txt"
    out.write_text("".join(lines), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
