"""retention_ladder.py — retention for the 25 motifs (paper §2.4.2, Table 2, SI Tables S1-1 and S1-3).

The published artefact of the direction measurement (§2.4.2): for each of the 25
motifs, the human mean share, the generated mean share (the five form-compliant
systems pooled), retention (= generated / human, computed before rounding), the
per-system retentions and their maximum. The set-qualification judgment of §3.2
(adjacent gap, per-system separation) and the retention column of SI Table S1-1 come
from this file.
Run: python3 scripts/retention_ladder.py -> results/phase2/retention_ladder.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIVE = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]
# Motif names are written into the CSV in Korean; see docs/glossary.md for the
# English terms used in the paper.
NAMES = {0: "바다", 1: "꽃", 2: "가족", 3: "음식·동물", 4: "정서", 5: "학교·아이",
         6: "겨울", 7: "비", 8: "여인", 9: "노인·지하", 10: "새·나비", 11: "메타시",
         12: "편지·글", 13: "실내", 14: "음악", 15: "거리", 16: "고양이", 17: "우주",
         18: "상처·술", 19: "과일", 20: "조부모", 21: "거울", 22: "병원·질병",
         23: "아파트·벽", 24: "버스"}


def main():
    rec = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
    h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
    gen = rec[rec["model"].isin(FIVE)]
    tcols = sorted([c for c in rec.columns if c.startswith("T") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    rows = []
    for c in tcols:
        hm = h[f"{c}_pct"].mean()
        gm = gen[c].mean() * 100
        per = {m: (rec[rec["model"] == m][c].mean() * 100) / hm for m in FIVE}
        rows.append({"motif": c, "name": NAMES[int(c[1:])], "human_mean_pct": round(hm, 6),
                     "gen_mean_pct": round(gm, 6), "retention": round(gm / hm, 6),
                     **{f"retention_{m}": round(v, 6) for m, v in per.items()},
                     "retention_system_max": round(max(per.values()), 6)})
    df = pd.DataFrame(rows).sort_values("retention", ascending=False)
    out = ROOT / "results/phase2/retention_ladder.csv"
    df.to_csv(out, index=False)
    print(df[["motif", "name", "human_mean_pct", "gen_mean_pct", "retention",
              "retention_system_max"]].to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
