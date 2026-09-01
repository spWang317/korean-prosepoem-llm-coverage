"""closure_audit_sampling.py — 결말 감사(본문 §2.4.6·§3.6)의 표본 추출.

층: 형식을 지키는 다섯 시스템의 비-집약 층, Opus의 겨울·가족 층, Gemini의 겨울 층, 인간의 비·가족·겨울·상처·노인·병원·학교 층(총 15층, 층당 12편).

전 집단 공통 절차:
  모집단 = 프로토콜 유효물(topic_records.csv; 셀별 유효 25편) 중 해당 모티프-집약(지분>=0.10) / 인간은 T*_pct>=10.
  순서   = results/phase2/topic_records.csv 행 순서(인간은 human_doc_motif_distribution.csv 행 순서).
  추출   = random.Random(seed).sample(pop, k)  # 층별 고정 시드(42–53); 같은 코퍼스에서 뽑는 층끼리는 서로 다른 시드
분류 기준과 판정 결과는 docs/closure_audit.md에 있다.
실행: python3 scripts/closure_audit_sampling.py  -> results/phase2/closure_audit_samples.txt
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




def gen_sample(model, tcol, thr, k, seed):
    sub = rec[rec["model"] == model].reset_index(drop=True)
    pop = []
    for _, row in sub[sub[tcol] >= thr].iterrows():
        t = row["text_full"]
        pop.append(t)
    return pop, random.Random(seed).sample(pop, min(k, len(pop)))


def main():
    lines = []
    # 층마다 다른 시드(같은 코퍼스에서 뽑는 층들이 같은 위치를 재추출하지 않도록).
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
        lines.append(f"\n### {name} (모집단 {len(pop)}, 표본 {len(sample)})\n")
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
        lines.append(f"\n### {name} (모집단 {len(sub)}, 표본 {k})\n")
        for i, j in enumerate(idx, 1):
            row = sub.iloc[j]; t = str(row["원문"])
            lines.append(f"[{name}-{i}] {row['작가']} 「{row['작품 제목']}」 …{t[-170:]}\n")
    out = ROOT / "results/phase2/closure_audit_samples.txt"
    out.write_text("".join(lines), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
