"""company_weighted.py — 동행 분석(본문 §2.4.4): 이웃 모티프 지분의 앵커 지분 가중 평균.

이웃 모티프 지분의 가중 평균(가중치 = 앵커 모티프 지분). 절단·임계·대상 선택 없음:
25개 앵커 전수 × (회피 넷 합, 유지율>1 여덟 합, 전 이웃 개별 행렬) × (인간, 생성).
Run: python3 scripts/company_weighted.py -> results/phase2/company_weighted.json
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOC = ["T5", "T9", "T18", "T22"]  # 회피 넷(본문 §3.2.2의 집합 자격 판정)
ATM = ["T7", "T15", "T6", "T17", "T13", "T16", "T23", "T12"]  # 유지율>1인 여덟(본문 §3.2.1; 연속체)

rec = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
gen = rec[rec["model"] != "HCX-005"]


def wmean(df, anchor, targets, pct=False):
    w = df[anchor + "_pct"] / 100 if pct else df[anchor]
    tgt = df[[t + "_pct" for t in targets]].sum(axis=1) if pct else df[targets].sum(axis=1) * 100
    return float((w * tgt).sum() / w.sum())


out = {"note": "가중 동행: 이웃 지분을 앵커 지분으로 가중 평균(본문 §2.4.4, 표 4의 원천).",
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
