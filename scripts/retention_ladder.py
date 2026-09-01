"""retention_ladder.py — 25개 모티프 유지율 산출(본문 §2.4.2, 표 2, SI 표 S1-1·S1-3).

방향 측정(§2.4.2)의 공시 산출물: 25개 모티프 각각의 인간 평균 지분, 생성 평균 지분
(형식을 지키는 다섯 시스템 합), 유지율(= 생성 ÷ 인간, 반올림 전 값으로 계산),
시스템별 유지율과 그 최대. §3.2의 집합 자격 판정(인접 간격·시스템별 분리)과
SI 표 S1-1의 유지율 열이 이 파일에서 나온다.
실행: python3 scripts/retention_ladder.py -> results/phase2/retention_ladder.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIVE = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]
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
