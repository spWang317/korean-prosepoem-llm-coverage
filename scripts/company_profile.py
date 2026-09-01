"""company_profile.py — 동행 분석의 이웃 프로필 요약(본문 §2.4.4, 표 4).

company_weighted.json(25앵커 × 24이웃의 가중 평균 지분, 인간·생성)을 앵커마다 이웃 프로필
(이웃 24개의 지분을 합 1로)로 바꾸고 파라미터 없는 요약 셋을 낸다:
  - tv          : 인간·생성 이웃 프로필의 총변동 거리(0–1)
  - eff_h/eff_g : 이웃의 유효 수 1/Σp²(최대 24)
  - up_h/up_g   : 이웃 가운데 늘어난 쪽(유지율>1, 앵커 자신 제외)의 몫
  - avoid_h/avoid_g : 이웃 가운데 회피 넷(상처·노인·병원·학교)의 몫
집합은 §2.4.2가 경계(유지율 1)와 자격 판정으로 정한 것만 쓴다. 앵커 선택 없음(25개 전부).
실행: python3 scripts/company_profile.py -> results/phase2/company_profile.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
cw = json.load(open(ROOT / "results/phase2/company_weighted.json"))["anchors"]
lad = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
UP = set(lad.index[lad["retention"] > 1]); AVOID = {"T18", "T9", "T22", "T5"}
rows = []
for a in sorted(cw, key=lambda m: int(m[1:])):
    nb = cw[a]["neighbors"]; ms = sorted(nb, key=lambda m: int(m[1:]))
    h = np.array([nb[m]["human"] for m in ms], float); g = np.array([nb[m]["gen"] for m in ms], float)
    hp, gp = h / h.sum(), g / g.sum()
    up = [i for i, m in enumerate(ms) if m in UP and m != a]; av = [i for i, m in enumerate(ms) if m in AVOID and m != a]
    rows.append({"anchor": a, "name": lad.loc[a, "name"], "anchor_share_h": round(lad.loc[a, "human_mean_pct"], 2),
                 "anchor_share_g": round(lad.loc[a, "gen_mean_pct"], 2), "retention": round(lad.loc[a, "retention"], 3),
                 "tv": round(0.5 * np.abs(hp - gp).sum(), 3),
                 "eff_h": round(1 / (hp ** 2).sum(), 2), "eff_g": round(1 / (gp ** 2).sum(), 2),
                 "up_h": round(hp[up].sum() * 100, 2), "up_g": round(gp[up].sum() * 100, 2),
                 "avoid_h": round(hp[av].sum() * 100, 2), "avoid_g": round(gp[av].sum() * 100, 2)})
df = pd.DataFrame(rows).sort_values("anchor_share_g", ascending=False)
df.to_csv(ROOT / "results/phase2/company_profile.csv", index=False)
print(df.to_string(index=False))
