"""human_motif_ci.py — 인간 25모티프 평균 지분의 부트스트랩 95% CI (2,000회, 시드 42).
SI 표 S1-2(표 2 캡션이 가리키는 인간 평균의 신뢰구간)의 산출 산출물. -> results/phase2/human_motif_ci.json"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
cols = [f"T{i}_pct" for i in range(25)]
X = h[cols].values
rng = np.random.default_rng(42)
n = len(X)
boots = np.array([X[rng.integers(0, n, n)].mean(0) for _ in range(2000)])
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
out = {f"T{i}": {"mean": round(float(X[:, i].mean()), 3),
                 "ci95": [round(float(lo[i]), 3), round(float(hi[i]), 3)]} for i in range(25)}
json.dump(out, open(ROOT / "results/phase2/human_motif_ci.json", "w"), ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k in ("T3","T9","T5","T22","T18")})
