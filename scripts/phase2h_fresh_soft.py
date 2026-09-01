"""phase2h_fresh_soft.py — 생성 축(모티프 15개) 위의 NNLS 투영(본문 §2.4.5·§3.5, SI 표 S3-2).

생성 축 모티프 중심(발견 군집의 정규화 평균 임베딩)에 인간 축과 같은 NNLS 투영을 적용한다:
(i) 생성물 1,625편, (ii) 인간 4,642편. 본문이 쓰는 값은 두 코퍼스의 잔차(두 축의 비대칭 점검)다.
Run: python3 scripts/phase2h_fresh_soft.py -> results/phase2/fresh_soft_projection.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
OUT2C = ROOT / "results/phase2c"

# ---------- generated docs: same order as phase2c ----------
import importlib.util
spec = importlib.util.spec_from_file_location("p2c", ROOT / "scripts/phase2c_bertopic_fresh.py")
p2c = importlib.util.module_from_spec(spec)
import sys
sys.modules["p2c"] = p2c
spec.loader.exec_module(p2c)

from mecab import MeCab
texts, labels = p2c.load_valid()
tagger = MeCab()
docs = [" ".join(w for w, t in tagger.pos(x) if t.split("+")[0] in p2c.CONTENT_POS) for x in texts]
keep = [i for i, d in enumerate(docs) if len(d.split()) >= 3]
texts = [texts[i] for i in keep]
labels = [labels[i] for i in keep]

E = np.load(OUT2C / "gen_embeddings.npy")
assert E.shape[0] == len(texts), (E.shape, len(texts))

# ---------- baseline (seed 100) doc->topic from indicator ----------
z = np.load(OUT2C / "indicator_matrices.npz", allow_pickle=True)
ind = z["indicator"]  # (15, 10, N)
seeds = list(z["seeds"])
si = seeds.index(100)
A = ind[:, si, :]  # (15, N) binary
assert A.sum(0).max() <= 1, "doc assigned to >1 topic?"
doc_topic = np.where(A.sum(0) == 0, -1, A.argmax(0))
print(f"baseline(seed100) 복원: outlier {(doc_topic == -1).mean():.1%} "
      f"(요약파일 14.8%와 일치해야 함)")

# ---------- fresh centroids ----------
K = ind.shape[0]
C = np.zeros((E.shape[1], K), dtype=np.float32)
for t in range(K):
    m = E[doc_topic == t].mean(0)
    C[:, t] = m / np.linalg.norm(m)

def project(X):
    W = np.zeros((len(X), K)); R = np.zeros(len(X))
    for i in range(len(X)):
        w, r = nnls(C, X[i], maxiter=2000)
        W[i] = w; R[i] = r
    d = np.maximum(W.sum(1) + R, 1e-12)
    return W / d[:, None], R / d

W_gen, R_gen = project(E)

# ---------- human embeddings + projection ----------
h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
meta = pd.read_csv(ROOT / "data/atlas/metadata.csv", usecols=["work_id", "원문"])
h["work_id"] = h["work_id"].astype(str); meta["work_id"] = meta["work_id"].astype(str)
d = h.merge(meta, on="work_id")
hemb_path = OUT2C / "human_embeddings.npy"
if hemb_path.exists():
    HE = np.load(hemb_path)
else:
    from sentence_transformers import SentenceTransformer
    HE = SentenceTransformer(p2c.EMBED_MODEL).encode(
        d["원문"].astype(str).tolist(), normalize_embeddings=True, batch_size=64
    ).astype(np.float32)
    np.save(hemb_path, HE)
W_hum, R_hum = project(HE)

# ---------- system-level stats (phase2f와 동일: 평균 지분 벡터를 토픽 안에서 재정규화) ----------
rows = []
def stats(name, W, R):
    mean_share = W.mean(0)
    p = mean_share / mean_share.sum()
    o = np.argsort(p)[::-1]
    rows.append({"group": name,
                 "freshsoft_top1_topic": int(o[0]),
                 "freshsoft_top1_pct": round(p[o[0]] * 100, 2),
                 "freshsoft_top3_pct": round(p[o[:3]].sum() * 100, 2),
                 "freshsoft_eff_ntopics": round(1 / np.sum(p ** 2), 2),
                 "mean_residual_pct": round(R.mean() * 100, 2)})

stats("human", W_hum, R_hum)
lab = np.array(labels)
for m in ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0", "HCX-005"]:
    sel = lab == m
    stats(m, W_gen[sel], R_gen[sel])

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/phase2/fresh_soft_projection.csv", index=False)
print(df.to_string(index=False))
