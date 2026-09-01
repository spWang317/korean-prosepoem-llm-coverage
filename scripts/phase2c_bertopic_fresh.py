"""
phase2c_bertopic_fresh.py — the generation axis (paper §2.2.2, §3.5): BERTopic refit on the
generated corpus with the companion study's reliability audit, replicated from its scripts.
Used to check whether a coherent topic absent from the human axis exists in the generated
corpus, and which generated-corpus motifs are reliability-robust (SI table S3-1).

Parameters identical to the companion study:
  BASE_UMAP     n_neighbors=15, n_components=5, min_dist=0.0, cosine, random_state=100
  BASE_HDBSCAN  min_cluster_size=30, min_samples=10, euclidean, eom
  BASE_VEC      token_pattern=r"\\S+", min_df=2, max_df=0.95
  R1  parameter grid, 24 OAT variations                     persistence >= 0.7
  R2  bootstrap subsample 80%, 10 replicates                persistence >= 0.7
  R3  random seeds {0,7,42,100,123,256,511,777,1024,2048}   persistence >= 0.7
  alpha  Cronbach's alpha on R3 doc-membership (10 x N)      alpha >= 0.7
  topic matching = Jaccard >= 0.5 on top-10 keywords ; robust = pass all 4 axes.
Tokenizer Mecab-ko and content POS {NNG,NP,VA,VV} identical to the companion study;
corpus = generated valid outputs (N=1,625) at the companion study's baseline parameters.

Run: python3 scripts/phase2c_bertopic_fresh.py
"""
from __future__ import annotations
import json, re, sys, warnings
from copy import deepcopy
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data/generated/arm_ab"
OUT = ROOT / "results/phase2c"; OUT.mkdir(parents=True, exist_ok=True)
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
PER_CELL = 25
KO_RATIO_MIN, REP_DEGEN = 0.5, 10
CONTENT_POS = {"NNG", "NP", "VA", "VV"}
JACCARD_MATCH, TAU, ALPHA_TAU = 0.5, 0.7, 0.7

BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=100)
BASE_HDBSCAN = dict(min_cluster_size=30, min_samples=10, metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)
SEEDS = [0, 7, 42, 100, 123, 256, 511, 777, 1024, 2048]
BOOT_N, BOOT_FRAC = 10, 0.80
R1_PARAMS = {"umap_n_neighbors": [10, 15, 20, 30, 50], "umap_n_components": [3, 5, 8],
             "umap_min_dist": [0.0, 0.1, 0.25], "hdbscan_min_cluster_size": [20, 30, 50, 80],
             "hdbscan_min_samples": [5, 10, 15], "cv_min_df": [2, 3, 5], "cv_max_df": [0.9, 0.95, 0.99]}

MODELS = {"openai_gpt-5.6": "GPT-5.6", "anthropic_opus-4-8": "Opus-4.8",
          "google_gemini-2.5-pro": "Gemini-2.5", "upstage_solar-pro3": "Solar-Pro3",
          "naver_hcx-005": "HCX-005", "lg_exaone": "EXAONE-4.0"}
RNG = np.random.default_rng(100)


def is_valid(t):
    t = (t or "").strip()
    if not t or "\n" in t: return False
    tot = len(re.findall(r"\S", t)); ko = len(re.findall(r"[가-힣]", t))
    if (ko / tot if tot else 0) < KO_RATIO_MIN: return False
    w = t.split()
    if len(w) >= 4:
        g = Counter(" ".join(w[i:i+4]) for i in range(len(w)-3))
        if (max(g.values()) if g else 0) >= REP_DEGEN: return False
    return True


def load_valid():
    texts, labels = [], []
    for stem, label in MODELS.items():
        rows = [json.loads(l) for l in open(GEN / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
        by = {}
        for r in rows:
            by.setdefault(r["prompt_id"], []).append(r)
        for pid, rs in by.items():
            rs.sort(key=lambda r: r.get("attempt", r.get("seed_idx", 0)))
            for t in [(r.get("response") or "").strip() for r in rs if is_valid(r.get("response") or "")][:PER_CELL]:
                texts.append(t); labels.append(label)
    return texts, labels


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb)) if (sa or sb) else 0.0


def run_bertopic(emb, docs, up, hp, vp):
    """INSTRUMENT-exact. Returns (topics_arr, topic_ids, keyword_lists) or None on failure."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    tm = BERTopic(embedding_model=None, umap_model=UMAP(**up),
                  hdbscan_model=HDBSCAN(**hp, prediction_data=True),
                  vectorizer_model=CountVectorizer(**vp), language="multilingual",
                  top_n_words=10, calculate_probabilities=False, verbose=False)
    try:
        topics, _ = tm.fit_transform(docs, emb)
    except Exception:
        return None
    topics = np.asarray(topics)
    info = tm.get_topic_info(); info = info[info["Topic"] != -1]
    tids = list(info["Topic"])
    kws = [[w for w, _ in (tm.get_topic(t) or [])] for t in tids]
    return topics, tids, kws


def present_per_baseline(perturbed_kws, base_kws, minj=JACCARD_MATCH):
    """Per baseline topic: does a Jaccard>=0.5 match exist among perturbed?"""
    if not perturbed_kws:
        return np.zeros(len(base_kws), dtype=bool)
    return np.array([max((jaccard(ba, pe) for pe in perturbed_kws), default=0.0) >= minj
                     for ba in base_kws])


def best_match_baseline(perturbed_kws, base_kws, minj=JACCARD_MATCH):
    """Per perturbed topic: index of best baseline (>=0.5) or -1. (INSTRUMENT 05)"""
    out = []
    for pe in perturbed_kws:
        js = [jaccard(pe, ba) for ba in base_kws]
        bi = int(np.argmax(js)) if js else -1
        out.append(bi if (js and js[bi] >= minj) else -1)
    return out


def cronbach_alpha(X):
    """INSTRUMENT-exact. X: (K raters=seeds, N docs) binary."""
    X = np.asarray(X, dtype=float); K, N = X.shape
    if K < 2 or N < 2: return float("nan")
    item_var = X.var(axis=1, ddof=1).sum()
    st_var = X.sum(axis=0).var(ddof=1)
    if st_var <= 0: return float("nan")
    return float((K / (K - 1)) * (1 - item_var / st_var))


def main():
    from sentence_transformers import SentenceTransformer
    from mecab import MeCab
    texts, labels = load_valid()
    tagger = MeCab()
    docs = [" ".join(w for w, t in tagger.pos(x) if t.split("+")[0] in CONTENT_POS) for x in texts]
    keep = [i for i, d in enumerate(docs) if len(d.split()) >= 3]
    texts = [texts[i] for i in keep]; docs = [docs[i] for i in keep]; labels = [labels[i] for i in keep]
    N = len(texts)
    print(f"N={N}  models={Counter(labels)}")

    emb_path = OUT / "gen_embeddings.npy"
    if emb_path.exists():
        embs = np.load(emb_path)
    else:
        embs = SentenceTransformer(EMBED_MODEL).encode(texts, normalize_embeddings=True,
                                                       batch_size=64).astype(np.float32)
        np.save(emb_path, embs)

    # ---------- baseline (seed 100) ----------
    base = run_bertopic(embs, docs, BASE_UMAP, BASE_HDBSCAN, BASE_VEC)
    base_topics, base_tids, base_kws = base
    K = len(base_kws)
    print(f"baseline: {K} topics, outlier {(base_topics==-1).mean():.0%}")

    # ---------- R1 param grid (24 OAT) ----------
    r1_hits = np.zeros(K); r1_n = 0
    for pname, vals in R1_PARAMS.items():
        for v in vals:
            up, hp, vp = deepcopy(BASE_UMAP), deepcopy(BASE_HDBSCAN), deepcopy(BASE_VEC)
            if pname.startswith("umap_"): up[pname[5:]] = v
            elif pname.startswith("hdbscan_"): hp[pname[8:]] = v
            elif pname.startswith("cv_"): vp[pname[3:]] = v
            res = run_bertopic(embs, docs, up, hp, vp)
            r1_hits += present_per_baseline(res[2] if res else [], base_kws)
            r1_n += 1
    r1 = r1_hits / r1_n
    print(f"R1 done ({r1_n} runs)")

    # ---------- R2 bootstrap 80% x 10 ----------
    r2_hits = np.zeros(K); m = int(BOOT_FRAC * N)
    for b in range(BOOT_N):
        idx = RNG.choice(N, size=m, replace=False)
        res = run_bertopic(embs[idx], [docs[j] for j in idx], BASE_UMAP, BASE_HDBSCAN, BASE_VEC)
        r2_hits += present_per_baseline(res[2] if res else [], base_kws)
    r2 = r2_hits / BOOT_N
    print("R2 done")

    # ---------- R3 seeds x 10 + indicator for alpha ----------
    r3_hits = np.zeros(K); indicator = np.zeros((K, len(SEEDS), N), dtype=np.int8)
    for ri, sd in enumerate(SEEDS):
        up = deepcopy(BASE_UMAP); up["random_state"] = sd
        res = run_bertopic(embs, docs, up, BASE_HDBSCAN, BASE_VEC)
        if res is None: continue
        topics_arr, tids, kws = res
        r3_hits += present_per_baseline(kws, base_kws)
        for j, base_id in enumerate(best_match_baseline(kws, base_kws)):
            if base_id >= 0:
                indicator[base_id, ri, :] = (topics_arr == tids[j]).astype(np.int8)
    r3 = r3_hits / len(SEEDS)
    alpha = np.array([cronbach_alpha(indicator[t]) for t in range(K)])
    np.savez_compressed(OUT / "indicator_matrices.npz", indicator=indicator, seeds=np.array(SEEDS))
    print("R3 + alpha done")

    # ---------- assemble ----------
    rows = []
    for t in range(K):
        members = [labels[j] for j in range(N) if base_topics[j] == base_tids[t]]
        p = {"R1": r1[t] >= TAU, "R2": r2[t] >= TAU, "R3": r3[t] >= TAU,
             "alpha": (not np.isnan(alpha[t])) and alpha[t] >= ALPHA_TAU}
        rows.append({"topic": base_tids[t], "size": int((base_topics == base_tids[t]).sum()),
                     "keywords": " ".join(base_kws[t][:8]),
                     "R1_persist": round(float(r1[t]), 2), "R2_persist": round(float(r2[t]), 2),
                     "R3_persist": round(float(r3[t]), 2),
                     "cronbach_alpha": None if np.isnan(alpha[t]) else round(float(alpha[t]), 3),
                     "n_axes_passed": int(sum(p.values())),
                     "pass": "".join(k for k, v in [("1", p["R1"]), ("2", p["R2"]),
                                                    ("3", p["R3"]), ("α", p["alpha"])] if v) or "-",
                     "dominant_models": ", ".join(f"{mm}:{c}" for mm, c in Counter(members).most_common(3))})
    df = pd.DataFrame(rows).sort_values(["n_axes_passed", "size"], ascending=False)
    df.to_csv(OUT / "bertopic_fresh_topics.csv", index=False)
    json.dump({"N": N, "n_topics": K, "outlier_pct": round(float((base_topics == -1).mean())*100, 1),
               "jaccard_match": JACCARD_MATCH, "tau": TAU, "alpha_tau": ALPHA_TAU,
               "tokenizer": "python-mecab-ko (identical to INSTRUMENT; content POS identical)",
               "params": "INSTRUMENT baseline (mcs=30 etc.), absolute (not scaled to N)",
               "axes_pass_distribution": Counter(df["n_axes_passed"]).most_common(),
               "robust_4of4": int((df["n_axes_passed"] == 4).sum())},
              open(OUT / "bertopic_fresh_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    pd.set_option("display.width", 240, "display.max_colwidth", 48, "display.max_rows", 40)
    print(f"\n=== BERTopic refit + reliability axes (R1/R2/R3 persist>=0.7, alpha>=0.7; robust=4/4) ===")
    print(df.to_string(index=False))
    print(f"\naxes passed: {Counter(df['n_axes_passed']).most_common()}  | robust(4/4): {(df['n_axes_passed']==4).sum()}")
    print(f"Saved -> {OUT}/bertopic_fresh_topics.csv , bertopic_fresh_summary.json , indicator_matrices.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
