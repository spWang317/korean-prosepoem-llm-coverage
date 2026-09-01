"""
phase2_topic_full.py — full 6-model topic-distribution comparison (human vs generated).

Requires torch and sentence-transformers:
    python3 scripts/phase2_topic_full.py

Design:
  Two levels.
  Level 1 (form compliance): per model, valid-rate = valid / completed calls, and per-prompt
     success. Complete data for all 6 systems.
  Level 2 (topic among valid): take VALID outputs only (line-break-free Korean prose,
     non-degenerate; common.is_valid_poem), balanced to <=25/cell (prompt-robust). Whole-work
     embedding (ko-sroberta) -> NNLS onto the frozen 25 INSTRUMENT motif centroids -> motif
     distribution + unexplained residual + entropy. Compare to human (INSTRUMENT, N=4,642).
  HCX enters Level 1 fully; in Level 2 it has valid outputs on only its complying prompts
     and is excluded from the human-comparison topic measures (paper §3).

Rejection-sampling note: accept/reject is on FORM, so P(topic | valid) is unbiased in topic.
The form x topic caveat is checked (topic of valid vs discarded; topic vs attempt-count).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent

# is_valid_poem inlined from common.py (common imports dotenv/API libs absent in this
# torch interpreter). Kept byte-identical to common.is_valid_poem: reject empty / any
# line break / ko_ratio < 0.5 / a word 4-gram repeated >= 10 times.
from collections import Counter as _Counter  # noqa: E402
KO_RATIO_MIN, REP_DEGEN_THRESHOLD = 0.5, 10


def is_valid_poem(text):
    t = (text or "").strip()
    if len(t) == 0:
        return False, "empty"
    if "\n" in t:
        return False, "linebreak"
    tot = len(re.findall(r"\S", t))
    ko = len(re.findall(r"[가-힣]", t))
    if (ko / tot if tot else 0.0) < KO_RATIO_MIN:
        return False, "ko_ratio"
    w = t.split()
    rep_max = 0
    if len(w) >= 4:
        g = _Counter(" ".join(w[i:i + 4]) for i in range(len(w) - 3))
        rep_max = max(g.values()) if g else 0
    if rep_max >= REP_DEGEN_THRESHOLD:
        return False, "degenerate"
    return True, "ok"

ATLAS = ROOT / "data/atlas"
INSTRUMENT = ATLAS / "instrument_frozen"
GEN = ROOT / "data/generated/arm_ab"
OUT = ROOT / "results/phase2"; OUT.mkdir(parents=True, exist_ok=True)
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
PER_CELL = 25          # balance: <=25 valid per (model, prompt) cell
N_BOOT = 2000
RNG = np.random.default_rng(42)

# file stem -> display label
MODELS = {
    "openai_gpt-5.6": "GPT-5.6",
    "anthropic_opus-4-8": "Opus-4.8",
    "google_gemini-2.5-pro": "Gemini-2.5",
    "upstage_solar-pro3": "Solar-Pro3",
    "naver_hcx-005": "HCX-005",
    "lg_exaone": "EXAONE-4.0",
}


def nnls_dist(X, C):
    n, K = X.shape[0], C.shape[1]
    W = np.zeros((n, K)); R = np.zeros(n)
    for i in range(n):
        w, r = nnls(C, X[i], maxiter=2000); W[i] = w; R[i] = r
    denom = np.maximum(W.sum(1) + R, 1e-12)
    return W / denom[:, None], R / denom


def entropy(share):
    p = np.clip(share, 1e-12, 1.0); p = p / p.sum(1, keepdims=True)
    return (-(p * np.log(p)).sum(1)) / np.log(p.shape[1])


def order_key(r):
    return r.get("attempt", r.get("seed_idx", 0))


def load_valid_balanced(stem):
    """Return (valid_texts<=25/cell, n_valid_total, n_attempts, percell dict)."""
    rows = [json.loads(l) for l in open(GEN / f"{stem}.jsonl", encoding="utf-8") if l.strip()]
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["prompt_id"], []).append(r)
    texts, ids, pids, percell = [], [], [], {}
    n_valid_total = 0
    for pid, rs in sorted(by_cell.items()):
        rs.sort(key=order_key)
        if stem == "lg_exaone":
            # EXAONE was generated as a fixed batch per cell (gen_exaone_full.py); apply the paper's
            # stopping rule afterwards: keep each cell up to and including its 25th valid output.
            n_ok, cut = 0, len(rs)
            for k, r in enumerate(rs):
                if is_valid_poem((r.get("response") or "").strip())[0]:
                    n_ok += 1
                    if n_ok == PER_CELL:
                        cut = k + 1; break
            rs = rs[:cut]
        valid = [r for r in rs if is_valid_poem((r.get("response") or "").strip())[0]]
        n_valid_total += len(valid)
        completed = sum(1 for r in rs if (r.get("response") or "").strip())
        percell[pid] = {"valid": len(valid), "attempts": len(rs), "completed": completed}
        for r in valid[:PER_CELL]:
            texts.append((r.get("response") or "").strip()); ids.append(r["id"]); pids.append(pid)
    n_rows = sum(c["attempts"] for c in percell.values())
    return texts, n_valid_total, n_rows, percell, ids, pids


def boot_mean_ci(x, n=N_BOOT):
    x = np.asarray(x)
    bs = np.array([x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    from sentence_transformers import SentenceTransformer
    npz = np.load(INSTRUMENT / "motif_centroids.npz", allow_pickle=True)
    centroids = npz["centroids"].astype(np.float32)
    topwords = [str(w) for w in npz["topwords"]]
    C = centroids.T; K = centroids.shape[0]
    emb = SentenceTransformer(EMBED_MODEL)
    print(f"centroids {centroids.shape}  embedder {EMBED_MODEL}")

    # ---- VERIFY NNLS reproduces INSTRUMENT on a human sample ----
    human_csv = pd.read_csv(INSTRUMENT / "human_doc_motif_distribution.csv")
    Tcols = [f"T{i}_pct" for i in range(K)]
    meta = pd.read_csv(ATLAS / "metadata.csv", usecols=["work_id", "원문"])
    meta["work_id"] = meta["work_id"].astype(str)
    samp = human_csv.sample(min(50, len(human_csv)), random_state=42).copy()
    samp["work_id"] = samp["work_id"].astype(str)
    samp = samp.merge(meta, on="work_id", how="inner").dropna(subset=["원문"])
    Xs = emb.encode(samp["원문"].tolist(), normalize_embeddings=True, batch_size=16).astype(np.float32)
    sh, _ = nnls_dist(Xs, C)
    mad = float(np.abs(sh - samp[Tcols].values / 100.0).mean())
    print(f"[VERIFY] NNLS vs INSTRUMENT CSV on {len(samp)} works: MAD={mad:.4f} (small=reproduces)")

    # ---- human baseline ----
    human_share = human_csv[Tcols].values / 100.0
    human_ent = entropy(human_share)
    human_motif_mean = human_share.mean(0)
    hent_m, hent_lo, hent_hi = boot_mean_ci(human_ent)

    # ---- per model: valid -> embed -> NNLS ----
    records, per_model, motif_means = [], [], {}
    print(f"\n{'model':12s} {'validN':>7} {'rate':>6} {'topicN':>7} {'entropy(95%CI)':>22} {'Δvshuman':>9} {'unexpl':>7}")
    for stem, label in MODELS.items():
        texts, n_valid, n_att, percell, ids, pids = load_valid_balanced(stem)
        X = emb.encode(texts, normalize_embeddings=True, batch_size=32).astype(np.float32)
        share, unexpl = nnls_dist(X, C)
        ent = entropy(share)
        em, elo, ehi = boot_mean_ci(ent)
        motif_means[label] = share.mean(0)
        rate = float(np.mean([c["valid"] / c["completed"] for c in percell.values() if c["completed"]]))  # cell-mean of valid/completed (transport failures excluded), = paper Table 1
        per_model.append({
            "model": label, "stem": stem, "valid_total": n_valid, "attempts": n_att,
            "valid_rate": rate, "topic_n": len(texts),
            "entropy_mean": em, "entropy_ci": [elo, ehi],
            "entropy_delta_vs_human": em - hent_m,
            "unexplained_mean": float(unexpl.mean()),
            "motif_share_mean": motif_means[label].tolist(),
            "per_cell": percell,
        })
        for i in range(len(texts)):
            records.append({"model": label, "work_id": ids[i], "prompt_id": pids[i],
                            "text": texts[i][:60], "text_full": texts[i],
                            "top1_motif": int(share[i].argmax()),
                            "entropy": round(float(ent[i]), 3),
                            "unexplained": round(float(unexpl[i]), 3),
                            **{f"T{j}": round(float(share[i, j]), 4) for j in range(K)}})
        print(f"{label:12s} {n_valid:7d} {rate:6.0%} {len(texts):7d}  "
              f"{em:.3f}[{elo:.3f},{ehi:.3f}]  {em-hent_m:+.3f}  {unexpl.mean():7.2f}")
    print(f"{'HUMAN':12s} {len(human_csv):7d} {'—':>6} {len(human_csv):7d}  "
          f"{hent_m:.3f}[{hent_lo:.3f},{hent_hi:.3f}]  {'—':>9}")

    # ---- motif over/under map: per model + pooled (gen - human), with bootstrap CI on pooled ----
    pooled = np.mean([per_model[i]["motif_share_mean"] for i in range(len(per_model))
                      if per_model[i]["model"] != "HCX-005"], 0)  # HCX excluded from pooled topic
    diff = pooled - human_motif_mean
    # bootstrap CI of the pooled per-motif diff over human works (human sampling variability)
    hb = np.array([human_share[RNG.integers(0, len(human_share), len(human_share))].mean(0)
                   for _ in range(N_BOOT)])
    hlo, hhi = np.percentile(hb, 2.5, 0), np.percentile(hb, 97.5, 0)
    motif_rows = []
    for j in range(K):
        sig = "**" if (pooled[j] < hlo[j] or pooled[j] > hhi[j]) else ""
        row = {"motif": f"T{j}", "topwords": topwords[j][:32],
               "human": round(float(human_motif_mean[j]), 4),
               "gen_pooled": round(float(pooled[j]), 4),
               "diff": round(float(diff[j]), 4), "sig_vs_human_CI": sig}
        for pm in per_model:
            row[pm["model"]] = round(pm["motif_share_mean"][j], 4)
        motif_rows.append(row)
    motif_df = pd.DataFrame(motif_rows)

    # ---- save ----
    pd.DataFrame(records).to_csv(OUT / "topic_records.csv", index=False)
    motif_df.to_csv(OUT / "motif_map.csv", index=False)
    bal = pd.DataFrame({"motif": [f"T{j}" for j in range(K)], "topwords": [topwords[j] for j in range(K)],
                        "human_%": (human_motif_mean * 100).round(2)})
    for pm in per_model:
        bal[f"{pm['model']}_%"] = (np.array(pm["motif_share_mean"]) * 100).round(2)
    bal.to_csv(OUT / "motif_full_distribution_balanced.csv", index=False)  # balanced (<=25/cell) mean shares; source of Table 3, Fig. 1, SI S1
    (OUT / "summary.json").write_text(json.dumps({
        "embedder": EMBED_MODEL, "per_cell_cap": PER_CELL, "verify_mad_vs_instrument": mad,
        "human_n": len(human_csv), "human_entropy_mean": hent_m,
        "human_entropy_ci": [hent_lo, hent_hi],
        "per_model": per_model,
        "pooled_note": "pooled generated = 5 form-capable models (HCX excluded from Level-2 pooled topic)",
    }, ensure_ascii=False, indent=2))

    # ---- console: over/under motifs ----
    print("\n=== 모티프 과잉 생산 top6 (생성풀 − 인간, **=인간 CI 벗어남) ===")
    for j in np.argsort(diff)[::-1][:6]:
        s = "**" if (pooled[j] < hlo[j] or pooled[j] > hhi[j]) else "  "
        print(f"  {s} {diff[j]:+.3f}  T{j} [{topwords[j][:26]}]  human {human_motif_mean[j]:.3f} → gen {pooled[j]:.3f}")
    print("=== 모티프 과소(회피) top6 ===")
    for j in np.argsort(diff)[:6]:
        s = "**" if (pooled[j] < hlo[j] or pooled[j] > hhi[j]) else "  "
        print(f"  {s} {diff[j]:+.3f}  T{j} [{topwords[j][:26]}]  human {human_motif_mean[j]:.3f} → gen {pooled[j]:.3f}")
    print(f"\nSaved -> {OUT}/topic_records.csv , motif_map.csv , summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
