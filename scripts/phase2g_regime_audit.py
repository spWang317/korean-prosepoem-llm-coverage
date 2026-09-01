"""phase2g_regime_audit.py — protocol-window recomputation (paper §2.3: each cell stops at the
25th valid output; EXAONE rows beyond that window are excluded).
(1) checks the §3.2 pooled numbers against the balanced CSV,
(2) form×topic filter check on protocol samples -> results/phase2/form_x_topic_protocol.json,
Run with torch python3: python3 scripts/phase2g_regime_audit.py"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data/generated/arm_ab"
STEM = {"GPT-5.6": "openai_gpt-5.6", "Opus-4.8": "anthropic_opus-4-8",
        "Gemini-2.5": "google_gemini-2.5-pro", "Solar-Pro3": "upstage_solar-pro3",
        "EXAONE-4.0": "lg_exaone", "HCX-005": "naver_hcx-005"}
SYS5 = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]


def protocol_rows(stem):
    """Rows within the until-25-valid window per cell (EXAONE truncated; others = as logged)."""
    raw = defaultdict(list)
    for l in open(GEN / f"{stem}.jsonl", encoding="utf-8"):
        r = json.loads(l); raw[r["prompt_id"]].append(r)
    out = []
    for pid, rs in raw.items():
        rs = sorted(rs, key=lambda r: r.get("attempt", r.get("seed_idx", 0)))
        v = 0
        for r in rs:
            out.append(r); v += bool(r.get("is_valid"))
            if v == 25: break
    return out


def main():
    # ---------- (1) §3.2 pooled numbers vs balanced CSV ----------
    m = pd.read_csv(ROOT / "results/phase2/motif_full_distribution_balanced.csv")
    pooled = m[[f"{s}_%" for s in SYS5]].values.mean(1)
    human = m["human_%"].values
    checks = {"T3": (2.6, 0.4), "T9": (2.6, 0.2), "T5": (2.0, 0.2), "T22": (1.9, 0.2),
              "T18": (2.0, 0.1), "T7": (2.1, 11.5), "T15": (2.0, 7.6), "T6": (2.4, 7.0), "T17": (1.8, 4.9)}
    print("== (1) check of the §3.2 figures (paper values vs balanced recomputation) ==")
    for t, (h_paper, g_paper) in checks.items():
        j = int(t[1:])
        ok = (abs(human[j] - h_paper) < 0.05) and (abs(pooled[j] - g_paper) < 0.05)
        print(f"  {t}: human {human[j]:.1f} (paper {h_paper}) | pooled {pooled[j]:.1f} (paper {g_paper}) {'✓' if ok else '✗✗'}")

    # ---------- (2) form×topic on protocol samples ----------
    from sentence_transformers import SentenceTransformer
    from scipy.optimize import nnls
    npz = np.load(ROOT / "data/atlas/instrument_frozen/motif_centroids.npz", allow_pickle=True)
    C = npz["centroids"].astype(np.float32).T
    emb = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def topic_mean(texts):
        if len(texts) < 3: return None
        X = emb.encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)
        W = np.zeros((len(texts), C.shape[1])); R = np.zeros(len(texts))
        for i in range(len(texts)):
            w, r = nnls(C, X[i], maxiter=2000); W[i] = w; R[i] = r
        d = np.maximum(W.sum(1) + R, 1e-12)
        return (W / d[:, None]).mean(0)

    fx = {}
    print("\n== (2) form x topic (protocol sample) ==")
    for name, stem in STEM.items():
        rows = protocol_rows(stem)
        vt = [(r["response"] or "").strip() for r in rows if r.get("is_valid")]
        dt = [(r["response"] or "").strip() for r in rows if not r.get("is_valid") and (r.get("response") or "").strip()]
        entry = {"n_valid": len(vt), "n_discard_nonempty": len(dt),
                 "n_discard_empty": sum(1 for r in rows if not r.get("is_valid") and not (r.get("response") or "").strip())}
        if len(dt) >= 10:
            vm, dm = topic_mean(vt), topic_mean(dt)
            entry["mad"] = round(float(np.abs(vm - dm).mean()), 4)
            entry["corr"] = round(float(np.corrcoef(vm, dm)[0, 1]), 2)
        else:
            entry["note"] = "fewer than 10 rejected texts (empty responses only, or no rejection) - subject comparison not applied"
        fx[name] = entry
        print(f"  {name:11s} valid {entry['n_valid']:4d} | 기각(텍스트) {entry['n_discard_nonempty']:4d} "
              f"| 기각(빈) {entry['n_discard_empty']:3d} | {entry.get('mad','—')} / {entry.get('corr','—')}")
    json.dump({"form_x_topic_protocol": fx}, open(ROOT / "results/phase2/form_x_topic_protocol.json", "w"),
              ensure_ascii=False, indent=2)


    return 0


if __name__ == "__main__":
    main()
