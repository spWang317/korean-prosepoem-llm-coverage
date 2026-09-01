"""
robustness_checks.py — robustness computations reported in the paper (§3.2.2, §3.4, §4.6; SI S1).

  B1  retention 95% CI by cell-cluster bootstrap (resample the 12 prompt cells within each
      system, with replacement; B=2,000)                      -> results/phase2/retention_ci.csv
  B2  stability of the avoided-four set under that bootstrap, and the linear gap rule of §2.4.2 (bootstrap + per-system-max re-ranking)
                                                              -> results/phase2/avoidance_set_stability.json
  B3  effective sample size of the anchor-share weights (generation side), added to
      company_profile.csv as ess_g and n_g_10pct (works with weight >= 10%)
   B5  length distributions and a 300–500-character length-matched recomputation of
      effective neighbours and within-work entropy            -> results/phase2/length_check.json
  B6  human-baseline splits (2000–2011 / 2012–2024 / without Minumsa): avoided-four human
      shares                                                  -> results/phase2/human_split_check.json

Inputs: results/phase2/topic_records.csv (generated works, 25 shares as fractions, text_full),
data/atlas/instrument_frozen/human_doc_motif_distribution.csv (human shares in %),
data/atlas/metadata.csv (human bibliographic fields and text; text is not redistributed).
Run: python3 scripts/robustness_checks.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/phase2"
FIVE = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]
T = [f"T{i}" for i in range(25)]
AV = ["T18", "T9", "T22", "T5"]                                  # avoided four (paper §3.2.2)
UP = ["T7", "T15", "T6", "T17", "T13", "T16", "T23", "T12"]      # increased eight
B, SEED, BAND = 2000, 42, (300, 500)


def eff_neighbors(df, anchor):
    w = df[anchor].values; others = [t for t in T if t != anchor]
    c = np.array([(w * df[t].values).sum() / w.sum() for t in others]); p = c / c.sum()
    return float(1 / (p ** 2).sum())


def entropy(df):
    P = df[T].values; P = P / P.sum(1, keepdims=True); P = np.clip(P, 1e-12, 1)
    return float((-(P * np.log(P)).sum(1) / np.log(25)).mean())


def main():
    rec = pd.read_csv(R / "topic_records.csv")
    h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
    meta = pd.read_csv(ROOT / "data/atlas/metadata.csv")
    names = dict(pd.read_csv(R / "retention_ladder.csv")[["motif", "name"]].values)
    gen = rec[rec.model.isin(FIVE)].copy()
    hm = np.array([h[f"{t}_pct"].mean() for t in T])

    # B1 / B2
    rng = np.random.default_rng(SEED)
    cells = {m: {p: g[T].values * 100 for p, g in gen[gen.model == m].groupby("prompt_id")} for m in FIVE}
    boots = np.zeros((B, 25))
    for b in range(B):
        acc = []
        for m in FIVE:
            ks = list(cells[m]); pick = rng.choice(len(ks), len(ks), replace=True)
            acc.append(np.vstack([cells[m][ks[i]] for i in pick]))
        boots[b] = np.vstack(acc).mean(0) / hm
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0); pt = gen[T].values.mean(0) * 100 / hm
    ci = pd.DataFrame({"motif": T, "name": [names[t] for t in T], "retention": pt.round(4),
                       "ci_lo": lo.round(4), "ci_hi": hi.round(4)}).sort_values("retention", ascending=False)
    ci.to_csv(R / "retention_ci.csv", index=False)
    order = np.argsort(pt)
    def separates(v): return bool(v[4] - v[3] > np.diff(v[:4]).max())
    members = sum(set(np.argsort(boots[b])[:4]) == set(order[:4]) for b in range(B))
    holds = sum(separates(np.sort(boots[b])) for b in range(B))
    # how stable the bottom-k set is for every k
    kcurve = {k: float(np.mean([set(np.argsort(boots[b])[:k]) == set(order[:k]) for b in range(B)])) for k in range(2, 9)}
    p_next_above = {k: float(np.mean(boots[:, order[k]] > boots[:, order[k - 1]])) for k in range(2, 9)}
    # second qualification of §2.4.2: re-rank by each motif's highest per-system retention
    lad = pd.read_csv(R / "retention_ladder.csv").set_index("motif")
    smax = np.array([lad.loc[t, "retention_system_max"] for t in T]); so = np.argsort(smax)
    smax_rule = {"system_max_bottom4": [names[T[i]] for i in so[:4]], "system_max_gap_4_5": float(smax[so[4]] - smax[so[3]]),
                 "system_max_internal_max_gap": float(np.diff(smax[so[:4]]).max()), "system_max_gap_rule_holds": separates(smax[so])}
    json.dump({"linear_separates": separates(pt[order]), **smax_rule,
               "bottom4_membership_stable": members / B, "linear_gap_rule_holds": holds / B, "B": B,
               "bottom_k_membership_stable": kcurve, "p_rank_k_plus_1_above_rank_k": p_next_above,
               "order_low_to_high": [names[T[i]] for i in order]},
              open(R / "avoidance_set_stability.json", "w"), ensure_ascii=False, indent=1)

    # B3
    cp = pd.read_csv(R / "company_profile.csv")
    cp["ess_g"] = [round(float(gen[a].sum() ** 2 / (gen[a] ** 2).sum()), 1) for a in cp["anchor"]]
    cp["n_g_10pct"] = [int((gen[a] >= 0.10).sum()) for a in cp["anchor"]]
    cp.to_csv(R / "company_profile.csv", index=False)


    # B5
    meta["work_id"] = meta.work_id.astype(str); h["work_id"] = h.work_id.astype(str)
    d = h.merge(meta, on="work_id"); d["len"] = d["원문"].astype(str).str.len()
    gen["len"] = gen.text_full.str.len()
    hh = d.rename(columns={f"{t}_pct": t for t in T}); hh[T] = hh[T] / 100
    hb = hh[(hh.len >= BAND[0]) & (hh.len <= BAND[1])]; gb = gen[(gen.len >= BAND[0]) & (gen.len <= BAND[1])]
    json.dump({"band": BAND, "n_human": int(len(hb)), "n_gen": int(len(gb)),
               "eff_nb_rain": [eff_neighbors(hb, "T7"), eff_neighbors(gb, "T7")],
               "eff_nb_street": [eff_neighbors(hb, "T15"), eff_neighbors(gb, "T15")],
               "entropy": [entropy(hb), entropy(gb)],
               "length_pct": {"human": np.percentile(d["len"], [5, 50, 95]).tolist(),
                              **{m: np.percentile(gen[gen.model == m]["len"], [5, 50, 95]).tolist() for m in FIVE}}},
              open(R / "length_check.json", "w"), indent=1)

    # B6
    d["yr"] = pd.to_numeric(d["출판년도"], errors="coerce")
    splits = {"2000–2011": d[d.yr <= 2011], "2012–2024": d[d.yr >= 2012], "excluding Minumsa": d[~d["출판사"].str.contains("민음")]}
    out = {}
    for k, s in splits.items():
        out[k] = {"n": int(len(s)), "avoid4_human_mean_pct": {t: round(s[f"{t}_pct"].mean(), 2) for t in AV}}
    json.dump(out, open(R / "human_split_check.json", "w"), ensure_ascii=False, indent=1)
    print("retention CI (bottom 5):\n", ci.tail(5).to_string(index=False))
    print(json.load(open(R / "avoidance_set_stability.json")))
    print(cp[["anchor", "ess_g", "n_g_10pct"]].sort_values("ess_g").head(4).to_string(index=False))


if __name__ == "__main__":
    main()
