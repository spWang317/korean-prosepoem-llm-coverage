"""close_reading_provenance.py — the selection record for the quoted works (paper §2.4.7).

Generated typical works: ranked by cosine proximity to the mean embedding weighted by
the anchor motif share. The naming set: the motifs with retention > 1 in all five
systems, crossed with the per-system typical work. Human top-share works: the per-motif
share ranking and the reason for each selection.

Run: python3 scripts/close_reading_provenance.py -> results/phase2/close_reading_provenance.json
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rec = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
E = np.load(ROOT / "results/phase2c/gen_embeddings.npy")
spec = importlib.util.spec_from_file_location("p2c", ROOT / "scripts/phase2c_bertopic_fresh.py")
p2c = importlib.util.module_from_spec(spec); sys.modules["p2c"] = p2c; spec.loader.exec_module(p2c)
from mecab import MeCab
texts, labels = p2c.load_valid()
tagger = MeCab()
docs = [" ".join(w for w, t in tagger.pos(x) if t.split("+")[0] in p2c.CONTENT_POS) for x in texts]
keep = [i for i, d in enumerate(docs) if len(d.split()) >= 3]
texts = [texts[i] for i in keep]; labels = [labels[i] for i in keep]

S5 = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]


def topk(model_filter, tcol, k=3):
    """rank list of the k works nearest the share-weighted centroid (system + opening); paper §4.1."""
    w = np.zeros(len(texts))
    pool = rec if model_filter is None else rec[rec["model"].isin(model_filter)]
    pref = {r["text_full"]: r[tcol] for _, r in pool.iterrows()}
    for i, (t, l) in enumerate(zip(texts, labels)):
        if (model_filter is None or l in model_filter) and t in pref:
            w[i] = pref[t]
    c = (E * w[:, None]).sum(0) / w.sum(); c /= np.linalg.norm(c)
    sims = E @ c
    idx = [i for i in range(len(texts)) if (model_filter is None or labels[i] in model_filter)]
    order = sorted(idx, key=lambda i: -sims[i])
    return [{"rank": r, "system": labels[i], "opening": texts[i][:30]} for r, i in enumerate(order[:k], 1)]


def wrank(model_filter, tcol, probe):
    w = np.zeros(len(texts))
    pool = rec if model_filter is None else rec[rec["model"].isin(model_filter)]
    pref = {r["text_full"]: r[tcol] for _, r in pool.iterrows()}
    for i, (t, l) in enumerate(zip(texts, labels)):
        if (model_filter is None or l in model_filter) and t in pref:
            w[i] = pref[t]
    c = (E * w[:, None]).sum(0) / w.sum(); c /= np.linalg.norm(c)
    sims = E @ c
    idx = [i for i in range(len(texts)) if (model_filter is None or labels[i] in model_filter)]
    order = sorted(idx, key=lambda i: -sims[i])
    for rk, i in enumerate(order, 1):
        if probe in texts[i]:
            return {"weighted_centroid_rank": rk, "of": len(order)}
    return {"weighted_centroid_rank": None, "of": len(order)}


out = {
    "겨울 전형(전체 가중) 상위 3(§4.1)": topk(S5, "T6", 3),
    "GPT 비 전형작(§4.1)": {"probe": "주머니 속 열쇠를 만지며",
                            **wrank(["GPT-5.6"], "T7", "비가 그친 골목에는 아직 돌아가지 못한 저녁")},
    "겨울 전형(전체 가중) 1위(§4.1)": {"probe": "새벽은 아직 이름을 얻지 못한 빛을 창문 가장자리에",
                                      **wrank(S5, "T6", "새벽은 아직 이름을 얻지 못한 빛을 창문 가장자리에"),
                                      "system": "GPT-5.6"},
    "Gemini 버스(§4.3)": {"probe": "아직 내릴 곳을 정하지 못한 채", "source": "결말 감사 표본 Gemini 비-11 (SI S8)"},
    "Gemini 여름 끝(§4.3)": {"probe": "이름 붙일 수 없는 감정의 문턱", "source": "결말 감사 표본 Gemini 비-6 (SI S8)"},
    "Opus 겨울 전형(§4.5)": {"probe": "성에는 이미 절반쯤 녹아 흘러내리고",
                              **wrank(["Opus-4.8"], "T6", "성에는 이미 절반쯤 녹아 흘러내리고")},
    "Opus 가족 전형(§4.5)": {"probe": "새벽이 오기 전 부엌에서 어머니는 쌀을 씻었다",
                              **wrank(["Opus-4.8"], "T2", "새벽이 오기 전 부엌에서 어머니는 쌀을 씻었다")},
}
json.dump(out, open(ROOT / "results/phase2/close_reading_provenance.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))


def naming_set():
    """The naming set (§2.4.7; the result is in §4.1): the motifs with retention > 1 in
    all five systems, crossed with each system's typical work (nearest to the weighted centroid)."""
    import re
    lad = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
    core = [t for t in lad.index if all(lad.loc[t, f"retention_{m}"] > 1 for m in S5)]
    WIN = re.compile(r"창|유리"); NIGHT = re.compile(r"저녁|밤|새벽|어둠|어스름|노을|늦은 오후|가로등|별")
    rows = []
    for tcol in core:
        for m in S5:
            w = np.zeros(len(texts)); pref = {r["text_full"]: r[tcol] for _, r in rec[rec["model"] == m].iterrows()}
            for i, (t, l) in enumerate(zip(texts, labels)):
                if l == m and t in pref: w[i] = pref[t]
            c = (E * w[:, None]).sum(0) / w.sum(); c /= np.linalg.norm(c); sims = E @ c
            i = max((i for i in range(len(texts)) if labels[i] == m), key=lambda i: sims[i])
            rows.append({"motif": tcol, "name": lad.loc[tcol, "name"], "system": m, "head": texts[i][:60], "text": texts[i],
                         "window": bool(WIN.search(texts[i])), "night": bool(NIGHT.search(texts[i]))})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/phase2/naming_set_typicals.csv", index=False)
    print("naming set:", core, "| window/glass", int(df.window.sum()), "/", len(df), "| evening/night/lamplight", int(df.night.sum()), "/", len(df))
    return df


def human_top_works():
    """The ranking of human top-share works. By the rule of §2.4.7 (choose among the five
    highest shares, publish the reason), this releases the ranking and the top five for the
    human works quoted in §4.2 (the avoided four, plus the family and food that only Opus
    retains) and §4.3 (rain, winter). -> results/phase2/human_top_works.csv"""
    import pandas as pd
    h = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
    m = pd.read_csv(ROOT / "data/atlas/metadata.csv")
    h["work_id"] = h.work_id.astype(str); m["work_id"] = m.work_id.astype(str); d = h.merge(m, on="work_id")
    motifs = {"T18": "상처·술", "T9": "노인·지하", "T22": "병원·질병", "T5": "학교·아이",
              "T3": "음식·동물", "T2": "가족", "T7": "비", "T6": "겨울"}
    # Selection rule (§2.4.7): among the five highest shares, the researcher takes the one
    # that can show how the material is used within the quotation limit, and publishes the reason.
    # (Korean as recorded, English translation). Written to the reason / reason_en columns.
    chosen = {("T18", 2): ("1위 「홍도야」는 171자로 인용 상한(3어구·15%) 안에서 재료를 보일 수 없어 다음 순위", "the highest-ranked work is 171 characters long, too short to show how the material is used within the quotation limit (3 phrases, 15%), so the next rank was taken"),
              ("T9", 1): ("1위", "highest share"), ("T22", 4): ("병원 재료와 비 어휘가 한 작품에 있어 §4.3의 비 맞세우기에 재사용", "this work carries both the hospital material and rain vocabulary, so it is reused for the rain comparison in §4.3"),
              ("T5", 3): ("모티프 키워드(학교·교실·선생·학생)가 표면에 있는 작품; 1·2위는 아이는 있으나 학교 장면이 아님", "the motif's keywords (school, classroom, teacher, student) are on the surface of this work; ranks 1 and 2 have a child but no school scene"),
              ("T3", 1): ("1위", "highest share"), ("T2", 1): ("1위", "highest share"), ("T7", 1): ("1위", "highest share"), ("T6", 1): ("1위", "highest share")}
    cited = {"T18": "§4.2", "T9": "§4.2", "T22": "§4.2", "T5": "§4.2", "T3": "§4.2", "T2": "§4.2", "T7": "§4.3", "T6": "§4.3"}
    rows = []
    for t, n in motifs.items():
        s = d.sort_values(f"{t}_pct", ascending=False).head(5).reset_index(drop=True)
        for i, r in s.iterrows():
            rows.append({"motif": t, "name": n, "rank": i + 1, "work_id": r["work_id"], "poet": r["작가"],
                         "title": r["작품 제목"], "year": r["출판년도"], "share_pct": round(r[f"{t}_pct"], 2),
                         "chosen": "Y" if (t, i + 1) in chosen else "", "reason": chosen.get((t, i + 1), ("", ""))[0],
                         "reason_en": chosen.get((t, i + 1), ("", ""))[1],
                         "cited": cited[t] if (t, i + 1) in chosen else ""})
    pd.DataFrame(rows).to_csv(ROOT / "results/phase2/human_top_works.csv", index=False)


if __name__ == "__main__":
    naming_set()
    human_top_works()
