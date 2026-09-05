# -*- coding: utf-8 -*-
"""Direct count of person-denoting nouns in the human and generated corpora.

The manuscript's central claim — that what the systems drop is not rare material
but material with people in it — currently rests on whether a motif's top ten
c-TF-IDF keywords contain a word denoting a person (SI Table S2-2). That is an
inference about motif labels. This script measures the thing itself: how often a
person-denoting noun occurs in the works of each corpus.

The lexicon is fixed in advance and printed in full below, so the measurement can
be re-run or contested. It has three parts: (a) the person words the instrument's
own keyword table already marks, minus personal names; (b) a closed list of
kinship terms; (c) a closed list of generic person nouns, age terms and social
roles. Counting is on morphemes tagged as nouns by Mecab-ko, the analyser the
instrument uses, so a match is a whole noun and not a substring.

Run: python3 scripts/person_word_density.py
-> results/phase2/person_word_density_by_motif.csv        (per motif)
-> results/phase2/person_word_density.json     (corpus level)
"""
from pathlib import Path
import json
import re

import mecab
import numpy as np
import pandas as pd
from common import ATLAS_META

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase2"

# ---------------------------------------------------------------- the lexicon
# (a) marked as person words in SI Table S2-2 (personal names excluded)
FROM_S2_2 = """아버지 어머니 엄마 아들 고모 형 동생 누나 애인 아이 선생 소녀 학생
그녀 여인 사내 노인 인간 남자 시인 동무 필자 가수 할머니 할아버지 손주 할망구
환자 오빠 간호사 기사 승객"""
# (b) kinship
KINSHIP = """아빠 부모 딸 언니 남매 형제 자매 조부 조모 손자 손녀 삼촌 이모
외삼촌 사촌 며느리 사위 아내 남편 부인 처남 처제 시어머니 시아버지 장모 장인
조카 자식 자녀 아우 누이"""
# (c) generic person nouns, age terms, social roles
GENERIC = """사람 인간 남 타인 이웃 친구 벗 손님 주인 아저씨 아줌마 아주머니
영감 할멈 노파 어른 청년 소년 아기 아가 어린이 늙은이 젊은이 여자 여성 남성
학생 교사 의사 간호사 환자 승객 기사 운전사 경찰 군인 농부 어부 상인 점원
직원 사장 노동자 시인 작가 화가 가수 배우 무용수 스님 신부 목사 시인
동네사람 행인 아이들 사람들 군중 무리 관객 독자 청중"""

LEXICON = sorted({w for block in (FROM_S2_2, KINSHIP, GENERIC) for w in block.split()})
NOUN_TAGS = {"NNG", "NNP"}          # common and proper nouns

# The paper's claim is that OTHER PEOPLE disappear, not that person words disappear.
# Korean prose poems often drop the subject, and generated works tend to state the
# speaker, so pooling 'I' with 'my mother' would measure the opposite of the claim.
# The three registers are therefore counted and reported apart. Pronouns carry the
# tag NP, so the noun lexicon above never touches them.
SPEAKER = {"나", "내", "저", "제", "우리", "우리들"}          # NP, speaker reference
THIRD = {"그", "그녀", "그들", "너", "네", "당신", "그이", "당신들"}   # NP, other reference
PRON_TAG = "NP"


def noun_tokens(tagger, text):
    """Returns (other-referring nouns, speaker pronouns, third-person pronouns)."""
    nouns, spk, thd = [], 0, 0
    for w, t in tagger.pos(str(text)):
        head = t.split("+")[0]
        if head in NOUN_TAGS:
            nouns.append(w)
        elif head == PRON_TAG:
            if w in SPEAKER:
                spk += 1
            elif w in THIRD:
                thd += 1
    return nouns, spk, thd


def main():
    tagger = mecab.MeCab()
    lex = set(LEXICON)

    # The human full texts are not redistributed; the released output is counts only.
    meta = ATLAS_META if ATLAS_META.exists() else ROOT.parent / "data/atlas/metadata.csv"
    human = pd.read_csv(meta)[["work_id", "원문"]]
    dist = pd.read_csv(ROOT / "data/atlas/instrument_frozen/human_doc_motif_distribution.csv")
    human = dist[["work_id"]].merge(human, on="work_id")          # the 4,642 baseline works
    gen = pd.read_csv(ROOT / "results/phase2/topic_records.csv")
    TCOL = "text_full" if "text_full" in gen.columns else "text"   # 'text' is a 60-char head
    five = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]
    gen5 = gen[gen.model.isin(five)]

    def measure(texts):
        n_person, n_noun, has, n_spk, n_thd, has_o = [], [], [], [], [], []
        for t in texts:
            nouns, spk, thd = noun_tokens(tagger, t)
            k = sum(1 for w in nouns if w in lex)
            n_person.append(k); n_noun.append(len(nouns)); has.append(k > 0)
            n_spk.append(spk); n_thd.append(thd); has_o.append(k + thd > 0)
        n_person, n_noun = np.array(n_person), np.array(n_noun)
        n_spk, n_thd = np.array(n_spk), np.array(n_thd)
        tot = n_noun.sum()
        return {"works": int(len(n_person)),
                "mean_person_nouns_per_work": round(float(n_person.mean()), 3),
                "person_nouns_per_100_nouns": round(float(100 * n_person.sum() / tot), 3),
                "works_with_at_least_one_pct": round(float(100 * has.count(True) / len(has)), 1),
                # speaker vs other, reported apart
                "speaker_pron_per_100_nouns": round(float(100 * n_spk.sum() / tot), 3),
                "third_pron_per_100_nouns": round(float(100 * n_thd.sum() / tot), 3),
                "other_reference_per_100_nouns": round(float(100 * (n_person.sum() + n_thd.sum()) / tot), 3),
                "works_with_any_other_reference_pct": round(float(100 * has_o.count(True) / len(has_o)), 1),
                "median_nouns_per_work": int(np.median(n_noun))}

    summary = {"lexicon_size": len(LEXICON), "human": measure(human["원문"])}
    for m in five:
        summary[m] = measure(gen5[gen5.model == m][TCOL])
    summary["generated_five_pooled"] = measure(gen5[TCOL])

    # per motif: intensive works (share >= 10%) on each side
    cols = [c for c in dist.columns if re.fullmatch(r"T\d+_pct", c)]
    names = pd.read_csv(ROOT / "results/phase2/retention_ladder.csv").set_index("motif")
    hj = dist.merge(human, on="work_id")
    rows = []
    for c in cols:
        t = c[:-4]
        hi = hj[hj[c] >= 10]
        gi = gen5[gen5[t] * 100 >= 10] if t in gen5.columns else gen5.iloc[0:0]
        r = {"motif": t, "name_en": names.loc[t, "name_en"],
             "retention": round(float(names.loc[t, "retention"]), 3),
             "n_human": len(hi), "n_gen": len(gi)}
        r.update({f"human_{k}": v for k, v in measure(hi["원문"]).items() if k != "works"})
        if len(gi):
            r.update({f"gen_{k}": v for k, v in measure(gi[TCOL]).items() if k != "works"})
        rows.append(r)
    per_motif = pd.DataFrame(rows)
    per_motif.to_csv(OUT / "person_word_density_by_motif.csv", index=False)

    # ------------------------------------------------------------- inference
    from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

    def counts(texts):
        a = []
        for t in texts:
            ns, _spk, thd = noun_tokens(tagger, t)
            a.append((sum(1 for w in ns if w in lex) + thd, len(ns)))
        return np.array(a, dtype=float)

    ch, cg = counts(human["원문"]), counts(gen5[TCOL])
    rh = 100 * ch[:, 0] / np.maximum(ch[:, 1], 1)
    rg = 100 * cg[:, 0] / np.maximum(cg[:, 1], 1)
    rng = np.random.default_rng(42)
    boot = [ (100*ch[i,0].sum()/ch[i,1].sum()) / (100*cg[j,0].sum()/cg[j,1].sum())
             for i, j in ((rng.integers(0, len(ch), len(ch)), rng.integers(0, len(cg), len(cg)))
                          for _ in range(2000)) ]
    u = mannwhitneyu(rh, rg, alternative="greater")
    f = fisher_exact([[int((ch[:,0]>0).sum()), int((ch[:,0]==0).sum())],
                      [int((cg[:,0]>0).sum()), int((cg[:,0]==0).sum())]], alternative="greater")
    sp = spearmanr(per_motif.retention, per_motif.human_person_nouns_per_100_nouns)
    summary["tests"] = {
        "density_ratio_human_over_generated": round(float(np.mean(boot)), 3),
        "density_ratio_ci95": [round(float(np.percentile(boot, 2.5)), 3),
                               round(float(np.percentile(boot, 97.5)), 3)],
        "mannwhitney_u_p_one_sided": float(u.pvalue),
        "fisher_p_one_sided_share_with_at_least_one": float(f.pvalue),
        "spearman_retention_vs_human_person_density": {
            "rho": round(float(sp.statistic), 3), "p": round(float(sp.pvalue), 4), "n_motifs": len(per_motif)},
    }
    json.dump({"lexicon_other_nouns": LEXICON, "speaker_pronouns": sorted(SPEAKER),
               "third_person_pronouns": sorted(THIRD), "summary": summary}, open(OUT / "person_word_density.json", "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
