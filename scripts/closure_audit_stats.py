"""
closure_audit_stats.py — Table 5 from the closure-audit labels (paper §3.6, SI S8).

Reads results/phase2/closure_audit_labels.csv (one row per audited ending: stratum, item
1-12, category in {sealing, unresolved, borderline} as the Korean values 봉합/미해소/경계, reason)
and prints, per stratum, the counts, the
sealing rate with its Wilson 95% interval, and a one-sided Fisher exact test (generated >
human) against the paired human stratum (the rain strata against 인간 비 = human rain;
Opus 겨울/가족 = winter/family against the human ones). Stratum and category names are
the CSV values; docs/glossary.md maps them to the English terms used in the paper.
The labels themselves come from the reading procedure in docs/closure_audit.md.
Run: python3 scripts/closure_audit_stats.py
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
PAIR = {"GPT 비": "인간 비", "Solar 비": "인간 비", "EXAONE 비": "인간 비", "Opus 비": "인간 비",
        "Gemini 비": "인간 비", "Gemini 겨울": "인간 겨울", "Opus 겨울": "인간 겨울", "Opus 가족": "인간 가족"}
ORDER = ["GPT 비", "Solar 비", "EXAONE 비", "Opus 겨울", "Opus 가족", "Opus 비", "Gemini 비", "Gemini 겨울",
         "인간 비", "인간 겨울", "인간 가족", "인간 상처", "인간 노인", "인간 병원", "인간 학교"]
# English stratum labels, exactly the row labels of Table 5 of the paper. The Korean
# values stay in the `stratum` column; the English goes to `stratum_en` beside it.
STRATUM_EN = {"GPT 비": "GPT rain-intensive", "Solar 비": "Solar rain-intensive",
              "EXAONE 비": "EXAONE rain-intensive", "Opus 겨울": "Opus winter",
              "Opus 가족": "Opus family", "Opus 비": "Opus rain-intensive",
              "Gemini 비": "Gemini rain-intensive", "Gemini 겨울": "Gemini winter",
              "인간 비": "Human rain-intensive", "인간 겨울": "Human winter",
              "인간 가족": "Human family", "인간 상처": "Human wounds",
              "인간 노인": "Human old age", "인간 병원": "Human hospital",
              "인간 학교": "Human school"}


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - hw)), round(100 * (c + hw))


def main():
    df = pd.read_csv(ROOT / "results/phase2/closure_audit_labels.csv")
    counts = {s: g.category.value_counts().to_dict() for s, g in df.groupby("stratum")}
    rows = []
    for s in ORDER:
        c = counts[s]; b, u, e = c.get("봉합", 0), c.get("미해소", 0), c.get("경계", 0); n = b + u + e
        lo, hi = wilson(b, n)
        p = ""
        if s in PAIR:
            hb = counts[PAIR[s]].get("봉합", 0); hn = sum(counts[PAIR[s]].values())
            pv = fisher_exact([[b, n - b], [hb, hn - hb]], alternative="greater")[1]
            p = "<0.0001" if pv < 1e-4 else f"{pv:.4f}"
        rows.append([s, STRATUM_EN[s], b, u, e, f"{round(100 * b / n)}% [{lo}, {hi}]", p])
    if "object_present" in df.columns:
        u = df[df.category == "미해소"]
        print("\nunresolved endings, presence of an object (a person, an event or a thing within the final 170 characters):")
        print(u.groupby("stratum").object_present.value_counts().unstack(fill_value=0).reindex([s for s in ORDER if s in set(u.stratum)]).to_string())
    out = pd.DataFrame(rows, columns=["stratum", "stratum_en", "sealing", "unresolved_open", "borderline",
                                      "sealing_rate [Wilson 95% CI]", "Fisher p (vs paired human)"])
    print(out.to_string(index=False))
    out.to_csv(ROOT / "results/phase2/closure_audit_table.csv", index=False)


if __name__ == "__main__":
    main()
