# Korean labels in this package, and where their English equivalents are

The documentation here is in English. The **data** are in Korean, because Korean prose
poetry is the object of study. Where a released file carries Korean labels or Korean
text, this page gives the English term used in the paper, and points to the place in
the Supplementary Information (SI) where the same material appears with an English
translation beside the Korean.

---

## 1. Ending-audit categories

`results/phase2/closure_audit_labels.csv`, column `category`; `docs/closure_audit.md`.

| Korean | English (paper) | Definition |
|---|---|---|
| 봉합 | **sealing** | the ending is closed by consolation, insight, acceptance or forgiveness |
| 미해소 | **unresolved/open** | the ending is a question, a break, a refusal or a continuing action, so tension remains |
| 경계 | **borderline** | the ending cannot be settled either way; not counted in the numerator of sealing |

Column `object_present` records the sub-judgment made on every unresolved item:
whether the ending holds a specific person, an event with an agent, or a tangible
object. An unresolved ending holding none of the three is called **free-floating** in
the paper (§3.6).

Full definitions: §2.4.6 of the paper. All 180 items with English translations: **SI S8**.

## 2. Ending-audit strata

`results/phase2/closure_audit_labels.csv`, column `stratum`; `closure_audit_table.csv`.

| Korean | English (Table 5) | | Korean | English (Table 5) |
|---|---|---|---|---|
| GPT 비 | GPT rain-intensive | | 인간 비 | Human rain-intensive |
| Solar 비 | Solar rain-intensive | | 인간 겨울 | Human winter |
| EXAONE 비 | EXAONE rain-intensive | | 인간 가족 | Human family |
| Opus 비 | Opus rain-intensive | | 인간 상처 | Human wounds |
| Opus 겨울 | Opus winter | | 인간 노인 | Human old age |
| Opus 가족 | Opus family | | 인간 병원 | Human hospital |
| Gemini 비 | Gemini rain-intensive | | 인간 학교 | Human school |
| Gemini 겨울 | Gemini winter | | | |

## 3. Motif names (T0–T24)

Used wherever a motif is named: `motif_map.csv`, `retention_ladder.csv`,
`retention_ci.csv`, `company_profile.csv`, `motif_full_distribution_balanced.csv`,
`concentration_two_axes.csv`, `avoidance_set_stability.json`, `topic_records.csv`,
`naming_set_typicals.csv`, `human_top_works.csv`, `human_top20_by_motif.csv`,
`centroid_keyword_alignment.csv`, and `human_per_motif_summary.csv` in
the frozen instrument. Every file that carries a Korean motif name also carries the
English name of the paper in a parallel column (`name_en`; `fx25_top1_motif_en` in
`concentration_two_axes.csv`; `*_en` arrays in `avoidance_set_stability.json`). The
remaining non-numeric columns are the Korean `topwords` (below), `sig_vs_human_CI` in
`motif_map.csv` (`**` marks a motif whose generated mean lies outside the human
bootstrap 95% CI, blank otherwise), and the generated texts in `topic_records.csv`
(`text`, `text_full`) and `naming_set_typicals.csv` (`head`, `text`).

| | | | | |
|---|---|---|---|---|
| T0 sea | T1 flowers | T2 family | T3 food/animals | T4 affect |
| T5 school/child | T6 winter | T7 rain | T8 woman | T9 old age/underground |
| T10 birds/butterflies | T11 meta-poetic | T12 letters/writing | T13 interior | T14 music |
| T15 street | T16 cat | T17 cosmos | T18 wounds/drink | T19 fruit |
| T20 grandparents | T21 mirror | T22 hospital/illness | T23 apartment/wall | T24 bus |

The `topwords` column of these files holds the ten c-TF-IDF keywords of each motif in
Korean. **SI Table S2-2** prints the same ten words with an English gloss, and marks
which of them denote a person.

The two groups named in the paper:

- **the avoided four** (회피 넷) — T18 wounds, T9 old age, T22 hospital, T5 school
- **the increased eight** (늘어난 여덟) — T7 rain, T15 street, T6 winter, T17 cosmos,
  T13 interior, T16 cat, T23 apartment/wall, T12 letters/writing

## 4. Where the Korean text has an English counterpart

| File in this package | Korean content | English counterpart |
|---|---|---|
| `results/phase2/closure_audit_samples_public.txt` | the judgment unit of all 180 audit items: the final ~170 characters of the generated works; bibliography and one phrase for the human works | **SI S8** — every item with an English translation beside the Korean |
| `results/phase2/closure_audit_labels.csv` | the `reason` column (why each of the 180 endings was judged as it was) and the `object_note` column, both free text in Korean | the parallel columns `stratum_en`, `category_en`, `reason_en` and `object_note_en` in the same file (the English of **SI S8**) |
| `data/generated/arm_ab/*.jsonl` | the generated poems themselves, 4,876 rows across six systems | not translated as a corpus. The works the paper quotes are translated where they are quoted (§3.6, §4.1, §4.4, §4.5); the 180 audited endings are translated in **SI S8** |
| `docs/closure_audit.md` | Part B is the criteria document as it was frozen, in Korean; the appendix holds all 180 judgments with their reasons | Part A of the same file is an English translation of the criteria; the procedure is §2.4.6; per-item categories and reasons in English: **SI S8** |
| `docs/prompts_frozen_2026-07-28.json` | the twelve prompts as sent, in Korean, with SHA-256 | **SI S5** — each prompt in Korean and English with the same hash |
| `data/atlas/instrument_frozen/topic_info.csv` | the motif keywords | **SI Table S2-2** — the ten keywords per motif, Korean with English gloss |
| `results/phase2c/bertopic_fresh_topics.csv` | keywords of the fifteen generated-axis motifs | **SI Table S3-1** (keywords with gloss) and **SI Table S3-3** (their mapping onto the human-axis motifs) |
| `results/phase2/naming_set_typicals.csv` | the 20 typical-work slots (18 distinct works) read in naming the window-and-street scene | quoted and translated in §4.1; the feature tally in `naming_set_features.csv` is summarized there |
| `results/phase2/human_top20_by_motif.csv` | `poet` and `title` in Hangul (bibliographic data), `name` | `name_en` beside the motif name; poet and title are the works' bibliographic identifiers and are left in the original script, as in SI Table S2-3 |
| `results/phase2/human_top_works.csv` | poet, title, motif name and the `reason` column as recorded | the parallel columns `poet_en` (romanization), `title_en`, `name_en` and `reason_en`; the six works read closely are quoted in §4.2 with the Korean original beside the English translation |
| `results/phase2/topic_records.csv` | the full text of the 1,625 generated works (column `text_full`) | the generated works quoted in the paper appear in §3.6, §4.1, §4.4 and §4.5 with translations |

## 5. Column names that appear in Korean

A few derived tables carry Korean in a value rather than a header. Besides the
categories and strata above:

| Where | Korean value | English |
|---|---|---|
| `motif_map.csv`, `retention_ladder.csv` etc., `topwords` | the ten motif keywords | SI Table S2-2 |
| `naming_set_features.csv`, coded values | 예 / 아니오, 없음(…), 골목·길, 창가, 걷기, 저녁 … | the six `*_en` columns beside them give each coded value in English (§4.1) |
| `close_reading_provenance.json` | the `label_ko` field of each entry (the label the record was first written under) | the entry's key is the English label; `source` values are English |
| `closure_audit_samples_public.txt`, stratum headers | `GPT 비 (population 213, sample 12)` | each header carries the English stratum label after a slash |
| `bertopic_fresh_topics.csv`, `pass` | — | which reliability checks the motif passed: `1`, `2`, `3` for R1–R3 and `α` for document-membership alpha (`123α` = all four); the same information as `n_axes_passed` |

Free-text notes inside `summary.json`, `form_x_topic_protocol.json`,
`human_split_check.json` and `company_weighted.json` are in English and match the strings
the scripts emit (`company_weighted.json` keeps the original Korean note as `note_ko`).

## 6. Spelling in code and data

The prose in the paper and in this documentation uses British spelling
(*neighbour*, *centre*). Code identifiers, API parameter names and JSON keys keep the
spelling of the libraries they come from and are not changed: `neighbors` (a key in
`company_weighted.json`), `n_neighbors` (a UMAP parameter), `eff_neighbors` (a function
in `robustness_checks.py`), `ha="center"` (matplotlib). They mean the same thing as
*neighbour* in the paper.
