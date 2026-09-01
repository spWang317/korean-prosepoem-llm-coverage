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
| 미해소 | **unresolved / open** | the ending is a question, a break, a refusal or a continuing action, so tension remains |
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

Used in `motif_map.csv`, `retention_ladder.csv`, `company_profile.csv`,
`motif_full_distribution_balanced.csv`, `topic_records.csv` and elsewhere.

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
| `docs/closure_audit.md` | the frozen criteria document and all 180 judgments with their reasons | criteria: §2.4.6; per-item categories and reasons in English: **SI S8** |
| `docs/prompts_frozen_2026-07-28.json` | the twelve prompts as sent, in Korean, with SHA-256 | **SI S5** — each prompt in Korean and English with the same hash |
| `data/atlas/instrument_frozen/topic_info.csv` | the motif keywords | **SI Table S2-2** — the ten keywords per motif, Korean with English gloss |
| `results/phase2c/bertopic_fresh_topics.csv` | keywords of the fifteen generated-axis motifs | **SI Table S3-1** (keywords with gloss) and **SI Table S3-3** (their mapping onto the human-axis motifs) |
| `results/phase2/naming_set_typicals.csv` | the 20 typical-work slots (18 distinct works) read in naming the window-and-street scene | quoted and translated in §4.1; the feature tally in `naming_set_features.csv` is summarized there |
| `results/phase2/human_top_works.csv` | poet, title and reason for the human top-share works | the six read closely are quoted in §4.2 with the Korean original beside the English translation |
| `results/phase2/topic_records.csv` | the full text of the 1,625 generated works (column `text_full`) | the generated works quoted in the paper appear in §3.6, §4.1, §4.3 and §4.5 with translations |

## 5. Column names that appear in Korean

A few derived tables carry Korean in a value rather than a header. Besides the
categories and strata above:

| Where | Korean value | English |
|---|---|---|
| `motif_map.csv`, `retention_ladder.csv` etc., `topwords` | the ten motif keywords | SI Table S2-2 |
| `naming_set_features.csv`, feature names | 혼자, 창가, 길, 사건 없음, 부재자 | alone, at a window, on a road, no event, an absent figure (§4.1) |
| `close_reading_provenance.json` | 전형작 / 상위작 | typical work / top-share work (§2.4.7) |

## 6. Spelling in code and data

The prose in the paper and in this documentation uses British spelling
(*neighbour*, *centre*). Code identifiers, API parameter names and JSON keys keep the
spelling of the libraries they come from and are not changed: `neighbors` (a key in
`company_weighted.json`), `n_neighbors` (a UMAP parameter), `eff_neighbors` (a function
in `robustness_checks.py`), `ha="center"` (matplotlib). They mean the same thing as
*neighbour* in the paper.
