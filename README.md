# Korean prose poetry and LLM subject-matter coverage — code and data package

Generation logs, analysis code and derived artefacts for the manuscript submitted to
*Digital Scholarship in the Humanities*. The scope of what is and is not released, and
the copyright handling, are in `docs/data_release_notes.md`; the frozen criteria
document and all 180 judgments of the ending audit are in `docs/closure_audit.md`.

- Script filenames carry no ordering prefix because the scripts import one another as
  modules. Run them in the order of the step numbers below.
- The frozen instrument assets (motif centroid vectors, human share distribution) are
  included in `data/atlas/instrument_frozen/`, so most steps — including 2, 5, 6 and
  12 — re-run from this package alone. Only the two steps marked † need the full text
  of the human poems, which is not redistributed for copyright reasons; their outputs
  are released here, so those steps can be verified by comparing artefacts.
- The embedding and BERTopic steps need an interpreter with torch. The morphological
  analyser is python-mecab-ko.

## What this measures

Six deployed systems were asked for Korean prose poems under twelve prompts. Every
form-compliant output is projected onto a frozen 25-motif instrument built on the
complete set of human prose poems published in 2000--2024 by four publishers (4,642
works), and the two corpora are compared on four measurements.

| Measurement | Statistic | Artefact | Paper |
|---|---|---|---|
| Form | per-cell validity rate and failure types; reported per system as the cell-mean validity rate with its cell range | `summary.json`, `form_x_topic_protocol.json` | Table 1, SI S6, S9 |
| Direction | retention = generated mean share ÷ human mean share, with a cell-cluster bootstrap CI | `retention_ladder.csv`, `retention_ci.csv`, `avoidance_set_stability.json` | Table 2, Fig. 1 |
| Shape | effective number of motifs (inverse Simpson), largest share, residual; per-work normalized entropy | `concentration_two_axes.csv`, `repetition_check.csv` | Table 3, SI Table S1-4 |
| Company | share-weighted neighbour profile per anchor: total variation distance, effective neighbours, side shares, ESS | `company_weighted.json`, `company_profile.csv` | Table 4 |
| Endings | fixed-criteria audit of a 180-work stratified sample, Wilson CI and one-sided Fisher test | `closure_audit_labels.csv`, `closure_audit_table.csv` | Table 5, SI S8 |

A second axis, refitted on the generated corpus alone, is compared with the human axis
to test whether the generated poems contain subjects the human axis cannot hold
(`results/phase2c/`, §3.5).

## Run order

| Step | Script | Input | Output | Paper |
|---|---|---|---|---|
| 0 | `common.py`, `prompt_set.py` | (modules: form judgment and model registry / the twelve prompts) | — | §2.3 |
| 1 | `gen_full.py` (five APIs), `gen_exaone_full.py` (local vLLM; 200 works per cell in one batch) | frozen prompts | `data/generated/arm_ab/*.jsonl` | §2.3 |
| 2 | `phase2_topic_full.py` | generation logs + motif centroids | `topic_records.csv`, `summary.json`, `motif_map.csv`, `motif_full_distribution_balanced.csv` | §2.2; source of Table 3 and Fig. 1 |
| 3 | `retention_ladder.py` | topic_records + human means | `retention_ladder.csv` | Table 2, SI Tables S1-1 and S1-3 |
| 4 | `company_weighted.py` → `company_profile.py` | topic_records | `company_weighted.json`, `company_profile.csv` | Table 4, §3.4 |
| 5 | `robustness_checks.py` | topic_records + human distribution | `retention_ci.csv`, `avoidance_set_stability.json`, `length_check.json`, `human_split_check.json`, the ESS column of company_profile | Table 2 CI, §3.2.2, SI S1.1–S1.2 |
| 6 | `phase2f_concentration.py` | summary + human distribution | `concentration_two_axes.csv` | Table 3 |
| 7 | `repetition_check.py` | generation logs | `repetition_check.csv` | SI Table S1-4 |
| 8 | `phase2c_bertopic_fresh.py` → `phase2h_fresh_soft.py` | generation logs | `results/phase2c/*`, `fresh_soft_projection.csv` | §3.5, SI Tables S3-1 and S3-2 |
| 9 | `phase2g_regime_audit.py` | generation logs | `form_x_topic_protocol.json` | SI Table S6-1 |
| 10† | `closure_audit_sampling.py` → (human classification: `docs/closure_audit.md`) → `closure_audit_stats.py` | topic_records + human full text (not included) | `closure_audit_labels.csv`, `closure_audit_table.csv`, `closure_audit_samples_public.txt` | Table 5, SI S8 |
| 10b | `closure_audit_power.py` | — (design only: n = 12 per stratum) | `closure_audit_power.csv` | SI S8 (detection threshold of the per-stratum Fisher test) |
| 10c† | `human_top20_by_motif.py` | human shares + bibliographic columns of the human metadata (not included) | `human_top20_by_motif.csv` | SI Table S2-3 |
| 10d | `centroid_keyword_alignment.py` | frozen centroids + `motif_map.csv` keywords (needs torch) | `centroid_keyword_alignment.csv` | SI Table S2-4 |
| 10e | `retention_by_prompt_factor.py` | `topic_records.csv`, `retention_ladder.csv` | `retention_by_prompt_factor.csv` | SI Table S1-8 (retention within each prompt factor) |
| 10f† | `person_word_density.py` | generated texts in `topic_records.csv` + the human full texts (not included); needs mecab-ko | `person_word_density.json`, `person_word_density_by_motif.csv` | SI Table S1-9 (person words counted directly) |
| 11† | `close_reading_provenance.py` | topic_records, embeddings + human full text (not included) | `close_reading_provenance.json`, `naming_set_typicals.csv`, `human_top_works.csv` | §4.1–§4.5, SI S7 |
| 12 | `human_motif_ci.py` | human distribution | `human_motif_ci.json` | SI Table S1-2 |
| 13 | `make_figures.py` | artefacts | `figures/*` (submission names figure1, figureS1-1, figureS9-1) | Figs 1, S1-1, S9-1 |

## File inventory

### data/
- `generated/arm_ab/*.jsonl` — the complete generation logs of the six systems
  (attempts and rejections included, with a SHA-256 record of each prompt). For EXAONE
  this is the full batch of 200 works per cell (2,400 rows); the analysis code cuts
  each cell at the 25th valid work to form the protocol sample.
- `atlas/instrument_frozen/` — the frozen instrument: motif centroids, the human
  per-work motif distribution, the per-motif corpus summary and motif metadata. See
  `PROVENANCE.md` in that directory.

### docs/
- `closure_audit.md` — the ending audit's classification criteria (frozen before
  classification began) and all 180 judgments.
- `data_release_notes.md` — what is and is not released, copyright handling, and the
  EXAONE sampling scheme.
- `prompts_frozen_2026-07-28.json` — the twelve frozen prompts with their SHA-256.

### results/phase2/ (the source of the tables in the paper and the SI)
- `topic_records.csv` — for each of the 1,625 generated works: its 25 motif shares,
  residual, entropy and full text.
- `summary.json` — per-system summary of the projection step (validity rate, entropy,
  mean shares).
- `motif_map.csv` — human and generated mean shares over the 25 motifs (the data
  behind Fig. 1).
- `motif_full_distribution_balanced.csv` — per-system distribution over the 25 motifs
  (SI Table S1-1, Fig. S1-1).
- `retention_ladder.csv` — retention for the 25 motifs (pooled, per system, and the
  per-motif maximum across systems).
- `retention_ci.csv` — cell-cluster bootstrap 95% CI of retention (Table 2).
- `avoidance_set_stability.json` — resampling stability of the avoided four (§3.2.2).
- `human_motif_ci.json` — bootstrap 95% CI of the human mean shares (SI Table S1-2).
- `concentration_two_axes.csv` — concentration: effective number of motifs, largest
  share, residual (Table 3).
- `repetition_check.csv` — repetition measures: distinct full texts and openings
  (SI Table S1-4).
- `length_check.json`, `human_split_check.json` — the length and split checks
  (SI S1.1, S1.2).
- `company_weighted.json` — the full neighbour matrix for all 25 anchors (§3.4).
- `company_profile.csv` — per-anchor summary of the company profile: total variation
  distance, effective number of neighbours, the two side shares, ESS (Table 4).
- `form_x_topic_protocol.json` — the form-filter check: subject distribution of valid
  against rejected outputs (SI Table S6-1).
- `fresh_soft_projection.csv` — summary of the projection onto the generated axis
  (SI Table S3-2, §3.5).
- `closure_audit_labels.csv` — the 180 ending-audit judgments (no phrase column).
- `closure_audit_table.csv` — the computation behind Table 5 (Wilson CI, Fisher p).
- `human_top20_by_motif.csv` — for each of the 25 motifs, the twenty human works with the highest share of it: poet, title, year, publisher, work id, share. Bibliography only; no text. Lets a reader with library access check the motif labels against the poems (SI Table S2-3).
- `centroid_keyword_alignment.csv` — cosine between each motif's ten keywords (embedded as one string with ko-sroberta-multitask) and the 25 frozen centroids; own-centroid rank first for 24/25 (SI Table S2-4). Re-runs from this package alone.
- `retention_by_prompt_factor.csv` — retention recomputed on the six cells without a genre definition, the six with one, and each of the three framings; same eight above 1 and same lowest four in every condition (SI Table S1-8).
- `person_word_density.json`, `person_word_density_by_motif.csv` — words referring to people counted directly on noun morphemes, speaker reference (나/내/우리) kept apart from reference to others; the fixed 116-word lexicon is in the json. Corpus-level and per-motif counts only, no text (SI Table S1-9).
- `closure_audit_power.csv` — the detection threshold of that test: one-sided Fisher p for every generated sealing count against a paired human count of 0, 1 or 2 out of 12 (SI S8). A property of n = 12, not of the data.
- `closure_audit_samples_public.txt` — the public version of the audit sample: the
  final 170 characters for generated works, bibliography and one phrase for human ones.
- `close_reading_provenance.json` — the selection record for the quoted typical works:
  ranking by weighted centroid similarity.
- `naming_set_typicals.csv` — the twenty typical-work slots read in naming 'the
  window-and-street scene' (18 distinct works).
- `naming_set_features.csv` — the feature tally over those 18 works, coded by the
  researcher on reading (not a script output).
- `human_top_works.csv` — the human top-share works: bibliography, share and the
  reason for selection (no full text).

### results/phase2c/ (the generated axis)
- `bertopic_fresh_topics.csv` — the fifteen motifs of the generated axis: keywords,
  size, dominant system, reliability checks (SI Table S3-1).
- `bertopic_fresh_summary.json` — summary of the refit (parameters, noise proportion).
- `gen_embeddings.npy` — whole-work embeddings of the generated poems. The human
  embeddings are withheld because the source text could be partly reconstructed
  from them.
- `indicator_matrices.npz` — document-membership matrices over the seed repetitions
  (input to Cronbach's α).

### figures/
- `figure1.tif`, `figureS1-1.tif`, `figureS9-1.tif` — 600 dpi TIFFs as submitted
  (Fig. 1 of the paper, Figs S1-1 and S9-1 of the SI).
- `fig2_motif_map.png`, `figS6_motif_by_system.png`, `fig1_form_heatmap.png` — PNG
  versions of the same figures.

### scripts/ (19 files)
See the run-order table above. `common.py` (paths, form judgment, model registry) and
`prompt_set.py` (construction of the twelve prompts) are modules imported by the other
scripts.

## Not included, and why

| Path | Reason |
|---|---|
| `data/atlas/metadata.csv` (source-text column) | full text of the human poems; not redistributed for copyright reasons |
| `results/phase2/closure_audit_samples.txt` | the working version of the audit sample; contains human full text. The public version is `closure_audit_samples_public.txt` |
| human sentence/work embeddings | the source text could be partly reconstructed from them |

The two steps marked † in the run order need the excluded human full text. Their
outputs are released in full, so those steps can be verified by comparing artefacts
rather than by re-running them.

## Row counts

| File | Rows |
|---|---|
| `results/phase2/topic_records.csv` | 1,625 generated works |
| `results/phase2/retention_ladder.csv` | 25 motifs |
| `results/phase2/company_profile.csv` | 25 anchors |
| `results/phase2/closure_audit_labels.csv` | 180 judgments over 15 strata |
| `results/phase2/human_top_works.csv` | 40 (top five for each of eight motifs) |
| `results/phase2/human_top20_by_motif.csv` | 500 (top twenty for each of 25 motifs) |
| `results/phase2/naming_set_typicals.csv` | 20 slots (18 distinct works) |
| `results/phase2c/bertopic_fresh_topics.csv` | 15 generated-axis motifs |
| `data/atlas/instrument_frozen/human_doc_motif_distribution.csv` | 4,642 human works |
| `data/generated/arm_ab/*.jsonl` | rows by system (all attempts): Gemini 300, GPT 300, Opus 369, Solar 505, HCX 1,002, EXAONE 2,400 |

## A note on language

The documentation is in English; the data are in Korean, because Korean prose poetry
is the object of study. The generation logs, the prompts, the motif keywords and the
quoted endings are left exactly as they were produced.

**`docs/glossary.md`** maps every Korean label that appears in a released table — the
three ending categories, the fifteen stratum names, the 25 motif names, the feature
names — to the English term used in the paper, and gives a file-by-file cross
reference to the place in the Supplementary Information where the same Korean material
is printed with an English translation beside it.
