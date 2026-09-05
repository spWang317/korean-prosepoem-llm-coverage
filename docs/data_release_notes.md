# Scope of the data and code release

## Released

- **The complete generation logs** (`data/generated/arm_ab/*.jsonl`): attempts and
  rejections included, with a SHA-256 record of each prompt. For EXAONE the full batch
  of 200 works per cell is included; the analysis code (`phase2_topic_full.py`) cuts
  each cell at the 25th valid work to form the protocol sample used in the paper.
- **The analysis code** (`scripts/`) **and the derived artefacts**
  (`results/phase2/`, `results/phase2c/`): retention (`retention_ladder.csv`), company
  (`company_weighted.json`, `company_profile.csv`), the generated-axis audit
  (`bertopic_fresh_topics.csv`, `fresh_soft_projection.csv`), the ending audit
  (`closure_audit_samples_public.txt`, `closure_audit_labels.csv`,
  `closure_audit_table.csv`, `closure_audit_power.csv`), the two instrument checks that need no corpus
  (`human_top20_by_motif.csv`, `centroid_keyword_alignment.csv`), the selection record for the quoted works
  (`close_reading_provenance.json`, `naming_set_typicals.csv`,
  `naming_set_features.csv`, `human_top_works.csv`), the bootstrap confidence
  intervals (`human_motif_ci.json`, `retention_ci.csv`), the robustness checks
  (`avoidance_set_stability.json`, `length_check.json`, `human_split_check.json`) and
  the filter check (`form_x_topic_protocol.json`).
- **The ending audit's criteria document and every judgment**
  (`docs/closure_audit.md`, `results/phase2/closure_audit_labels.csv`), together with
  the judgment units of the sample (`results/phase2/closure_audit_samples_public.txt`:
  roughly the last 170 characters for the 96 generated works, bibliography and the
  single phrase on which the judgment rests for the 84 human ones).
  `closure_audit_labels.csv` carries only the category, the reason and the
  object sub-judgment; the human phrases appear, with their bibliography, in the
  samples file alone.
- **The frozen instrument assets** (motif centroid vectors, human share percentages,
  motif metadata; `data/atlas/instrument_frozen/`). These are the product of a
  companion study. Once that study's publication process is complete they will be
  removed from this repository, which will then point to the companion study's own
  repository.

## Not released

- **The full text of the human poems** (the source-text column of
  `data/atlas/metadata.csv`). It is not redistributed, for copyright reasons. The
  paper and the released artefacts carry only share percentages, bibliographic
  details, short phrases, and the phrases on which the ending-audit judgments rest.

## Sampling protocol (summary)

Generation stops in each cell (system × prompt) once 25 valid works have accumulated.
The base cap is 60 attempts, transport failures included; only a cell that produced at
least one valid output at the cap is extended to 400. Calls that returned no response
(HTTP 400 usage limit, HTTP 429 rate limit) stay in the log but are excluded from the
form judgment and from the validity rates. HCX made 1,002 attempts, of which 224 were
429 transport failures; its 778 completed calls are 88 in the three cells that filled
early, 321 in the seven cells with no valid work, and 369 in the two extended cells
(SI S9). Opus made 369 attempts, of which 69 were 400 transport failures, and all 300
completed calls are valid.

The construction of the human baseline (the four rules of §2.1 and the 5,166 → 4,642
filter) and the fitting of the 25-motif instrument belong to the companion study; this
package takes their output — the human shares and the motif metadata — as input only.

Re-running `gen_full.py` also writes `run_manifest.json` (the per-run call log). It is a working file and is not part of this release.
