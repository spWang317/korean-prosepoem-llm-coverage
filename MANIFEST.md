# Contents of the release package

The layout, the run order and the per-file description are in `README.md`; the scope
of the release is in `docs/data_release_notes.md`; the ending audit's criteria
document is in `docs/closure_audit.md`.

**Not included:** the full text of the human poems (`data/atlas/metadata.csv`). The
human items of the ending audit carry only bibliographic details and the single
phrase on which the judgment rests. The two steps that read human poems — drawing the
human strata and ranking the human top-share works — therefore do not re-run from
this package; their outputs (`human_top_works.csv`, the appendix of
`docs/closure_audit.md`) are released, carrying bibliography, shares and judgments
only.

**Included:** the frozen instrument assets, in `data/atlas/instrument_frozen/` —
`motif_centroids.npz` (the 25 motif centroids, L2-normalized),
`human_doc_motif_distribution.csv` (per-work motif shares and residual for the 4,642
human works), `human_per_motif_summary.csv` (per-motif corpus summary),
`topic_info.csv` (motif metadata, with the column from which source text could be
reconstructed removed) and `PROVENANCE.md` (their origin and scope).
