# Frozen instrument assets (derived/copied 2026-07-16)

Source: the companion study, currently under review elsewhere. These files are reused
here, unchanged, as the FROZEN instrument onto which the generated works are projected.

- `motif_centroids.npz` — the 25 motif centroids (L2-normalized, in the ko-sroberta
  work-embedding space) together with topic ids and top words. Derived from the
  companion study's baseline fit (seed 100). Generated works are projected onto these
  by NNLS.
- `topic_info.csv` — motif metadata. The `Representative_Docs` lemma column has been
  REMOVED; no source text is included.
- `human_doc_motif_distribution.csv` — per-work motif percentages and unexplained
  residual for the 4,642 human works (the companion study's NNLS output). This is the
  human comparison baseline. Percentages only; no source text.
- `human_per_motif_summary.csv` — per-motif corpus summary.

Motifs passing all four reliability checks: T0 sea, T1 flowers, T7 rain,
T11 meta-poetic, T14 music, T16 cat.

No poem full text and no lemma-reconstructable content is included in these files.
