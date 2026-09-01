# 공개 패키지 목록

구성·실행 순서·파일별 설명은 `README.md`에 있다. 공개 범위는 `docs/data_release_notes.md`, 결말 감사 기준 문서는 `docs/closure_audit.md`.
포함하지 않는 것: 인간 작품 원문(`data/atlas/metadata.csv`). 결말 감사의 인간 표본은 서지와 판정 근거 구절만 포함한다. 인간 작품을 읽는 단계(인간 층 표본 추출, 인간 상위작 순위표)는 원문 없이는 재실행되지 않으며, 그 산출물(`human_top_works.csv`, `docs/closure_audit.md` 부록)은 서지·지분·판정만 담아 공개한다. 계측기 동결 자산은 `data/atlas/instrument_frozen/`에 포함되어 있다: `motif_centroids.npz`(25모티프 중심, L2 정규화), `human_doc_motif_distribution.csv`(인간 4,642편의 작품별 모티프 지분·잔차), `human_per_motif_summary.csv`(모티프별 코퍼스 요약), `topic_info.csv`(모티프 메타데이터; 원문 재구성 가능한 열 제거), `PROVENANCE.md`(출처와 범위).
