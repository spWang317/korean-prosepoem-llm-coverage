# 산문시 LLM 커버리지 — 공개 코드·데이터 패키지

논문 "산문시 LLM 커버리지"(DSH 투고)의 생성 로그, 분석 코드, 파생 산출물이다.
공개 범위와 저작권 처리는 `docs/data_release_notes.md`, 결말 감사의 기준 문서와 전 항목 판정은 `docs/closure_audit.md`에 있다.

- 스크립트 파일명에 순서 번호를 붙이지 않은 것은 스크립트 간 import(모듈명) 때문이다. 실행 순서는 아래 단계 번호를 따른다.
- 계측기 동결 자산(모티프 중심 벡터·인간 지분 분포)은 `data/atlas/instrument_frozen/`에 포함되어 있으므로, 단계 2·5·6·12를 포함한 대부분의 단계가 이 패키지만으로 재실행된다. † 표시 두 단계(10·11)만 인간 작품 원문이 필요해 재실행되지 않는다(저작권상 원문은 재배포하지 않는다). 그 산출물은 함께 공개되어 있으므로 검증은 산출물 대조로 가능하다.
- 임베딩·BERTopic 단계는 torch가 있는 인터프리터가 필요하다. 형태소 분석기는 python-mecab-ko.

## 실행 순서

| 단계 | 스크립트 | 입력 | 산출 | 본문 |
|---|---|---|---|---|
| 0 | `common.py`, `prompt_set.py` | (모듈: 형식 판정·모델 레지스트리 / 프롬프트 12종) | — | §2.3 |
| 1 | `gen_full.py`(API 5종), `gen_exaone_full.py`(로컬 vLLM; 셀당 200편 일괄 생성) | 프롬프트 동결본 | `data/generated/arm_ab/*.jsonl` | §2.3 |
| 2 | `phase2_topic_full.py` | 생성 로그 + 모티프 중심 | `topic_records.csv`, `summary.json`, `motif_map.csv`, `motif_full_distribution_balanced.csv` | §2.2, 표 3·그림 1의 원천 |
| 3 | `retention_ladder.py` | topic_records + 인간 평균 | `retention_ladder.csv` | 표 2, 표 S1-1·S1-3 |
| 4 | `company_weighted.py` → `company_profile.py` | topic_records | `company_weighted.json`, `company_profile.csv` | 표 4, §3.4 |
| 5 | `robustness_checks.py` | topic_records + 인간 분포 | `retention_ci.csv`, `avoidance_set_stability.json`, `length_check.json`, `human_split_check.json`, company_profile의 ESS 열 | 표 2 CI, §3.2.2, S1.1–S1.2 |
| 6 | `phase2f_concentration.py` | summary + 인간 분포 | `concentration_two_axes.csv` | 표 3 |
| 7 | `repetition_check.py` | 생성 로그 | `repetition_check.csv` | 표 S1-4 |
| 8 | `phase2c_bertopic_fresh.py` → `phase2h_fresh_soft.py` | 생성 로그 | `results/phase2c/*`, `fresh_soft_projection.csv` | §3.5, 표 S3-1·S3-2 |
| 9 | `phase2g_regime_audit.py` | 생성 로그 | `form_x_topic_protocol.json` | 표 S6-1 |
| 10† | `closure_audit_sampling.py` → (사람 판정: `docs/closure_audit.md`) → `closure_audit_stats.py` | topic_records + 인간 원문(미포함) | `closure_audit_labels.csv`, `closure_audit_table.csv`, `closure_audit_samples_public.txt` | 표 5, SI S8 |
| 11† | `close_reading_provenance.py` | topic_records, 임베딩 + 인간 원문(미포함) | `close_reading_provenance.json`, `naming_set_typicals.csv`, `human_top_works.csv` | §4.1–§4.3, SI S7 |
| 12 | `human_motif_ci.py` | 인간 분포 | `human_motif_ci.json` | 표 S1-2 |
| 13 | `make_figures.py` | 산출물 | `figures/*`(제출명 figure1, figureS1-1, figureS9-1) | 그림 1, S1-1, S9-1 |

## 파일 인벤토리

### data/
- `generated/arm_ab/*.jsonl` — 여섯 시스템의 생성 로그 전체(시도·기각 포함, 프롬프트 SHA-256 기록). EXAONE은 셀당 200편의 전체 배치(2,400행)이며, 분석 코드가 셀마다 25번째 유효작까지를 프로토콜 표본으로 자른다.

### docs/
- `closure_audit.md` — 결말 감사 분류 기준 문서(사전 고정본)와 180항목 판정 전부.
- `data_release_notes.md` — 공개/비공개 범위, 저작권 처리, EXAONE 표집 방식.
- `prompts_frozen_2026-07-28.json` — 프롬프트 12종 동결본과 SHA-256.

### results/phase2/ (본문·SI 표의 원천)
- `topic_records.csv` — 생성 1,625편의 작품별 25모티프 지분·잔차·엔트로피·전문.
- `summary.json` — 투영 단계의 시스템별 요약(유효율·엔트로피·평균 지분).
- `motif_map.csv` — 25모티프 인간·생성 평균 지분(그림 1의 자료).
- `motif_full_distribution_balanced.csv` — 시스템별 25모티프 분포(표 S1-1, 그림 S1-1).
- `retention_ladder.csv` — 25모티프 유지율(전체·시스템별·시스템별 최대).
- `retention_ci.csv` — 유지율의 셀-군집 부트스트랩 95% CI(표 2).
- `avoidance_set_stability.json` — 회피 넷 재추출 안정성(§3.2.2).
- `human_motif_ci.json` — 인간 평균 지분의 부트스트랩 95% CI(표 S1-2).
- `concentration_two_axes.csv` — 집중도: 유효 모티프 수·최대 지분·잔차(표 3).
- `repetition_check.csv` — 반복 지표: 전문·서두 종수(표 S1-4).
- `length_check.json`, `human_split_check.json` — 길이·분할 점검(S1.1, S1.2).
- `company_weighted.json` — 25앵커 가중 동행의 전 이웃 행렬(§3.4).
- `company_profile.csv` — 앵커별 이웃 프로필 요약: 총변동 거리·유효 이웃 수·두 쪽 몫·ESS(표 4).
- `form_x_topic_protocol.json` — 형식 필터 점검: 유효 대 기각 주제분포(표 S6-1).
- `fresh_soft_projection.csv` — 생성 축 투영 요약(표 S3-2, §3.5).
- `closure_audit_labels.csv` — 결말 감사 180항목 판정(구절 열 없음).
- `closure_audit_table.csv` — 표 5 산출(Wilson CI·Fisher p).
- `closure_audit_samples_public.txt` — 감사 표본 공개판(생성 결말 170자 전문, 인간은 서지+한 구절).
- `close_reading_provenance.json` — 인용작(전형작) 선별 기록: 가중 중심 순위.
- `naming_set_typicals.csv` — '창가와 길의 정경' 명명에 읽은 전형작 스무 자리(서로 다른 18편).
- `naming_set_features.csv` — 그 18편을 연구자가 읽어 코딩한 특징 집계(스크립트 산출 아님).
- `human_top_works.csv` — 인간 상위작 순위표(서지·지분·선택 사유; 원문 없음).

### results/phase2c/ (생성 축)
- `bertopic_fresh_topics.csv` — 생성 축 모티프 15개: 키워드·크기·지배 시스템·신뢰성 검사(표 S3-1).
- `bertopic_fresh_summary.json` — 재적합 요약(파라미터·이상치 비율).
- `gen_embeddings.npy` — 생성물 통글 임베딩(인간 임베딩은 원문 복원 가능성 때문에 비공개).
- `indicator_matrices.npz` — 시드 반복의 문서소속 행렬(Cronbach's α 입력).

### figures/
- `figure1.tif`, `figureS1-1.tif`, `figureS9-1.tif` — 제출용 600dpi TIFF(본문 그림 1, SI 그림 S1-1·S9-1).
- `fig2_motif_map.png`, `figS6_motif_by_system.png`, `fig1_form_heatmap.png` — 같은 그림의 PNG 판.

### scripts/ (19개)
위 실행 순서 표 참조. `common.py`(경로·형식 판정·모델 레지스트리)와 `prompt_set.py`(프롬프트 12종 생성)는 다른 스크립트가 import하는 모듈이다.
