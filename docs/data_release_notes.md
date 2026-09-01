# 데이터·코드 공개 범위

## 공개하는 것
- 생성 로그 전체(`data/generated/arm_ab/*.jsonl`): 시도·기각 포함, 프롬프트 SHA-256 기록. EXAONE은 셀당 200편의 전체 배치를 싣고, 분석 코드(`phase2_topic_full.py`)가 셀마다 25번째 유효작까지의 행을 본문 프로토콜 창으로 잘라 쓴다.
- 분석 코드(`scripts/`)와 파생 산출물(`results/phase2/`, `results/phase2c/`): 유지율(`retention_ladder.csv`), 동행(`company_weighted.json`, `company_profile.csv`), 생성 축 감사(`bertopic_fresh_topics.csv`, `fresh_soft_projection.csv`), 결말 감사(`closure_audit_samples_public.txt`, `closure_audit_labels.csv`, `closure_audit_table.csv`), 인용작 선별 기록(`close_reading_provenance.json`, `naming_set_typicals.csv`, `naming_set_features.csv`, `human_top_works.csv`), 부트스트랩 CI(`human_motif_ci.json`, `retention_ci.csv`), 강건성 점검(`avoidance_set_stability.json`, `length_check.json`, `human_split_check.json`), 필터 점검(`form_x_topic_protocol.json`).
- 결말 감사 기준 문서와 판정 전 항목(`docs/closure_audit.md`, `results/phase2/closure_audit_labels.csv`), 표본의 판정 단위(`results/phase2/closure_audit_samples_public.txt`: 생성 96편은 결말 약 170자, 인간 84편은 서지와 판정 근거 구절). `closure_audit_labels.csv`에는 범주·사유·대상 유무만 있고 인간 작품의 구절은 서지와 함께 위 파일에만 있다.
- 계측기 동결 자산(모티프 중심 벡터, 인간 지분 백분율, 모티프 정보; `data/atlas/instrument_frozen/`). 이 자산은 동반 연구의 산출이며, 그 연구의 출판 절차가 끝나면 이 저장소에서 빼고 동반 연구 저장소를 가리킨다.

## 공개하지 않는 것
- 인간 작품 원문(`data/atlas/metadata.csv`의 원문 열). 저작권상 재배포하지 않으며, 논문과 공개 산출물에는 지분 백분율·서지·짧은 구절, 그리고 결말 감사의 판정 근거 구절만 싣는다.

## 표집 프로토콜(요약)
셀(시스템×프롬프트)마다 유효 25편에 도달하면 중단한다. 기본 상한 60회(전송 실패 포함), 상한에서 유효가 하나라도 나온 셀만 400회까지 확대한다. 응답이 돌아오지 않은 호출(HTTP 400 사용량 한도, 429 요청 제한)은 로그에 남되 형식 판정과 유효율에서 제외한다. HCX는 시도 1,002회 가운데 224회가 429 전송 실패이고, 완료 호출 778회 = 조기 충족 3셀 88회 + 유효 0인 7셀 321회 + 확대 2셀 369회(SI S9). Opus는 시도 369회 가운데 69회가 400 전송 실패이고 완료 호출 300회가 전부 유효하다.
- 인간 기준선의 구축(본문 §2.1의 네 규칙과 5,166→4,642 필터)과 계측기(25모티프) 적합은 동반 연구의 공개본에 속하며, 이 패키지는 그 산출(인간 지분·모티프 정보)을 입력으로만 쓴다.
