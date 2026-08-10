# 📁 LG Aimers 해커톤 — 문서 인덱스

> **주제:** 야구 투수의 제구 성공 여부 예측 (`control_success`)  
> **주최:** LG AI연구원  
> **형식:** 이진 분류 → 확률값 예측 (0.0 ~ 1.0)

---

## 📂 폴더 구조

```
lg aimers 관련/
│
├── 📁 docs/                          ← 문서 모음 (현재 폴더)
│   ├── README.md                      ← 이 파일 (인덱스)
│   ├── 01_trackman_history_columns.md ← 데이터 컬럼 분석
│   ├── 02_대회_방향성_및_목표.md       ← 대회 목표 & 공략법
│   ├── 03_샘플_테스트_데이터_기록.md   ← 샘플 테스트 데이터 기록
│   ├── 04_대회_참가_및_제출_규칙.md    ← 참가·모델링·제출 규칙
│   ├── 05_평가_제출_서버_및_Mac_개발_가이드.md ← 평가·서버·Mac 개발 가이드
│   ├── 06_공식_RandomForest_베이스라인_기록.md ← 공식 학습·추론 베이스라인
│   └── 환경설정_트러블슈팅.md           ← 환경 설정 가이드
│
├── 📁 submissions/                   ← 제출 ZIP 파일 모음
│   ├── lg_aimers_submission.zip
│   └── my_submission.zip
│
├── 📁 공모전 dataset/                ← 원본 데이터셋
│   └── open/baseline_submit/data/
│       ├── train.csv                  (학습 데이터 + 정답)
│       ├── test.csv                   (예측 대상)
│       ├── trackman_history.csv       (투구 물리 지표 이력)
│       └── sample_submission.csv      (제출 형식 예시)
│
├── 📁 output/                        ← 모델 출력 결과
│   └── submission.csv
│
└── 환경설정_트러블슈팅.md             ← (docs/에도 복사본 있음)
```

---

## 📄 문서 목록

| 번호 | 파일 | 내용 | 상태 |
|------|------|------|------|
| 01 | [01_trackman_history_columns.md](./01_trackman_history_columns.md) | trackman_history.csv 30개 컬럼 분석 | ✅ 완료 |
| 02 | [02_대회_방향성_및_목표.md](./02_대회_방향성_및_목표.md) | 대회 목표, control_success 기준, 공략 로드맵 | ✅ 완료 |
| 03 | [03_샘플_테스트_데이터_기록.md](./03_샘플_테스트_데이터_기록.md) | 대회측 제공 test.csv 5건 전체 기록 + 결측 패턴 분석 | ✅ 완료 |
| 04 | [04_대회_참가_및_제출_규칙.md](./04_대회_참가_및_제출_규칙.md) | 참가, 허용 모델·데이터, 독립 추론 및 제출 규칙 | ✅ 완료 |
| 05 | [05_평가_제출_서버_및_Mac_개발_가이드.md](./05_평가_제출_서버_및_Mac_개발_가이드.md) | Brier Skill Score, 제출 구조, 평가 서버와 M5 Mac 개발 안내 | ✅ 완료 |
| 06 | [06_공식_RandomForest_베이스라인_기록.md](./06_공식_RandomForest_베이스라인_기록.md) | 공식 RandomForest 학습·추론 노트북 및 현재 반영 사항 | ✅ 완료 |
| 07 | [환경설정_트러블슈팅.md](./환경설정_트러블슈팅.md) | 개발 환경 설정 & 에러 해결법 | ✅ 완료 |

> 새 문서를 만들 때 위 표에 추가해 주세요.

---

## 📦 제출 파일 이력

> `submissions/` 폴더에 ZIP 파일로 관리

| 파일명 | 제출일 | 점수 | 비고 |
|--------|--------|------|------|
| `my_submission.zip` | - | - | 초기 baseline |
| `lg_aimers_submission.zip` | - | - | - |

> 새 제출 시: 파일명에 날짜/버전 포함 권장  
> 예: `submission_v2_20250805.zip`

---

## 🚀 빠른 참고

### Target이 뭔가요?
→ **`control_success`** : 투구 제구 성공 확률 (0~1)

### 어떤 데이터를 써야 하나요?
→ `train.csv` (메인) + `trackman_history.csv` (파생 변수 소스)

### 어떤 모델을 써야 하나요?
→ **LightGBM / XGBoost** 로 시작 → 앙상블로 성능 향상

### 가장 중요한 피처는?
→ `asof_pitcher_prev1/3/5_game_success_rate` (최근 폼)  
→ `li` (레버리지 인덱스 - 압박 상황 수치)  
→ `trackman_history`에서 추출한 투수별 물리 지표 통계

---

## ✅ 진행 체크리스트

- [ ] EDA 완료 (train.csv 분포, 결측값, 이상치 확인)
- [ ] trackman_history.csv → 투수별 파생 변수 집계
- [ ] Baseline 모델 구축 (LightGBM)
- [ ] Cross-validation 전략 수립 (시계열 고려)
- [ ] Feature Importance 분석
- [ ] 앙상블 / 하이퍼파라미터 튜닝
- [ ] 최종 제출 파일 `submissions/`에 저장
