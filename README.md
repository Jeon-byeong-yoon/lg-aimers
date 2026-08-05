# 🏆 LG Aimers 해커톤 — 야구 투구 제구 성공 확률 예측

> LG AI연구원 주최 해커톤 참가 레포지토리  
> **팀원은 이 README를 먼저 읽어주세요** 👇

---

## 📌 대회 개요

| 항목 | 내용 |
|------|------|
| **주최** | LG AI연구원 |
| **태스크** | 야구 투구의 **제구 성공 확률** 예측 (이진 분류 → 0~1 확률 출력) |
| **Target** | `control_success` — 제구 성공 여부 (1: 성공, 0: 실패) |
| **데이터** | KBO 리그 실제 투구 데이터 + Trackman 측정 시스템 |
| **제출 형식** | `row_id`, `control_success` (확률값) CSV → zip 압축 후 리더보드 업로드 |

---

## 📁 디렉토리 구조

```
lg aimers 관련/
│
├── 📁 docs/                              ← ✅ 팀 문서 모음 (여기부터 읽으세요)
│   ├── README.md                          # 문서 인덱스 + 진행 체크리스트
│   ├── 01_trackman_history_columns.md    # 데이터 컬럼 30개 상세 분석
│   ├── 02_대회_방향성_및_목표.md           # 대회 목표·공략 로드맵·주의사항
│   └── 환경설정_트러블슈팅.md             # 환경 설정 에러 해결 기록
│
├── 📁 submissions/                       ← ✅ 제출 ZIP 버전 관리
│   ├── lg_aimers_submission.zip          # 제출본 (버전 관리용)
│   └── my_submission.zip
│   (새 제출 시 → submission_v2_YYYYMMDD.zip 형식으로 추가)
│
├── 📁 공모전 dataset/
│   └── open/
│       ├── 📁 data/                      ← ✅ 원본 데이터 (git 제외, 직접 배치)
│       │   ├── train.csv                  # 학습 데이터 (약 1,475,092행 × 49컬럼)
│       │   ├── test.csv                   # 예측 대상 데이터
│       │   ├── trackman_history.csv       # 2019~ Trackman 투구 물리 로그
│       │   └── sample_submission.csv      # 제출 형식 예시
│       ├── 📁 baseline_submit/           ← ✅ 베이스라인 실행 폴더
│       │   ├── script.py                  # 추론 스크립트 (여기서 실행)
│       │   ├── requirements.txt           # 패키지 목록
│       │   ├── data → (심볼릭 링크 → ../data)
│       │   ├── model/
│       │   │   └── rf.pkl                 # 베이스라인 Random Forest 모델
│       │   └── output/
│       │       └── submission.csv         # 실행 후 생성되는 제출 파일
│       └── data_description.md            # 공식 컬럼 설명서
│
├── 📁 output/                            ← 모델 출력 결과
│   └── submission.csv
│
├── 환경설정_트러블슈팅.md                  # (docs/에 동일 파일 있음)
├── README.md                              # 이 파일
│
└── 📁 강의자료/
    ├── 『LG AI연구원 해커톤 문제 소개』.pdf
    ├── 『지도학습』.pdf
    ├── 『딥러닝 자연어처리 기초와 LLM Agent』.pdf
    ├── 『LLM Application & Evaluation』.pdf
    ├── 『Mathematics for ML』.pdf
    ├── 『Tabular ML: From Classical to Foundation Models』/
    └── 『Optimization & Time-Series Analysis』/
```

> ⚠️ `train.csv`, `trackman_history.csv`는 100MB 이상으로 GitHub에 올라가지 않습니다.  
> 팀원은 아래 **팀원 온보딩** 섹션을 참고해 직접 데이터를 배치하세요.

---

## 📊 데이터 설명

### 피처 구성 (48개 입력 컬럼)

| 그룹 | 컬럼 수 | 주요 내용 |
|------|--------|----------|
| 기본 식별자 · 경기 정보 | 7 | season, game_month, inning, top_bottom 등 |
| 투구 직전 카운트 · 점수 | 8 | balls/strikes/outs_before, score_diff 등 |
| 주자 상황 · 중요도 | 7 | runner_on_1b/2b/3b, base_state, li, win_expectancy |
| 선수 · 팀 정보 | 6 | pitcher_id, batter_id, pitcher/batter_hand, team_id |
| 과거 이력 피처 (`asof_*`) | 20+ | 투수 성공률, 구종 비율, 최근 N경기 성적 등 |

### 핵심 제약사항

- `test.csv` 내부 행 간 통계/rolling/target encoding **금지**
- 현재 투구 **이후** 확정되는 정보 사용 **금지** (Trackman 측정값, 실제 판정 등)
- `trackman_history.csv`는 1:1 join 불가 — 피처 엔지니어링 보조 자료로만 활용

---

## 👋 팀원 온보딩 (레포 처음 받았을 때)

### Step 1. 문서 먼저 읽기

| 순서 | 파일 | 내용 |
|------|------|------|
| 1️⃣ | [docs/README.md](docs/README.md) | 전체 인덱스 + 진행 체크리스트 |
| 2️⃣ | [docs/02_대회_방향성_및_목표.md](docs/02_대회_방향성_및_목표.md) | 대회 목표, Target 변수, 공략법 |
| 3️⃣ | [docs/01_trackman_history_columns.md](docs/01_trackman_history_columns.md) | 데이터 컬럼 상세 설명 |
| 4️⃣ | [docs/환경설정_트러블슈팅.md](docs/환경설정_트러블슈팅.md) | 환경 설정 에러 대처법 |

### Step 2. 대용량 데이터 배치

레포를 받으면 아래 파일들이 없습니다. **LG Aimers 대회 페이지**에서 직접 다운받아 경로에 배치하세요.

```bash
# 아래 경로에 파일을 직접 넣어주세요
공모전 dataset/open/data/
├── train.csv                ← 대회 사이트에서 다운로드
├── test.csv                 ← 대회 사이트에서 다운로드
├── trackman_history.csv     ← 대회 사이트에서 다운로드
└── sample_submission.csv    ← 대회 사이트에서 다운로드
```

### Step 3. 심볼릭 링크 설정 후 베이스라인 실행

```bash
cd "공모전 dataset/open/baseline_submit"
ln -sf "../data" "./data"   # 심볼릭 링크 생성
```

---

## 🚀 빠른 시작 (베이스라인 실행)

### 1. 환경 설정

```bash
# conda 가상환경 생성 (Python 3.11 권장)
conda create -n lgaimers python=3.11 -y
conda activate lgaimers

# 패키지 설치
pip install scikit-learn==1.8.0 joblib==1.5.3 pandas==2.3.3 "numpy<2"
```

> 💡 Anaconda base 환경의 NumPy가 2.x인 경우 충돌 발생 → 반드시 새 환경에서 실행  
> 자세한 내용은 [환경설정_트러블슈팅.md](환경설정_트러블슈팅.md) 참고

### 2. 데이터 연결 (심볼릭 링크)

```bash
cd "공모전 dataset/open/baseline_submit"
ln -s "../data" "./data"
```

### 3. 추론 실행

```bash
conda activate lgaimers
cd "공모전 dataset/open/baseline_submit"
python script.py
```

**실행 성공 시 출력:**
```
Load model...      OK. n_features=47
Load test data...  test=5  submission=5
Build features...  features=47
Inference model... preds=5
Build submission...
✅ Saved: ./output/submission.csv (rows=5)
```

### 4. 제출 파일 생성

```bash
cd baseline_submit
zip -r submission.zip script.py requirements.txt model/ output/
```

---

## 📋 제출 방법

> ⚠️ **팀 구성 전 팀원당 1회 개인 제출 필수**

1. 위 과정으로 `lg_aimers_submission.zip` 생성
2. LG Aimers 리더보드에 zip 파일 업로드
3. 평가 서버에서 실제 `test.csv`로 교체 후 `script.py` 자동 실행 → 채점

**제출 파일 구조:**
```
submission.zip
├── script.py
├── requirements.txt
├── model/
│   └── rf.pkl
└── output/
    └── submission.csv
```

---

## 📈 모델 개선 계획

### Phase 1 — 베이스라인 제출 ✅
- [x] Random Forest 베이스라인 (`rf.pkl`) 실행
- [x] `submission.csv` 생성 완료
- [x] 리더보드 제출용 zip 생성

### Phase 2 — 피처 엔지니어링
- [ ] `trackman_history.csv`에서 투수별 구종 특성 집계
- [ ] cold-start 처리 전략 (smoothing, fallback)
- [ ] `pitcher_hand × batter_hand` 상호작용 피처

### Phase 3 — 모델 실험
- [ ] LightGBM / XGBoost
- [ ] CatBoost (범주형 피처 강점)
- [ ] TabPFN (소규모 실험)
- [ ] 앙상블 (Stacking / Blending)

### Phase 4 — 최적화
- [ ] Optuna 하이퍼파라미터 튜닝
- [ ] GroupKFold CV (시계열 특성 고려)

---

## 📚 강의자료 활용 매핑

| 단계 | 관련 강의자료 |
|------|-------------|
| EDA + Feature Engineering | Tabular ML 01~02 |
| 트리 기반 모델 | Tabular ML 02, 지도학습 |
| 딥러닝 Tabular 모델 | Tabular ML 03~04 |
| TabPFN | Tabular ML 06 |
| 시계열 피처 | 이용재 교수 Time Series (4~6강) |
| 최적화 | 이용재 교수 Opt & DFL (1~3강) |
| LLM 피처 아이디어 | LLM Application & Evaluation |

---

## ⚙️ 기술 스택

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![pandas](https://img.shields.io/badge/pandas-2.3.3-lightblue)
![numpy](https://img.shields.io/badge/numpy-1.26.4-yellow)

---

---

## 📝 문서 기여 가이드

- 분석 결과, 실험 노트 등 새 문서는 **`docs/` 폴더**에 `03_`, `04_` 번호 붙여 저장
- 제출 파일은 **`submissions/` 폴더**에 `submission_v버전_날짜.zip` 형식으로 저장
- 새 문서 추가 시 [docs/README.md](docs/README.md)의 문서 목록 표에 추가

---

*LG Aimers 해커톤 참가 레포지토리 | 전병윤*
