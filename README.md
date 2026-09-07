# 🏆 LG Aimers 9기 해커톤 — 야구 투구 제구 성공 확률 예측

> LG AI연구원 주최 온라인 해커톤(Phase 2) 참가 레포지토리  
> **팀원은 이 README를 먼저 읽어주세요** 👇

---

## 📌 대회 개요

| 항목 | 내용 |
|------|------|
| **주최** | LG AI연구원 (DACON 운영, 대회 ID 236743) |
| **태스크** | 야구 투구의 **제구 성공 확률** 예측 (이진 분류 → 0~1 확률 출력) |
| **Target** | `control_success` — 제구 성공 여부 (1: 성공, 0: 실패) |
| **데이터** | KBO 리그 실제 투구 데이터 + Trackman 측정 시스템 |
| **제출 형식** | `row_id`, `control_success` (확률값) CSV → zip 압축 후 리더보드 업로드 |
| **평가** | Public = Private = 테스트 100% (홀드아웃 갭 없음), 일일 제출 5회 |

---

## 🥇 최종 결과 (전병윤 트랙)

베이스라인 Random Forest **Public 900.7385**에서 출발해 **약 210차례의 실험(V1~V208)**을 거쳐
최종 **Public 1083.2462**에 도달했습니다. 베이스라인 대비 **+182.5점**, 로컬 시간순 검증
기준선(V114)이 잡힌 이후로만 **누적 +80.67점**입니다.

### 채택된 상위 5개 제출 (Public Score)

| 순위 | 버전 | Public Score | 무엇이 바뀌었나 | 직전 대비 |
|:---:|:---:|:---|:---|:---:|
| 🥇 | **V199** | **1083.2461959655** | CatBoost 성분을 4개 시드 평균으로 (복권 축 완성) | +0.41 |
| 2 | V196 | 1082.8399355688 | factorization 9개·Context 6개 시드 평균 | +5.86 |
| 3 | V193 | 1076.9782849555 | 임베딩 신경망 9개·Form 6개 시드 평균 | +5.52 |
| 4 | V189 | 1071.4549348488 | 임베딩 신경망 3개 시드 평균 (시드 복권 축 발견) | +3.59 |
| 5 | V187 | 1067.8617513573 | V6에서 버렸던 인코딩 그룹을 신경망에 재투입 | — |

> **V199가 최종 확정본입니다.** 이후 V204~V208(CatBoost 반복수·reliability·구조 다양성·
> 시드 확장)은 전부 기각됐습니다 — 남은 축들이 리더보드 전이 잡음(±1점)을 넘는 이득을
> 내지 못함을 **측정으로** 확인했기 때문입니다. 자세한 판정 근거는
> [`submissions/by_submitter/전병윤_RF_기준선/`](submissions/by_submitter/전병윤_RF_기준선/)의
> 90번대 기록 문서에 있습니다.

### 최종 모델 아키텍처 — 6성분 가중 블렌드 + 세그먼트 보정층

최종 예측은 **서로 다른 6개 모델의 가중 평균**에 **세그먼트 보정층**을 얹은 구조입니다.

| 성분 | 가중치 | 모델 | 비고 |
|------|:---:|------|------|
| **Form** | 0.32 | HistGradientBoosting | 투수 안정형 이력 피처(`asof_*`) 중심, 6시드 평균 |
| **CatBoost** | 0.27 | CatBoost | ordered target statistics로 범주형 처리, 4시드 평균 |
| **임베딩 신경망** | 0.20 | Entity Embedding NN | 범주형을 임베딩, 9시드 평균 |
| **Context** | 0.14 | HistGradientBoosting | 경기 상황·Trackman 맥락 피처, 6시드 평균 |
| **Factorization** | 0.07 | Interaction/Factorization NN | 피처 간 상호작용 학습, 9시드 평균 |
| **V17 (로지스틱+릿지)** | 0.00 | Logistic + Residual Ridge | 초기 성분, 현재 가중치 0으로 은퇴 |

그 위에 **세그먼트 보정층**이 잔차를 네 축으로 나눠 교정합니다: 볼카운트, 투수×볼카운트,
투수 경험 구간, **투수 좌우스플릿 × 2스트라이크**. 마지막으로 시즌 내 드리프트 보정을 더합니다.

### 가장 효과가 컸던 방법 두 가지

1. **투수 좌우스플릿 × 2스트라이크 세그먼트 보정** (V169~V175, **누적 +14.65**)  
   투수의 좌/우타자 상대 성적을 2스트라이크 상황과 교차한 세그먼트에서 잔차를 교정한
   것이 단일 아이디어로는 최대 이득이었습니다. "이미 닫혔다"고 두 번 선언됐던 보정층에서
   상호작용 세그먼트를 전수 조사해 되살린 발견입니다.

2. **시드 복권 평균화** (V189~V199, **누적 +15.38**)  
   확률적 성분(신경망·CatBoost·Form·Context·factorization)은 시드마다 다른 뽑기를
   내놓는데, 이를 여러 시드로 평균하면 배포 분산이 줄어 리더보드가 오릅니다. 성분별
   2024 폴드의 "복권 범위"가 평균화의 가치를 예측한다는 규칙을 세우고 다섯 성분을
   전부 평균화했습니다.

이 두 축이 후반부 상승의 대부분을 설명하며, **로컬 검증의 "최소 시즌 규칙"이 16번 중
14번 리더보드 방향을 맞혔습니다.**

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
├── 📁 submissions/                       ← 제출 산출물 보관
│   └── 📁 by_submitter/                  ← ✅ 실제 제출 작업의 단일 기준 위치
│       ├── README.md                     # 담당 모델·점수·제출 시각 기록
│       ├── 전병윤_RF_기준선/
│       │   ├── submit.zip                # 전병윤이 그대로 업로드
│       │   └── 설명.md
│       ├── 이준형_안정형/
│       │   ├── submit.zip                # 이준형이 그대로 업로드
│       │   └── 설명.md
│       └── 오현기_보정형/
│           ├── submit.zip                # 오현기가 그대로 업로드
│           └── 설명.md
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

앞으로 모든 제출 준비와 관리는 `submissions/by_submitter/`에서 수행합니다.
루트나 `submissions/` 바로 아래의 과거 ZIP은 새 제출에 사용하지 않습니다.

1. 새 모델을 학습하고 로컬 시간순 검증 결과를 기록합니다.
2. 평가 서버 방식으로 스모크 테스트와 ZIP 무결성 검사를 수행합니다.
3. 담당자 폴더의 기존 `submit.zip`을 검증된 새 파일로 교체합니다.
4. 담당자는 자기 폴더 안의 `submit.zip`을 이름 변경이나 재압축 없이 그대로 업로드합니다.
5. 제출 직후 `submissions/by_submitter/README.md`에 Public Score, 제출 시각, 정상 실행 여부를 기록합니다.

현재 담당은 전병윤=RF 기준선, 이준형=안정형, 오현기=보정형입니다. 모델 배정을
바꾸면 ZIP 파일뿐 아니라 각 폴더의 `설명.md`와 배정표도 함께 갱신합니다.

**각 `submit.zip`의 내부 구조:**
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

## 📈 실험 여정 요약 (V1 → V208)

| 단계 | 대표 버전 | 방법 | 도달 점수 |
|------|:---:|------|:---:|
| 베이스라인 | V0 | Random Forest (`rf.pkl`) | 900.7385 |
| 피처 복원·이력 | V92~V114 | 시즌 내 이력·계층 인코딩·엔티티 임베딩 | 1002.5720 |
| 성분 확장 | V117~V156 | CatBoost·factorization·Context 성분 추가 | ~1052 |
| **보정층 상호작용** | **V169~V175** | **투수 좌우스플릿 × 2스트라이크 세그먼트** | 1060.5434 |
| **시드 복권 평균화** | **V189~V199** | **다섯 성분 전부 시드 평균화** | **1083.2462** |
| 축 폐쇄 (기각) | V200~V208 | 상수·구조 재검토 → 전부 현행 유지로 확정 | 1083.2462 |

> 로컬 시간순 검증(2022·2023·2024 폴드)과 리더보드를 짝지어, 효과의 **부호와 유의성**을
> 오차막대로 측정하는 방법론을 후반부에 확립했습니다. 각 실험의 사전 등록 가설과 판정
> 근거는 `submissions/by_submitter/전병윤_RF_기준선/`의 번호순 기록 문서에 남아 있습니다.

### 재현 방법

최종 제출본(V199)은 아래로 정확히 재생성됩니다. 모델 아티팩트는 반드시
`lgaimers_server` 환경(numpy 1.26.4 — 평가 서버와 동일)에서 저장해야 합니다.

```bash
# 1. 학습 아티팩트 빌드 (성분 모델 + 보정층)
/opt/anaconda3/envs/lgaimers_server/bin/python scripts/build_v199_artifacts.py
# 2. 서버 호환성 게이트 통과 후 제출 ZIP 패키징
/opt/anaconda3/envs/lgaimers_server/bin/python scripts/package_v199_submission.py
```

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
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-brightgreen)
![pandas](https://img.shields.io/badge/pandas-2.0.3-lightblue)
![numpy](https://img.shields.io/badge/numpy-1.26.4-yellow)

> ⚠️ **제출 아티팩트는 반드시 `lgaimers_server` 환경(numpy 1.26.4)에서 저장**합니다.
> numpy 2.x로 저장한 모델은 평가 서버(numpy 1.26.4)에서 로드가 깨져 제출이 실패합니다.
> 패키징 전 `scripts/check_server_pickle_compat.py`로 로드 게이트를 통과시켜야 합니다.

---

---

## 📝 문서 기여 가이드

- 분석 결과, 실험 노트 등 새 문서는 **`docs/` 폴더**에 `03_`, `04_` 번호 붙여 저장
- 제출 파일은 **`submissions/` 폴더**에 `submission_v버전_날짜.zip` 형식으로 저장
- 새 문서 추가 시 [docs/README.md](docs/README.md)의 문서 목록 표에 추가

---

*LG Aimers 9기 해커톤 참가 레포지토리 | 전병윤 · 이준형 · 오현기*
*최종 Public 1083.2462 (V199) · 베이스라인 대비 +182.5점*
