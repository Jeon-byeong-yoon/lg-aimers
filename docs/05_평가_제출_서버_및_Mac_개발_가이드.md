# 평가·코드 제출·Mac 개발 가이드

> 대회 평가 산식, `submit.zip` 규격, 평가 서버 환경과 MacBook Pro M5(메모리 32GB) 개발 시 주의사항을 정리합니다.
> 공지 내용이 변경되거나 서로 다르게 표기된 경우 DACON의 최신 공식 공지와 운영진 답변을 우선합니다.

---

## 1. 리더보드 평가 산식

이 대회는 각 투구의 `control_success = 1`일 확률을 예측합니다.

```text
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)
평균 제구율 Brier Score = r × (1 - r)
Score = max(0, 100000 × (1 - Brier Score / (r × (1 - r))))
```

- `p_i`: i번째 샘플의 제구 성공 예측 확률
- `y_i`: i번째 샘플의 실제 정답(0 또는 1)
- `r`: 전체 평가 데이터의 평균 제구 성공률(비공개)
- Brier Score는 낮을수록 좋고, 최종 Score는 높을수록 좋습니다.
- 상수로 전체 평균 확률만 예측하는 기준 모델보다 Brier Score가 좋아야 양의 점수를 받습니다.
- 지나치게 확신한 오답은 제곱 오차가 커지므로 분류 정확도뿐 아니라 확률 보정(calibration)이 중요합니다.

### 리더보드 구성

- Public Score: 전체 테스트 데이터 100%
- Private Score: 대회 종료 시점의 Public Score

즉, 별도의 Public/Private 데이터 분할로 순위가 뒤집히는 방식이 아니라 종료 시점의 점수가 Private Score가 됩니다.

### 로컬 검증 권장 방식

- 교차 검증의 핵심 지표로 `mean_squared_error(y, p)`를 사용합니다.
- 최종 대회 Score를 흉내 낼 때는 각 검증 폴드의 실제 평균 `r`로 위 공식을 계산합니다.
- 확률은 제출 전에 `[0, 1]` 범위인지 확인합니다.
- 모델 선택 시 정확도보다 OOF Brier Score를 우선합니다.
- 필요하면 OOF 예측만으로 calibration을 학습하며, 검증 정답이 calibration 학습에 누출되지 않도록 폴드를 분리합니다.

---

## 2. 평가 및 Phase 3 진출

- LG Aimers 수료 조건: Phase 1 이수 및 Phase 2 Public Score `549.51` 이상
- 기준 점수는 운영진 베이스라인 추론 코드를 운영진 평가 환경에서 실행한 결과입니다.
- 1차 평가: 리더보드 Private Score 100%
- 동점자는 기존 리더보드 순위 산정 방식을 따릅니다.
- 2차 평가: Phase 3 진출 희망 팀의 코드 검증
- Private 리더보드 상위 팀(약 100명)은 코드와 PPT를 필수로 제출합니다.
- 코드 및 PPT 제출과 검증을 모두 통과한 상위 팀을 기준으로 약 100명이 Phase 3에 진출합니다.

---

## 3. 제출 ZIP 필수 구조

평가 서버는 ZIP **내부 구조만** 확인하고 ZIP 파일명은 보지 않습니다.
따라서 팀 규칙에 따라 `submit_v<번호>.zip`처럼 버전 번호를 붙여도 채점에 영향이 없습니다.
아래 구조에서 `submit.zip`은 파일명 예시일 뿐이며, 최상위 구성이 정확히 같아야 합니다.

```text
submit.zip
├── model/
│   └── 모델 가중치 파일
├── script.py
└── requirements.txt
```

- `script.py`: 평가 서버에서 자동 실행되는 추론 코드
- `requirements.txt`: `pip install -r requirements.txt`로 설치 가능한 패키지 목록
- `model/`: 인터넷 없이 추론할 수 있도록 필요한 모델과 가중치를 포함
- `submit/`, `baseline_submit/` 같은 추가 최상위 폴더를 ZIP 안에 넣지 않습니다.
- 로컬의 `data/` 및 `output/`은 제출 ZIP에 포함하지 않습니다.

평가 서버는 압축 해제 후 다음 항목을 추가합니다.

```text
실행 디렉토리/
├── model/
├── script.py
├── requirements.txt
├── data/                  # 평가 데이터, 읽기 전용
└── output/                # 결과 저장 위치
    └── submission.csv
```

`script.py`는 반드시 `output/submission.csv`를 생성해야 합니다.

### 공식 베이스라인에서 확인된 데이터 경로

공식 학습·추론 노트북은 `./data/test.csv`와 `./data/sample_submission.csv`를 사용합니다. 배포본은 형식 확인용 5행이고, 평가 서버가 실제 테스트 데이터 245,789행을 동일한 `./data/test.csv` 경로에 제공합니다. 앞선 안내의 `open/` 표기는 배포 패키지 상위 구조를 가리키는 것으로 보고, 제출용 `script.py`에서는 공식 베이스라인과 동일한 `./data` 상대경로를 사용합니다.

---

## 4. 용량과 시간 제한

- ZIP 파일: 최대 10GB
- 압축 해제 후: 최대 32GB
- 패키지 설치: 최대 10분
- 추론 실행: 최대 10분

모델 로딩, 전처리, 예측, CSV 저장까지 모두 추론 시간에 포함된다고 보고 여유 있게 설계합니다.

---

## 5. 평가 서버 환경

| 항목 | 평가 서버 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| CUDA | 12.8 |
| CPU | 6 vCPU |
| RAM | 28GB |
| 인터넷 | 패키지 설치 외 비활성화 |

### 주요 기본 Python 패키지

```text
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```

버전이 명시된 기본 패키지는 다른 버전을 강제로 재설치할 때 충돌할 수 있으므로, 가능하면 `requirements.txt`에서 제외하고 서버 기본 버전을 사용합니다.

---

## 5-2. 직렬화 호환성 — 필수 게이트

### 반드시 지킬 것

제출 ZIP에 들어갈 `.joblib` 모델을 **새로 학습하거나 다시 저장할 때는 반드시
서버 미러 환경**을 사용한다.

```bash
/opt/anaconda3/envs/lgaimers_server/bin/python scripts/train_....py
```

패키징 전에 항상 로드 검사를 통과시킨다. `scripts/package_v9*.py`는 이 검사를
내부에서 강제하므로, 실패하면 ZIP이 생성되지 않는다.

```bash
/opt/anaconda3/envs/lgaimers_server/bin/python scripts/check_server_pickle_compat.py
```

### 환경 구성

| 환경 | numpy | 용도 |
|---|---|---|
| `lgaimers` | **2.4.6** | 탐색·검증 전용. **제출 아티팩트 저장 금지** |
| `lgaimers_server` | **1.26.4** | 서버와 동일. 제출 아티팩트 학습·저장 전용 |

```bash
conda create -y -n lgaimers_server python=3.11
/opt/anaconda3/envs/lgaimers_server/bin/pip install \
  numpy==1.26.4 pandas==2.0.3 scikit-learn==1.8.0 joblib==1.5.3 scipy==1.15.3
```

### 왜 필요한가 — V93 1차 제출 실패 사례

2026-08-24 V93 1차 제출이 평가 서버에서 다음 오류로 실패해 **일일 제출 기회 1회를
낭비했다.** (7절 기준 `script.py` 실행 후 오류는 일일 횟수에 반영된다.)

```
ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module.
```

numpy 2.x는 `np.random.Generator`를 **클래스 참조**로 pickle하지만 numpy 1.x는
**문자열 이름**만 해석한다. 원인 객체는
`HistGradientBoostingClassifier._feature_subsample_rng`로, fit 전용이고 추론에는
쓰이지도 않지만 pickle에 함께 저장된다.

V41·V92가 무사했던 이유는 기존에 저장된 `v6_ensemble.joblib`,
`v38_feature_models.joblib`을 **재사용만** 했기 때문이다. V93에서 처음으로 모델을
새로 저장했고 그때 numpy 2.4.6 환경을 썼다.

### 일반 교훈

**로컬 추론 성공은 서버 호환성의 증거가 되지 않는다.** 직렬화 객체는 저장 시점의
라이브러리 버전을 각인한다. 새로 저장한 아티팩트가 하나라도 있으면 서버와 같은
버전의 인터프리터로 로드 검사를 통과시킨 뒤에만 제출한다.

---

## 6. MacBook Pro M5·32GB 개발 안내

32GB 메모리는 이 대회의 일반적인 표형 데이터 전처리와 LightGBM, XGBoost, CatBoost, scikit-learn 계열 모델 개발에 충분한 편입니다. 다만 로컬 Mac과 평가 서버는 하드웨어 및 운영체제가 다르므로 로컬 성공만으로 서버 실행을 보장할 수 없습니다.

| 차이 | MacBook Pro M5 | 평가 서버 | 대응 |
|---|---|---|---|
| CPU 아키텍처 | Apple Silicon ARM64 | 일반적으로 Linux x86_64 | 플랫폼 종속 바이너리와 직렬화 호환성 확인 |
| OS | macOS | Ubuntu Linux | 절대경로와 macOS 전용 기능 사용 금지 |
| GPU | Apple GPU/MPS | NVIDIA L4/CUDA | 추론 코드에서 장치를 자동 선택 |
| 메모리 | 32GB 통합 메모리 | CPU RAM 28GB, VRAM 22.4GB | 로컬 사용량을 28GB보다 충분히 낮게 유지 |
| 파일 시스템 | 보통 대소문자 구분이 느슨함 | 대소문자 구분 | 파일명 대소문자를 정확히 일치 |

### GPU 코드 작성 원칙

Mac에서는 CUDA를 사용할 수 없습니다. PyTorch 개발 시 Mac은 `mps`, 서버는 `cuda`, 둘 다 없으면 `cpu`를 선택하도록 작성합니다.

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

단, 제출 직전에는 CUDA 경로가 실제로 작동하는지 Linux/NVIDIA 환경에서 별도로 검증하는 것이 안전합니다. MPS 결과와 CUDA 결과는 부동소수점 연산 차이로 완전히 같지 않을 수 있습니다.

### 패키지 관리 원칙

- 로컬 개발 환경도 Python 3.11 계열로 맞춥니다.
- 서버 기본 패키지 버전과 가능한 한 동일한 버전으로 테스트합니다.
- `torch==2.7.1+cu128`은 Mac에 그대로 설치하는 항목이 아닙니다. 로컬에는 macOS용 PyTorch를 사용하고 제출 ZIP에는 PyTorch 자체를 넣지 않습니다.
- Mac 전용 wheel, `.dylib`, 가상환경 폴더를 ZIP에 포함하지 않습니다.
- `joblib`이나 pickle 모델은 학습·저장 라이브러리 버전과 서버 로딩 버전을 맞춥니다.
- 외부 다운로드가 필요한 Hugging Face 모델 등은 허용 라이선스를 먼저 확인하고, 필요한 모든 설정·토크나이저·가중치를 `model/`에 포함합니다.

### 자원 사용 권장선

- 평가 서버 RAM은 28GB이므로 최대 메모리를 이에 딱 맞추지 말고 여유를 둡니다.
- pandas 복사본을 반복 생성하지 않고 필요한 열만 읽습니다.
- 모델과 데이터를 한꺼번에 여러 벌 메모리에 올리지 않습니다.
- CPU 병렬도는 6코어 이하로 제한해 서버에서 과도한 스레드 경쟁을 피합니다.
- 전체 로컬 추론 시간이 10분에 근접하면 서버에서는 초과할 수 있으므로 충분한 여유를 확보합니다.

---

## 7. 오류 분류

### 설치 오류 — 일일 제출 횟수에 미반영

- ZIP 내부 구조 불일치
- 패키지 설치 실패

### 제출 오류 — 일일 제출 횟수에 반영

- `script.py` 실행 후 발생하는 모든 오류
- 입력 경로 오류
- 모델 로딩 오류
- 시간 또는 메모리 초과
- `output/submission.csv` 미생성 또는 형식 오류

---

## 8. 제출 전 체크리스트

- [ ] ZIP 최상위에 `model/`, `script.py`, `requirements.txt`만 필요한 구조로 배치
- [ ] 압축 해제 후 용량 32GB 미만
- [ ] 상대경로만 사용하고 현재 작업 디렉토리 기준으로 실행
- [ ] 실제 평가 입력 경로를 공식 베이스라인과 대조
- [ ] 인터넷을 끈 상태에서도 모델 로딩과 추론 가능
- [ ] Python 3.11 및 서버 패키지 버전으로 검증
- [ ] **`check_server_pickle_compat.py`를 `lgaimers_server`로 실행해 전 아티팩트 OK 확인**
- [ ] **새로 저장한 `.joblib`이 있으면 `lgaimers_server`에서 학습·저장했는지 확인**
- [ ] CPU RAM 28GB 및 6 vCPU 조건을 고려
- [ ] 전체 추론이 10분보다 충분히 빠름
- [ ] 예측값이 유한한 `[0, 1]` 확률이며 행 수와 순서가 입력과 일치
- [ ] 결과 파일명이 정확히 `output/submission.csv`
- [ ] ZIP 내부에 macOS의 `.DS_Store`, `__MACOSX`, 가상환경 파일이 없음
