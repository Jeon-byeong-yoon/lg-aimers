# 공식 RandomForest 베이스라인 기록

> DACON 공지에서 제공된 학습·추론 노트북의 구조와 현재 프로젝트 반영 사항을 기록합니다.

## 공식 노트북

- `[Baseline_Train]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습).ipynb`
- `[Baseline_Inference]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (추론).ipynb`

두 파일은 프로젝트 루트에 보관되어 있습니다. 노트북의 Python 커널 표기는 `Python 3`이며, 실제 평가 서버와 로컬 `lgaimers` 환경은 Python 3.11.15를 사용합니다.

## 학습 베이스라인 요약

- 문제: 각 투구의 `control_success = 1` 확률 예측
- 입력: `test.csv`를 기준으로 선택한 47개 피처
- 학습 데이터: 2019~2024 시즌 `train.csv`
- 평가 데이터: 2025 시즌
- `trackman_history.csv`: 공식 베이스라인에서는 사용하지 않음
- 평가 지표: Brier Skill Score
- 저장 모델: `./model/rf.pkl`

### 전처리

- 범주형 3개: `top_bottom`, `game_type`, `base_state`
- 범주형 처리: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
- 나머지 수치형 44개: 중앙값 결측 대치
- 전처리와 모델을 하나의 scikit-learn `Pipeline`으로 저장

### RandomForest 설정

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=200,
    n_jobs=-1,
    random_state=42,
)
```

### 검증 및 최종 학습

1. 2019~2023 시즌으로 학습합니다.
2. 2024 시즌을 홀드아웃 검증 데이터로 사용합니다.
3. 검증 예측으로 Brier Skill Score를 계산합니다.
4. 성능 확인 후 2019~2024 전체 데이터로 다시 학습합니다.
5. 파이프라인을 `joblib.dump(..., compress=3)`으로 `model/rf.pkl`에 저장합니다.

검증 산식은 다음과 같습니다.

```text
r = mean(y_val)
Brier = mean((prediction - y_val)^2)
기준 Brier = r × (1 - r)
Score = max(0, 100000 × (1 - Brier / 기준 Brier))
```

## 추론 베이스라인 요약

- 모델 입력: `./data/test.csv`
- 제출 양식: `./data/sample_submission.csv`
- 모델: `./model/rf.pkl`
- 결과: `./output/submission.csv`
- 배포된 테스트 파일: 형식 확인용 5행
- 평가 서버의 실제 테스트 파일: 245,789행
- 서버가 실제 데이터를 동일한 `./data/test.csv` 경로에 제공합니다.

공식 제출 구조는 다음과 같습니다.

```text
submit.zip
├── model/
│   └── rf.pkl
├── script.py
└── requirements.txt
```

## 현재 프로젝트에 반영한 차이

현재 `baseline_submit/script.py`는 공식 추론 흐름을 유지하면서 다음 안전 검사를 추가한 버전입니다.

- test와 sample submission의 ID 집합 일치 확인
- ID 중복 및 예측 개수 불일치 확인
- NaN, 무한대, `[0, 1]` 범위 밖 확률 차단
- 예측이 누락된 행에 placeholder를 남기지 않고 오류 발생

이 검사는 테스트 행 사이의 통계나 분포를 피처로 사용하지 않습니다. 제출 파일의 무결성만 확인하므로 각 테스트 행의 독립 예측 원칙을 침해하지 않습니다.

## 다음 성능 개선 방향

공식 베이스라인은 `trackman_history.csv`의 2019~2024 과거 로그 약 179만 행을 사용하지 않습니다. 이를 활용할 때는 공식 데이터만 사용하고, 평가 행 간 집계 없이 과거 로그에서 미리 계산한 피처만 각 행에 적용해야 합니다. 또한 Brier Skill Score에서는 순위뿐 아니라 확률 보정이 중요하므로 OOF Brier Score를 기준으로 개선 여부를 판단합니다.

