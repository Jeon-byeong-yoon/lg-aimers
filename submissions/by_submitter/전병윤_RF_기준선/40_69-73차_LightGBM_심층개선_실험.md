# V69–V73 LightGBM 심층 개선 및 전방위 최적화 실험 계획

## 배경 및 문제의식

- **V65 실험의 핵심 발견**: LightGBM을 Base Layer에 추가했을 때 **2022(+), 2023(+1.05e-4), 2024(+) 전 시즌 동시 개선**을 달성한 유일한 이종 모델이었음.
- **V65의 한계와 병목 원인**:
  1. **2022년 순수 OOF의 부재**: 2022 시즌은 LightGBM OOF가 없어 sklearn의 `trackman_hgb`로 임시 대체하여 Calibration 연쇄 왜곡 발생.
  2. **정규화 부재**: 기본 하이퍼파라미터(`max_depth=5, lr=0.05`)만 사용하여 leaf-wise의 세밀한 일반화 정규화 미흡.
  3. **피처 활용 범위 제한**: Base Trackman 외 Form Layer(32%)에는 미적용.

---

## 5단계 심층 개선 로드맵

### 1️⃣ V69: 2022년 5-Fold OOF 구축 및 LightGBM 정규화 튜닝
- **목표**: 2022년 5-Fold Stratified K-Fold로 완전한 순수 OOF를 추출하여 전 시즌 데이터 정합성 완성.
- **탐색 공간**:
  - `num_leaves` ∈ {15, 31}
  - `learning_rate` ∈ {0.03, 0.05}
  - `min_child_samples` ∈ {100, 200, 300}
  - `reg_lambda` ∈ {10.0, 20.0, 50.0}
  - `w_lgbm` ∈ [0.01 ~ 0.15]
- **통과 기준**: 세 시즌 동시 개선 + 2024 gain > 5e-6

### 2️⃣ V70: Base `trackman_hgb`(23.4%) 1:1 전면 교체 검증
- **목표**: 4-Tree 앙상블에서 sklearn `trackman_hgb` 자리를 `trackman_lgbm`으로 1:1 전면 교체하여 leaf-wise 분할의 이점을 온전히 활용.
- **통과 기준**: 세 시즌 동시 개선 + 2024 gain > 5e-6

### 3️⃣ V71: Form Layer 전용 LightGBM 도입 (Form Diversity 앙상블)
- **목표**: V41 최종 가중치의 32%를 차지하는 `Form HGB`에 `Form LightGBM`을 이종 앙상블하여 Form 신호의 분산 대폭 축소.
- **통과 기준**: 세 시즌 동시 개선 + 2024 gain > 5e-6

### 4️⃣ V72: LightGBM Extremely Randomized Trees (`extra_trees=True`) 모드
- **목표**: LightGBM의 무작위 분할 모드를 활성화하여 트랙맨 물리 피처의 노이즈 흡수 및 일반화 강화.
- **통과 기준**: 세 시즌 동시 개선 + 2024 gain > 5e-6

### 5️⃣ V73: LightGBM 직접 Brier Loss(MSE) Regression Objective 검증
- **목표**: `objective='regression'`을 사용하여 Brier score의 제곱 오차 손실을 직접 최적화.
- **통과 기준**: 세 시즌 동시 개선 + 2024 gain > 5e-6

---

## 진행 현황

| 실험 | 핵심 내용 | 2022 raw | 2023 calib | 2024 calib | 최종 판정 |
|---|---|---|---|---|:---:|
| **V69** | 2022 완전 5-Fold OOF + 정규화 LightGBM | **+3.61e-6** | **+2.85e-5** | -1.15e-7 | ❌ 기각 (2022/2023 대폭 개선, 2024 미세 손실) |
| **V70** | Base Trackman HGB 1:1 전면 교체 | **+1.58e-5** | **+1.27e-4** | -8.75e-7 | ❌ 기각 (2024년 일반화 손실) |
| **V71** | Form Layer 전용 LightGBM 앙상블 | **+1.16e-5** | **+1.00e-4** | **+6.76e-8** | ⚠️ 세 시즌 동시 개선 (2024 gain 5e-6 미달) |
| **V72** | Extremely Randomized LightGBM 모드 | -8.09e-8 | **+2.08e-5** | -6.05e-8 | ❌ 기각 (2022/2024 미세 손실) |
| **V73** | Brier Loss 직접 최적화 (MSE Objective) | **+2.51e-6** | **+2.01e-5** | -4.30e-8 | ❌ 기각 (2024 미세 손실) |

> **현재 확정 기준선**: `submit_v41.zip` (Public Score: **900.7385360187**)

