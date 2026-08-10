# script.py
import os
import math

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def build_features(df):
    """모델 입력 추출 — 학습 때와 동일하게 row_id만 빼고 전부 사용.

    범주형 인코딩(top_bottom, game_type, base_state)과 결측 대치는
    모델 파일 안의 파이프라인이 함께 수행하므로 여기서는 컬럼만 고른다.
    """
    return df.drop(columns=[ID_COL])


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    ID 누락·중복 또는 비정상 확률이 있으면 잘못된 파일을 저장하지 않고 중단한다.
    """
    if len(ids) != len(preds):
        raise ValueError(f"ID 수({len(ids)})와 예측 수({len(preds)})가 다름")
    if len(ids) != len(set(ids)):
        raise ValueError("test 데이터의 row_id에 중복이 있음")
    if sub[ID_COL].duplicated().any():
        raise ValueError("sample_submission의 row_id에 중복이 있음")

    test_ids = set(ids)
    submission_ids = set(sub[ID_COL])
    if test_ids != submission_ids:
        missing = len(submission_ids - test_ids)
        extra = len(test_ids - submission_ids)
        raise ValueError(
            f"test와 sample_submission의 row_id 불일치: "
            f"예측 누락={missing}, 제출 양식에 없는 ID={extra}"
        )

    if any(not math.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0 for p in preds):
        raise ValueError("예측값에 NaN, 무한대 또는 [0, 1] 범위 밖의 값이 있음")

    pred_map = dict(zip(ids, preds))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map)
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
