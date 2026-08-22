"""V63: 볼카운트 전환 경로 인코딩 피처

현재 모델: balls_before, strikes_before 만 참조 (현재 카운트).
V63 가설: "어떤 경로로 이 카운트에 도달했는가" 정보를 추가 피처로 인코딩.

구현:
  - 타석 내 이전 투구 결과(볼/스트라이크 추가) 추론: 
    * 전환 타입 = (prev_balls, prev_strikes) → (balls_before, strikes_before)
    * 0-0 카운트는 타석 시작이므로 path=(−1,−1) 처리 (NaN)
  - path_type = prev_balls*10 + prev_strikes (혹은 첫 투구 표시자)
  - 이 path 피처를 Context HGB 및 Encoded HGB에 추가

단, 데이터에 이전 투구 볼/스트라이크 직접 컬럼이 없으므로:
  → 동일 게임/타자/타석 내 이전 행의 balls_before, strikes_before를 shift(1)로 파생
  → 첫 투구(0-0)는 prev=(−1,−1) 표시
"""

import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

W_V17  = 0.55
W_FORM = 0.32
W_CTX  = 0.13

# Context HGB 피처 (기존 no_matchup_hte 기준)
CONTEXT_FEATURES_BASE = [
    "inning", "top_bottom", "outs_before",
    "balls_before", "strikes_before",
    "score_diff_pitcher_team", "li",
    "num_runners_on", "base_state",
]


def add_count_path_feature(df: pd.DataFrame) -> pd.DataFrame:
    """타석 내 이전 투구 카운트를 shift로 파생 → 전환 경로 피처."""
    df = df.copy()
    df = df.reset_index(drop=True)

    # 타석 그룹 식별: 볼카운트가 0-0으로 리셋되면 새 타석 시작
    # 단순 접근: 행 단위 shift, 볼이 줄거나 스트라이크가 0으로 리셋되면 첫 투구로 표시
    prev_balls   = df["balls_before"].shift(1).fillna(-1).astype(int)
    prev_strikes = df["strikes_before"].shift(1).fillna(-1).astype(int)

    # 현재 카운트가 0-0이면 첫 투구 (이전 경로 없음)
    is_first_pitch = (df["balls_before"] == 0) & (df["strikes_before"] == 0)
    prev_balls[is_first_pitch]   = -1
    prev_strikes[is_first_pitch] = -1

    # 전환 경로: prev_balls * 10 + prev_strikes (−1은 첫 투구 표시)
    df["count_path"] = prev_balls * 10 + prev_strikes

    return df


def train_context_hgb_v63(train_frame, train_y):
    tf = add_count_path_feature(train_frame)
    feats = [f for f in CONTEXT_FEATURES_BASE + ["count_path"] if f in tf.columns]
    X = tf[feats].copy()
    for c in feats:
        if X[c].dtype == object:
            X[c] = X[c].fillna("unknown")
            X[c] = X[c].astype("category").cat.codes.astype(float)
        else:
            X[c] = X[c].fillna(X[c].median())

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        random_state=42,
    )
    clf.fit(X, train_y)
    return clf, feats



def predict_context_hgb(clf, frame, feature_cols):
    tf = add_count_path_feature(frame)
    X = tf[feature_cols].copy()
    for c in feature_cols:
        if X[c].dtype == object:
            X[c] = X[c].fillna("unknown")
            X[c] = X[c].astype("category").cat.codes.astype(float)
        else:
            X[c] = X[c].fillna(X[c].median())
    return clf.predict_proba(X)[:, 1]



def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof       = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic  = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames  = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    y_dict  = {str(yr): y_all.loc[oof[str(yr)]["row_index"]] for yr in years}

    v11_tr   = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95*v11_tr[str(yr)] + 0.05*logistic[str(yr)] for yr in years}

    # ── V63 Context HGB (count_path 피처 포함) 재학습 ──────────────────
    print("Training V63 Context HGB (2022 → 2023)...", flush=True)
    clf_23, feats_23 = train_context_hgb_v63(frames["2022"], y_dict["2022"])
    cp23 = predict_context_hgb(clf_23, frames["2023"], feats_23)

    print("Training V63 Context HGB (2022+2023 → 2024)...", flush=True)
    f2223 = pd.concat([frames["2022"], frames["2023"]])
    y2223 = pd.concat([y_dict["2022"], y_dict["2023"]])
    clf_24, feats_24 = train_context_hgb_v63(f2223, y2223)
    cp24 = predict_context_hgb(clf_24, frames["2024"], feats_24)

    # 2022 Context: 기존 V31 no_matchup_hte 사용 (자체 OOF 없음)
    v31    = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_base = v31["no_matchup_hte"]["context"]
    # ctx_base가 연도별 dict인지 배열인지 확인 후 처리
    if isinstance(ctx_base, dict):
        cp22 = ctx_base["2022"]
    else:
        cp22 = ctx_base

    ctx_v63 = {"2022": cp22, "2023": cp23, "2024": cp24}


    # ── raw blend & calibration ─────────────────────────────────────
    raw_blend = {
        str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_v63[str(yr)]
        for yr in years
    }

    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]])

    res22    = targets["2022"] - raw_blend["2022"]
    res_2223 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])

    p22 = np.clip(raw_blend["2022"], 0, 1)

    cnt23  = segment_correction(f23_tr, res22,    frames["2023"], ["balls_before","strikes_before"], 500)
    pcnt23 = segment_correction(f23_tr, res22,    frames["2023"], ["pitcher_id","balls_before","strikes_before"], 300)
    p23    = np.clip(raw_blend["2023"] + res22.mean() + 0.75*cnt23 + 0.25*pcnt23, 0, 1)

    cnt24  = segment_correction(f24_tr, res_2223, frames["2024"], ["balls_before","strikes_before"], 500)
    pcnt24 = segment_correction(f24_tr, res_2223, frames["2024"], ["pitcher_id","balls_before","strikes_before"], 300)
    p24    = np.clip(raw_blend["2024"] + res_2223.mean() + 0.75*cnt24 + 0.25*pcnt24, 0, 1)

    s22 = float(brier_score_loss(targets["2022"], p22))
    s23 = float(brier_score_loss(targets["2023"], p23))
    s24 = float(brier_score_loss(targets["2024"], p24))

    g22 = V41_BASELINE[2022] - s22
    g23 = V41_BASELINE[2023] - s23
    g24 = V41_BASELINE[2024] - s24

    result = {
        "experiment": "V63_count_path_encoding",
        "description": "볼카운트 전환 경로 인코딩 피처 추가 (Context HGB 재학습)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "scores":       {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
    }

    Path("artifacts/v63_count_path_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n✅ all_improved={result['all_improved']}, submit_ready={result['submit_ready']}", flush=True)


if __name__ == "__main__":
    main()
