"""V61: 경기 내 단기 볼넷 비율 이동 평균 피처 (Form HGB 추가)

현재 Form 피처는 과거 경기 단위 통계 (prev1/3/5_game_success_rate).
경기 내 최근 N구의 볼넷 비율 이동 평균을 추가해 투수의 실시간 컨디션 신호 포착.

구현:
  - 경기 내 투구 인덱스(game_pitch_index) 파생
  - 최근 20/30/50구 ball_rate 이동 평균 (경기 단위 그룹화, 초반 패딩)
  - Form HGB 재학습 (lr=0.03, iter=500, 기존 6개 + 3개 신규 피처)
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

# V41 기준선
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

W_V17  = 0.55
W_FORM = 0.32
W_CTX  = 0.13

FORM_FEATURES_BASE = [
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "recent_form_trend",
    "pitcher_era_change",
]


def add_ball_rate_ma(df: pd.DataFrame, windows=(20, 30, 50)) -> pd.DataFrame:
    """경기 내 투구 인덱스 및 볼넷 비율 이동 평균 피처 생성."""
    df = df.copy()

    # ball_rate: pitch_type 관련 컬럼이 없으면 볼넷 여부(balls_after > balls_before)로 추정
    if "ball_rate" in df.columns:
        is_ball = df["ball_rate"].fillna(0)   # 이미 있으면 사용
    else:
        # balls_before → balls_after 변화 여부로 볼 여부 근사
        balls_before = df.get("balls_before", pd.Series(0, index=df.index))
        balls_after  = df.get("balls_after",  pd.Series(0, index=df.index))
        is_ball = (balls_after > balls_before).astype(float)

    # 경기 내 투구 순서 인덱스 (game_id, pitcher_id 기준)
    group_cols = []
    for c in ["game_id", "pitcher_id", "inning"]:
        if c in df.columns:
            group_cols.append(c)

    if group_cols:
        # 경기 내 순서 = 그룹 내 행 순서 (데이터가 투구 순서대로 정렬 가정)
        df["_ball"] = is_ball
        for w in windows:
            col = f"ball_rate_ma{w}"
            df[col] = (
                df.groupby(group_cols[:1])["_ball"]   # game_id 단위
                  .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
        df.drop(columns=["_ball"], inplace=True)
    else:
        for w in windows:
            df[f"ball_rate_ma{w}"] = is_ball.expanding().mean()

    return df


def train_form_hgb_v61(train_frame, train_y, windows=(20, 30, 50)):
    """신규 볼넷 MA 피처 포함 Form HGB 학습."""
    tf = add_ball_rate_ma(train_frame, windows)
    new_features = [f"ball_rate_ma{w}" for w in windows]
    all_features = FORM_FEATURES_BASE + new_features
    present = [f for f in all_features if f in tf.columns]

    X = tf[present].copy()
    # 결측치 처리
    for c in present:
        X[c] = X[c].fillna(X[c].median())

    clf = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=500,
        l2_regularization=20.0,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        random_state=42,
    )
    clf.fit(X, train_y)
    return clf, present


def predict_form_hgb(clf, frame, feature_cols, windows=(20, 30, 50)):
    tf = add_ball_rate_ma(frame, windows)
    X = tf[feature_cols].copy()
    for c in feature_cols:
        X[c] = X[c].fillna(X[c].median())
    return clf.predict_proba(X)[:, 1]


def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof        = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic   = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v31        = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds  = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames  = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    y_dict  = {str(yr): y_all.loc[oof[str(yr)]["row_index"]] for yr in years}

    v11_tr   = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95*v11_tr[str(yr)] + 0.05*logistic[str(yr)] for yr in years}

    # ── V61 Form HGB (볼넷 MA 피처 포함) ──────────────────────────────
    print("Training V61 Form HGB (2022 train → 2023 predict)...", flush=True)
    clf_23, feats_23 = train_form_hgb_v61(frames["2022"], y_dict["2022"])
    fp23 = predict_form_hgb(clf_23, frames["2023"], feats_23)

    print("Training V61 Form HGB (2022+2023 train → 2024 predict)...", flush=True)
    f2223 = pd.concat([frames["2022"], frames["2023"]])
    y2223 = pd.concat([y_dict["2022"], y_dict["2023"]])
    clf_24, feats_24 = train_form_hgb_v61(f2223, y2223)
    fp24 = predict_form_hgb(clf_24, frames["2024"], feats_24)

    # 2022 Form: 자체 OOF 없으므로 기존 V38 gentle_500 사용
    form_v38 = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    fp22 = form_v38["2022"]

    form_preds_v61 = {"2022": fp22, "2023": fp23, "2024": fp24}

    # ── raw blend & calibration ────────────────────────────────────────
    raw_blend = {
        str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds_v61[str(yr)] + W_CTX*ctx_preds[str(yr)]
        for yr in years
    }

    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]])

    res22    = targets["2022"] - raw_blend["2022"]
    res_2223 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])

    p22 = np.clip(raw_blend["2022"], 0, 1)

    cnt23  = segment_correction(f23_tr, res22, frames["2023"], ["balls_before","strikes_before"], 500)
    pcnt23 = segment_correction(f23_tr, res22, frames["2023"], ["pitcher_id","balls_before","strikes_before"], 300)
    p23 = np.clip(raw_blend["2023"] + res22.mean() + 0.75*cnt23 + 0.25*pcnt23, 0, 1)

    cnt24  = segment_correction(f24_tr, res_2223, frames["2024"], ["balls_before","strikes_before"], 500)
    pcnt24 = segment_correction(f24_tr, res_2223, frames["2024"], ["pitcher_id","balls_before","strikes_before"], 300)
    p24 = np.clip(raw_blend["2024"] + res_2223.mean() + 0.75*cnt24 + 0.25*pcnt24, 0, 1)

    s22 = float(brier_score_loss(targets["2022"], p22))
    s23 = float(brier_score_loss(targets["2023"], p23))
    s24 = float(brier_score_loss(targets["2024"], p24))

    g22 = V41_BASELINE[2022] - s22
    g23 = V41_BASELINE[2023] - s23
    g24 = V41_BASELINE[2024] - s24

    submit_ready = (g22 > 0 and g23 > 0 and g24 > 5e-6)

    result = {
        "experiment": "V61_ball_rate_ma_form_feature",
        "description": "경기 내 최근 N구 볼넷 비율 MA 피처 추가 (Form HGB 재학습)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "scores":       {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": submit_ready,
        "features_used_2023": feats_23,
        "features_used_2024": feats_24,
    }

    Path("artifacts/v61_ball_rate_ma_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n✅ all_improved={result['all_improved']}, submit_ready={submit_ready}", flush=True)


if __name__ == "__main__":
    main()
