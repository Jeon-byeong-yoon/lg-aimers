"""V64: 이닝 × 이닝 내 누적 투구수 교호작용 피로 신호 피처

현재 inning 단독 사용. 이닝 내 누적 투구수(이닝 내 투수의 누적 투구 부담)와
교호작용을 통해 피로 임계점을 포착.

구현:
  - inning_within_pitch_count: 경기 내 해당 이닝에서의 투수 누적 투구 순서
  - fatigue_signal = inning * inning_within_pitch_count
  - Context HGB에 추가
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

CONTEXT_FEATURES_BASE = [
    "inning", "top_bottom", "outs_before",
    "balls_before", "strikes_before",
    "score_diff_pitcher_team", "li",
    "num_runners_on", "base_state",
]


def add_fatigue_feature(df: pd.DataFrame) -> pd.DataFrame:
    """이닝 내 누적 투구수 및 피로 교호작용 피처 생성."""
    df = df.copy().reset_index(drop=True)

    # 경기 내 (pitcher_id, inning) 단위 누적 투구 순서 (1-indexed, 현재 구 포함)
    group_cols = [c for c in ["pitcher_id", "inning"] if c in df.columns]
    if group_cols:
        df["inning_within_pitch_count"] = df.groupby(group_cols).cumcount() + 1
    else:
        df["inning_within_pitch_count"] = 1

    # 피로 교호작용
    df["fatigue_signal"] = df["inning"] * df["inning_within_pitch_count"]

    return df


def encode_features(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    X = df[feats].copy()
    for c in feats:
        if X[c].dtype == object:
            X[c] = X[c].fillna("unknown").astype("category").cat.codes.astype(float)
        else:
            X[c] = X[c].fillna(X[c].median())
    return X


def train_context_hgb_v64(train_frame, train_y):
    tf = add_fatigue_feature(train_frame)
    feats = [f for f in CONTEXT_FEATURES_BASE + [
        "inning_within_pitch_count", "fatigue_signal"
    ] if f in tf.columns]
    X = encode_features(tf, feats)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        random_state=42,
    )
    clf.fit(X, train_y)
    return clf, feats


def predict_context_v64(clf, frame, feature_cols):
    tf = add_fatigue_feature(frame)
    X = encode_features(tf, feature_cols)
    return clf.predict_proba(X)[:, 1]


def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof        = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic   = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31        = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_base   = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames  = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    y_dict  = {str(yr): y_all.loc[oof[str(yr)]["row_index"]] for yr in years}

    v11_tr   = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95*v11_tr[str(yr)] + 0.05*logistic[str(yr)] for yr in years}

    # ── V64 Context HGB (피로 신호 피처 포함) ────────────────────────
    print("Training V64 Context HGB (2022 → 2023)...", flush=True)
    clf_23, feats_23 = train_context_hgb_v64(frames["2022"], y_dict["2022"])
    cp23 = predict_context_v64(clf_23, frames["2023"], feats_23)

    print("Training V64 Context HGB (2022+2023 → 2024)...", flush=True)
    f2223 = pd.concat([frames["2022"], frames["2023"]])
    y2223 = pd.concat([y_dict["2022"], y_dict["2023"]])
    clf_24, feats_24 = train_context_hgb_v64(f2223, y2223)
    cp24 = predict_context_v64(clf_24, frames["2024"], feats_24)

    cp22 = ctx_base["2022"]
    ctx_v64 = {"2022": cp22, "2023": cp23, "2024": cp24}

    # ── raw blend & calibration ─────────────────────────────────────
    raw_blend = {
        str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_v64[str(yr)]
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
        "experiment": "V64_fatigue_signal",
        "description": "이닝×이닝 내 누적 투구수 교호작용 피로 신호 피처 추가 (Context HGB)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "scores":       {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
    }

    Path("artifacts/v64_fatigue_signal_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n✅ all_improved={result['all_improved']}, submit_ready={result['submit_ready']}", flush=True)


if __name__ == "__main__":
    main()
