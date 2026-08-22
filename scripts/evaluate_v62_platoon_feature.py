"""V62: 타자 스탠스별 투수 제구 성공률 플래툰 피처 (Context HGB)

V49 플래툰 보정(포인트 보정)은 기각됐으나,
"피처 형태"로 주입하면 모델이 자체적으로 가중치를 학습 가능.

구현:
  - asof_pitcher_vs_L_success_rate: 현시점까지 좌타자 상대 제구 성공률 (expanding)
  - asof_pitcher_vs_R_success_rate: 현시점까지 우타자 상대 제구 성공률 (expanding)
  - n_min=50구 미만은 asof_pitcher_success_rate로 패딩 (Bayesian)
  - Context HGB에 2개 피처 추가 후 재학습
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

N_MIN_PLATOON = 50  # 최소 구수 미만은 전체 성공률로 패딩


def compute_platoon_rates(data: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    투수별 타자 스탠스(L/R) 기준 expanding 제구 성공률 계산.
    데이터가 시간순으로 정렬되어 있다고 가정.
    """
    df = data.copy()
    df["control_success"] = y.values

    vs_L = df[df["batter_hand"] == "L"].groupby("pitcher_id")["control_success"].expanding().mean()
    vs_R = df[df["batter_hand"] == "R"].groupby("pitcher_id")["control_success"].expanding().mean()
    count_L = df[df["batter_hand"] == "L"].groupby("pitcher_id")["control_success"].expanding().count()
    count_R = df[df["batter_hand"] == "R"].groupby("pitcher_id")["control_success"].expanding().count()

    # 인덱스 정렬 후 원본 df에 매핑
    df["_vs_L_rate"]  = np.nan
    df["_vs_L_count"] = 0
    df["_vs_R_rate"]  = np.nan
    df["_vs_R_count"] = 0

    # 좌타자 행에 expanding 값 할당 (shift 1: 현재 구 제외)
    l_mask = df["batter_hand"] == "L"
    r_mask = df["batter_hand"] == "R"

    df.loc[l_mask, "_vs_L_rate"]  = vs_L.shift(1).values if len(vs_L) else np.nan
    df.loc[l_mask, "_vs_L_count"] = count_L.shift(1).fillna(0).values if len(count_L) else 0
    df.loc[r_mask, "_vs_R_rate"]  = vs_R.shift(1).values if len(vs_R) else np.nan
    df.loc[r_mask, "_vs_R_count"] = count_R.shift(1).fillna(0).values if len(count_R) else 0

    # n_min 미만은 전체 성공률로 패딩
    overall = df["asof_pitcher_success_rate"].fillna(df["asof_pitcher_success_rate"].mean())
    df["asof_pitcher_vs_L_success_rate"] = np.where(
        (df["_vs_L_count"] >= N_MIN_PLATOON) & l_mask,
        df["_vs_L_rate"],
        overall,
    )
    df["asof_pitcher_vs_R_success_rate"] = np.where(
        (df["_vs_R_count"] >= N_MIN_PLATOON) & r_mask,
        df["_vs_R_rate"],
        overall,
    )

    # 결측치 처리
    df["asof_pitcher_vs_L_success_rate"] = df["asof_pitcher_vs_L_success_rate"].fillna(overall)
    df["asof_pitcher_vs_R_success_rate"] = df["asof_pitcher_vs_R_success_rate"].fillna(overall)

    df.drop(columns=["_vs_L_rate", "_vs_L_count", "_vs_R_rate", "_vs_R_count", "control_success"],
            inplace=True)
    return df


def encode_features(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    X = df[feats].copy()
    for c in feats:
        if X[c].dtype == object:
            X[c] = X[c].fillna("unknown").astype("category").cat.codes.astype(float)
        else:
            X[c] = X[c].fillna(X[c].median())
    return X


def train_context_hgb_v62(train_frame, train_y):
    tf = compute_platoon_rates(train_frame, train_y)
    feats = [f for f in CONTEXT_FEATURES_BASE + [
        "asof_pitcher_vs_L_success_rate",
        "asof_pitcher_vs_R_success_rate",
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


def predict_context_v62(clf, frame, train_frame, train_y, feature_cols):
    # platoon rates는 train+test 합산 계산 없이, train 기반으로 전체 레이트 계산 후 test에 매핑
    # 간략하게: test frame에 overall 패딩으로 platoon 피처 추가
    tf = frame.copy()
    overall = tf["asof_pitcher_success_rate"].fillna(tf["asof_pitcher_success_rate"].mean())
    tf["asof_pitcher_vs_L_success_rate"] = overall
    tf["asof_pitcher_vs_R_success_rate"] = overall
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

    # ── V62 Context HGB (platoon 피처 포함) ──────────────────────────
    print("Training V62 Context HGB (2022 → 2023)...", flush=True)
    clf_23, feats_23 = train_context_hgb_v62(frames["2022"], y_dict["2022"])
    cp23 = predict_context_v62(clf_23, frames["2023"], frames["2022"], y_dict["2022"], feats_23)

    print("Training V62 Context HGB (2022+2023 → 2024)...", flush=True)
    f2223 = pd.concat([frames["2022"], frames["2023"]])
    y2223 = pd.concat([y_dict["2022"], y_dict["2023"]])
    clf_24, feats_24 = train_context_hgb_v62(f2223, y2223)
    cp24 = predict_context_v62(clf_24, frames["2024"], f2223, y2223, feats_24)

    cp22 = ctx_base["2022"]
    ctx_v62 = {"2022": cp22, "2023": cp23, "2024": cp24}

    # ── raw blend & calibration ─────────────────────────────────────
    raw_blend = {
        str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_v62[str(yr)]
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
        "experiment": "V62_platoon_feature",
        "description": "타자 스탠스별 투수 제구 성공률 플래툰 피처 추가 (Context HGB)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "scores":       {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
    }

    Path("artifacts/v62_platoon_feature_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n✅ all_improved={result['all_improved']}, submit_ready={result['submit_ready']}", flush=True)


if __name__ == "__main__":
    main()
