"""V69: 2022년 5-Fold 완전 OOF 구축 및 LightGBM 정규화 튜닝

V65의 치명적 결함(2022 OOF 부재로 인한 임시 대체)을 해결:
  1. 2022년 데이터에 5-Fold StratifiedKFold를 적용하여 LightGBM 순수 OOF 예측값 생성.
  2. 2023년 예측: 2022 데이터 전체로 학습한 LightGBM.
  3. 2024년 예측: 2022+2023 데이터 전체로 학습한 LightGBM.
  4. 하이퍼파라미터 (num_leaves, min_child_samples, reg_lambda, learning_rate) 및
     가중치(w_lgbm 0.01~0.12) 전수 격자 평가.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

W_V17 = 0.55
W_FORM = 0.32
W_CTX = 0.13

W_ET = 0.2948
W_THGB = 0.2344
W_EHGB = 0.1162
W_HHGB = 0.3545

TRACKMAN_FEATURES = [
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
]


def get_trackman_X(frame):
    feats = [f for f in TRACKMAN_FEATURES if f in frame.columns]
    X = frame[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(X[c].median())
    return X


def build_lgbm_oof_and_preds(f22, y22, f23, y23, f24, params):
    """2022년 5-Fold OOF 및 2023/2024 Expanding 예측 생성"""
    X22 = get_trackman_X(f22)
    X23 = get_trackman_X(f23)
    X24 = get_trackman_X(f24)
    
    # 1. 2022 5-Fold OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgbm_22_oof = np.zeros(len(y22), dtype=float)
    
    for tr_idx, val_idx in skf.split(X22, y22):
        X_tr, y_tr = X22.iloc[tr_idx], y22.iloc[tr_idx]
        X_val = X22.iloc[val_idx]
        
        clf = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        lgbm_22_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    # 2. 2023 예측 (2022 전체 학습)
    clf_23 = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    clf_23.fit(X22, y22)
    lgbm_23_pred = clf_23.predict_proba(X23)[:, 1]
    
    # 3. 2024 예측 (2022+2023 전체 학습)
    f2223 = pd.concat([f22, f23]).reset_index(drop=True)
    y2223 = pd.concat([y22, y23]).reset_index(drop=True)
    X2223 = get_trackman_X(f2223)
    
    clf_24 = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    clf_24.fit(X2223, y2223)
    lgbm_24_pred = clf_24.predict_proba(X24)[:, 1]
    
    return {
        "2022": lgbm_22_oof,
        "2023": lgbm_23_pred,
        "2024": lgbm_24_pred,
    }


def evaluate_lgbm_blend(oof, targets, frames, logistic, form_preds, ctx_preds, lgbm_preds, w_lgbm):
    years = (2022, 2023, 2024)
    scale = 1.0 - w_lgbm
    wt_et = W_ET * scale
    wt_thgb = W_THGB * scale
    wt_ehgb = W_EHGB * scale
    wt_hhgb = W_HHGB * scale

    v11_base_preds = {}
    for yr in years:
        comp = oof[str(yr)]["components"]
        v11_base_preds[str(yr)] = (
            wt_et * comp["extra_trees"]
            + wt_thgb * comp["trackman_hgb"]
            + wt_ehgb * comp["te_trackman_hgb"]
            + wt_hhgb * comp["hierarchical_hgb"]
            + w_lgbm * lgbm_preds[str(yr)]
        )

    v17_pred = {
        str(yr): 0.95 * v11_base_preds[str(yr)] + 0.05 * logistic[str(yr)]
        for yr in years
    }
    raw_blend = {
        str(yr): W_V17 * v17_pred[str(yr)] + W_FORM * form_preds[str(yr)] + W_CTX * ctx_preds[str(yr)]
        for yr in years
    }

    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]]).reset_index(drop=True)
    res22 = np.asarray(targets["2022"]) - np.asarray(raw_blend["2022"])
    res_2223 = np.concatenate([res22, np.asarray(targets["2023"]) - np.asarray(raw_blend["2023"])])

    p22 = np.clip(raw_blend["2022"], 0, 1)
    cnt23 = segment_correction(f23_tr, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
    pcnt23 = segment_correction(f23_tr, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
    p23 = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt23 + 0.25 * pcnt23, 0, 1)
    
    cnt24 = segment_correction(f24_tr, res_2223, frames["2024"], ["balls_before", "strikes_before"], 500)
    pcnt24 = segment_correction(f24_tr, res_2223, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
    p24 = np.clip(raw_blend["2024"] + res_2223.mean() + 0.75 * cnt24 + 0.25 * pcnt24, 0, 1)

    s22 = float(brier_score_loss(targets["2022"], p22))
    s23 = float(brier_score_loss(targets["2023"], p23))
    s24 = float(brier_score_loss(targets["2024"], p24))
    
    g22 = V41_BASELINE[2022] - s22
    g23 = V41_BASELINE[2023] - s23
    g24 = V41_BASELINE[2024] - s24

    return {
        "scores": {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
    }


def main():
    print("Loading data and V41 artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]].reset_index(drop=True) for yr in years}
    y_dict = {str(yr): y_all.loc[oof[str(yr)]["row_index"]].reset_index(drop=True) for yr in years}

    # 정규화 파라미터 조합 탐색
    param_grid = [
        {"num_leaves": 15, "min_child_samples": 200, "reg_lambda": 20.0, "learning_rate": 0.03, "n_estimators": 500},
        {"num_leaves": 15, "min_child_samples": 300, "reg_lambda": 50.0, "learning_rate": 0.03, "n_estimators": 500},
        {"num_leaves": 31, "min_child_samples": 200, "reg_lambda": 20.0, "learning_rate": 0.03, "n_estimators": 400},
        {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 10.0, "learning_rate": 0.05, "n_estimators": 300},
        {"num_leaves": 31, "min_child_samples": 100, "reg_lambda": 10.0, "learning_rate": 0.05, "n_estimators": 300},
    ]

    all_results = []
    w_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

    for p_idx, params in enumerate(param_grid):
        print(f"\n--- Training Param Config [{p_idx+1}/{len(param_grid)}]: {params} ---", flush=True)
        lgbm_preds = build_lgbm_oof_and_preds(
            frames["2022"], y_dict["2022"],
            frames["2023"], y_dict["2023"],
            frames["2024"],
            params
        )
        
        for w in w_list:
            res = evaluate_lgbm_blend(
                oof, targets, frames, logistic, form_preds, ctx_preds, lgbm_preds, w
            )
            res_entry = {
                "config_id": p_idx + 1,
                "params": params,
                "w_lgbm": w,
                "scores": res["scores"],
                "gains_vs_v41": res["gains_vs_v41"],
                "all_improved": res["all_improved"],
                "submit_ready": res["submit_ready"],
            }
            all_results.append(res_entry)

    rank = lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"], r["gains_vs_v41"]["2022"])
    all_results.sort(key=rank, reverse=True)
    accepted = [r for r in all_results if r["all_improved"]]
    submit_ready = [r for r in all_results if r["submit_ready"]]

    output = {
        "experiment": "V69_lgbm_full_oof_and_regularization",
        "description": "2022년 5-Fold 완전 OOF 구축 및 LightGBM 정규화 파라미터 전수 튜닝",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(all_results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": all_results[:5],
    }

    Path("artifacts/v69_lgbm_full_oof_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
