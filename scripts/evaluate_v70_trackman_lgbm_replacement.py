"""V70: Base `trackman_hgb`(23.4%)를 정규화된 `trackman_lgbm`으로 1:1 전면 교체 검증

Hypothesis:
-----------
기존 V11 Base Layer의 4-Tree 구성:
  - ExtraTrees: 29.48%
  - Trackman HGB: 23.44%
  - Encoded HGB: 11.62%
  - Hierarchical HGB: 35.45%

여기서 동일한 8개 트랙맨 물리 피처를 보는 `Trackman HGB` 자리를
LightGBM(Leaf-wise GBDT)으로 1:1 전면 교체하여
깊이 분할(Depth-wise HGB)과 리프 분할(Leaf-wise LGBM)의 완전한 시너지 효과를 검증.

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib).
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


def build_trackman_lgbm_predictions(f22, y22, f23, y23, f24, params):
    """5-Fold OOF 및 2023/2024 Trackman LightGBM 예측 생성"""
    X22 = get_trackman_X(f22)
    X23 = get_trackman_X(f23)
    X24 = get_trackman_X(f24)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_22 = np.zeros(len(y22), dtype=float)

    for tr_idx, val_idx in skf.split(X22, y22):
        X_tr, y_tr = X22.iloc[tr_idx], y22.iloc[tr_idx]
        X_val = X22.iloc[val_idx]

        clf = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        oof_22[val_idx] = clf.predict_proba(X_val)[:, 1]

    clf_23 = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    clf_23.fit(X22, y22)
    pred_23 = clf_23.predict_proba(X23)[:, 1]

    f2223 = pd.concat([f22, f23]).reset_index(drop=True)
    y2223 = pd.concat([y22, y23]).reset_index(drop=True)
    X2223 = get_trackman_X(f2223)

    clf_24 = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    clf_24.fit(X2223, y2223)
    pred_24 = clf_24.predict_proba(X24)[:, 1]

    return {
        "2022": oof_22,
        "2023": pred_23,
        "2024": pred_24,
    }


def main():
    print("Loading data and V41 artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_hgb = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]].reset_index(drop=True) for yr in years}
    y_dict = {str(yr): y_all.loc[oof[str(yr)]["row_index"]].reset_index(drop=True) for yr in years}

    # 정규화 파라미터 조합
    param_configs = [
        {"num_leaves": 15, "min_child_samples": 200, "reg_lambda": 20.0, "learning_rate": 0.03, "n_estimators": 500},
        {"num_leaves": 15, "min_child_samples": 300, "reg_lambda": 50.0, "learning_rate": 0.03, "n_estimators": 500},
        {"num_leaves": 31, "min_child_samples": 200, "reg_lambda": 20.0, "learning_rate": 0.03, "n_estimators": 400},
        {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 10.0, "learning_rate": 0.05, "n_estimators": 300},
    ]

    all_results = []

    for p_idx, params in enumerate(param_configs):
        print(f"\n--- Training Trackman LightGBM [{p_idx+1}/{len(param_configs)}]: {params} ---", flush=True)
        tm_lgbm = build_trackman_lgbm_predictions(
            frames["2022"], y_dict["2022"],
            frames["2023"], y_dict["2023"],
            frames["2024"],
            params
        )

        # 1. 100% 1:1 교체 (W_THGB = 0.2344 자리를 그대로 tm_lgbm으로 교체)
        # 2. 비율 혼합 (hgb vs lgbm 비중 탐색: 0:100, 20:80, 40:60, 50:50, 60:40, 80:20)
        for ratio_lgbm in [1.0, 0.8, 0.6, 0.5, 0.4, 0.2]:
            v11_custom = {}
            for yr in years:
                comp = oof[str(yr)]["components"]
                thgb_blend = (1.0 - ratio_lgbm) * comp["trackman_hgb"] + ratio_lgbm * tm_lgbm[str(yr)]
                v11_custom[str(yr)] = (
                    W_ET * comp["extra_trees"]
                    + W_THGB * thgb_blend
                    + W_EHGB * comp["te_trackman_hgb"]
                    + W_HHGB * comp["hierarchical_hgb"]
                )

            v17_pred = {
                str(yr): 0.95 * v11_custom[str(yr)] + 0.05 * logistic[str(yr)]
                for yr in years
            }
            raw_blend = {
                str(yr): W_V17 * v17_pred[str(yr)] + W_FORM * form_hgb[str(yr)] + W_CTX * ctx_preds[str(yr)]
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

            entry = {
                "config_id": p_idx + 1,
                "params": params,
                "ratio_lgbm": ratio_lgbm,
                "scores": {"2022": s22, "2023": s23, "2024": s24},
                "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
                "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
                "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
            }
            all_results.append(entry)

    rank = lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"], r["gains_vs_v41"]["2022"])
    all_results.sort(key=rank, reverse=True)
    accepted = [r for r in all_results if r["all_improved"]]
    submit_ready = [r for r in all_results if r["submit_ready"]]

    output = {
        "experiment": "V70_trackman_lgbm_replacement",
        "description": "Base Trackman HGB(23.4%)의 Trackman LightGBM 1:1 전면 교체 및 혼합 검증",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(all_results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": all_results[:5],
    }

    Path("artifacts/v70_trackman_lgbm_replacement_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
