"""V72: LightGBM Extremely Randomized Trees (`extra_trees=True`) 모드 및 융합 검증

Hypothesis:
-----------
sklearn ExtraTrees가 Base Layer에서 29.5%의 최고 비중을 차지하는 것과 마찬가지로,
LightGBM의 `extra_trees=True` (무작위 분할) 모드를 활성화하면
트랙맨 및 폼 피처의 노이즈를 효과적으로 흡수하고 2024년 일반화 성능을 대폭 끌어올릴 수 있음.

Form Layer 및 Base Trackman Layer 각각에 대해 `extra_trees=True` LightGBM을 학습하고
미세 가중치 앙상블 전수 평가.
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

FORM_FEATURES = [
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


def get_features_X(frame, feat_list):
    feats = [f for f in feat_list if f in frame.columns]
    X = frame[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(X[c].median())
    return X


def build_et_lgbm_predictions(f22, y22, f23, y23, f24, feat_list, params):
    """5-Fold OOF 및 2023/2024 ExtraTrees LightGBM 예측 생성"""
    X22 = get_features_X(f22, feat_list)
    X23 = get_features_X(f23, feat_list)
    X24 = get_features_X(f24, feat_list)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_22 = np.zeros(len(y22), dtype=float)

    for tr_idx, val_idx in skf.split(X22, y22):
        X_tr, y_tr = X22.iloc[tr_idx], y22.iloc[tr_idx]
        X_val = X22.iloc[val_idx]

        clf = LGBMClassifier(**params, extra_trees=True, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        oof_22[val_idx] = clf.predict_proba(X_val)[:, 1]

    clf_23 = LGBMClassifier(**params, extra_trees=True, random_state=42, verbose=-1, n_jobs=-1)
    clf_23.fit(X22, y22)
    pred_23 = clf_23.predict_proba(X23)[:, 1]

    f2223 = pd.concat([f22, f23]).reset_index(drop=True)
    y2223 = pd.concat([y22, y23]).reset_index(drop=True)
    X2223 = get_features_X(f2223, feat_list)

    clf_24 = LGBMClassifier(**params, extra_trees=True, random_state=42, verbose=-1, n_jobs=-1)
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

    v11_tr = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95 * v11_tr[str(yr)] + 0.05 * logistic[str(yr)] for yr in years}

    # Form ET LightGBM 모델 학습
    et_params = {
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 20.0,
        "learning_rate": 0.03,
        "n_estimators": 500,
    }

    print("\n--- Training Form ExtraTrees-LightGBM ---", flush=True)
    form_et_lgbm = build_et_lgbm_predictions(
        frames["2022"], y_dict["2022"],
        frames["2023"], y_dict["2023"],
        frames["2024"],
        FORM_FEATURES,
        et_params
    )

    print("\n--- Training Base Trackman ExtraTrees-LightGBM ---", flush=True)
    base_et_lgbm = build_et_lgbm_predictions(
        frames["2022"], y_dict["2022"],
        frames["2023"], y_dict["2023"],
        frames["2024"],
        TRACKMAN_FEATURES,
        et_params
    )

    all_results = []
    
    # 1. Form Layer ET-LGBM 블렌드 격자
    for w_form in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for w_base in [0.0, 0.01, 0.02, 0.03, 0.05]:
            form_blended = {
                str(yr): (1.0 - w_form) * form_hgb[str(yr)] + w_form * form_et_lgbm[str(yr)]
                for yr in years
            }
            
            if w_base > 0:
                scale = 1.0 - w_base
                v11_custom = {}
                for yr in years:
                    comp = oof[str(yr)]["components"]
                    v11_custom[str(yr)] = (
                        W_ET * scale * comp["extra_trees"]
                        + W_THGB * scale * comp["trackman_hgb"]
                        + W_EHGB * scale * comp["te_trackman_hgb"]
                        + W_HHGB * scale * comp["hierarchical_hgb"]
                        + w_base * base_et_lgbm[str(yr)]
                    )
                v17_curr = {
                    str(yr): 0.95 * v11_custom[str(yr)] + 0.05 * logistic[str(yr)]
                    for yr in years
                }
            else:
                v17_curr = v17_pred

            raw_blend = {
                str(yr): W_V17 * v17_curr[str(yr)] + W_FORM * form_blended[str(yr)] + W_CTX * ctx_preds[str(yr)]
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
                "w_form_et_lgbm": w_form,
                "w_base_et_lgbm": w_base,
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
        "experiment": "V72_lgbm_extra_trees_fusion",
        "description": "LightGBM ExtraTrees(무작위 분할) 모드 기반 Form/Base 융합 평가",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(all_results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": all_results[:5],
    }

    Path("artifacts/v72_lgbm_extra_trees_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
