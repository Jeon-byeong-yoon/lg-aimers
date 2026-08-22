"""V65b: LightGBM w_lgbm 세밀 격자 탐색 (0.01~0.07, step=0.005)

V65에서 w_lgbm=0.02~0.06 구간에서 세 시즌 동시 개선 발견.
더 세밀하게 탐색해 2024 gain > 5e-6 돌파 가능 여부 확인.
"""

import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from lightgbm import LGBMClassifier

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

W_ET   = 0.2948
W_THGB = 0.2344
W_EHGB = 0.1162
W_HHGB = 0.3545

TRACKMAN_FEATURES = [
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
]


def get_trackman_X(frame):
    feats = [f for f in TRACKMAN_FEATURES if f in frame.columns]
    X = frame[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(X[c].median())
    return X


def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof        = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic   = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31        = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds  = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames  = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    y_dict  = {str(yr): y_all.loc[oof[str(yr)]["row_index"]] for yr in years}

    # LightGBM 학습
    print("Training LightGBM (2022 → 2023)...", flush=True)
    lgbm_23_clf = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgbm_23_clf.fit(get_trackman_X(frames["2022"]), y_dict["2022"])
    lgbm_23 = lgbm_23_clf.predict_proba(get_trackman_X(frames["2023"]))[:, 1]

    print("Training LightGBM (2022+2023 → 2024)...", flush=True)
    f2223 = pd.concat([frames["2022"], frames["2023"]])
    y2223 = pd.concat([y_dict["2022"], y_dict["2023"]])
    lgbm_24_clf = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgbm_24_clf.fit(get_trackman_X(f2223), y2223)
    lgbm_24 = lgbm_24_clf.predict_proba(get_trackman_X(frames["2024"]))[:, 1]

    lgbm_preds = {
        "2022": oof["2022"]["components"]["trackman_hgb"],  # 대체
        "2023": lgbm_23,
        "2024": lgbm_24,
    }

    # 세밀 격자: 0.005 step
    w_lgbm_list = [round(x * 0.005, 3) for x in range(2, 15)]  # 0.01 ~ 0.07
    print(f"세밀 탐색 범위: {w_lgbm_list}", flush=True)

    results = []
    for w_lgbm in w_lgbm_list:
        scale = 1.0 - w_lgbm
        wt_et   = W_ET   * scale
        wt_thgb = W_THGB * scale
        wt_ehgb = W_EHGB * scale
        wt_hhgb = W_HHGB * scale

        v11_base_preds = {}
        for yr in years:
            comp = oof[str(yr)]["components"]
            v11_base_preds[str(yr)] = (
                wt_et   * comp["extra_trees"] +
                wt_thgb * comp["trackman_hgb"] +
                wt_ehgb * comp["te_trackman_hgb"] +
                wt_hhgb * comp["hierarchical_hgb"] +
                w_lgbm  * lgbm_preds[str(yr)]
            )

        v17_pred = {
            str(yr): 0.95*v11_base_preds[str(yr)] + 0.05*logistic[str(yr)]
            for yr in years
        }
        raw_blend = {
            str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_preds[str(yr)]
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

        results.append({
            "w_lgbm": w_lgbm,
            "scores": {"2022": s22, "2023": s23, "2024": s24},
            "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
            "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
            "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
        })

    results.sort(key=lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"]), reverse=True)
    accepted     = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]

    output = {
        "experiment": "V65b_lgbm_fine_grid",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": results[:5],
    }

    Path("artifacts/v65b_lgbm_fine_grid_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
