"""V65: LightGBM 이종 GBDT Base 트리 추가

현재 Base Layer: 4개 트리 (ExtraTrees, Trackman HGB, Encoded HGB, Hierarchical HGB)
V65 가설: LightGBM (leaf-wise 알고리즘)을 5번째 트리로 추가해 이종 GBDT 다양성 확보

구현:
  1. Trackman 피처 셋으로 LGBMClassifier OOF 예측 생성 (3-Fold 시간순)
  2. 기존 4-Tree 가중치 고정 후 5-Simplex 중 w_lgbm 비율만 격자 탐색
     (w_lgbm ∈ [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14] 탐색 후
      나머지를 V41 비율로 스케일링)
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

# V41 내 4-Tree 가중치 (V11 구조)
W_ET   = 0.2948
W_THGB = 0.2344
W_EHGB = 0.1162
W_HHGB = 0.3545

# LightGBM용 Trackman 피처 (trackman HGB와 동일 피처 셋)
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


def get_trackman_X(frame: pd.DataFrame) -> pd.DataFrame:
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

    v11_tr   = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}

    # ── LightGBM OOF 예측 생성 (시간순: 2022 → 2023, 2022+2023 → 2024) ──
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

    # 2022 LGBM: 자체 OOF 없으므로 leave-one-season 사용 불가 → 기존 Trackman HGB OOF 사용
    lgbm_preds = {
        "2022": v11_tr["2022"],  # 대체
        "2023": lgbm_23,
        "2024": lgbm_24,
    }

    # ── 격자 탐색: w_lgbm 비율 변화, 나머지는 V41 비율로 스케일링 ──
    print("Grid search w_lgbm...", flush=True)
    results = []
    w_lgbm_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

    for w_lgbm in w_lgbm_list:
        scale = 1.0 - w_lgbm
        # V11 4-tree 가중치를 스케일링
        wt_et   = W_ET   * scale
        wt_thgb = W_THGB * scale
        wt_ehgb = W_EHGB * scale
        wt_hhgb = W_HHGB * scale

        v11_base_preds = {}
        for yr in years:
            oof_yr = oof[str(yr)]
            comp = oof_yr["components"]
            tree_pred = (
                wt_et   * comp["extra_trees"] +
                wt_thgb * comp["trackman_hgb"] +
                wt_ehgb * comp["te_trackman_hgb"] +
                wt_hhgb * comp["hierarchical_hgb"] +
                w_lgbm  * lgbm_preds[str(yr)]
            )
            v11_base_preds[str(yr)] = tree_pred


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

    rank = lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"], r["gains_vs_v41"]["2022"])
    results.sort(key=rank, reverse=True)
    accepted     = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]

    output = {
        "experiment": "V65_lgbm_diversity",
        "description": "LightGBM 이종 GBDT 5번째 Base 트리 추가",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": results[:5],
    }

    Path("artifacts/v65_lgbm_diversity_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
