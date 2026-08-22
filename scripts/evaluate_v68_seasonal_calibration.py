"""V68: 시즌 전반/후반(월별/계절성) 분기 Calibration 검증

Hypothesis:
-----------
시즌 전반기(봄~초여름: 3~6월)와 후반기(한여름~가을: 7~10월)는 날씨(기온/습도) 및
투수 피로도, 로스터 변화 등으로 인해 볼카운트별 잔차 보정 분포가 달라질 수 있음.
전반기(first_half: game_month <= 6)와 후반기(second_half: game_month >= 7)를 분기하여
각각 독립적으로 Segment Calibration (count 500구, pitcher_count 300구)을 적용.

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


def apply_seasonal_calibration(train_frame, train_residuals, test_frame, split_month=6, w_count=0.75, w_pitcher=0.25):
    """
    split_month 기준으로 전반부(<= split_month)와 후반부(> split_month)로 나누어
    독립적으로 segment correction을 계산하고 결합.
    """
    test_pred_adj = np.zeros(len(test_frame), dtype=float)
    
    train_first = train_frame["game_month"] <= split_month
    train_second = train_frame["game_month"] > split_month
    
    test_first = test_frame["game_month"] <= split_month
    test_second = test_frame["game_month"] > split_month
    
    # 1. 전반부 보정
    if train_first.sum() > 0 and test_first.sum() > 0:
        tf_tr = train_frame.loc[train_first]
        res_tr = train_residuals[train_first.to_numpy()]
        tf_te = test_frame.loc[test_first]
        
        cnt_1 = segment_correction(tf_tr, res_tr, tf_te, ["balls_before", "strikes_before"], 500)
        pcnt_1 = segment_correction(tf_tr, res_tr, tf_te, ["pitcher_id", "balls_before", "strikes_before"], 300)
        
        adj_1 = res_tr.mean() + w_count * cnt_1 + w_pitcher * pcnt_1
        test_pred_adj[test_first.to_numpy()] = adj_1
    elif test_first.sum() > 0:
        # fallback: 전체 평균
        test_pred_adj[test_first.to_numpy()] = train_residuals.mean()

    # 2. 후반부 보정
    if train_second.sum() > 0 and test_second.sum() > 0:
        tf_tr = train_frame.loc[train_second]
        res_tr = train_residuals[train_second.to_numpy()]
        tf_te = test_frame.loc[test_second]
        
        cnt_2 = segment_correction(tf_tr, res_tr, tf_te, ["balls_before", "strikes_before"], 500)
        pcnt_2 = segment_correction(tf_tr, res_tr, tf_te, ["pitcher_id", "balls_before", "strikes_before"], 300)
        
        adj_2 = res_tr.mean() + w_count * cnt_2 + w_pitcher * pcnt_2
        test_pred_adj[test_second.to_numpy()] = adj_2
    elif test_second.sum() > 0:
        # fallback: 전체 평균
        test_pred_adj[test_second.to_numpy()] = train_residuals.mean()
        
    return test_pred_adj


def main():
    print("Loading data and V41 artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id").copy()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]].reset_index(drop=True) for yr in years}
    v11_tr = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95 * v11_tr[str(yr)] + 0.05 * logistic[str(yr)] for yr in years}

    raw_blend = {
        str(yr): W_V17 * v17_pred[str(yr)] + W_FORM * form_preds[str(yr)] + W_CTX * ctx_preds[str(yr)]
        for yr in years
    }

    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]]).reset_index(drop=True)

    res22 = np.asarray(targets["2022"]) - np.asarray(raw_blend["2022"])
    res_2223 = np.concatenate([res22, np.asarray(targets["2023"]) - np.asarray(raw_blend["2023"])])


    p22_raw = np.clip(raw_blend["2022"], 0, 1)

    results = []

    # 다양한 split_month 기준 및 가중치 탐색
    for split_m in [5, 6, 7]:
        for w_cnt in [0.65, 0.75, 0.85]:
            w_pit = round(1.0 - w_cnt, 4)

            # 2023 seasonal calibration
            adj_23 = apply_seasonal_calibration(f23_tr, res22, frames["2023"], split_month=split_m, w_count=w_cnt, w_pitcher=w_pit)
            p23 = np.clip(raw_blend["2023"] + adj_23, 0, 1)

            # 2024 seasonal calibration
            adj_24 = apply_seasonal_calibration(f24_tr, res_2223, frames["2024"], split_month=split_m, w_count=w_cnt, w_pitcher=w_pit)
            p24 = np.clip(raw_blend["2024"] + adj_24, 0, 1)

            s22 = float(brier_score_loss(targets["2022"], p22_raw))
            s23 = float(brier_score_loss(targets["2023"], p23))
            s24 = float(brier_score_loss(targets["2024"], p24))

            g22 = V41_BASELINE[2022] - s22
            g23 = V41_BASELINE[2023] - s23
            g24 = V41_BASELINE[2024] - s24

            results.append({
                "split_month": split_m,
                "w_count": w_cnt,
                "w_pitcher": w_pit,
                "scores": {"2022": s22, "2023": s23, "2024": s24},
                "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
                "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
                "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
            })

    rank = lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"], r["gains_vs_v41"]["2022"])
    results.sort(key=rank, reverse=True)
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]

    output = {
        "experiment": "V68_seasonal_calibration",
        "description": "시즌 전반/후반 분기 Segment Calibration 평가",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": results[:5],
    }

    Path("artifacts/v68_seasonal_calibration_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)


if __name__ == "__main__":
    main()
