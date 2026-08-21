"""Evaluate calibration segment parameters (thresholds and weights) on top of V41.

Current V41 calibration uses fixed legacy parameters:
  - count_columns: ['balls_before', 'strikes_before'] with min_count = 500, weight = 0.75
  - pitcher_count_columns: ['pitcher_id', 'balls_before', 'strikes_before'] with min_count = 300, weight = 0.25

V41 changed the 3-way blend to 0.55 / 0.32 / 0.13, which reduced residual variance.
This experiment tests whether tuning the smoothing thresholds and weights of the
two existing safe calibration segments improves generalization on all 3 seasons.

Grid:
  - count_min_samples: [300, 500, 700, 1000, 1500]
  - pitcher_count_min_samples: [150, 200, 300, 500, 700]
  - count_weight: [0.65, 0.70, 0.75, 0.80, 0.85]
  - pitcher_count_weight: [0.15, 0.20, 0.25, 0.30, 0.35]

Constraint: No new segment dimensions are added (per V29 failure lesson).
Only the thresholds and balance between count vs pitcher_count are tested.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


# V41 blend weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

# V41 baseline scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def fold_score_calib(train_frame, train_residuals, valid_frame, valid_target,
                     valid_raw_preds, count_min, pitcher_min, count_w, pitcher_w):
    count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["balls_before", "strikes_before"], count_min,
    )
    pitcher_count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], pitcher_min,
    )
    prediction = np.clip(
        valid_raw_preds + train_residuals.mean() + count_w * count_corr + pitcher_w * pitcher_count_corr,
        0, 1,
    )
    return float(brier_score_loss(valid_target, prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = raw.drop(columns=["row_id", "control_success"])
    
    # Compute raw blend predictions and residuals for 2022, 2023, 2024
    raw_preds = {}
    targets = {}
    frames = {}
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17_y = v17_prediction(item, logistic[str(year)])
        raw_preds[year] = (
            W_V17 * v17_y + W_FORM * form_preds[str(year)] + W_CONTEXT * context_preds[str(year)]
        )
        targets[year] = item["target"].astype(float)
        frames[year] = frame.loc[item["row_index"]]
        
    # Pre-build train folds
    # Fold 2023: train on 2022
    f23_train_frame = frames[2022]
    f23_train_res = targets[2022] - raw_preds[2022]
    f23_valid_frame = frames[2023]
    f23_valid_target = targets[2023]
    f23_valid_raw = raw_preds[2023]
    
    # Fold 2024: train on 2022 + 2023
    f24_train_frame = pd.concat([frames[2022], frames[2023]])
    f24_train_res = np.concatenate([targets[2022] - raw_preds[2022], targets[2023] - raw_preds[2023]])
    f24_valid_frame = frames[2024]
    f24_valid_target = targets[2024]
    f24_valid_raw = raw_preds[2024]
    
    count_mins = [300, 500, 700, 1000, 1500]
    pitcher_mins = [150, 200, 300, 500, 700]
    weight_pairs = [
        (0.65, 0.35),
        (0.70, 0.30),
        (0.75, 0.25),  # baseline
        (0.80, 0.20),
        (0.85, 0.15),
        (0.90, 0.10),
    ]
    
    candidates = []
    for count_min in count_mins:
        for pitcher_min in pitcher_mins:
            for count_w, pitcher_w in weight_pairs:
                score23 = fold_score_calib(
                    f23_train_frame, f23_train_res, f23_valid_frame, f23_valid_target,
                    f23_valid_raw, count_min, pitcher_min, count_w, pitcher_w,
                )
                score24 = fold_score_calib(
                    f24_train_frame, f24_train_res, f24_valid_frame, f24_valid_target,
                    f24_valid_raw, count_min, pitcher_min, count_w, pitcher_w,
                )
                
                # 2022 raw score is unaffected by calibration params
                score22 = float(brier_score_loss(targets[2022], raw_preds[2022]))
                
                gain23 = V41_BASELINE[2023] - score23
                gain24 = V41_BASELINE[2024] - score24
                
                candidates.append({
                    "count_min": count_min,
                    "pitcher_min": pitcher_min,
                    "count_weight": count_w,
                    "pitcher_weight": pitcher_w,
                    "scores": {
                        "2022": score22,
                        "2023": score23,
                        "2024": score24,
                    },
                    "gains_vs_v41": {
                        "2022": 0.0,
                        "2023": gain23,
                        "2024": gain24,
                    },
                    "improved_both_folds": (gain23 > 0 and gain24 > 0),
                    "submit_ready": (gain23 > 0 and gain24 > 5e-6),
                })
                
    accepted = [c for c in candidates if c["improved_both_folds"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    rank = lambda item: (
        item["gains_vs_v41"]["2024"],
        item["gains_vs_v41"]["2023"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V42_calibration_tuning",
        "description": "Grid search over count/pitcher_count smoothing thresholds and weights on V41 blend",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v42_calibration_tuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
