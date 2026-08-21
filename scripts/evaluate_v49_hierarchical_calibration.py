"""Evaluate hierarchical pitcher-group calibration on top of V41.

Hypothesis & Architecture:
--------------------------
In V41:
- Segment 1: balls_before, strikes_before (global count, min_count=500, w=0.75)
- Segment 2: pitcher_id, balls_before, strikes_before (pitcher-specific count, min_count=300, w=0.25)

Limitation:
- Pitchers with < 300 observed pitches in history receive ZERO pitcher-level correction.
- A natural intermediate hierarchical segment is:
    Segment Intermediate: pitcher_hand, balls_before, strikes_before (min_count=1000)
    or
    Segment Intermediate: pitcher_hand, pitch_type_preference, balls_before, strikes_before (min_count=1000)

This smooths the shrinkage gap between coarse global counts and fine pitcher-specific counts.

Combinations tested:
  - hand_count: [pitcher_hand, balls_before, strikes_before]
  - hand_batter_hand_count: [pitcher_hand, batter_hand, balls_before, strikes_before]
  - hand_outs_count: [pitcher_hand, outs_before, balls_before, strikes_before]

Weights tested:
  - (w_count, w_inter, w_pitcher) in:
    (0.65, 0.15, 0.20), (0.70, 0.10, 0.20), (0.60, 0.20, 0.20), (0.70, 0.15, 0.15), (0.75, 0.10, 0.15)

Evaluation:
  2022 raw OOF, 2023 expanding calib, 2024 expanding calib against V41 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


# V41 outer blend weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

# V41 baseline scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

INTERMEDIATE_SEGMENTS = {
    "hand_count": {
        "columns": ["pitcher_hand", "balls_before", "strikes_before"],
        "min_samples": 500,
    },
    "hand_count_1000": {
        "columns": ["pitcher_hand", "balls_before", "strikes_before"],
        "min_samples": 1000,
    },
    "platoon_count": {
        "columns": ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"],
        "min_samples": 500,
    },
    "platoon_count_1000": {
        "columns": ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"],
        "min_samples": 1000,
    },
}

WEIGHT_TRIPLETS = [
    (0.65, 0.10, 0.25),
    (0.65, 0.15, 0.20),
    (0.70, 0.05, 0.25),
    (0.70, 0.10, 0.20),
    (0.60, 0.15, 0.25),
    (0.60, 0.20, 0.20),
    (0.75, 0.05, 0.20),
]


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def fold_score_hierarchical(train_frame, train_residuals, valid_frame, valid_target,
                            valid_raw_preds, inter_cols, inter_min,
                            w_count, w_inter, w_pitcher):
    # 1. Global count correction
    count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    # 2. Intermediate segment correction (e.g. pitcher_hand x count)
    inter_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        inter_cols, inter_min,
    )
    # 3. Fine pitcher x count correction
    pitcher_count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    
    prediction = np.clip(
        valid_raw_preds + train_residuals.mean()
        + w_count * count_corr
        + w_inter * inter_corr
        + w_pitcher * pitcher_count_corr,
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
    
    score22 = float(brier_score_loss(targets[2022], raw_preds[2022]))
    
    candidates = []
    for inter_name, inter_cfg in INTERMEDIATE_SEGMENTS.items():
        inter_cols = inter_cfg["columns"]
        inter_min = inter_cfg["min_samples"]
        
        for w_count, w_inter, w_pitcher in WEIGHT_TRIPLETS:
            score23 = fold_score_hierarchical(
                f23_train_frame, f23_train_res, f23_valid_frame, f23_valid_target,
                f23_valid_raw, inter_cols, inter_min,
                w_count, w_inter, w_pitcher,
            )
            score24 = fold_score_hierarchical(
                f24_train_frame, f24_train_res, f24_valid_frame, f24_valid_target,
                f24_valid_raw, inter_cols, inter_min,
                w_count, w_inter, w_pitcher,
            )
            
            gain23 = V41_BASELINE[2023] - score23
            gain24 = V41_BASELINE[2024] - score24
            
            candidates.append({
                "segment": inter_name,
                "w_count": w_count,
                "w_inter": w_inter,
                "w_pitcher": w_pitcher,
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
        "experiment": "V49_hierarchical_calibration",
        "description": "Evaluate 3-tier hierarchical calibration (global count -> platoon/hand count -> pitcher count)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v49_hierarchical_calibration_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
