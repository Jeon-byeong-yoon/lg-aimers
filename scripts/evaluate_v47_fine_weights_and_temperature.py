"""Evaluate fine-grained 3-way blend weights (step 0.005) and probability temperature scaling on V41.

Context & Design:
-----------------
V41 achieved Public 900.7385 using macro weights: 0.55 (V17) / 0.32 (Form) / 0.13 (Context) found with step=0.01.
This experiment:
1. Explores a high-resolution grid around the V41 optimal point:
   - v17_weight: [0.52, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.58]
   - form_weight: [0.30, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34]
   - context_weight: [0.11, 0.12, 0.125, 0.13, 0.135, 0.14]
2. Evaluates temperature scaling (logit sharpening / flattening) on the raw blend before segment calibration:
   - Temperature T: [0.95, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.05]
   where calibrated_p = sigmoid(logit(p) / T).

Evaluation:
  2022 raw OOF, 2023 expanding calib, and 2024 expanding calib against V41 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


# V41 baseline scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

V17_WEIGHTS = [0.52, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.58]
FORM_WEIGHTS = [0.29, 0.30, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35]
CONTEXT_WEIGHTS = [0.10, 0.11, 0.12, 0.125, 0.13, 0.135, 0.14, 0.15]
TEMPERATURES = [0.96, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04]


def fold_score_v47(oof, logistic, form_preds, context_preds, raw_frame,
                   w_v17, w_form, w_ctx, temp, history_years, valid_year):
    indices, targets, predictions = [], [], []
    for year in history_years:
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        raw_blend = w_v17 * v17 + w_form * form_preds[str(year)] + w_ctx * context_preds[str(year)]
        if temp != 1.0:
            clipped_p = np.clip(raw_blend, 1e-6, 1 - 1e-6)
            raw_blend = expit(logit(clipped_p) / temp)
        predictions.append(raw_blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
        
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    train_frame = raw_frame.loc[index]
    
    valid_item = oof[str(valid_year)]
    v17_valid = 0.95 * v11_prediction(valid_item) + 0.05 * logistic[str(valid_year)]
    valid_blend_raw = w_v17 * v17_valid + w_form * form_preds[str(valid_year)] + w_ctx * context_preds[str(valid_year)]
    if temp != 1.0:
        clipped_valid = np.clip(valid_blend_raw, 1e-6, 1 - 1e-6)
        valid_blend_raw = expit(logit(clipped_valid) / temp)
        
    valid_frame = raw_frame.loc[valid_item["row_index"]]
    valid_y = valid_item["target"].astype(float)
    
    count_corr = segment_correction(
        train_frame, residual, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    pitcher_count_corr = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(
        valid_blend_raw + residual.mean() + 0.75 * count_corr + 0.25 * pitcher_count_corr,
        0, 1,
    )
    return float(brier_score_loss(valid_y, prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = raw.drop(columns=["row_id", "control_success"])
    
    item22 = oof["2022"]
    v17_22 = 0.95 * v11_prediction(item22) + 0.05 * logistic["2022"]
    target22 = item22["target"].astype(float)
    
    # Phase 1: High-resolution weights (temp=1.0)
    candidates = []
    for w_form in FORM_WEIGHTS:
        for w_ctx in CONTEXT_WEIGHTS:
            w_v17 = round(1.0 - (w_form + w_ctx), 4)
            if w_v17 < 0.45 or w_v17 > 0.65:
                continue
            for temp in [1.0]:
                pred22_raw = w_v17 * v17_22 + w_form * form_preds["2022"] + w_ctx * context_preds["2022"]
                score22 = float(brier_score_loss(target22, pred22_raw))
                score23 = fold_score_v47(oof, logistic, form_preds, context_preds, frame, w_v17, w_form, w_ctx, temp, [2022], 2023)
                score24 = fold_score_v47(oof, logistic, form_preds, context_preds, frame, w_v17, w_form, w_ctx, temp, [2022, 2023], 2024)
                
                gain22 = V41_BASELINE[2022] - score22
                gain23 = V41_BASELINE[2023] - score23
                gain24 = V41_BASELINE[2024] - score24
                
                candidates.append({
                    "w_v17": w_v17,
                    "w_form": w_form,
                    "w_context": w_ctx,
                    "temperature": temp,
                    "scores": {"2022": score22, "2023": score23, "2024": score24},
                    "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
                    "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
                    "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
                })
                
    # Phase 2: Temperature scaling on top of optimal weights
    for temp in TEMPERATURES:
        if temp == 1.0:
            continue
        w_v17, w_form, w_ctx = 0.55, 0.32, 0.13
        pred22_raw = w_v17 * v17_22 + w_form * form_preds["2022"] + w_ctx * context_preds["2022"]
        clipped_p = np.clip(pred22_raw, 1e-6, 1 - 1e-6)
        pred22_scaled = expit(logit(clipped_p) / temp)
        score22 = float(brier_score_loss(target22, pred22_scaled))
        score23 = fold_score_v47(oof, logistic, form_preds, context_preds, frame, w_v17, w_form, w_ctx, temp, [2022], 2023)
        score24 = fold_score_v47(oof, logistic, form_preds, context_preds, frame, w_v17, w_form, w_ctx, temp, [2022, 2023], 2024)
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        candidates.append({
            "w_v17": w_v17,
            "w_form": w_form,
            "w_context": w_ctx,
            "temperature": temp,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        
    accepted = [c for c in candidates if c["all_improved"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    rank = lambda item: (
        item["gains_vs_v41"]["2024"],
        item["gains_vs_v41"]["2023"],
        item["gains_vs_v41"]["2022"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V47_fine_weights_and_temperature",
        "description": "High-resolution 3-way weights and temperature scaling on V41 baseline",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v47_fine_weights_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
