"""Validate V41 robustness and calibration rebuild under the optimal 3-way weights.

Selected optimal weight candidates:
  Candidate A (highest 2024 gain):
    v17: 0.55, form: 0.32, context: 0.13
  Candidate B (balanced across all 3 years):
    v17: 0.56, form: 0.32, context: 0.12
  Candidate C (slightly more conservative):
    v17: 0.58, form: 0.30, context: 0.12

This script:
1. Re-computes chronological fold calibration for each candidate.
2. Checks raw predictions correlation with V38 and V31.
3. Builds the full training set calibration artifact for packaging.
4. Outputs comprehensive metrics.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


CANDIDATES = {
    "v41_a_55_32_13": {"v17": 0.55, "form": 0.32, "context": 0.13},
    "v41_b_56_32_12": {"v17": 0.56, "form": 0.32, "context": 0.12},
    "v41_c_58_30_12": {"v17": 0.58, "form": 0.30, "context": 0.12},
    "v41_d_60_28_12": {"v17": 0.60, "form": 0.28, "context": 0.12},
}

V38_BASELINE = {
    2022: 0.243390002346232,
    2023: 0.25324438180845804,
    2024: 0.24783152211261955,
}

V31_BASELINE = {
    2022: 0.24339835671689145,
    2023: 0.25326829650157456,
    2024: 0.24783690211517923,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def fold_score_candidate(oof, logistic, form_preds, context_preds, raw_frame,
                         weights, history_years, valid_year):
    w_v17 = weights["v17"]
    w_form = weights["form"]
    w_ctx = weights["context"]
    
    indices, targets, predictions = [], [], []
    for year in history_years:
        item = oof[str(year)]
        v17 = v17_prediction(item, logistic[str(year)])
        blend = (
            w_v17 * v17
            + w_form * form_preds[str(year)]
            + w_ctx * context_preds[str(year)]
        )
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
        
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    train_frame = raw_frame.loc[index]
    
    valid_item = oof[str(valid_year)]
    v17_valid = v17_prediction(valid_item, logistic[str(valid_year)])
    valid_pred_raw = (
        w_v17 * v17_valid
        + w_form * form_preds[str(valid_year)]
        + w_ctx * context_preds[str(valid_year)]
    )
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
        valid_pred_raw + residual.mean() + 0.75 * count_corr + 0.25 * pitcher_count_corr,
        0, 1,
    )
    return float(brier_score_loss(valid_y, prediction)), valid_pred_raw


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = raw.drop(columns=["row_id", "control_success"])
    
    item22 = oof["2022"]
    v17_22 = v17_prediction(item22, logistic["2022"])
    
    # V38 predictions for correlation check
    v38_raw_preds = {}
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17_y = v17_prediction(item, logistic[str(year)])
        v38_raw_preds[year] = (
            0.75 * v17_y + 0.16 * form_preds[str(year)] + 0.09 * context_preds[str(year)]
        )
        
    results = {}
    for name, weights in CANDIDATES.items():
        w_v17 = weights["v17"]
        w_form = weights["form"]
        w_ctx = weights["context"]
        
        # 2022 raw prediction
        pred22_raw = (
            w_v17 * v17_22
            + w_form * form_preds["2022"]
            + w_ctx * context_preds["2022"]
        )
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22_raw))
        
        score23, pred23_raw = fold_score_candidate(
            oof, logistic, form_preds, context_preds, frame,
            weights, [2022], 2023,
        )
        score24, pred24_raw = fold_score_candidate(
            oof, logistic, form_preds, context_preds, frame,
            weights, [2022, 2023], 2024,
        )
        
        # Correlations with V38
        corr22 = float(np.corrcoef(pred22_raw, v38_raw_preds[2022])[0, 1])
        corr23 = float(np.corrcoef(pred23_raw, v38_raw_preds[2023])[0, 1])
        corr24 = float(np.corrcoef(pred24_raw, v38_raw_preds[2024])[0, 1])
        
        gains_v38 = {
            "2022": V38_BASELINE[2022] - score22,
            "2023": V38_BASELINE[2023] - score23,
            "2024": V38_BASELINE[2024] - score24,
        }
        gains_v31 = {
            "2022": V31_BASELINE[2022] - score22,
            "2023": V31_BASELINE[2023] - score23,
            "2024": V31_BASELINE[2024] - score24,
        }
        
        results[name] = {
            "weights": weights,
            "scores": {
                "2022_raw": score22,
                "2023_calib": score23,
                "2024_calib": score24,
            },
            "gains_vs_v38": gains_v38,
            "gains_vs_v31": gains_v31,
            "correlations_with_v38": {
                "2022": corr22,
                "2023": corr23,
                "2024": corr24,
            },
            "all_improved_vs_v38": (
                gains_v38["2022"] > 0 and gains_v38["2023"] > 0 and gains_v38["2024"] > 0
            ),
            "submit_ready_vs_v38": (
                gains_v38["2022"] > 0 and gains_v38["2023"] > 0 and gains_v38["2024"] > 5e-6
            ),
        }
        
    output_path = Path("artifacts/v41_robustness_metrics.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
