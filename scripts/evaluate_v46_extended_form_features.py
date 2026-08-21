"""Evaluate refined extended form & matchup stability features on gentle_500 Form HGB.

Available official asof columns:
  Pitcher recent: prev1, prev3, prev5 for success_rate and middle_rate.
  Pitcher season: success_rate, middle_rate, reverse_rate, ball_rate, strike_rate, pitchmix rates.
  Batter season: success_rate, middle_rate, n.

Feature sets tested:
  - base_v22 : the current 6 features (V41 baseline)
  - plus_prev1_momentum : + prev1 success/middle gaps & 1-3-5 exponential weighting
  - plus_control_margin : + strike/ball ratio, reverse/success ratio, location quality history
  - plus_batter_matchup_form : + pitcher-batter success/middle gap & batter sample reliability
  - plus_combined_stable : all carefully designed stable features combined

All models use the proven gentle_500 config:
  lr=0.03, max_iter=500, l2=20, max_leaf=15, min_samples=200.
And evaluated through the V41 3-way blend (0.55 / 0.32 / 0.13) + expanding calibration.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


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

FORM_HGB_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.03,
    "max_iter": 500,
}


def add_form_features_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    output = frame.copy()
    
    # 1. Base v22 features
    recent_success = (
        0.6 * output["asof_pitcher_prev3_game_success_rate"]
        + 0.4 * output["asof_pitcher_prev5_game_success_rate"]
    )
    output["stable_recent_success"] = recent_success
    output["stable_recent_gap"] = recent_success - output["asof_pitcher_success_rate"]
    output["stable_recent_disagreement"] = (
        output["asof_pitcher_prev3_game_success_rate"]
        - output["asof_pitcher_prev5_game_success_rate"]
    ).abs()
    reliability = output["asof_pitcher_n"].clip(lower=0) / (
        output["asof_pitcher_n"].clip(lower=0) + 200.0
    )
    output["stable_recent_gap_reliable"] = output["stable_recent_gap"] * reliability
    recent_middle = (
        0.6 * output["asof_pitcher_prev3_game_middle_rate"]
        + 0.4 * output["asof_pitcher_prev5_game_middle_rate"]
    )
    output["stable_recent_middle"] = recent_middle
    output["stable_recent_middle_gap"] = (
        recent_middle - output["asof_pitcher_middle_rate"]
    )
    
    if variant == "base_v22":
        return output
        
    # 2. prev1 short-term momentum
    if variant in ("plus_prev1_momentum", "plus_combined_stable"):
        recent_success_135 = (
            0.50 * output["asof_pitcher_prev1_game_success_rate"]
            + 0.30 * output["asof_pitcher_prev3_game_success_rate"]
            + 0.20 * output["asof_pitcher_prev5_game_success_rate"]
        )
        recent_middle_135 = (
            0.50 * output["asof_pitcher_prev1_game_middle_rate"]
            + 0.30 * output["asof_pitcher_prev3_game_middle_rate"]
            + 0.20 * output["asof_pitcher_prev5_game_middle_rate"]
        )
        output["stable_recent_success_135"] = recent_success_135
        output["stable_recent_middle_135"] = recent_middle_135
        output["stable_prev1_success_gap"] = (
            output["asof_pitcher_prev1_game_success_rate"] - output["asof_pitcher_success_rate"]
        ) * reliability
        output["stable_prev1_middle_gap"] = (
            output["asof_pitcher_prev1_game_middle_rate"] - output["asof_pitcher_middle_rate"]
        ) * reliability
        
    # 3. Control Margin features
    if variant in ("plus_control_margin", "plus_combined_stable"):
        output["stable_strike_ball_margin"] = (
            output["asof_pitcher_strike_rate"] - output["asof_pitcher_ball_rate"]
        )
        output["stable_quality_margin"] = (
            output["asof_pitcher_success_rate"]
            - (output["asof_pitcher_middle_rate"] + output["asof_pitcher_reverse_rate"])
        )
        
    # 4. Batter matchup stability
    if variant in ("plus_batter_matchup_form", "plus_combined_stable"):
        batter_reliability = output["asof_batter_n"].clip(lower=0) / (
            output["asof_batter_n"].clip(lower=0) + 200.0
        )
        output["stable_batter_success_gap"] = (
            output["asof_pitcher_success_rate"] - output["asof_batter_success_rate"]
        ) * batter_reliability
        output["stable_batter_middle_gap"] = (
            output["asof_pitcher_middle_rate"] - output["asof_batter_middle_rate"]
        ) * batter_reliability
        
    return output


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def fold_score_v46(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
    indices, targets, predictions = [], [], []
    for year in history_years:
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        blend = W_V17 * v17 + W_FORM * form_preds[str(year)] + W_CONTEXT * context_preds[str(year)]
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    train_frame = raw_frame.loc[index]
    
    valid_item = oof[str(valid_year)]
    v17_valid = 0.95 * v11_prediction(valid_item) + 0.05 * logistic[str(valid_year)]
    valid_pred_raw = (
        W_V17 * v17_valid
        + W_FORM * form_preds[str(valid_year)]
        + W_CONTEXT * context_preds[str(valid_year)]
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
    return float(brier_score_loss(valid_y, prediction))


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    item22 = oof["2022"]
    v17_22 = 0.95 * v11_prediction(item22) + 0.05 * logistic["2022"]
    
    VARIANTS = ["base_v22", "plus_prev1_momentum", "plus_control_margin", "plus_batter_matchup_form", "plus_combined_stable"]
    form_predictions = {v: {} for v in VARIANTS}
    
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        
        for variant in VARIANTS:
            form_raw = add_form_features_variant(hierarchical, variant)
            form_features = select_v2_features(add_row_features(form_raw, prior))
            form_features = add_trackman_features(form_features, trackman)
            columns = v31_form_columns(form_features)
            candidate = form_features[columns]
            
            model, cols = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in FORM_HGB_CONFIG.items()
            })
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            form_predictions[variant][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, cols]
            )[:, 1]
        print(f"year={year} all form feature variants done", flush=True)
        
    joblib.dump(form_predictions, "artifacts/v46_form_feature_predictions.joblib", compress=3)
    
    results = []
    for variant in VARIANTS:
        f_preds = form_predictions[variant]
        pred22_raw = W_V17 * v17_22 + W_FORM * f_preds["2022"] + W_CONTEXT * context_preds["2022"]
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22_raw))
        score23 = fold_score_v46(oof, logistic, f_preds, context_preds, raw_frame, [2022], 2023)
        score24 = fold_score_v46(oof, logistic, f_preds, context_preds, raw_frame, [2022, 2023], 2024)
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        results.append({
            "variant": variant,
            "scores": {
                "2022_raw": score22,
                "2023_calib": score23,
                "2024_calib": score24,
            },
            "gains_vs_v41": {
                "2022": gain22,
                "2023": gain23,
                "2024": gain24,
            },
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        
    results.sort(
        key=lambda x: (x["gains_vs_v41"]["2024"], x["gains_vs_v41"]["2023"]),
        reverse=True,
    )
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V46_extended_form_features",
        "description": "Evaluate location failure & quality form feature variants on gentle_500 Form HGB",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v46_form_features_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
