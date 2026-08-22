"""Evaluate Logistic diversity model tuning (C parameter & blend weight) on top of V41.

Current V41 setup:
  - Base Tree Blend: V11 (extra_trees 29.5%, trackman_hgb 23.4%, te_trackman_hgb 11.6%, hierarchical_hgb 35.5%)
  - V17 linear blend: 0.95 * V11_trees + 0.05 * Logistic(C=0.3)
  - V41 3-way blend: 0.55 * V17 + 0.32 * Form + 0.13 * Context + calibration

Since V41 changed the macro weights (V17 was reduced from 0.75 to 0.55),
the micro weight of the linear diversity model (w_log) and its regularization C
can be re-optimized across all 3 seasons.

Grid:
  - C values: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
  - Logistic weights w_log: [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20]

Evaluation:
  Full V41 expanding calibration pipeline on 2022 raw OOF, 2023 calib, 2024 calib.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


# V41 outer blend weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

# V41 baseline scores (with C=0.3, w_log=0.05)
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

ADDITIONAL_CS = [0.5, 1.0, 3.0]
W_LOGS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]


def logistic_pipeline(frame, c):
    categorical_candidates = LOW_CARDINAL_CATEGORICAL + [
        "game_month", "pitcher_team_id", "batter_team_id"
    ]
    categorical = [column for column in categorical_candidates if column in frame]
    dropped = {"pitcher_id", "batter_id", "season"}
    numeric = [column for column in frame if column not in categorical and column not in dropped]
    model = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ), categorical),
            ("numeric", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
            ), numeric),
        ]),
        LogisticRegression(C=c, max_iter=500, solver="lbfgs"),
    )
    return model, categorical + numeric


def fold_score_logistic(v11_tree_preds, log_preds, w_log, form_preds, context_preds,
                        raw_frame, targets, data_index, history_years, valid_year):
    train_y, train_preds = [], []
    for year in history_years:
        v17_y = (1.0 - w_log) * v11_tree_preds[year] + w_log * log_preds[str(year)]
        blend_y = W_V17 * v17_y + W_FORM * form_preds[str(year)] + W_CONTEXT * context_preds[str(year)]
        train_preds.append(blend_y)
        train_y.append(targets[year])
        
    train_frame = pd.concat([raw_frame.loc[data_index[year]] for year in history_years])
    train_residuals = np.concatenate(train_y) - np.concatenate(train_preds)
    
    valid_v17 = (1.0 - w_log) * v11_tree_preds[valid_year] + w_log * log_preds[str(valid_year)]
    valid_blend_raw = W_V17 * valid_v17 + W_FORM * form_preds[str(valid_year)] + W_CONTEXT * context_preds[str(valid_year)]
    valid_frame = raw_frame.loc[data_index[valid_year]]
    valid_target = targets[valid_year]
    
    count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    pitcher_count_corr = segment_correction(
        train_frame, train_residuals, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(
        valid_blend_raw + train_residuals.mean() + 0.75 * count_corr + 0.25 * pitcher_count_corr,
        0, 1,
    )
    return float(brier_score_loss(valid_target, prediction)), valid_blend_raw


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    # Load existing V17 logistic predictions
    all_log_preds = joblib.load("artifacts/v17_logistic_predictions.joblib")
    
    # Train additional C models if not present
    missing_cs = [c for c in ADDITIONAL_CS if str(c) not in all_log_preds]
    if missing_cs:
        print(f"Training additional C values: {missing_cs}...", flush=True)
        encoded = add_prior_season_target_encodings(data, y)
        hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
        raw_trackman = pd.read_csv(
            "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
        )
        trackman = prepare_trackman(raw_trackman)
        
        for c in missing_cs:
            c_preds = {}
            for year in (2022, 2023, 2024):
                train_mask = data["season"] < year
                valid_mask = data["season"] == year
                prior = float(y.loc[train_mask].mean())
                features = select_v2_features(add_row_features(hierarchical, prior))
                features = add_trackman_features(features, trackman)
                model, columns = logistic_pipeline(features, c)
                model.fit(features.loc[train_mask, columns], y.loc[train_mask])
                c_preds[str(year)] = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
            all_log_preds[str(c)] = c_preds
            print(f"C={c} done", flush=True)
        joblib.dump(all_log_preds, "artifacts/v17_logistic_predictions.joblib", compress=3)
        
    # Load other components
    v6_oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    data_index = {year: v6_oof[str(year)]["row_index"] for year in (2022, 2023, 2024)}
    targets = {year: v6_oof[str(year)]["target"].astype(float) for year in (2022, 2023, 2024)}
    
    # V11 composite tree prediction for each year
    v11_tree_preds = {
        year: v11_prediction(v6_oof[str(year)])
        for year in (2022, 2023, 2024)
    }
    
    candidates = []
    for c_str, log_preds in all_log_preds.items():
        c_val = float(c_str)
        for w_log in W_LOGS:
            # 2022 raw score
            v17_22 = (1.0 - w_log) * v11_tree_preds[2022] + w_log * log_preds["2022"]
            pred22_raw = W_V17 * v17_22 + W_FORM * form_preds["2022"] + W_CONTEXT * context_preds["2022"]
            score22 = float(brier_score_loss(targets[2022], pred22_raw))
            
            # 2023 expanding calib
            score23, pred23_raw = fold_score_logistic(
                v11_tree_preds, log_preds, w_log, form_preds, context_preds,
                raw_frame, targets, data_index, [2022], 2023,
            )
            # 2024 expanding calib
            score24, pred24_raw = fold_score_logistic(
                v11_tree_preds, log_preds, w_log, form_preds, context_preds,
                raw_frame, targets, data_index, [2022, 2023], 2024,
            )
            
            gain22 = V41_BASELINE[2022] - score22
            gain23 = V41_BASELINE[2023] - score23
            gain24 = V41_BASELINE[2024] - score24
            
            candidates.append({
                "C": c_val,
                "w_log": w_log,
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
        "experiment": "V45_logistic_diversity_tuning",
        "description": "Grid search over Logistic C parameter and blend weight w_log on V41 pipeline",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v45_logistic_tuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
