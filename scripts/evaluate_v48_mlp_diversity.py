"""Evaluate regularized Multi-Layer Perceptron (MLP) as a non-linear diversity component.

Hypothesis & Architecture:
--------------------------
Current V41 relies on:
- Tree ensembles (ExtraTrees + 3 HGBs): orthogonal axis-aligned partition models.
- Logistic Regression (5%): purely linear model.
- Form & Context HGBs.

A strongly regularized shallow neural network (MLP) provides smooth non-linear decision
boundaries that are fundamentally different from both tree splits and linear hyperplanes.

Pipeline:
  ColumnTransformer (OneHotEncoder for low-cardinal categoricals, StandardScaler for numerics)
  -> MLPClassifier(hidden_layer_sizes, alpha, early_stopping=True)

Configurations tested:
  - mlp_32_16_a5  : hidden=(32, 16), alpha=5.0
  - mlp_32_16_a10 : hidden=(32, 16), alpha=10.0
  - mlp_64_32_a10 : hidden=(64, 32), alpha=10.0
  - mlp_32_a5     : hidden=(32,), alpha=5.0
  - mlp_32_a10    : hidden=(32,), alpha=10.0

Evaluation:
  1. Measure standalone MLP Brier score per season.
  2. Test blending MLP into the V41 ensemble with weights w_mlp = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10].
  3. Validate on 2022 raw OOF, 2023 expanding calib, and 2024 expanding calib against V41 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.neural_network import MLPClassifier
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

# V41 baseline scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

MLP_CONFIGS = {
    "mlp_32_16_a5":  {"hidden_layer_sizes": (32, 16), "alpha": 5.0},
    "mlp_32_16_a10": {"hidden_layer_sizes": (32, 16), "alpha": 10.0},
    "mlp_64_32_a10": {"hidden_layer_sizes": (64, 32), "alpha": 10.0},
    "mlp_32_a5":     {"hidden_layer_sizes": (32,), "alpha": 5.0},
    "mlp_32_a10":    {"hidden_layer_sizes": (32,), "alpha": 10.0},
}

W_MLP_CANDIDATES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]


def mlp_pipeline(frame, hidden_layer_sizes, alpha):
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
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            batch_size=512,
            random_state=42,
        ),
    )
    return model, categorical + numeric


def fold_score_with_mlp(v11_tree_preds, logistic_preds, mlp_preds, w_mlp,
                        form_preds, context_preds, raw_frame, targets,
                        data_index, history_years, valid_year):
    # In V17 component: mix trees (0.95 - w_mlp), logistic (0.05), and mlp (w_mlp)
    w_trees = 0.95 - w_mlp
    w_log = 0.05
    
    train_y, train_preds = [], []
    for year in history_years:
        v17_y = w_trees * v11_tree_preds[year] + w_log * logistic_preds[str(year)] + w_mlp * mlp_preds[year]
        blend_y = W_V17 * v17_y + W_FORM * form_preds[str(year)] + W_CONTEXT * context_preds[str(year)]
        train_preds.append(blend_y)
        train_y.append(targets[year])
        
    train_frame = pd.concat([raw_frame.loc[data_index[year]] for year in history_years])
    train_residuals = np.concatenate(train_y) - np.concatenate(train_preds)
    
    valid_v17 = w_trees * v11_tree_preds[valid_year] + w_log * logistic_preds[str(valid_year)] + w_mlp * mlp_preds[valid_year]
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
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic_preds = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    data_index = {year: oof[str(year)]["row_index"] for year in (2022, 2023, 2024)}
    targets = {year: oof[str(year)]["target"].astype(float) for year in (2022, 2023, 2024)}
    
    v11_tree_preds = {
        year: v11_prediction(oof[str(year)])
        for year in (2022, 2023, 2024)
    }
    
    # Train each MLP variant per year
    mlp_predictions = {name: {} for name in MLP_CONFIGS}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(hierarchical, prior))
        features = add_trackman_features(features, trackman)
        
        for name, cfg in MLP_CONFIGS.items():
            model, cols = mlp_pipeline(features, cfg["hidden_layer_sizes"], cfg["alpha"])
            model.fit(features.loc[train_mask, cols], y.loc[train_mask])
            pred = model.predict_proba(features.loc[valid_mask, cols])[:, 1]
            mlp_predictions[name][year] = pred
            brier = brier_score_loss(y.loc[valid_mask], pred)
            print(f"year={year} {name} standalone Brier={brier:.6f}", flush=True)
            
    joblib.dump(mlp_predictions, "artifacts/v48_mlp_predictions.joblib", compress=3)
    
    candidates = []
    for mlp_name, mlp_preds in mlp_predictions.items():
        for w_mlp in W_MLP_CANDIDATES:
            w_trees = 0.95 - w_mlp
            w_log = 0.05
            
            # 2022 raw score
            v17_22 = w_trees * v11_tree_preds[2022] + w_log * logistic_preds["2022"] + w_mlp * mlp_preds[2022]
            pred22_raw = W_V17 * v17_22 + W_FORM * form_preds["2022"] + W_CONTEXT * context_preds["2022"]
            score22 = float(brier_score_loss(targets[2022], pred22_raw))
            
            # 2023 expanding calib
            score23, pred23_raw = fold_score_with_mlp(
                v11_tree_preds, logistic_preds, mlp_preds, w_mlp,
                form_preds, context_preds, raw_frame, targets,
                data_index, [2022], 2023,
            )
            # 2024 expanding calib
            score24, pred24_raw = fold_score_with_mlp(
                v11_tree_preds, logistic_preds, mlp_preds, w_mlp,
                form_preds, context_preds, raw_frame, targets,
                data_index, [2022, 2023], 2024,
            )
            
            gain22 = V41_BASELINE[2022] - score22
            gain23 = V41_BASELINE[2023] - score23
            gain24 = V41_BASELINE[2024] - score24
            
            candidates.append({
                "mlp_variant": mlp_name,
                "w_mlp": w_mlp,
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
        "experiment": "V48_mlp_diversity",
        "description": "Evaluate shallow regularized MLP neural networks as a non-linear diversity component in V41",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v48_mlp_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
