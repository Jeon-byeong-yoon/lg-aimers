"""Evaluate regularization and lower learning rate on the base Hierarchical HGB model.

Context & Hypothesis:
---------------------
In the V41 ensemble:
- Form HGB achieved dramatic improvements via gentle_500 (lr=0.03, max_iter=500, l2=20, max_leaf=15, min_samples=200).
- However, the base tree ensemble (V6/V11) still uses default V2 parameters for all its HGBs:
    lr=0.06, max_iter=200, max_leaf_nodes=31, min_samples_leaf=100, l2=5.0.
- `hierarchical_hgb` has the largest single tree weight in V11 (35.45%).
- Applying the proven gentle/strong regularization to `hierarchical_hgb` should reduce
  tree-level overfitting and improve the entire V17/V41 base layer for all seasons.

Configurations tested for `hierarchical_hgb`:
  - v6_baseline : default V2 (lr=0.06, iter=200, leaf=31, min_leaf=100, l2=5.0)
  - gentle_500  : lr=0.03, iter=500, leaf=15, min_leaf=200, l2=20.0
  - gentle_400  : lr=0.03, iter=400, leaf=15, min_leaf=200, l2=20.0
  - balanced_400: lr=0.03, iter=400, leaf=23, min_leaf=150, l2=10.0
  - mild_300    : lr=0.04, iter=300, leaf=23, min_leaf=150, l2=10.0

Evaluation:
  Recompute the full V41 pipeline (Base Tree Ensemble -> Logistic -> V41 3-way blend -> calibration)
  and evaluate 2022 raw OOF, 2023 expanding calib, and 2024 expanding calib against V41 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


# Tree weights inside V11
TREE_WEIGHTS = {
    "extra_trees": 0.29483562599237795,
    "trackman_hgb": 0.2344456574665245,
    "te_trackman_hgb": 0.11617615437521091,
    "hierarchical_hgb": 0.35454256216588664,
}

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

HGB_CONFIGS = {
    "gentle_500":   {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500},
    "gentle_400":   {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 400},
    "balanced_400": {"max_leaf_nodes": 23, "min_samples_leaf": 150, "l2_regularization": 10.0, "learning_rate": 0.03, "max_iter": 400},
    "mild_300":     {"max_leaf_nodes": 23, "min_samples_leaf": 150, "l2_regularization": 10.0, "learning_rate": 0.04, "max_iter": 300},
}


def fold_score_full(oof_tree_preds, logistic_preds, form_preds, context_preds,
                    raw_frame, targets, history_years, valid_year):
    train_y, train_preds = [], []
    for year in history_years:
        v17_y = 0.95 * oof_tree_preds[year] + 0.05 * logistic_preds[str(year)]
        blend_y = W_V17 * v17_y + W_FORM * form_preds[str(year)] + W_CONTEXT * context_preds[str(year)]
        train_preds.append(blend_y)
        train_y.append(targets[year])
        
    train_frame = pd.concat([raw_frame.loc[data_index[year]] for year in history_years])
    train_residuals = np.concatenate(train_y) - np.concatenate(train_preds)
    
    valid_v17 = 0.95 * oof_tree_preds[valid_year] + 0.05 * logistic_preds[str(valid_year)]
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
    pred_path = Path("artifacts/v43_hierarchical_hgb_predictions.joblib")
    if pred_path.exists():
        print(f"Loading existing predictions from {pred_path}...", flush=True)
        hierarchical_preds = joblib.load(pred_path)
    else:
        data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
        y = data.pop("control_success").astype("uint8")
        data = data.drop(columns="row_id")
        
        encoded = add_prior_season_target_encodings(data, y)
        hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
        raw_trackman = pd.read_csv(
            "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
        )
        trackman = prepare_trackman(raw_trackman)
        
        hierarchical_preds = {name: {} for name in HGB_CONFIGS}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            hierarchical_base = select_v2_features(add_row_features(hierarchical, prior))
            hierarchical_features = add_trackman_features(hierarchical_base, trackman)
            
            for name, config in HGB_CONFIGS.items():
                model, cols = hist_gbdt_pipeline(hierarchical_features)
                model.set_params(**{
                    f"histgradientboostingclassifier__{k}": v for k, v in config.items()
                })
                model.fit(hierarchical_features.loc[train_mask, cols], y.loc[train_mask])
                hierarchical_preds[name][year] = model.predict_proba(
                    hierarchical_features.loc[valid_mask, cols]
                )[:, 1]
            print(f"year={year} hierarchical_hgb all variants done", flush=True)
        joblib.dump(hierarchical_preds, pred_path, compress=3)
        
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = data.drop(columns=["row_id", "control_success"])
    
    # Load existing V6 OOF components
    v6_oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic_preds = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    global data_index
    data_index = {year: v6_oof[str(year)]["row_index"] for year in (2022, 2023, 2024)}
    targets = {year: v6_oof[str(year)]["target"].astype(float) for year in (2022, 2023, 2024)}
    
    # Other fixed tree components in V6
    extra_trees_preds = {year: v6_oof[str(year)]["components"]["extra_trees"] for year in (2022, 2023, 2024)}
    trackman_hgb_preds = {year: v6_oof[str(year)]["components"]["trackman_hgb"] for year in (2022, 2023, 2024)}
    te_trackman_hgb_preds = {year: v6_oof[str(year)]["components"]["te_trackman_hgb"] for year in (2022, 2023, 2024)}
    
    # Evaluate each variant
    results = []
    for name, h_preds in hierarchical_preds.items():
        # Build composite tree prediction for each year
        tree_preds = {}
        for year in (2022, 2023, 2024):
            tree_preds[year] = (
                TREE_WEIGHTS["extra_trees"] * extra_trees_preds[year]
                + TREE_WEIGHTS["trackman_hgb"] * trackman_hgb_preds[year]
                + TREE_WEIGHTS["te_trackman_hgb"] * te_trackman_hgb_preds[year]
                + TREE_WEIGHTS["hierarchical_hgb"] * h_preds[year]
            )
            
        # 2022 raw score
        v17_22 = 0.95 * tree_preds[2022] + 0.05 * logistic_preds["2022"]
        pred22_raw = W_V17 * v17_22 + W_FORM * form_preds["2022"] + W_CONTEXT * context_preds["2022"]
        score22 = float(brier_score_loss(targets[2022], pred22_raw))
        
        # 2023 expanding calib
        score23, pred23_raw = fold_score_full(
            tree_preds, logistic_preds, form_preds, context_preds,
            raw_frame, targets, [2022], 2023,
        )
        # 2024 expanding calib
        score24, pred24_raw = fold_score_full(
            tree_preds, logistic_preds, form_preds, context_preds,
            raw_frame, targets, [2022, 2023], 2024,
        )
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        results.append({
            "variant": name,
            "config": HGB_CONFIGS[name],
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
        "experiment": "V43_base_hierarchical_hgb_regularization",
        "description": "Evaluate gentle LR + regularization on base hierarchical_hgb (35.45% of base trees)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v43_base_hgb_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
