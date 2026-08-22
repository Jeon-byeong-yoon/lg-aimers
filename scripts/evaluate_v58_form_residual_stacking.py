"""V58: Sequential Residual Boosting of Form features on top of V17 predictions.

Hypothesis:
-----------
Currently, Form HGB is trained independently on the binary target y,
and then blended as: 0.55 * V17 + 0.32 * Form + 0.13 * Context.

What if Form HGB directly fits the residual of V17:
  residual_i = y_i - V17_pred_i
using HistGradientBoostingRegressor(loss='squared_error', max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=20.0)?

Then:
  v17_form_pred = V17_pred + w_res * residual_form_pred

This matches the foundational theory of gradient boosting (fitting negative gradient of MSE).

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib) with V41 calibration.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


# V41 baseline constants
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

FORM_REG_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.03,
    "max_iter": 500,
    "random_state": 42,
}


def build_hgb_regressor(frame, config):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical and not c.endswith("_id") and c != "row_id"]
    pipe = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]),
        HistGradientBoostingRegressor(**config, loss="squared_error"),
    )
    return pipe, categorical + numeric


def main():
    print("Loading data...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv")
    trackman = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    v11_tree_preds = {str(yr): v11_prediction(oof[str(yr)]) for yr in (2022, 2023, 2024)}
    v17_preds = {str(yr): 0.95 * v11_tree_preds[str(yr)] + 0.05 * logistic[str(yr)] for yr in (2022, 2023, 2024)}
    
    # Form features
    form_raw = add_stable_form_features(hierarchical)
    
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    # We test two residual configurations:
    # 1. Standard Residual Regressor on (y - v17_pred)
    # 2. Context + Residual Form
    res_form_preds = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        
        form_base = select_v2_features(add_row_features(form_raw, prior))
        form_tm = add_trackman_features(form_base, trackman)
        form_cols = [c for c in form_tm.columns if not (c.startswith("tm_") and c.endswith("_std"))]
        
        # Target for regressor: residual on train seasons
        # For training before year: we use train OOF or in-sample V17
        # To maintain strict temporal validity:
        train_indices = data.loc[train_mask].index
        # Get train targets and train v17 preds
        train_targets = []
        train_v17 = []
        for yr in sorted([y_i for y_i in (2022, 2023, 2024) if y_i < year]):
            train_targets.append(targets[str(yr)])
            train_v17.append(v17_preds[str(yr)])
            
        if not train_targets:
            # For 2022 (history before 2022 is 2020/2021 where we don't have separate OOF):
            # fallback to 0
            res_form_preds[str(year)] = np.zeros(valid_mask.sum())
            continue
            
        y_res_train = np.concatenate(train_targets) - np.concatenate(train_v17)
        train_submask = data["season"].isin([y_i for y_i in (2022, 2023, 2024) if y_i < year])
        
        pipe, cols = build_hgb_regressor(form_tm[form_cols], FORM_REG_CONFIG)
        pipe.fit(form_tm.loc[train_submask, cols], y_res_train)
        res_form_preds[str(year)] = pipe.predict(form_tm.loc[valid_mask, cols])
        
    candidates = []
    # Test residual shrinkage weight w_res in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for w_res in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        v17_plus_res = {
            str(yr): (0.87 * (v17_preds[str(yr)] + w_res * res_form_preds[str(yr)]) + 0.13 * context_preds[str(yr)])
            for yr in (2022, 2023, 2024)
        }
        
        score22 = float(brier_score_loss(targets["2022"], v17_plus_res["2022"]))
        
        # 2023 calib
        res22 = targets["2022"] - v17_plus_res["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(v17_plus_res["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # 2024 calib
        res_22_23 = np.concatenate([res22, targets["2023"] - v17_plus_res["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(v17_plus_res["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        candidates.append({
            "w_res": w_res,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        
    accepted = [c for c in candidates if c["all_improved"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    rank = lambda item: (item["gains_vs_v41"]["2024"], item["gains_vs_v41"]["2023"], item["gains_vs_v41"]["2022"])
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V58_form_residual_stacking",
        "description": "Sequential Residual Boosting: Form HGB fits V17 prediction residuals",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": candidates,
    }
    
    Path("artifacts/v58_form_residual_stacking_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
