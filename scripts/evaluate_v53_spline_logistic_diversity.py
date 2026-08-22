"""Fast evaluation of Spline Logistic & Ridge linear diversity on top of V41 (lbfgs only)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}


def spline_logistic(frame, n_knots=4, C=0.3):
    categorical_candidates = LOW_CARDINAL_CATEGORICAL + ["game_month", "pitcher_team_id", "batter_team_id"]
    categorical = [c for c in categorical_candidates if c in frame]
    numeric = [c for c in frame if c not in categorical and c not in {"pitcher_id", "batter_id", "season"}]
    pipe = make_pipeline(
        ColumnTransformer([
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SplineTransformer(n_knots=n_knots, degree=2, extrapolation="linear")), numeric),
        ]),
        LogisticRegression(C=C, max_iter=100, solver="lbfgs", random_state=42),
    )
    return pipe, categorical + numeric


def standard_logistic_tuned(frame, C=0.1):
    categorical_candidates = LOW_CARDINAL_CATEGORICAL + ["game_month", "pitcher_team_id", "batter_team_id"]
    categorical = [c for c in categorical_candidates if c in frame]
    numeric = [c for c in frame if c not in categorical and c not in {"pitcher_id", "batter_id", "season"}]
    pipe = make_pipeline(
        ColumnTransformer([
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ]),
        LogisticRegression(C=C, max_iter=100, solver="lbfgs", random_state=42),
    )
    return pipe, categorical + numeric


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    trackman = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    v11_tree_preds = {str(yr): v11_prediction(oof[str(yr)]) for yr in (2022, 2023, 2024)}
    
    models = {
        "spline_k4_deg2_c03": lambda f: spline_logistic(f, n_knots=4, C=0.3),
        "spline_k3_deg2_c01": lambda f: spline_logistic(f, n_knots=3, C=0.1),
        "logistic_c01": lambda f: standard_logistic_tuned(f, C=0.1),
        "logistic_c005": lambda f: standard_logistic_tuned(f, C=0.05),
    }
    
    lin_preds = {name: {} for name in models}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(hierarchical, prior))
        features = add_trackman_features(features, trackman)
        
        for name, pipe_fn in models.items():
            pipe, cols = pipe_fn(features)
            pipe.fit(features.loc[train_mask, cols], y.loc[train_mask])
            lin_preds[name][str(year)] = pipe.predict_proba(features.loc[valid_mask, cols])[:, 1]

    candidates = []
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    for name, p_dict in lin_preds.items():
        for w_log in [0.03, 0.05, 0.08]:
            w_trees = 1.0 - w_log
            v17_variant = {str(yr): w_trees * v11_tree_preds[str(yr)] + w_log * p_dict[str(yr)] for yr in (2022, 2023, 2024)}
            raw_blend = {str(yr): W_V17 * v17_variant[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds[str(yr)] for yr in (2022, 2023, 2024)}
            
            score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
            
            res22 = targets["2022"] - raw_blend["2022"]
            cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
            pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
            pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
            score23 = float(brier_score_loss(targets["2023"], pred23_cal))
            
            res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
            cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
            pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
            pred24_cal = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
            score24 = float(brier_score_loss(targets["2024"], pred24_cal))
            
            gain22 = V41_BASELINE[2022] - score22
            gain23 = V41_BASELINE[2023] - score23
            gain24 = V41_BASELINE[2024] - score24
            
            candidates.append({
                "model": name,
                "w_linear": w_log,
                "scores": {"2022": score22, "2023": score23, "2024": score24},
                "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
                "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
                "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
            })
            
    accepted = [c for c in candidates if c["all_improved"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    rank = lambda item: (item["gains_vs_v41"]["2024"], item["gains_vs_v41"]["2023"])
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V53_spline_logistic_diversity",
        "description": "Evaluate fast Spline (GAM) and tuned L-BFGS linear diversity models on V41",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": candidates,
    }
    
    Path("artifacts/v53_spline_logistic_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
