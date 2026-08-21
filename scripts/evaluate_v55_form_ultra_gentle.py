"""V55: Evaluate Ultra-Gentle learning rates on Form HGB for V41 3-way blend (32% weight).

Hypothesis:
-----------
In V38, lowering learning rate (0.05 -> 0.03) with iter=500 boosted Public Score by +2.89 points.
Back then, Form HGB had only 16% weight. Now in V41, Form HGB has 32% weight (2x larger).
Under this 32% weight structure, testing even gentler learning rates:
- lr=0.025, iter=600
- lr=0.020, iter=750
- lr=0.015, iter=1000
- lr=0.012, iter=1200
- lr=0.030, iter=500 with l2=25

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib).
  Threshold: 2022 > 0, 2023 > 0, 2024 gain > 5e-6 vs V41 baseline.
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
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


# V41 constants
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

FORM_CANDIDATES = {
    "ultra_gentle_600": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0,
        "learning_rate": 0.025, "max_iter": 600, "random_state": 42
    },
    "ultra_gentle_750": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0,
        "learning_rate": 0.020, "max_iter": 750, "random_state": 42
    },
    "ultra_gentle_1000": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0,
        "learning_rate": 0.015, "max_iter": 1000, "random_state": 42
    },
    "ultra_gentle_1200": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0,
        "learning_rate": 0.012, "max_iter": 1200, "random_state": 42
    },
    "gentle_500_l2_25": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 25.0,
        "learning_rate": 0.030, "max_iter": 500, "random_state": 42
    },
}


def main():
    print("Loading precomputed OOF artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    trackman = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    v11_tree_preds = {str(yr): v11_prediction(oof[str(yr)]) for yr in (2022, 2023, 2024)}
    v17_preds = {str(yr): 0.95 * v11_tree_preds[str(yr)] + 0.05 * logistic[str(yr)] for yr in (2022, 2023, 2024)}
    
    form_raw = add_stable_form_features(hierarchical)
    
    # Pre-build form features for each season
    form_features_by_year = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_base = select_v2_features(add_row_features(form_raw, prior))
        form_tm = add_trackman_features(form_base, trackman)
        form_cols = [c for c in form_tm.columns if not (c.startswith("tm_") and c.endswith("_std"))]
        form_features_by_year[year] = (form_tm, form_cols, train_mask, valid_mask)
    
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    results = []
    
    for name, config in FORM_CANDIDATES.items():
        print(f"Evaluating Form candidate: {name} (lr={config['learning_rate']}, iter={config['max_iter']})...", flush=True)
        form_preds = {}
        for year in (2022, 2023, 2024):
            form_tm, form_cols, train_mask, valid_mask = form_features_by_year[year]
            pipe, cols = hist_gbdt_pipeline(form_tm[form_cols])
            pipe.set_params(**{f"histgradientboostingclassifier__{k}": v for k, v in config.items()})
            pipe.fit(form_tm.loc[train_mask, form_cols], y.loc[train_mask])
            form_preds[str(year)] = pipe.predict_proba(form_tm.loc[valid_mask, form_cols])[:, 1]
            
        raw_blend = {
            str(yr): W_V17 * v17_preds[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        
        score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
        
        # 2023 calib
        res22 = targets["2022"] - raw_blend["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # 2024 calib
        res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        results.append({
            "variant": name,
            "config": config,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        print(f"  -> gains: 2022={gain22:+.8f}, 2023={gain23:+.8f}, 2024={gain24:+.8f}", flush=True)
        
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V55_form_ultra_gentle_learning_rates",
        "description": "Evaluate ultra-gentle learning rates (0.012~0.025) on Form HGB with V41 32% weight",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v55_form_ultra_gentle_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n--- Summary ---", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
