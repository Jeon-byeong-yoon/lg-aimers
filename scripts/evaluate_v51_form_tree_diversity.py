"""Ultra-fast evaluation of Form ExtraTrees diversity on top of V41."""

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
from evaluate_v2 import extra_trees_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
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

W_BAGGING_VALUES = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


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
    form_hgb_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    v17_preds = {
        str(yr): 0.95 * v11_prediction(oof[str(yr)]) + 0.05 * logistic[str(yr)]
        for yr in (2022, 2023, 2024)
    }
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    
    # Fast ExtraTrees on Form features (60 trees, min_samples_leaf=200)
    form_et_preds = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        columns = v31_form_columns(form_features)
        candidate = form_features[columns]
        
        et_model, et_cols = extra_trees_pipeline(candidate)
        et_model.set_params(**{
            "extratreesclassifier__n_estimators": 60,
            "extratreesclassifier__min_samples_leaf": 200,
            "extratreesclassifier__max_features": 0.7,
            "extratreesclassifier__random_state": 42,
            "extratreesclassifier__n_jobs": -1,
        })
        et_model.fit(candidate.loc[train_mask, et_cols], y.loc[train_mask])
        form_et_preds[str(year)] = et_model.predict_proba(candidate.loc[valid_mask, et_cols])[:, 1]

    candidates = []
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    for w_bag in W_BAGGING_VALUES:
        w_hgb = 1.0 - w_bag
        composite_form = {
            str(yr): w_hgb * form_hgb_preds[str(yr)] + w_bag * form_et_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        raw_blend = {
            str(yr): W_V17 * v17_preds[str(yr)] + W_FORM * composite_form[str(yr)] + W_CONTEXT * context_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
        
        # Fast Fold 2023
        res22 = targets["2022"] - raw_blend["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # Fast Fold 2024
        res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        candidates.append({
            "model": "form_et",
            "w_bagging": w_bag,
            "w_hgb": w_hgb,
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
        "experiment": "V51_form_tree_diversity",
        "description": "Evaluate fast ExtraTrees diversity on the Form feature space",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": candidates,
    }
    
    Path("artifacts/v51_form_diversity_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
