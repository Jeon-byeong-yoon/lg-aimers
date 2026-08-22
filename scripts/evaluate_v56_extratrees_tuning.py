"""V56: Fine-tuning ExtraTreesClassifier (Base Layer 29.5% weight).

Background:
-----------
In V11 tree ensemble, ExtraTrees carries a massive 29.48% weight.
However, ExtraTrees is still using the ancient V2 default hyperparameters:
  n_estimators=150, max_depth=14, min_samples_leaf=100, max_features=0.8

While all HGBs were heavily refined (gentle_500, l2, etc.), ExtraTrees was untouched!
Tuning:
- n_estimators: 150 -> 300 / 400 (more trees = lower variance with zero overfitting)
- max_features: 0.8 -> 0.5, 0.6, 0.7 (more feature randomness = stronger bagging diversity)
- min_samples_leaf: 100 -> 50, 75, 120, 150

Evaluation:
  Full V41 expanding calibration pipeline across 2022 raw OOF, 2023 calib, 2024 calib.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from evaluate_residual_ridge_v13 import COMPONENTS
from evaluate_segment_calibration_v12 import segment_correction
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings


# V41 constants
TREE_WEIGHTS = {
    "extra_trees": 0.29483562599237795,
    "trackman_hgb": 0.2344456574665245,
    "te_trackman_hgb": 0.11617615437521091,
    "hierarchical_hgb": 0.35454256216588664,
}
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

ET_VARIANTS = {
    "et_n300_feat06": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 100, "max_features": 0.6},
    "et_n300_feat07": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 100, "max_features": 0.7},
    "et_n300_leaf75_feat06": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 75, "max_features": 0.6},
    "et_n300_leaf120_feat06": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 120, "max_features": 0.6},
    "et_n400_feat06": {"n_estimators": 400, "max_depth": 14, "min_samples_leaf": 100, "max_features": 0.6},
}


def build_et_pipeline(frame, params):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical]
    pipe = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]),
        ExtraTreesClassifier(**params, n_jobs=-1, random_state=42),
    )
    return pipe, list(frame.columns)


def main():
    print("Loading data and precomputed artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    
    # Pre-extract other 3 base HGB components from existing OOF
    other_hgb_preds = {}
    for yr in (2022, 2023, 2024):
        item = oof[str(yr)]
        other_hgb_preds[str(yr)] = (
            TREE_WEIGHTS["trackman_hgb"] * item["components"]["trackman_hgb"] +
            TREE_WEIGHTS["te_trackman_hgb"] * item["components"]["te_trackman_hgb"] +
            TREE_WEIGHTS["hierarchical_hgb"] * item["components"]["hierarchical_hgb"]
        )
        
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    # Pre-build base features
    base_features = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        base = select_v2_features(add_row_features(data, prior))
        base_features[year] = (base, train_mask, valid_mask)
        
    results = []
    
    for name, params in ET_VARIANTS.items():
        print(f"Training ExtraTrees variant: {name}...", flush=True)
        et_preds = {}
        for year in (2022, 2023, 2024):
            base, train_mask, valid_mask = base_features[year]
            pipe, cols = build_et_pipeline(base, params)
            pipe.fit(base.loc[train_mask, cols], y.loc[train_mask])
            et_preds[str(year)] = pipe.predict_proba(base.loc[valid_mask, cols])[:, 1]
            
        # Blend: V11_trees = w_et * new_et + other_hgbs
        v11_variant = {
            str(yr): TREE_WEIGHTS["extra_trees"] * et_preds[str(yr)] + other_hgb_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        v17_variant = {
            str(yr): 0.95 * v11_variant[str(yr)] + 0.05 * logistic[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        raw_blend = {
            str(yr): W_V17 * v17_variant[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds[str(yr)]
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
            "params": params,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        print(f"  -> {name}: gains 2022={gain22:+.8f}, 2023={gain23:+.8f}, 2024={gain24:+.8f}", flush=True)
        
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V56_extratrees_hyperparameter_tuning",
        "description": "Fine-tune ExtraTrees in Base Layer (29.5% weight): n_estimators, max_features, min_samples_leaf",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v56_extratrees_tuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n--- Summary ---", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
