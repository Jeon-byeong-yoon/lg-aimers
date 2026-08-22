"""Evaluate exponentially decayed chronological Target Encodings on top of V41.

Hypothesis & Mathematics:
-------------------------
In standard target encoding (V5/V6):
  smoothed_rate = (sum(target) + prior * m) / (count + m)
where all prior seasons are weighted equally (1.0).

In baseball, recent seasons (e.g. 1 season ago) are more predictive of present control
than older seasons (e.g. 3-4 seasons ago) due to aging curves and mechanics changes.

With exponential decay rate gamma in (0, 1]:
  For a row in season T, a historical row in season s < T gets weight:
    w = gamma ** (T - s - 1)
  (i.e. s = T-1 has w=1.0, s = T-2 has w=gamma, s = T-3 has w=gamma^2, etc.)

  weighted_sum = sum(target * w)
  weighted_count = sum(w)
  smoothed_rate = (weighted_sum + prior * m) / (weighted_count + m)
  log_count = log1p(weighted_count)

Decay rates tested:
  gamma in [1.0 (baseline), 0.90, 0.80, 0.70, 0.50]

Evaluation:
  Re-train Form HGB (gentle_500) and Context HGB (V31) with decayed encodings,
  and evaluate the full V41 3-way blend (0.55/0.32/0.13) + calibration
  across 2022 raw OOF, 2023 expanding calib, and 2024 expanding calib.
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
from stable_form_features_v22 import add_stable_form_features
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

GAMMA_VALUES = [1.0, 0.90, 0.80, 0.70, 0.50]
ENTITY_COLUMNS = ["pitcher_id", "batter_id"]
STRENGTHS = (50.0, 200.0)


def add_decayed_target_encodings(frame: pd.DataFrame, target: pd.Series, gamma: float) -> pd.DataFrame:
    source = frame.copy()
    source["__target"] = target.to_numpy()
    pieces = []
    
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"] == season].drop(columns="__target").copy()
        current["__original_index"] = current.index
        history = source.loc[source["season"] < season].copy()
        
        if history.empty:
            for entity in ENTITY_COLUMNS:
                for strength in STRENGTHS:
                    current[f"te_{entity}_{int(strength)}"] = float("nan")
                current[f"te_{entity}_log_count"] = float("nan")
        else:
            # Apply exponential decay weight based on season distance
            history["__weight"] = gamma ** (season - history["season"] - 1)
            history["__weighted_target"] = history["__target"] * history["__weight"]
            
            prior = float(history["__target"].mean())
            for entity in ENTITY_COLUMNS:
                stats = history.groupby(entity, observed=True).agg({
                    "__weighted_target": "sum",
                    "__weight": "sum",
                }).reset_index()
                
                for strength in STRENGTHS:
                    stats[f"te_{entity}_{int(strength)}"] = (
                        stats["__weighted_target"] + prior * strength
                    ) / (stats["__weight"] + strength)
                stats[f"te_{entity}_log_count"] = np.log1p(stats["__weight"])
                
                lookup = stats.drop(columns=["__weighted_target", "__weight"])
                current = current.merge(lookup, how="left", on=entity, validate="many_to_one")
                
        pieces.append(current.set_index("__original_index"))
        
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    return output


def add_decayed_hierarchical_encodings(frame: pd.DataFrame, target: pd.Series, gamma: float) -> pd.DataFrame:
    source = frame.copy()
    source["__target"] = target.to_numpy()
    source["pitcher_batter"] = source["pitcher_id"].astype(str) + "_" + source["batter_id"].astype(str)
    
    pieces = []
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"] == season].drop(columns="__target").copy()
        current["__original_index"] = current.index
        history = source.loc[source["season"] < season].copy()
        
        if history.empty:
            for strength in (100.0, 500.0):
                current[f"hte_pitcher_batter_{int(strength)}"] = float("nan")
            current["hte_pitcher_batter_log_count"] = float("nan")
        else:
            history["__weight"] = gamma ** (season - history["season"] - 1)
            history["__weighted_target"] = history["__target"] * history["__weight"]
            prior = float(history["__target"].mean())
            
            stats = history.groupby("pitcher_batter", observed=True).agg({
                "__weighted_target": "sum",
                "__weight": "sum",
            }).reset_index()
            
            for strength in (100.0, 500.0):
                stats[f"hte_pitcher_batter_{int(strength)}"] = (
                    stats["__weighted_target"] + prior * strength
                ) / (stats["__weight"] + strength)
            stats["hte_pitcher_batter_log_count"] = np.log1p(stats["__weight"])
            
            lookup = stats.drop(columns=["__weighted_target", "__weight"])
            current = current.merge(lookup, how="left", on="pitcher_batter", validate="many_to_one")
            
        pieces.append(current.set_index("__original_index"))
        
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    return output.drop(columns="pitcher_batter")


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def fold_score_v50(oof, logistic, form_preds, context_preds, raw_frame,
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
    
    results = []
    for gamma in GAMMA_VALUES:
        print(f"Evaluating decay gamma={gamma}...", flush=True)
        encoded = add_decayed_target_encodings(data, y, gamma)
        hierarchical = add_decayed_hierarchical_encodings(encoded, y, gamma)
        
        # Train Form HGB with decayed target encodings per year
        form_preds_gamma = {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            
            form_raw = add_stable_form_features(hierarchical)
            form_features = select_v2_features(add_row_features(form_raw, prior))
            form_features = add_trackman_features(form_features, trackman)
            columns = v31_form_columns(form_features)
            candidate = form_features[columns]
            
            model, cols = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in FORM_HGB_CONFIG.items()
            })
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            form_preds_gamma[str(year)] = model.predict_proba(
                candidate.loc[valid_mask, cols]
            )[:, 1]
            
        pred22_raw = W_V17 * v17_22 + W_FORM * form_preds_gamma["2022"] + W_CONTEXT * context_preds["2022"]
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22_raw))
        score23 = fold_score_v50(oof, logistic, form_preds_gamma, context_preds, raw_frame, [2022], 2023)
        score24 = fold_score_v50(oof, logistic, form_preds_gamma, context_preds, raw_frame, [2022, 2023], 2024)
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        results.append({
            "gamma": gamma,
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
        print(f"gamma={gamma} gains: 2022={gain22:.8f}, 2023={gain23:.8f}, 2024={gain24:.8f}", flush=True)
        
    results.sort(
        key=lambda x: (x["gains_vs_v41"]["2024"], x["gains_vs_v41"]["2023"]),
        reverse=True,
    )
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V50_target_encoding_decay",
        "description": "Evaluate exponential time decay on historical Target Encodings in Form HGB",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v50_te_decay_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
