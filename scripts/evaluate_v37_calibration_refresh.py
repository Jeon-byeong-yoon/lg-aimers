"""V37-A: Replace V31 calibration with one fitted to V35-strong form residuals.

V31 calibration was built from V25-era OOF residuals.  V35-strong form shifts
the predicted probability distribution, so recomputing the segment correction
from the new residuals may remove a systematic bias that the old calibration
can no longer correct accurately.

Experiment scope
----------------
- form    : V35-strong (max_leaf_nodes=15, min_samples_leaf=200, l2=20, lr=0.05, iter=240)
- context : V31 original (no_matchup_hte)
- weights : unchanged (0.75 / 0.16 / 0.09)
- calibration : rebuilt from V35-strong form OOF residuals (same segment structure
                as V31: count + pitcher×count with 500 / 300 min-count thresholds)

Evaluation
----------
- 2022 : raw OOF Brier (no calibration applied, same as V31 robustness check)
- 2023 : fold_score using 2022 train residuals → calibration → apply to 2023
- 2024 : fold_score using 2022+2023 train residuals → calibration → apply to 2024
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from build_v12_calibration import make_lookup
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


# V35-strong configuration (best from V35 experiment)
FORM_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.05,
    "max_iter": 240,
}

MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

# V31 baseline values
V31_BASELINE = {
    2022: 0.24339835671689145,
    2023: 0.25326829650157456,
    2024: 0.24783690211517923,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def v31_context_columns(frame):
    return [c for c in frame if c not in MATCHUP_HTE]


def build_calibration_from_residuals(oof, logistic, form_preds, context_preds, raw_frame):
    """Rebuild segment correction from all three seasons of new OOF residuals."""
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = v17_prediction(item, logistic[str(year)])
        blend = (
            0.75 * v17
            + 0.16 * form_preds[str(year)]
            + 0.09 * context_preds[str(year)]
        )
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw_frame.loc[index]
    corrections = [
        {
            "columns": ["balls_before", "strikes_before"],
            "weight": 0.75,
            "lookup": make_lookup(frame, centered, ["balls_before", "strikes_before"], 500),
        },
        {
            "columns": ["pitcher_id", "balls_before", "strikes_before"],
            "weight": 0.25,
            "lookup": make_lookup(
                frame, centered,
                ["pitcher_id", "balls_before", "strikes_before"], 300,
            ),
        },
    ]
    return {"global_shift": shift, "corrections": corrections}


def apply_calibration(base_p, frame, calibration):
    shift = calibration["global_shift"]
    correction = np.zeros(len(frame))
    for spec in calibration["corrections"]:
        correction += spec["weight"] * segment_correction(
            frame, np.zeros(len(frame)),  # dummy; use lookup path
            frame, spec["columns"], 0,
        )
    # Re-derive correction properly using the stored lookup
    correction = np.zeros(len(frame))
    for spec in calibration["corrections"]:
        lookup = spec["lookup"]
        key = spec["columns"]
        merged = frame[key].copy().reset_index(drop=True)
        merged = merged.merge(lookup.rename(columns={"correction": "__corr"}),
                              on=key, how="left")
        correction += spec["weight"] * merged["__corr"].fillna(0.0).values
    return np.clip(base_p + shift + correction, 0.0, 1.0)


def fold_score_v37(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
    """Chronological fold score with freshly built calibration per fold."""
    # Build calibration from history years only
    indices, targets, predictions = [], [], []
    for year in history_years:
        item = oof[str(year)]
        v17 = v17_prediction(item, logistic[str(year)])
        blend = (
            0.75 * v17
            + 0.16 * form_preds[str(year)]
            + 0.09 * context_preds[str(year)]
        )
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    train_frame = raw_frame.loc[index]
    # Valid prediction
    valid_item = oof[str(valid_year)]
    v17_valid = v17_prediction(valid_item, logistic[str(valid_year)])
    valid_pred_raw = (
        0.75 * v17_valid
        + 0.16 * form_preds[str(valid_year)]
        + 0.09 * context_preds[str(valid_year)]
    )
    valid_frame = raw_frame.loc[valid_item["row_index"]]
    valid_y = valid_item["target"].astype(float)
    # Apply same segment correction structure as V31
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
    context_trackman = prepare_context_trackman(trackman)

    # Train V35-strong form model per year
    form_preds = {}
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
            f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
        })
        model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
        form_preds[str(year)] = model.predict_proba(candidate.loc[valid_mask, cols])[:, 1]
        print(f"form year={year} done", flush=True)

    # Load V31 context predictions (no_matchup_hte)
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]  # keys are str years

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]

    # 2022: raw OOF (no calibration), same as V31 robustness check
    item22 = oof["2022"]
    v17_22 = v17_prediction(item22, logistic["2022"])
    pred22 = (
        0.75 * v17_22
        + 0.16 * form_preds["2022"]
        + 0.09 * context_preds["2022"]
    )
    score22 = float(brier_score_loss(item22["target"].astype(float), pred22))

    # 2023, 2024: chronological calibration folds
    score23 = fold_score_v37(
        oof, logistic, form_preds, context_preds, raw_frame,
        history_years=[2022], valid_year=2023,
    )
    score24 = fold_score_v37(
        oof, logistic, form_preds, context_preds, raw_frame,
        history_years=[2022, 2023], valid_year=2024,
    )

    scores = {2022: score22, 2023: score23, 2024: score24}
    gains = {year: V31_BASELINE[year] - scores[year] for year in (2022, 2023, 2024)}

    accepted = all(gains[y] > 0 for y in (2022, 2023, 2024))
    submit_ready = accepted and gains[2024] > 5e-6

    output = {
        "experiment": "V37a_calibration_refresh",
        "description": "V35-strong form + refreshed calibration residuals; context=V31",
        "form_config": FORM_CONFIG,
        "baseline_v31": {str(k): v for k, v in V31_BASELINE.items()},
        "scores": {str(k): v for k, v in scores.items()},
        "gains_vs_v31": {str(k): v for k, v in gains.items()},
        "all_seasons_improved": accepted,
        "submit_ready_2024_gain_gt_5e6": submit_ready,
    }
    Path("artifacts/v37_calibration_refresh_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
