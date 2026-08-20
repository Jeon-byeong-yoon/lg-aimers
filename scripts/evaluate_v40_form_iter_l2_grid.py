"""V40: Extend V38 form model grid — iter extension + l2 combined with gentle LR.

V38 tested lr=0.03 with iter=400 and 500 only.
  - iter=600, 700 with lr=0.03 are genuinely untested.
  - The sweet spot might extend further.

V36 found l2=30 gave best 2023 gain (though 2024 was only +3.2e-6 at lr=0.05).
  - Combining l2=30 with lr=0.03 is genuinely untested.
  - Also l2=15 and l2=25 at gentle LR haven't been checked.

All use:
  - form: V38 base config except the varied parameter
  - context: V31 original (no_matchup_hte) — proven best
  - weights: 0.75 / 0.16 / 0.09 (unchanged)

V38 baseline (form=gentle_500, context=v31):
  2022: 0.243390002346232
  2023: 0.253244381808458
  2024: 0.247831522112620
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


MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

# V38 gentle_500 reference (baseline)
V38_REFERENCE = {
    "max_leaf_nodes": 15, "min_samples_leaf": 200,
    "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500,
}

# Candidates: extend iter (same l2=20) + tune l2 (same iter=500/lr=0.03)
FORM_CONFIGS = {
    # iter extension at lr=0.03
    "g500_ref":  {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500},  # V38 reference
    "g550":      {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 550},
    "g600":      {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 600},
    "g700":      {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 700},
    # l2 tuning at gentle_500 LR
    "l2_15_g500": {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 15.0, "learning_rate": 0.03, "max_iter": 500},
    "l2_25_g500": {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 25.0, "learning_rate": 0.03, "max_iter": 500},
    "l2_30_g500": {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 30.0, "learning_rate": 0.03, "max_iter": 500},
    "l2_30_g600": {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 30.0, "learning_rate": 0.03, "max_iter": 600},
}

# V38 baseline
V38_BASELINE = {
    2022: 0.243390002346232,
    2023: 0.25324438180845804,
    2024: 0.24783152211261955,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def fold_score_v40(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
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
    train_frame = raw_frame.loc[index]
    valid_item = oof[str(valid_year)]
    v17_valid = v17_prediction(valid_item, logistic[str(valid_year)])
    valid_pred_raw = (
        0.75 * v17_valid
        + 0.16 * form_preds[str(valid_year)]
        + 0.09 * context_preds[str(valid_year)]
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
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]

    # Context: V31 original (fixed — proven best)
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]

    # Train form variants per year
    form_predictions = {name: {} for name in FORM_CONFIGS}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        columns = v31_form_columns(form_features)
        candidate = form_features[columns]
        for name, config in FORM_CONFIGS.items():
            model, cols = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in config.items()
            })
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            form_predictions[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, cols]
            )[:, 1]
        print(f"form year={year} all variants done", flush=True)

    joblib.dump(form_predictions, "artifacts/v40_form_grid_predictions.joblib", compress=3)

    # Evaluate
    results = []
    item22 = oof["2022"]
    v17_22 = v17_prediction(item22, logistic["2022"])

    for name, form_preds in form_predictions.items():
        pred22 = 0.75 * v17_22 + 0.16 * form_preds["2022"] + 0.09 * context_preds["2022"]
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22))
        score23 = fold_score_v40(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022], valid_year=2023,
        )
        score24 = fold_score_v40(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022, 2023], valid_year=2024,
        )
        scores = {2022: score22, 2023: score23, 2024: score24}
        gains = {y: V38_BASELINE[y] - scores[y] for y in (2022, 2023, 2024)}
        accepted = all(gains[y] > 0 for y in (2022, 2023, 2024))
        submit_ready = accepted and gains[2024] > 5e-6
        results.append({
            "variant": name,
            "config": FORM_CONFIGS[name],
            "scores": {str(k): v for k, v in scores.items()},
            "gains_vs_v38": {str(k): v for k, v in gains.items()},
            "all_improved": accepted,
            "submit_ready": submit_ready,
        })

    results.sort(
        key=lambda x: (x["gains_vs_v38"]["2024"], x["gains_vs_v38"]["2023"]),
        reverse=True,
    )
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]

    output = {
        "experiment": "V40_form_iter_l2_grid",
        "description": "form: iter extension (550-700) + l2 tuning at gentle LR; context=V31",
        "baseline_v38": {str(k): v for k, v in V38_BASELINE.items()},
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    Path("artifacts/v40_form_grid_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
