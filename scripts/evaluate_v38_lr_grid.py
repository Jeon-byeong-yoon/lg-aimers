"""V38: Lower learning rate grid on form HGB (genuinely unexplored axis).

V35 best used lr=0.05, max_iter=240 with strong regularization.
V36 only varied max_leaf_nodes, min_samples_leaf, l2 — learning_rate was
never changed below 0.05.

Hypothesis: a lower learning rate combined with proportionally more iterations
improves out-of-distribution generalization (especially 2024), because each
tree makes smaller corrections and the ensemble is less prone to fitting
late-season idiosyncrasies of the training set.

Grid (all share V35-strong base: max_leaf=15, min_samples=200, l2=20):
  - lr=0.04, iter=300   ("gentle_300")
  - lr=0.03, iter=400   ("gentle_400")
  - lr=0.03, iter=500   ("gentle_500")
  - lr=0.02, iter=600   ("slow_600")
  - lr=0.02, iter=800   ("slow_800")

context: V31 original (no_matchup_hte, proven best)
weights: unchanged (0.75 / 0.16 / 0.09)
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


# All share base regularization from V35-strong
BASE_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
}

LR_CONFIGS = {
    "v35_strong": {"learning_rate": 0.05, "max_iter": 240},   # V35-best (reference)
    "gentle_300": {"learning_rate": 0.04, "max_iter": 300},
    "gentle_400": {"learning_rate": 0.03, "max_iter": 400},
    "gentle_500": {"learning_rate": 0.03, "max_iter": 500},
    "slow_600":   {"learning_rate": 0.02, "max_iter": 600},
    "slow_800":   {"learning_rate": 0.02, "max_iter": 800},
}

MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

V31_BASELINE = {
    2022: 0.24339835671689145,
    2023: 0.25326829650157456,
    2024: 0.24783690211517923,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def fold_score_v38(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
    """Chronological fold with same calibration structure as V31."""
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
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]

    # V31 context predictions (no_matchup_hte) — proven best
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]  # str year keys

    # Train each LR variant's form model per year
    form_predictions = {name: {} for name in LR_CONFIGS}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        columns = v31_form_columns(form_features)
        candidate = form_features[columns]

        for name, lr_cfg in LR_CONFIGS.items():
            config = {**BASE_CONFIG, **lr_cfg}
            model, cols = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in config.items()
            })
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            form_predictions[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, cols]
            )[:, 1]
        print(f"year={year} all LR variants done", flush=True)

    # Evaluate each variant
    results = []
    for name, form_preds in form_predictions.items():
        # 2022 raw OOF Brier
        item22 = oof["2022"]
        v17_22 = v17_prediction(item22, logistic["2022"])
        pred22 = (
            0.75 * v17_22
            + 0.16 * form_preds["2022"]
            + 0.09 * context_preds["2022"]
        )
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22))
        score23 = fold_score_v38(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022], valid_year=2023,
        )
        score24 = fold_score_v38(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022, 2023], valid_year=2024,
        )
        scores = {2022: score22, 2023: score23, 2024: score24}
        gains = {y: V31_BASELINE[y] - scores[y] for y in (2022, 2023, 2024)}
        accepted = all(gains[y] > 0 for y in (2022, 2023, 2024))
        submit_ready = accepted and gains[2024] > 5e-6
        lr_cfg = LR_CONFIGS[name]
        results.append({
            "variant": name,
            "config": {**BASE_CONFIG, **lr_cfg},
            "scores": {str(k): v for k, v in scores.items()},
            "gains_vs_v31": {str(k): v for k, v in gains.items()},
            "all_seasons_improved": accepted,
            "submit_ready": submit_ready,
        })

    results.sort(
        key=lambda x: (x["gains_vs_v31"]["2024"], x["gains_vs_v31"]["2023"]),
        reverse=True,
    )
    accepted_results = [r for r in results if r["all_seasons_improved"]]
    submit_ready_results = [r for r in results if r["submit_ready"]]

    joblib.dump(
        form_predictions, "artifacts/v38_lr_grid_predictions.joblib", compress=3
    )

    output = {
        "experiment": "V38_lr_grid",
        "description": "form HGB: lower LR grid (0.02-0.04) + V31 context; unexplored axis",
        "base_config": BASE_CONFIG,
        "baseline_v31": {str(k): v for k, v in V31_BASELINE.items()},
        "accepted_count": len(accepted_results),
        "submit_ready_count": len(submit_ready_results),
        "best_accepted": accepted_results[0] if accepted_results else None,
        "best_submit_ready": submit_ready_results[0] if submit_ready_results else None,
        "all_results": results,
    }
    Path("artifacts/v38_lr_grid_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
