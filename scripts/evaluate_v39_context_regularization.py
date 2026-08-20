"""V39: Apply regularization + lower LR to context HGB — genuinely unexplored.

V38 applied gentle_500 (lr=0.03, iter=500, max_leaf=15, min_samples=200, l2=20)
to the form model and achieved Public +2.89 over V31.

The context model has NEVER had regularization or LR tuning applied —
it still uses default HGB params (max_leaf=31, min_samples=100, l2=5, lr=0.06).

This experiment applies the same treatment independently to the context model
while keeping form=V38-gentle_500 fixed.

Context variants tested (form is always V38-gentle_500):
  v38_base     : V31 original context (no_matchup_hte) — V38 baseline
  ctx_strong   : max_leaf=15, min_samples=200, l2=20, lr=0.05, iter=240  (V35-strong style)
  ctx_g400     : max_leaf=15, min_samples=200, l2=20, lr=0.03, iter=400
  ctx_g500     : max_leaf=15, min_samples=200, l2=20, lr=0.03, iter=500  (mirror of form)
  ctx_g600     : max_leaf=15, min_samples=200, l2=20, lr=0.03, iter=600
  ctx_wide500  : max_leaf=23, min_samples=150, l2=10, lr=0.03, iter=500  (balanced-style)

V38 baseline scores:
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
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

CTX_CONFIGS = {
    "ctx_strong":  {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240},
    "ctx_g400":    {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 400},
    "ctx_g500":    {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500},
    "ctx_g600":    {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 600},
    "ctx_wide500": {"max_leaf_nodes": 23, "min_samples_leaf": 150, "l2_regularization": 10.0, "learning_rate": 0.03, "max_iter": 500},
}

# V38 baseline (form=gentle_500, context=v31)
V38_BASELINE = {
    2022: 0.243390002346232,
    2023: 0.25324438180845804,
    2024: 0.24783152211261955,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def v31_context_columns(frame):
    return [c for c in frame if c not in MATCHUP_HTE]


def fold_score_v39(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
    """Same calibration structure as V31/V38."""
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

    # Form predictions: V38 gentle_500 (fixed)
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]

    # V31 context OOF (baseline — no regularization applied)
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    v31_context = v31_removed["no_matchup_hte"]["context"]

    # Train context variants per year
    ctx_predictions = {name: {} for name in CTX_CONFIGS}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        ctx_features = select_v2_features(add_row_features(hierarchical, prior))
        ctx_features = add_trackman_features(ctx_features, trackman)
        ctx_features = add_context_trackman_features(ctx_features, context_trackman)
        columns = v31_context_columns(ctx_features)
        candidate = ctx_features[columns]
        for name, config in CTX_CONFIGS.items():
            model, cols = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in config.items()
            })
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            ctx_predictions[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, cols]
            )[:, 1]
        print(f"context year={year} all variants done", flush=True)

    joblib.dump(ctx_predictions, "artifacts/v39_context_regularization_predictions.joblib", compress=3)

    # Evaluate: v38_base (V31 context) + each new context variant
    results = []

    # V38 baseline (v31 context, already known)
    item22 = oof["2022"]
    v17_22 = v17_prediction(item22, logistic["2022"])
    pred22_base = 0.75 * v17_22 + 0.16 * form_preds["2022"] + 0.09 * v31_context["2022"]
    score22_base = float(brier_score_loss(item22["target"].astype(float), pred22_base))
    results.append({
        "variant": "v38_base",
        "config": "V31 original context (no_matchup_hte)",
        "scores": {"2022": score22_base,
                   "2023": V38_BASELINE[2023], "2024": V38_BASELINE[2024]},
        "gains_vs_v38": {"2022": 0.0, "2023": 0.0, "2024": 0.0},
        "all_improved": False, "submit_ready": False,
    })

    for name, ctx_preds in ctx_predictions.items():
        # 2022 raw OOF
        pred22 = 0.75 * v17_22 + 0.16 * form_preds["2022"] + 0.09 * ctx_preds["2022"]
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22))
        score23 = fold_score_v39(
            oof, logistic, form_preds, ctx_preds, raw_frame,
            history_years=[2022], valid_year=2023,
        )
        score24 = fold_score_v39(
            oof, logistic, form_preds, ctx_preds, raw_frame,
            history_years=[2022, 2023], valid_year=2024,
        )
        scores = {2022: score22, 2023: score23, 2024: score24}
        gains = {y: V38_BASELINE[y] - scores[y] for y in (2022, 2023, 2024)}
        accepted = all(gains[y] > 0 for y in (2022, 2023, 2024))
        submit_ready = accepted and gains[2024] > 5e-6
        results.append({
            "variant": name,
            "config": CTX_CONFIGS[name],
            "scores": {str(k): v for k, v in scores.items()},
            "gains_vs_v38": {str(k): v for k, v in gains.items()},
            "all_improved": accepted,
            "submit_ready": submit_ready,
        })

    results.sort(
        key=lambda x: (x["gains_vs_v38"].get("2024", 0), x["gains_vs_v38"].get("2023", 0)),
        reverse=True,
    )
    accepted = [r for r in results if r.get("all_improved")]
    submit_ready = [r for r in results if r.get("submit_ready")]

    output = {
        "experiment": "V39_context_regularization",
        "description": "form=V38-gentle_500 fixed; context HGB regularization+LR grid",
        "baseline_v38": {str(k): v for k, v in V38_BASELINE.items()},
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    Path("artifacts/v39_context_regularization_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
