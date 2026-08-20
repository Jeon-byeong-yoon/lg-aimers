"""Regularize V31 HGB components without changing features or calibration."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v19_blend import fold_score
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
CONFIGS = {
    "balanced": {
        "max_leaf_nodes": 23, "min_samples_leaf": 150,
        "l2_regularization": 10.0, "learning_rate": 0.05, "max_iter": 220,
    },
    "strong": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200,
        "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240,
    },
    "wide_stable": {
        "max_leaf_nodes": 31, "min_samples_leaf": 200,
        "l2_regularization": 10.0, "learning_rate": 0.05, "max_iter": 220,
    },
}


def v31_form_columns(frame):
    return [column for column in frame if not (
        column.startswith("tm_") and column.endswith("_std")
    )]


def v31_context_columns(frame):
    return [column for column in frame if column not in MATCHUP_HTE]


def train_predict(features, columns, config, train_mask, valid_mask, target):
    candidate = features[columns]
    model, columns = hist_gbdt_pipeline(candidate)
    model.set_params(**{
        f"histgradientboostingclassifier__{key}": value
        for key, value in config.items()
    })
    model.fit(candidate.loc[train_mask, columns], target.loc[train_mask])
    return model.predict_proba(candidate.loc[valid_mask, columns])[:, 1]


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)
    predictions = {
        name: {"form": {}, "context": {}} for name in CONFIGS
    }
    for year in (2022, 2023, 2024):
        train_mask, valid_mask = data["season"] < year, data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        context_features = select_v2_features(add_row_features(hierarchical, prior))
        context_features = add_trackman_features(context_features, trackman)
        context_features = add_context_trackman_features(context_features, context_trackman)
        for name, config in CONFIGS.items():
            predictions[name]["form"][str(year)] = train_predict(
                form_features, v31_form_columns(form_features), config,
                train_mask, valid_mask, y,
            )
            predictions[name]["context"][str(year)] = train_predict(
                context_features, v31_context_columns(context_features), config,
                train_mask, valid_mask, y,
            )
            print(year, name, flush=True)
    joblib.dump(
        predictions, "artifacts/v35_hgb_regularization_predictions.joblib", compress=3
    )

    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    base_form = v31["no_trackman_std"]["form"]
    base_context = v31["no_matchup_hte"]["context"]
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    choices = ["v31"] + list(CONFIGS)
    item22 = oof["2022"]
    base22_prediction = (
        0.75 * v17_prediction(item22, logistic["2022"])
        + 0.16 * base_form["2022"] + 0.09 * base_context["2022"]
    )
    baseline = {
        2022: float(brier_score_loss(item22["target"], base22_prediction)),
        2023: 0.25326829650157456,
        2024: 0.24783690211517923,
    }
    candidates = []
    for form_name in choices:
        for context_name in choices:
            if form_name == context_name == "v31":
                continue
            form = base_form if form_name == "v31" else predictions[form_name]["form"]
            context = (
                base_context if context_name == "v31"
                else predictions[context_name]["context"]
            )
            component = {
                year: (0.16 * form[year] + 0.09 * context[year]) / 0.25
                for year in form
            }
            scores = {
                2023: fold_score(oof, logistic, component, data, [2022], 2023, 0.25),
                2024: fold_score(
                    oof, logistic, component, data, [2022, 2023], 2024, 0.25
                ),
            }
            prediction22 = (
                0.75 * v17_prediction(item22, logistic["2022"])
                + 0.16 * form["2022"] + 0.09 * context["2022"]
            )
            scores[2022] = float(brier_score_loss(item22["target"], prediction22))
            item = {"form": form_name, "context": context_name}
            for year in (2022, 2023, 2024):
                item[f"{year}_brier"] = scores[year]
                item[f"{year}_gain_vs_v31"] = baseline[year] - scores[year]
            candidates.append(item)
    accepted = [item for item in candidates if min(
        item["2022_gain_vs_v31"], item["2023_gain_vs_v31"],
        item["2024_gain_vs_v31"],
    ) > 0]
    rank = lambda item: (
        item["2024_gain_vs_v31"], item["2023_gain_vs_v31"],
        item["2022_gain_vs_v31"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v31": {str(k): v for k, v in baseline.items()},
        "configs": CONFIGS,
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_overall": candidates[:10],
    }
    Path("artifacts/v35_hgb_regularization_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
