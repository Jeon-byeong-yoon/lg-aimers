"""Refine the positive V35 form-only HGB regularization candidate."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v19_blend import fold_score
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from evaluate_v35_hgb_regularization import train_predict, v31_form_columns


CONFIGS = {
    "leaf11": {
        "max_leaf_nodes": 11, "min_samples_leaf": 200,
        "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240,
    },
    "leaf19": {
        "max_leaf_nodes": 19, "min_samples_leaf": 200,
        "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240,
    },
    "min150": {
        "max_leaf_nodes": 15, "min_samples_leaf": 150,
        "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240,
    },
    "min250": {
        "max_leaf_nodes": 15, "min_samples_leaf": 250,
        "l2_regularization": 20.0, "learning_rate": 0.05, "max_iter": 240,
    },
    "l2_30": {
        "max_leaf_nodes": 15, "min_samples_leaf": 200,
        "l2_regularization": 30.0, "learning_rate": 0.05, "max_iter": 240,
    },
}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    predictions = {name: {} for name in CONFIGS}
    for year in (2022, 2023, 2024):
        train_mask, valid_mask = data["season"] < year, data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        features = select_v2_features(add_row_features(form_raw, prior))
        features = add_trackman_features(features, trackman)
        columns = v31_form_columns(features)
        for name, config in CONFIGS.items():
            predictions[name][str(year)] = train_predict(
                features, columns, config, train_mask, valid_mask, y
            )
            print(year, name, flush=True)
    joblib.dump(predictions, "artifacts/v36_form_refinement_predictions.joblib", compress=3)

    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = v31["no_matchup_hte"]["context"]
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    baseline = {2023: 0.25326829650157456, 2024: 0.24783690211517923}
    results = []
    for name, form in predictions.items():
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
        results.append({
            "variant": name, "config": CONFIGS[name],
            "2023_brier": scores[2023], "2024_brier": scores[2024],
            "2023_gain_vs_v31": baseline[2023] - scores[2023],
            "2024_gain_vs_v31": baseline[2024] - scores[2024],
        })
    accepted = [item for item in results if min(
        item["2023_gain_vs_v31"], item["2024_gain_vs_v31"]
    ) > 0]
    rank = lambda item: (item["2024_gain_vs_v31"], item["2023_gain_vs_v31"])
    accepted.sort(key=rank, reverse=True)
    results.sort(key=rank, reverse=True)
    output = {
        "baseline_v31": {str(k): v for k, v in baseline.items()},
        "candidate_count": len(results), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "all": results,
    }
    Path("artifacts/v36_form_refinement_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
