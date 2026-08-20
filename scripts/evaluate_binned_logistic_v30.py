"""Evaluate strongly regularized binned Logistic models for V25 diversity."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


VARIANTS = [(8, 0.003), (8, 0.01)]


def pipeline(frame, bins, c):
    categorical_candidates = LOW_CARDINAL_CATEGORICAL + [
        "game_month", "pitcher_team_id", "batter_team_id"
    ]
    categorical = [column for column in categorical_candidates if column in frame]
    dropped = {"pitcher_id", "batter_id", "season"}
    numeric = [column for column in frame if column not in categorical and column not in dropped]
    model = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ), categorical),
            ("binned", make_pipeline(
                SimpleImputer(strategy="median"),
                KBinsDiscretizer(
                    n_bins=bins, encode="onehot", strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
            ), numeric),
        ]),
        LogisticRegression(C=c, max_iter=150, solver="saga", tol=1e-3),
    )
    return model, categorical + numeric


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    results, predictions = {}, {}
    for bins, c in VARIANTS:
        name = f"bins{bins}_c{c}"
        fold_results, fold_predictions = {}, {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            features = select_v2_features(add_row_features(hierarchical, prior))
            features = add_trackman_features(features, trackman)
            model, columns = pipeline(features, bins, c)
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
            fold_results[str(year)] = {
                "brier": float(brier_score_loss(y.loc[valid_mask], prediction)),
                "prediction_mean": float(prediction.mean()),
            }
            fold_predictions[str(year)] = prediction
            print(name, year, fold_results[str(year)], flush=True)
        results[name], predictions[name] = fold_results, fold_predictions
    Path("artifacts/v30_binned_logistic_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(predictions, "artifacts/v30_binned_logistic_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
