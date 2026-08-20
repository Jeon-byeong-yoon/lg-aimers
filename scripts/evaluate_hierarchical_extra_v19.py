"""Evaluate hierarchical Trackman ExtraTrees variants for V17 diversity."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from evaluate_v2 import score
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


VARIANTS = {
    "balanced": {"max_depth": 16, "min_samples_leaf": 100, "max_features": 0.7},
    "stable": {"max_depth": 12, "min_samples_leaf": 300, "max_features": 1.0},
}


def pipeline(frame, parameters):
    categorical = [column for column in LOW_CARDINAL_CATEGORICAL if column in frame]
    numeric = [column for column in frame if column not in categorical]
    model = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ), categorical),
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ]),
        ExtraTreesClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=20260819,
            **parameters,
        ),
    )
    return model, list(frame.columns)


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
    for name, parameters in VARIANTS.items():
        variant_result, variant_prediction = {}, {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            features = select_v2_features(add_row_features(hierarchical, prior))
            features = add_trackman_features(features, trackman)
            model, columns = pipeline(features, parameters)
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
            variant_result[str(year)] = score(y.loc[valid_mask], prediction)
            variant_prediction[str(year)] = prediction
            print(name, year, variant_result[str(year)], flush=True)
        results[name] = variant_result
        predictions[name] = variant_prediction
    Path("artifacts/v19_hierarchical_extra_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(predictions, "artifacts/v19_hierarchical_extra_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
