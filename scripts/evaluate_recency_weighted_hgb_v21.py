"""Train hierarchical HGB with recency-weighted training loss."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


DECAYS = [0.7, 0.85]


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
    for decay in DECAYS:
        decay_results, decay_predictions = {}, {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            features = select_v2_features(add_row_features(hierarchical, prior))
            features = add_trackman_features(features, trackman)
            model, columns = hist_gbdt_pipeline(features)
            weights = decay ** (year - 1 - data.loc[train_mask, "season"].to_numpy())
            model.fit(
                features.loc[train_mask, columns], y.loc[train_mask],
                histgradientboostingclassifier__sample_weight=weights,
            )
            prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
            decay_results[str(year)] = score(y.loc[valid_mask], prediction)
            decay_predictions[str(year)] = prediction
            print(decay, year, decay_results[str(year)], flush=True)
        results[str(decay)] = decay_results
        predictions[str(decay)] = decay_predictions
    Path("artifacts/v21_recency_weighted_hgb_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(
        predictions, "artifacts/v21_recency_weighted_hgb_predictions.joblib", compress=3
    )


if __name__ == "__main__":
    main()
