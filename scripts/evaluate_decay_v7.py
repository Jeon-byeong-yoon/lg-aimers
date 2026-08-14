"""Temporal validation of decay=0.85 and its blend with ordinary TE HGB."""

import json
from pathlib import Path

import joblib
import pandas as pd

from decayed_target_encoding_v7 import add_prior_season_decayed_encodings
from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


def main() -> None:
    ordinary_predictions = joblib.load("artifacts/v5_target_encoding_predictions.joblib")
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    ordinary = add_prior_season_target_encodings(data, y)
    decayed = add_prior_season_decayed_encodings(ordinary, y, 0.85)
    trackman = prepare_trackman(
        pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    )
    results, saved = {}, {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        features = select_v2_features(
            add_row_features(decayed, float(y.loc[train_mask].mean()))
        )
        features = add_trackman_features(features, trackman)
        model, columns = hist_gbdt_pipeline(features)
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        decay_prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        ordinary_prediction = ordinary_predictions[str(year)]["te_trackman_hgb"]
        blended = 0.55 * ordinary_prediction + 0.45 * decay_prediction
        results[str(year)] = {
            "decay_hgb": score(y.loc[valid_mask], decay_prediction),
            "ordinary_55_decay_45": score(y.loc[valid_mask], blended),
        }
        saved[str(year)] = {"target": y.loc[valid_mask].to_numpy(), "decay_hgb": decay_prediction}
        print(year, results[str(year)], flush=True)
    Path("artifacts/v7_decay_temporal.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(saved, "artifacts/v7_decay_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
