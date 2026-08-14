"""Compare recency-decay strengths on the 2024 chronological holdout."""

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
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    train_mask = data["season"] < 2024
    valid_mask = data["season"] == 2024
    ordinary = add_prior_season_target_encodings(data, y)
    trackman = prepare_trackman(
        pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    )
    results, predictions = {}, {}
    for decay in (0.5, 0.7, 0.85):
        encoded = add_prior_season_decayed_encodings(ordinary, y, decay)
        features = select_v2_features(
            add_row_features(encoded, float(y.loc[train_mask].mean()))
        )
        features = add_trackman_features(features, trackman)
        model, columns = hist_gbdt_pipeline(features)
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        results[str(decay)] = score(y.loc[valid_mask], prediction)
        predictions[str(decay)] = prediction
        print(decay, results[str(decay)], flush=True)
    Path("artifacts/v7_decay_tuning.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"target": y.loc[valid_mask].to_numpy(), "predictions": predictions},
        "artifacts/v7_decay_2024_predictions.joblib",
        compress=3,
    )


if __name__ == "__main__":
    main()
