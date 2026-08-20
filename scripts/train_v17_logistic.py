"""Train the final C=0.3 Logistic diversity component for V17."""

from pathlib import Path

import joblib
import pandas as pd

from evaluate_logistic_diversity_v17 import pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(
        encoded, y, ["pitcher_batter"]
    )
    features = select_v2_features(add_row_features(hierarchical, float(y.mean())))
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    features = add_trackman_features(features, trackman)
    model, columns = pipeline(features, 0.3)
    model.fit(features[columns], y)
    output = Path("artifacts/v17_logistic_model.joblib")
    joblib.dump({"model": model, "columns": columns}, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
