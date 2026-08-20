"""Train the final enhanced contextual Trackman model for V29."""

from pathlib import Path

import joblib
import pandas as pd

from contextual_trackman_v24 import prepare_context_trackman
from contextual_trackman_v26 import (
    add_enhanced_context_features, build_enhanced_context_lookup,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


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
    features = select_v2_features(add_row_features(hierarchical, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = add_enhanced_context_features(features, context_trackman)
    model, columns = hist_gbdt_pipeline(features)
    model.fit(features[columns], y)
    artifact = {
        "context_model": model,
        "context_columns": columns,
        "context_lookup_2025": build_enhanced_context_lookup(context_trackman, 2025),
    }
    output = Path("artifacts/v29_context_model.joblib")
    joblib.dump(artifact, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
