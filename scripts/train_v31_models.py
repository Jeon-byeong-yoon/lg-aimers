"""Train V31 models after removing unstable V25 features."""

from pathlib import Path

import joblib
import pandas as pd

from contextual_trackman_v24 import (
    add_context_trackman_features, build_context_lookup, prepare_context_trackman,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
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
    prior = float(y.mean())

    form_raw = add_stable_form_features(hierarchical)
    form_features = select_v2_features(add_row_features(form_raw, prior))
    form_features = add_trackman_features(form_features, trackman)
    form_columns = [
        column for column in form_features
        if not (column.startswith("tm_") and column.endswith("_std"))
    ]
    form_model, form_columns = hist_gbdt_pipeline(form_features[form_columns])
    form_model.fit(form_features[form_columns], y)

    context_features = select_v2_features(add_row_features(hierarchical, prior))
    context_features = add_trackman_features(context_features, trackman)
    context_features = add_context_trackman_features(context_features, context_trackman)
    context_columns = [column for column in context_features if column not in MATCHUP_HTE]
    context_model, context_columns = hist_gbdt_pipeline(context_features[context_columns])
    context_model.fit(context_features[context_columns], y)

    artifact = {
        "form_model": form_model, "form_columns": form_columns,
        "context_model": context_model, "context_columns": context_columns,
        "context_lookup_2025": build_context_lookup(context_trackman, 2025),
    }
    output = Path("artifacts/v31_feature_models.joblib")
    joblib.dump(artifact, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
