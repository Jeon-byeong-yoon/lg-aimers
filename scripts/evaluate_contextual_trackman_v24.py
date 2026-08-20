"""Evaluate hierarchical context Trackman features with chronological folds."""

import json
from pathlib import Path

import joblib
import pandas as pd

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


TRACKMAN_COLUMNS = [
    "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
    "batter_hand", "pitch_type_group", "rel_speed", "spin_rate",
    "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side",
    "zone_speed", "outs_before", "inning", "top_bottom",
]


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
    results, saved = {}, {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(hierarchical, prior))
        features = add_trackman_features(features, trackman)
        features = add_context_trackman_features(features, context_trackman)
        model, columns = hist_gbdt_pipeline(features)
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        results[str(year)] = score(y.loc[valid_mask], prediction)
        saved[str(year)] = prediction
        print(year, results[str(year)], flush=True)
    Path("artifacts/v24_contextual_trackman_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(saved, "artifacts/v24_contextual_trackman_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
