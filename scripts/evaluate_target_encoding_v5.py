"""Evaluate chronological player target encoding, with V4 Trackman features."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


TRACKMAN_COLUMNS = [
    "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
    "batter_hand", "pitch_type_group", "rel_speed", "spin_rate",
    "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side",
    "zone_speed",
]


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    trackman = prepare_trackman(
        pd.read_csv(
            "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
        )
    )
    results = {}
    saved = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        base = select_v2_features(
            add_row_features(encoded, float(y.loc[train_mask].mean()))
        )
        candidate = add_trackman_features(base, trackman)
        model, columns = hist_gbdt_pipeline(candidate)
        model.fit(candidate.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(candidate.loc[valid_mask, columns])[:, 1]
        results[str(year)] = score(y.loc[valid_mask], prediction)
        results[str(year)]["feature_count"] = len(columns)
        saved[str(year)] = {"target": y.loc[valid_mask].to_numpy(), "te_trackman_hgb": prediction}
        print(year, results[str(year)], flush=True)
    Path("artifacts/v5_target_encoding_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(saved, "artifacts/v5_target_encoding_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
