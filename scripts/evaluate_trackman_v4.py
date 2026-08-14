"""Evaluate strictly prior-season Trackman aggregates on top of V2."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from trackman_features import add_trackman_features, prepare_trackman


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    trackman_columns = [
        "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
        "batter_hand", "pitch_type_group", "rel_speed", "spin_rate",
        "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side",
        "zone_speed",
    ]
    trackman = prepare_trackman(
        pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=trackman_columns)
    )
    results = {}
    saved_predictions = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        base = select_v2_features(
            add_row_features(data, float(y.loc[train_mask].mean()))
        )
        with_trackman = add_trackman_features(base, trackman)
        # 2019 has no earlier Trackman rows; model imputation handles these missing values.
        model, columns = hist_gbdt_pipeline(with_trackman)
        model.fit(with_trackman.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(with_trackman.loc[valid_mask, columns])[:, 1]
        results[str(year)] = score(y.loc[valid_mask], prediction)
        results[str(year)]["feature_count"] = len(columns)
        saved_predictions[str(year)] = {
            "row_index": data.index[valid_mask].to_numpy(),
            "target": y.loc[valid_mask].to_numpy(),
            "trackman_hgb": prediction,
        }
        print(year, results[str(year)], flush=True)
    Path("artifacts/v4_trackman_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(saved_predictions, "artifacts/v4_trackman_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
