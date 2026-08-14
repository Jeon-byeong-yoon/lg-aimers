"""Train V5 with chronological player encodings and prior Trackman features."""

from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from target_encoding_v5 import add_prior_season_target_encodings, build_test_lookups
from trackman_features import add_trackman_features, build_lookup, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    prior = float(y.mean())
    base = select_v2_features(add_row_features(data, prior))
    encoded_raw = add_prior_season_target_encodings(data, y)
    encoded_base = select_v2_features(add_row_features(encoded_raw, prior))
    trackman = prepare_trackman(
        pd.read_csv(
            "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
        )
    )
    trackman_base = add_trackman_features(base, trackman)
    encoded_trackman_base = add_trackman_features(encoded_base, trackman)

    extra, extra_columns = extra_trees_pipeline(base)
    trackman_hgb, trackman_columns = hist_gbdt_pipeline(trackman_base)
    encoded_hgb, encoded_columns = hist_gbdt_pipeline(encoded_trackman_base)
    extra.fit(base[extra_columns], y)
    trackman_hgb.fit(trackman_base[trackman_columns], y)
    encoded_hgb.fit(encoded_trackman_base[encoded_columns], y)

    bundle = {
        "models": {
            "extra_trees": extra,
            "trackman_hgb": trackman_hgb,
            "encoded_hgb": encoded_hgb,
        },
        "model_columns": {
            "extra_trees": extra_columns,
            "trackman_hgb": trackman_columns,
            "encoded_hgb": encoded_columns,
        },
        "weights": {"extra_trees": 0.40, "trackman_hgb": 0.36, "encoded_hgb": 0.24},
        "target_prior": prior,
        "raw_columns": list(data.columns),
        "trackman_lookup_2025": build_lookup(trackman, 2025),
        "target_encoding_lookups": build_test_lookups(data, y),
        "training_seasons": sorted(map(int, data["season"].unique())),
    }
    output = Path("artifacts/v5_ensemble.joblib")
    joblib.dump(bundle, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
