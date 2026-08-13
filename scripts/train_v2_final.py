"""Fit the selected V2 ensemble on all official 2019--2024 training rows."""

from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    prior = float(y.mean())
    features = select_v2_features(add_row_features(data, prior))

    extra, extra_columns = extra_trees_pipeline(features)
    hist, hist_columns = hist_gbdt_pipeline(features)
    extra.fit(features[extra_columns], y)
    hist.fit(features[hist_columns], y)
    bundle = {
        "models": {"extra_trees": extra, "hist_gbdt": hist},
        "model_columns": {
            "extra_trees": extra_columns,
            "hist_gbdt": hist_columns,
        },
        "weights": {"extra_trees": 0.5, "hist_gbdt": 0.5},
        "target_prior": prior,
        "raw_columns": list(data.columns),
        "training_seasons": sorted(map(int, data["season"].unique())),
    }
    output = Path("artifacts/v2_ensemble.joblib")
    joblib.dump(bundle, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
