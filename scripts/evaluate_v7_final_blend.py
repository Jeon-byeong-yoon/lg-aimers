"""Validate fixed V7 ensemble weights against V6."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    v4 = joblib.load("artifacts/v4_trackman_predictions.joblib")
    v5 = joblib.load("artifacts/v5_target_encoding_predictions.joblib")
    v6 = joblib.load("artifacts/v6_hierarchical_predictions.joblib")
    v7 = joblib.load("artifacts/v7_decay_predictions.joblib")
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    results = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        base = select_v2_features(add_row_features(data, float(y.loc[train_mask].mean())))
        extra, columns = extra_trees_pipeline(base)
        extra.fit(base.loc[train_mask, columns], y.loc[train_mask])
        extra_prediction = extra.predict_proba(base.loc[valid_mask, columns])[:, 1]
        prediction = (
            0.40 * extra_prediction
            + 0.36 * v4[str(year)]["trackman_hgb"]
            + 0.048 * v5[str(year)]["te_trackman_hgb"]
            + 0.12 * v6[str(year)]["hierarchical_hgb"]
            + 0.072 * v7[str(year)]["decay_hgb"]
        )
        results[str(year)] = score(y.loc[valid_mask], prediction)
        print(year, results[str(year)], flush=True)
    Path("artifacts/v7_final_blend_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
