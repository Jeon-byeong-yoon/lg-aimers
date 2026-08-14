"""Blend V2 ExtraTrees with the prior-season Trackman HGB across three folds."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    saved = joblib.load("artifacts/v4_trackman_predictions.joblib")
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    results = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        features = select_v2_features(
            add_row_features(data, float(y.loc[train_mask].mean()))
        )
        extra, columns = extra_trees_pipeline(features)
        extra.fit(features.loc[train_mask, columns], y.loc[train_mask])
        extra_prediction = extra.predict_proba(features.loc[valid_mask, columns])[:, 1]
        trackman_prediction = saved[str(year)]["trackman_hgb"]
        candidates = []
        for extra_weight in np.linspace(0.2, 0.8, 7):
            prediction = extra_weight * extra_prediction + (1 - extra_weight) * trackman_prediction
            result = score(y.loc[valid_mask], prediction)
            result["extra_weight"] = float(extra_weight)
            candidates.append(result)
        candidates.sort(key=lambda item: item["brier"])
        results[str(year)] = {"best": candidates[0], "all_weights": candidates}
        print(year, candidates[0], flush=True)

    # Pick one fixed weight by mean normalized Brier deterioration across folds.
    weight_summary = []
    for weight in np.linspace(0.2, 0.8, 7):
        fold_results = []
        for year in (2022, 2023, 2024):
            match = next(
                item for item in results[str(year)]["all_weights"]
                if abs(item["extra_weight"] - weight) < 1e-9
            )
            fold_results.append(match["brier"])
        weight_summary.append({"extra_weight": float(weight), "mean_brier": float(np.mean(fold_results))})
    weight_summary.sort(key=lambda item: item["mean_brier"])
    results["weight_summary"] = weight_summary
    Path("artifacts/v4_blend_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("fixed weight ranking", weight_summary)


if __name__ == "__main__":
    main()
