"""Validate selected V2 preprocessing on the 2022, 2023, and 2024 seasons."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    results = {}

    for valid_season in (2022, 2023, 2024):
        started = perf_counter()
        train_mask = data["season"] < valid_season
        valid_mask = data["season"] == valid_season
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(data, prior))

        extra, extra_columns = extra_trees_pipeline(features)
        extra.fit(features.loc[train_mask, extra_columns], y.loc[train_mask])
        extra_prediction = extra.predict_proba(features.loc[valid_mask, extra_columns])[:, 1]

        hist, hist_columns = hist_gbdt_pipeline(features)
        hist.fit(features.loc[train_mask, hist_columns], y.loc[train_mask])
        hist_prediction = hist.predict_proba(features.loc[valid_mask, hist_columns])[:, 1]

        blend_prediction = 0.50 * extra_prediction + 0.50 * hist_prediction
        season_result = {
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "train_prior": prior,
            "extra_trees": score(y.loc[valid_mask], extra_prediction),
            "hist_gbdt": score(y.loc[valid_mask], hist_prediction),
            "blend_50_50": score(y.loc[valid_mask], blend_prediction),
            "elapsed_seconds": perf_counter() - started,
        }
        results[str(valid_season)] = season_result
        print(valid_season, json.dumps(season_result, indent=2), flush=True)

    briers = [results[str(year)]["blend_50_50"]["brier"] for year in (2022, 2023, 2024)]
    results["summary"] = {
        "mean_blend_brier": float(np.mean(briers)),
        "all_blend_bss_positive": all(
            results[str(year)]["blend_50_50"]["bss"] > 0 for year in (2022, 2023, 2024)
        ),
    }
    output = Path("artifacts/v2_temporal_metrics.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
