"""Build V14-style correction lookups for the V17 blended OOF predictions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        indices.append(item["row_index"])
        targets.append(item["target"].astype(float))
        predictions.append(0.95 * v11_prediction(item) + 0.05 * logistic[str(year)])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    corrections = [
        {
            "name": "count", "columns": ["balls_before", "strikes_before"],
            "weight": 0.75,
            "lookup": make_lookup(
                frame, centered, ["balls_before", "strikes_before"], 500
            ),
        },
        {
            "name": "pitcher_count",
            "columns": ["pitcher_id", "balls_before", "strikes_before"],
            "weight": 0.25,
            "lookup": make_lookup(
                frame, centered,
                ["pitcher_id", "balls_before", "strikes_before"], 300,
            ),
        },
    ]
    output = Path("artifacts/v17_calibration.joblib")
    joblib.dump({"global_shift": shift, "corrections": corrections}, output, compress=3)
    print(f"Saved {output}, shift={shift:.12f}")


if __name__ == "__main__":
    main()
