"""Build V25 correction lookups from joint chronological OOF predictions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v24_contextual_trackman_predictions.joblib")
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        predictions.append(0.75 * v17 + 0.16 * form[str(year)] + 0.09 * context[str(year)])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    corrections = [
        {
            "columns": ["balls_before", "strikes_before"], "weight": 0.75,
            "lookup": make_lookup(
                frame, centered, ["balls_before", "strikes_before"], 500
            ),
        },
        {
            "columns": ["pitcher_id", "balls_before", "strikes_before"],
            "weight": 0.25,
            "lookup": make_lookup(
                frame, centered,
                ["pitcher_id", "balls_before", "strikes_before"], 300,
            ),
        },
    ]
    output = Path("artifacts/v25_calibration.joblib")
    joblib.dump({"global_shift": shift, "corrections": corrections}, output, compress=3)
    print(f"Saved {output}, shift={shift:.12f}")


if __name__ == "__main__":
    main()
