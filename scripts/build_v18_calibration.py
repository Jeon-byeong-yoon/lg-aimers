"""Build V18 calibration artifact.

Extends V17 with two additional OOF-fixed segment corrections:
  - base_state   : smoothing=10000, weight=1.0
  - count_hand   : smoothing=50,    weight=0.25
Residuals are recomputed from V17 blended OOF (95% tree + 5% Logistic).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


LOGISTIC_C = "0.3"
LOGISTIC_WEIGHT = 0.05
COUNT_SMOOTHING = 500
COUNT_WEIGHT = 0.75
PITCHER_COUNT_SMOOTHING = 300
PITCHER_COUNT_WEIGHT = 0.25
BASE_STATE_SMOOTHING = 10000
BASE_STATE_WEIGHT = 1.0
COUNT_HAND_SMOOTHING = 50
COUNT_HAND_WEIGHT = 0.25


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")[LOGISTIC_C]
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")

    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        indices.append(item["row_index"])
        targets.append(item["target"].astype(float))
        tree_pred = v11_prediction(item)
        logistic_pred = logistic[str(year)]
        predictions.append(
            (1 - LOGISTIC_WEIGHT) * tree_pred + LOGISTIC_WEIGHT * logistic_pred
        )

    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    corrections = [
        {
            "name": "count",
            "columns": ["balls_before", "strikes_before"],
            "weight": COUNT_WEIGHT,
            "lookup": make_lookup(frame, centered, ["balls_before", "strikes_before"], COUNT_SMOOTHING),
        },
        {
            "name": "pitcher_count",
            "columns": ["pitcher_id", "balls_before", "strikes_before"],
            "weight": PITCHER_COUNT_WEIGHT,
            "lookup": make_lookup(
                frame, centered,
                ["pitcher_id", "balls_before", "strikes_before"], PITCHER_COUNT_SMOOTHING,
            ),
        },
        {
            "name": "base_state",
            "columns": ["base_state"],
            "weight": BASE_STATE_WEIGHT,
            "lookup": make_lookup(frame, centered, ["base_state"], BASE_STATE_SMOOTHING),
        },
        {
            "name": "count_hand",
            "columns": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
            "weight": COUNT_HAND_WEIGHT,
            "lookup": make_lookup(
                frame, centered,
                ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
                COUNT_HAND_SMOOTHING,
            ),
        },
    ]

    output = Path("artifacts/v18_calibration.joblib")
    joblib.dump({"global_shift": shift, "corrections": corrections}, output, compress=3)
    print(f"Saved {output}")
    print(f"  global_shift : {shift:.12f}")
    print(f"  corrections  : {[c['name'] for c in corrections]}")


if __name__ == "__main__":
    main()
