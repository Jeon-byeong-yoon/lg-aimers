"""Build V41 calibration from V38 gentle_500 form + V31 context with 0.55/0.32/0.13 blend weights."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


# Optimal V41 weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    
    # V38 gentle_500 form OOF predictions
    v38_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    
    # V31 context OOF predictions (no_matchup_hte)
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = v31_removed["no_matchup_hte"]["context"]
    
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        blend = W_V17 * v17 + W_FORM * v38_form[str(year)] + W_CONTEXT * context[str(year)]
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
        
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    
    corrections = [
        {
            "columns": ["balls_before", "strikes_before"],
            "weight": 0.75,
            "lookup": make_lookup(frame, centered, ["balls_before", "strikes_before"], 500),
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
    
    output = Path("artifacts/v41_calibration.joblib")
    joblib.dump({"global_shift": shift, "corrections": corrections}, output, compress=3)
    print(f"Saved {output}, shift={shift:.12f}")


if __name__ == "__main__":
    main()
