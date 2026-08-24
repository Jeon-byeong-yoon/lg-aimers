"""Rebuild V41-structure calibration on the V93 Form OOF predictions.

The blend weights (0.55 / 0.32 / 0.13) and the correction structure are
unchanged; only the Form component's out-of-fold predictions differ, so the
global shift and the count / pitcher-count lookups must be recomputed. The V92
drift term is applied after calibration, exactly as it was validated, so it is
not included in these residuals.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


OUTPUT = Path("artifacts/v93_calibration.joblib")
W_V17, W_FORM, W_CONTEXT = 0.55, 0.32, 0.13
FORM_VARIANT = "with_2019_nan"


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    form = form["new_form"][FORM_VARIANT]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        blend = W_V17 * v17 + W_FORM * form[str(year)] + W_CONTEXT * context[str(year)]
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
                frame, centered, ["pitcher_id", "balls_before", "strikes_before"], 300
            ),
        },
    ]
    joblib.dump({"global_shift": shift, "corrections": corrections}, OUTPUT, compress=3)
    print(f"Saved {OUTPUT}, shift={shift:.12f} (V41 shift was -0.011287550929)")


if __name__ == "__main__":
    main()
