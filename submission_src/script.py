"""Evaluation-server inference for the ExtraTrees + HistGBDT ensemble."""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
# "stable" is the primary submission. "affine" is the calibrated experiment.
PREDICTION_VARIANT = os.environ.get("PREDICTION_VARIANT", "stable")


def main() -> None:
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/ensemble.joblib")

    if ID_COL not in test or list(submission.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError("Unexpected test or sample_submission schema")
    if test[ID_COL].duplicated().any() or submission[ID_COL].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test[ID_COL]) != set(submission[ID_COL]):
        raise ValueError("test and sample_submission row_id sets differ")

    features = test.drop(columns=ID_COL)
    expected = bundle["feature_columns"]
    missing = sorted(set(expected) - set(features.columns))
    extra = sorted(set(features.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch: missing={missing}, extra={extra}")
    features = features[expected]

    predictions = {
        name: model.predict_proba(features)[:, 1]
        for name, model in bundle["models"].items()
    }
    if PREDICTION_VARIANT == "stable":
        prediction = sum(
            bundle["stable_weights"][name] * values
            for name, values in predictions.items()
        )
    elif PREDICTION_VARIANT == "affine":
        intercept, extra_weight, hgb_weight = bundle["affine_coefficients"]
        prediction = (
            intercept
            + extra_weight * predictions["extra_trees"]
            + hgb_weight * predictions["hist_gbdt"]
        )
        prediction = np.clip(prediction, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown PREDICTION_VARIANT={PREDICTION_VARIANT!r}")

    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    prediction_by_id = dict(zip(test[ID_COL], prediction))
    submission[TARGET_COL] = submission[ID_COL].map(prediction_by_id)
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(
        f"Saved ./output/submission.csv: rows={len(submission)}, "
        f"variant={PREDICTION_VARIANT}, mean={prediction.mean():.6f}"
    )


if __name__ == "__main__":
    main()
