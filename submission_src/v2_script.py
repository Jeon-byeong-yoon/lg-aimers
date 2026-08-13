"""Leakage-safe V2 ensemble inference for the DACON evaluation server."""

import math
import os

import joblib
import pandas as pd

from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v2_ensemble.joblib")
    if test["row_id"].duplicated().any() or submission["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test["row_id"]) != set(submission["row_id"]):
        raise ValueError("test and sample_submission row_id sets differ")

    raw = test.drop(columns="row_id")
    if set(raw.columns) != set(bundle["raw_columns"]):
        raise ValueError("Raw feature schema mismatch")
    raw = raw[bundle["raw_columns"]]
    features = select_v2_features(add_row_features(raw, bundle["target_prior"]))
    prediction = sum(
        bundle["weights"][name]
        * model.predict_proba(features[bundle["model_columns"][name]])[:, 1]
        for name, model in bundle["models"].items()
    )
    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    prediction_by_id = dict(zip(test["row_id"], prediction))
    submission["control_success"] = submission["row_id"].map(prediction_by_id)
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved rows={len(submission)}, mean={prediction.mean():.6f}")


if __name__ == "__main__":
    main()
