"""V4 evaluation inference using a fixed official 2019--2024 Trackman lookup."""

import math
import os

import joblib
import pandas as pd

from feature_engineering_v2 import add_row_features, select_v2_features
from trackman_features import KEYS


def main() -> None:
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v4_ensemble.joblib")
    if test["row_id"].duplicated().any() or submission["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test["row_id"]) != set(submission["row_id"]):
        raise ValueError("test and sample_submission row_id sets differ")

    raw = test.drop(columns="row_id")[bundle["raw_columns"]]
    base = select_v2_features(add_row_features(raw, bundle["target_prior"]))
    with_trackman = base.merge(
        bundle["trackman_lookup_2025"], how="left", on=KEYS, validate="many_to_one"
    )
    with_trackman.index = base.index
    predictions = {
        "extra_trees": bundle["models"]["extra_trees"].predict_proba(
            base[bundle["model_columns"]["extra_trees"]]
        )[:, 1],
        "trackman_hgb": bundle["models"]["trackman_hgb"].predict_proba(
            with_trackman[bundle["model_columns"]["trackman_hgb"]]
        )[:, 1],
    }
    prediction = sum(bundle["weights"][name] * values for name, values in predictions.items())
    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    submission["control_success"] = submission["row_id"].map(
        dict(zip(test["row_id"], prediction))
    )
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved rows={len(submission)}, mean={prediction.mean():.6f}")


if __name__ == "__main__":
    main()
