"""V8: V6 inference plus a fixed earlier-OOF additive calibration."""

import math
import os

import joblib
import numpy as np
import pandas as pd

from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_test_hierarchical_encodings
from target_encoding_v5 import add_test_target_encodings
from trackman_features import KEYS


# Mean target-minus-prediction residual from the 2022, 2023, 2024 chronological
# OOF folds. It is fixed before inference and never uses evaluation-row statistics.
CALIBRATION_SHIFT = -0.011639291780759972


def attach_trackman(frame, lookup):
    output = frame.merge(lookup, how="left", on=KEYS, validate="many_to_one")
    output.index = frame.index
    return output


def main() -> None:
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v6_ensemble.joblib")
    if test["row_id"].duplicated().any() or submission["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test["row_id"]) != set(submission["row_id"]):
        raise ValueError("row_id sets differ")
    raw = test.drop(columns="row_id")[bundle["raw_columns"]]
    base = select_v2_features(add_row_features(raw, bundle["target_prior"]))
    encoded_raw = add_test_target_encodings(raw, bundle["target_encoding_lookups"])
    encoded_base = select_v2_features(add_row_features(encoded_raw, bundle["target_prior"]))
    hierarchical_raw = add_test_hierarchical_encodings(
        encoded_raw, bundle["hierarchical_lookups"]
    )
    hierarchical_base = select_v2_features(
        add_row_features(hierarchical_raw, bundle["target_prior"])
    )
    feature_sets = {
        "extra_trees": base,
        "trackman_hgb": attach_trackman(base, bundle["trackman_lookup_2025"]),
        "encoded_hgb": attach_trackman(encoded_base, bundle["trackman_lookup_2025"]),
        "hierarchical_hgb": attach_trackman(
            hierarchical_base, bundle["trackman_lookup_2025"]
        ),
    }
    raw_prediction = sum(
        bundle["weights"][name]
        * model.predict_proba(feature_sets[name][bundle["model_columns"][name]])[:, 1]
        for name, model in bundle["models"].items()
    )
    prediction = np.clip(raw_prediction + CALIBRATION_SHIFT, 0.0, 1.0)
    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    submission["control_success"] = submission["row_id"].map(
        dict(zip(test["row_id"], prediction))
    )
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(
        f"Saved rows={len(submission)}, raw_mean={raw_prediction.mean():.6f}, "
        f"calibrated_mean={prediction.mean():.6f}"
    )


if __name__ == "__main__":
    main()
