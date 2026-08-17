"""V17: 95% V11 tree ensemble + 5% regularized Logistic with OOF calibration."""

import math
import os

import joblib
import numpy as np
import pandas as pd

from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_test_hierarchical_encodings
from target_encoding_v5 import add_test_target_encodings
from trackman_features import KEYS


TREE_WEIGHTS = {
    "extra_trees": 0.29483562599237795,
    "trackman_hgb": 0.2344456574665245,
    "encoded_hgb": 0.11617615437521091,
    "hierarchical_hgb": 0.35454256216588664,
}


def lookup_values(frame, lookup, columns):
    keyed = frame[columns].copy()
    keyed["_order"] = np.arange(len(keyed))
    merged = keyed.merge(lookup, how="left", on=columns, validate="many_to_one")
    return merged.sort_values("_order")["correction"].fillna(0).to_numpy()


def attach_trackman(frame, lookup):
    output = frame.merge(lookup, how="left", on=KEYS, validate="many_to_one")
    output.index = frame.index
    return output


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v6_ensemble.joblib")
    logistic = joblib.load("./model/v17_logistic_model.joblib")
    calibration = joblib.load("./model/v17_calibration.joblib")
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
    trackman_base = attach_trackman(base, bundle["trackman_lookup_2025"])
    encoded_trackman = attach_trackman(encoded_base, bundle["trackman_lookup_2025"])
    hierarchical_trackman = attach_trackman(
        hierarchical_base, bundle["trackman_lookup_2025"]
    )
    feature_sets = {
        "extra_trees": base,
        "trackman_hgb": trackman_base,
        "encoded_hgb": encoded_trackman,
        "hierarchical_hgb": hierarchical_trackman,
    }
    tree_prediction = sum(
        TREE_WEIGHTS[name]
        * model.predict_proba(feature_sets[name][bundle["model_columns"][name]])[:, 1]
        for name, model in bundle["models"].items()
    )
    logistic_prediction = logistic["model"].predict_proba(
        hierarchical_trackman[logistic["columns"]]
    )[:, 1]
    prediction = 0.95 * tree_prediction + 0.05 * logistic_prediction
    prediction += calibration["global_shift"]
    for correction in calibration["corrections"]:
        prediction += correction["weight"] * lookup_values(
            raw, correction["lookup"], correction["columns"]
        )
    prediction = np.clip(prediction, 0, 1)
    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    submission["control_success"] = submission["row_id"].map(
        dict(zip(test["row_id"], prediction))
    )
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved rows={len(submission)}, prediction_mean={prediction.mean():.6f}")


if __name__ == "__main__":
    main()
