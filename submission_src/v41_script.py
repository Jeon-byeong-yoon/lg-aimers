"""V41: V17 + V38 gentle_500 form + contextual-Trackman with retuned weights (0.55 / 0.32 / 0.13)."""

import math
import os

import joblib
import numpy as np
import pandas as pd

from contextual_trackman_v24 import CONTEXT, add_context_keys
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_test_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_test_target_encodings
from trackman_features import KEYS


TREE_WEIGHTS = {
    "extra_trees": 0.29483562599237795,
    "trackman_hgb": 0.2344456574665245,
    "encoded_hgb": 0.11617615437521091,
    "hierarchical_hgb": 0.35454256216588664,
}


def lookup_values(frame, lookup, columns, value="correction"):
    keyed = frame[columns].copy()
    keyed["_order"] = np.arange(len(keyed))
    merged = keyed.merge(lookup, how="left", on=columns, validate="many_to_one")
    return merged.sort_values("_order")[value].fillna(0).to_numpy()


def attach_lookup(frame, lookup, columns):
    output = frame.copy()
    output["__original_index"] = output.index
    output = output.merge(lookup, how="left", on=columns, validate="many_to_one")
    output = output.set_index("__original_index")
    output.index.name = frame.index.name
    return output


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v6_ensemble.joblib")
    logistic = joblib.load("./model/v17_logistic_model.joblib")
    feature_models = joblib.load("./model/v25_feature_models.joblib")
    calibration = joblib.load("./model/v25_calibration.joblib")
    
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
    trackman_base = attach_lookup(base, bundle["trackman_lookup_2025"], KEYS)
    encoded_trackman = attach_lookup(
        encoded_base, bundle["trackman_lookup_2025"], KEYS
    )
    hierarchical_trackman = attach_lookup(
        hierarchical_base, bundle["trackman_lookup_2025"], KEYS
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
    v17_prediction = 0.95 * tree_prediction + 0.05 * logistic_prediction

    form_raw = add_stable_form_features(hierarchical_raw)
    form_base = select_v2_features(add_row_features(form_raw, bundle["target_prior"]))
    form_features = attach_lookup(form_base, bundle["trackman_lookup_2025"], KEYS)
    form_prediction = feature_models["form_model"].predict_proba(
        form_features[feature_models["form_columns"]]
    )[:, 1]

    context_keys = add_context_keys(hierarchical_trackman)
    context_features = attach_lookup(
        context_keys, feature_models["context_lookup_2025"], KEYS + CONTEXT
    )
    context_prediction = feature_models["context_model"].predict_proba(
        context_features[feature_models["context_columns"]]
    )[:, 1]

    # V41 retuned weights: 55% V17 + 32% Form + 13% Context
    prediction = 0.55 * v17_prediction + 0.32 * form_prediction + 0.13 * context_prediction
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
