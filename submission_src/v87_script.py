"""V87: V41 + V85 3-seed rates/hierarchy/middle late-blend inference."""

import math
import os

import joblib
import numpy as np
import pandas as pd

from asof_features_v79 import add_asof_reliability_features
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


def apply_calibration(prediction, raw, calibration):
    output = prediction + calibration["global_shift"]
    for correction in calibration["corrections"]:
        output += correction["weight"] * lookup_values(
            raw, correction["lookup"], correction["columns"]
        )
    return np.clip(output, 0, 1)


def mean_model_prediction(models, features, columns):
    total = np.zeros(len(features), dtype=float)
    for model in models:
        total += model.predict_proba(features[columns])[:, 1]
    return total / len(models)


def middle_features(frame):
    columns = [
        "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
        "batter_hand", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n", "asof_batter_middle_rate",
    ]
    output = frame[columns].copy()
    n = output["asof_pitcher_n"].fillna(0).clip(lower=0)
    reliability = n / (n + 500.0)
    recent = (
        0.6 * output["asof_pitcher_prev3_game_middle_rate"]
        + 0.4 * output["asof_pitcher_prev5_game_middle_rate"]
    )
    output["v80_middle_delta"] = (
        recent - output["asof_pitcher_middle_rate"]
    ) * reliability
    output["v80_reliability"] = reliability
    return output


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v6_ensemble.joblib")
    logistic = joblib.load("./model/v17_logistic_model.joblib")
    feature_models = joblib.load("./model/v25_feature_models.joblib")
    calibration = joblib.load("./model/v25_calibration.joblib")
    v87 = joblib.load("./model/v87_final_models.joblib")

    if test["row_id"].duplicated().any() or submission["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test["row_id"]) != set(submission["row_id"]):
        raise ValueError("row_id sets differ")

    raw = test.drop(columns="row_id")[bundle["raw_columns"]]
    base = select_v2_features(add_row_features(raw, bundle["target_prior"]))
    encoded_raw = add_test_target_encodings(raw, bundle["target_encoding_lookups"])
    encoded_base = select_v2_features(add_row_features(encoded_raw, bundle["target_prior"]))
    hierarchical_raw = add_test_hierarchical_encodings(encoded_raw, bundle["hierarchical_lookups"])
    hierarchical_base = select_v2_features(add_row_features(hierarchical_raw, bundle["target_prior"]))
    trackman_base = attach_lookup(base, bundle["trackman_lookup_2025"], KEYS)
    encoded_trackman = attach_lookup(encoded_base, bundle["trackman_lookup_2025"], KEYS)
    hierarchical_trackman = attach_lookup(hierarchical_base, bundle["trackman_lookup_2025"], KEYS)
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
    base_form_prediction = feature_models["form_model"].predict_proba(
        form_features[feature_models["form_columns"]]
    )[:, 1]
    context_keys = add_context_keys(hierarchical_trackman)
    context_features = attach_lookup(
        context_keys, feature_models["context_lookup_2025"], KEYS + CONTEXT
    )
    context_prediction = feature_models["context_model"].predict_proba(
        context_features[feature_models["context_columns"]]
    )[:, 1]

    base_raw_prediction = (
        0.55 * v17_prediction + 0.32 * base_form_prediction + 0.13 * context_prediction
    )
    baseline = apply_calibration(base_raw_prediction, raw, calibration)

    signal_predictions = {}
    for variant in ("rates", "hierarchy"):
        asof = add_asof_reliability_features(
            hierarchical_raw, v87["asof_priors"], variant
        )
        variant_raw = add_stable_form_features(asof)
        variant_base = select_v2_features(
            add_row_features(variant_raw, v87["target_prior"])
        )
        variant_features = attach_lookup(
            variant_base, bundle["trackman_lookup_2025"], KEYS
        )
        form_prediction = mean_model_prediction(
            v87[f"{variant}_models"], variant_features, v87[f"{variant}_columns"]
        )
        raw_prediction = (
            0.55 * v17_prediction + 0.32 * form_prediction + 0.13 * context_prediction
        )
        signal_predictions[variant] = apply_calibration(
            raw_prediction, raw, calibration
        )
        del asof, variant_raw, variant_base, variant_features

    middle = middle_features(raw)
    middle_prediction = mean_model_prediction(
        v87["middle_models"], middle, v87["middle_columns"]
    )
    middle_form_prediction = 0.99 * base_form_prediction + 0.01 * middle_prediction
    middle_raw_prediction = (
        0.55 * v17_prediction
        + 0.32 * middle_form_prediction
        + 0.13 * context_prediction
    )
    signal_predictions["middle"] = apply_calibration(
        middle_raw_prediction, raw, calibration
    )

    prediction = np.clip(
        baseline
        + 0.2 * (signal_predictions["rates"] - baseline)
        + 0.2 * (signal_predictions["hierarchy"] - baseline)
        + 0.6 * (signal_predictions["middle"] - baseline),
        0, 1,
    )
    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    submission["control_success"] = submission["row_id"].map(
        dict(zip(test["row_id"], prediction))
    )
    if submission["control_success"].isna().any():
        raise ValueError("Missing predictions after row_id mapping")
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved rows={len(submission)}, prediction_mean={prediction.mean():.6f}")


if __name__ == "__main__":
    main()
