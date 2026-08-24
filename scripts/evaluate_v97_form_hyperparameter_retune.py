"""V97: refit the Form HGB hyperparameters for its enlarged feature set.

V96 succeeded by fixing a parameter that had not been updated after the model
changed. The same kind of staleness remains. The Form configuration
(max_leaf_nodes=15, min_samples_leaf=200, l2=20, lr=0.03, max_iter=500) was
selected in V38 against roughly 87 features. V93 grew that to 105, and the 18
added columns produced the largest single gain of the project — a stronger,
denser signal than the capacity was chosen for. Form is now also the heaviest
component of the blend at 0.52, so its configuration matters more than before.

Baseline is V96 (Public 962.8787800874) at weights 0.27 / 0.52 / 0.21 and drift
0.15. Because a better Form model may deserve more weight, each configuration is
also evaluated one step up the form-weight axis, with calibration refitted for
every blend.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v97_form_hyperparameter_metrics.json")
PREDICTIONS = Path("artifacts/v97_form_hyperparameter_predictions.joblib")
V93_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
              "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
CONFIGS = {
    "wide_23": {**V93_CONFIG, "max_leaf_nodes": 23},
    "wide_31": {**V93_CONFIG, "max_leaf_nodes": 31},
    "wide_23_reg": {**V93_CONFIG, "max_leaf_nodes": 23,
                    "min_samples_leaf": 300, "l2_regularization": 30.0},
    "wide_23_long": {**V93_CONFIG, "max_leaf_nodes": 23,
                     "l2_regularization": 30.0, "max_iter": 700},
}
BASE_WEIGHTS = (0.52, 0.21)
FORM_WEIGHTS = (0.52, 0.56)
DRIFT_WEIGHT = 0.15
INSEASON = feature_names()
MIN_GAIN = 1.06e-5
RATIO = 0.932


def points(gain):
    return float(gain * 100000.0 / 0.25)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    inseason_full = add_training_inseason_features(raw_frame)
    inseason_block = inseason_full[INSEASON]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v93_form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    v93_form = v93_form["new_form"]["with_2019_nan"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    forms = {name: {} for name in CONFIGS}
    n_features = None
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        frame = select_v2_features(add_row_features(form_raw, prior))
        frame = add_trackman_features(frame, trackman)
        frame = pd.concat([frame, inseason_block], axis=1)
        columns = v31_form_columns(frame)
        n_features = len(columns)
        for name, config in CONFIGS.items():
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in config.items()
            })
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[name][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"year={year} config={name} trained", flush=True)

    validation_frame = make_validation_frame()
    unit = np.concatenate([
        drift_correction(inseason_full.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])

    baseline = np.clip(
        blend_and_calibrate(BASE_WEIGHTS[0], BASE_WEIGHTS[1], oof, logistic,
                            v93_form, context, raw_frame) + DRIFT_WEIGHT * unit, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for name, form in forms.items():
        for w_form in FORM_WEIGHTS:
            label = f"{name}_form{w_form:.2f}"
            blend = blend_and_calibrate(
                w_form, BASE_WEIGHTS[1], oof, logistic, form, context, raw_frame)
            candidates[label] = np.clip(blend + DRIFT_WEIGHT * unit, 0, 1)
            development[label] = development_metrics(reference, candidates[label])
            print(f"evaluated {label}", flush=True)

    eligible = [
        label for label, r in development.items()
        if r["season_gain_development"]["2022"] > -1e-5
        and r["season_gain_development"]["2023"] > -1e-5
        and r["gain_2024_mar_aug"] >= MIN_GAIN
        and r["gain_2024_jul_aug"] > 0
        and r["monthly_block_win_rate"] >= 0.75
    ]
    promoted = max(eligible, key=lambda l: development[l]["gain_2024_mar_aug"], default=None)
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {"candidate": promoted,
                              "comparison": compare_candidate(reference, candidates[promoted])}

    output = {
        "experiment": "V97_form_hyperparameter_retune",
        "baseline": "V96 (0.27 / 0.52 / 0.21, drift 0.15)",
        "baseline_public_score": 962.8787800874,
        "rationale": (
            "The Form configuration was chosen in V38 for ~87 features; V93 grew it to "
            "105 with the project's strongest new signal, and Form now carries 0.52 of "
            "the blend. Same class of staleness that V96 fixed."
        ),
        "v93_config": V93_CONFIG,
        "configs": CONFIGS,
        "form_feature_count": n_features,
        "form_weights": list(FORM_WEIGHTS),
        "context_weight": BASE_WEIGHTS[1],
        "drift_weight": DRIFT_WEIGHT,
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "calibration_refitted_per_blend": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "baseline": baseline}, PREDICTIONS, compress=3)

    print(f"\nform features = {n_features}")
    print("gains vs V96 baseline, development windows only:")
    print(f"{'candidate':>26} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>26} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e} "
              f"{r['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
