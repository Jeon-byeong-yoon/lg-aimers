"""V93: put the current-season as-of reconstruction inside the Form HGB.

V92 added the reconstruction as a single linear post-hoc term on top of the
finished prediction, which was worth +24.17 leaderboard points. That term can
only express one global slope. Letting the Form HGB consume the reconstructed
rates directly allows the model to interact them with the count, inning, runner
state and sample size, and to use them *instead of* the drift-contaminated
career rates rather than merely correcting for them.

Everything else is held fixed: V17 tree/logistic layer, Context HGB, the V38
gentle_500 Form configuration, the 0.55/0.32/0.13 blend and the calibration
structure. The baseline for every comparison is V92 (Public 924.9037964219),
not V41. Candidate selection uses 2022, 2023 and 2024 March-August only.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import BASE_CONFIG, v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v93_inseason_form_model_metrics.json")
PREDICTIONS = Path("artifacts/v93_inseason_form_model_predictions.joblib")
FORM_CONFIG = {**BASE_CONFIG, "learning_rate": 0.03, "max_iter": 500}
V92_WEIGHT = 0.20
INSEASON = feature_names()
# How much of the V41 Form layer the in-season Form model replaces.
FORM_MIXES = (0.50, 1.00)
# Post-hoc V92 drift term retained on top of the new Form model.
DRIFT_WEIGHTS = (0.0, 0.10, 0.20)
TRAIN_STARTS = {"with_2019_nan": 2019, "drop_2019": 2020}
LOCAL_TO_LEADERBOARD = 1.475


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
    inseason_block = add_training_inseason_features(raw_frame)[INSEASON]

    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    new_form = {name: {} for name in TRAIN_STARTS}
    feature_count = None
    for year in YEARS:
        prior = float(y.loc[raw_frame["season"] < year].mean())
        features = select_v2_features(add_row_features(form_raw, prior))
        features = add_trackman_features(features, trackman)
        features = pd.concat([features, inseason_block], axis=1)
        candidate = features[v31_form_columns(features)]
        feature_count = int(candidate.shape[1])
        valid_mask = raw_frame["season"] == year
        for name, start in TRAIN_STARTS.items():
            train_mask = (raw_frame["season"] < year) & (raw_frame["season"] >= start)
            model, columns = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{key}": value
                for key, value in FORM_CONFIG.items()
            })
            model.fit(candidate.loc[train_mask, columns], y.loc[train_mask])
            new_form[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, columns]
            )[:, 1]
            print(f"year={year} form={name} trained (n={int(train_mask.sum())})", flush=True)

    validation_frame = make_validation_frame()
    unit_correction = np.concatenate([
        drift_correction(
            add_training_inseason_features(raw_frame).loc[oof[str(year)]["row_index"]], 1.0
        ) for year in YEARS
    ])
    v92_baseline = np.clip(
        validation_frame["v41_prediction"].to_numpy() + V92_WEIGHT * unit_correction, 0, 1
    )
    v92_frame = validation_frame.copy()
    v92_frame["v41_prediction"] = v92_baseline
    v92_frame["v41_squared_error"] = (v92_baseline - v92_frame["target"]) ** 2

    candidates, development = {}, {}
    for name in TRAIN_STARTS:
        for mix in FORM_MIXES:
            blended_form = {
                key: (1.0 - mix) * baseline_form[key] + mix * new_form[name][key]
                for key in baseline_form
            }
            parts = []
            for year in YEARS:
                prediction, _, _, _ = calibrated_prediction(
                    year, oof, logistic, blended_form, context, raw_frame
                )
                parts.append(prediction)
            base = np.concatenate(parts)
            for drift in DRIFT_WEIGHTS:
                label = f"{name}_mix{mix:.2f}_drift{drift:.2f}"
                candidates[label] = np.clip(base + drift * unit_correction, 0, 1)
                development[label] = development_metrics(v92_frame, candidates[label])

    eligible = [
        label for label, result in development.items()
        if result["season_gain_development"]["2022"] > -1e-5
        and result["season_gain_development"]["2023"] > -1e-5
        and result["gain_2024_mar_aug"] >= 6.8e-6
        and result["gain_2024_jul_aug"] > 0
        and result["monthly_block_win_rate"] >= 0.75
    ]
    promoted = max(
        eligible, key=lambda label: development[label]["gain_2024_mar_aug"], default=None
    )
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {
            "candidate": promoted,
            "comparison": compare_candidate(v92_frame, candidates[promoted]),
        }

    output = {
        "experiment": "V93_inseason_form_model",
        "baseline": "V92",
        "baseline_public_score": 924.9037964219,
        "form_config": FORM_CONFIG,
        "form_feature_count": feature_count,
        "inseason_features": INSEASON,
        "form_mixes": list(FORM_MIXES),
        "drift_weights": list(DRIFT_WEIGHTS),
        "train_starts": TRAIN_STARTS,
        "local_to_leaderboard_factor": LOCAL_TO_LEADERBOARD,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "current_pitch_post_event_information_used": False,
            "trackman_2025_used": False,
            "inseason_anchor_frozen_at_training_time": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"new_form": new_form, "v92_baseline": v92_baseline,
                 "unit_correction": unit_correction}, PREDICTIONS, compress=3)

    print(f"\nform features = {feature_count} (in-season {len(INSEASON)} added)")
    print("gains vs V92 baseline, development windows only:")
    print(f"{'candidate':>34} {'2022':>9} {'2023':>9} {'2024 pt':>9} {'2024 gain':>12} {'LB est':>8} {'blocks':>8} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>34} {points(sg['2022']):9.1f} {points(sg['2023']):9.1f} "
              f"{points(r['gain_2024_mar_aug']):9.1f} {r['gain_2024_mar_aug']:12.3e} "
              f"{points(r['gain_2024_mar_aug'])*LOCAL_TO_LEADERBOARD:8.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
