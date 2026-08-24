"""V101: put the in-season reconstruction into the two V17 HGB components.

Last remaining branch of the axis that produced +50 leaderboard points. The
reconstruction reached the Form model in V93 and was tried on Context in V94a,
but the two target-encoded HGBs inside the V17 tree layer never received it:

  te_trackman_hgb    inner tree weight 0.116
  hierarchical_hgb   inner tree weight 0.355

Together that is 47% of the tree layer, so 0.27 * 0.95 * 0.47 = 12% of the final
prediction. extra_trees and trackman_hgb are left alone; they consume no target
encodings and V101 changes nothing they read.

The prior is not strong. Context carried 13% and produced only +6.5 development
points before failing the sealed window, which suggested the signal is largely
already expressed through Form's 0.52 share. This tests whether the V17 HGBs
behave differently — they lack the stable-form block, so their view is not a
subset of Form's.

Because a stronger V17 layer may deserve more weight, the blend is re-swept.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import COMPONENTS, WEIGHTS
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v101_v17_inseason_components_metrics.json")
PREDICTIONS = Path("artifacts/v101_v17_inseason_components_predictions.joblib")
FORM_WEIGHTS = (0.44, 0.48, 0.52)
CONTEXT_WEIGHT = 0.21
DRIFT_WEIGHT = 0.15
INSEASON = feature_names()
MIN_GAIN = 1.06e-5
RATIO = 0.932


def points(gain):
    return float(gain * 100000.0 / 0.25)


def tree_layer(year, oof, overrides):
    columns = [
        overrides[name][str(year)] if name in overrides
        else oof[str(year)]["components"][name]
        for name in COMPONENTS
    ]
    return np.column_stack(columns) @ WEIGHTS


def blend_and_calibrate(w_form, w_context, oof, logistic, form, context, frame, overrides):
    w_v17 = 1.0 - w_form - w_context
    raw = {}
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * tree_layer(year, oof, overrides) + 0.05 * logistic[key]
        raw[year] = w_v17 * v17 + w_form * form[key] + w_context * context[key]
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            target.append(oof[str(history)]["target"].astype(float))
            prediction.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(target) - np.concatenate(prediction)
        train_frame = frame.loc[index]
        valid_frame = frame.loc[oof[str(year)]["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        pieces.append(np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1))
    return np.concatenate(pieces)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    inseason_full = add_training_inseason_features(raw_frame)
    inseason_block = inseason_full[INSEASON]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    form = form["new_form"]["with_2019_nan"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    new = {"te_trackman_hgb": {}, "hierarchical_hgb": {}}
    counts = {}
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        sources = {
            "te_trackman_hgb": encoded,
            "hierarchical_hgb": hierarchical,
        }
        for name, source in sources.items():
            frame = select_v2_features(add_row_features(source, prior))
            frame = add_trackman_features(frame, trackman)
            frame = pd.concat([frame, inseason_block], axis=1)
            columns = list(frame.columns)
            counts[name] = len(columns)
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            new[name][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"year={year} component={name} ({len(columns)} feat)", flush=True)

    validation_frame = make_validation_frame()
    unit = np.concatenate([
        drift_correction(inseason_full.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])
    baseline = np.clip(
        blend_and_calibrate(0.52, CONTEXT_WEIGHT, oof, logistic, form, context,
                            raw_frame, {}) + DRIFT_WEIGHT * unit, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    variants = {
        "hier_only": {"hierarchical_hgb": new["hierarchical_hgb"]},
        "te_only": {"te_trackman_hgb": new["te_trackman_hgb"]},
        "both": dict(new),
    }
    candidates, development = {}, {}
    for name, overrides in variants.items():
        for w_form in FORM_WEIGHTS:
            label = f"{name}_form{w_form:.2f}"
            blend = blend_and_calibrate(
                w_form, CONTEXT_WEIGHT, oof, logistic, form, context, raw_frame, overrides)
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
        "experiment": "V101_v17_inseason_components",
        "baseline": "V96 (0.27 / 0.52 / 0.21, drift 0.15)",
        "baseline_public_score": 962.8787800874,
        "retrained_components": ["te_trackman_hgb", "hierarchical_hgb"],
        "inner_tree_weights": {name: float(w) for name, w in zip(COMPONENTS, WEIGHTS)},
        "feature_counts": counts,
        "form_weights": list(FORM_WEIGHTS),
        "context_weight": CONTEXT_WEIGHT,
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
    joblib.dump({"new": new, "baseline": baseline}, PREDICTIONS, compress=3)

    print(f"\nfeature counts: {counts}")
    print("gains vs V96 baseline, development windows only:")
    print(f"{'candidate':>22} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>22} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
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
