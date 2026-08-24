"""V98: refit the calibration correction structure for V96's blend.

V96 rebuilt the calibration *lookups* for the new weights but kept the structure
those lookups sit in: a 0.75 / 0.25 split between the count correction and the
pitcher-by-count correction, smoothed with 500 and 300. Those four numbers were
chosen in V12 and V14, when the blend was 0.75 V17 / 0.16 Form / 0.09 Context and
there was no in-season reconstruction. The blend has since moved to
0.27 / 0.52 / 0.21, so the residual these corrections operate on is a different
object.

This is the same class of staleness V96 fixed, and it is the cheapest one left:
no model is retrained, only the correction weights and smoothing strengths move.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v98_calibration_structure_metrics.json")
PREDICTIONS = Path("artifacts/v98_calibration_structure_predictions.joblib")
W_V17, W_FORM, W_CONTEXT = 0.27, 0.52, 0.21
DRIFT_WEIGHT = 0.15
BASELINE = (0.75, 0.25, 500, 300)
SPLITS = ((1.00, 0.00), (0.85, 0.15), (0.75, 0.25), (0.60, 0.40), (0.50, 0.50))
COUNT_SMOOTHING = (300, 500, 800)
PITCHER_SMOOTHING = (150, 300, 600)
MIN_GAIN = 1.06e-5
RATIO = 0.932


def points(gain):
    return float(gain * 100000.0 / 0.25)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = data.drop(columns=["row_id", "control_success"])

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    form = form["new_form"]["with_2019_nan"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    inseason = add_training_inseason_features(frame)
    unit = np.concatenate([
        drift_correction(inseason.loc[oof[str(year)]["row_index"]], 1.0) for year in YEARS
    ])

    # Raw blend and residual history are identical across candidates; only the
    # correction weights and smoothing strengths vary, so compute them once.
    raw, residual_pool = {}, {}
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        raw[year] = W_V17 * v17 + W_FORM * form[key] + W_CONTEXT * context[key]
    for year in YEARS:
        if year == 2022:
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            hkey = str(history)
            index.append(oof[hkey]["row_index"])
            target.append(oof[hkey]["target"].astype(float))
            prediction.append(raw[history])
        index = np.concatenate(index)
        residual_pool[year] = (
            index,
            np.concatenate(target) - np.concatenate(prediction),
        )

    cache = {}

    def correction(year, columns, smoothing):
        cache_key = (year, tuple(columns), smoothing)
        if cache_key not in cache:
            index, residual = residual_pool[year]
            cache[cache_key] = segment_correction(
                frame.loc[index], residual, frame.loc[oof[str(year)]["row_index"]],
                list(columns), smoothing,
            )
        return cache[cache_key]

    def assemble(w_count, w_pitcher, count_smoothing, pitcher_smoothing):
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            _, residual = residual_pool[year]
            total = residual.mean()
            total = total + w_count * correction(
                year, ("balls_before", "strikes_before"), count_smoothing)
            if w_pitcher > 0:
                total = total + w_pitcher * correction(
                    year, ("pitcher_id", "balls_before", "strikes_before"),
                    pitcher_smoothing)
            pieces.append(np.clip(raw[year] + total, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * unit, 0, 1)

    validation_frame = make_validation_frame()
    baseline = assemble(*BASELINE)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for w_count, w_pitcher in SPLITS:
        for count_smoothing in COUNT_SMOOTHING:
            pitcher_options = PITCHER_SMOOTHING if w_pitcher > 0 else (BASELINE[3],)
            for pitcher_smoothing in pitcher_options:
                setting = (w_count, w_pitcher, count_smoothing, pitcher_smoothing)
                if setting == BASELINE:
                    continue
                label = (f"count{w_count:.2f}_pit{w_pitcher:.2f}"
                         f"_cs{count_smoothing}_ps{pitcher_smoothing}")
                candidates[label] = assemble(*setting)
                development[label] = development_metrics(reference, candidates[label])
        print(f"split {w_count:.2f}/{w_pitcher:.2f} evaluated", flush=True)

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
        "experiment": "V98_calibration_structure_retune",
        "baseline": "V96 (0.75/0.25 split, smoothing 500/300)",
        "baseline_public_score": 962.8787800874,
        "rationale": (
            "The 0.75/0.25 split and 500/300 smoothing were chosen in V12/V14 for a "
            "0.75/0.16/0.09 blend with no in-season reconstruction. The blend is now "
            "0.27/0.52/0.21, so the residual these corrections act on has changed."
        ),
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT},
        "drift_weight": DRIFT_WEIGHT,
        "splits": [list(s) for s in SPLITS],
        "count_smoothing": list(COUNT_SMOOTHING),
        "pitcher_smoothing": list(PITCHER_SMOOTHING),
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
            "corrections_fitted_on_earlier_seasons_only": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"baseline": baseline}, PREDICTIONS, compress=3)

    print("\ntop 15 by 2024 development gain (vs V96):")
    print(f"{'candidate':>36} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"])[:15]:
        sg = r["season_gain_development"]
        print(f"{label:>36} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
