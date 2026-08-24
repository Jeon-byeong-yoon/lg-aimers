"""V96: re-optimise the blend and drift weights for V93's components.

The 0.55 / 0.32 / 0.13 split was fitted in V41, when the Form layer was the V38
model. V93 replaced that Form model with a materially stronger one (the
current-season reconstruction added 18 features and produced the single largest
gain of the project), but the weights were never refitted. The same situation
occurred once before and was worth a lot: V38 improved the Form model, V41
refitted the three-way weights, and the leaderboard moved +6.54 points.

The V92 drift weight (0.20) has the same problem — it was tuned against V41's
prediction vector, not V93's.

This step adds no features and no new information, so it is not subject to the
V94/V95 "new information" filter: it corrects a genuine mis-specification.
Calibration is rebuilt for every blend, exactly as V41 did. The drift term is
applied after calibration, so it can be swept without refitting the lookups.
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


OUTPUT = Path("artifacts/v96_weight_reoptimization_metrics.json")
PREDICTIONS = Path("artifacts/v96_weight_reoptimization_predictions.joblib")
FORM_WEIGHTS = (0.32, 0.36, 0.40, 0.44, 0.48)
CONTEXT_WEIGHTS = (0.09, 0.13, 0.17)
DRIFT_WEIGHTS = (0.15, 0.20, 0.25)
BASELINE = (0.32, 0.13, 0.20)
MIN_GAIN = 1.06e-5
RATIO = 0.944


def points(gain):
    return float(gain * 100000.0 / 0.25)


def blend_and_calibrate(w_form, w_context, oof, logistic, form, context, frame):
    """V41 calibration structure, refitted for this particular blend."""
    w_v17 = 1.0 - w_form - w_context
    pieces = []
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        raw = w_v17 * v17 + w_form * form[key] + w_context * context[key]
        if year == 2022:
            pieces.append(np.clip(raw, 0, 1))
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            hkey = str(history)
            hv17 = 0.95 * v11_prediction(oof[hkey]) + 0.05 * logistic[hkey]
            prediction.append(
                w_v17 * hv17 + w_form * form[hkey] + w_context * context[hkey]
            )
            target.append(oof[hkey]["target"].astype(float))
            index.append(oof[hkey]["row_index"])
        index = np.concatenate(index)
        residual = np.concatenate(target) - np.concatenate(prediction)
        train_frame = frame.loc[index]
        valid_frame = frame.loc[oof[key]["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
        )
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300,
        )
        pieces.append(np.clip(
            raw + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1
        ))
    return np.concatenate(pieces)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data = data.drop(columns=["row_id", "control_success"])

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    form = form["new_form"]["with_2019_nan"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    inseason = add_training_inseason_features(data)
    unit_correction = np.concatenate([
        drift_correction(inseason.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])

    blends = {}
    for w_form in FORM_WEIGHTS:
        for w_context in CONTEXT_WEIGHTS:
            key = (w_form, w_context)
            blends[key] = blend_and_calibrate(
                w_form, w_context, oof, logistic, form, context, data
            )
            print(f"blend v17={1 - w_form - w_context:.2f} form={w_form:.2f} "
                  f"context={w_context:.2f} calibrated", flush=True)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blends[(BASELINE[0], BASELINE[1])] + BASELINE[2] * unit_correction, 0, 1
    )
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for (w_form, w_context), blend in blends.items():
        for drift in DRIFT_WEIGHTS:
            if (w_form, w_context, drift) == BASELINE:
                continue
            label = f"form{w_form:.2f}_ctx{w_context:.2f}_drift{drift:.2f}"
            candidates[label] = np.clip(blend + drift * unit_correction, 0, 1)
            development[label] = development_metrics(reference, candidates[label])

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
        "experiment": "V96_v93_weight_reoptimization",
        "baseline": "V93 (form 0.32 / context 0.13 / drift 0.20)",
        "baseline_public_score": 951.1067955895,
        "rationale": (
            "V93 materially improved the Form model but kept V41's weights. The same "
            "situation after V38 was worth +6.54 leaderboard points when V41 refitted them."
        ),
        "form_weights": list(FORM_WEIGHTS),
        "context_weights": list(CONTEXT_WEIGHTS),
        "drift_weights": list(DRIFT_WEIGHTS),
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": eligible,
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
    joblib.dump({"blends": {f"{k[0]:.2f}_{k[1]:.2f}": v for k, v in blends.items()},
                 "baseline": baseline, "unit_correction": unit_correction},
                PREDICTIONS, compress=3)

    print("\ntop 15 by 2024 development gain (vs V93):")
    print(f"{'candidate':>34} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'worst':>12}")
    ranked = sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"])
    for label, r in ranked[:15]:
        sg = r["season_gain_development"]
        print(f"{label:>34} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e} "
              f"{r['worst_monthly_gain']:12.3e}")
    print(f"\neligible ({len(eligible)}): {eligible}")
    print(f"promoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
