"""V96b: extend the V96 weight grid, whose optimum sat on the boundary.

V96 promoted form=0.48 / context=0.17 / drift=0.15 — the maximum form weight,
the maximum context weight and the minimum drift weight in that grid. An optimum
on three boundaries is not an optimum, so the grid is extended outward here.

Selection is deliberately not "take the maximum". Scanning many weight
combinations against one development window invites fitting that window, so the
chosen candidate must sit on a plateau: its immediate neighbours in the grid must
also pass, and among passing candidates the most interior one is preferred.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v96b_weight_grid_extension_metrics.json")
PREDICTIONS = Path("artifacts/v96b_weight_grid_extension_predictions.joblib")
FORM_WEIGHTS = (0.44, 0.48, 0.52, 0.56, 0.60)
CONTEXT_WEIGHTS = (0.13, 0.17, 0.21, 0.25)
DRIFT_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
BASELINE = (0.32, 0.13, 0.20)
MIN_GAIN = 1.06e-5
RATIO = 0.944


def points(gain):
    return float(gain * 100000.0 / 0.25)


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
    unit = np.concatenate([
        drift_correction(inseason.loc[oof[str(year)]["row_index"]], 1.0) for year in YEARS
    ])

    grid = {(BASELINE[0], BASELINE[1])}
    grid.update((f, c) for f in FORM_WEIGHTS for c in CONTEXT_WEIGHTS)
    blends = {}
    for w_form, w_context in sorted(grid):
        blends[(w_form, w_context)] = blend_and_calibrate(
            w_form, w_context, oof, logistic, form, context, data
        )
        print(f"blend v17={1 - w_form - w_context:.2f} form={w_form:.2f} "
              f"context={w_context:.2f} calibrated", flush=True)

    validation_frame = make_validation_frame()
    baseline = np.clip(blends[(BASELINE[0], BASELINE[1])] + BASELINE[2] * unit, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development, table = {}, {}, {}
    for (w_form, w_context), blend in blends.items():
        if (w_form, w_context) == (BASELINE[0], BASELINE[1]):
            continue
        for drift in DRIFT_WEIGHTS:
            label = f"form{w_form:.2f}_ctx{w_context:.2f}_drift{drift:.2f}"
            candidates[label] = np.clip(blend + drift * unit, 0, 1)
            development[label] = development_metrics(reference, candidates[label])
            table[(w_form, w_context, drift)] = label

    def passes(label):
        r = development[label]
        return (r["season_gain_development"]["2022"] > -1e-5
                and r["season_gain_development"]["2023"] > -1e-5
                and r["gain_2024_mar_aug"] >= MIN_GAIN
                and r["gain_2024_jul_aug"] > 0
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in development if passes(label)]

    # Plateau requirement: every immediate neighbour that exists must also pass.
    def neighbours(key):
        w_form, w_context, drift = key
        out = []
        for axis, values in ((0, FORM_WEIGHTS), (1, CONTEXT_WEIGHTS), (2, DRIFT_WEIGHTS)):
            if key[axis] not in values:
                continue
            index = values.index(key[axis])
            for step in (-1, 1):
                if 0 <= index + step < len(values):
                    candidate = list(key)
                    candidate[axis] = values[index + step]
                    if tuple(candidate) in table:
                        out.append(tuple(candidate))
        return out

    plateau = []
    for key, label in table.items():
        if label not in eligible:
            continue
        neighbour_labels = [table[n] for n in neighbours(key)]
        if neighbour_labels and all(n in eligible for n in neighbour_labels):
            plateau.append((key, label))

    def interiority(key):
        """Distance from the nearest grid edge on each axis; larger is safer."""
        w_form, w_context, drift = key
        score = 0
        for value, values in ((w_form, FORM_WEIGHTS), (w_context, CONTEXT_WEIGHTS),
                              (drift, DRIFT_WEIGHTS)):
            if value in values:
                index = values.index(value)
                score += min(index, len(values) - 1 - index)
        return score

    promoted = None
    if plateau:
        promoted = max(
            plateau,
            key=lambda item: (interiority(item[0]),
                              development[item[1]]["gain_2024_mar_aug"]),
        )[1]
    elif eligible:
        promoted = max(eligible, key=lambda l: development[l]["gain_2024_mar_aug"])

    final_confirmation = None
    if promoted is not None:
        final_confirmation = {"candidate": promoted,
                              "comparison": compare_candidate(reference, candidates[promoted])}

    output = {
        "experiment": "V96b_weight_grid_extension",
        "baseline": "V93 (form 0.32 / context 0.13 / drift 0.20)",
        "baseline_public_score": 951.1067955895,
        "form_weights": list(FORM_WEIGHTS),
        "context_weights": list(CONTEXT_WEIGHTS),
        "drift_weights": list(DRIFT_WEIGHTS),
        "selection_rule": (
            "must pass all development criteria, all existing immediate grid "
            "neighbours must also pass, then prefer the most interior point"
        ),
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": sorted(eligible),
        "plateau_candidates": sorted(label for _, label in plateau),
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "calibration_refitted_per_blend": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"blends": {f"{k[0]:.2f}_{k[1]:.2f}": v for k, v in blends.items()},
                 "baseline": baseline, "unit_correction": unit}, PREDICTIONS, compress=3)

    print("\ntop 18 by 2024 development gain (vs V93):")
    print(f"{'candidate':>34} {'v17':>5} {'2022':>7} {'2023':>7} {'2024pt':>7} {'LB':>6} {'blocks':>7} {'jul_aug':>11} {'plateau':>8}")
    names = {label for _, label in plateau}
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"])[:18]:
        sg = r["season_gain_development"]
        wf = float(label.split("form")[1][:4]); wc = float(label.split("ctx")[1][:4])
        print(f"{label:>34} {1 - wf - wc:5.2f} {points(sg['2022']):7.1f} {points(sg['2023']):7.1f} "
              f"{points(r['gain_2024_mar_aug']):7.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:6.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {r['gain_2024_jul_aug']:11.3e} "
              f"{'YES' if label in names else '-':>8}")
    print(f"\neligible={len(eligible)}  plateau={len(plateau)}")
    print(f"promoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
