"""V100: broaden the post-hoc drift term beyond the success-rate delta.

The drift term is the only component that extrapolates the league level. V93
proved this: with the in-season features already inside the Form model, deleting
this one scalar term still collapsed the monthly block win rate from 100% to 70%.
Trees cannot extrapolate past `season=2024`; this term can.

Yet it reads exactly one of the eighteen reconstruction signals,
`dlt_asof_pitcher_success_rate`. The other deltas map directly onto the three
official failure definitions and are unused in this role:

  middle_rate   -> "스트라이크존 가운데 부근"      higher is worse
  ball_rate     -> "스트라이크존에서 크게 벗어남"  higher is worse
  reverse_rate  -> "포수 요구 방향과 반대"          higher is worse

Sub-weights inside each composite are fixed a priori — equal weighting, or a
plain 3:1 split for the pitcher/batter mix — and only the overall scale is swept.
Fitting the sub-weights on the validation seasons would be fitting noise, which is
what sank V78 and V80.

Calibration depends on the blend, not on this term, so a single calibration is
reused across every candidate.
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
from inseason_asof_features_v92 import add_training_inseason_features


OUTPUT = Path("artifacts/v100_drift_term_composition_metrics.json")
PREDICTIONS = Path("artifacts/v100_drift_term_composition_predictions.joblib")
W_FORM, W_CONTEXT = 0.52, 0.21
BASELINE_SCALE = 0.15
SCALES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
MIN_GAIN = 1.06e-5
RATIO = 0.932


def points(gain):
    return float(gain * 100000.0 / 0.25)


def composites(frame):
    """Signed combinations of the reconstruction deltas, sub-weights fixed here."""
    def d(name):
        return np.nan_to_num(
            frame[f"dlt_asof_{name}"].to_numpy(dtype=float), nan=0.0
        )
    success = d("pitcher_success_rate")
    middle = d("pitcher_middle_rate")
    ball = d("pitcher_ball_rate")
    reverse = d("pitcher_reverse_rate")
    batter = d("batter_success_rate")
    failure = -(middle + ball + reverse) / 3.0
    return {
        "success": success,
        "success_middle": 0.5 * success - 0.5 * middle,
        "failure_modes": failure,
        "combined": 0.5 * success + 0.5 * failure,
        "pitcher_batter": 0.75 * success + 0.25 * batter,
    }


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
    rows = pd.concat([inseason.loc[oof[str(year)]["row_index"]] for year in YEARS])
    reliability = np.nan_to_num(
        rows["ins_pitcher_reliability"].to_numpy(dtype=float), nan=0.0
    )
    terms = {name: value * reliability for name, value in composites(rows).items()}
    stats = {
        name: {"mean": float(value.mean()), "std": float(value.std())}
        for name, value in terms.items()
    }

    blend = blend_and_calibrate(W_FORM, W_CONTEXT, oof, logistic, form, context, frame)
    validation_frame = make_validation_frame()
    baseline = np.clip(blend + BASELINE_SCALE * terms["success"], 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for name, term in terms.items():
        for scale in SCALES:
            if name == "success" and scale == BASELINE_SCALE:
                continue
            label = f"{name}_w{scale:.2f}"
            candidates[label] = np.clip(blend + scale * term, 0, 1)
            development[label] = development_metrics(reference, candidates[label])

    eligible = [
        label for label, r in development.items()
        if r["season_gain_development"]["2022"] > -1e-5
        and r["season_gain_development"]["2023"] > -1e-5
        and r["gain_2024_mar_aug"] >= MIN_GAIN
        and r["gain_2024_jul_aug"] > 0
        and r["monthly_block_win_rate"] >= 0.75
    ]

    # Plateau requirement on the scale axis: the neighbouring scales of the same
    # composite must also pass, so the choice is not a single lucky point.
    def neighbours(label):
        name, scale = label.rsplit("_w", 1)
        index = SCALES.index(float(scale))
        out = []
        for step in (-1, 1):
            if 0 <= index + step < len(SCALES):
                out.append(f"{name}_w{SCALES[index + step]:.2f}")
        return [n for n in out if n in development or n == f"success_w{BASELINE_SCALE:.2f}"]

    plateau = [
        label for label in eligible
        if all(n in eligible or n == f"success_w{BASELINE_SCALE:.2f}"
               for n in neighbours(label))
    ]
    pool = plateau or eligible
    promoted = max(pool, key=lambda l: development[l]["gain_2024_mar_aug"], default=None)
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {"candidate": promoted,
                              "comparison": compare_candidate(reference, candidates[promoted])}

    output = {
        "experiment": "V100_drift_term_composition",
        "baseline": "V96 (success delta only, scale 0.15)",
        "baseline_public_score": 962.8787800874,
        "rationale": (
            "The drift term is the only component that extrapolates the league level, "
            "but it reads only one of the eighteen reconstruction signals. The middle, "
            "ball and reverse deltas map onto the three official failure definitions."
        ),
        "blend_weights": {"v17": round(1 - W_FORM - W_CONTEXT, 2),
                          "form": W_FORM, "context": W_CONTEXT},
        "composites": {
            "success": "dlt_success",
            "success_middle": "0.5*dlt_success - 0.5*dlt_middle",
            "failure_modes": "-(dlt_middle + dlt_ball + dlt_reverse)/3",
            "combined": "0.5*dlt_success + 0.5*failure_modes",
            "pitcher_batter": "0.75*dlt_success + 0.25*dlt_batter_success",
        },
        "term_statistics": stats,
        "scales": list(SCALES),
        "selection_rule": "pass all criteria, and neighbouring scales must pass too",
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": sorted(eligible),
        "plateau_candidates": sorted(plateau),
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "sub_weights_fixed_a_priori": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"terms": terms, "blend": blend, "baseline": baseline},
                PREDICTIONS, compress=3)

    print("term statistics (after reliability shrinkage):")
    for name, s in stats.items():
        print(f"  {name:16s} mean={s['mean']:+.6f} std={s['std']:.6f}")
    print("\ngains vs V96 baseline, development windows only:")
    print(f"{'candidate':>24} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'plateau':>8}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"])[:18]:
        sg = r["season_gain_development"]
        print(f"{label:>24} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e} "
              f"{'YES' if label in plateau else '-':>8}")
    print(f"\neligible={len(eligible)}  plateau={len(plateau)}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
