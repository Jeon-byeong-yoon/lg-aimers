"""V92: correct V41 with the current-season reconstruction of pitcher control.

Hypothesis
----------
``asof_pitcher_success_rate`` is career-cumulative, so the league-wide decline of
the control-success rate (0.5647 in 2019 to 0.4861 in 2024) leaves it biased high
by roughly 2.6 percentage points by 2024. Used alone it scores below the constant
baseline in 2023 and 2024. Subtracting the frozen end-of-previous-season career
state recovers the pitcher's current-season-only rate, and the shrunk gap between
the two is a signal V41 cannot see.

The candidate adds ``w * (current_season_rate - career_rate) * n/(n+150)`` to the
finished V41 prediction. ``w`` is fixed in advance from a coarse grid; the graded
sweep is reported so that the choice can be seen to sit on a plateau rather than a
spike. Candidate selection uses 2022, 2023 and 2024 March-August only; the sealed
2024 September-October window is opened once, for the single promoted candidate.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from inseason_asof_features_v92 import (
    add_training_inseason_features,
    drift_correction,
)


OUTPUT = Path("artifacts/v92_inseason_drift_correction_metrics.json")
PREDICTIONS = Path("artifacts/v92_inseason_drift_correction_predictions.joblib")
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PROMOTION_WEIGHT = 0.20
BASE_BRIER = 0.25


def points(gain):
    """Leaderboard points for a Brier gain, using Score = 1e5 * (1 - B/B0)."""
    return float(gain * 100000.0 / BASE_BRIER)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    feature_frame = data.drop(columns=["row_id", "control_success"])

    inseason = add_training_inseason_features(feature_frame)
    audit = {
        "career_rate_mean_by_season": {
            str(int(season)): float(part["asof_pitcher_success_rate"].mean())
            for season, part in inseason.groupby("season")
        },
        "inseason_rate_mean_by_season": {
            str(int(season)): float(part["ins_asof_pitcher_success_rate"].mean())
            for season, part in inseason.groupby("season")
        },
        "target_mean_by_season": {
            str(int(season)): float(part.mean())
            for season, part in data.groupby("season")["control_success"]
        },
    }

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    baseline_parts, correction_parts = [], []
    for year in YEARS:
        prediction, _, _, _ = calibrated_prediction(
            year, oof, logistic, form, context, feature_frame
        )
        rows = inseason.loc[oof[str(year)]["row_index"]]
        baseline_parts.append(prediction)
        correction_parts.append(drift_correction(rows, 1.0))
    baseline = np.concatenate(baseline_parts)
    unit_correction = np.concatenate(correction_parts)

    validation_frame = make_validation_frame()
    if len(validation_frame) != len(baseline):
        raise ValueError("Validation frame and V41 predictions are misaligned")
    if not np.allclose(validation_frame["v41_prediction"].to_numpy(), baseline):
        raise ValueError("V41 baseline reproduction mismatch")

    candidates, development, sweep = {}, {}, []
    for weight in WEIGHTS:
        name = f"drift_w_{weight:.2f}"
        candidates[name] = np.clip(baseline + weight * unit_correction, 0, 1)
        development[name] = development_metrics(validation_frame, candidates[name])
        result = development[name]
        sweep.append({
            "weight": weight,
            "season_gain_development": result["season_gain_development"],
            "season_points_development": {
                season: points(value)
                for season, value in result["season_gain_development"].items()
            },
            "gain_2024_mar_aug": result["gain_2024_mar_aug"],
            "points_2024_mar_aug": points(result["gain_2024_mar_aug"]),
            "gain_2024_jul_aug": result["gain_2024_jul_aug"],
            "monthly_block_win_rate": result["monthly_block_win_rate"],
            "worst_monthly_gain": result["worst_monthly_gain"],
        })

    eligible = [
        name for name, result in development.items()
        if result["season_gain_development"]["2022"] > 0
        and result["season_gain_development"]["2023"] > 0
        and result["gain_2024_mar_aug"] >= 2e-5
        and result["gain_2024_jul_aug"] > 0
        and result["monthly_block_win_rate"] >= 0.75
    ]
    promoted = f"drift_w_{PROMOTION_WEIGHT:.2f}"
    if promoted not in eligible:
        promoted = max(
            eligible,
            key=lambda name: development[name]["gain_2024_mar_aug"],
            default=None,
        )
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {
            "candidate": promoted,
            "comparison": compare_candidate(validation_frame, candidates[promoted]),
        }

    output = {
        "experiment": "V92_inseason_asof_drift_correction",
        "baseline": "V41",
        "baseline_public_score": 900.7385360187,
        "hypothesis": (
            "Career-cumulative asof_pitcher_success_rate is biased by the league-level "
            "decline; the shrunk gap to the recovered current-season rate is signal V41 "
            "cannot represent."
        ),
        "weights": list(WEIGHTS),
        "promotion_weight": PROMOTION_WEIGHT,
        "drift_audit": audit,
        "sweep": sweep,
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
            "row_independent_transform": True,
            "anchor_frozen_at_training_time": True,
            "weight_fixed_before_final_confirmation": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(
        {"baseline": baseline, "unit_correction": unit_correction, "weights": list(WEIGHTS)},
        PREDICTIONS, compress=3,
    )

    print("drift audit (career vs recovered current-season vs target):")
    for season in sorted(audit["target_mean_by_season"]):
        print(f"  {season}: career={audit['career_rate_mean_by_season'][season]:.4f} "
              f"in_season={audit['inseason_rate_mean_by_season'][season]:.4f} "
              f"target={audit['target_mean_by_season'][season]:.4f}")
    print("\nweight sweep (development windows only):")
    header = f"{'w':>6} {'2022 pt':>9} {'2023 pt':>9} {'2024 pt':>9} {'2024 gain':>12} {'blocks':>8} {'worst':>12}"
    print(header)
    for row in sweep:
        sp = row["season_points_development"]
        print(f"{row['weight']:6.2f} {sp['2022']:9.1f} {sp['2023']:9.1f} "
              f"{points(row['gain_2024_mar_aug']):9.1f} {row['gain_2024_mar_aug']:12.3e} "
              f"{row['monthly_block_win_rate']:8.2%} {row['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
