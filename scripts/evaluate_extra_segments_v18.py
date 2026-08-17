"""Search additional OOF segment corrections on top of V17 blended predictions.

V17 baseline: 95% V11 tree ensemble + 5% Logistic (C=0.3), OOF-fixed
count and pitcher_count lookups, and global shift.
This script recomputes residuals from the V17 blended OOF and searches
for extra segment corrections (base_state, count_hand, etc.) that
improve Brier score in both 2023 and 2024 simultaneously.
"""

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


GROUPS = {
    "base_state": ["base_state"],
    "count_hand": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "outs": ["outs_before"],
    "inning_bucket": ["inning_bucket"],
    "month": ["game_month"],
    "top_bottom": ["top_bottom"],
    "batter_count": ["batter_id", "balls_before", "strikes_before"],
    "pitcher_team": ["pitcher_team_id"],
    "batter_team": ["batter_team_id"],
}
SMOOTHING = [50, 100, 300, 1000, 3000, 10000]
STRENGTHS = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

LOGISTIC_C = "0.3"
LOGISTIC_WEIGHT = 0.05
COUNT_SMOOTHING = 500
COUNT_WEIGHT = 0.75
PITCHER_COUNT_SMOOTHING = 300
PITCHER_COUNT_WEIGHT = 0.25


def add_features(frame):
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], [0, 3, 6, 9, np.inf], labels=False, include_lowest=True
    ).astype("int8")
    return output


def v17_prediction(oof_item, logistic, year):
    """Return V17 blended prediction for a single OOF fold."""
    tree_pred = v11_prediction(oof_item)
    logistic_pred = logistic[str(year)]
    return (1 - LOGISTIC_WEIGHT) * tree_pred + LOGISTIC_WEIGHT * logistic_pred


def make_fold(oof, logistic, frame, history_years, valid_year):
    """Build training residuals and validation baseline from V17 OOF."""
    train_y, train_p, train_indices = [], [], []
    for year in history_years:
        item = oof[str(year)]
        train_y.append(item["target"].astype(float))
        train_p.append(v17_prediction(item, logistic, year))
        train_indices.append(item["row_index"])

    train_y = np.concatenate(train_y)
    train_p = np.concatenate(train_p)
    train_index = np.concatenate(train_indices)
    train_frame = frame.loc[train_index]
    residual = train_y - train_p

    # Recompute count / pitcher_count corrections from V17 residuals
    count_corr = segment_correction(
        train_frame, residual, frame.loc[oof[str(valid_year)]["row_index"]],
        ["balls_before", "strikes_before"], COUNT_SMOOTHING,
    )
    pitcher_count_corr = segment_correction(
        train_frame, residual, frame.loc[oof[str(valid_year)]["row_index"]],
        ["pitcher_id", "balls_before", "strikes_before"], PITCHER_COUNT_SMOOTHING,
    )

    valid = oof[str(valid_year)]
    valid_y = valid["target"].astype(float)
    valid_p = v17_prediction(valid, logistic, valid_year)
    valid_frame = frame.loc[valid["row_index"]]

    # V17 baseline prediction (shift + count + pitcher_count already baked in)
    shift = float(residual.mean())
    baseline_pred = np.clip(
        valid_p + shift
        + COUNT_WEIGHT * count_corr
        + PITCHER_COUNT_WEIGHT * pitcher_count_corr,
        0, 1,
    )

    # Extra corrections (search space)
    corrections = {
        (name, smoothing): segment_correction(
            train_frame, residual - shift, valid_frame, columns, smoothing
        )
        for name, columns in GROUPS.items()
        for smoothing in SMOOTHING
    }
    return valid_y, baseline_pred, corrections, train_frame, residual - shift, valid_frame


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")[LOGISTIC_C]
    raw = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    frame = add_features(raw)

    folds = {
        2023: make_fold(oof, logistic, frame, [2022], 2023),
        2024: make_fold(oof, logistic, frame, [2022, 2023], 2024),
    }

    baseline = {
        year: float(brier_score_loss(fold[0], fold[1]))
        for year, fold in folds.items()
    }
    print(f"V17 baseline  2023={baseline[2023]:.9f}  2024={baseline[2024]:.9f}")

    singles = []
    for name in GROUPS:
        for smoothing in SMOOTHING:
            for strength in STRENGTHS:
                scores = {
                    year: float(brier_score_loss(
                        fold[0],
                        np.clip(fold[1] + strength * fold[2][(name, smoothing)], 0, 1),
                    ))
                    for year, fold in folds.items()
                }
                gains = {year: baseline[year] - scores[year] for year in folds}
                if all(v > 0 for v in gains.values()):
                    singles.append({
                        "name": name, "smoothing": smoothing, "strength": strength,
                        "2023_brier": scores[2023], "2024_brier": scores[2024],
                        "2023_gain": gains[2023], "2024_gain": gains[2024],
                    })

    rank = lambda x: (min(x["2023_gain"], x["2024_gain"]),
                      x["2023_gain"] + x["2024_gain"])
    singles.sort(key=rank, reverse=True)

    best_by_group = {}
    for item in singles:
        best_by_group.setdefault(item["name"], item)
    top_groups = list(best_by_group)[:6]

    mixtures = []
    for left, right in combinations(top_groups, 2):
        a, b = best_by_group[left], best_by_group[right]
        scores = {}
        for year, fold in folds.items():
            correction = (
                a["strength"] * fold[2][(left, a["smoothing"])]
                + b["strength"] * fold[2][(right, b["smoothing"])]
            )
            scores[year] = float(brier_score_loss(
                fold[0], np.clip(fold[1] + correction, 0, 1)
            ))
        gains = {year: baseline[year] - scores[year] for year in folds}
        if all(v > 0 for v in gains.values()):
            mixtures.append({
                "left": a, "right": b,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain": gains[2023], "2024_gain": gains[2024],
            })

    all_candidates = singles + mixtures
    all_candidates.sort(key=rank, reverse=True)

    output = {
        "baseline_v17": {str(year): value for year, value in baseline.items()},
        "accepted_single_count": len(singles),
        "accepted_mixture_count": len(mixtures),
        "best": all_candidates[0] if all_candidates else None,
        "top10": all_candidates[:10],
    }
    Path("artifacts/v18_extra_segment_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
