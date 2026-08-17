"""Search an additional stable OOF segment correction on top of V14."""

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction


GROUPS = {
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "count_hand": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "base_state": ["base_state"],
    "outs": ["outs_before"],
    "inning_bucket": ["inning_bucket"],
    "month": ["game_month"],
    "game_type": ["game_type"],
    "top_bottom": ["top_bottom"],
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "batter_count": ["batter_id", "balls_before", "strikes_before"],
    "pitcher_team": ["pitcher_team_id"],
    "batter_team": ["batter_team_id"],
}
SMOOTHING = [50, 100, 300, 1000, 3000, 10000]
STRENGTHS = [0.1, 0.25, 0.5, 0.75, 1.0]


def add_features(frame):
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], [0, 3, 6, 9, np.inf], labels=False, include_lowest=True
    ).astype("int8")
    return output


def make_fold(oof, frame, history_years, valid_year):
    history = [oof[str(year)] for year in history_years]
    train_y = np.concatenate([item["target"] for item in history]).astype(float)
    train_p = np.concatenate([raw_prediction(item) for item in history])
    train_index = np.concatenate([item["row_index"] for item in history])
    valid = oof[str(valid_year)]
    valid_y = valid["target"].astype(float)
    valid_p = raw_prediction(valid)
    train_frame = frame.loc[train_index]
    valid_frame = frame.loc[valid["row_index"]]
    residual = train_y - train_p
    count = segment_correction(
        train_frame, residual, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    pitcher_count = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    baseline = np.clip(
        valid_p + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1
    )
    corrections = {
        (name, smoothing): segment_correction(
            train_frame, residual, valid_frame, columns, smoothing
        )
        for name, columns in GROUPS.items()
        for smoothing in SMOOTHING
    }
    return valid_y, baseline, corrections


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    raw = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    frame = add_features(raw)
    folds = {
        2023: make_fold(oof, frame, [2022], 2023),
        2024: make_fold(oof, frame, [2022, 2023], 2024),
    }
    baseline = {
        year: float(brier_score_loss(fold[0], fold[1]))
        for year, fold in folds.items()
    }
    singles = []
    for name in GROUPS:
        for smoothing in SMOOTHING:
            for strength in STRENGTHS:
                scores = {
                    year: float(brier_score_loss(
                        fold[0], np.clip(
                            fold[1] + strength * fold[2][(name, smoothing)], 0, 1
                        ),
                    )) for year, fold in folds.items()
                }
                gains = {year: baseline[year] - scores[year] for year in folds}
                if all(value > 0 for value in gains.values()):
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
            correction = 0.5 * (
                a["strength"] * fold[2][(left, a["smoothing"])]
                + b["strength"] * fold[2][(right, b["smoothing"])]
            )
            scores[year] = float(brier_score_loss(
                fold[0], np.clip(fold[1] + correction, 0, 1)
            ))
        gains = {year: baseline[year] - scores[year] for year in folds}
        if all(value > 0 for value in gains.values()):
            mixtures.append({
                "left": a, "right": b,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain": gains[2023], "2024_gain": gains[2024],
            })
    all_candidates = singles + mixtures
    all_candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v14": {str(year): value for year, value in baseline.items()},
        "accepted_single_count": len(singles),
        "accepted_mixture_count": len(mixtures),
        "best": all_candidates[0] if all_candidates else None,
        "top10": all_candidates[:10],
    }
    Path("artifacts/v15_extra_segment_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
