"""Fine-tune the two proven V12 segment corrections on chronological folds."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction


COUNT_SMOOTHING = [10, 25, 50, 100, 200, 500]
PITCHER_SMOOTHING = [100, 200, 300, 500, 1000, 2000]
COUNT_WEIGHTS = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
PITCHER_WEIGHTS = [0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]


def fold_data(oof, frame, history_years, valid_year):
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
    shifted = valid_p + residual.mean()
    count = {
        smoothing: segment_correction(
            train_frame, residual, valid_frame,
            ["balls_before", "strikes_before"], smoothing,
        ) for smoothing in COUNT_SMOOTHING
    }
    pitcher = {
        smoothing: segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], smoothing,
        ) for smoothing in PITCHER_SMOOTHING
    }
    return valid_y, shifted, count, pitcher


def score(fold, cs, ps, cw, pw):
    target, shifted, count, pitcher = fold
    prediction = np.clip(shifted + cw * count[cs] + pw * pitcher[ps], 0, 1)
    return float(brier_score_loss(target, prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    folds = {
        2023: fold_data(oof, frame, [2022], 2023),
        2024: fold_data(oof, frame, [2022, 2023], 2024),
    }
    baseline = {
        year: score(fold, 50, 300, 0.5, 0.125)
        for year, fold in folds.items()
    }
    accepted = []
    for cs in COUNT_SMOOTHING:
        for ps in PITCHER_SMOOTHING:
            for cw in COUNT_WEIGHTS:
                for pw in PITCHER_WEIGHTS:
                    scores = {year: score(fold, cs, ps, cw, pw)
                              for year, fold in folds.items()}
                    gains = {year: baseline[year] - scores[year] for year in folds}
                    if all(gain > 0 for gain in gains.values()):
                        accepted.append({
                            "count_smoothing": cs, "pitcher_smoothing": ps,
                            "count_weight": cw, "pitcher_weight": pw,
                            "2023_brier": scores[2023], "2024_brier": scores[2024],
                            "2023_gain": gains[2023], "2024_gain": gains[2024],
                        })
    accepted.sort(
        key=lambda x: (min(x["2023_gain"], x["2024_gain"]),
                       x["2023_gain"] + x["2024_gain"]), reverse=True
    )
    output = {
        "baseline_v12": {str(year): value for year, value in baseline.items()},
        "candidate_count": len(COUNT_SMOOTHING) * len(PITCHER_SMOOTHING)
                           * len(COUNT_WEIGHTS) * len(PITCHER_WEIGHTS),
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10": accepted[:10],
    }
    Path("artifacts/v14_segment_tuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
