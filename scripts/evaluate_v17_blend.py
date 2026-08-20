"""Blend regularized Logistic candidates into V14 with recalibrated OOF lookups."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]


def fold_score(oof, candidate, frame, history_years, valid_year, weight):
    train_y, train_base, train_new, indices = [], [], [], []
    for year in history_years:
        item = oof[str(year)]
        train_y.append(item["target"].astype(float))
        train_base.append(v11_prediction(item))
        train_new.append(candidate[str(year)])
        indices.append(item["row_index"])
    train_y = np.concatenate(train_y)
    train_p = ((1 - weight) * np.concatenate(train_base)
               + weight * np.concatenate(train_new))
    train_frame = frame.loc[np.concatenate(indices)]
    valid = oof[str(valid_year)]
    valid_p = ((1 - weight) * v11_prediction(valid)
               + weight * candidate[str(valid_year)])
    valid_y = valid["target"].astype(float)
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
    prediction = np.clip(
        valid_p + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1
    )
    return float(brier_score_loss(valid_y, prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    candidates = joblib.load("artifacts/v17_logistic_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    any_candidate = next(iter(candidates.values()))
    baseline = {
        2023: fold_score(oof, any_candidate, frame, [2022], 2023, 0),
        2024: fold_score(oof, any_candidate, frame, [2022, 2023], 2024, 0),
    }
    accepted, all_results = [], []
    for c, candidate in candidates.items():
        for weight in WEIGHTS:
            scores = {
                2023: fold_score(oof, candidate, frame, [2022], 2023, weight),
                2024: fold_score(oof, candidate, frame, [2022, 2023], 2024, weight),
            }
            item = {
                "C": float(c), "weight": weight,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain": baseline[2023] - scores[2023],
                "2024_gain": baseline[2024] - scores[2024],
            }
            all_results.append(item)
            if item["2023_gain"] > 0 and item["2024_gain"] > 0:
                accepted.append(item)
    rank = lambda x: (min(x["2023_gain"], x["2024_gain"]),
                      x["2023_gain"] + x["2024_gain"])
    accepted.sort(key=rank, reverse=True)
    all_results.sort(key=rank, reverse=True)
    output = {
        "baseline_v14": {str(k): v for k, v in baseline.items()},
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_accepted": accepted[:10],
        "top5_overall": all_results[:5],
    }
    Path("artifacts/v17_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
