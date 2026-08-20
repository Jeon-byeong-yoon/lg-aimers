"""Test whether TargetEncoder HGB adds diversity to the V14 ensemble."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


BLEND_WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]


def evaluate(oof, candidate, frame, history_years, valid_year, weight):
    history_v11, history_new, train_y, indices = [], [], [], []
    for year in history_years:
        item = oof[str(year)]
        history_v11.append(v11_prediction(item))
        history_new.append(candidate[str(year)]["prediction"])
        train_y.append(item["target"].astype(float))
        indices.append(item["row_index"])
    train_p = (1 - weight) * np.concatenate(history_v11) + weight * np.concatenate(history_new)
    train_y = np.concatenate(train_y)
    train_frame = frame.loc[np.concatenate(indices)]
    valid = oof[str(valid_year)]
    valid_p = ((1 - weight) * v11_prediction(valid)
               + weight * candidate[str(valid_year)]["prediction"])
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
    candidate = joblib.load("artifacts/v16_target_encoder_hgb_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    baseline = {
        2023: evaluate(oof, candidate, frame, [2022], 2023, 0.0),
        2024: evaluate(oof, candidate, frame, [2022, 2023], 2024, 0.0),
    }
    results = []
    for weight in BLEND_WEIGHTS:
        scores = {
            2023: evaluate(oof, candidate, frame, [2022], 2023, weight),
            2024: evaluate(oof, candidate, frame, [2022, 2023], 2024, weight),
        }
        item = {
            "weight": weight,
            "2023_brier": scores[2023], "2024_brier": scores[2024],
            "2023_gain": baseline[2023] - scores[2023],
            "2024_gain": baseline[2024] - scores[2024],
        }
        if item["2023_gain"] > 0 and item["2024_gain"] > 0:
            results.append(item)
    results.sort(key=lambda x: (min(x["2023_gain"], x["2024_gain"]),
                                x["2023_gain"] + x["2024_gain"]), reverse=True)
    output = {
        "baseline_v14": {str(k): v for k, v in baseline.items()},
        "accepted_count": len(results),
        "best": results[0] if results else None,
        "accepted": results,
    }
    Path("artifacts/v16_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
