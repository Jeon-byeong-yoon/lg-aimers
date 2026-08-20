"""Blend binned Logistic candidates into V25 with expanding calibration."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


WEIGHTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def v25_prediction(item, logistic, form, context):
    v17 = 0.95 * v11_prediction(item) + 0.05 * logistic
    return 0.75 * v17 + 0.16 * form + 0.09 * context


def fold_score(oof, logistic, form, context, candidate, frame, history_years, valid_year, weight):
    train_y, train_p, indices = [], [], []
    for year in history_years:
        item = oof[str(year)]
        base = v25_prediction(item, logistic[str(year)], form[str(year)], context[str(year)])
        train_y.append(item["target"].astype(float))
        train_p.append((1 - weight) * base + weight * candidate[str(year)])
        indices.append(item["row_index"])
    train_y, train_p = np.concatenate(train_y), np.concatenate(train_p)
    residual = train_y - train_p
    train_frame = frame.loc[np.concatenate(indices)]
    valid = oof[str(valid_year)]
    valid_base = v25_prediction(
        valid, logistic[str(valid_year)], form[str(valid_year)], context[str(valid_year)]
    )
    valid_p = (1 - weight) * valid_base + weight * candidate[str(valid_year)]
    valid_frame = frame.loc[valid["row_index"]]
    count = segment_correction(
        train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
    )
    pitcher = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(valid_p + residual.mean() + 0.75 * count + 0.25 * pitcher, 0, 1)
    return float(brier_score_loss(valid["target"].astype(float), prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v24_contextual_trackman_predictions.joblib")
    candidates = joblib.load("artifacts/v30_binned_logistic_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    any_candidate = next(iter(candidates.values()))
    baseline = {
        year: fold_score(
            oof, logistic, form, context, any_candidate, frame,
            [2022] if year == 2023 else [2022, 2023], year, 0,
        ) for year in (2023, 2024)
    }
    results = []
    for name, candidate in candidates.items():
        for weight in WEIGHTS:
            scores = {
                year: fold_score(
                    oof, logistic, form, context, candidate, frame,
                    [2022] if year == 2023 else [2022, 2023], year, weight,
                ) for year in (2023, 2024)
            }
            results.append({
                "variant": name, "weight": weight,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain_vs_v25": baseline[2023] - scores[2023],
                "2024_gain_vs_v25": baseline[2024] - scores[2024],
            })
    accepted = [x for x in results if min(x["2023_gain_vs_v25"], x["2024_gain_vs_v25"]) > 0]
    rank = lambda x: (
        min(x["2023_gain_vs_v25"], x["2024_gain_vs_v25"]),
        x["2023_gain_vs_v25"] + x["2024_gain_vs_v25"],
    )
    accepted.sort(key=rank, reverse=True)
    results.sort(key=rank, reverse=True)
    output = {
        "baseline_v25": {str(k): v for k, v in baseline.items()},
        "accepted_count": len(accepted), "best": accepted[0] if accepted else None,
        "top10_overall": results[:10],
    }
    Path("artifacts/v30_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
