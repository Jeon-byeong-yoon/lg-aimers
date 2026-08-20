"""Retune leakage-safe residual calibration for the enhanced V26 blend."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


BLEND = {"v17": 0.69, "form": 0.12, "context_v26": 0.19}
COUNT_SMOOTHING = [100, 200, 500, 1000]
PITCHER_SMOOTHING = [100, 200, 300, 500, 1000]
COUNT_WEIGHTS = [0.40, 0.55, 0.70, 0.85, 1.00]
PITCHER_WEIGHTS = [0.40, 0.475, 0.55, 0.625, 0.70]
V25_BRIER = {2023: 0.2533208415598194, 2024: 0.24784544408284379}


def blended_prediction(oof_item, logistic, form, context):
    v17 = 0.95 * v11_prediction(oof_item) + 0.05 * logistic
    return BLEND["v17"] * v17 + BLEND["form"] * form + BLEND["context_v26"] * context


def make_fold(oof, logistic, form, context, frame, history_years, valid_year):
    train_y, train_p, train_indices = [], [], []
    for year in history_years:
        item = oof[str(year)]
        train_y.append(item["target"].astype(float))
        train_p.append(blended_prediction(
            item, logistic[str(year)], form[str(year)], context[str(year)]
        ))
        train_indices.append(item["row_index"])
    train_y = np.concatenate(train_y)
    train_p = np.concatenate(train_p)
    residual = train_y - train_p
    train_frame = frame.loc[np.concatenate(train_indices)]
    valid = oof[str(valid_year)]
    valid_p = blended_prediction(
        valid, logistic[str(valid_year)], form[str(valid_year)], context[str(valid_year)]
    )
    valid_frame = frame.loc[valid["row_index"]]
    counts = {
        smoothing: segment_correction(
            train_frame, residual, valid_frame,
            ["balls_before", "strikes_before"], smoothing,
        ) for smoothing in COUNT_SMOOTHING
    }
    pitchers = {
        smoothing: segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], smoothing,
        ) for smoothing in PITCHER_SMOOTHING
    }
    return valid["target"].astype(float), valid_p + residual.mean(), counts, pitchers


def score(fold, count_smoothing, pitcher_smoothing, count_weight, pitcher_weight):
    target, shifted, counts, pitchers = fold
    prediction = np.clip(
        shifted
        + count_weight * counts[count_smoothing]
        + pitcher_weight * pitchers[pitcher_smoothing],
        0, 1,
    )
    return float(brier_score_loss(target, prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v26_contextual_trackman_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    folds = {
        2023: make_fold(oof, logistic, form, context, frame, [2022], 2023),
        2024: make_fold(oof, logistic, form, context, frame, [2022, 2023], 2024),
    }
    candidates = []
    for cs in COUNT_SMOOTHING:
        for ps in PITCHER_SMOOTHING:
            for cw in COUNT_WEIGHTS:
                for pw in PITCHER_WEIGHTS:
                    scores = {year: score(fold, cs, ps, cw, pw) for year, fold in folds.items()}
                    candidates.append({
                        "count_smoothing": cs, "pitcher_smoothing": ps,
                        "count_weight": cw, "pitcher_weight": pw,
                        "2023_brier": scores[2023], "2024_brier": scores[2024],
                        "2023_gain_vs_v25": V25_BRIER[2023] - scores[2023],
                        "2024_gain_vs_v25": V25_BRIER[2024] - scores[2024],
                    })
    accepted = [item for item in candidates if min(
        item["2023_gain_vs_v25"], item["2024_gain_vs_v25"]
    ) > 0]
    rank = lambda item: (
        min(item["2023_gain_vs_v25"], item["2024_gain_vs_v25"]),
        item["2023_gain_vs_v25"] + item["2024_gain_vs_v25"],
    )
    accepted.sort(key=rank, reverse=True)
    output = {
        "blend": BLEND,
        "baseline_v25": {str(k): v for k, v in V25_BRIER.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10": accepted[:10],
    }
    Path("artifacts/v27_calibration_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
