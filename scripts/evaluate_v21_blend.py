"""Test recency-weighted HGB candidates as additions to V17."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v19_blend import fold_score


WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    candidates = joblib.load("artifacts/v21_recency_weighted_hgb_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    any_candidate = next(iter(candidates.values()))
    baseline = {
        2023: fold_score(oof, logistic, any_candidate, frame, [2022], 2023, 0),
        2024: fold_score(oof, logistic, any_candidate, frame, [2022, 2023], 2024, 0),
    }
    accepted, all_results = [], []
    for decay, candidate in candidates.items():
        for weight in WEIGHTS:
            scores = {
                2023: fold_score(oof, logistic, candidate, frame, [2022], 2023, weight),
                2024: fold_score(oof, logistic, candidate, frame, [2022, 2023], 2024, weight),
            }
            item = {
                "decay": float(decay), "weight": weight,
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
        "baseline_v17": {str(k): v for k, v in baseline.items()},
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top5_overall": all_results[:5],
    }
    Path("artifacts/v21_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
