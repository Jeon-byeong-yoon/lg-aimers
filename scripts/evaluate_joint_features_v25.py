"""Jointly blend stable-form and contextual-Trackman components into V17."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_v19_blend import fold_score


FORM_WEIGHTS = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
TRACKMAN_WEIGHTS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    trackman = joblib.load("artifacts/v24_contextual_trackman_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    baseline = {
        2023: fold_score(oof, logistic, form, frame, [2022], 2023, 0),
        2024: fold_score(oof, logistic, form, frame, [2022, 2023], 2024, 0),
    }
    accepted, all_results = [], []
    for form_weight in FORM_WEIGHTS:
        for trackman_weight in TRACKMAN_WEIGHTS:
            total = form_weight + trackman_weight
            candidate = {
                year: (form_weight * form[year] + trackman_weight * trackman[year]) / total
                for year in form
            }
            scores = {
                2023: fold_score(oof, logistic, candidate, frame, [2022], 2023, total),
                2024: fold_score(oof, logistic, candidate, frame, [2022, 2023], 2024, total),
            }
            item = {
                "form_weight": form_weight, "trackman_weight": trackman_weight,
                "v17_weight": 1 - total,
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
    eligible = [
        item for item in accepted
        if item["2024_gain"] >= 0.000005
        and (item["2023_gain"] + item["2024_gain"]) / 2 >= 0.00002
    ]
    eligible.sort(key=lambda item: item["2023_gain"] + item["2024_gain"], reverse=True)
    output = {
        "baseline_v17": {str(k): v for k, v in baseline.items()},
        "candidate_count": len(all_results),
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "eligible_count": len(eligible),
        "best_eligible": eligible[0] if eligible else None,
        "top10": accepted[:10],
    }
    Path("artifacts/v25_joint_feature_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
