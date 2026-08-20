"""Retune V31 form/context weights under three-season robustness constraints."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v19_blend import fold_score


FORM_WEIGHTS = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
CONTEXT_WEIGHTS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
V31_BRIER = {2023: 0.25326829650157456, 2024: 0.24783690211517923}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    form = removed["no_trackman_std"]["form"]
    context = removed["no_matchup_hte"]["context"]
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    item22 = oof["2022"]
    baseline22_prediction = (
        0.75 * v17_prediction(item22, logistic["2022"])
        + 0.16 * form["2022"] + 0.09 * context["2022"]
    )
    baseline22 = float(brier_score_loss(item22["target"], baseline22_prediction))
    candidates = []
    for form_weight in FORM_WEIGHTS:
        for context_weight in CONTEXT_WEIGHTS:
            total = form_weight + context_weight
            v17_weight = 1.0 - total
            component = {
                year: (
                    form_weight * form[year] + context_weight * context[year]
                ) / total
                for year in form
            }
            scores = {
                2023: fold_score(
                    oof, logistic, component, frame, [2022], 2023, total
                ),
                2024: fold_score(
                    oof, logistic, component, frame, [2022, 2023], 2024, total
                ),
            }
            prediction22 = (
                v17_weight * v17_prediction(item22, logistic["2022"])
                + form_weight * form["2022"]
                + context_weight * context["2022"]
            )
            score22 = float(brier_score_loss(item22["target"], prediction22))
            candidates.append({
                "v17_weight": v17_weight,
                "form_weight": form_weight,
                "context_weight": context_weight,
                "2022_raw_brier": score22,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2022_gain_vs_v31": baseline22 - score22,
                "2023_gain_vs_v31": V31_BRIER[2023] - scores[2023],
                "2024_gain_vs_v31": V31_BRIER[2024] - scores[2024],
            })
    accepted = [item for item in candidates if min(
        item["2022_gain_vs_v31"], item["2023_gain_vs_v31"],
        item["2024_gain_vs_v31"],
    ) > 0]
    rank = lambda item: (
        item["2024_gain_vs_v31"],
        min(item["2022_gain_vs_v31"], item["2023_gain_vs_v31"]),
        item["2022_gain_vs_v31"] + item["2023_gain_vs_v31"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v31": {
            "2022_raw": baseline22,
            "2023": V31_BRIER[2023], "2024": V31_BRIER[2024],
        },
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_accepted": accepted[:10],
        "top5_overall": candidates[:5],
    }
    Path("artifacts/v33_weight_retuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
