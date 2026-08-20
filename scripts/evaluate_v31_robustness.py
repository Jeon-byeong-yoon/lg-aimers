"""Add 2022 raw-fold and prediction-correlation checks to V31 candidates."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


def v17(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def main():
    metrics = json.loads(Path("artifacts/v31_feature_removal_metrics.json").read_text())
    predictions = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    base_form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    base_context = joblib.load("artifacts/v24_contextual_trackman_predictions.joblib")
    baseline_predictions, baseline_scores = {}, {}
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        baseline_predictions[year] = (
            0.75 * v17(item, logistic[str(year)])
            + 0.16 * base_form[str(year)] + 0.09 * base_context[str(year)]
        )
        baseline_scores[year] = float(brier_score_loss(
            item["target"], baseline_predictions[year]
        ))
    checked = []
    for item in metrics["top10_overall"]:
        form_name, context_name = item["form"], item["context"]
        form = base_form if form_name == "base" else predictions[form_name]["form"]
        context = (
            base_context if context_name == "base"
            else predictions[context_name]["context"]
        )
        candidate_predictions = {}
        raw_scores = {}
        correlations = {}
        for year in (2022, 2023, 2024):
            candidate_predictions[year] = (
                0.75 * v17(oof[str(year)], logistic[str(year)])
                + 0.16 * form[str(year)] + 0.09 * context[str(year)]
            )
            raw_scores[year] = float(brier_score_loss(
                oof[str(year)]["target"], candidate_predictions[year]
            ))
            correlations[year] = float(np.corrcoef(
                baseline_predictions[year], candidate_predictions[year]
            )[0, 1])
        checked.append({
            **item,
            "2022_raw_brier": raw_scores[2022],
            "2022_raw_gain_vs_v25": baseline_scores[2022] - raw_scores[2022],
            "raw_brier": {str(k): v for k, v in raw_scores.items()},
            "correlation_with_v25": {str(k): v for k, v in correlations.items()},
        })
    robust = [item for item in checked if min(
        item["2022_raw_gain_vs_v25"],
        item["2023_gain_vs_v25"], item["2024_gain_vs_v25"],
    ) > 0]
    robust.sort(key=lambda item: (
        item["2024_gain_vs_v25"], item["2023_gain_vs_v25"],
        item["2022_raw_gain_vs_v25"],
    ), reverse=True)
    output = {
        "baseline_raw_brier": {str(k): v for k, v in baseline_scores.items()},
        "checked_count": len(checked), "robust_count": len(robust),
        "best_robust": robust[0] if robust else None,
        "robust_candidates": robust,
    }
    Path("artifacts/v31_robustness_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
