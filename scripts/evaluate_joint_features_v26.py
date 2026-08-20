"""Replace V25 context component with enhanced Trackman and retune weights."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v19_blend import fold_score


FORM_WEIGHTS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
CONTEXT_WEIGHTS = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
V25_BRIER = {2023: 0.2533208415598194, 2024: 0.24784544408284379}


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v26_contextual_trackman_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    candidates = []
    for form_weight in FORM_WEIGHTS:
        for context_weight in CONTEXT_WEIGHTS:
            total = form_weight + context_weight
            component = {
                year: (
                    form_weight * form[year] + context_weight * context[year]
                ) / total
                for year in form
            }
            scores = {
                2023: fold_score(oof, logistic, component, frame, [2022], 2023, total),
                2024: fold_score(
                    oof, logistic, component, frame, [2022, 2023], 2024, total
                ),
            }
            candidates.append({
                "v17_weight": 1 - total,
                "form_weight": form_weight,
                "context_v26_weight": context_weight,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain_vs_v25": V25_BRIER[2023] - scores[2023],
                "2024_gain_vs_v25": V25_BRIER[2024] - scores[2024],
            })
    accepted = [
        item for item in candidates
        if item["2023_gain_vs_v25"] > 0 and item["2024_gain_vs_v25"] > 0
    ]
    rank = lambda item: (
        min(item["2023_gain_vs_v25"], item["2024_gain_vs_v25"]),
        item["2023_gain_vs_v25"] + item["2024_gain_vs_v25"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v25": {str(k): v for k, v in V25_BRIER.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10": accepted[:10],
    }
    Path("artifacts/v26_joint_feature_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
