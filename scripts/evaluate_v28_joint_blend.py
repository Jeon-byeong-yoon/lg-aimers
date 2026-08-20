"""Test whether Trackman archetypes add diverse signal to V26/V25 components."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v19_blend import fold_score


FORM_WEIGHTS = [0.08, 0.10, 0.12, 0.14]
CONTEXT_WEIGHTS = [0.10, 0.13, 0.16, 0.19, 0.22]
ARCHETYPE_WEIGHTS = [0.02, 0.04, 0.06, 0.08, 0.10]
V25_BRIER = {2023: 0.2533208415598194, 2024: 0.24784544408284379}


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v26_contextual_trackman_predictions.joblib")
    archetype = joblib.load("artifacts/v28_trackman_archetype_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    candidates = []
    for fw in FORM_WEIGHTS:
        for cw in CONTEXT_WEIGHTS:
            for aw in ARCHETYPE_WEIGHTS:
                total = fw + cw + aw
                if total >= 0.5:
                    continue
                component = {
                    year: (fw * form[year] + cw * context[year] + aw * archetype[year]) / total
                    for year in form
                }
                scores = {
                    2023: fold_score(oof, logistic, component, frame, [2022], 2023, total),
                    2024: fold_score(oof, logistic, component, frame, [2022, 2023], 2024, total),
                }
                candidates.append({
                    "v17_weight": 1 - total, "form_weight": fw,
                    "context_v26_weight": cw, "archetype_v28_weight": aw,
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
        "baseline_v25": {str(k): v for k, v in V25_BRIER.items()},
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None, "top10": accepted[:10],
    }
    Path("artifacts/v28_joint_blend_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
