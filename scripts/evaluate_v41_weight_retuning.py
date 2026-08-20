"""Evaluate 3-way blend weights (V17 vs V38 gentle_500 form vs V31 context).

V38 upgraded the form model with gentle_500 (lr=0.03, iter=500, strong reg),
achieving Public 894.19 (+2.89 vs V31).
However, the blend weights are still fixed at the legacy V25 values:
  0.75 (V17) / 0.16 (form) / 0.09 (context).

This script performs an exhaustive grid search over form_weight (0.10 to 0.35)
and context_weight (0.03 to 0.15), checking 2022 raw OOF, 2023 expanding calib,
and 2024 expanding calib against the V38 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v19_blend import fold_score


# Grid definitions
FORM_WEIGHTS = np.linspace(0.10, 0.32, 23)      # 0.10 to 0.32 with step 0.01
CONTEXT_WEIGHTS = np.linspace(0.03, 0.15, 13)   # 0.03 to 0.15 with step 0.01

# V38 baseline scores (form=gentle_500 @ 0.16, context=v31 @ 0.09, v17 @ 0.75)
V38_BASELINE = {
    2022: 0.243390002346232,
    2023: 0.25324438180845804,
    2024: 0.24783152211261955,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    
    # V38 gentle_500 form OOF predictions
    form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    
    # V31 context OOF predictions
    removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = removed["no_matchup_hte"]["context"]
    
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    
    item22 = oof["2022"]
    v17_22 = v17_prediction(item22, logistic["2022"])
    
    candidates = []
    for form_weight in FORM_WEIGHTS:
        for context_weight in CONTEXT_WEIGHTS:
            total_fc = form_weight + context_weight
            if total_fc >= 0.70:  # keep V17 at least 30%
                continue
            v17_weight = 1.0 - total_fc
            
            # component representation for fold_score (which expects a normalized component)
            component = {
                year: (
                    form_weight * form[year] + context_weight * context[year]
                ) / total_fc
                for year in form
            }
            
            scores = {
                2023: fold_score(
                    oof, logistic, component, frame, [2022], 2023, total_fc
                ),
                2024: fold_score(
                    oof, logistic, component, frame, [2022, 2023], 2024, total_fc
                ),
            }
            
            pred22 = (
                v17_weight * v17_22
                + form_weight * form["2022"]
                + context_weight * context["2022"]
            )
            score22 = float(brier_score_loss(item22["target"], pred22))
            
            gain_22 = V38_BASELINE[2022] - score22
            gain_23 = V38_BASELINE[2023] - scores[2023]
            gain_24 = V38_BASELINE[2024] - scores[2024]
            
            candidates.append({
                "v17_weight": round(float(v17_weight), 4),
                "form_weight": round(float(form_weight), 4),
                "context_weight": round(float(context_weight), 4),
                "scores": {
                    "2022": score22,
                    "2023": scores[2023],
                    "2024": scores[2024],
                },
                "gains_vs_v38": {
                    "2022": gain_22,
                    "2023": gain_23,
                    "2024": gain_24,
                },
                "all_improved": (gain_22 > 0 and gain_23 > 0 and gain_24 > 0),
                "submit_ready": (gain_22 > 0 and gain_23 > 0 and gain_24 > 5e-6),
            })
            
    accepted = [c for c in candidates if c["all_improved"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    # rank by 2024 gain then 2023 gain
    rank = lambda item: (
        item["gains_vs_v38"]["2024"],
        item["gains_vs_v38"]["2023"],
        item["gains_vs_v38"]["2022"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V41_weight_retuning",
        "description": "Exhaustive 3-way blend weight search on V38 gentle_500 form + V31 context",
        "baseline_v38": {str(k): v for k, v in V38_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top10_accepted": accepted[:10],
        "top10_overall": candidates[:10],
    }
    
    Path("artifacts/v41_weight_retuning_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
