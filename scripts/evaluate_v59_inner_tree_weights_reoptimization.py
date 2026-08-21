"""V59: Inner Tree Weights Re-optimization under the V41 3-way blend.

Background:
-----------
The V11 inner tree weights:
  ExtraTrees: 29.48%, Trackman HGB: 23.44%, TE Trackman HGB: 11.62%, Hierarchical HGB: 35.45%
were optimized back in V11 when there were NO Form HGB (32%) and NO Context HGB (13%).

Now that Form HGB handles recent pitching form and Context HGB handles trackman context,
the optimal balance among the 4 Base Trees within the 0.55 * (0.95 * Trees + 0.05 * Logistic)
component may have shifted.

Grid:
  Grid search over 4-simplex weights (step=0.03, ~5,000 combinations)
  under the full V41 3-way blend and expanding calibration.

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib).
  Threshold: 2022 > 0, 2023 > 0, 2024 gain > 5e-6 vs V41 baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import COMPONENTS
from evaluate_segment_calibration_v12 import segment_correction


# V41 macro weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13
W_TREES = 0.95
W_LOG = 0.05

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}


def main():
    print("Loading precomputed OOF artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = data.drop(columns="row_id").copy()
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    
    # Extract the 4 component matrices for each season
    comp_matrices = {}
    for yr in (2022, 2023, 2024):
        item = oof[str(yr)]
        comp_matrices[str(yr)] = np.column_stack([item["components"][name] for name in COMPONENTS])
        
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    # Generate 4-simplex grid with step 0.03
    steps = np.arange(0.05, 0.55, 0.03)
    grid = []
    for w_et in steps:
        for w_tm in steps:
            for w_te in steps:
                w_h = 1.0 - (w_et + w_tm + w_te)
                if 0.05 <= w_h <= 0.55:
                    grid.append((w_et, w_tm, w_te, w_h))
                    
    print(f"Generated {len(grid)} 4-simplex tree weight candidates. Evaluating...", flush=True)
    
    candidates = []
    
    for w_et, w_tm, w_te, w_h in grid:
        weights = np.array([w_et, w_tm, w_te, w_h])
        
        # 1. Compute tree predictions
        tree_preds = {str(yr): comp_matrices[str(yr)] @ weights for yr in (2022, 2023, 2024)}
        
        # 2. V17 = 0.95 * trees + 0.05 * logistic
        v17_variant = {str(yr): W_TREES * tree_preds[str(yr)] + W_LOG * logistic[str(yr)] for yr in (2022, 2023, 2024)}
        
        # 3. 3-way blend
        raw_blend = {
            str(yr): W_V17 * v17_variant[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        
        score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
        
        # 2023 calib
        res22 = targets["2022"] - raw_blend["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # 2024 calib
        res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        candidates.append({
            "weights": {
                "extra_trees": float(w_et),
                "trackman_hgb": float(w_tm),
                "te_trackman_hgb": float(w_te),
                "hierarchical_hgb": float(w_h),
            },
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        
    accepted = [c for c in candidates if c["all_improved"]]
    submit_ready = [c for c in candidates if c["submit_ready"]]
    
    rank = lambda item: (item["gains_vs_v41"]["2024"], item["gains_vs_v41"]["2023"], item["gains_vs_v41"]["2022"])
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V59_inner_tree_weights_reoptimization",
        "description": "Re-optimize 4-simplex tree weights under V41 3-way blend (55% V17, 32% Form, 13% Context)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": candidates[:5],
    }
    
    Path("artifacts/v59_inner_tree_weights_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("\n--- Summary ---", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
