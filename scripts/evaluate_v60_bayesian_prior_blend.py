"""V60: Evaluate Bayesian Prior / Career Base Rate Micro-Blending on top of V41.

Hypothesis:
-----------
Tree models can sometimes overfit local interactions and deviate slightly from
a pitcher's true intrinsic base control ability (asof_pitcher_success_rate).
Testing micro-blending (1% ~ 5%) of a Bayesian smoothed pitcher career control rate:
  bayes_pitcher_rate = (n_pitches * asof_success_rate + M * league_mean) / (n_pitches + M)
with M in [50, 100, 200, 500].

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


# V41 constants
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}


def main():
    print("Loading data and V41 artifacts...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31_removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_preds = v31_removed["no_matchup_hte"]["context"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    v11_tree_preds = {str(yr): v11_prediction(oof[str(yr)]) for yr in (2022, 2023, 2024)}
    v17_preds = {str(yr): 0.95 * v11_tree_preds[str(yr)] + 0.05 * logistic[str(yr)] for yr in (2022, 2023, 2024)}
    
    raw_blend = {
        str(yr): W_V17 * v17_preds[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds[str(yr)]
        for yr in (2022, 2023, 2024)
    }
    
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    v41_preds = {}
    v41_preds["2022"] = np.clip(raw_blend["2022"], 0, 1)
    
    res22 = targets["2022"] - raw_blend["2022"]
    cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
    pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
    v41_preds["2023"] = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
    
    res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
    cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
    pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
    v41_preds["2024"] = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
    
    # Compute Bayesian pitcher base rate for each season
    results = []
    
    for M in [50, 100, 200, 500]:
        for w_bayes in [0.005, 0.01, 0.02, 0.03, 0.05]:
            p_blended = {}
            for yr in (2022, 2023, 2024):
                f = frames[str(yr)]
                league_prior = float(y.loc[data["season"] < yr].mean())
                n_pitches = f["asof_pitcher_n"].fillna(0).to_numpy()
                success_rate = f["asof_pitcher_success_rate"].fillna(league_prior).to_numpy()
                bayes_rate = (n_pitches * success_rate + M * league_prior) / (n_pitches + M)
                p_blended[str(yr)] = np.clip((1.0 - w_bayes) * v41_preds[str(yr)] + w_bayes * bayes_rate, 0, 1)
                
            s22 = float(brier_score_loss(targets["2022"], p_blended["2022"]))
            s23 = float(brier_score_loss(targets["2023"], p_blended["2023"]))
            s24 = float(brier_score_loss(targets["2024"], p_blended["2024"]))
            
            g22 = V41_BASELINE[2022] - s22
            g23 = V41_BASELINE[2023] - s23
            g24 = V41_BASELINE[2024] - s24
            
            results.append({
                "M": M,
                "w_bayes": w_bayes,
                "scores": {"2022": s22, "2023": s23, "2024": s24},
                "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
                "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
                "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
            })
            
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    rank = lambda item: (item["gains_vs_v41"]["2024"], item["gains_vs_v41"]["2023"], item["gains_vs_v41"]["2022"])
    accepted.sort(key=rank, reverse=True)
    results.sort(key=rank, reverse=True)
    
    output = {
        "experiment": "V60_bayesian_prior_blend",
        "description": "Evaluate micro-blending of Bayesian smoothed pitcher career base rate",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results": results[:5],
    }
    
    Path("artifacts/v60_bayesian_prior_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
