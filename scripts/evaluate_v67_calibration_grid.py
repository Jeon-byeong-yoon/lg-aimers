"""V67: Calibration 세그먼트 보정 파라미터 격자 탐색

현재 고정값:
  count segment:   n_min=500, w=0.75
  pitcher×count:   n_min=300, w=0.25

탐색 공간:
  n_count  ∈ {300, 400, 500, 600, 700}
  n_pitcher ∈ {200, 300, 400, 500}
  w1 (count weight) ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.90}
  w2 = 1 - w1
  총 5 × 4 × 6 = 120개 조합
"""

import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction

# V41 기준선
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

W_V17   = 0.55
W_FORM  = 0.32
W_CTX   = 0.13

def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof         = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic    = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds  = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31         = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds   = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets     = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames      = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    v11_tr      = {str(yr): v11_prediction(oof[str(yr)])            for yr in years}
    v17_preds   = {str(yr): 0.95*v11_tr[str(yr)] + 0.05*logistic[str(yr)] for yr in years}

    raw_blend   = {
        str(yr): W_V17*v17_preds[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_preds[str(yr)]
        for yr in years
    }

    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]])

    # ── 잔차 (고정) ──────────────────────────────────────────
    res22    = targets["2022"] - raw_blend["2022"]
    res_2223 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])

    # ── V41 raw 2022 (캘리브 없음, 고정) ─────────────────────
    p22_raw = np.clip(raw_blend["2022"], 0, 1)

    n_counts  = [300, 400, 500, 600, 700]
    n_pitchers = [200, 300, 400, 500]
    w1_list   = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]

    results   = []
    total     = len(n_counts) * len(n_pitchers) * len(w1_list)
    print(f"탐색 조합 수: {total}", flush=True)

    for nc in n_counts:
        for np_ in n_pitchers:
            for w1 in w1_list:
                w2 = round(1.0 - w1, 4)

                # 2023 calibration
                cnt23  = segment_correction(f23_tr, res22,    frames["2023"],
                                            ["balls_before","strikes_before"], nc)
                pcnt23 = segment_correction(f23_tr, res22,    frames["2023"],
                                            ["pitcher_id","balls_before","strikes_before"], np_)
                p23    = np.clip(raw_blend["2023"] + res22.mean() + w1*cnt23 + w2*pcnt23, 0, 1)

                # 2024 calibration
                cnt24  = segment_correction(f24_tr, res_2223, frames["2024"],
                                            ["balls_before","strikes_before"], nc)
                pcnt24 = segment_correction(f24_tr, res_2223, frames["2024"],
                                            ["pitcher_id","balls_before","strikes_before"], np_)
                p24    = np.clip(raw_blend["2024"] + res_2223.mean() + w1*cnt24 + w2*pcnt24, 0, 1)

                s22 = float(brier_score_loss(targets["2022"], p22_raw))
                s23 = float(brier_score_loss(targets["2023"], p23))
                s24 = float(brier_score_loss(targets["2024"], p24))

                g22 = V41_BASELINE[2022] - s22
                g23 = V41_BASELINE[2023] - s23
                g24 = V41_BASELINE[2024] - s24

                results.append({
                    "n_count": nc, "n_pitcher": np_, "w1": w1, "w2": w2,
                    "scores":        {"2022": s22, "2023": s23, "2024": s24},
                    "gains_vs_v41":  {"2022": g22, "2023": g23, "2024": g24},
                    "all_improved":  (g22 > 0 and g23 > 0 and g24 > 0),
                    "submit_ready":  (g22 > 0 and g23 > 0 and g24 > 5e-6),
                })

    rank = lambda r: (r["gains_vs_v41"]["2024"], r["gains_vs_v41"]["2023"], r["gains_vs_v41"]["2022"])
    results.sort(key=rank, reverse=True)
    accepted      = [r for r in results if r["all_improved"]]
    submit_ready  = [r for r in results if r["submit_ready"]]

    output = {
        "experiment":       "V67_calibration_param_grid",
        "baseline_v41":     {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count":  len(results),
        "accepted_count":   len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "top_5_results":    results[:5],
    }

    Path("artifacts/v67_calibration_grid_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)
    print(f"\n✅ accepted={len(accepted)}, submit_ready={len(submit_ready)}", flush=True)

if __name__ == "__main__":
    main()
