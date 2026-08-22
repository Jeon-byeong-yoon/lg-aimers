"""V66: 투수별 Isotonic Regression 개인화 보정

V41 raw blend에 투수별 단조 보정을 적용 후 count correction 진행.

구현:
  - n_min=300구 이상 투수: 개인별 IsotonicRegression 학습
  - n_min 미만: 글로벌 IR 적용
  - 순서: V41 raw blend → 투수별 IR → count/pitcher_count correction
"""

import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

W_V17  = 0.55
W_FORM = 0.32
W_CTX  = 0.13
N_MIN_PITCHER_IR = 300


def apply_pitcher_ir(raw_pred: np.ndarray, frame: pd.DataFrame,
                     train_pred: np.ndarray, train_target: np.ndarray,
                     train_frame: pd.DataFrame,
                     n_min: int = N_MIN_PITCHER_IR) -> np.ndarray:
    """
    투수별 Isotonic Regression 보정 적용.
    학습 데이터로 투수별 IR 모델 구성, 테스트 데이터에 적용.
    """
    corrected = raw_pred.copy()
    train_pids = train_frame["pitcher_id"].values
    test_pids  = frame["pitcher_id"].values

    # 글로벌 IR 학습 (n_min 미만 투수 대체용)
    global_ir = IsotonicRegression(out_of_bounds="clip")
    global_ir.fit(train_pred, train_target)

    unique_pids = np.unique(test_pids)
    for pid in unique_pids:
        test_mask  = (test_pids  == pid)
        train_mask = (train_pids == pid)

        if train_mask.sum() >= n_min:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(train_pred[train_mask], train_target[train_mask])
            corrected[test_mask] = ir.predict(raw_pred[test_mask])
        else:
            corrected[test_mask] = global_ir.predict(raw_pred[test_mask])

    return corrected


def main():
    print("Loading artifacts...", flush=True)
    data      = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all     = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()

    oof        = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic   = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    v31        = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    ctx_preds  = v31["no_matchup_hte"]["context"]

    years = (2022, 2023, 2024)
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in years}
    frames  = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in years}
    y_dict  = {str(yr): y_all.loc[oof[str(yr)]["row_index"]] for yr in years}

    v11_tr   = {str(yr): v11_prediction(oof[str(yr)]) for yr in years}
    v17_pred = {str(yr): 0.95*v11_tr[str(yr)] + 0.05*logistic[str(yr)] for yr in years}

    raw_blend = {
        str(yr): W_V17*v17_pred[str(yr)] + W_FORM*form_preds[str(yr)] + W_CTX*ctx_preds[str(yr)]
        for yr in years
    }

    # ── 투수별 IR 적용 ────────────────────────────────────────────────
    # 2022: raw blend 자체 (train 없음 → 보정 없이 raw 사용)
    p22_raw = raw_blend["2022"]

    # 2023: 2022 데이터로 IR 학습, 2023에 적용
    print("Applying pitcher IR (2022 → 2023)...", flush=True)
    ir_23 = apply_pitcher_ir(
        raw_pred=raw_blend["2023"].copy(),
        frame=frames["2023"],
        train_pred=raw_blend["2022"],
        train_target=np.array(targets["2022"]),
        train_frame=frames["2022"],
    )

    # 2024: 2022+2023 데이터로 IR 학습, 2024에 적용
    print("Applying pitcher IR (2022+2023 → 2024)...", flush=True)
    train_pred_2223 = np.concatenate([raw_blend["2022"], raw_blend["2023"]])
    train_tgt_2223  = np.concatenate([np.array(targets["2022"]), np.array(targets["2023"])])

    train_frame_2223 = pd.concat([frames["2022"], frames["2023"]])

    ir_24 = apply_pitcher_ir(
        raw_pred=raw_blend["2024"].copy(),
        frame=frames["2024"],
        train_pred=train_pred_2223,
        train_target=train_tgt_2223,
        train_frame=train_frame_2223,
    )

    blend_ir = {
        "2022": p22_raw,
        "2023": ir_23,
        "2024": ir_24,
    }

    # ── count correction ─────────────────────────────────────────────
    f23_tr = frames["2022"]
    f24_tr = pd.concat([frames["2022"], frames["2023"]])

    res22    = targets["2022"] - blend_ir["2022"]
    res_2223 = np.concatenate([res22, targets["2023"] - blend_ir["2023"]])

    p22 = np.clip(blend_ir["2022"], 0, 1)

    cnt23  = segment_correction(f23_tr, res22,    frames["2023"], ["balls_before","strikes_before"], 500)
    pcnt23 = segment_correction(f23_tr, res22,    frames["2023"], ["pitcher_id","balls_before","strikes_before"], 300)
    p23    = np.clip(blend_ir["2023"] + res22.mean() + 0.75*cnt23 + 0.25*pcnt23, 0, 1)

    cnt24  = segment_correction(f24_tr, res_2223, frames["2024"], ["balls_before","strikes_before"], 500)
    pcnt24 = segment_correction(f24_tr, res_2223, frames["2024"], ["pitcher_id","balls_before","strikes_before"], 300)
    p24    = np.clip(blend_ir["2024"] + res_2223.mean() + 0.75*cnt24 + 0.25*pcnt24, 0, 1)

    s22 = float(brier_score_loss(targets["2022"], p22))
    s23 = float(brier_score_loss(targets["2023"], p23))
    s24 = float(brier_score_loss(targets["2024"], p24))

    g22 = V41_BASELINE[2022] - s22
    g23 = V41_BASELINE[2023] - s23
    g24 = V41_BASELINE[2024] - s24

    result = {
        "experiment": "V66_pitcher_isotonic_regression",
        "description": "투수별 Isotonic Regression 개인화 보정 (n_min=300구)",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "scores":       {"2022": s22, "2023": s23, "2024": s24},
        "gains_vs_v41": {"2022": g22, "2023": g23, "2024": g24},
        "all_improved": (g22 > 0 and g23 > 0 and g24 > 0),
        "submit_ready": (g22 > 0 and g23 > 0 and g24 > 5e-6),
    }

    Path("artifacts/v66_pitcher_ir_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n✅ all_improved={result['all_improved']}, submit_ready={result['submit_ready']}", flush=True)


if __name__ == "__main__":
    main()
