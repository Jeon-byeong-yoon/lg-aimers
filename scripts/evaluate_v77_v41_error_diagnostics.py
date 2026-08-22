"""V77: V41 OOF 예측의 시간 블록·세그먼트별 Brier 오류 진단.

진단 결과는 모델 선택용이며 추론 피처로 사용하지 않는다.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


YEARS = (2022, 2023, 2024)


def calibrated_prediction(year, oof, logistic, form, context, frame):
    key = str(year)
    v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
    raw = 0.55 * v17 + 0.32 * form[key] + 0.13 * context[key]
    if year == 2022:
        return np.clip(raw, 0, 1), v17, form[key], context[key]

    history_years = [y for y in YEARS if y < year]
    history_index, history_target, history_prediction = [], [], []
    for history_year in history_years:
        history_key = str(history_year)
        history_v17 = 0.95 * v11_prediction(oof[history_key]) + 0.05 * logistic[history_key]
        history_raw = 0.55 * history_v17 + 0.32 * form[history_key] + 0.13 * context[history_key]
        history_index.append(oof[history_key]["row_index"])
        history_target.append(oof[history_key]["target"].astype(float))
        history_prediction.append(history_raw)

    index = np.concatenate(history_index)
    residual = np.concatenate(history_target) - np.concatenate(history_prediction)
    train_frame = frame.loc[index]
    valid_frame = frame.loc[oof[key]["row_index"]]
    count = segment_correction(
        train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
    )
    pitcher_count = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(raw + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)
    return prediction, v17, form[key], context[key]


def summarize(group, key):
    rows = []
    for value, part in group.groupby(key, observed=True, dropna=False):
        error = part["squared_error"].to_numpy()
        rows.append({
            "group": str(value),
            "n": int(len(part)),
            "brier": float(error.mean()),
            "loss_share": float(error.sum() / group["squared_error"].sum()),
            "prediction_mean": float(part["prediction"].mean()),
            "target_mean": float(part["target"].mean()),
            "calibration_gap": float(part["prediction"].mean() - part["target"].mean()),
        })
    return sorted(rows, key=lambda row: row["brier"], reverse=True)


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    feature_frame = data.drop(columns=["row_id", "control_success"])

    all_rows = []
    season_summary = {}
    for year in YEARS:
        key = str(year)
        prediction, v17, form_prediction, context_prediction = calibrated_prediction(
            year, oof, logistic, form, context, feature_frame
        )
        indices = oof[key]["row_index"]
        part = data.loc[indices].copy().reset_index(drop=True)
        part["target"] = oof[key]["target"].astype(float)
        part["prediction"] = prediction
        part["squared_error"] = (part["prediction"] - part["target"]) ** 2
        components = np.column_stack([v17, form_prediction, context_prediction])
        part["model_disagreement"] = components.std(axis=1)
        part["season_block"] = pd.cut(
            part["game_month"], bins=[2, 5, 7, 8, 10],
            labels=["early_3_5", "mid_6_7", "late_8", "finish_9_10"],
        )
        part["pitcher_history_bin"] = pd.cut(
            part["asof_pitcher_n"], bins=[-1, 49, 199, 999, np.inf],
            labels=["cold_0_49", "low_50_199", "mid_200_999", "high_1000_plus"],
        )
        part["pitchmix_history_bin"] = pd.cut(
            part["asof_pitcher_pitchmix_n"], bins=[-1, 49, 199, 999, np.inf],
            labels=["cold_0_49", "low_50_199", "mid_200_999", "high_1000_plus"],
        )
        part["disagreement_bin"] = pd.qcut(
            part["model_disagreement"], q=5,
            labels=["q1_low", "q2", "q3", "q4", "q5_high"], duplicates="drop",
        )
        part["count"] = part["balls_before"].astype(str) + "-" + part["strikes_before"].astype(str)
        season_summary[key] = {
            "n": int(len(part)),
            "brier": float(part["squared_error"].mean()),
            "prediction_mean": float(part["prediction"].mean()),
            "target_mean": float(part["target"].mean()),
        }
        all_rows.append(part)

    combined = pd.concat(all_rows, ignore_index=True)
    dimensions = [
        "season_block", "pitcher_history_bin", "pitchmix_history_bin",
        "disagreement_bin", "count", "base_state", "pitcher_hand", "batter_hand",
    ]
    by_season = {
        str(year): {
            dimension: summarize(combined.loc[combined["season"] == year], dimension)
            for dimension in dimensions
        }
        for year in YEARS
    }
    overall = {dimension: summarize(combined, dimension) for dimension in dimensions}
    block_rows = []
    for (year, block), part in combined.groupby(["season", "season_block"], observed=True):
        block_rows.append({
            "season": int(year), "block": str(block), "n": int(len(part)),
            "brier": float(part["squared_error"].mean()),
            "loss_share_within_season": float(
                part["squared_error"].sum()
                / combined.loc[combined["season"] == year, "squared_error"].sum()
            ),
        })

    output = {
        "experiment": "V77_v41_error_diagnostics",
        "compliance": {
            "official_data_only": True,
            "test_row_aggregation_used": False,
            "current_pitch_post_event_information_used": False,
            "diagnostics_used_as_inference_features": False,
        },
        "season_summary": season_summary,
        "time_blocks": block_rows,
        "overall_segments": overall,
        "segments_by_season": by_season,
    }
    path = Path("artifacts/v77_v41_error_diagnostics.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"season_summary": season_summary, "time_blocks": block_rows}, indent=2, ensure_ascii=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
