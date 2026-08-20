"""Diagnose stable V17 OOF residual patterns across 2023 and 2024."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


FEATURES = [
    "inning", "outs_before", "run_total_before", "score_diff_pitcher_team",
    "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def v17_raw(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def fold_prediction(oof, logistic, frame, history_years, valid_year):
    history = [oof[str(year)] for year in history_years]
    train_y = np.concatenate([item["target"] for item in history]).astype(float)
    train_p = np.concatenate([
        v17_raw(item, logistic[str(year)])
        for year, item in zip(history_years, history)
    ])
    train_index = np.concatenate([item["row_index"] for item in history])
    valid = oof[str(valid_year)]
    valid_y = valid["target"].astype(float)
    valid_p = v17_raw(valid, logistic[str(valid_year)])
    train_frame = frame.loc[train_index]
    valid_frame = frame.loc[valid["row_index"]]
    residual = train_y - train_p
    count = segment_correction(
        train_frame, residual, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    pitcher_count = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(
        valid_p + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1
    )
    return valid_y, prediction, valid_frame


def feature_diagnostic(values, residual):
    valid = values.notna()
    x = values.loc[valid].astype(float)
    r = residual[valid.to_numpy()]
    if len(x) < 100 or x.nunique() < 2:
        return None
    correlation = float(np.corrcoef(x.to_numpy(), r)[0, 1])
    try:
        bins = pd.qcut(x, q=10, duplicates="drop")
        table = pd.DataFrame({"bin": bins.astype(str), "residual": r})
        grouped = table.groupby("bin", observed=True)["residual"].agg(["mean", "size"])
        spread = float(grouped["mean"].max() - grouped["mean"].min())
    except ValueError:
        spread = 0.0
    return {"correlation": correlation, "decile_residual_spread": spread}


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    folds = {
        2023: fold_prediction(oof, logistic, frame, [2022], 2023),
        2024: fold_prediction(oof, logistic, frame, [2022, 2023], 2024),
    }
    results = {"folds": {}, "features": {}}
    for year, (target, prediction, valid_frame) in folds.items():
        residual = target - prediction
        results["folds"][str(year)] = {
            "brier": float(brier_score_loss(target, prediction)),
            "residual_mean": float(residual.mean()),
        }
        for feature in FEATURES:
            diagnostic = feature_diagnostic(valid_frame[feature], residual)
            if diagnostic:
                results["features"].setdefault(feature, {})[str(year)] = diagnostic
    stable = []
    for feature, metrics in results["features"].items():
        if "2023" not in metrics or "2024" not in metrics:
            continue
        corr23 = metrics["2023"]["correlation"]
        corr24 = metrics["2024"]["correlation"]
        if np.sign(corr23) == np.sign(corr24):
            stable.append({
                "feature": feature,
                "corr_2023": corr23,
                "corr_2024": corr24,
                "min_abs_corr": min(abs(corr23), abs(corr24)),
                "spread_2023": metrics["2023"]["decile_residual_spread"],
                "spread_2024": metrics["2024"]["decile_residual_spread"],
            })
    stable.sort(key=lambda item: item["min_abs_corr"], reverse=True)
    results["stable_ranking"] = stable
    Path("artifacts/v22_v17_residual_diagnostics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({"folds": results["folds"], "top15": stable[:15]}, indent=2))


if __name__ == "__main__":
    main()
