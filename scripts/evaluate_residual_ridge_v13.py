"""Evaluate a chronological Ridge residual corrector on top of V12."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_segment_calibration_v12 import segment_correction


COMPONENTS = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
WEIGHTS = np.array([
    0.29483562599237795, 0.2344456574665245,
    0.11617615437521091, 0.35454256216588664,
])
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
ALPHAS = [30, 100, 300, 1000, 3000, 10000, 30000]
STRENGTHS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def raw_prediction(item):
    matrix = np.column_stack([item["components"][name] for name in COMPONENTS])
    return matrix @ WEIGHTS


def prepare_fold(oof, frame, history_years, valid_year):
    history = [oof[str(year)] for year in history_years]
    train_y = np.concatenate([item["target"] for item in history]).astype(float)
    train_p = np.concatenate([raw_prediction(item) for item in history])
    train_index = np.concatenate([item["row_index"] for item in history])
    valid = oof[str(valid_year)]
    valid_y = valid["target"].astype(float)
    valid_p = raw_prediction(valid)
    valid_index = valid["row_index"]
    train_frame, valid_frame = frame.loc[train_index], frame.loc[valid_index]

    residual = train_y - train_p
    shift = float(residual.mean())
    centered = residual - shift
    count_columns = ["balls_before", "strikes_before"]
    pitcher_count_columns = ["pitcher_id", "balls_before", "strikes_before"]
    train_count = segment_correction(
        train_frame, residual, train_frame, count_columns, 50
    )
    train_pitcher_count = segment_correction(
        train_frame, residual, train_frame, pitcher_count_columns, 300
    )
    valid_count = segment_correction(
        train_frame, residual, valid_frame, count_columns, 50
    )
    valid_pitcher_count = segment_correction(
        train_frame, residual, valid_frame, pitcher_count_columns, 300
    )
    ridge_target = centered - 0.5 * train_count - 0.125 * train_pitcher_count
    baseline = np.clip(
        valid_p + shift + 0.5 * valid_count + 0.125 * valid_pitcher_count, 0, 1
    )
    return (
        train_frame[FEATURES], ridge_target,
        valid_frame[FEATURES], valid_y, baseline,
    )


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    folds = {
        2023: prepare_fold(oof, frame, [2022], 2023),
        2024: prepare_fold(oof, frame, [2022, 2023], 2024),
    }
    baselines = {
        year: float(brier_score_loss(values[3], values[4]))
        for year, values in folds.items()
    }
    results = []
    for alpha in ALPHAS:
        corrections = {}
        for year, (train_x, train_y, valid_x, _, _) in folds.items():
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=alpha),
            )
            model.fit(train_x, train_y)
            corrections[year] = model.predict(valid_x)
        for strength in STRENGTHS:
            scores = {}
            for year, (_, _, _, valid_y, baseline) in folds.items():
                scores[year] = float(brier_score_loss(
                    valid_y, np.clip(baseline + strength * corrections[year], 0, 1)
                ))
            if all(scores[year] < baselines[year] for year in folds):
                results.append({
                    "alpha": alpha,
                    "strength": strength,
                    "2023_brier": scores[2023], "2024_brier": scores[2024],
                    "2023_gain": baselines[2023] - scores[2023],
                    "2024_gain": baselines[2024] - scores[2024],
                })
    results.sort(
        key=lambda x: (min(x["2023_gain"], x["2024_gain"]),
                       x["2023_gain"] + x["2024_gain"]), reverse=True
    )
    output = {
        "baseline": {str(year): score for year, score in baselines.items()},
        "accepted_count": len(results),
        "best": results[0] if results else None,
        "top10": results[:10],
        "features": FEATURES,
    }
    Path("artifacts/v13_residual_ridge_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
