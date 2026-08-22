"""V88 step 1: build a transfer-focused validation baseline for V41.

This is a diagnostic/evaluation script only. It never reads test.csv and none of
its segments or validation labels are used as inference features.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction


OUTPUT = Path("artifacts/v88_transfer_validation_metrics.json")
BOOTSTRAP_SEED = 20250823
BOOTSTRAP_ROUNDS = 1000


def brier(target, prediction):
    return float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))


def make_validation_frame():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    feature_frame = data.drop(columns=["row_id", "control_success"])

    rows = []
    for year in YEARS:
        key = str(year)
        prediction, _, _, _ = calibrated_prediction(
            year, oof, logistic, form, context, feature_frame
        )
        part = data.loc[oof[key]["row_index"]].copy().reset_index(drop=True)
        part["target"] = oof[key]["target"].astype(float)
        part["v41_prediction"] = prediction
        part["v41_squared_error"] = (part["v41_prediction"] - part["target"]) ** 2
        rows.append(part)

    frame = pd.concat(rows, ignore_index=True)
    frame["month_block"] = frame["season"].astype(str) + "_m" + frame["game_month"].astype(str)
    frame["half"] = np.where(frame["game_month"] <= 6, "first_mar_jun", "second_jul_oct")
    frame["validation_role"] = np.where(
        (frame["season"] == 2024) & (frame["game_month"] >= 9),
        "final_confirmation_2024_sep_oct",
        "development",
    )
    frame["history_bin"] = pd.cut(
        frame["asof_pitcher_n"],
        bins=[-1, 49, 199, 999, np.inf],
        labels=["cold_0_49", "low_50_199", "mid_200_999", "high_1000_plus"],
    ).astype(str)
    frame["pitchmix_history_bin"] = pd.cut(
        frame["asof_pitcher_pitchmix_n"],
        bins=[-1, 49, 199, 999, np.inf],
        labels=["cold_0_49", "low_50_199", "mid_200_999", "high_1000_plus"],
    ).astype(str)
    return frame


def summarize(frame, columns):
    result = []
    grouper = columns[0] if len(columns) == 1 else columns
    for keys, part in frame.groupby(grouper, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: str(value) for column, value in zip(columns, keys)}
        row.update({
            "n": int(len(part)),
            "pitchers": int(part["pitcher_id"].nunique()),
            "brier": brier(part["target"], part["v41_prediction"]),
            "prediction_mean": float(part["v41_prediction"].mean()),
            "target_mean": float(part["target"].mean()),
            "calibration_gap": float(part["v41_prediction"].mean() - part["target"].mean()),
        })
        result.append(row)
    return result


def pitcher_cluster_bootstrap(frame, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """Bootstrap V41 Brier by pitcher; candidate comparisons use the same clusters."""
    grouped = []
    for _, part in frame.groupby("pitcher_id", observed=True):
        grouped.append((len(part), float(part["v41_squared_error"].sum())))
    counts = np.asarray([item[0] for item in grouped], dtype=float)
    losses = np.asarray([item[1] for item in grouped], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(rounds, dtype=float)
    for index in range(rounds):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = losses[selected].sum() / counts[selected].sum()
    return {
        "unit": "pitcher_id",
        "pitchers": int(len(grouped)),
        "rounds": int(rounds),
        "seed": int(seed),
        "mean": float(estimates.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def compare_candidate(frame, candidate_prediction, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """Compare aligned OOF predictions with V41 under the fixed V88 policy."""
    candidate_prediction = np.asarray(candidate_prediction, dtype=float)
    if candidate_prediction.shape != (len(frame),):
        raise ValueError(f"Expected {len(frame)} candidate predictions, got {candidate_prediction.shape}")
    work = frame.copy()
    work["candidate_prediction"] = np.clip(candidate_prediction, 0, 1)
    work["candidate_squared_error"] = (
        work["candidate_prediction"] - work["target"]
    ) ** 2
    work["gain"] = work["v41_squared_error"] - work["candidate_squared_error"]

    monthly_gains = work.groupby(["season", "game_month"], observed=True)["gain"].mean()
    pitcher_gain = work.groupby("pitcher_id", observed=True)["gain"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    estimates = np.empty(rounds, dtype=float)
    for index in range(rounds):
        selected = rng.integers(0, len(pitcher_gain), size=len(pitcher_gain))
        estimates[index] = (
            pitcher_gain["sum"].to_numpy()[selected].sum()
            / pitcher_gain["count"].to_numpy()[selected].sum()
        )

    def mean_gain(mask):
        return float(work.loc[mask, "gain"].mean())

    return {
        "season_gain": {
            str(year): mean_gain(work["season"] == year) for year in YEARS
        },
        "gain_2024_jul_oct": mean_gain(
            (work["season"] == 2024) & (work["game_month"] >= 7)
        ),
        "gain_2024_final_confirmation_sep_oct": mean_gain(
            work["validation_role"] == "final_confirmation_2024_sep_oct"
        ),
        "monthly_blocks_won": int((monthly_gains > 0).sum()),
        "monthly_blocks_total": int(len(monthly_gains)),
        "monthly_block_win_rate": float((monthly_gains > 0).mean()),
        "worst_monthly_gain": float(monthly_gains.min()),
        "paired_pitcher_bootstrap": {
            "rounds": int(rounds),
            "seed": int(seed),
            "mean_gain": float(estimates.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
        },
    }


def main():
    frame = make_validation_frame()
    season = summarize(frame, ["season"])
    monthly = summarize(frame, ["season", "game_month"])
    halves = summarize(frame, ["season", "half"])
    transfer_windows = summarize(frame, ["validation_role"])
    history = summarize(frame, ["season", "history_bin"])
    pitchmix_history = summarize(frame, ["season", "pitchmix_history_bin"])

    block_briers = [row["brier"] for row in monthly]
    output = {
        "experiment": "V88_transfer_focused_validation_baseline",
        "baseline": "V41",
        "compliance": {
            "official_train_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "current_pitch_post_event_information_used": False,
            "validation_segments_used_as_inference_features": False,
        },
        "selection_policy": {
            "development_windows": "2022-2023 and 2024 March-August",
            "final_confirmation_window": "2024 September-October; inspect once per promoted candidate",
            "primary_requirements": [
                "positive 2024 full-season gain",
                "positive 2024 July-October gain",
                "no material 2022 or 2023 regression",
                "at least 75% monthly-block wins",
                "target 2024 gain >= 2e-5; preferred >= 5e-5",
            ],
            "important_note": "The final confirmation window is reported for the baseline now, but future candidate tuning must not use it.",
        },
        "season_summary": season,
        "half_summary": halves,
        "transfer_windows": transfer_windows,
        "monthly_blocks": monthly,
        "history_segments": history,
        "pitchmix_history_segments": pitchmix_history,
        "monthly_block_difficulty": {
            "count": int(len(block_briers)),
            "best_brier": float(min(block_briers)),
            "worst_brier": float(max(block_briers)),
        },
        "pitcher_cluster_bootstrap": {
            str(year): pitcher_cluster_bootstrap(frame.loc[frame["season"] == year])
            for year in YEARS
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "season_summary": season,
        "2024_halves": [row for row in halves if row["season"] == "2024"],
        "monthly_block_difficulty": output["monthly_block_difficulty"],
        "bootstrap_2024": output["pitcher_cluster_bootstrap"]["2024"],
    }, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
