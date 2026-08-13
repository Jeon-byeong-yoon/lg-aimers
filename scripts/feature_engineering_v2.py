"""Leakage-safe, row-wise features for the second submission generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


LOW_CARDINAL_CATEGORICAL = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "count_state",
    "hand_matchup",
]

ID_COLUMNS = ["pitcher_id", "batter_id"]

FEATURE_GROUPS = {
    "categorical_interactions": ["count_state", "hand_matchup"],
    "game_context": [
        "is_full_count",
        "is_two_strike",
        "is_three_ball",
        "is_late_inning",
        "is_high_leverage",
        "is_close_game",
        "pressure_index",
        "count_pressure",
    ],
    "recent_form": [
        "success_form_gap_1",
        "middle_form_gap_1",
        "success_form_gap_3",
        "middle_form_gap_3",
        "success_form_gap_5",
        "middle_form_gap_5",
        "recent_success_weighted",
        "recent_middle_weighted",
    ],
    "failure_proxies": [
        "location_failure_history",
        "location_quality_history",
    ],
    "reliability": [
        "pitcher_success_smoothed_50",
        "pitcher_success_smoothed_200",
        "log1p_pitcher_n",
        "is_cold_pitcher",
        "batter_success_smoothed_50",
        "batter_success_smoothed_200",
        "log1p_batter_n",
        "is_cold_batter",
    ],
    "missingness": ["missing_history_count", "missing_recent_form"],
}

# Selected after the 2024 HGB ablation. These hand-crafted groups added noise
# beyond the original official as-of columns and are omitted from the candidate.
V2_DROPPED_GROUPS = ["game_context", "recent_form", "failure_proxies"]


def select_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    dropped = [column for group in V2_DROPPED_GROUPS for column in FEATURE_GROUPS[group]]
    return frame.drop(columns=dropped)


def add_row_features(frame: pd.DataFrame, target_prior: float) -> pd.DataFrame:
    """Create features using only values available in the current input row.

    ``target_prior`` must be learned from the training split only. No statistic of
    the test frame, including frequencies or missing rates, is used here.
    """
    data = frame.copy()

    # Explicit categorical interactions; numeric identifier magnitude is not used.
    data["count_state"] = (
        data["balls_before"].astype(str) + "-" + data["strikes_before"].astype(str)
    )
    data["hand_matchup"] = (
        data["pitcher_hand"].astype(str) + "-" + data["batter_hand"].astype(str)
    )

    # Pressure/context features based on current-row information only.
    data["is_full_count"] = (
        (data["balls_before"] == 3) & (data["strikes_before"] == 2)
    ).astype("int8")
    data["is_two_strike"] = (data["strikes_before"] == 2).astype("int8")
    data["is_three_ball"] = (data["balls_before"] == 3).astype("int8")
    data["is_late_inning"] = (data["inning"] >= 7).astype("int8")
    data["is_high_leverage"] = (data["li"] >= 2.0).astype("int8")
    data["is_close_game"] = (data["score_diff_pitcher_team"].abs() <= 1).astype("int8")
    data["pressure_index"] = (
        data["li"]
        * (1.0 + data["num_runners_on"])
        * (1.0 + data["is_late_inning"])
    )
    data["count_pressure"] = data["balls_before"] - data["strikes_before"]

    # Career-vs-recent form deltas capture deterioration or recovery.
    for window in (1, 3, 5):
        recent_success = f"asof_pitcher_prev{window}_game_success_rate"
        recent_middle = f"asof_pitcher_prev{window}_game_middle_rate"
        data[f"success_form_gap_{window}"] = (
            data[recent_success] - data["asof_pitcher_success_rate"]
        )
        data[f"middle_form_gap_{window}"] = (
            data[recent_middle] - data["asof_pitcher_middle_rate"]
        )

    data["recent_success_weighted"] = (
        0.50 * data["asof_pitcher_prev1_game_success_rate"]
        + 0.30 * data["asof_pitcher_prev3_game_success_rate"]
        + 0.20 * data["asof_pitcher_prev5_game_success_rate"]
    )
    data["recent_middle_weighted"] = (
        0.50 * data["asof_pitcher_prev1_game_middle_rate"]
        + 0.30 * data["asof_pitcher_prev3_game_middle_rate"]
        + 0.20 * data["asof_pitcher_prev5_game_middle_rate"]
    )

    # Proxies corresponding to the three official failure definitions.
    data["location_failure_history"] = (
        data["asof_pitcher_middle_rate"]
        + data["asof_pitcher_reverse_rate"]
        + data["asof_pitcher_ball_rate"]
    )
    data["location_quality_history"] = (
        data["asof_pitcher_success_rate"] - data["asof_pitcher_middle_rate"]
    )

    # Reliability: rates based on tiny histories are shrunk toward train prior.
    for owner in ("pitcher", "batter"):
        n_column = f"asof_{owner}_n"
        success_column = f"asof_{owner}_success_rate"
        n = data[n_column].clip(lower=0)
        for strength in (50.0, 200.0):
            data[f"{owner}_success_smoothed_{int(strength)}"] = (
                data[success_column].fillna(target_prior) * n
                + target_prior * strength
            ) / (n + strength)
        data[f"log1p_{owner}_n"] = np.log1p(n)
        data[f"is_cold_{owner}"] = (n < 50).astype("int8")

    # Preserve missingness itself before the model median-imputes values.
    history_rates = [
        column
        for column in data.columns
        if column.startswith("asof_") and column.endswith("_rate")
    ]
    data["missing_history_count"] = data[history_rates].isna().sum(axis=1)
    data["missing_recent_form"] = data[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ].isna().any(axis=1).astype("int8")
    return data
