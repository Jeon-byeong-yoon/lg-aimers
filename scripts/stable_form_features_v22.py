"""Row-wise recent-form features selected by stable V17 residual diagnostics."""

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "stable_recent_success",
    "stable_recent_gap",
    "stable_recent_disagreement",
    "stable_recent_gap_reliable",
    "stable_recent_middle",
    "stable_recent_middle_gap",
]


def add_stable_form_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    recent_success = (
        0.6 * output["asof_pitcher_prev3_game_success_rate"]
        + 0.4 * output["asof_pitcher_prev5_game_success_rate"]
    )
    output["stable_recent_success"] = recent_success
    output["stable_recent_gap"] = recent_success - output["asof_pitcher_success_rate"]
    output["stable_recent_disagreement"] = (
        output["asof_pitcher_prev3_game_success_rate"]
        - output["asof_pitcher_prev5_game_success_rate"]
    ).abs()
    reliability = output["asof_pitcher_n"].clip(lower=0) / (
        output["asof_pitcher_n"].clip(lower=0) + 200.0
    )
    output["stable_recent_gap_reliable"] = output["stable_recent_gap"] * reliability
    recent_middle = (
        0.6 * output["asof_pitcher_prev3_game_middle_rate"]
        + 0.4 * output["asof_pitcher_prev5_game_middle_rate"]
    )
    output["stable_recent_middle"] = recent_middle
    output["stable_recent_middle_gap"] = (
        recent_middle - output["asof_pitcher_middle_rate"]
    )
    return output
