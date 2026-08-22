"""학습·검증·추론에 공통 적용하는 행 단위 As-of 신뢰도 피처."""

import numpy as np
import pandas as pd


RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)


def learn_asof_priors(frame: pd.DataFrame) -> dict:
    """이전 학습 구간에서만 각 비율의 고정 prior를 학습한다."""
    return {
        column: float(frame[column].dropna().median())
        for column in RATE_COLUMNS
    }


def _smoothed(rate, count, prior, strength):
    count = count.fillna(0).clip(lower=0)
    rate = rate.fillna(prior)
    return (rate * count + prior * strength) / (count + strength)


def add_asof_reliability_features(frame, priors, variant):
    """현재 행과 이전 학습 구간에서 고정한 prior만 사용한다."""
    output = frame.copy()
    pitcher_n = output["asof_pitcher_n"].fillna(0).clip(lower=0)
    batter_n = output["asof_batter_n"].fillna(0).clip(lower=0)
    for column in RATE_COLUMNS:
        count = batter_n if column.startswith("asof_batter_") else pitcher_n
        short = column.removeprefix("asof_").removesuffix("_rate")
        for strength in (100.0, 500.0):
            output[f"v79_{short}_smooth_{int(strength)}"] = _smoothed(
                output[column], count, priors[column], strength
            )

    if variant in {"hierarchy", "pitchmix"}:
        pitcher_success = output["v79_pitcher_success_smooth_500"]
        batter_success = output["v79_batter_success_smooth_500"]
        total_n = pitcher_n + batter_n
        output["v79_pitcher_batter_consensus"] = np.where(
            total_n > 0,
            (pitcher_n * pitcher_success + batter_n * batter_success) / total_n.clip(lower=1),
            priors["asof_pitcher_success_rate"],
        )
        output["v79_pitcher_batter_gap"] = pitcher_success - batter_success
        output["v79_joint_reliability"] = total_n / (total_n + 500.0)
        output["v79_pitcher_reliability"] = pitcher_n / (pitcher_n + 500.0)
        output["v79_batter_reliability"] = batter_n / (batter_n + 500.0)

    if variant == "pitchmix":
        mix_n = output["asof_pitcher_pitchmix_n"].fillna(0).clip(lower=0)
        rates = output[[
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]].fillna(0).clip(lower=0)
        rate_sum = rates.sum(axis=1).replace(0, 1.0)
        normalized = rates.div(rate_sum, axis=0)
        output["v79_pitchmix_entropy"] = -(
            normalized * np.log(normalized.clip(lower=1e-12))
        ).sum(axis=1)
        output["v79_pitchmix_max_share"] = normalized.max(axis=1)
        output["v79_pitchmix_reliability"] = mix_n / (mix_n + 500.0)
        output["v79_pitchmix_entropy_reliable"] = (
            output["v79_pitchmix_entropy"] * output["v79_pitchmix_reliability"]
        )
    return output

