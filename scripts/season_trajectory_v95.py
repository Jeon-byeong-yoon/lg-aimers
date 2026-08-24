"""Per-season control rates for each pitcher, and the multi-season trajectory.

The feature set currently holds two views of a pitcher: the career-cumulative
``asof_*`` rates, and (since V92) the current-season reconstruction. Neither
encodes a *trajectory* — whether a pitcher has been declining for three seasons
or is stable. That is information no existing column can reproduce.

Because ``asof_pitcher_n`` is strictly career-cumulative and continuous across
seasons, freezing the cumulative state at the end of each season lets a single
season's rate be recovered by differencing two adjacent anchors:

    rate(season s) = (S_end(s) - S_end(s-1)) / (n_end(s) - n_end(s-1))

All anchors come from official training rows. A row in season s only ever reads
anchors from seasons strictly before s, so nothing leaks forward, and a 2025
evaluation row reads the 2022/2023/2024 anchors exactly the way a 2024 training
row reads the 2021/2022/2023 ones. No evaluation row consults another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ENTITY = "pitcher_id"
COUNT = "asof_pitcher_n"
RATE = "asof_pitcher_success_rate"
LOOKBACK = 3
MIN_SEASON_PITCHES = 50.0
FEATURE_NAMES = [
    "traj_prev1_season_rate",
    "traj_prev2_season_rate",
    "traj_prev3_season_rate",
    "traj_prev1_minus_prev2",
    "traj_prev2_minus_prev3",
    "traj_slope",
    "traj_dispersion",
    "traj_seasons_available",
    "traj_prev1_log_pitches",
]


def season_endpoints(frame):
    """Cumulative (n, successes) at the end of every season, per pitcher."""
    work = frame[[ENTITY, "season", COUNT, RATE]].copy()
    work["__sum"] = work[COUNT] * work[RATE].fillna(0.0)
    ordered = work.sort_values(COUNT)
    last = ordered.groupby([ENTITY, "season"], observed=True).tail(1)
    return {
        int(season): part.set_index(ENTITY)[[COUNT, "__sum"]]
        for season, part in last.groupby("season", observed=True)
    }


def _season_rate(entities, endpoints, season, league_level):
    """Rate for one season, from the gap between its two bracketing anchors."""
    current = endpoints.get(season)
    if current is None:
        return np.full(len(entities), np.nan), np.full(len(entities), np.nan)
    n_now = entities.map(current[COUNT]).to_numpy(dtype=float)
    s_now = entities.map(current["__sum"]).to_numpy(dtype=float)
    previous = endpoints.get(season - 1)
    if previous is None:
        n_before = np.zeros(len(entities))
        s_before = np.zeros(len(entities))
    else:
        n_before = entities.map(previous[COUNT]).to_numpy(dtype=float)
        s_before = entities.map(previous["__sum"]).to_numpy(dtype=float)
        n_before = np.where(np.isnan(n_before), 0.0, n_before)
        s_before = np.where(np.isnan(s_before), 0.0, s_before)
    pitches = n_now - n_before
    successes = s_now - s_before
    valid = np.isfinite(pitches) & (pitches >= MIN_SEASON_PITCHES)
    rate = np.where(valid, successes / np.where(pitches > 0, pitches, 1.0), np.nan)
    if league_level is not None:
        rate = rate - league_level
    return rate, np.where(valid, pitches, np.nan)


def add_trajectory_features(frame, endpoints, reference_season=None, league_levels=None):
    """Attach the trajectory block, reading only seasons before the row's own."""
    output = frame.copy()
    entities = output[ENTITY]
    if reference_season is None:
        seasons = output["season"].astype(int).to_numpy()
    else:
        seasons = np.full(len(output), int(reference_season))

    rates, counts = [], []
    for offset in range(1, LOOKBACK + 1):
        column_rate = np.full(len(output), np.nan)
        column_count = np.full(len(output), np.nan)
        for season in np.unique(seasons):
            target = int(season) - offset
            level = None if league_levels is None else league_levels.get(target)
            values, pitches = _season_rate(entities, endpoints, target, level)
            mask = seasons == season
            column_rate[mask] = values[mask]
            column_count[mask] = pitches[mask]
        rates.append(column_rate)
        counts.append(column_count)

    output["traj_prev1_season_rate"] = rates[0]
    output["traj_prev2_season_rate"] = rates[1]
    output["traj_prev3_season_rate"] = rates[2]
    output["traj_prev1_minus_prev2"] = rates[0] - rates[1]
    output["traj_prev2_minus_prev3"] = rates[1] - rates[2]

    stack = np.vstack(rates)
    available = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        # Slope of a least-squares line through the available season rates,
        # oriented so that a positive value means "improving toward the present".
        weights = np.array([1.0, 0.0, -1.0])[:, None]
        centered = stack - np.nanmean(stack, axis=0)
        slope = np.nansum(centered * weights, axis=0) / 2.0
        output["traj_slope"] = np.where(available >= 2, slope, np.nan)
        output["traj_dispersion"] = np.where(available >= 2, np.nanstd(stack, axis=0), np.nan)
    output["traj_seasons_available"] = available.astype(float)
    output["traj_prev1_log_pitches"] = np.log1p(np.nan_to_num(counts[0], nan=0.0))
    return output


def build_training_trajectory(frame, league_levels=None):
    """Season-by-season construction, mirroring how a 2025 row is transformed."""
    endpoints = season_endpoints(frame)
    pieces = []
    for season in sorted(frame["season"].unique()):
        current = frame.loc[frame["season"] == season]
        usable = {k: v for k, v in endpoints.items() if k < int(season)}
        pieces.append(
            add_trajectory_features(current, usable, None, league_levels)
        )
    output = pd.concat(pieces).sort_index()
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Trajectory construction changed row identity")
    return output
