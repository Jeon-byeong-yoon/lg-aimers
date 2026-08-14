"""Strictly prior-season Trackman aggregates with no player-ID assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["pitcher_hand", "batter_hand", "game_month", "balls_before", "strikes_before"]
PHYSICAL = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
    "rel_height", "rel_side", "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
HAND_MAP = {"Left": 1, "Right": 2}


def prepare_trackman(trackman: pd.DataFrame) -> pd.DataFrame:
    data = trackman.copy()
    data["pitcher_hand"] = data["pitcher_hand"].map(HAND_MAP)
    data["batter_hand"] = data["batter_hand"].map(HAND_MAP)
    for group in PITCH_GROUPS:
        data[f"pitch_group_{group}"] = (data["pitch_type_group"] == group).astype("float32")
    return data


def build_lookup(trackman: pd.DataFrame, before_season: int) -> pd.DataFrame:
    """Aggregate only Trackman rows strictly before ``before_season``."""
    history = trackman.loc[trackman["season"] < before_season]
    if history.empty:
        return pd.DataFrame(columns=KEYS)
    aggregations = {column: ["mean", "std"] for column in PHYSICAL}
    aggregations.update({f"pitch_group_{group}": "mean" for group in PITCH_GROUPS})
    lookup = history.groupby(KEYS, observed=True).agg(aggregations)
    lookup.columns = [
        f"tm_{column}_{stat}" if stat else f"tm_{column}"
        for column, stat in lookup.columns.to_flat_index()
    ]
    lookup["tm_history_n"] = history.groupby(KEYS, observed=True).size()
    return lookup.reset_index()


def add_trackman_features(
    frame: pd.DataFrame, trackman: pd.DataFrame, season_column: str = "season"
) -> pd.DataFrame:
    """Attach fixed prior-season lookups independently to every main-data row."""
    output = frame.copy()
    feature_frames = []
    for season in sorted(output[season_column].unique()):
        rows = output.loc[output[season_column] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_lookup(trackman, int(season))
        rows = rows.merge(lookup, how="left", on=KEYS, validate="many_to_one")
        rows = rows.set_index("__original_index")
        feature_frames.append(rows)
    combined = pd.concat(feature_frames).sort_index()
    combined.index.name = frame.index.name
    if len(combined) != len(frame) or not combined.index.equals(frame.index):
        raise ValueError("Trackman feature join changed row identity")
    return combined
