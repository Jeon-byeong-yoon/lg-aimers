"""Enhanced contextual Trackman aggregates with deltas and physical composites."""

import numpy as np
import pandas as pd

from contextual_trackman_v24 import (
    CONTEXT, SMOOTHING, VALUE_COLUMNS, add_context_keys, build_context_lookup,
)
from trackman_features import KEYS, PHYSICAL, PITCH_GROUPS


def build_enhanced_context_lookup(
    trackman: pd.DataFrame, before_season: int
) -> pd.DataFrame:
    history = trackman.loc[
        (trackman["season"] < before_season)
        & trackman["outs_before"].between(0, 2)
        & trackman["top_bottom"].isin(["T", "B"])
    ]
    lookup = build_context_lookup(trackman, before_season)
    parent = history.groupby(KEYS, observed=True)[VALUE_COLUMNS].mean().reset_index()
    parent = parent.rename(columns={column: f"__parent_{column}" for column in VALUE_COLUMNS})
    lookup = lookup.merge(parent, how="left", on=KEYS, validate="many_to_one")
    count = lookup["tm_ctx_history_n"].astype(float)
    for smoothing in SMOOTHING:
        suffix = f"s{int(smoothing)}"
        lookup[f"tm_ctx_reliability_{suffix}"] = count / (count + smoothing)
        for column in VALUE_COLUMNS:
            context = lookup[f"tm_ctx_{column}_{suffix}"]
            lookup[f"tm_ctx_delta_{column}_{suffix}"] = (
                context - lookup[f"__parent_{column}"]
            )
        lookup[f"tm_ctx_speed_loss_{suffix}"] = (
            lookup[f"tm_ctx_rel_speed_{suffix}"]
            - lookup[f"tm_ctx_zone_speed_{suffix}"]
        )
        lookup[f"tm_ctx_break_magnitude_{suffix}"] = np.hypot(
            lookup[f"tm_ctx_induced_vert_break_{suffix}"],
            lookup[f"tm_ctx_horz_break_{suffix}"],
        )
        lookup[f"tm_ctx_release_radius_{suffix}"] = np.hypot(
            lookup[f"tm_ctx_rel_height_{suffix}"],
            lookup[f"tm_ctx_rel_side_{suffix}"],
        )
        pitch_columns = [
            f"tm_ctx_pitch_group_{group}_{suffix}" for group in PITCH_GROUPS
        ]
        probabilities = lookup[pitch_columns].clip(lower=1e-8)
        lookup[f"tm_ctx_pitchmix_entropy_{suffix}"] = -(
            probabilities * np.log(probabilities)
        ).sum(axis=1)
    return lookup.drop(
        columns=[f"__parent_{column}" for column in VALUE_COLUMNS]
    )


def add_enhanced_context_features(
    frame: pd.DataFrame, trackman: pd.DataFrame
) -> pd.DataFrame:
    source = add_context_keys(frame)
    pieces = []
    for season in sorted(source["season"].unique()):
        rows = source.loc[source["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_enhanced_context_lookup(trackman, int(season))
        rows = rows.merge(
            lookup, how="left", on=KEYS + CONTEXT, validate="many_to_one"
        )
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Enhanced context Trackman join changed row identity")
    return output
