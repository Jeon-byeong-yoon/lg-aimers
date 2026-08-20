"""Hierarchically smoothed prior-season Trackman context aggregates."""

import numpy as np
import pandas as pd

from trackman_features import KEYS, PHYSICAL, PITCH_GROUPS


CONTEXT = ["outs_before", "inning_bucket", "top_bottom"]
VALUE_COLUMNS = PHYSICAL + [f"pitch_group_{group}" for group in PITCH_GROUPS]
SMOOTHING = (100.0, 500.0)


def add_context_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], bins=[-1, 3, 6, 9, np.inf],
        labels=False, include_lowest=True,
    ).astype("int8")
    return output


def prepare_context_trackman(trackman: pd.DataFrame) -> pd.DataFrame:
    output = add_context_keys(trackman)
    output["top_bottom"] = output["top_bottom"].map({"Top": "T", "Bottom": "B"})
    return output


def build_context_lookup(trackman: pd.DataFrame, before_season: int) -> pd.DataFrame:
    history = trackman.loc[
        (trackman["season"] < before_season)
        & trackman["outs_before"].between(0, 2)
        & trackman["top_bottom"].isin(["T", "B"])
    ]
    detail_keys = KEYS + CONTEXT
    parent = history.groupby(KEYS, observed=True)[VALUE_COLUMNS].mean().reset_index()
    parent = parent.rename(columns={column: f"__parent_{column}" for column in VALUE_COLUMNS})
    detail = history.groupby(detail_keys, observed=True)[VALUE_COLUMNS].agg(["mean", "count"])
    detail.columns = [f"{column}__{stat}" for column, stat in detail.columns]
    detail = detail.reset_index().merge(parent, how="left", on=KEYS, validate="many_to_one")
    output = detail[detail_keys].copy()
    counts = detail[f"{VALUE_COLUMNS[0]}__count"].astype(float)
    output["tm_ctx_history_n"] = counts
    output["tm_ctx_log1p_n"] = np.log1p(counts)
    for column in VALUE_COLUMNS:
        mean = detail[f"{column}__mean"]
        parent_mean = detail[f"__parent_{column}"]
        for smoothing in SMOOTHING:
            output[f"tm_ctx_{column}_s{int(smoothing)}"] = (
                mean * counts + parent_mean * smoothing
            ) / (counts + smoothing)
    return output


def add_context_trackman_features(frame: pd.DataFrame, trackman: pd.DataFrame) -> pd.DataFrame:
    source = add_context_keys(frame)
    pieces = []
    for season in sorted(source["season"].unique()):
        rows = source.loc[source["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_context_lookup(trackman, int(season))
        rows = rows.merge(lookup, how="left", on=KEYS + CONTEXT, validate="many_to_one")
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Context Trackman join changed row identity")
    return output
