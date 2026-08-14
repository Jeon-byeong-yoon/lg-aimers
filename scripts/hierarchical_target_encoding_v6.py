"""Strictly prior-season hierarchical target encodings for baseball matchups."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUPS = {
    "pitcher_vs_batter_hand": {
        "keys": ["pitcher_id", "batter_hand"],
        "parent": ["pitcher_id"],
        "strengths": (30.0, 100.0),
    },
    "pitcher_count": {
        "keys": ["pitcher_id", "balls_before", "strikes_before"],
        "parent": ["pitcher_id"],
        "strengths": (50.0, 200.0),
    },
    "batter_vs_pitcher_hand": {
        "keys": ["batter_id", "pitcher_hand"],
        "parent": ["batter_id"],
        "strengths": (30.0, 100.0),
    },
    "pitcher_batter": {
        "keys": ["pitcher_id", "batter_id"],
        "parent": ["pitcher_id"],
        "strengths": (100.0, 500.0),
    },
}


def _name(group: str, strength: float) -> str:
    return f"hte_{group}_{int(strength)}"


def _make_lookup(
    history: pd.DataFrame, group: str, global_prior: float
) -> pd.DataFrame:
    spec = GROUPS[group]
    keys = spec["keys"]
    parent_keys = spec["parent"]
    parent = (
        history.groupby(parent_keys, observed=True)["__target"]
        .agg(parent_sum="sum", parent_count="count")
        .reset_index()
    )
    parent["__parent_rate"] = (
        parent["parent_sum"] + global_prior * 100.0
    ) / (parent["parent_count"] + 100.0)
    child = (
        history.groupby(keys, observed=True)["__target"]
        .agg(child_sum="sum", child_count="count")
        .reset_index()
    )
    child = child.merge(
        parent[parent_keys + ["__parent_rate"]],
        how="left",
        on=parent_keys,
        validate="many_to_one",
    )
    for strength in spec["strengths"]:
        child[_name(group, strength)] = (
            child["child_sum"] + child["__parent_rate"] * strength
        ) / (child["child_count"] + strength)
    child[f"hte_{group}_log_count"] = np.log1p(child["child_count"])
    return child.drop(
        columns=["child_sum", "child_count", "__parent_rate"]
    )


def add_prior_season_hierarchical_encodings(
    frame: pd.DataFrame,
    target: pd.Series,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    selected = groups or list(GROUPS)
    source = frame.copy()
    source["__target"] = target.to_numpy()
    pieces = []
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"] == season].drop(columns="__target").copy()
        current["__original_index"] = current.index
        history = source.loc[source["season"] < season]
        if history.empty:
            for group in selected:
                for strength in GROUPS[group]["strengths"]:
                    current[_name(group, strength)] = np.nan
                current[f"hte_{group}_log_count"] = np.nan
        else:
            prior = float(history["__target"].mean())
            for group in selected:
                lookup = _make_lookup(history, group, prior)
                current = current.merge(
                    lookup,
                    how="left",
                    on=GROUPS[group]["keys"],
                    validate="many_to_one",
                )
        pieces.append(current.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Hierarchical encoding changed row identity")
    return output


def build_hierarchical_test_lookups(
    frame: pd.DataFrame, target: pd.Series, groups: list[str]
) -> dict[str, pd.DataFrame]:
    source = frame.copy()
    source["__target"] = target.to_numpy()
    prior = float(source["__target"].mean())
    return {group: _make_lookup(source, group, prior) for group in groups}


def add_test_hierarchical_encodings(
    frame: pd.DataFrame, lookups: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    output = frame.copy()
    output["__original_index"] = output.index
    for group, lookup in lookups.items():
        output = output.merge(
            lookup,
            how="left",
            on=GROUPS[group]["keys"],
            validate="many_to_one",
        )
    output = output.set_index("__original_index")
    output.index.name = frame.index.name
    return output
