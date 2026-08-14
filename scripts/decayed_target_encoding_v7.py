"""Strictly prior-season player target encoding with recency decay."""

from __future__ import annotations

import numpy as np
import pandas as pd


ENTITIES = ["pitcher_id", "batter_id"]
STRENGTHS = (50.0, 200.0)


def _lookup(
    history: pd.DataFrame, entity: str, prediction_season: int, decay: float
) -> pd.DataFrame:
    weighted = history[[entity, "season", "__target"]].copy()
    weighted["__weight"] = decay ** (
        prediction_season - 1 - weighted["season"]
    )
    weighted["__weighted_target"] = weighted["__target"] * weighted["__weight"]
    prior = float(weighted["__weighted_target"].sum() / weighted["__weight"].sum())
    stats = (
        weighted.groupby(entity, observed=True)
        .agg(weighted_sum=("__weighted_target", "sum"), weighted_n=("__weight", "sum"))
        .reset_index()
    )
    for strength in STRENGTHS:
        stats[f"dte_{entity}_{decay:g}_{int(strength)}"] = (
            stats["weighted_sum"] + prior * strength
        ) / (stats["weighted_n"] + strength)
    stats[f"dte_{entity}_{decay:g}_log_weighted_n"] = np.log1p(stats["weighted_n"])
    return stats.drop(columns=["weighted_sum", "weighted_n"])


def add_prior_season_decayed_encodings(
    frame: pd.DataFrame, target: pd.Series, decay: float
) -> pd.DataFrame:
    source = frame.copy()
    source["__target"] = target.to_numpy()
    pieces = []
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"] == season].drop(columns="__target").copy()
        current["__original_index"] = current.index
        history = source.loc[source["season"] < season]
        if history.empty:
            for entity in ENTITIES:
                for strength in STRENGTHS:
                    current[f"dte_{entity}_{decay:g}_{int(strength)}"] = np.nan
                current[f"dte_{entity}_{decay:g}_log_weighted_n"] = np.nan
        else:
            for entity in ENTITIES:
                current = current.merge(
                    _lookup(history, entity, int(season), decay),
                    how="left",
                    on=entity,
                    validate="many_to_one",
                )
        pieces.append(current.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Decayed encoding changed row identity")
    return output


def build_decayed_test_lookups(
    frame: pd.DataFrame, target: pd.Series, decay: float, prediction_season: int = 2025
) -> dict[str, pd.DataFrame]:
    source = frame.copy()
    source["__target"] = target.to_numpy()
    return {
        entity: _lookup(source, entity, prediction_season, decay)
        for entity in ENTITIES
    }


def add_test_decayed_encodings(
    frame: pd.DataFrame, lookups: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    output = frame.copy()
    output["__original_index"] = output.index
    for entity in ENTITIES:
        output = output.merge(
            lookups[entity], how="left", on=entity, validate="many_to_one"
        )
    output = output.set_index("__original_index")
    output.index.name = frame.index.name
    return output
