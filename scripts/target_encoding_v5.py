"""Chronological target encodings using strictly earlier seasons."""

from __future__ import annotations

import pandas as pd
import numpy as np


ENTITY_COLUMNS = ["pitcher_id", "batter_id"]
STRENGTHS = (50.0, 200.0)


def _lookup(history: pd.DataFrame, entity: str, prior: float) -> pd.DataFrame:
    stats = history.groupby(entity, observed=True)["__target"].agg(["sum", "count"])
    lookup = stats.reset_index()
    for strength in STRENGTHS:
        lookup[f"te_{entity}_{int(strength)}"] = (
            lookup["sum"] + prior * strength
        ) / (lookup["count"] + strength)
    lookup[f"te_{entity}_log_count"] = np.log1p(lookup["count"])
    return lookup.drop(columns=["sum", "count"])


def add_prior_season_target_encodings(
    frame: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    """Add encodings to training rows using only strictly earlier seasons."""
    source = frame.copy()
    source["__target"] = target.to_numpy()
    pieces = []
    for season in sorted(source["season"].unique()):
        current = source.loc[source["season"] == season].drop(columns="__target").copy()
        current["__original_index"] = current.index
        history = source.loc[source["season"] < season]
        if history.empty:
            for entity in ENTITY_COLUMNS:
                for strength in STRENGTHS:
                    current[f"te_{entity}_{int(strength)}"] = float("nan")
                current[f"te_{entity}_log_count"] = float("nan")
        else:
            prior = float(history["__target"].mean())
            for entity in ENTITY_COLUMNS:
                current = current.merge(
                    _lookup(history, entity, prior),
                    how="left",
                    on=entity,
                    validate="many_to_one",
                )
        pieces.append(current.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Target encoding changed row identity")
    return output


def build_test_lookups(frame: pd.DataFrame, target: pd.Series) -> dict[str, pd.DataFrame]:
    """Build fixed mappings from all official training rows for future test rows."""
    source = frame.copy()
    source["__target"] = target.to_numpy()
    prior = float(source["__target"].mean())
    return {entity: _lookup(source, entity, prior) for entity in ENTITY_COLUMNS}


def add_test_target_encodings(
    frame: pd.DataFrame, lookups: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Apply fixed training-only mappings independently to test rows."""
    output = frame.copy()
    output["__original_index"] = output.index
    for entity in ENTITY_COLUMNS:
        output = output.merge(
            lookups[entity], how="left", on=entity, validate="many_to_one"
        )
    output = output.set_index("__original_index")
    output.index.name = frame.index.name
    return output
