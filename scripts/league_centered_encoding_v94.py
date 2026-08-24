"""Recentre Target Encodings on the league level of the seasons that built them.

``te_*`` and ``hte_*`` are prior-season target means, so they inherit the league
control-success level of whichever seasons produced them. While that level fell
from 0.5647 (2019) to 0.4861 (2024) the encodings drifted above the truth:

    season  te_pitcher_200  target   difference
    2020            0.5657  0.5327      +0.0330
    2022            0.5386  0.5289      +0.0097
    2023            0.5365  0.5000      +0.0366
    2024            0.5260  0.4861      +0.0399

Used alone as a prediction each encoding scores strongly *negative* on 2023 and
2024 (te_pitcher_200: +527, -668, -870), i.e. worse than a constant. Subtracting
the prior-season league mean turns "how often did this player succeed" into "how
much better than the league was this player", which is stationary across seasons.
The absolute level is then carried by the calibration shift and the V92 drift
term, both of which already track it.

Only official training rows and their own season labels are used, and the 2025
offset is a constant frozen at training time, so evaluation rows never interact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encoding_columns(frame):
    """Rate-like target encodings. Count columns are left alone."""
    return [
        column for column in frame.columns
        if column.startswith(("te_", "hte_")) and not column.endswith("log_count")
    ]


def prior_season_levels(seasons, target):
    """League control-success mean over all seasons strictly before each season."""
    frame = pd.DataFrame({"season": np.asarray(seasons), "target": np.asarray(target, dtype=float)})
    totals = frame.groupby("season")["target"].agg(["sum", "count"]).sort_index()
    levels = {}
    running_sum = running_count = 0.0
    for season, row in totals.iterrows():
        levels[int(season)] = running_sum / running_count if running_count else np.nan
        running_sum += float(row["sum"])
        running_count += float(row["count"])
    return levels, (running_sum / running_count)


def add_centered_encodings(frame, levels, columns, keep_original, fallback_level=None):
    """Attach ``<name>_vs_league`` columns; optionally drop the raw encodings."""
    output = frame.copy()
    if fallback_level is None:
        level = output["season"].astype(int).map(levels).to_numpy(dtype=float)
    else:
        level = np.full(len(output), float(fallback_level))
    for column in columns:
        output[f"{column}_vs_league"] = output[column].to_numpy(dtype=float) - level
    if not keep_original:
        output = output.drop(columns=columns)
    return output
