"""A prior for the in-season reconstruction that tracks the season, not the career.

`build_priors` in `inseason_asof_features_v92` sets the shrinkage prior to the mean of
the career as-of rate column. But `inside_rate` estimates a *current-season* quantity,
so shrinking it toward a career average pulls it toward a league that no longer exists.
Measured against what each fold's target season actually was:

    fold   history      prior      target      gap
    2022   2019-2021   0.547341   0.528920   +0.0184
    2023   2019-2022   0.544313   0.499957   +0.0444
    2024   2019-2023   0.540175   0.486105   +0.0541
    2025   2019-2024   0.535228   ~0.470     +0.060 to +0.073

The gap grows every season because the league rate fell from 0.5647 to 0.4861 while a
career average lags behind it. At `inside_n = 100` the prior carries weight 20/120, so a
+0.07 error puts +0.0117 into the feature -- which is what the low-experience rows of the
2024 fold show (+0.0157 in F, +0.0077 in R, crossing zero near a thousand pitches).

The fix keeps the same machinery. The population's in-season rate for a season is
recoverable by the very anchor-differencing that V92 built for individual pitchers:

    population inside rate = sum(career_sum - anchor_sum) / sum(career_n - anchor_n)

computed over that season's rows. That is a genuine within-season population value for
every rate column, not just the target, so one uniform rule covers all seven. Those
values are then fitted linearly across the history seasons and projected to the target
season, which uses no information from the season being predicted.

Projecting rather than taking the last season matters: the decline is steady but not
uniform (-0.029 from 2022 to 2023 against -0.014 from 2023 to 2024), and a projection
uses the whole trend instead of one noisy step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from inseason_asof_features_v92 import ENTITIES, build_anchor


def population_inside_rates(frame, groups=("pitcher", "batter")):
    """Per-season within-season population rate for every reconstructed column."""
    seasons = sorted(frame["season"].unique())
    records = {}
    for season in seasons:
        current = frame.loc[frame["season"] == season]
        history = frame.loc[frame["season"] < season]
        if history.empty:
            continue
        row = {}
        for group in groups:
            entity, count_column, rates = ENTITIES[group]
            anchor = build_anchor(history, group)
            merged = current[[entity, count_column, *rates]].merge(
                anchor, how="left", on=entity, validate="many_to_one")
            career_n = merged[count_column].fillna(0.0).to_numpy(dtype=float)
            anchor_n = merged[f"anchor_n_{group}"].fillna(0.0).to_numpy(dtype=float)
            inside_n = np.clip(career_n - anchor_n, 0.0, None)
            for rate in rates:
                career_sum = career_n * merged[rate].fillna(0.0).to_numpy(dtype=float)
                anchor_sum = merged[f"anchor_sum_{rate}"].fillna(
                    0.0).to_numpy(dtype=float)
                inside_sum = np.minimum(
                    np.clip(career_sum - anchor_sum, 0.0, None), inside_n)
                total = inside_n.sum()
                row[rate] = float(inside_sum.sum() / total) if total > 0 else np.nan
        records[season] = row
    return pd.DataFrame.from_dict(records, orient="index").sort_index()


def projected_priors(frame, target_season, groups=("pitcher", "batter"),
                     table=None):
    """Linear projection of the population in-season rates to ``target_season``.

    ``frame`` must contain only seasons strictly before ``target_season``; the caller is
    responsible for that, and the assertion below enforces it so a leak cannot pass
    silently.
    """
    assert frame["season"].max() < target_season, (
        f"prior frame reaches {frame['season'].max()} for target {target_season}")
    if table is None:
        table = population_inside_rates(frame, groups)
    priors = {}
    for group in groups:
        for rate in ENTITIES[group][2]:
            series = table[rate].dropna() if rate in table else pd.Series(dtype=float)
            if len(series) >= 2:
                slope, intercept = np.polyfit(
                    series.index.to_numpy(dtype=float), series.to_numpy(), 1)
                value = slope * target_season + intercept
            elif len(series) == 1:
                value = float(series.iloc[0])
            else:
                # No reconstructed history to project from; fall back to the career
                # average, which is what V92 always used.
                value = float(frame[rate].mean())
            priors[rate] = float(np.clip(value, 0.0, 1.0))
    return priors


__all__ = ["population_inside_rates", "projected_priors"]
