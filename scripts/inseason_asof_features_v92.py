"""Current-season reconstruction of the official as-of histories.

The official ``asof_*`` counters are strictly career-cumulative and continue
across season boundaries (verified: for every pitcher the first ``asof_*_n`` of
a season equals the last value of the previous season plus one). Therefore the
career state at the end of a fixed anchor period can be subtracted from a row's
own as-of values to recover that entity's *current-season-only* history.

Compliance
----------
Every feature here is a function of two quantities only:

1. the values on the row being transformed, all of which are official ``asof_*``
   columns describing the state strictly before the current pitch, and
2. a lookup table frozen at training time from official training rows.

No other evaluation row is consulted, so a test batch produces identical output
whether it is transformed all at once or one row at a time. No post-event
information about the current pitch is used, and no Trackman data is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PITCHER_RATES = (
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_strike_rate",
)
BATTER_RATES = ("asof_batter_success_rate", "asof_batter_middle_rate")
PITCHMIX_RATES = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
ENTITIES = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", PITCHER_RATES),
    "batter": ("batter_id", "asof_batter_n", BATTER_RATES),
    "pitchmix": ("pitcher_id", "asof_pitcher_pitchmix_n", PITCHMIX_RATES),
}
# Chosen a priori in V92 and never refitted until V102. Kept as the defaults so
# every existing artifact and script reproduces byte-for-byte.
SHRINKAGE = 50.0
RELIABILITY_SCALE = 150.0


def feature_names(groups=("pitcher", "batter")):
    names = []
    for group in groups:
        _, _, rates = ENTITIES[group]
        for rate in rates:
            names.append(f"ins_{rate}")
            names.append(f"dlt_{rate}")
        names.append(f"ins_log_n_{group}")
    if "pitcher" in groups:
        names.append("ins_pitcher_reliability")
        names.append("ins_pitcher_seasons_since_seen")
    return names


def build_anchor(frame, group):
    """Freeze the career cumulative state at the end of ``frame`` per entity.

    The state is taken *exclusive* of each entity's last row, so every stored
    quantity is an official as-of value and no current-pitch outcome is needed.
    """
    entity, count_column, rates = ENTITIES[group]
    ordered = frame.sort_values(count_column)
    last = ordered.groupby(entity, observed=True).tail(1)
    anchor = {entity: last[entity].to_numpy(), f"anchor_n_{group}": last[count_column].to_numpy()}
    for rate in rates:
        anchor[f"anchor_sum_{rate}"] = (
            last[count_column].to_numpy() * last[rate].fillna(0.0).to_numpy()
        )
    if "season" in last.columns:
        anchor[f"anchor_season_{group}"] = last["season"].to_numpy()
    return pd.DataFrame(anchor)


def build_anchors(frame, groups=("pitcher", "batter")):
    return {group: build_anchor(frame, group) for group in groups}


def build_priors(frame, groups=("pitcher", "batter")):
    priors = {}
    for group in groups:
        for rate in ENTITIES[group][2]:
            priors[rate] = float(frame[rate].mean())
    return priors


def add_inseason_features(frame, anchors, priors, current_season=None,
                          shrinkage=SHRINKAGE, reliability_scale=RELIABILITY_SCALE):
    """Attach current-season reconstructions to ``frame`` row by row."""
    output = frame.copy()
    output["__order"] = np.arange(len(output))
    for group, anchor in anchors.items():
        entity, count_column, rates = ENTITIES[group]
        merged = output[["__order", entity, count_column, *rates]].merge(
            anchor, how="left", on=entity, validate="many_to_one"
        )
        merged = merged.sort_values("__order")
        career_n = merged[count_column].fillna(0.0).to_numpy(dtype=float)
        anchor_n = merged[f"anchor_n_{group}"].fillna(0.0).to_numpy(dtype=float)
        inside_n = np.clip(career_n - anchor_n, 0.0, None)
        for rate in rates:
            career_sum = career_n * merged[rate].fillna(0.0).to_numpy(dtype=float)
            anchor_sum = merged[f"anchor_sum_{rate}"].fillna(0.0).to_numpy(dtype=float)
            inside_sum = np.clip(career_sum - anchor_sum, 0.0, None)
            inside_sum = np.minimum(inside_sum, inside_n)
            prior = priors[rate]
            inside_rate = (inside_sum + prior * shrinkage) / (inside_n + shrinkage)
            output[f"ins_{rate}"] = inside_rate
            output[f"dlt_{rate}"] = inside_rate - merged[rate].fillna(prior).to_numpy(dtype=float)
        output[f"ins_log_n_{group}"] = np.log1p(inside_n)
        if group == "pitcher":
            output["ins_pitcher_reliability"] = inside_n / (inside_n + reliability_scale)
            season_column = f"anchor_season_{group}"
            if current_season is None:
                reference = output["season"].to_numpy(dtype=float)
            else:
                reference = np.full(len(output), float(current_season))
            last_seen = merged[season_column].to_numpy(dtype=float) if season_column in merged else np.full(len(output), np.nan)
            gap = reference - last_seen
            output["ins_pitcher_seasons_since_seen"] = np.where(np.isnan(gap), -1.0, gap)
    return output.drop(columns="__order")


def add_training_inseason_features(frame, groups=("pitcher", "batter"),
                                  shrinkage=SHRINKAGE,
                                  reliability_scale=RELIABILITY_SCALE):
    """Build the same features for training rows, one season at a time.

    Each season uses an anchor frozen at the end of the *previous* season, which
    is exactly the relation a 2025 evaluation row has to the 2019-2024 training
    data. The earliest season has no anchor and is left missing.
    """
    pieces = []
    for season in sorted(frame["season"].unique()):
        current = frame.loc[frame["season"] == season]
        history = frame.loc[frame["season"] < season]
        if history.empty:
            block = pd.DataFrame(
                np.nan, index=current.index, columns=feature_names(groups)
            )
            pieces.append(pd.concat([current, block], axis=1))
            continue
        anchors = build_anchors(history, groups)
        priors = build_priors(history, groups)
        pieces.append(add_inseason_features(
            current, anchors, priors, current_season=season,
            shrinkage=shrinkage, reliability_scale=reliability_scale,
        ))
    output = pd.concat(pieces).sort_index()
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("In-season feature construction changed row identity")
    return output


def _delta(frame, name):
    return np.nan_to_num(frame[f"dlt_asof_{name}"].to_numpy(dtype=float), nan=0.0)


def _success(frame):
    return _delta(frame, "pitcher_success_rate")


def _combined(frame):
    """Success delta averaged with the three official failure-mode deltas.

    The task defines a failed pitch as one down the middle, far outside the zone,
    or opposite to the catcher's request, so those three rates move against
    control and enter with a negative sign. Sub-weights are fixed here rather
    than fitted, because fitting them on the validation seasons is what sank V78
    and V80.
    """
    failure = -(
        _delta(frame, "pitcher_middle_rate")
        + _delta(frame, "pitcher_ball_rate")
        + _delta(frame, "pitcher_reverse_rate")
    ) / 3.0
    return 0.5 * _success(frame) + 0.5 * failure


COMPOSITIONS = {"success": _success, "combined": _combined}


def drift_correction(frame, weight, composition="success"):
    """Shrunk current-season drift signal. ``success`` reproduces V92 exactly."""
    delta = COMPOSITIONS[composition](frame)
    reliability = np.nan_to_num(
        frame["ins_pitcher_reliability"].to_numpy(dtype=float), nan=0.0
    )
    return weight * np.nan_to_num(delta * reliability, nan=0.0)
