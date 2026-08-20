"""Prior-season Trackman archetypes learned without cross-dataset player matching."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from trackman_features import KEYS, PHYSICAL, build_lookup


CLUSTER_COUNTS = (6, 10, 16)


def build_archetype_lookup(trackman: pd.DataFrame, before_season: int) -> pd.DataFrame:
    lookup = build_lookup(trackman, before_season)
    if lookup.empty:
        return lookup
    physical = [
        f"tm_{column}_{stat}" for column in PHYSICAL for stat in ("mean", "std")
    ]
    matrix = SimpleImputer(strategy="median").fit_transform(lookup[physical])
    matrix = StandardScaler().fit_transform(matrix)
    output = lookup[KEYS].copy()
    for requested_clusters in CLUSTER_COUNTS:
        clusters = min(requested_clusters, len(output))
        model = KMeans(n_clusters=clusters, n_init=20, random_state=42)
        labels = model.fit_predict(matrix)
        distances = model.transform(matrix)
        output[f"tm_arch{requested_clusters}_distance"] = distances[
            np.arange(len(output)), labels
        ]
        output[f"tm_arch{requested_clusters}_margin"] = (
            np.partition(distances, 1, axis=1)[:, 1]
            - distances[np.arange(len(output)), labels]
            if clusters > 1 else 0.0
        )
        for cluster in range(requested_clusters):
            output[f"tm_arch{requested_clusters}_{cluster}"] = (
                labels == cluster
            ).astype("int8")
    return output


def add_trackman_archetypes(frame: pd.DataFrame, trackman: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for season in sorted(frame["season"].unique()):
        rows = frame.loc[frame["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_archetype_lookup(trackman, int(season))
        rows = rows.merge(lookup, how="left", on=KEYS, validate="many_to_one")
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Trackman archetype join changed row identity")
    return output
