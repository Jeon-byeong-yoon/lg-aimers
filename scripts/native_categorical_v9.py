"""HistGradientBoosting pipeline with true low-cardinality categorical splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL


ADDITIONAL_NATIVE_CATEGORICAL = [
    "game_month",
    "pitcher_team_id",
    "batter_team_id",
]


def native_hist_pipeline(frame: pd.DataFrame):
    categorical = [
        column
        for column in LOW_CARDINAL_CATEGORICAL + ADDITIONAL_NATIVE_CATEGORICAL
        if column in frame
    ]
    numeric = [column for column in frame if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]
    )
    categorical_mask = [True] * len(categorical) + [False] * len(numeric)
    model = make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=5.0,
            categorical_features=categorical_mask,
            random_state=42,
        ),
    )
    return model, categorical + numeric
