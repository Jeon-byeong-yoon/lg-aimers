"""Cross-fitted sklearn TargetEncoder plus HGB candidate."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL


HIGH_CARDINAL_CATEGORICAL = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "game_month"
]


def target_encoder_hgb_pipeline(frame):
    categorical = [
        column for column in LOW_CARDINAL_CATEGORICAL + HIGH_CARDINAL_CATEGORICAL
        if column in frame
    ]
    numeric = [column for column in frame if column not in categorical]
    model = make_pipeline(
        ColumnTransformer(
            [
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        TargetEncoder(
                            target_type="binary", smooth="auto", cv=5,
                            shuffle=True, random_state=20260816,
                        ),
                    ),
                    categorical,
                ),
                ("numeric", SimpleImputer(strategy="median"), numeric),
            ]
        ),
        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.045,
            max_leaf_nodes=31,
            min_samples_leaf=150,
            l2_regularization=10.0,
            random_state=20260816,
        ),
    )
    return model, list(frame.columns)
