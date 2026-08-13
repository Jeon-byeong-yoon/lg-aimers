"""Evaluate second-generation preprocessing on a chronological holdout."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from feature_engineering_v2 import (
    ID_COLUMNS,
    LOW_CARDINAL_CATEGORICAL,
    add_row_features,
    select_v2_features,
)


def score(y: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    brier = brier_score_loss(y, prediction)
    reference = float(y.mean() * (1.0 - y.mean()))
    return {
        "brier": float(brier),
        "bss": float(max(0.0, 100000.0 * (1.0 - brier / reference))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(y.mean()),
    }


def extra_trees_pipeline(frame: pd.DataFrame):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical]
    pipeline = make_pipeline(
        ColumnTransformer(
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
        ),
        ExtraTreesClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=100,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
    )
    return pipeline, list(frame.columns)


def hist_gbdt_pipeline(frame: pd.DataFrame):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical]
    pipeline = make_pipeline(
        ColumnTransformer(
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
        ),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=5.0,
            random_state=42,
        ),
    )
    return pipeline, list(frame.columns)


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    train_mask = data["season"] < 2024
    valid_mask = data["season"] == 2024
    prior = float(y.loc[train_mask].mean())
    features = select_v2_features(add_row_features(data, prior))

    started = perf_counter()
    results: dict[str, object] = {
        "split": "season < 2024 -> season == 2024",
        "train_target_prior": prior,
        "raw_feature_count": int(data.shape[1]),
        "engineered_feature_count": int(features.shape[1]),
        "models": {},
    }

    extra, extra_columns = extra_trees_pipeline(features)
    extra.fit(features.loc[train_mask, extra_columns], y.loc[train_mask])
    extra_prediction = extra.predict_proba(features.loc[valid_mask, extra_columns])[:, 1]
    results["models"]["extra_trees_v2"] = score(y.loc[valid_mask], extra_prediction)

    hist, hist_columns = hist_gbdt_pipeline(features)
    hist.fit(features.loc[train_mask, hist_columns], y.loc[train_mask])
    hist_prediction = hist.predict_proba(features.loc[valid_mask, hist_columns])[:, 1]
    results["models"]["hist_gbdt_v2"] = score(y.loc[valid_mask], hist_prediction)

    best = None
    for extra_weight in np.linspace(0.0, 1.0, 11):
        prediction = extra_weight * extra_prediction + (1.0 - extra_weight) * hist_prediction
        candidate = score(y.loc[valid_mask], prediction)
        candidate["extra_trees_weight"] = float(extra_weight)
        if best is None or candidate["brier"] < best["brier"]:
            best = candidate
    results["best_grid_blend"] = best
    results["elapsed_seconds"] = perf_counter() - started

    output = Path("artifacts/v2_validation_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
