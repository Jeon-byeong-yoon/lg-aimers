"""Tune model capacity while keeping the validated V2 features fixed."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features


HGB_CANDIDATES = {
    "hgb_v2": dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                   min_samples_leaf=100, l2_regularization=5.0),
    "hgb_small_leaf": dict(max_iter=220, learning_rate=0.05, max_leaf_nodes=15,
                           min_samples_leaf=100, l2_regularization=5.0),
    "hgb_large_leaf": dict(max_iter=180, learning_rate=0.05, max_leaf_nodes=63,
                           min_samples_leaf=150, l2_regularization=8.0),
    "hgb_more_regularized": dict(max_iter=220, learning_rate=0.05, max_leaf_nodes=31,
                                 min_samples_leaf=200, l2_regularization=12.0),
    "hgb_finer": dict(max_iter=300, learning_rate=0.035, max_leaf_nodes=31,
                      min_samples_leaf=100, l2_regularization=8.0),
}

EXTRA_CANDIDATES = {
    "extra_v2": dict(max_depth=14, min_samples_leaf=100, max_features=0.8),
    "extra_shallow": dict(max_depth=11, min_samples_leaf=150, max_features=0.8),
    "extra_deeper": dict(max_depth=18, min_samples_leaf=100, max_features=0.7),
    "extra_fine_leaf": dict(max_depth=16, min_samples_leaf=50, max_features=0.7),
}


def replace_estimator(pipeline, estimator) -> None:
    pipeline.steps[-1] = (pipeline.steps[-1][0], estimator)


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    train_mask = data["season"] < 2024
    valid_mask = data["season"] == 2024
    prior = float(y.loc[train_mask].mean())
    features = select_v2_features(add_row_features(data, prior))
    results = {"split": "season < 2024 -> season == 2024", "models": {}}
    predictions = {}

    for name, params in HGB_CANDIDATES.items():
        started = perf_counter()
        model, columns = hist_gbdt_pipeline(features)
        replace_estimator(model, HistGradientBoostingClassifier(**params, random_state=42))
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        predictions[name] = prediction
        results["models"][name] = score(y.loc[valid_mask], prediction)
        results["models"][name]["elapsed_seconds"] = perf_counter() - started
        print(name, results["models"][name], flush=True)

    for name, params in EXTRA_CANDIDATES.items():
        started = perf_counter()
        model, columns = extra_trees_pipeline(features)
        replace_estimator(
            model,
            ExtraTreesClassifier(
                n_estimators=120, n_jobs=-1, random_state=42, **params
            ),
        )
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        predictions[name] = prediction
        results["models"][name] = score(y.loc[valid_mask], prediction)
        results["models"][name]["elapsed_seconds"] = perf_counter() - started
        print(name, results["models"][name], flush=True)

    hgb_names = list(HGB_CANDIDATES)
    extra_names = list(EXTRA_CANDIDATES)
    blends = []
    for hgb_name in hgb_names:
        for extra_name in extra_names:
            for extra_weight in np.linspace(0.2, 0.8, 7):
                prediction = (
                    extra_weight * predictions[extra_name]
                    + (1.0 - extra_weight) * predictions[hgb_name]
                )
                result = score(y.loc[valid_mask], prediction)
                result.update(
                    hgb=hgb_name,
                    extra=extra_name,
                    extra_weight=float(extra_weight),
                )
                blends.append(result)
    blends.sort(key=lambda item: item["brier"])
    results["top_blends"] = blends[:20]
    Path("artifacts/v3_tuning_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"y": y.loc[valid_mask].to_numpy(), "predictions": predictions},
        "artifacts/v3_validation_predictions.joblib",
        compress=3,
    )
    print("TOP BLENDS", json.dumps(blends[:10], indent=2))


if __name__ == "__main__":
    main()
