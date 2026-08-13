"""Train the two-model ensemble used by the evaluation-server submission.

Validation is strictly chronological: seasons 2019--2023 train, 2024 validates.
The final bundle is then refit on every official training row (2019--2024).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder


ID_COL = "row_id"
TARGET_COL = "control_success"
VALID_SEASON = 2024


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = list(frame.select_dtypes(exclude=np.number).columns)
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
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


def make_models(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "extra_trees": make_pipeline(
            make_preprocessor(frame),
            ExtraTreesClassifier(
                n_estimators=150,
                max_depth=14,
                min_samples_leaf=100,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        "hist_gbdt": make_pipeline(
            make_preprocessor(frame),
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=3.0,
                random_state=42,
            ),
        ),
    }


def bss(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    brier = brier_score_loss(y_true, prediction)
    rate = float(y_true.mean())
    reference = rate * (1.0 - rate)
    return {
        "brier": float(brier),
        "bss": float(max(0.0, 100000.0 * (1.0 - brier / reference))),
        "prediction_mean": float(prediction.mean()),
        "target_mean": rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv", default="공모전 dataset/open/data/train.csv"
    )
    parser.add_argument("--output", default="artifacts/ensemble.joblib")
    parser.add_argument("--metrics", default="artifacts/validation_metrics.json")
    args = parser.parse_args()

    data = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    y = data.pop(TARGET_COL).astype("uint8")
    row_ids = data.pop(ID_COL)
    if row_ids.duplicated().any():
        raise ValueError("Training row_id contains duplicates")

    train_mask = data["season"] < VALID_SEASON
    valid_mask = data["season"] == VALID_SEASON
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Expected pre-2024 training rows and 2024 validation rows")

    started = perf_counter()
    validation_models = make_models(data)
    validation_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {
        "split": "season < 2024 -> season == 2024",
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "models": {},
    }
    for name, model in validation_models.items():
        model.fit(data.loc[train_mask], y.loc[train_mask])
        prediction = model.predict_proba(data.loc[valid_mask])[:, 1]
        validation_predictions[name] = prediction
        metrics["models"][name] = bss(y.loc[valid_mask], prediction)

    stable_prediction = (
        0.70 * validation_predictions["extra_trees"]
        + 0.30 * validation_predictions["hist_gbdt"]
    )
    metrics["stable_blend"] = bss(y.loc[valid_mask], stable_prediction)

    # This affine fit is saved as an optional, higher-variance submission variant.
    design = np.column_stack(
        [
            np.ones(valid_mask.sum()),
            validation_predictions["extra_trees"],
            validation_predictions["hist_gbdt"],
        ]
    )
    coefficients = np.linalg.lstsq(design, y.loc[valid_mask], rcond=None)[0]
    calibrated_prediction = np.clip(design @ coefficients, 0.0, 1.0)
    metrics["affine_blend"] = bss(y.loc[valid_mask], calibrated_prediction)
    metrics["affine_coefficients"] = coefficients.tolist()

    final_models = make_models(data)
    for model in final_models.values():
        model.fit(data, y)

    bundle = {
        "models": final_models,
        "stable_weights": {"extra_trees": 0.70, "hist_gbdt": 0.30},
        "affine_coefficients": coefficients,
        "feature_columns": list(data.columns),
        "training_seasons": sorted(map(int, data["season"].unique())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    metrics["elapsed_seconds"] = perf_counter() - started
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model bundle: {output}")


if __name__ == "__main__":
    main()
