"""Evaluate a strongly regularized nonlinear residual corrector for V13."""

import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline

from evaluate_residual_ridge_v13 import FEATURES, prepare_fold


PARAMETERS = list(product([4, 8], [1500, 3000], [10.0, 50.0]))
STRENGTHS = [0.1, 0.2, 0.3, 0.5, 0.75]


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    folds = {
        2023: prepare_fold(oof, frame, [2022], 2023),
        2024: prepare_fold(oof, frame, [2022, 2023], 2024),
    }
    baselines = {
        year: float(brier_score_loss(values[3], values[4]))
        for year, values in folds.items()
    }
    accepted = []
    all_results = []
    for leaves, min_leaf, l2 in PARAMETERS:
        corrections = {}
        for year, (train_x, train_y, valid_x, _, _) in folds.items():
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=75,
                    max_leaf_nodes=leaves,
                    min_samples_leaf=min_leaf,
                    l2_regularization=l2,
                    random_state=20260816,
                ),
            )
            model.fit(train_x, train_y)
            corrections[year] = model.predict(valid_x)
        for strength in STRENGTHS:
            scores = {}
            for year, (_, _, _, valid_y, baseline) in folds.items():
                scores[year] = float(brier_score_loss(
                    valid_y, np.clip(baseline + strength * corrections[year], 0, 1)
                ))
            item = {
                "leaves": leaves, "min_samples_leaf": min_leaf, "l2": l2,
                "strength": strength,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
                "2023_gain": baselines[2023] - scores[2023],
                "2024_gain": baselines[2024] - scores[2024],
            }
            all_results.append(item)
            if item["2023_gain"] > 0 and item["2024_gain"] > 0:
                accepted.append(item)
    ranking = lambda x: (min(x["2023_gain"], x["2024_gain"]),
                         x["2023_gain"] + x["2024_gain"])
    accepted.sort(key=ranking, reverse=True)
    all_results.sort(key=ranking, reverse=True)
    output = {
        "baseline": {str(year): score for year, score in baselines.items()},
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_accepted": accepted[:10],
        "top5_overall": all_results[:5],
        "features": FEATURES,
    }
    Path("artifacts/v13_residual_hgb_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
