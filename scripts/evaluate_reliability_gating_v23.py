"""Leakage-safe pitcher-history reliability gated ensemble for V23."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction


COMPONENTS = [
    "extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb", "logistic"
]
BASE_WEIGHTS = np.array([
    0.95 * 0.29483562599237795,
    0.95 * 0.2344456574665245,
    0.95 * 0.11617615437521091,
    0.95 * 0.35454256216588664,
    0.05,
])
BIN_EDGES = [-np.inf, 50, 200, 1000, np.inf]
REGULARIZATIONS = [0.0001, 0.0003, 0.001, 0.003, 0.01]


def component_matrix(oof_item, logistic_prediction):
    tree = oof_item["components"]
    return np.column_stack([
        tree["extra_trees"], tree["trackman_hgb"], tree["te_trackman_hgb"],
        tree["hierarchical_hgb"], logistic_prediction,
    ])


def fit_weights(matrix, target, regularization):
    def objective(weights):
        error = target - matrix @ weights
        return np.mean(error**2) + regularization * np.sum((weights - BASE_WEIGHTS) ** 2)
    result = minimize(
        objective, BASE_WEIGHTS, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(BASE_WEIGHTS),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def gated_prediction(train_matrix, train_y, train_n, valid_matrix, valid_n, regularization):
    prediction = np.empty(len(valid_matrix), dtype=float)
    learned = []
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        train_mask = (train_n >= low) & (train_n < high)
        valid_mask = (valid_n >= low) & (valid_n < high)
        if train_mask.sum() < 1000:
            weights = BASE_WEIGHTS.copy()
        else:
            weights = fit_weights(
                train_matrix[train_mask], train_y[train_mask], regularization
            )
        prediction[valid_mask] = valid_matrix[valid_mask] @ weights
        learned.append({
            "low": None if np.isneginf(low) else float(low),
            "high": None if np.isposinf(high) else float(high),
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "weights": dict(zip(COMPONENTS, weights.tolist())),
        })
    return prediction, learned


def evaluate_fold(oof, logistic, frame, history_years, valid_year, regularization):
    history = [oof[str(year)] for year in history_years]
    train_matrix = np.vstack([
        component_matrix(item, logistic[str(year)])
        for year, item in zip(history_years, history)
    ])
    train_y = np.concatenate([item["target"] for item in history]).astype(float)
    train_index = np.concatenate([item["row_index"] for item in history])
    valid = oof[str(valid_year)]
    valid_matrix = component_matrix(valid, logistic[str(valid_year)])
    valid_y = valid["target"].astype(float)
    valid_index = valid["row_index"]
    train_frame, valid_frame = frame.loc[train_index], frame.loc[valid_index]
    train_n = train_frame["asof_pitcher_n"].fillna(0).to_numpy()
    valid_n = valid_frame["asof_pitcher_n"].fillna(0).to_numpy()
    train_raw, _ = gated_prediction(
        train_matrix, train_y, train_n, train_matrix, train_n, regularization
    )
    valid_raw, learned = gated_prediction(
        train_matrix, train_y, train_n, valid_matrix, valid_n, regularization
    )
    residual = train_y - train_raw
    count = segment_correction(
        train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
    )
    pitcher_count = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(
        valid_raw + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1
    )
    return float(brier_score_loss(valid_y, prediction)), learned


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    baseline = {"2023": 0.2533578938768331, "2024": 0.24785045715137144}
    accepted, all_results = [], []
    for regularization in REGULARIZATIONS:
        score23, weights23 = evaluate_fold(
            oof, logistic, frame, [2022], 2023, regularization
        )
        score24, weights24 = evaluate_fold(
            oof, logistic, frame, [2022, 2023], 2024, regularization
        )
        item = {
            "regularization": regularization,
            "2023_brier": score23, "2024_brier": score24,
            "2023_gain": baseline["2023"] - score23,
            "2024_gain": baseline["2024"] - score24,
            "weights_2023": weights23, "weights_2024": weights24,
        }
        all_results.append(item)
        if item["2023_gain"] > 0 and item["2024_gain"] > 0:
            accepted.append(item)
        print(regularization, score23, score24, flush=True)
    rank = lambda x: (min(x["2023_gain"], x["2024_gain"]),
                      x["2023_gain"] + x["2024_gain"])
    accepted.sort(key=rank, reverse=True)
    all_results.sort(key=rank, reverse=True)
    output = {
        "baseline_v17": baseline,
        "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "all": all_results,
    }
    Path("artifacts/v23_reliability_gating_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"accepted_count": len(accepted), "best": output["best"]}, indent=2))


if __name__ == "__main__":
    main()
