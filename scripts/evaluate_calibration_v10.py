"""Compare expanding OOF calibration methods without evaluation-data statistics."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def logit(prediction):
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def metric(target, prediction):
    brier = brier_score_loss(target, prediction)
    rate = float(target.mean())
    return {
        "brier": float(brier),
        "bss": float(max(0, 100000 * (1 - brier / (rate * (1 - rate))))),
        "mean": float(prediction.mean()),
    }


def fit_apply(method, train_y, train_p, valid_p):
    if method == "additive":
        shift = float(train_y.mean() - train_p.mean())
        return np.clip(valid_p + shift, 0, 1), {"shift": shift}
    if method == "affine":
        design = np.column_stack([np.ones(len(train_p)), train_p])
        intercept, slope = np.linalg.lstsq(design, train_y, rcond=None)[0]
        return np.clip(intercept + slope * valid_p, 0, 1), {
            "intercept": float(intercept), "slope": float(slope)
        }
    if method == "platt":
        model = LogisticRegression(C=1e6, max_iter=200)
        model.fit(logit(train_p).reshape(-1, 1), train_y)
        prediction = model.predict_proba(logit(valid_p).reshape(-1, 1))[:, 1]
        return prediction, {
            "intercept": float(model.intercept_[0]),
            "slope": float(model.coef_[0, 0]),
        }
    if method == "isotonic":
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(train_p, train_y)
        return model.predict(valid_p), {"threshold_count": len(model.X_thresholds_)}
    raise ValueError(method)


def main() -> None:
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    methods = ["additive", "affine", "platt", "isotonic"]
    results = {}
    history_y, history_p = [], []
    for year in (2022, 2023, 2024):
        target = oof[str(year)]["target"]
        prediction = oof[str(year)]["prediction"]
        year_result = {"raw": metric(target, prediction)}
        if history_y:
            train_y = np.concatenate(history_y)
            train_p = np.concatenate(history_p)
            for method in methods:
                calibrated, parameters = fit_apply(
                    method, train_y, train_p, prediction
                )
                year_result[method] = metric(target, calibrated)
                year_result[method]["parameters"] = parameters
        results[str(year)] = year_result
        history_y.append(target)
        history_p.append(prediction)
        print(year, year_result, flush=True)

    all_y = np.concatenate(history_y)
    all_p = np.concatenate(history_p)
    final_parameters = {}
    probe = np.array([0.25, 0.5, 0.75])
    for method in methods:
        _, parameters = fit_apply(method, all_y, all_p, probe)
        final_parameters[method] = parameters
    results["parameters_for_2025"] = final_parameters
    Path("artifacts/v10_calibration_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
