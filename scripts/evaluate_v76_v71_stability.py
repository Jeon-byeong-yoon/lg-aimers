"""V76: V71 Form LightGBM의 다중 시드·제한 비중 안정성 검증."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v71_form_lgbm_diversity import get_form_X


BASELINE = {2022: 0.24338879436541092, 2023: 0.25315389703549307, 2024: 0.24782554051940017}
PARAMS = {
    "num_leaves": 15,
    "min_child_samples": 100,
    "reg_lambda": 10.0,
    "learning_rate": 0.05,
    "n_estimators": 300,
}
SEEDS = (17, 42, 73)
WEIGHTS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075)
YEARS = (2022, 2023, 2024)


def form_predictions(frames, targets, seed):
    x22 = get_form_X(frames["2022"])
    x23 = get_form_X(frames["2023"])
    x24 = get_form_X(frames["2024"])
    y22, y23 = targets["2022"], targets["2023"]
    pred22 = np.zeros(len(y22), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in cv.split(x22, y22):
        model = LGBMClassifier(**PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
        model.fit(x22.iloc[train_idx], y22.iloc[train_idx])
        pred22[valid_idx] = model.predict_proba(x22.iloc[valid_idx])[:, 1]

    model23 = LGBMClassifier(**PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
    model23.fit(x22, y22)
    pred23 = model23.predict_proba(x23)[:, 1]

    train_2223 = pd.concat([frames["2022"], frames["2023"]], ignore_index=True)
    y_2223 = pd.concat([y22, y23], ignore_index=True)
    model24 = LGBMClassifier(**PARAMS, random_state=seed, verbose=-1, n_jobs=-1)
    model24.fit(get_form_X(train_2223), y_2223)
    pred24 = model24.predict_proba(x24)[:, 1]
    return {"2022": pred22, "2023": pred23, "2024": pred24}


def score_candidate(name, weight, lgbm, frames, targets, v17, form_hgb, context):
    raw = {}
    for year in YEARS:
        key = str(year)
        form = (1.0 - weight) * form_hgb[key] + weight * lgbm[key]
        raw[key] = 0.55 * v17[key] + 0.32 * form + 0.13 * context[key]

    residual22 = np.asarray(targets["2022"]) - raw["2022"]
    train24 = pd.concat([frames["2022"], frames["2023"]], ignore_index=True)
    residual_2223 = np.concatenate([
        residual22,
        np.asarray(targets["2023"]) - raw["2023"],
    ])
    predictions = {"2022": np.clip(raw["2022"], 0, 1)}
    for year, train_frame, residual in (
        (2023, frames["2022"], residual22),
        (2024, train24, residual_2223),
    ):
        key = str(year)
        count = segment_correction(train_frame, residual, frames[key], ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, frames[key], ["pitcher_id", "balls_before", "strikes_before"], 300
        )
        predictions[key] = np.clip(raw[key] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)

    scores = {
        str(year): float(brier_score_loss(targets[str(year)], predictions[str(year)]))
        for year in YEARS
    }
    gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
    return {
        "candidate": name,
        "weight": weight,
        "scores": scores,
        "gains_vs_v41": gains,
        "all_improved": all(gains[str(year)] > 0 for year in YEARS),
        "submit_ready": all(gains[str(year)] > 0 for year in YEARS) and gains["2024"] >= 5e-6,
    }


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y_all = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_hgb = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]

    frames = {str(y): raw_frame.loc[oof[str(y)]["row_index"]].reset_index(drop=True) for y in YEARS}
    targets = {str(y): oof[str(y)]["target"].astype(float) for y in YEARS}
    labels = {str(y): y_all.loc[oof[str(y)]["row_index"]].reset_index(drop=True) for y in YEARS}
    v11 = {str(y): v11_prediction(oof[str(y)]) for y in YEARS}
    v17 = {str(y): 0.95 * v11[str(y)] + 0.05 * logistic[str(y)] for y in YEARS}

    seed_predictions = {}
    results = []
    for seed in SEEDS:
        print(f"Training seed {seed}...", flush=True)
        seed_predictions[seed] = form_predictions(frames, labels, seed)
        for weight in WEIGHTS:
            results.append(score_candidate(f"seed_{seed}", weight, seed_predictions[seed], frames, targets, v17, form_hgb, context))

    averaged = {
        str(year): np.mean([seed_predictions[s][str(year)] for s in SEEDS], axis=0)
        for year in YEARS
    }
    for weight in WEIGHTS:
        results.append(score_candidate("seed_mean", weight, averaged, frames, targets, v17, form_hgb, context))

    results.sort(key=lambda x: (x["gains_vs_v41"]["2024"], x["gains_vs_v41"]["2023"]), reverse=True)
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V76_v71_stability",
        "baseline": {str(k): v for k, v in BASELINE.items()},
        "params": PARAMS,
        "seeds": list(SEEDS),
        "weights": list(WEIGHTS),
        "candidate_count": len(results),
        "submit_ready_count": len(ready),
        "best": results[0],
        "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v76_v71_stability_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best", "best_submit_ready")}, indent=2))


if __name__ == "__main__":
    main()
