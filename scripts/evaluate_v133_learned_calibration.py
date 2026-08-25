"""V133: upgrade the calibration layer from segment means to a learned residual model.

The calibration layer is a global shift plus two segment averages -- one over
(balls, strikes) at weight 0.75 with smoothing 500, one over (pitcher, balls, strikes)
at weight 0.25 with smoothing 300. V126 showed this layer doing heavy lifting: on the
2024 fold it took the level bias from +0.011804 down to +0.002361, and the drift term
then finished the job at +0.001024. A layer that demonstrably matters and has never
been anything more expressive than a lookup table is an obvious place to look.

The increment is stated additively so it cannot flatter itself. The existing
calibration is applied exactly as now, and a model is trained on what *remains* --
target minus fully calibrated prediction -- over the seasons before the fold, then
added back at a shrunk weight. At weight zero the candidate is bit-identical to the
baseline, so the ladder measures only what the model contributes beyond the lookups.

Two honest limitations, stated before the results.

2022 receives no calibration at all, because no earlier season exists to fit residuals
on, so its gain is exactly zero here by construction rather than by evidence. The
per-season condition adopted in V132 therefore binds only 2023 and 2024 for this class
of candidate, and the gate is correspondingly weaker than it looks. That is a property
of the pipeline, not a choice made for this experiment.

Residuals of a well-calibrated ensemble are mostly noise, so the models are regularised
far past what a first-stage learner would want: shallow trees, heavy L2, a low learning
rate. The failure mode to fear is a model that fits the training seasons' noise and
transfers a confident wrong correction, which is precisely what the shrinkage weight and
the chronological folds are there to catch.
"""

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v119_five_way_weight_refit import NAMES
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v133_learned_calibration_metrics.json")
PREDICTIONS = Path("artifacts/v133_learned_calibration_predictions.joblib")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"
CONFIGS = {
    "shallow": dict(iterations=600, depth=4, learning_rate=0.02, l2_leaf_reg=20.0),
    "medium": dict(iterations=900, depth=6, learning_rate=0.015, l2_leaf_reg=30.0),
}
RESIDUAL_WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def three_season(metrics):
    seasons = metrics["season_bootstrap"]
    means = [seasons[str(y)]["mean"] for y in YEARS]
    errors = [(seasons[str(y)]["ci95_high"] - seasons[str(y)]["ci95_low"]) / 3.9199
              for y in YEARS]
    average = sum(means) / 3.0
    combined = math.sqrt(sum(e * e for e in errors)) / 3.0
    return {"average_points": average * P, "se_points": combined * P,
            "ci95_low_points": (average - 1.96 * combined) * P,
            "season_points": [m * P for m in means]}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    raw = {year: sum(w * parts[n][str(year)] for w, n in zip(BASE, NAMES))
           for year in YEARS}

    # The fully calibrated prediction per fold, reproducing the deployed layer exactly.
    calibrated, target_of = {}, {}
    for year in YEARS:
        item = oof[str(year)]
        target_of[year] = item["target"].astype(float)
        if year == 2022:
            calibrated[year] = np.clip(raw[year], 0, 1)
            continue
        index, targets, predictions = [], [], []
        for history in [h for h in YEARS if h < year]:
            index.append(oof[str(history)]["row_index"])
            targets.append(oof[str(history)]["target"].astype(float))
            predictions.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(targets) - np.concatenate(predictions)
        train_frame = raw_frame.loc[index]
        valid_frame = raw_frame.loc[item["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        calibrated[year] = np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)

    categorical = [c for c, _ in EMBEDDING_SPECS if c in raw_frame.columns]
    numeric = [c for c in raw_frame.columns if c not in categorical]
    frame = raw_frame.copy()
    for name in categorical:
        frame[name] = frame[name].astype(str)
    frame[numeric] = frame[numeric].astype(np.float32)
    print(f"residual model frame: {frame.shape[1]} columns "
          f"({len(categorical)} categorical)", flush=True)

    corrections = {}
    for name, config in CONFIGS.items():
        corrections[name] = {}
        for year in YEARS:
            if year == 2022:
                corrections[name][str(year)] = np.zeros(len(target_of[year]))
                continue
            started = time.time()
            index, residual = [], []
            for history in [h for h in YEARS if h < year]:
                index.append(oof[str(history)]["row_index"])
                residual.append(target_of[history] - calibrated[history])
            index = np.concatenate(index)
            residual = np.concatenate(residual)
            model = CatBoostRegressor(**config, loss_function="RMSE", random_seed=42,
                                      verbose=0, thread_count=6,
                                      cat_features=categorical,
                                      allow_writing_files=False)
            model.fit(frame.loc[index], residual)
            prediction = model.predict(frame.loc[oof[str(year)]["row_index"]])
            corrections[name][str(year)] = prediction
            print(f"  {name} {year}: trained on {len(index):,} residuals "
                  f"(sd {residual.std():.4f}), correction sd {prediction.std():.5f}, "
                  f"mean {prediction.mean():+.6f}  [{time.time() - started:.0f}s]",
                  flush=True)

    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    baseline = np.clip(
        np.concatenate([calibrated[year] for year in YEARS]) + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for name in CONFIGS:
        for weight in RESIDUAL_WEIGHTS:
            candidate = np.clip(
                np.concatenate([calibrated[year] + weight * corrections[name][str(year)]
                                for year in YEARS]) + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate,
                (validation_frame["season"] == 2024).to_numpy())
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["three_season"] = three_season(metrics)
            metrics["residual_weight"] = weight
            metrics["config"] = CONFIGS[name]
            label = f"{name}_r{weight:.1f}"
            results[label] = metrics
            t = metrics["three_season"]
            print(f"  {label:14s} avg {t['average_points']:+7.2f}  "
                  f"CIlo {t['ci95_low_points']:+7.2f}  "
                  f"2022 {t['season_points'][0]:+6.2f}  "
                  f"2023 {t['season_points'][1]:+8.2f}  "
                  f"2024 {t['season_points'][2]:+7.2f}  "
                  f"blocks {metrics['monthly_block_win_rate']:5.0%}", flush=True)

    def passes(label):
        r = results[label]
        t = r["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and r["monthly_block_win_rate"] >= BLOCK_FLOOR)

    eligible = [l for l in results if passes(l)]
    promoted = max(eligible,
                   key=lambda l: results[l]["three_season"]["ci95_low_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V133_learned_calibration",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27), Public 1047.03653",
        "rationale": (
            "The calibration layer is a global shift plus two segment averages and has "
            "never been more expressive than a lookup. V126 showed it moving the 2024 "
            "level bias from +0.011804 to +0.002361, so it demonstrably matters."
        ),
        "construction": (
            "Additive increment: the deployed calibration is applied unchanged, a model "
            "is trained on what remains over prior seasons, and added at a shrunk "
            "weight. Weight zero is bit-identical to the baseline."
        ),
        "limitations": {
            "2022": ("Receives no calibration, so its gain is exactly zero by "
                     "construction. The per-season condition binds only 2023 and 2024 "
                     "for this class of candidate."),
            "regularisation": ("Residuals of a calibrated ensemble are mostly noise, so "
                               "the models are shallow with heavy L2; the fear is a "
                               "confident wrong correction transferring forward."),
        },
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}"},
        "configs": CONFIGS,
        "residual_weights": list(RESIDUAL_WEIGHTS),
        "results": {k: {"residual_weight": v["residual_weight"],
                        "three_season": v["three_season"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"corrections": corrections}, PREDICTIONS, compress=3)
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
