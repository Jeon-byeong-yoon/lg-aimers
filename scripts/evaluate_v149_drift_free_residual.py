"""V149: the residual model again, with the columns that made it fail removed.

Three findings now point at the same experiment.

V148 established that structure remains. A regressor given half a season and the blend's
own 96 features recovers 135-177 points of residual on 2024, 265-364 on 2022. The blend
is not at the frontier of what these features support; it is at the frontier of what
*transfers across seasons*.

V133 tried to collect it and collapsed. A residual model trained on prior seasons moved
the 2024 fold's level by -0.0188, wrecking a level that V126 had measured at +0.001, and
every weight from 0.2 upward made it worse. The diagnosis at the time was "it re-learned
league drift the global shift had already removed" -- correct but unactioned.

V142 named the culprit. Removing nine league-environment columns -- `season`, both win
expectancies, the leverage index, the running score and the score differentials -- moved
CatBoost's 2023 standalone from -844 to -519, and `no_season` alone accounted for almost
all of it. Those quantities are re-based every season, so a model leaning on them
extrapolates confidently and wrongly. The league rate fell 0.5647 to 0.4861 and the
per-season drop is not even uniform: -0.029 from 2022 to 2023 against -0.014 from 2023 to
2024.

A residual model is far more exposed to this than a first-stage model, because a residual
*is* mostly drift once the signal has been fitted. So this rebuilds V133 with those nine
columns withheld, and reports the correction's mean on each fold as the primary
diagnostic. If the mean collapses toward zero, the fix worked and the question becomes
whether the remaining structure transfers. If the mean stays large, the drift was never
in those columns and the whole residual-model idea is dead for a reason worth recording.

Weights start much lower than V133's -- 0.05 rather than 0.2 -- because V148 measures the
recoverable structure with same-season data and the transferable part must be smaller.

Pre-registered gate, unchanged since V132, ranked by the weakest season:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import gc
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
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v149_drift_free_residual_metrics.json")
PREDICTIONS = Path("artifacts/v149_drift_free_residual_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
# The nine columns V142 identified as re-based every season.
ENVIRONMENT = ("season", "home_win_expectancy", "away_win_expectancy", "li",
               "run_top_before", "run_bot_before", "run_total_before",
               "score_diff_home", "score_diff_pitcher_team")
CONFIG = dict(iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=10.0)
VARIANTS = ("with_environment", "without_environment")
WEIGHTS = (0.05, 0.10, 0.20, 0.35)
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    full = ([c for c in model_columns if c not in encoding_columns]
            + [c for c in categorical if c not in model_columns])
    frames = {}
    for variant in VARIANTS:
        keep = full if variant == "with_environment" else [
            c for c in full if c not in ENVIRONMENT]
        work = features[keep].copy()
        cats = [c for c in categorical if c in keep]
        for column in cats:
            work[column] = work[column].astype(str)
        numeric = [c for c in keep if c not in cats]
        work[numeric] = work[numeric].astype(np.float32)
        frames[variant] = (work, cats)
        print(f"  {variant}: {work.shape[1]} columns", flush=True)
    season = raw_frame["season"].to_numpy()
    del features, hierarchical, encoded, form_raw, trackman, data
    gc.collect()

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
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    weights = tuple(BASE[n] for n in NAMES6)
    raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
           for year in YEARS}

    # The deployed calibration, reproduced so the residual model sees what it will correct.
    calibrated, targets_of = {}, {}
    for year in YEARS:
        item = oof[str(year)]
        targets_of[year] = item["target"].astype(float)
        if year == 2022:
            calibrated[year] = np.clip(raw[year], 0, 1)
            continue
        history = [h for h in YEARS if h < year]
        index = np.concatenate([oof[str(h)]["row_index"] for h in history])
        past_target = np.concatenate([targets_of[h] for h in history])
        residual = past_target - np.concatenate([raw[h] for h in history])
        count = segment_correction(
            raw_frame.loc[index], residual, raw_frame.loc[item["row_index"]],
            ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            raw_frame.loc[index], residual, raw_frame.loc[item["row_index"]],
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        calibrated[year] = np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)

    corrections = {}
    print("\ncorrection means -- the primary diagnostic (V133 reached -0.018835)",
          flush=True)
    for variant in VARIANTS:
        work, cats = frames[variant]
        corrections[variant] = {}
        for year in YEARS:
            if year == 2022:
                corrections[variant][str(year)] = np.zeros(len(targets_of[year]))
                continue
            started = time.time()
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate([targets_of[h] - calibrated[h] for h in history])
            model = CatBoostRegressor(**CONFIG, loss_function="RMSE", random_seed=42,
                                      verbose=0, thread_count=6, cat_features=cats,
                                      allow_writing_files=False)
            model.fit(work.loc[index], residual)
            train_mean = float(model.predict(work.loc[index]).mean())
            predicted = model.predict(work.loc[oof[str(year)]["row_index"]])
            corrections[variant][str(year)] = predicted
            del model
            gc.collect()
            print(f"  {variant:20s} {year}: mean {predicted.mean():+.6f}  "
                  f"sd {predicted.std():.5f}  (train mean {train_mean:+.6f})  "
                  f"[{time.time() - started:.0f}s]", flush=True)

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

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    def passes(m):
        t = m["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and m["monthly_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"\n{'candidate':>30} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    for variant in VARIANTS:
        for weight in WEIGHTS:
            candidate = np.clip(
                np.concatenate([calibrated[year]
                                + weight * corrections[variant][str(year)]
                                for year in YEARS]) + DRIFT_WEIGHT * term, 0, 1)
            label = f"{variant}_w{weight:.2f}"
            results[label] = evaluate(candidate)
            m = results[label]; t = m["three_season"]
            print(f"{label:>30} {m['min_season_points']:7.2f} "
                  f"{t['average_points']:7.2f} {t['season_points'][0]:7.2f} "
                  f"{t['season_points'][1]:8.2f} {t['season_points'][2]:7.2f} "
                  f"{m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V149_drift_free_residual",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "synthesis": {
            "V148": ("135-177 points of residual structure on 2024 and 265-364 on 2022 "
                     "are recoverable by a regressor given same-season data, so the "
                     "blend is at the frontier of what transfers, not of what the "
                     "features support."),
            "V133": ("a residual model on prior seasons moved the 2024 level by "
                     "-0.018835 and collapsed; the diagnosis was drift re-learning."),
            "V142": ("removing nine league-environment columns moved CatBoost's 2023 "
                     "standalone from -844 to -519, and `season` alone accounted for "
                     "almost all of it."),
        },
        "environment_columns": list(ENVIRONMENT),
        "primary_diagnostic": "the correction's mean per fold; V133 reached -0.018835",
        "config": CONFIG,
        "weights": list(WEIGHTS),
        "correction_summary": {
            variant: {year: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                      for year, v in values.items()}
            for variant, values in corrections.items()},
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
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
