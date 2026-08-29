"""V202: the factorization network's shape was never measured, only guessed.

V200 built an instrument with error bars -- pair at the same seed, five seeds, and the 2024
reading's noise falls from sd 8.67 to 1.07 -- and immediately used it to show that the
reliability-scale decision had been made on 0.13 standard deviations. The same instrument
now goes to the next constant chosen the same way.

The factorization network ships `LATENT = 8`, `epochs = 8`, `hidden = (128, 64)`. Its module
default is `LATENT = 24`; V130 moved it to 8 and V140 tried a "high latent slot", both read
on single draws. `epochs = 8` has never been varied at all -- the file says the choices are
"same operational choices as V111, for the same reasons".

This is the cheapest re-ask available, because **the feature frame does not change**. Only
the factorization network's own hyperparameters move, so one frame build serves every arm,
and the embedding network, Form, Context and CatBoost are all held at what V199 ships. Six
arms:

    latent   4, 16, 24        at the shipped 8 epochs
    epochs   5, 12            at the shipped latent 8
    wide     latent 16 + hidden (256, 128)

Paired at the same seed against the shipped configuration, five seeds, with the measured
null carried over from V200 (sd 8.67 on 2024) so the readings can be compared to the noise
they replace.
"""

import gc
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import interaction_network_v130 as inet
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, encode_categorical,
    encode_numeric, numeric_statistics,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
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


OUTPUT = Path("artifacts/v202_factorization_shape_metrics.json")
PREDICTIONS = Path("artifacts/v202_factorization_shape_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SHIPPED = {"latent": 8, "epochs": 8, "hidden": (128, 64)}
ARMS = {
    "latent4": {"latent": 4},
    "latent16": {"latent": 16},
    "latent24": {"latent": 24},
    "epochs5": {"epochs": 5},
    "epochs12": {"epochs": 12},
    "latent16_wide": {"latent": 16, "hidden": (256, 128)},
}
PAIRED_SEEDS = (42, 1004, 2024, 777, 999)
NULL_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
SHIPPED_NETWORK_SEEDS = NULL_SEEDS
SHIPPED_FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_CONTEXT_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_CATBOOST_SEEDS = (42, 1004, 2024, 777)
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    frame = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical, encoded, trackman
    gc.collect()
    frame = add_trackman_features(frame, prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)))
    frame = pd.concat([frame, block], axis=1)
    keep = v31_form_columns(frame)
    assert len(keep) == 105, len(keep)
    categorical_columns = [c for c, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in keep
                       if c not in categorical_columns and c != "season"]
    identity = frame.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)
    del frame, block
    gc.collect()
    print(f"one frame serves every arm: {len(keep)} columns", flush=True)

    arms = {name: {s: {} for s in PAIRED_SEEDS} for name in ARMS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        train_identity = identity.loc[train_mask]
        valid_identity = identity.loc[valid_mask]
        vocabularies = build_vocabularies(train_identity)
        statistics = numeric_statistics(
            train_identity[numeric_columns].to_numpy(dtype=np.float64))
        categorical = encode_categorical(train_identity, vocabularies)
        numeric = encode_numeric(
            train_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        valid_categorical = encode_categorical(valid_identity, vocabularies)
        valid_numeric = encode_numeric(
            valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        del train_identity, valid_identity
        gc.collect()
        for name, override in ARMS.items():
            config = {**SHIPPED, **override}
            inet.LATENT = config["latent"]
            inet.FIELD_SPECS = [(c, config["latent"]) for c in inet.FIELDS]
            spec = inet.field_cardinalities(vocabularies)
            for seed in PAIRED_SEEDS:
                model = inet.train(categorical, numeric, targets[train_mask], spec,
                                   epochs=config["epochs"], seed=seed,
                                   hidden=tuple(config["hidden"]), verbose=False)
                arms[name][seed][str(year)] = inet.predict(
                    model, valid_categorical, valid_numeric)
                del model
                gc.collect()
            print(f"  {year} {name:14s} done [{time.time() - started:.0f}s]", flush=True)
        del categorical, numeric, valid_categorical, valid_numeric
        gc.collect()
    del identity
    gc.collect()
    joblib.dump({"arms": arms}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    v197 = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")
    incumbent_fm = {**v164["factorizations"], **v194["factorizations"]}
    network_store = {**v164["networks"], **v190["networks"]}
    form_store = {**v190["forms"], **v192["forms"]}

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": mean_of(form_store, SHIPPED_FORM_SEEDS),
        "network": mean_of(network_store, SHIPPED_NETWORK_SEEDS),
        "context": mean_of(v195["contexts"], SHIPPED_CONTEXT_SEEDS),
        "catboost": mean_of(v197["catboosts"], SHIPPED_CATBOOST_SEEDS),
    }
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    calibration_frame["two_strike"] = (
        calibration_frame["strikes_before"] == 2).astype("int64")
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y_: calibration_frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y_])]
        for y_ in YEARS if y_ != 2022}
    valid_frames = {y_: calibration_frame.loc[oof[str(y_)]["row_index"]] for y_ in YEARS}
    scale_sum = 1.0 / sum(w for _, w, _ in TERMS)

    def errors(factorization):
        parts = dict(fixed)
        parts["factorization"] = factorization
        blend = {y_: sum(BASE[n] * parts[n][str(y_)] for n in NAMES6) for y_ in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for cols, weight, smoothing in TERMS:
                value = value + scale_sum * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        prediction = np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)
        return (prediction - target) ** 2

    def points(candidate_error, base_error):
        return [float(P * (base_error[masks[y_]].mean()
                           - candidate_error[masks[y_]].mean())) for y_ in YEARS]

    incumbent_errors = {s: errors({str(y_): incumbent_fm[s][str(y_)] for y_ in YEARS})
                        for s in NULL_SEEDS}
    null = np.array([points(incumbent_errors[a], incumbent_errors[b])
                     for a, b in combinations(NULL_SEEDS, 2)])
    print(f"\nmeasured null -- two shipped-configuration draws ({len(null)} pairs):",
          flush=True)
    print(f"  sd    {null.std(axis=0, ddof=1)[0]:7.2f} "
          f"{null.std(axis=0, ddof=1)[1]:7.2f} {null.std(axis=0, ddof=1)[2]:7.2f}",
          flush=True)

    print(f"\npaired against latent {SHIPPED['latent']} / epochs {SHIPPED['epochs']} "
          f"at the same seed:", flush=True)
    print(f"  {'arm':14s} {'2022':>17s} {'2023':>17s} {'2024':>17s}", flush=True)
    results = {}
    for name in ARMS:
        rows = [points(errors(arms[name][s]), incumbent_errors[s]) for s in PAIRED_SEEDS]
        block_ = np.array(rows)
        mean = block_.mean(axis=0)
        se = block_.std(axis=0, ddof=1) / np.sqrt(len(PAIRED_SEEDS))
        results[name] = {"config": {**SHIPPED, **ARMS[name]},
                         "per_seed": {str(s): r for s, r in zip(PAIRED_SEEDS, rows)},
                         "paired_mean": mean.tolist(), "paired_se": se.tolist(),
                         "all_positive": bool((mean > 0).all())}
        results[name]["config"]["hidden"] = list(results[name]["config"]["hidden"])
        print(f"  {name:14s} " + "  ".join(
            f"{mean[i]:+7.2f}+-{se[i]:5.2f}" for i in range(3))
            + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    survivors = [n for n in results if results[n]["all_positive"]]
    print(f"\narms positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V202_factorization_shape",
        "baseline": "V199 (Public 1083.2461959655)",
        "why": ("LATENT 8, epochs 8 and hidden (128, 64) were chosen on single-draw "
                "readings; the module default is LATENT 24, V130 moved it to 8, and epochs "
                "has never been varied. V200's instrument makes a one-point effect "
                "measurable for the first time."),
        "why_cheapest": ("the feature frame does not change, so one build serves every arm "
                         "and only the factorization network is retrained"),
        "shipped": {**SHIPPED, "hidden": list(SHIPPED["hidden"])},
        "arms": {k: {**v, "hidden": list(v.get("hidden", SHIPPED["hidden"]))}
                 for k, v in ARMS.items()},
        "paired_seeds": list(PAIRED_SEEDS),
        "measured_null": {"pairs": len(null), "sd": null.std(axis=0, ddof=1).tolist()},
        "results": results,
        "survivors": survivors,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
