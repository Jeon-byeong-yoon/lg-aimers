"""V194: re-measure the factorization network's lottery, because V188's reading was biased.

V188 rejected averaging the factorization network on this row:

    fact3   vs seed42   +0.28  +0.80  -1.67      expected  -2.67  -8.31  +10.02

and the expected column is where the rejection came from -- negative 2022 and 2023. But
V191 then showed exactly how that estimator misleads: the averaged set contained seed 42,
and every baseline it was scored against included seed 42 as well, so the same draw sat on
both sides of the comparison. Correcting that turned the network's curve from *falling* in
k to rising monotonically, which is the shape the mechanism predicts.

So the factorization network gets the corrected estimator: the inside set pooled over
random subsets, each average scored only against the seeds it excludes. Four more draws are
trained so the curve reaches k=8 rather than k=4.

The baseline is V193, not V189 -- the network is held at its nine-draw average and Form at
its six-draw average, because a curve on one component should hold the others at what is
deployed.

The factorization network carries 0.07, the smallest weight in the blend, so the honest
expectation is a small number. It is measured because the rejection that stands against it
was produced by an estimator now known to read backwards.
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


OUTPUT = Path("artifacts/v194_factorization_lottery_metrics.json")
PREDICTIONS = Path("artifacts/v194_factorization_lottery_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
FACTORIZATION_LATENT, FACTORIZATION_EPOCHS = 8, 8
FACTORIZATION_HIDDEN = (128, 64)
CACHED_SEEDS = (42, 1004, 2024, 777, 999)
NEW_SEEDS = (13, 314, 2718, 65537)
SHIPPED_NETWORK_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
SHIPPED_FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)
MAX_SUBSETS = 20
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
    del hierarchical, encoded
    gc.collect()
    frame = add_trackman_features(frame, trackman)
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
    del frame
    gc.collect()
    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]

    fresh = {s: {} for s in NEW_SEEDS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        train_identity, valid_identity = identity.loc[train_mask], identity.loc[valid_mask]
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
        actual = targets[valid_mask].astype(float)
        rate = actual.mean()
        for seed in NEW_SEEDS:
            model = inet.train(categorical, numeric, targets[train_mask],
                               inet.field_cardinalities(vocabularies),
                               epochs=FACTORIZATION_EPOCHS, seed=seed,
                               hidden=FACTORIZATION_HIDDEN, verbose=False)
            fresh[seed][str(year)] = inet.predict(model, valid_categorical, valid_numeric)
            del model
            gc.collect()
        skills = [100000 * (1 - ((fresh[s][str(year)] - actual) ** 2).mean()
                            / (rate * (1 - rate))) for s in NEW_SEEDS]
        print(f"  {year} new FM seeds: " + "  ".join(f"{v:7.0f}" for v in skills)
              + f"   [{time.time() - started:.0f}s]", flush=True)
        del categorical, numeric, valid_categorical, valid_numeric
        gc.collect()
    del identity
    gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    factorizations = dict(v164["factorizations"])
    factorizations.update(fresh)
    seeds = CACHED_SEEDS + NEW_SEEDS
    network_store = dict(v164["networks"])
    network_store.update(v190["networks"])
    form_store = dict(v190["forms"])
    form_store.update(v192["forms"])

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        # Held at what V193 deploys, so the curve is read on the shipped configuration.
        "form": mean_of(form_store, SHIPPED_FORM_SEEDS),
        "network": mean_of(network_store, SHIPPED_NETWORK_SEEDS),
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
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
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(factorization):
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
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    base_error = {s: (pipeline(factorizations[s]) - target) ** 2 for s in seeds}
    lottery = np.array([points(pipeline(factorizations[s]), base_error[seeds[0]])
                        for s in seeds])
    print(f"\nthe factorization network's lottery, every seed against seed 42:", flush=True)
    print(f"  sd    {lottery.std(axis=0, ddof=1)[0]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[1]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[2]:7.2f}", flush=True)
    print(f"  range {lottery[:, 0].ptp():7.2f} {lottery[:, 1].ptp():7.2f} "
          f"{lottery[:, 2].ptp():7.2f}   "
          f"(network 17.94, Form 2.97 on 2024)", flush=True)

    generator = np.random.default_rng(20260829)
    print(f"\naverage of k against the seeds it excludes, pooled over which go inside:",
          flush=True)
    print(f"  {'k':>3s} {'subsets':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    table = {}
    for k in range(1, len(seeds)):
        every = list(combinations(seeds, k))
        picked = ([every[i] for i in generator.choice(len(every), MAX_SUBSETS,
                                                      replace=False)]
                  if len(every) > MAX_SUBSETS else every)
        readings = []
        for inside in picked:
            avg = (factorizations[inside[0]] if k == 1
                   else mean_of(factorizations, inside))
            candidate = pipeline(avg)
            for s in seeds:
                if s in inside:
                    continue
                readings.append(points(candidate, base_error[s]))
        block_ = np.array(readings)
        mean = block_.mean(axis=0)
        table[k] = {"subsets": len(picked), "readings": len(readings),
                    "expected": mean.tolist(),
                    "sd": block_.std(axis=0, ddof=1).tolist()}
        print(f"  {k:3d} {len(picked):8d} {mean[0]:+8.2f} {mean[1]:+8.2f} {mean[2]:+8.2f}"
              + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    best = max(table, key=lambda k: table[k]["expected"][2])
    print(f"\nbest k by expected 2024 = {best} "
          f"({table[best]['expected'][2]:+.2f}); "
          f"V188 rejected this component on a biased estimate of -2.67 / -8.31 / +10.02",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V194_factorization_lottery",
        "baseline": "V193 (Public 1076.9782849555)",
        "why": ("V188 rejected averaging this component on an expected column of -2.67 / "
                "-8.31 / +10.02, produced by an estimator V191 showed to read backwards: "
                "the averaged set contained seed 42 and so did every baseline"),
        "seeds": list(seeds),
        "lottery_sd": lottery.std(axis=0, ddof=1).tolist(),
        "lottery_range": [float(lottery[:, i].ptp()) for i in range(3)],
        "k_table": table,
        "best_k_by_2024": best,
        "held_at": {"network": list(SHIPPED_NETWORK_SEEDS),
                    "form": list(SHIPPED_FORM_SEEDS)},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"factorizations": fresh}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
