"""V206: structure diversity -- does averaging across depths pay like averaging seeds did?

The lottery axis paid +15.38 by averaging draws of the SAME structure, and V199 closed it
with every component's k-curve saturated. But V203 measured that a structural change
(depth, leaf counts) decorrelates two fits even at the same seed -- which is exactly what
makes a draw worth averaging. Depth-5 and depth-7 CatBoost fits should be less correlated
with the shipped depth-6 average than a fifth depth-6 seed would be, so the average of
mixed structures could sit past the saturated part of the seed curve.

This is the one mechanism in the record that produced double-digit gains and still has an
untested variant. The readout mirrors V197: candidate averages are built from cached
depth-6 fits plus the new arms, and every combination is read against the shipped 4-seed
depth-6 average through the full blend + calibration pipeline.

Held: iterations 1200, learning rate 0.02, l2_leaf_reg 12 (the shipped config except
depth). If V204 adopts a different iteration count first, rerun with that count.
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
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v153_projected_prior import CATBOOST_CONFIG, FEATURE_SHRINKAGE, build_block
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v206_catboost_depth_diversity_metrics.json")
PREDICTIONS = Path("artifacts/v206_catboost_depth_diversity_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
INCUMBENT_DEPTH = 6
NEW_DEPTHS = (5, 7)
CATBOOST_SEEDS = (42, 1004, 2024, 777)
SHIPPED = {"network": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "form": (42, 1004, 2024, 777, 999, 13),
           "factorization": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "context": (42, 1004, 2024, 777, 999, 13)}
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    shared = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    shared = add_trackman_features(shared, trackman)
    del hierarchical, encoded, trackman
    gc.collect()
    started = time.time()
    block = build_block(raw_frame, FEATURE_SHRINKAGE, "projected")
    features = pd.concat([shared, block], axis=1)
    model_columns = v31_form_columns(features)
    del shared, block
    gc.collect()

    # Exactly the frame evaluate_v153_projected_prior builds for the shipped arm.
    categorical = [n for n, _ in EMBEDDING_SPECS if n in model_columns]
    work = features.copy()
    for name, _ in EMBEDDING_SPECS:
        if name not in work:
            work[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encodings = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encodings]
               + [c for c in categorical if c not in model_columns])
    work = work[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    del features
    gc.collect()
    print(f"CatBoost frame: {len(columns)} columns, {len(categorical)} categorical "
          f"[{time.time() - started:.0f}s]", flush=True)

    config = dict(CATBOOST_CONFIG)
    assert config.pop("depth") == INCUMBENT_DEPTH
    arms = {d: {s: {} for s in CATBOOST_SEEDS} for d in NEW_DEPTHS}
    started = time.time()
    for year in YEARS:
        train_slice = work.loc[season < year]
        train_target = targets[season < year]
        valid_slice = work.loc[season == year]
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        for depth in NEW_DEPTHS:
            for seed in CATBOOST_SEEDS:
                model = CatBoostClassifier(**config, depth=depth, random_seed=seed,
                                           verbose=0, thread_count=6,
                                           cat_features=categorical,
                                           allow_writing_files=False)
                model.fit(train_slice, train_target)
                arms[depth][seed][str(year)] = model.predict_proba(valid_slice)[:, 1]
                skill = 100000 * (1 - ((arms[depth][seed][str(year)] - actual) ** 2
                                       ).mean() / (rate * (1 - rate)))
                print(f"  {year} depth {depth} seed {seed:5d}: standalone {skill:8.0f} "
                      f"[{time.time() - started:.0f}s]", flush=True)
                del model
                gc.collect()
        del train_slice, valid_slice
        gc.collect()
    del work
    gc.collect()
    joblib.dump({"arms": arms}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    cached = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")["catboosts"]

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    stores = {
        "network": {**v164["networks"], **v190["networks"]},
        "form": {**v190["forms"], **v192["forms"]},
        "factorization": {**v164["factorizations"], **v194["factorizations"]},
        "context": v195["contexts"],
    }

    def mean_over(prediction_sets):
        return {str(y_): np.mean([p[str(y_)] for p in prediction_sets], axis=0)
                for y_ in YEARS}

    def mean_of(store, chosen):
        return mean_over([store[s] for s in chosen])

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        **{name: mean_of(stores[name], SHIPPED[name]) for name in stores},
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

    def pipeline(catboost):
        parts = dict(fixed)
        parts["catboost"] = catboost
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
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def points(candidate_error, base_error):
        return [float(P * (base_error[masks[y_]].mean() - candidate_error[masks[y_]].mean()))
                for y_ in YEARS]

    shipped_sets = [cached[s] for s in CATBOOST_SEEDS]
    base_error = (pipeline(mean_over(shipped_sets)) - target) ** 2

    # First the correlation structure: is a new-depth draw actually less like the
    # shipped average than a same-depth draw is?
    flat_base = np.concatenate([mean_over(shipped_sets)[str(y_)] for y_ in YEARS])
    print("\ncorrelation of each arm's fits with the shipped depth-6 average:", flush=True)
    correlations = {}
    for depth in NEW_DEPTHS + (INCUMBENT_DEPTH,):
        store = cached if depth == INCUMBENT_DEPTH else arms[depth]
        rho = float(np.mean([np.corrcoef(
            np.concatenate([store[s][str(y_)] for y_ in YEARS]), flat_base)[0, 1]
            for s in CATBOOST_SEEDS]))
        correlations[str(depth)] = rho
        print(f"  depth {depth}: {rho:.6f}", flush=True)

    # Candidate mixes, every one read against the shipped 4-seed depth-6 average.
    mixes = {}
    for depth in NEW_DEPTHS:
        for k in range(1, len(CATBOOST_SEEDS) + 1):
            rows = [points((pipeline(mean_over(
                shipped_sets + [arms[depth][s] for s in inside])) - target) ** 2,
                base_error) for inside in combinations(CATBOOST_SEEDS, k)]
            block_ = np.array(rows)
            mixes[f"6x4+{depth}x{k}"] = {
                "readings": len(rows), "mean": block_.mean(axis=0).tolist(),
                "sd_over_subsets": (block_.std(axis=0, ddof=1).tolist()
                                    if len(rows) > 1 else None)}
    for k in range(1, len(CATBOOST_SEEDS) + 1):
        rows = [points((pipeline(mean_over(
            shipped_sets + [arms[5][s] for s in inside]
            + [arms[7][s] for s in inside])) - target) ** 2,
            base_error) for inside in combinations(CATBOOST_SEEDS, k)]
        block_ = np.array(rows)
        mixes[f"6x4+5x{k}+7x{k}"] = {
            "readings": len(rows), "mean": block_.mean(axis=0).tolist(),
            "sd_over_subsets": (block_.std(axis=0, ddof=1).tolist()
                                if len(rows) > 1 else None)}

    print("\nmixes against the shipped 4-seed depth-6 average:", flush=True)
    print(f"  {'mix':>14s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    for name, row in mixes.items():
        mean = row["mean"]
        print(f"  {name:>14s} {mean[0]:+8.2f} {mean[1]:+8.2f} {mean[2]:+8.2f}"
              + ("   ALL POSITIVE" if all(v > 0 for v in mean) else ""), flush=True)

    survivors = [name for name, row in mixes.items()
                 if all(v > 0 for v in row["mean"])]
    print(f"\nmixes positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V206_catboost_depth_diversity",
        "baseline": "V199 (Public 1083.2461959655), CatBoost = 4-seed depth-6 average",
        "why": ("the seed-lottery axis paid +15.38 and saturated; V203 measured that a "
                "structural change decorrelates fits even at the same seed, which is "
                "exactly what makes a draw worth averaging -- the untested variant of "
                "the one mechanism that produced double-digit gains"),
        "design": ("depth-5 and depth-7 arms, four seeds x three folds each, every mix "
                   "with the cached depth-6 fits read against the shipped average"),
        "config_held": config,
        "incumbent_depth": INCUMBENT_DEPTH,
        "new_depths": list(NEW_DEPTHS),
        "seeds": list(CATBOOST_SEEDS),
        "correlation_with_shipped_average": correlations,
        "mixes": mixes,
        "survivors": survivors,
        "held_at": {k: list(v) for k, v in SHIPPED.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
