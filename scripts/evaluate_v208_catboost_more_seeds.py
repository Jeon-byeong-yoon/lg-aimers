"""V208: CatBoost seeds 4 -> 9, the last measured card.

Doc 87 already priced this: k 4->9 is worth about +0.3 at the fold level, so roughly
+0.15 on the leaderboard at the axis's realized ratio -- deferred then because other
components' lotteries were worth more per hour. Now every axis is closed, submission
slots are spare, and this is the only remaining candidate with a MEASURED positive
expectation.

It does not contradict the V207 rejection: V207 replaced a constant (transfer record
0/1), while this adds draws to an average (transfer record 4/4 -- V189 +3.59, V193
+5.52, V196 +5.86, V199 +0.41; V199 in particular showed a small additive effect still
transfers positive).

Five new seeds -- the same ones the other components use -- three folds each, shipped
config exactly. The readout mirrors V206: every "shipped 4 + j new" subset average is
read against the shipped 4-seed average through the full blend + calibration pipeline.
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


OUTPUT = Path("artifacts/v208_catboost_more_seeds_metrics.json")
PREDICTIONS = Path("artifacts/v208_catboost_more_seeds_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SHIPPED_CATBOOST_SEEDS = (42, 1004, 2024, 777)
NEW_SEEDS = (999, 13, 314, 2718, 65537)
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

    arms = {s: {} for s in NEW_SEEDS}
    started = time.time()
    for year in YEARS:
        train_slice = work.loc[season < year]
        train_target = targets[season < year]
        valid_slice = work.loc[season == year]
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        for seed in NEW_SEEDS:
            model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=seed, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            model.fit(train_slice, train_target)
            arms[seed][str(year)] = model.predict_proba(valid_slice)[:, 1]
            skill = 100000 * (1 - ((arms[seed][str(year)] - actual) ** 2).mean()
                              / (rate * (1 - rate)))
            print(f"  {year} seed {seed:5d}: standalone {skill:8.0f} "
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

    shipped_sets = [cached[s] for s in SHIPPED_CATBOOST_SEEDS]
    base_error = (pipeline(mean_over(shipped_sets)) - target) ** 2

    print("\n'shipped 4 + j new' against the shipped 4-seed average:", flush=True)
    print(f"  {'j':>3s} {'subsets':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s}",
          flush=True)
    curve = {}
    for j in range(1, len(NEW_SEEDS) + 1):
        rows = [points((pipeline(mean_over(
            shipped_sets + [arms[s] for s in inside])) - target) ** 2, base_error)
            for inside in combinations(NEW_SEEDS, j)]
        block_ = np.array(rows)
        mean = block_.mean(axis=0)
        curve[str(j)] = {
            "readings": len(rows), "mean": mean.tolist(),
            "sd_over_subsets": (block_.std(axis=0, ddof=1).tolist()
                                if len(rows) > 1 else None),
            "all_positive": bool((mean > 0).all())}
        print(f"  {j:3d} {len(rows):8d} {mean[0]:+8.2f} {mean[1]:+8.2f} {mean[2]:+8.2f}"
              + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    survivors = [j for j in curve if curve[j]["all_positive"]]
    print(f"\nj positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V208_catboost_more_seeds",
        "baseline": "V199 (Public 1083.2461959655), CatBoost = 4-seed average",
        "why": ("doc 87 priced k 4->9 at about +0.3 on the folds; deferred then, but "
                "now every axis is closed and this is the only candidate left with a "
                "measured positive expectation; adds draws (transfer 4/4), does not "
                "replace a constant (transfer 0/1)"),
        "design": ("five new seeds x three folds at the shipped config, every "
                   "'4 + j new' subset average read against the shipped average"),
        "shipped_seeds": list(SHIPPED_CATBOOST_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "curve": curve,
        "survivors": survivors,
        "held_at": {k: list(v) for k, v in SHIPPED.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
