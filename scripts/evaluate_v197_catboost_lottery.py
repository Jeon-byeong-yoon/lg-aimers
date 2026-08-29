"""V197: the last lottery -- CatBoost, weight 0.27, the only component still one draw.

Four components have been averaged and three submissions have paid for it:

    V189  network 1 -> 3                          +3.59
    V193  network 3 -> 9, Form 1 -> 6             +5.52
    V196  factorization 1 -> 9, Context 1 -> 6    +5.86

and the pattern across all four is the same: the 2024 lottery range predicts what averaging
is worth, and the curve rises monotonically in k and saturates.

    component        weight   2024 range   averaging is worth
    network            0.20        17.94   +5.30
    factorization      0.07         8.67   +1.62
    Context            0.14         6.54   +0.84
    Form               0.32         2.97   +1.72
    CatBoost           0.27            ?                  ?

CatBoost is the second-largest weight and its randomness is the least like the others':
`random_seed` chooses the permutations behind ordered target statistics, which is what
replaces the hand-built encodings in this configuration. That is a deeper kind of
randomness than a bin-threshold subsample, so the lottery could be wide.

It was deferred twice on a size argument that turns out to be wrong. `docs/05` gives the
real limits -- **ZIP 10GB, uncompressed 32GB, inference 10 minutes** -- so nine CatBoost
models at about 113 MB each come to roughly 1 GB, a tenth of the archive limit, and V196
already runs thirty model evaluations per row in 32 seconds against a 600-second cap. The
only real constraint is training time: 160 seconds per fold fit.

Four seeds, so the unbiased curve reaches k=3. The frame is rebuilt exactly as
`evaluate_v153_projected_prior.py` builds it for the shipped `projected` arm -- the
`te_*`/`hte_*` encodings dropped so ordered target statistics replace them, every
categorical stringified, numerics cast to float32 -- and the control requires seed 42 to
reproduce the stored predictions to 0.000e+00.
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


OUTPUT = Path("artifacts/v197_catboost_lottery_metrics.json")
PREDICTIONS = Path("artifacts/v197_catboost_lottery_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
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

    catboosts = {s: {} for s in CATBOOST_SEEDS}
    started = time.time()
    for year in YEARS:
        train_slice = work.loc[season < year]
        train_target = targets[season < year]
        valid_slice = work.loc[season == year]
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        for seed in CATBOOST_SEEDS:
            model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=seed, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            model.fit(train_slice, train_target)
            catboosts[seed][str(year)] = model.predict_proba(valid_slice)[:, 1]
            skill = 100000 * (1 - ((catboosts[seed][str(year)] - actual) ** 2).mean()
                              / (rate * (1 - rate)))
            print(f"  {year} seed {seed:5d}: standalone {skill:8.0f} "
                  f"[{time.time() - started:.0f}s]", flush=True)
            del model
            gc.collect()
        del train_slice, valid_slice
        gc.collect()
    del work
    gc.collect()
    joblib.dump({"catboosts": catboosts}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    stored = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                         )["catboost"]["projected"]
    control = max(float(np.abs(catboosts[42][str(y_)] - stored[str(y_)]).max())
                  for y_ in YEARS)
    print(f"\ncontrol: seed 42 reproduces the shipped CatBoost to {control:.3e}",
          flush=True)

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

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

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

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    base_error = {s: (pipeline(catboosts[s]) - target) ** 2 for s in CATBOOST_SEEDS}
    lottery = np.array([points(pipeline(catboosts[s]), base_error[CATBOOST_SEEDS[0]])
                        for s in CATBOOST_SEEDS])
    print(f"\nCatBoost's lottery, every seed against seed 42:", flush=True)
    for s, row in zip(CATBOOST_SEEDS, lottery):
        print(f"  seed {s:5d}: {row[0]:+8.2f} {row[1]:+8.2f} {row[2]:+8.2f}", flush=True)
    print(f"  sd    {lottery.std(axis=0, ddof=1)[0]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[1]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[2]:7.2f}", flush=True)
    print(f"  range {lottery[:, 0].ptp():7.2f} {lottery[:, 1].ptp():7.2f} "
          f"{lottery[:, 2].ptp():7.2f}   "
          f"(network 17.94, factorization 8.67, Context 6.54, Form 2.97 on 2024)",
          flush=True)

    print("\naverage of k against the seeds it excludes, pooled over which go inside:",
          flush=True)
    print(f"  {'k':>3s} {'subsets':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    table = {}
    for k in range(1, len(CATBOOST_SEEDS)):
        readings = []
        for inside in combinations(CATBOOST_SEEDS, k):
            avg = catboosts[inside[0]] if k == 1 else mean_of(catboosts, inside)
            candidate = pipeline(avg)
            for s in CATBOOST_SEEDS:
                if s in inside:
                    continue
                readings.append(points(candidate, base_error[s]))
        block_ = np.array(readings)
        mean = block_.mean(axis=0)
        table[k] = {"readings": len(readings), "expected": mean.tolist(),
                    "sd": block_.std(axis=0, ddof=1).tolist()}
        print(f"  {k:3d} {len(list(combinations(CATBOOST_SEEDS, k))):8d} "
              f"{mean[0]:+8.2f} {mean[1]:+8.2f} {mean[2]:+8.2f}"
              + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    best = max(table, key=lambda k: table[k]["expected"][2])
    print(f"\nbest k by expected 2024 = {best} ({table[best]['expected'][2]:+.2f})",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V197_catboost_lottery",
        "baseline": "V196 (Public 1082.8399355688)",
        "why_now": ("deferred twice on a size argument that was wrong: docs/05 gives ZIP "
                    "10GB, uncompressed 32GB and inference 10 minutes, so nine CatBoost "
                    "models at ~113 MB come to roughly a tenth of the archive limit"),
        "seeds": list(CATBOOST_SEEDS),
        "control_vs_shipped": control,
        "lottery_per_seed": {str(s): row.tolist()
                             for s, row in zip(CATBOOST_SEEDS, lottery)},
        "lottery_sd": lottery.std(axis=0, ddof=1).tolist(),
        "lottery_range": [float(lottery[:, i].ptp()) for i in range(3)],
        "k_table": table,
        "best_k_by_2024": best,
        "held_at": {k: list(v) for k, v in SHIPPED.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
