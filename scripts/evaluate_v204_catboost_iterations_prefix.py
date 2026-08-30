"""V204: the CatBoost iteration count, read through prefix pairing.

V123/V124 chose iterations=1200 on the old instrument, whose unpaired noise V200 later
measured at sd 8.67 on the 2024 fold. V202/V203 established that a paired instrument
hears a change in proportion to how little of the model it touches -- and the iteration
count admits the limiting case: with learning_rate fixed at 0.02, the first 1200 trees
of a 2400-iteration fit ARE the 1200-iteration model. Reading the same fit at nine
`ntree_end` prefixes gives the whole iteration curve from twelve trainings, with pairing
strictly better than the HGB `iter_only_400` arm (SE 0.43) that settled max_iter in V203.

The frame is byte-for-byte the V197 frame (itself the shipped `projected` arm of
`evaluate_v153_projected_prior`), and the control requires the 1200-prefix to reproduce
the cached V197 predictions, which V197 in turn tied to the shipped submission CatBoost.

Depth and the other structural constants are out of scope: V203 showed pairing cancels
nothing when the tree structure changes (improvement 0-1.1x).
"""

import gc
import json
import sys
import time
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


OUTPUT = Path("artifacts/v204_catboost_iterations_prefix_metrics.json")
PREDICTIONS = Path("artifacts/v204_catboost_iterations_prefix_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
CATBOOST_SEEDS = (42, 1004, 2024, 777)
MAX_ITERATIONS = 2400
PREFIXES = (400, 600, 800, 1000, 1200, 1500, 1800, 2100, 2400)
INCUMBENT_PREFIX = 1200
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
    assert config.pop("iterations") == INCUMBENT_PREFIX
    prefixes = {k: {s: {} for s in CATBOOST_SEEDS} for k in PREFIXES}
    started = time.time()
    for year in YEARS:
        train_slice = work.loc[season < year]
        train_target = targets[season < year]
        valid_slice = work.loc[season == year]
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        for seed in CATBOOST_SEEDS:
            model = CatBoostClassifier(**config, iterations=MAX_ITERATIONS,
                                       random_seed=seed, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            model.fit(train_slice, train_target)
            for k in PREFIXES:
                prefixes[k][seed][str(year)] = model.predict_proba(
                    valid_slice, ntree_end=k)[:, 1]
            skill = {k: 100000 * (1 - ((prefixes[k][seed][str(year)] - actual) ** 2
                                       ).mean() / (rate * (1 - rate)))
                     for k in (PREFIXES[0], INCUMBENT_PREFIX, PREFIXES[-1])}
            print(f"  {year} seed {seed:5d}: standalone "
                  + " ".join(f"@{k} {v:8.0f}" for k, v in skill.items())
                  + f" [{time.time() - started:.0f}s]", flush=True)
            del model
            gc.collect()
        del train_slice, valid_slice
        gc.collect()
    del work
    gc.collect()
    joblib.dump({"prefixes": prefixes}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    # Control: the 1200-prefix must be the cached V197 CatBoost, which V197 tied to
    # the shipped submission model.
    cached = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")["catboosts"]
    control = max(float(np.abs(prefixes[INCUMBENT_PREFIX][s][str(y_)]
                               - cached[s][str(y_)]).max())
                  for s in CATBOOST_SEEDS for y_ in YEARS)
    print(f"\ncontrol: 1200-prefix reproduces cached V197 to {control:.3e}", flush=True)

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

    def points(candidate_error, base_error):
        return [float(P * (base_error[masks[y_]].mean() - candidate_error[masks[y_]].mean()))
                for y_ in YEARS]

    incumbent_errors = {s: (pipeline(prefixes[INCUMBENT_PREFIX][s]) - target) ** 2
                        for s in CATBOOST_SEEDS}
    incumbent_average_error = (pipeline(mean_of(
        prefixes[INCUMBENT_PREFIX], CATBOOST_SEEDS)) - target) ** 2

    print(f"\npaired against the {INCUMBENT_PREFIX}-prefix at the same seed:", flush=True)
    print(f"  {'trees':>6s} {'2022':>17s} {'2023':>17s} {'2024':>17s}", flush=True)
    results = {}
    for k in PREFIXES:
        if k == INCUMBENT_PREFIX:
            continue
        rows = [points((pipeline(prefixes[k][s]) - target) ** 2, incumbent_errors[s])
                for s in CATBOOST_SEEDS]
        block_ = np.array(rows)
        mean = block_.mean(axis=0)
        se = block_.std(axis=0, ddof=1) / np.sqrt(len(CATBOOST_SEEDS))
        avg = points((pipeline(mean_of(prefixes[k], CATBOOST_SEEDS)) - target) ** 2,
                     incumbent_average_error)
        results[str(k)] = {
            "per_seed": {str(s): r for s, r in zip(CATBOOST_SEEDS, rows)},
            "paired_mean": mean.tolist(), "paired_se": se.tolist(),
            "average_vs_average": avg,
            "all_positive": bool((mean > 0).all())}
        print(f"  {k:6d} " + "  ".join(
            f"{mean[i]:+7.2f}+-{se[i]:5.2f}" for i in range(3))
            + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)
        print(f"         avg-vs-avg  {avg[0]:+7.2f}        {avg[1]:+7.2f}        "
              f"{avg[2]:+7.2f}", flush=True)

    survivors = [k for k in results if results[k]["all_positive"]]
    print(f"\nprefixes positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V204_catboost_iterations_prefix",
        "baseline": "V199 (Public 1083.2461959655), CatBoost iterations 1200",
        "why": ("iterations=1200 was chosen by V123/V124 on the old instrument (unpaired "
                "noise sd 8.67 on 2024); prefixes of one fit give the whole curve with "
                "pairing strictly better than the arm that settled HGB max_iter"),
        "design": ("twelve fits at 2400 iterations, read at nine ntree_end prefixes, "
                   "paired against the 1200-prefix at the same seed"),
        "config_held": {**config, "max_iterations_trained": MAX_ITERATIONS},
        "seeds": list(CATBOOST_SEEDS),
        "prefixes": list(PREFIXES),
        "incumbent_prefix": INCUMBENT_PREFIX,
        "control_vs_cached_v197": control,
        "results": results,
        "survivors": survivors,
        "held_at": {k: list(v) for k, v in SHIPPED.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
