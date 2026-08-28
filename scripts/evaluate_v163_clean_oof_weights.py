"""V163: rebuild the leaky out-of-fold predictions honestly, then refit the weights on them.

`add_row_features(frame, target_prior)` uses the prior to build
`{owner}_success_smoothed_{50,200}`. V102 passed the *fold's* prior,
`y[season < year].mean()`. V111, V116 and V130 passed the global `y.mean()`, which contains
the validation fold's labels -- a leak of one scalar, revealed when V158's control failed to
reproduce V102 by up to 3.8e-02.

The submission is unaffected: at deployment the training window is 2019-2024, so the global
mean *is* the training-only prior and no 2025 label enters. What is affected is every
weight decision. A leaky fold estimate makes a component look better than it generalises,
so its weight is biased upward -- and the three components on the global prior are CatBoost
(0.27), the network (0.20) and the factorization network (0.07), which is 54% of the blend.
Form and Context, on the fold prior, are clean. V159 found the weights locally optimal in
78 directions, but that was measured on top of this.

So: rebuild those three with the fold prior, check how much the leak was worth, and refit
the weights against the honest estimator. Both the incumbent weights and every alternative
are scored on the *clean* out-of-fold predictions, because the question is which weight
vector is better given an unbiased view -- not how the clean numbers compare to the biased
ones.

Each component keeps everything else it currently has: CatBoost its projected in-season
prior and reliability 150, the two networks the career prior and reliability 300.

Filter unchanged: no fold may lose, at least one must gain, ranked by weakest season then
the three-season average.
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
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
import interaction_network_v130 as inet
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
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
    add_inseason_features, add_training_inseason_features, build_anchor,
    drift_correction, feature_names,
)
from inseason_prior_v153 import population_inside_rates, projected_priors
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v163_clean_oof_metrics.json")
PREDICTIONS = Path("artifacts/v163_clean_oof_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
LIVE = ("form", "context", "network", "catboost", "factorization")
TRANSFER_STEPS = (0.02, 0.04, 0.07)
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
NETWORK_RELIABILITY = 300.0
CATBOOST_RELIABILITY = 150.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
CATBOOST_CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT = 8
FACTORIZATION_EPOCHS = 8
FACTORIZATION_HIDDEN = (128, 64)
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
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    del encoded, hierarchical
    gc.collect()

    network_block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=NETWORK_RELIABILITY)[feature_names()]
    table = population_inside_rates(raw_frame)
    pieces = []
    for value in sorted(raw_frame["season"].unique()):
        current = raw_frame.loc[raw_frame["season"] == value]
        history = raw_frame.loc[raw_frame["season"] < value]
        if history.empty:
            pieces.append(pd.DataFrame(np.nan, index=current.index,
                                       columns=feature_names()))
            continue
        pieces.append(add_inseason_features(
            current,
            {g: build_anchor(history, g) for g in ("pitcher", "batter")},
            projected_priors(history, value, table=table.loc[table.index < value]),
            current_season=value, shrinkage=FEATURE_SHRINKAGE,
            reliability_scale=CATBOOST_RELIABILITY)[feature_names()])
    catboost_block = pd.concat(pieces).loc[raw_frame.index]

    clean = {"network": {}, "factorization": {}, "catboost": {}}
    for year in YEARS:
        started = time.time()
        train_mask = season < year
        valid_mask = season == year
        fold_prior = float(y.loc[train_mask].mean())

        base_frame = select_v2_features(add_row_features(form_raw, fold_prior))
        base_frame = add_trackman_features(base_frame, trackman)

        identity = pd.concat([base_frame, network_block], axis=1)
        model_columns = v31_form_columns(identity)
        categorical_columns = [c for c, _ in EMBEDDING_SPECS]
        numeric_columns = [c for c in model_columns
                           if c not in categorical_columns and c != "season"]
        for column in categorical_columns:
            if column not in identity:
                identity[column] = raw_frame[column]
        assert_numeric(identity, numeric_columns)
        train_identity = identity.loc[train_mask]
        valid_identity = identity.loc[valid_mask]
        vocabularies = build_vocabularies(train_identity)
        statistics = numeric_statistics(
            train_identity[numeric_columns].to_numpy(dtype=np.float64))
        categorical = encode_categorical(train_identity, vocabularies)
        numeric = encode_numeric(
            train_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        net = train(categorical, numeric, targets[train_mask],
                    cardinalities(vocabularies), epochs=NETWORK_EPOCHS, verbose=False)
        clean["network"][str(year)] = predict(
            net, encode_categorical(valid_identity, vocabularies),
            encode_numeric(
                valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics))
        del net
        gc.collect()
        inet.LATENT = FACTORIZATION_LATENT
        inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
        fmodel = inet.train(categorical, numeric, targets[train_mask],
                            inet.field_cardinalities(vocabularies),
                            epochs=FACTORIZATION_EPOCHS, hidden=FACTORIZATION_HIDDEN,
                            verbose=False)
        clean["factorization"][str(year)] = inet.predict(
            fmodel, inet.encode_categorical(valid_identity, vocabularies),
            inet.encode_numeric(
                valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics))
        del fmodel, identity, train_identity, valid_identity
        gc.collect()

        cb_frame = pd.concat([base_frame, catboost_block], axis=1)
        cb_columns = v31_form_columns(cb_frame)
        cats = [n for n, _ in EMBEDDING_SPECS if n in cb_columns]
        for name, _ in EMBEDDING_SPECS:
            if name not in cb_frame:
                cb_frame[name] = raw_frame[name]
                if name not in cats:
                    cats.append(name)
        encodings = [c for c in cb_columns if c.startswith(("te_", "hte_"))]
        keep = ([c for c in cb_columns if c not in encodings]
                + [c for c in cats if c not in cb_columns])
        work = cb_frame[keep].copy()
        for name in cats:
            work[name] = work[name].astype(str)
        numeric_cols = [c for c in keep if c not in cats]
        work[numeric_cols] = work[numeric_cols].astype(np.float32)
        model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=42, verbose=0,
                                   thread_count=6, cat_features=cats,
                                   allow_writing_files=False)
        train_slice = work.loc[train_mask]
        model.fit(train_slice, targets[train_mask])
        del train_slice
        gc.collect()
        clean["catboost"][str(year)] = model.predict_proba(work.loc[valid_mask])[:, 1]
        del model, work, cb_frame, base_frame
        gc.collect()

        actual = targets[valid_mask].astype(float)
        rate = actual.mean()
        line = "  ".join(
            f"{k[:4]} {100000 * (1 - ((clean[k][str(year)] - actual) ** 2).mean() / (rate * (1 - rate))):7.0f}"
            for k in ("network", "factorization", "catboost"))
        print(f"  clean {year} (fold prior {fold_prior:.6f}): {line}  "
              f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                       )["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                          )["no_matchup_hte"]["context"]
    leaky = {
        "network": joblib.load("artifacts/v160_reliability_scale_predictions.joblib"
                               )["predictions"]["network"][NETWORK_RELIABILITY],
        "factorization": joblib.load(
            "artifacts/v160_reliability_scale_predictions.joblib"
        )["predictions"]["factorization"][NETWORK_RELIABILITY],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
    }
    print("\nwhat the leak was worth (standalone skill, clean minus leaky):", flush=True)
    for slot in clean:
        line = []
        for year in YEARS:
            actual = oof[str(year)]["target"].astype(float)
            rate = actual.mean()
            def skill(p):
                return 100000 * (1 - ((p - actual) ** 2).mean() / (rate * (1 - rate)))
            line.append(f"{year} {skill(clean[slot][str(year)]) - skill(leaky[slot][str(year)]):+7.1f}")
        print(f"  {slot:14s} " + "  ".join(line), flush=True)

    v17 = {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
           for v in YEARS}
    parts = dict(v17=v17, form=form, context=context, **clean)
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE,
            reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(weights):
        raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
               for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            train_f = calibration_frame.loc[index]
            valid_f = calibration_frame.loc[oof[str(year)]["row_index"]]
            pieces.append(np.clip(
                raw[year] + residual.mean()
                + W_COUNT * segment_correction(
                    train_f, residual, valid_f, ["balls_before", "strikes_before"], 500)
                + W_PITCHER_COUNT * segment_correction(
                    train_f, residual, valid_f,
                    ["pitcher_id", "balls_before", "strikes_before"], 300)
                + W_EXPERIENCE * segment_correction(
                    train_f, residual, valid_f, ["experience_bin"],
                    EXPERIENCE_SMOOTHING), 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(tuple(BASE[n] for n in NAMES6))
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    from itertools import permutations
    results = {}
    print("\nweight transfers on the clean estimator "
          "(does the leak's removal move the optimum?):", flush=True)
    for source, target in permutations(LIVE, 2):
        for step in TRANSFER_STEPS:
            weights = dict(BASE)
            if weights[source] - step < -1e-12:
                continue
            weights[source] = round(weights[source] - step, 6)
            weights[target] = round(weights[target] + step, 6)
            label = f"{source}->{target}_{step:.2f}"
            results[label] = evaluate(pipeline(tuple(weights[n] for n in NAMES6)))
            m = results[label]; t = m["three_season"]
            if min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0:
                print(f"  {label:30s} min {m['min_season_points']:+7.2f}  "
                      f"avg {t['average_points']:+7.2f}  "
                      f"2022 {t['season_points'][0]:+7.2f}  "
                      f"2023 {t['season_points'][1]:+7.2f}  "
                      f"2024 {t['season_points'][2]:+7.2f}  "
                      f"blk {m['monthly_block_win_rate']:4.0%}  SAFE", flush=True)

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V163_clean_oof_weights",
        "baseline": ("V161's weights scored on the clean estimator; V161 is Public "
                     "1053.2326413884"),
        "leak": ("add_row_features' target_prior was global in V111, V116 and V130 and "
                 "per-fold in V102, so the validation fold's labels reached three "
                 "components as one scalar. The submission is unaffected -- at deployment "
                 "the global mean is the training-only prior -- but a leaky fold estimate "
                 "biases a component's weight upward, and those three hold 54% of the "
                 "blend."),
        "design": ("both the incumbent weights and every alternative are scored on the "
                   "clean out-of-fold predictions, since the question is which weight "
                   "vector is better under an unbiased view"),
        "transfer_steps": list(TRANSFER_STEPS),
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fold_prior": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"clean": clean}, PREDICTIONS, compress=3)
    print(f"\nsafe = {len(survivors)} of {len(results)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
