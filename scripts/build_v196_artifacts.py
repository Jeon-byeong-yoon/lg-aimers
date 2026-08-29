"""Build V196: average the last two cheap lotteries -- factorization and Context.

Four learned components have now been measured with the corrected estimator, which pools
over which seeds go inside so the luckiest draw does not sit on both sides of the
comparison. Every one of them gains from averaging, and the size tracks the lottery:

    component        weight   2024 range   averaging is worth, 2024   status after V193
    network            0.20        17.94   +5.30 at k=8               9 draws
    factorization      0.07         8.67   +1.62 at k=7               1 draw
    Context            0.14         6.54   +0.84 at k=5               1 draw
    Form               0.32         2.97   +1.72 at k=5               6 draws
    CatBoost           0.27            ?   deferred                   1 draw

This ships nine factorization draws and six Context draws. Expected against V193: about
+1.6 from the factorization network and +0.84 from Context on the 2024 fold.

Two notes on the measurements behind it. The factorization network was *rejected* at V188 on
an expected column of -2.67 / -8.31 / +10.02; that estimator averaged a set containing seed
42 against baselines that also contained it, and V191 showed the same bias reversing the
network's curve. Corrected, the factorization curve rises to +1.62 and is positive on every
season at k = 4, 5, 7 and 8. Context's curve is the cleanest of the four -- monotone on all
three seasons, +0.53 / +0.70 / +0.79 / +0.84 -- and its control is exact: seed 42 reproduces
the shipped Context to 0.000e+00.

Context's lottery is *wider* than Form's (6.54 against 2.97) despite being the same model
class, which fits: it is the weaker model of the two, around 670-725 skill on 2024 against
Form's 780, so it has more room to wobble.

CatBoost stays deferred: 160 seconds per fold fit, and three shipped models would put the
archive near 220 MB.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import interaction_network_v130 as inet
from build_v12_calibration import make_lookup
from contextual_trackman_v24 import (
    add_context_trackman_features, build_context_lookup, prepare_context_trackman,
)
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, encode_categorical,
    encode_numeric, numeric_statistics,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v31_feature_removal import REMOVALS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
RAW_SEGMENT_WEIGHTS = {"count": 0.55, "pitcher_count": 0.25,
                       "experience": 0.20, "platoon_two_strike": 0.80}
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
PLATOON_TWO_STRIKE_SMOOTHING = 1500.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
CATBOOST_RELIABILITY = 150.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
FACTORIZATION_LATENT, FACTORIZATION_EPOCHS = 8, 8
FACTORIZATION_HIDDEN = (128, 64)
FACTORIZATION_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
CONTEXT_SEEDS = (42, 1004, 2024, 777, 999, 13)
NETWORK_SEEDS = FACTORIZATION_SEEDS
FORM_SEEDS = CONTEXT_SEEDS
MATCHUP_HTE = REMOVALS["no_matchup_hte"]


def main():
    assert abs(W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST
               + W_FACTORIZATION - 1.0) < 1e-12
    denominator = sum(RAW_SEGMENT_WEIGHTS.values())
    weights = {k: v / denominator for k, v in RAW_SEGMENT_WEIGHTS.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-12

    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    prior = float(y.mean())
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    context_trackman = prepare_context_trackman(trackman)
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    del encoded
    gc.collect()

    # ---- Context, six draws; same construction as train_v31_models.py ----
    context_features = select_v2_features(add_row_features(hierarchical, prior))
    context_features = add_trackman_features(context_features, trackman)
    context_features = add_context_trackman_features(context_features, context_trackman)
    keep = [c for c in context_features if c not in MATCHUP_HTE]
    context_models, context_columns = [], None
    for seed in CONTEXT_SEEDS:
        started = time.time()
        model, columns = hist_gbdt_pipeline(context_features[keep])
        model.set_params(histgradientboostingclassifier__random_state=seed)
        model.fit(context_features[columns], y)
        context_columns = columns if context_columns is None else context_columns
        assert columns == context_columns
        context_models.append(model)
        print(f"  Context seed {seed} trained [{time.time() - started:.0f}s]", flush=True)
    del context_features
    gc.collect()

    # ---- the frame the Form models and both neural components share ----
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    features = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), prior))
    del hierarchical
    gc.collect()
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    form_keep = v31_form_columns(features)
    assert len(form_keep) == 105, len(form_keep)

    v193 = joblib.load("artifacts/v193_feature_models.joblib")
    assert v193["form_seeds"] == list(FORM_SEEDS), v193["form_seeds"]
    path = Path("artifacts/v196_feature_models.joblib")
    joblib.dump({
        "form_models": v193["form_models"], "form_columns": v193["form_columns"],
        "form_seeds": list(FORM_SEEDS),
        "context_models": context_models, "context_columns": context_columns,
        "context_seeds": list(CONTEXT_SEEDS),
        "context_lookup_2025": build_context_lookup(context_trackman, 2025),
        "feature_reliability_scale": FEATURE_RELIABILITY,
    }, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB, "
          f"{len(v193['form_models'])} Form + {len(context_models)} Context)")
    del context_models
    gc.collect()

    # ---- factorization network, nine draws ----
    categorical_columns = [column for column, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in form_keep
                       if c not in categorical_columns and c != "season"]
    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)
    del features
    gc.collect()
    vocabularies = build_vocabularies(identity)
    statistics = numeric_statistics(
        identity[numeric_columns].to_numpy(dtype=np.float64))
    categorical = encode_categorical(identity, vocabularies)
    numeric = encode_numeric(
        identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
    del identity
    gc.collect()
    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
    spec = inet.field_cardinalities(vocabularies)
    bundles = []
    for seed in FACTORIZATION_SEEDS:
        started = time.time()
        model = inet.train(categorical, numeric, y.to_numpy(), spec,
                           epochs=FACTORIZATION_EPOCHS, seed=seed,
                           hidden=FACTORIZATION_HIDDEN, verbose=False)
        bundle = inet.state_bundle(model, vocabularies, statistics, numeric_columns, spec)
        bundle["latent"] = FACTORIZATION_LATENT
        bundle["epochs"] = FACTORIZATION_EPOCHS
        bundle["seed"] = seed
        bundles.append(bundle)
        print(f"  factorization seed {seed} trained [{time.time() - started:.0f}s]",
              flush=True)
        del model
        gc.collect()
    for bundle in bundles[1:]:
        assert bundle["numeric_columns"] == bundles[0]["numeric_columns"]
    path = Path("artifacts/v196_factorizations.joblib")
    joblib.dump(bundles, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB, "
          f"{len(bundles)} bundles)")
    del categorical, numeric, bundles
    gc.collect()

    # ---- calibration, refitted because two more components moved ----
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    network_store = dict(v164["networks"]); network_store.update(v190["networks"])
    form_store = dict(v190["forms"]); form_store.update(v192["forms"])
    fm_store = dict(v164["factorizations"]); fm_store.update(v194["factorizations"])
    context_store = v195["contexts"]
    cached_context = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                                 )["no_matchup_hte"]["context"]
    control = max(float(np.abs(context_store[42][str(y_)] - cached_context[str(y_)]).max())
                  for y_ in YEARS)
    print(f"control: V195's seed-42 Context matches the shipped one to {control:.3e}")
    assert control == 0.0, "the cached seed-42 Context must be the one V193 shipped"

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    parts = {
        "form": mean_of(form_store, FORM_SEEDS),
        "network": mean_of(network_store, NETWORK_SEEDS),
        "factorization": mean_of(fm_store, FACTORIZATION_SEEDS),
        "context": mean_of(context_store, CONTEXT_SEEDS),
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
    }
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = pd.cut(raw["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
    raw["two_strike"] = (raw["strikes_before"] == 2).astype("int64")
    indices, targets, predictions = [], [], []
    for year in YEARS:
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * parts["form"][key] + W_CONTEXT * parts["context"][key]
            + W_NETWORK * parts["network"][key] + W_CATBOOST * parts["catboost"][key]
            + W_FACTORIZATION * parts["factorization"][key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    fine = make_lookup(frame, centered, ["pitcher_id", "batter_hand", "two_strike"],
                       PLATOON_TWO_STRIKE_SMOOTHING)
    assert fine["two_strike"].dtype == np.int64
    path = Path("artifacts/v196_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": weights["count"],
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": weights["pitcher_count"],
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["experience_bin"], "weight": weights["experience"],
             "lookup": make_lookup(frame, centered, ["experience_bin"],
                                   EXPERIENCE_SMOOTHING)},
            {"columns": ["pitcher_id", "batter_hand", "two_strike"],
             "weight": weights["platoon_two_strike"], "lookup": fine},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "feature_reliability_scale": FEATURE_RELIABILITY,
        "catboost_reliability_scale": CATBOOST_RELIABILITY,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
        "experience_edges": [float(v) for v in EDGES],
        "experience_labels": list(LABELS),
        "network_seeds": list(NETWORK_SEEDS), "form_seeds": list(FORM_SEEDS),
        "factorization_seeds": list(FACTORIZATION_SEEDS),
        "context_seeds": list(CONTEXT_SEEDS),
    }, path, compress=3)
    previous = joblib.load("artifacts/v193_calibration.joblib")["global_shift"]
    print(f"Saved {path}, shift={shift:.12f} (V193 was {previous:.12f})")

    Path("artifacts/v196_build_summary.json").write_text(json.dumps({
        "version": "V196",
        "baseline": "V193 (1076.9782849555)",
        "change": ("nine factorization draws and six Context draws, both previously "
                   "single; the network stays at nine and Form at six"),
        "seeds": {"network": list(NETWORK_SEEDS), "form": list(FORM_SEEDS),
                  "factorization": list(FACTORIZATION_SEEDS),
                  "context": list(CONTEXT_SEEDS)},
        "validation": {
            "factorization_2024_curve": {"1": 0.0, "3": 1.36, "5": 1.15, "7": 1.62,
                                         "8": 1.59},
            "context_2024_curve": {"1": 0.0, "2": 0.53, "3": 0.70, "4": 0.79, "5": 0.84},
            "expected_over_v193_2024": 2.4,
            "lottery_range_2024": {"network": 17.94, "factorization": 8.67,
                                   "context": 6.54, "form": 2.97},
            "controls": ("V195's seed-42 Context reproduces the shipped one to 0.000e+00; "
                         "the build asserts it"),
            "v188_reversal": ("the factorization network was rejected at V188 on an "
                              "estimator that averaged a set containing seed 42 against "
                              "baselines that also contained it"),
        },
        "deferred": {"catboost": "160s per fold fit; three models would near 220 MB"},
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
