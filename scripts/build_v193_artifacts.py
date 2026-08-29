"""Build V193: average both HistGradientBoosting Form and the network over many draws.

V189 replaced one network draw with three and gained +3.59. V191 then measured the curve
properly -- pooling over *which* seeds go inside, so the luckiest draw stops sitting in
every averaged set -- and both components rise monotonically and saturate:

    k     network 2024     Form 2024
    1           +0.00         +0.00
    3           +3.37         +1.43   <- network shipped here
    5           +4.34         +1.72
    8           +5.30             -

Nine network draws and six Form draws are cached per fold, so this ships those. Expected
against a fresh single draw: about +5.3 on 2024 from the network (+1.9 over V189) and about
+1.7 from Form, with every season positive at every k for both.

Form was never suspected of having a lottery -- its `random_state` only picks the
200,000-row subsample that sets the bin thresholds -- and its range on 2024 is 2.97 against
the network's 17.94. Small, but consistently signed, and it carries the largest weight in
the blend at 0.32.

Context (0.14) is left alone: it is the same model class as Form so its lottery is
presumably as narrow, but its cached predictions come from a different feature-removal
configuration and reproducing that is a separate job. CatBoost (0.27) is deferred because
one fit takes 160 seconds on the smallest fold and three shipped models would put the
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
from build_v12_calibration import make_lookup
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, state_bundle, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
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
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NETWORK_EPOCHS = 6
NETWORK_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)


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
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    features = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical, encoded
    gc.collect()
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    keep = v31_form_columns(features)
    assert len(keep) == 105, f"expected V175's 105-column frame, got {len(keep)}"
    print(f"feature frame: {len(keep)} columns", flush=True)

    form_models, form_columns = [], None
    for seed in FORM_SEEDS:
        started = time.time()
        model, columns = hist_gbdt_pipeline(features[keep])
        model.set_params(**{f"histgradientboostingclassifier__{k}": v
                            for k, v in FORM_CONFIG.items()},
                         histgradientboostingclassifier__random_state=seed)
        model.fit(features[columns], y)
        form_columns = columns if form_columns is None else form_columns
        assert columns == form_columns
        form_models.append(model)
        print(f"  Form seed {seed} trained [{time.time() - started:.0f}s]", flush=True)
    existing = joblib.load("artifacts/v105_feature_models.joblib")
    path = Path("artifacts/v193_feature_models.joblib")
    joblib.dump({
        "form_models": form_models, "form_columns": form_columns,
        "form_seeds": list(FORM_SEEDS),
        "context_model": existing["context_model"],
        "context_columns": existing["context_columns"],
        "context_lookup_2025": existing["context_lookup_2025"],
        "feature_reliability_scale": FEATURE_RELIABILITY,
    }, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB, "
          f"{len(form_models)} Form models)")
    del form_models
    gc.collect()

    categorical_columns = [column for column, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in keep
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

    bundles = []
    for seed in NETWORK_SEEDS:
        started = time.time()
        network = train(categorical, numeric, y.to_numpy(), cardinalities(vocabularies),
                        epochs=NETWORK_EPOCHS, seed=seed, verbose=False)
        bundle = state_bundle(network, vocabularies, statistics, numeric_columns,
                              cardinalities(vocabularies))
        bundle["seed"] = seed
        bundles.append(bundle)
        print(f"  network seed {seed} trained [{time.time() - started:.0f}s]", flush=True)
        del network
        gc.collect()
    for bundle in bundles[1:]:
        assert bundle["numeric_columns"] == bundles[0]["numeric_columns"]
    path = Path("artifacts/v193_networks.joblib")
    joblib.dump(bundles, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB, "
          f"{len(bundles)} bundles)")
    del categorical, numeric
    gc.collect()

    # ---- calibration, refitted because both averaged components moved ----
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    network_store = dict(v164["networks"])
    network_store.update(joblib.load(
        "artifacts/v190_form_lottery_and_k_predictions.joblib")["networks"])
    form_store = dict(v190["forms"])
    form_store.update(v192["forms"])
    assert set(NETWORK_SEEDS) <= set(network_store)
    assert set(FORM_SEEDS) <= set(form_store)
    network_oof = {str(year): np.mean(
        [network_store[s][str(year)] for s in NETWORK_SEEDS], axis=0) for year in YEARS}
    form_oof = {str(year): np.mean(
        [form_store[s][str(year)] for s in FORM_SEEDS], axis=0) for year in YEARS}
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                              )["no_matchup_hte"]["context"]
    catboost_oof = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                               )["catboost"]["projected"]
    factorization_oof = v160["factorization"][FEATURE_RELIABILITY]

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
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * network_oof[key] + W_CATBOOST * catboost_oof[key]
            + W_FACTORIZATION * factorization_oof[key])
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
    path = Path("artifacts/v193_calibration.joblib")
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
        "network_seeds": list(NETWORK_SEEDS),
        "form_seeds": list(FORM_SEEDS),
    }, path, compress=3)
    previous = joblib.load("artifacts/v189_calibration.joblib")["global_shift"]
    print(f"Saved {path}, shift={shift:.12f} (V189 was {previous:.12f})")

    Path("artifacts/v193_build_summary.json").write_text(json.dumps({
        "version": "V193",
        "baseline": "V189 (1071.4549348488)",
        "change": ("nine network draws instead of three, and six Form draws instead of "
                   "one; Context, CatBoost and the factorization network unchanged"),
        "network_seeds": list(NETWORK_SEEDS),
        "form_seeds": list(FORM_SEEDS),
        "validation": {
            "network_k_curve_2024": {"1": 0.0, "2": 2.19, "3": 3.37, "4": 3.86,
                                     "5": 4.34, "6": 4.59, "7": 5.16, "8": 5.30},
            "form_k_curve_2024": {"1": 0.0, "2": 1.07, "3": 1.43, "4": 1.61, "5": 1.72},
            "expected_over_v189": ("about +1.9 from the network going 3 -> 9 and about "
                                   "+1.7 from Form going 1 -> 6, on the 2024 fold"),
            "every_season_positive_at_every_k": True,
            "estimator": ("pooled over which seeds go inside, so the luckiest draw does "
                          "not sit in every averaged set -- the bias that made V190's "
                          "table read backwards"),
        },
        "deferred": {"context": "same model class, different cached configuration",
                     "catboost": "160s per fold fit; three models would near 220 MB"},
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
