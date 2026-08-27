"""Build V161: RELIABILITY_SCALE 150 -> 300 for the three components that read it as a feature.

`RELIABILITY_SCALE` builds one of the eighteen in-season columns,

    ins_pitcher_reliability = inside_n / (inside_n + reliability_scale)

so it sets the half-trust point in current-season pitches. V92 chose 150 a priori. V155
gridded the *drift term's* use of the same constant across eighteen combinations and found
150 optimal, but the drift term reads the reconstruction at shrinkage 3 and multiplies by
the column, whereas Form, the network and the factorization network read it at shrinkage 20
as an input. Only the first question had been asked.

V160 asked the second:

    reliability    min    2022    2023    2024   blocks
             75  -0.82   -0.35   -0.82   -0.72     40%
            150   0.00    0.00    0.00    0.00        -
            300  +0.54   +0.54   +3.37   +0.77     65%   <- no fold loses
            600  -0.25   +0.64   +5.74   -0.25     60%

The control at 150 reproduced the stored predictions of all three components to
0.000e+00 -- Form, the network and the factorization network -- so the comparison isolates
the constant. Reaching that required refitting each with its own historical prior
convention: per-fold for Form as V102 used, global for the two networks as V112 and V130
did. Form's predictions turned out identical at every scale, so the movement is entirely
the two networks.

CatBoost is untouched. It already takes its own reconstruction -- V154 gave it the
projected prior while these three kept the career one -- and it keeps reliability 150,
which is why no fourth reconstruction is needed at inference: only the existing
career-prior block changes.

Run with the server-mirror interpreter.
"""

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
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, state_bundle, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
FEATURE_RELIABILITY = 300.0          # was 150.0
CATBOOST_RELIABILITY = 150.0         # CatBoost's own block, unchanged
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT = 8
FACTORIZATION_EPOCHS = 8
FACTORIZATION_HIDDEN = (128, 64)


def main():
    total = W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST + W_FACTORIZATION
    assert abs(total - 1.0) < 1e-12, total
    assert abs(W_COUNT + W_PITCHER_COUNT + W_EXPERIENCE - 1.0) < 1e-12

    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    print(f"ins_pitcher_reliability mean at scale {FEATURE_RELIABILITY:g}: "
          f"{block['ins_pitcher_reliability'].mean():.5f}", flush=True)

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    keep = v31_form_columns(features)

    started = time.time()
    form_model, form_columns = hist_gbdt_pipeline(features[keep])
    form_model.set_params(**{f"histgradientboostingclassifier__{k}": v
                             for k, v in FORM_CONFIG.items()})
    form_model.fit(features[form_columns], y)
    print(f"Form trained on {len(features):,} rows, {len(form_columns)} features "
          f"[{time.time() - started:.0f}s]", flush=True)

    # Context and its 2025 Trackman lookup are unchanged; only the Form half is replaced.
    existing = joblib.load("artifacts/v105_feature_models.joblib")
    bundle_path = Path("artifacts/v161_feature_models.joblib")
    joblib.dump({
        "form_model": form_model, "form_columns": form_columns,
        "context_model": existing["context_model"],
        "context_columns": existing["context_columns"],
        "context_lookup_2025": existing["context_lookup_2025"],
        "feature_reliability_scale": FEATURE_RELIABILITY,
    }, bundle_path, compress=3)
    print(f"Saved {bundle_path} ({bundle_path.stat().st_size / 2**20:.2f} MiB)")

    categorical_columns = [column for column, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in keep
                       if c not in categorical_columns and c != "season"]
    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)
    vocabularies = build_vocabularies(identity)
    statistics = numeric_statistics(
        identity[numeric_columns].to_numpy(dtype=np.float64))
    categorical = encode_categorical(identity, vocabularies)
    numeric = encode_numeric(
        identity[numeric_columns].to_numpy(dtype=np.float64), statistics)

    started = time.time()
    network = train(categorical, numeric, y.to_numpy(), cardinalities(vocabularies),
                    epochs=NETWORK_EPOCHS)
    print(f"network trained [{time.time() - started:.0f}s]", flush=True)
    network_bundle = state_bundle(network, vocabularies, statistics, numeric_columns,
                                  cardinalities(vocabularies))
    network_bundle["variant"] = "without_season"
    network_bundle["feature_shrinkage"] = FEATURE_SHRINKAGE
    network_bundle["feature_reliability_scale"] = FEATURE_RELIABILITY
    path = Path("artifacts/v161_network.joblib")
    joblib.dump(network_bundle, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB)")

    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
    spec = inet.field_cardinalities(vocabularies)
    started = time.time()
    factorization = inet.train(categorical, numeric, y.to_numpy(), spec,
                               epochs=FACTORIZATION_EPOCHS,
                               hidden=FACTORIZATION_HIDDEN, verbose=True)
    print(f"factorization network trained "
          f"[{time.time() - started:.0f}s]", flush=True)
    factorization_bundle = inet.state_bundle(
        factorization, vocabularies, statistics, numeric_columns, spec)
    assert tuple(factorization_bundle["hidden"]) == FACTORIZATION_HIDDEN
    factorization_bundle["latent"] = FACTORIZATION_LATENT
    factorization_bundle["epochs"] = FACTORIZATION_EPOCHS
    factorization_bundle["feature_reliability_scale"] = FEATURE_RELIABILITY
    path = Path("artifacts/v161_factorization.joblib")
    joblib.dump(factorization_bundle, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB)")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")
    form_oof = v160["predictions"]["form"][FEATURE_RELIABILITY]
    network_oof = v160["predictions"]["network"][FEATURE_RELIABILITY]
    factorization_oof = v160["predictions"]["factorization"][FEATURE_RELIABILITY]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                              )["no_matchup_hte"]["context"]
    catboost_oof = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                               )["catboost"]["projected"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = pd.cut(raw["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
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

    calibration = Path("artifacts/v161_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": W_COUNT,
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"],
                                   COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": W_PITCHER_COUNT,
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["experience_bin"], "weight": W_EXPERIENCE,
             "lookup": make_lookup(frame, centered, ["experience_bin"],
                                   EXPERIENCE_SMOOTHING)},
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
    }, calibration, compress=3)
    print(f"\nSaved {calibration}, shift={shift:.12f} (V156 was -0.010678264995)")

    Path("artifacts/v161_build_summary.json").write_text(json.dumps({
        "version": "V161",
        "baseline": "V156 (1052.1807428872)",
        "change": ("feature-side RELIABILITY_SCALE 150 -> 300 for Form, the network and "
                   "the factorization network; CatBoost keeps 150 on its own block"),
        "validation": {
            "season_points": {"2022": 0.54, "2023": 3.37, "2024": 0.77},
            "min_season_points": 0.54, "average_points": 1.56,
            "monthly_block_win_rate": 0.65,
            "control": ("reliability 150 reproduced the stored Form, network and "
                        "factorization predictions to 0.000e+00"),
            "note": ("Form's predictions were identical at every scale, so the movement "
                     "is entirely the two networks."),
        },
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
