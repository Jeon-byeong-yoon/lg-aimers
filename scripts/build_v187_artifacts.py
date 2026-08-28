"""Build V187: give the unused hierarchical encodings to the embedding network.

`hierarchical_target_encoding_v6.py` defines four groups and every script in this
repository has passed only `["pitcher_batter"]` since V6, where the other three were
dropped on a 3e-5 Brier difference from a single HGB at the V5 baseline. The leaderboard
has since paid +14.65 for the platoon content delivered through the calibration layer, so
the question was only whether a model can also use it.

V185 measured every component separately and V186 settled it on five paired seeds. Only
the **embedding network** gains -- Form loses 2024 (-5.04) and the factorization network
swings 30 points between seeds -- so only the network changes here.

    arm `all`, network only, paired across five seeds
      2024   +2.52  +1.23  +5.36  +0.52  +3.88     positive at every seed, mean +2.70
      2023   +5.38  -0.42  +5.26  +30.81 +3.05     mean +8.81
      seed 42 (shipped)   2022 +0.07   2023 +5.38   2024 +2.52

`all` is shipped rather than the platoon-only arm, whose 2024 is stronger (+4.42 mean,
+9.45 at seed 42, bootstrap CI low +1.78) but which loses 2023 by 0.17 at the shipping
seed. Overriding the no-losing-fold gate has been tried twice and lost twice, so the arm
that passes it outright goes first; the platoon-only arm is the immediate follow-up.

Everything else is V175: Form, Context, CatBoost and the factorization network are reused
byte for byte, the blend weights are unchanged, and the four calibration segments keep
their weights and smoothings. The calibration is refitted because the network's
out-of-fold predictions moved, which moves the residual the layer is fitted on.
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
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import (
    add_prior_season_hierarchical_encodings, build_hierarchical_test_lookups,
)
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
NETWORK_EPOCHS, NETWORK_SEED = 6, 42
GROUPS = ["pitcher_batter", "pitcher_vs_batter_hand", "pitcher_count",
          "batter_vs_pitcher_hand"]
NEW_GROUPS = ["pitcher_vs_batter_hand", "pitcher_count", "batter_vs_pitcher_hand"]


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

    started = time.time()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, GROUPS)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    features = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical
    gc.collect()
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    keep = v31_form_columns(features)
    print(f"feature frame: {len(keep)} columns [{time.time() - started:.0f}s]", flush=True)

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

    started = time.time()
    network = train(categorical, numeric, y.to_numpy(), cardinalities(vocabularies),
                    epochs=NETWORK_EPOCHS, seed=NETWORK_SEED)
    print(f"network trained on {len(y):,} rows [{time.time() - started:.0f}s]", flush=True)
    bundle = state_bundle(network, vocabularies, statistics, numeric_columns,
                          cardinalities(vocabularies))
    bundle["feature_reliability_scale"] = FEATURE_RELIABILITY
    bundle["hierarchical_groups"] = GROUPS
    path = Path("artifacts/v187_network.joblib")
    joblib.dump(bundle, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB)")
    del network, categorical, numeric
    gc.collect()

    lookups = build_hierarchical_test_lookups(data, y, NEW_GROUPS)
    for group, table in lookups.items():
        print(f"  lookup {group}: {len(table)} rows, {len(table.columns)} columns")
    path = Path("artifacts/v187_hierarchical_lookups.joblib")
    joblib.dump(lookups, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB)")

    # ---- calibration, refitted because the network's out-of-fold predictions moved ----
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    network_oof = joblib.load(
        "artifacts/v186_platoon_network_five_seeds_predictions.joblib"
    )["networks"]["all"][NETWORK_SEED]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                           )["forms"][FEATURE_SHRINKAGE]
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

    fine = make_lookup(frame, centered, ["pitcher_id", "batter_hand", "two_strike"],
                       PLATOON_TWO_STRIKE_SMOOTHING)
    assert fine["two_strike"].dtype == np.int64, fine["two_strike"].dtype
    path = Path("artifacts/v187_calibration.joblib")
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
    }, path, compress=3)
    previous = joblib.load("artifacts/v175_calibration.joblib")["global_shift"]
    print(f"Saved {path}, shift={shift:.12f} (V175 was {previous:.12f}; the network "
          f"moved, so this one is expected to differ)")

    Path("artifacts/v187_build_summary.json").write_text(json.dumps({
        "version": "V187",
        "baseline": "V175 (1067.8617513573)",
        "change": ("pass all four hierarchical encoding groups instead of only "
                   "pitcher_batter, and retrain the embedding network on them; Form, "
                   "Context, CatBoost and the factorization network are unchanged"),
        "groups": GROUPS,
        "models_retrained": ["embedding network"],
        "validation": {
            "paired_across_five_seeds": {
                "2024": [2.52, 1.23, 5.36, 0.52, 3.88],
                "2023": [5.38, -0.42, 5.26, 30.81, 3.05],
                "2024_mean": 2.70, "2023_mean": 8.81,
                "positive_2024_every_seed": True},
            "shipping_seed_42": {"2022": 0.07, "2023": 5.38, "2024": 2.52,
                                 "monthly_block_win_rate": 0.60},
            "not_shipped": ("the platoon-only arm has a stronger 2024 (+4.42 mean, +9.45 "
                            "at seed 42, bootstrap CI low +1.78) but loses 2023 by 0.17 "
                            "at the shipping seed; the no-losing-fold gate has been "
                            "overridden twice and lost twice"),
        },
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
