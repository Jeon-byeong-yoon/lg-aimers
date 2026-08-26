"""Build V144: the pre-registered gate-violation probe.

This is the one submission in the project that deliberately fails its own gate, and the
reason is written down in V143 before any result was seen. Nine experiments -- V123,
V124, V127, V132, V135, V136, V139, V141, V142 -- were rejected with a single signature:
2023 improves substantially, another fold goes negative, the monthly block win rate falls
short. All the remaining headroom lives in that trade, and the four leaderboard results
cannot settle it, because every one of them concerned a candidate whose folds agreed in
sign. They established that the magnitude reading is unreliable. They say nothing about a
fold conflict.

The candidate is `onehot` CatBoost at weight 0.36, funded from Form. `onehot` raises
`one_hot_max_size` to 16, so the twelve count states, eight base states, thirteen team
codes and the hand columns are one-hot encoded and ordered target statistics are left to
`pitcher_id` and `batter_id` alone. V123 found it individually stronger than the
incumbent on all three seasons -- 2403 / -765 / 797 against 2397 / -844 / 752 -- and
rejected it because being more correlated with the rest cost more in the blend than the
extra skill was worth.

Against V138 it reads:

    three-season average   +8.43      2022  -1.86
    weakest season         -1.86      2023  +22.57
    monthly blocks           80%      2024   +4.58

It clears three of the four conditions, blocks included, and its two readings point in
opposite directions. The objection comes from 2022 alone, which is the weakest veto
available: 2022 receives no calibration at all, because no earlier season exists to fit
residuals on, so the raw blend is used (V126 measured the level through that path). It is
also the oldest fold and the furthest from 2025, so its measurement conditions least
resemble deployment.

The `no_env` and `no_season` candidates carry larger 2023 gains -- +77.52 and +72.93 --
but both have 2022 *and* 2024 negative, so a loss would not say which fold was right.
Confining the objection to one fold is what makes this a measurement.

Pre-registered interpretation, from V143:
  * gain above +2 points -> the 2022 veto is too strict; reopen the family
  * loss                 -> the veto is validated; keep V138 and close the line
  * within +/-2          -> inconclusive, counted as validated

Run with the server-mirror interpreter.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v38_lr_grid import v31_form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT = 0.00, 0.23, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.36, 0.07
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
NETWORK_VARIANT = "without_season"
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "onehot"
# `one_hot_max_size=16` one-hot encodes the count states, base states, team codes and
# hand columns, leaving ordered target statistics for `pitcher_id` and `batter_id`.
CATBOOST_CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02,
                       l2_leaf_reg=12.0, one_hot_max_size=16)
# Columns whose value space is produced by our own feature code, so a 2025 value
# outside the training set is suspicious. The identifier columns are deliberately
# excluded: the official test sample already carries pitcher ids above the training
# maximum, so unseen entities there are expected rather than a defect.
CLOSED_VOCABULARY = ("count_state", "base_state", "hand_matchup", "pitcher_hand",
                     "batter_hand", "top_bottom", "game_type")


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encoding_columns]
               + [c for c in categorical if c not in model_columns])

    work = features[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    print(f"catboost frame: {work.shape[0]:,} rows x {work.shape[1]} columns "
          f"({len(categorical)} categorical, {len(encoding_columns)} encodings dropped)",
          flush=True)

    started = time.time()
    model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=42, verbose=0,
                               thread_count=6, cat_features=categorical,
                               allow_writing_files=False)
    model.fit(work, y.to_numpy())
    print(f"catboost trained on all six seasons [{time.time() - started:.0f}s]", flush=True)

    model_path = Path("artifacts/v144_catboost.cbm")
    model.save_model(str(model_path))
    print(f"Saved {model_path} ({model_path.stat().st_size / 2**20:.2f} MiB)")

    # Saved so inference can rebuild the frame identically and, more importantly, so a
    # silent dtype drift (which would turn every categorical value into an unseen one
    # and quietly degrade the whole component) fails loudly instead.
    values = {name: sorted(work[name].unique().tolist()) for name in categorical}
    meta_path = Path("artifacts/v144_catboost_meta.joblib")
    joblib.dump({
        "columns": columns,
        "categorical": categorical,
        "numeric": numeric,
        "categorical_values": values,
        "closed_vocabulary": [c for c in CLOSED_VOCABULARY if c in categorical],
        "config": CATBOOST_CONFIG,
        "source": CATBOOST_SOURCE,
        "feature_shrinkage": FEATURE_SHRINKAGE,
    }, meta_path, compress=3)
    print(f"Saved {meta_path} ({meta_path.stat().st_size / 2**20:.2f} MiB)")
    print("  categorical cardinalities: " + ", ".join(
        f"{name}={len(v)}" for name, v in values.items()))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_oof = context_oof["no_matchup_hte"]["context"]
    network_oof = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network_oof = network_oof["networks"][NETWORK_VARIANT]
    catboost_oof = joblib.load(
        "artifacts/v123_catboost_capacity_predictions.joblib")["predictions"][
            CATBOOST_SOURCE]
    factorization_oof = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"][
            FACTORIZATION_SOURCE]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
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

    calibration = Path("artifacts/v144_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": 0.75,
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], 500)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"], "weight": 0.25,
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"], 300)},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, calibration, compress=3)
    print(f"Saved {calibration}, shift={shift:.12f} (V138 was -0.010764370502)")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} "
          f"network={W_NETWORK} catboost={W_CATBOOST} "
          f"factorization={W_FACTORIZATION}")

    Path("artifacts/v144_build_summary.json").write_text(json.dumps({
        "version": "V144",
        "baseline": "V138 (1050.5510725821)",
        "change": ("onehot CatBoost replaces no_te_strong and rises to 0.36, funded "
                   "from Form; every other weight unchanged from V138"),
        "weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                    "network": W_NETWORK, "catboost": W_CATBOOST,
                    "factorization": W_FACTORIZATION},
        "global_shift": shift,
        "catboost_config": CATBOOST_CONFIG,
        "catboost_columns": len(columns),
        "catboost_categorical": categorical,
        "encodings_dropped": encoding_columns,
        "gate_status": "deliberately violated once; see V143 for the pre-registration",
        "validation": {
            "instrument": "three-season equally weighted average, V138 baseline",
            "three_season_average_points": 8.43,
            "min_season_points": -1.86,
            "season_2022_points": -1.86, "season_2023_points": 22.57,
            "season_2024_points": 4.58,
            "monthly_block_win_rate": 0.80,
            "failing_condition": ("2022 mean below zero; 2022 is the only fold that "
                                 "receives no calibration, so the raw blend is used"),
        },
        "pre_registered_interpretation": {
            "gain_above_2_points": "the 2022 veto is too strict; reopen the family",
            "loss": "the veto is validated; keep V138 and close the line",
            "within_2_points": "inconclusive, counted as validated",
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
