"""Build V117: V114 plus CatBoost as a fifth blend component.

V116 tested four CatBoost variants at three blend weights and eleven of the twelve
candidates cleared the pre-registered gate. The promoted one, `no_te_strong_cb0.14`,
had a 2024 paired pitcher bootstrap lower bound of +4.1 (mean +9.9) and improved
2022 by +12.7 and 2023 by +71.5. Improving all three seasons at once has happened
only twice before in roughly a hundred experiments, and both times it was a
gradient-boosting library.

`no_te` drops the nine `te_*`/`hte_*` columns so CatBoost derives its own ordered
target statistics instead of consuming ours. It beat `full` at every weight and both
configs, which is the V94b finding arriving from the other direction: our hand-built
encodings carry league-level contamination that an ordered scheme avoids.

Two things about this artifact differ from every earlier one.

First, the model is saved in CatBoost's native `cbm` format rather than pickled.
`cbm` is a self-describing binary that carries no numpy or Python object graph, so
the numpy 1.x/2.x `BitGenerator` incompatibility that cost a submission on V93
cannot arise for it at all. The packaging gate loads it anyway.

Second, `season` stays in the feature set even though 2025 lies outside the training
range. For a tree that is not extrapolation: an out-of-range value simply falls in
the terminal bin, so 2025 is scored as 2024 was. And it is exactly what validation
measured, because every chronological fold predicted a season strictly later than
any it trained on. Unlike the network, where an unbounded linear response made the
same choice unsafe, the validated behaviour here transfers verbatim.

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


W_V17, W_FORM, W_CONTEXT, W_NETWORK, W_CATBOOST = 0.05, 0.40, 0.21, 0.20, 0.14
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
NETWORK_VARIANT = "without_season"
CATBOOST_SOURCE = "no_te_strong"
CATBOOST_CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
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

    model_path = Path("artifacts/v117_catboost.cbm")
    model.save_model(str(model_path))
    print(f"Saved {model_path} ({model_path.stat().st_size / 2**20:.2f} MiB)")

    # Saved so inference can rebuild the frame identically and, more importantly, so a
    # silent dtype drift (which would turn every categorical value into an unseen one
    # and quietly degrade the whole component) fails loudly instead.
    values = {name: sorted(work[name].unique().tolist()) for name in categorical}
    meta_path = Path("artifacts/v117_catboost_meta.joblib")
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
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * network_oof[key] + W_CATBOOST * catboost_oof[key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    calibration = Path("artifacts/v117_calibration.joblib")
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
                          "network": W_NETWORK, "catboost": W_CATBOOST},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, calibration, compress=3)
    print(f"Saved {calibration}, shift={shift:.12f} (V114 was -0.010123350182)")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} "
          f"network={W_NETWORK} catboost={W_CATBOOST}")

    Path("artifacts/v117_build_summary.json").write_text(json.dumps({
        "version": "V117",
        "baseline": "V114 (1002.5719943737)",
        "change": "CatBoost no_te/strong added at 0.14, taken from the V17 share",
        "weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                    "network": W_NETWORK, "catboost": W_CATBOOST},
        "global_shift": shift,
        "catboost_config": CATBOOST_CONFIG,
        "catboost_columns": len(columns),
        "catboost_categorical": categorical,
        "encodings_dropped": encoding_columns,
        "validation": {
            "instrument": "2024 paired pitcher-cluster bootstrap",
            "ci95_low_points": 4.1, "mean_points": 9.9,
            "season_2022_points": 12.7, "season_2023_points": 71.5,
            "monthly_block_win_rate": 0.85,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
