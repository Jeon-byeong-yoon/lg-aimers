"""Build V114: V106 plus the entity-embedding network as a fourth component.

Weights 0.19 v17 / 0.40 Form / 0.21 Context / 0.20 network, selected in V113 by the
2024 paired pitcher bootstrap lower bound (+13.0, mean +24.5) with a 95% monthly
block win rate.

The network omits `season` from its numeric inputs. Passing it through and clipping
it at 2024 are identical on every validation fold, so the extrapolation to 2025
cannot be measured; and because the calibration shift is a constant fitted where
`season` was in range, an extrapolated shift would not be absorbed by it. Dropping
it costs about 3 points of local CI lower bound and improves 2022 and the block win
rate, so the risk is not worth carrying.

Unseen identifiers map to a dedicated index. That path is well exercised: the
chronological folds saw 13.8%-19.9% unseen pitchers and the network still scored
+645 standalone on 2024. The final model knows all six seasons, so 2025's unseen
rate should be lower than the folds'.

Run with the server-mirror interpreter. Only weights are saved, as numpy arrays in
a plain dict, so the artifact carries no torch or numpy pickle internals.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from contextual_trackman_v24 import prepare_context_trackman
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, state_bundle, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v38_lr_grid import v31_form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT, W_NETWORK = 0.19, 0.40, 0.21, 0.20
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
EPOCHS = 6
NETWORK_VARIANT = "without_season"


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
    categorical_columns = [column for column, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in model_columns
                       if c not in categorical_columns and c != "season"]
    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)

    vocabularies = build_vocabularies(identity)
    statistics = numeric_statistics(
        identity[numeric_columns].to_numpy(dtype=np.float64))
    spec = cardinalities(vocabularies)
    started = time.time()
    model = train(
        encode_categorical(identity, vocabularies),
        encode_numeric(identity[numeric_columns].to_numpy(dtype=np.float64), statistics),
        y.to_numpy(), spec, epochs=EPOCHS)
    print(f"network trained on {len(identity):,} rows, {len(numeric_columns)} numeric "
          f"and {len(categorical_columns)} embedded inputs [{time.time() - started:.0f}s]",
          flush=True)

    bundle = state_bundle(model, vocabularies, statistics, numeric_columns, spec)
    bundle["variant"] = NETWORK_VARIANT
    bundle["feature_shrinkage"] = FEATURE_SHRINKAGE
    network_path = Path("artifacts/v114_network.joblib")
    joblib.dump(bundle, network_path, compress=3)
    print(f"Saved {network_path} ({network_path.stat().st_size / 2**20:.2f} MiB)")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_oof = context_oof["no_matchup_hte"]["context"]
    network_oof = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network_oof = network_oof["networks"][NETWORK_VARIANT]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * network_oof[key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    calibration = Path("artifacts/v114_calibration.joblib")
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
                          "network": W_NETWORK},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, calibration, compress=3)
    print(f"Saved {calibration}, shift={shift:.12f} (V106 was -0.010057027756)")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} network={W_NETWORK}")


if __name__ == "__main__":
    main()
