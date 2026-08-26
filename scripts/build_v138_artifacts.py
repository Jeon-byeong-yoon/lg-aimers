"""Build V138: the factorization network takes a third of the Context slot.

V135 ran twelve ordered pairwise weight transfers and eighteen of them touch Context;
all eighteen agree it is over-weighted, up to +32.61 on the three-season average when
weight leaves it and down to -49.09 when weight arrives. V136 then showed the axis
completely blocked: reducing Context by even one point sends 2024 negative and leaves
the monthly block win rate at 65-70% against a floor of 75%.

The diagnosis was that Context is the next v17 -- second-worst on 2023 at -1333,
effectively tied with the layer V120 retired for that reason, and highly correlated with
everything else -- but that it still earns diversity on 2024, where every inter-component
correlation drops (0.778-0.875 against 0.884-0.947 on 2022). So the weight cannot simply
be moved to components that already exist.

V120 solved the same shape of problem by *replacing* v17 rather than emptying it. The
replacement here is the factorization network, which V130 rejected as a sixth component
on its standalone skill and which nevertheless fixes exactly what is wrong with Context:

    slot content        2022 skill  corr   2023 skill  corr   2024 skill  corr
    context                   2224  0.919        -1333  0.903         725  0.817
    factorization L8          2046  0.876         -854  0.797         147  0.710

Latent 8 is the weakest of the three sizes tested and the least correlated, and it is the
one that passes. Latent 6 scores better standalone on every season yet fails, because its
higher correlation lets 2024 go negative. What the slot rewards is independence, not
skill -- the V123 lesson (`onehot` individually stronger on all three seasons, worse in
the blend) running in the other direction.

The handover is 0.07 of Context's 0.21, chosen by the pre-registered rule -- the largest
three-season lower bound among candidates clearing the gate. Latent 8 keeps 2024 positive
through 0.07 (+0.64) and crosses at 0.08 (-0.05), so this sits at the boundary the
per-season condition draws, not past it.

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


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
LATENT = 8
EPOCHS = 8
# Stated rather than left to the default. V130c produced the validated predictions with
# `train`'s default of (128, 64), and an earlier build recorded (256, 128) in the bundle
# while training at (128, 64), which failed at load time on a size mismatch.
HIDDEN = (128, 64)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"


def main():
    total = W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST + W_FACTORIZATION
    assert abs(total - 1.0) < 1e-12, total

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
    inet.assert_numeric(identity, numeric_columns)

    # The evaluation sweep set these globals per latent size; the fitted module now
    # reads its width from the spec, but `field_cardinalities` still consults them.
    inet.LATENT = LATENT
    inet.FIELD_SPECS = [(column, LATENT) for column in inet.FIELDS]

    vocabularies = inet.build_vocabularies(identity)
    statistics = inet.numeric_statistics(
        identity[numeric_columns].to_numpy(dtype=np.float64))
    spec = inet.field_cardinalities(vocabularies)
    started = time.time()
    model = inet.train(
        inet.encode_categorical(identity, vocabularies),
        inet.encode_numeric(identity[numeric_columns].to_numpy(dtype=np.float64),
                            statistics),
        y.to_numpy(), spec, epochs=EPOCHS, hidden=HIDDEN, verbose=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"factorization network trained on {len(identity):,} rows, "
          f"{len(numeric_columns)} numeric and {len(inet.FIELDS)} fields at latent "
          f"{LATENT}, {parameters:,} parameters [{time.time() - started:.0f}s]",
          flush=True)

    bundle = inet.state_bundle(model, vocabularies, statistics, numeric_columns, spec)
    assert tuple(bundle["hidden"]) == HIDDEN, bundle["hidden"]
    bundle["latent"] = LATENT
    bundle["epochs"] = EPOCHS
    bundle["feature_shrinkage"] = FEATURE_SHRINKAGE
    path = Path("artifacts/v138_factorization.joblib")
    joblib.dump(bundle, path, compress=3)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.2f} MiB)")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_oof = context_oof["no_matchup_hte"]["context"]
    network_oof = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network_oof = network_oof["networks"]["without_season"]
    catboost_oof = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    factorization_oof = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    factorization_oof = factorization_oof[f"latent{LATENT}"]

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

    calibration = Path("artifacts/v138_calibration.joblib")
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
    print(f"Saved {calibration}, shift={shift:.12f} (V122 was -0.010340829144)")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} "
          f"network={W_NETWORK} catboost={W_CATBOOST} "
          f"factorization={W_FACTORIZATION}")

    Path("artifacts/v138_build_summary.json").write_text(json.dumps({
        "version": "V138",
        "baseline": "V122 (1047.03653)",
        "change": ("factorization network at 0.07 taken from Context, "
                   "Context 0.21 -> 0.14"),
        "weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                    "network": W_NETWORK, "catboost": W_CATBOOST,
                    "factorization": W_FACTORIZATION},
        "global_shift": shift,
        "latent": LATENT, "epochs": EPOCHS, "hidden": list(HIDDEN),
        "parameters": int(parameters),
        "validation": {
            "instrument": "three-season equally weighted average (V131 correction)",
            "three_season_average_points": 25.08,
            "three_season_ci95_low_points": 18.72,
            "season_2022_points": 6.35, "season_2023_points": 68.26,
            "season_2024_points": 0.64,
            "monthly_block_win_rate": 0.80,
            "boundary": ("latent 8 keeps 2024 positive through handover 0.07 and "
                         "crosses at 0.08 (-0.05), so this sits at the boundary the "
                         "per-season condition draws."),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
