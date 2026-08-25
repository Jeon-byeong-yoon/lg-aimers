"""V111: add an entity-embedding network as a fourth blend component.

The ensemble is entirely scikit-learn trees plus one logistic model, so every
component sees a pitcher only as scalar summaries. This network learns a vector per
pitcher, batter and team, which is a different function class — the property that
made the logistic (V17) and joint-feature (V25) additions worth their weight.

The network is trained chronologically on the same folds as every other component:
2022 from 2019-2021, 2023 from 2019-2022, 2024 from 2019-2023. Its features are the
Form model's 105 columns, which already include the in-season reconstruction at
shrinkage 20, so the comparison isolates the model class rather than the inputs.

It enters as a fourth term, taken out of the V17 layer's share since that is the
weakest component at 0.19:

    (0.19 - n) * v17 + 0.64 * form + 0.17 * context + n * network

Ranked by the 2024 paired pitcher bootstrap. Candidates built from the same
out-of-fold predictions share most of their noise, so the trend across network
weights is far better determined than any single absolute interval — that is what
the V110 ladder established.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, state_bundle,
    train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v111_embedding_network_metrics.json")
PREDICTIONS = Path("artifacts/v111_embedding_network_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
W_FORM, W_CONTEXT, W_V17 = 0.64, 0.17, 0.19
NETWORK_WEIGHTS = (0.04, 0.08, 0.12, 0.19)
EPOCHS = 6
ROUNDS = 3000
SEED = 20260825
P = 100000.0 / 0.25


def bootstrap(frame, base, candidate, mask, rounds=ROUNDS, seed=SEED):
    part = frame.loc[mask]
    target = part["target"].to_numpy()
    gain = (base[mask] - target) ** 2 - (candidate[mask] - target) ** 2
    grouped = pd.DataFrame({"p": part["pitcher_id"].to_numpy(), "g": gain}) \
        .groupby("p")["g"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(sums), size=(rounds, len(sums)))
    draws = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return {"mean": float(gain.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975))}


def blend_and_calibrate(w_v17, w_network, oof, logistic, form, context, network, frame):
    raw = {}
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        raw[year] = (w_v17 * v17 + W_FORM * form[key] + W_CONTEXT * context[key]
                     + w_network * network[key])
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            target.append(oof[str(history)]["target"].astype(float))
            prediction.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(target) - np.concatenate(prediction)
        train_frame = frame.loc[index]
        valid_frame = frame.loc[oof[str(year)]["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        pieces.append(np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1))
    return np.concatenate(pieces)


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
    numeric_columns = [c for c in model_columns if c not in categorical_columns]
    print(f"network inputs: {len(categorical_columns)} embedded, "
          f"{len(numeric_columns)} numeric", flush=True)

    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)

    network_oof, bundles = {}, {}
    for year in YEARS:
        started = time.time()
        train_mask = (raw_frame["season"] < year).to_numpy()
        valid_mask = (raw_frame["season"] == year).to_numpy()
        vocabularies = build_vocabularies(identity.loc[train_mask])
        statistics = numeric_statistics(
            identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64))
        cat_train = encode_categorical(identity.loc[train_mask], vocabularies)
        cat_valid = encode_categorical(identity.loc[valid_mask], vocabularies)
        num_train = encode_numeric(
            identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64), statistics)
        num_valid = encode_numeric(
            identity.loc[valid_mask, numeric_columns].to_numpy(dtype=np.float64), statistics)
        spec = cardinalities(vocabularies)
        print(f"  {year}: training on {train_mask.sum():,} rows", flush=True)
        model = train(cat_train, num_train, y.to_numpy()[train_mask], spec, epochs=EPOCHS)
        network_oof[str(year)] = predict(model, cat_valid, num_valid)
        bundles[str(year)] = state_bundle(
            model, vocabularies, statistics, numeric_columns, spec)
        target = y.to_numpy()[valid_mask].astype(float)
        rate = target.mean()
        skill = 100000 * (1 - ((network_oof[str(year)] - target) ** 2).mean()
                          / (rate * (1 - rate)))
        print(f"  {year}: standalone skill {skill:8.0f}  "
              f"pred mean {network_oof[str(year)].mean():.4f} vs target {rate:.4f}  "
              f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    aligned = {str(year): network_oof[str(year)] for year in YEARS}
    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend_and_calibrate(W_V17, 0.0, oof, logistic, form, context, aligned, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for w_network in NETWORK_WEIGHTS:
        label = f"network{w_network:.2f}"
        candidate = np.clip(
            blend_and_calibrate(W_V17 - w_network, w_network, oof, logistic,
                                form, context, aligned, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["v17_weight"] = round(W_V17 - w_network, 4)
        results[label] = metrics
        print(f"evaluated {label} (v17 {metrics['v17_weight']:.2f})", flush=True)

    standalone = {}
    for year in YEARS:
        target = y.to_numpy()[(raw_frame["season"] == year).to_numpy()].astype(float)
        rate = target.mean()
        standalone[str(year)] = float(
            100000 * (1 - ((network_oof[str(year)] - target) ** 2).mean()
                      / (rate * (1 - rate))))

    OUTPUT.write_text(json.dumps({
        "experiment": "V111_entity_embedding_network",
        "baseline": "V106 (0.19 / 0.64 / 0.17, drift shrinkage 3, w 0.10)",
        "baseline_public_score": 973.0643764994,
        "architecture": {
            "embeddings": dict(EMBEDDING_SPECS),
            "hidden": [256, 128], "dropout": 0.15, "epochs": EPOCHS,
            "optimiser": "AdamW + OneCycleLR", "inference_device": "cpu",
        },
        "numeric_feature_count": len(numeric_columns),
        "network_standalone_skill": standalone,
        "network_weights": list(NETWORK_WEIGHTS),
        "results": results,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "pretrained_weights_used": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"network_oof": network_oof, "bundles": bundles},
                PREDICTIONS, compress=3)

    print(f"\nnetwork standalone skill: {standalone}")
    print("gains vs V106 baseline, by 2024 bootstrap:")
    print(f"{'candidate':>14} {'v17':>6} {'CIlo':>7} {'mean':>7} {'CIhi':>7} "
          f"{'2022':>7} {'2023':>7} {'blocks':>7}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["mean"]):
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>14} {r['v17_weight']:6.2f} {b['ci95_low']*P:7.1f} "
              f"{b['mean']*P:7.1f} {b['ci95_high']*P:7.1f} {s['2022']['mean']*P:7.1f} "
              f"{s['2023']['mean']*P:7.1f} {r['monthly_block_win_rate']:7.1%}")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
