"""V112: settle the network's `season` input, then re-optimise all four weights.

Two things V111 left open.

First, `season` reaches the network as a standardised number, and 2025 lands at
+2.83 against a training maximum of +2.12. Trees cannot extrapolate and simply
treat 2025 as 2024; a network extrapolates, which may track the league decline or
may drift unboundedly. Crucially this cannot be chosen locally: every validation
year is 2024 or earlier, so passing `season` through unchanged and clipping it at
2024 produce *identical* predictions on every fold. Only dropping it is
measurable. So the test is whether dropping `season` costs anything locally — if
not, dropping it removes the 2025 extrapolation risk at no measured cost, which is
the safer artifact.

Second, V111's ladder took the network's weight out of the V17 layer only and rose
monotonically until v17 hit 0.00 — a boundary, so the optimum is probably outside
it. Here all four weights move together.

Ranking is by the 2024 paired pitcher bootstrap lower bound. V110 established that
candidates sharing out-of-fold predictions share most of their noise, so the trend
across a grid is far better determined than any single absolute interval.
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


OUTPUT = Path("artifacts/v112_network_weight_metrics.json")
PREDICTIONS = Path("artifacts/v112_network_weight_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
EPOCHS = 6
SEASON_VARIANTS = ("with_season", "without_season")
NETWORK_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30)
FORM_WEIGHTS = (0.48, 0.56, 0.64)
CONTEXT_WEIGHTS = (0.13, 0.17)
BASELINE = (0.19, 0.64, 0.17, 0.0)
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


def blend(weights, oof, logistic, form, context, network, frame):
    w_v17, w_form, w_context, w_network = weights
    raw = {}
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        total = w_v17 * v17 + w_form * form[key] + w_context * context[key]
        if w_network:
            total = total + w_network * network[key]
        raw[year] = total
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
    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]

    numeric_sets = {
        "with_season": [c for c in model_columns if c not in categorical_columns],
        "without_season": [c for c in model_columns
                           if c not in categorical_columns and c != "season"],
    }
    for columns in numeric_sets.values():
        assert_numeric(identity, columns)

    networks, bundles, standalone = {}, {}, {}
    for variant in SEASON_VARIANTS:
        numeric_columns = numeric_sets[variant]
        networks[variant], bundles[variant], standalone[variant] = {}, {}, {}
        for year in YEARS:
            started = time.time()
            train_mask = (raw_frame["season"] < year).to_numpy()
            valid_mask = (raw_frame["season"] == year).to_numpy()
            vocabularies = build_vocabularies(identity.loc[train_mask])
            statistics = numeric_statistics(
                identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64))
            spec = cardinalities(vocabularies)
            model = train(
                encode_categorical(identity.loc[train_mask], vocabularies),
                encode_numeric(
                    identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64),
                    statistics),
                y.to_numpy()[train_mask], spec, epochs=EPOCHS, verbose=False)
            prediction = predict(
                model,
                encode_categorical(identity.loc[valid_mask], vocabularies),
                encode_numeric(
                    identity.loc[valid_mask, numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            networks[variant][str(year)] = prediction
            bundles[variant][str(year)] = state_bundle(
                model, vocabularies, statistics, numeric_columns, spec)
            target = y.to_numpy()[valid_mask].astype(float)
            rate = target.mean()
            standalone[variant][str(year)] = float(
                100000 * (1 - ((prediction - target) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {variant} {year}: standalone "
                  f"{standalone[variant][str(year)]:8.0f}  [{time.time() - started:.0f}s]",
                  flush=True)

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

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASELINE, oof, logistic, form, context, None, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for variant in SEASON_VARIANTS:
        for w_network in NETWORK_WEIGHTS:
            for w_form in FORM_WEIGHTS:
                for w_context in CONTEXT_WEIGHTS:
                    w_v17 = round(1.0 - w_form - w_context - w_network, 4)
                    if w_v17 < 0:
                        continue
                    label = (f"{variant}_net{w_network:.2f}_form{w_form:.2f}"
                             f"_ctx{w_context:.2f}")
                    candidate = np.clip(
                        blend((w_v17, w_form, w_context, w_network), oof, logistic,
                              form, context, networks[variant], raw_frame)
                        + DRIFT_WEIGHT * term, 0, 1)
                    metrics = development_metrics(reference, candidate)
                    metrics["bootstrap_2024"] = bootstrap(
                        validation_frame, baseline, candidate, mask_2024)
                    metrics["season_bootstrap"] = {
                        str(year): bootstrap(
                            validation_frame, baseline, candidate,
                            (validation_frame["season"] == year).to_numpy())
                        for year in YEARS}
                    metrics["weights"] = {"v17": w_v17, "form": w_form,
                                          "context": w_context, "network": w_network}
                    results[label] = metrics
        print(f"{variant} grid evaluated", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V112_network_weight_and_season",
        "baseline": "V106 (0.19 / 0.64 / 0.17, no network)",
        "baseline_public_score": 973.0643764994,
        "season_note": (
            "with_season and a version clipped at 2024 are identical on every "
            "validation fold, so only dropping it is locally measurable. If dropping "
            "costs nothing, drop it and the 2025 extrapolation risk disappears."
        ),
        "network_standalone_skill": standalone,
        "network_weights": list(NETWORK_WEIGHTS),
        "form_weights": list(FORM_WEIGHTS),
        "context_weights": list(CONTEXT_WEIGHTS),
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "pretrained_weights_used": False, "chronological_folds": True,
                       "fixed_seed": True, "inference_device": "cpu"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"networks": networks, "bundles": bundles}, PREDICTIONS, compress=3)

    print("\nnetwork standalone skill:")
    for variant, values in standalone.items():
        print(f"  {variant}: " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print("\ntop 20 by 2024 bootstrap lower bound (vs V106):")
    print(f"{'candidate':>44} {'v17':>5} {'CIlo':>7} {'mean':>7} {'CIhi':>7} "
          f"{'2022':>7} {'2023':>7} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"])[:20]:
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>44} {r['weights']['v17']:5.2f} {b['ci95_low']*P:7.1f} "
              f"{b['mean']*P:7.1f} {b['ci95_high']*P:7.1f} {s['2022']['mean']*P:7.1f} "
              f"{s['2023']['mean']*P:7.1f} {r['monthly_block_win_rate']:7.1%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
