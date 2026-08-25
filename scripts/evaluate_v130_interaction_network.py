"""V130: the factorization network as a sixth blend component.

Rationale is in `interaction_network_v130`: every existing component reaches an
interaction only implicitly -- trees by stacking splits, the V111 network by asking a
ReLU stack to imitate a product -- while control success is close to pitcher command
times situation difficulty by nature. This model computes the pairwise field products
directly and feeds only those to its dense stack, so it is a different function class
rather than a variation on one already present.

That distinction is the whole point, and V123 showed why. `onehot` CatBoost beat the
incumbent on all three seasons standalone and still lost 2022 in the blend, because a
component that is individually stronger but more correlated with the others contributes
less than a weaker, less correlated one. So the number to watch here is not the
standalone skill but the blend movement, and a mediocre standalone would not by itself
disqualify it.

Three latent dimensions are compared, since the pairwise term's capacity is set by that
one number, and the component is entered at three weights funded from Form -- the only
component still large enough to pay, now that v17 is retired and CatBoost sits at its
own laddered peak.
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
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v130c_interaction_network_metrics.json")
PREDICTIONS = Path("artifacts/v130c_interaction_network_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
# V122: v17 / form / context / network / catboost
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
NAMES6 = ("v17", "form", "context", "network", "catboost", "interaction")
LATENTS = (4, 6, 8)
EPOCHS = 8
COMPONENT_WEIGHTS = (0.05, 0.08, 0.11)
CATBOOST_SOURCE = "no_te_strong"
P = 100000.0 / 0.25


def blend6(weights, parts, oof, frame):
    raw = {}
    for year in YEARS:
        key = str(year)
        raw[year] = sum(w * parts[n][key] for w, n in zip(weights, NAMES6))
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, targets, predictions = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            targets.append(oof[str(history)]["target"].astype(float))
            predictions.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(targets) - np.concatenate(predictions)
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
    numeric_columns = [c for c in model_columns
                       if c not in categorical_columns and c != "season"]
    identity = features.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    inet.assert_numeric(identity, numeric_columns)
    print(f"{len(numeric_columns)} numeric, {len(categorical_columns)} fields",
          flush=True)

    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    predictions, standalone = {}, {}
    for latent in LATENTS:
        inet.LATENT = latent
        inet.FIELD_SPECS = [(column, latent) for column in inet.FIELDS]
        key = f"latent{latent}"
        predictions[key], standalone[key] = {}, {}
        for year in YEARS:
            started = time.time()
            train_mask = season < year
            valid_mask = season == year
            train_frame = identity.loc[train_mask]
            vocabularies = inet.build_vocabularies(train_frame)
            statistics = inet.numeric_statistics(
                train_frame[numeric_columns].to_numpy(dtype=np.float64))
            spec = inet.field_cardinalities(vocabularies)
            model = inet.train(
                inet.encode_categorical(train_frame, vocabularies),
                inet.encode_numeric(
                    train_frame[numeric_columns].to_numpy(dtype=np.float64), statistics),
                targets[train_mask], spec, epochs=EPOCHS, verbose=False)
            valid_frame = identity.loc[valid_mask]
            prediction = inet.predict(
                model,
                inet.encode_categorical(valid_frame, vocabularies),
                inet.encode_numeric(
                    valid_frame[numeric_columns].to_numpy(dtype=np.float64), statistics))
            predictions[key][str(year)] = prediction
            actual = targets[valid_mask].astype(float)
            rate = actual.mean()
            standalone[key][str(year)] = float(
                100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {key} {year}: standalone {standalone[key][str(year)]:8.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    zero = {str(year): 0.0 for year in YEARS}
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend6(BASE + (0.0,), dict(common, interaction=zero), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    # OOF predictions are stored in season order; the blend consumes them keyed by
    # season, so alignment is by construction.
    results = {}
    for key in predictions:
        aligned = {str(year): predictions[key][str(year)] for year in YEARS}
        for weight in COMPONENT_WEIGHTS:
            form_weight = round(BASE[1] - weight, 4)
            if form_weight <= 0:
                continue
            weights = (BASE[0], form_weight) + BASE[2:] + (round(weight, 4),)
            label = f"{key}_w{weight:.2f}"
            candidate = np.clip(
                blend6(weights, dict(common, interaction=aligned), oof, raw_frame)
                + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["weights"] = {n: w for n, w in zip(NAMES6, weights)}
            results[label] = metrics

    def passes(label):
        r = results[label]
        s = r["season_bootstrap"]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and s["2022"]["mean"] > -1e-5 and s["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [l for l in results if passes(l)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V130c_interaction_network_small_latent",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27)",
        "rationale": (
            "Every existing component reaches an interaction implicitly; this one "
            "computes the pairwise field products directly and feeds only those to its "
            "dense stack, so it is a different function class rather than a variation."
        ),
        "latents": list(LATENTS),
        "epochs": EPOCHS,
        "component_weights": list(COMPONENT_WEIGHTS),
        "standalone_skill": standalone,
        "results": {k: {"weights": v["weights"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "season_bootstrap": v["season_bootstrap"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)

    print("\nstandalone skill (V111 network: 2022 2162, 2023 -387, 2024 616):")
    for key, values in standalone.items():
        print(f"  {key:10s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print(f"\n{'candidate':>18} {'CIlo':>7} {'mean':>7} {'2022':>7} {'2023':>8} "
          f"{'sum':>8} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"]):
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        total = sum(s[str(v)]["mean"] * P for v in YEARS)
        print(f"{label:>18} {b['ci95_low']*P:7.2f} {b['mean']*P:7.2f} "
              f"{s['2022']['mean']*P:7.2f} {s['2023']['mean']*P:8.2f} {total:8.2f} "
              f"{r['monthly_block_win_rate']:7.0%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
