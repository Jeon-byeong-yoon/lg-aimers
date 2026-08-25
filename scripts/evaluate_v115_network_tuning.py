"""V115: tune the network, which is brand new and entirely un-fitted.

Every hyperparameter in V114's network was a first guess: 6 epochs, hidden
(256, 128), dropout 0.15, 24-dimensional pitcher and batter embeddings. That is
exactly the situation that paid off three times already — V96 (+11.77), V105
(+7.09) and V106 (+3.10) all came from refitting parameters nobody had fitted for
the current model. The network now carries 0.20 of the blend.

The strongest single clue is the loss curve, which was still falling at the last
epoch (0.6926 -> 0.6810): the model is undertrained, not overfitted. So epochs come
first, then capacity, then embedding width, then seed averaging.

Seed averaging deserves its own note. V99 rejected it for the Form HGB, but a
gradient-boosted ensemble of 500 trees is already an average while a single neural
network is one draw from a much wider distribution, so the variance available to
remove is far larger here.

Ranked by the 2024 paired pitcher bootstrap. V110 established that candidates
sharing out-of-fold predictions share most of their noise, so the trend across
variants is better determined than any single absolute interval.
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
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import blend, bootstrap
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


OUTPUT = Path("artifacts/v115_network_tuning_metrics.json")
PREDICTIONS = Path("artifacts/v115_network_tuning_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BASE_WEIGHTS = (0.19, 0.40, 0.21, 0.20)
WIDE_EMBEDDINGS = [(name, 48 if name in ("pitcher_id", "batter_id") else width)
                   for name, width in EMBEDDING_SPECS]
VARIANTS = {
    "base6": {"epochs": 6, "seeds": (42,)},
    "epochs12": {"epochs": 12, "seeds": (42,)},
    "epochs20": {"epochs": 20, "seeds": (42,)},
    "epochs12_wide": {"epochs": 12, "seeds": (42,), "hidden": (512, 256)},
    "epochs12_bigemb": {"epochs": 12, "seeds": (42,), "specs": WIDE_EMBEDDINGS},
    "epochs12_seeds3": {"epochs": 12, "seeds": (42, 1004, 2024)},
}
NETWORK_WEIGHTS = (0.20, 0.26, 0.32)
ROUNDS = 3000
SEED = 20260825
P = 100000.0 / 0.25


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
    targets = y.to_numpy()

    networks, standalone = {}, {}
    for name, config in VARIANTS.items():
        specs = config.get("specs", EMBEDDING_SPECS)
        hidden = config.get("hidden", (256, 128))
        networks[name], standalone[name] = {}, {}
        for year in YEARS:
            started = time.time()
            train_mask = (raw_frame["season"] < year).to_numpy()
            valid_mask = (raw_frame["season"] == year).to_numpy()
            vocabularies = build_vocabularies(identity.loc[train_mask], specs)
            statistics = numeric_statistics(
                identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64))
            spec = cardinalities(vocabularies, specs)
            cat_train = encode_categorical(identity.loc[train_mask], vocabularies, specs)
            cat_valid = encode_categorical(identity.loc[valid_mask], vocabularies, specs)
            num_train = encode_numeric(
                identity.loc[train_mask, numeric_columns].to_numpy(dtype=np.float64),
                statistics)
            num_valid = encode_numeric(
                identity.loc[valid_mask, numeric_columns].to_numpy(dtype=np.float64),
                statistics)
            draws = []
            for seed in config["seeds"]:
                model = train(cat_train, num_train, targets[train_mask], spec,
                              epochs=config["epochs"], seed=seed, hidden=hidden,
                              verbose=False)
                draws.append(predict(model, cat_valid, num_valid))
            prediction = np.mean(draws, axis=0)
            networks[name][str(year)] = prediction
            target = targets[valid_mask].astype(float)
            rate = target.mean()
            standalone[name][str(year)] = float(
                100000 * (1 - ((prediction - target) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {name} {year}: standalone {standalone[name][str(year)]:8.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    v114 = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    v114 = v114["networks"]["without_season"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASE_WEIGHTS, oof, logistic, form, context, v114, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for name in VARIANTS:
        for w_network in NETWORK_WEIGHTS:
            spare = 1.0 - 0.40 - 0.21 - w_network
            if spare < 0:
                continue
            label = f"{name}_net{w_network:.2f}"
            candidate = np.clip(
                blend((round(spare, 4), 0.40, 0.21, w_network), oof, logistic,
                      form, context, networks[name], raw_frame)
                + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["weights"] = {"v17": round(spare, 4), "form": 0.40,
                                  "context": 0.21, "network": w_network}
            results[label] = metrics
        print(f"{name} evaluated", flush=True)

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
        "experiment": "V115_network_tuning",
        "baseline": "V114 (0.19 / 0.40 / 0.21 / 0.20, 6 epochs, hidden 256-128)",
        "baseline_public_score": 1002.5719943737,
        "loss_curve_note": (
            "V114's training loss was still falling at epoch 6 (0.6926 -> 0.6810), "
            "so the network was undertrained rather than overfitted."
        ),
        "variants": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                         for kk, vv in v.items() if kk != "specs"}
                     for k, v in VARIANTS.items()},
        "network_standalone_skill": standalone,
        "network_weights": list(NETWORK_WEIGHTS),
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seeds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"networks": networks}, PREDICTIONS, compress=3)

    print("\nstandalone skill by variant:")
    for name, values in standalone.items():
        print(f"  {name:18s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print("\ntop 18 by 2024 bootstrap lower bound (vs V114):")
    print(f"{'candidate':>28} {'v17':>5} {'CIlo':>7} {'mean':>7} {'2022':>7} "
          f"{'2023':>7} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"])[:18]:
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>28} {r['weights']['v17']:5.2f} {b['ci95_low']*P:7.1f} "
              f"{b['mean']*P:7.1f} {s['2022']['mean']*P:7.1f} {s['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
