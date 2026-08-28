"""V164: seed-average the two neural components, and ask them for a Brier objective.

Two things were never asked of the network and the factorization network.

**Seeds.** Multi-seed averaging was rejected twice -- V54 (five seeds over six tree
models, at the V41 baseline) and V85 -- but both were about *trees*, whose seed variance
comes only from split sampling. The two components that hold 27% of the blend today are
neural: their variance comes from initialisation, shuffling and dropout, which is a much
larger source. V115 is the only place it was ever measured on a neural component, and it
gained on every season:

    embedding network, standalone skill      2022     2023     2024
      epochs12,  seed 42                     1711     -911      439
      epochs12,  seeds 42/1004/2024          2131     -633      638
      averaging bought                       +421     +279     +199

That was measured at **12 epochs**, and the config actually deployed is **6**. `base6`
with more than one seed was never in the grid, and the factorization network (V130) has
never had a seed test at all. Both ship as single seed 42 today.

Averaging is the one intervention with a principled reason to be non-negative on every
fold: it removes variance without touching bias. V54's stated failure mode -- shrinkage
toward the mean breaking a fixed calibration -- does not apply here, because the
calibration is refit on the blend downstream of the component.

**Objective.** Both networks train on `BCEWithLogitsLoss` while the contest scores Brier.
Log loss weights confident errors far more heavily than squared error does, so the two
disagree about where to spend capacity. V73 tried a squared-error objective once, for
LightGBM, at the V41 baseline; it has never been asked of these components. One caveat
recorded up front: MSE gradients are smaller than BCE's, and the learning rate is held at
2e-3, so a loss here is partly a learning-rate result and not only an objective result.

Everything else is V161 exactly -- weights, reliability 300, calibration, drift term.
Control: seed 42 must reproduce V160's stored predictions to 0.000e+00.

Filter: no fold may lose, and at least one must gain. Ranked by weakest season, then the
three-season average. The minimum has predicted the leaderboard sign 8/8; nothing predicts
the magnitude, its ratio running 0.13 to 2.15.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "scripts")
import interaction_network_v130 as inet
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
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


OUTPUT = Path("artifacts/v164_seed_averaging_metrics.json")
PREDICTIONS = Path("artifacts/v164_seed_averaging_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT = 8
FACTORIZATION_EPOCHS = 8
FACTORIZATION_HIDDEN = (128, 64)
SEEDS = (42, 1004, 2024, 777, 999)
LADDER = (2, 3, 5)


class BrierLoss(nn.Module):
    """Squared error on the probability -- the contest metric, as a training loss."""

    def forward(self, logits, target):
        return ((torch.sigmoid(logits) - target) ** 2).mean()


def skill(prediction, actual):
    rate = actual.mean()
    return 100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate)))


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    del encoded, hierarchical
    gc.collect()
    global_prior = float(y.mean())

    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    frame = select_v2_features(add_row_features(form_raw, global_prior))
    frame = add_trackman_features(frame, trackman)
    frame = pd.concat([frame, block], axis=1)
    model_columns = v31_form_columns(frame)
    categorical_columns = [c for c, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in model_columns
                       if c not in categorical_columns and c != "season"]
    identity = frame.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)
    del frame, block, form_raw
    gc.collect()

    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]

    # network[seed][year] and factorization[seed][year]; "mse" is seed 42 under BrierLoss.
    networks, factorizations = {}, {}
    for tag in list(SEEDS) + ["mse"]:
        networks[tag], factorizations[tag] = {}, {}

    started = time.time()
    for year in YEARS:
        train_mask = season < year
        valid_mask = season == year
        train_identity = identity.loc[train_mask]
        valid_identity = identity.loc[valid_mask]
        vocabularies = build_vocabularies(train_identity)
        statistics = numeric_statistics(
            train_identity[numeric_columns].to_numpy(dtype=np.float64))
        categorical = encode_categorical(train_identity, vocabularies)
        numeric = encode_numeric(
            train_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        valid_categorical = encode_categorical(valid_identity, vocabularies)
        valid_numeric = encode_numeric(
            valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        fvalid_categorical = inet.encode_categorical(valid_identity, vocabularies)
        fvalid_numeric = inet.encode_numeric(
            valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        spec = inet.field_cardinalities(vocabularies)
        actual = targets[valid_mask].astype(float)
        del train_identity, valid_identity
        gc.collect()

        for tag in list(SEEDS) + ["mse"]:
            seed = 42 if tag == "mse" else tag
            if tag == "mse":
                nn.BCEWithLogitsLoss = BrierLoss
            try:
                net = train(categorical, numeric, targets[train_mask],
                            cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                            seed=seed, verbose=False)
                networks[tag][str(year)] = predict(net, valid_categorical, valid_numeric)
                del net
                gc.collect()
                fmodel = inet.train(categorical, numeric, targets[train_mask], spec,
                                    epochs=FACTORIZATION_EPOCHS, seed=seed,
                                    hidden=FACTORIZATION_HIDDEN, verbose=False)
                factorizations[tag][str(year)] = inet.predict(
                    fmodel, fvalid_categorical, fvalid_numeric)
                del fmodel
                gc.collect()
            finally:
                nn.BCEWithLogitsLoss = torch.nn.modules.loss.BCEWithLogitsLoss
            print(f"  {year} {str(tag):>5s}: netw {skill(networks[tag][str(year)], actual):7.0f}"
                  f"  fact {skill(factorizations[tag][str(year)], actual):7.0f}"
                  f"  [{time.time() - started:.0f}s]", flush=True)
        del categorical, numeric, valid_categorical, valid_numeric
        del fvalid_categorical, fvalid_numeric
        gc.collect()
    del identity
    gc.collect()

    # Control: seed 42 must reproduce what V161 shipped.
    stored = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    control = {}
    for name, got in (("network", networks[42]), ("factorization", factorizations[42])):
        want = stored[name][FEATURE_RELIABILITY]
        control[name] = max(float(np.abs(got[str(y_)] - want[str(y_)]).max())
                            for y_ in YEARS)
        print(f"control {name:14s} max |new - stored| = {control[name]:.3e}", flush=True)

    def average(store, count):
        return {str(y_): np.mean([store[s][str(y_)] for s in SEEDS[:count]], axis=0)
                for y_ in YEARS}

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                       )["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                          )["no_matchup_hte"]["context"]
    catboost = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                           )["catboost"]["projected"]
    v17 = {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
           for v in YEARS}
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(network, factorization):
        parts = {"v17": v17, "form": form, "context": context, "network": network,
                 "catboost": catboost, "factorization": factorization}
        weights = tuple(BASE[n] for n in NAMES6)
        raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
               for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            train_f = calibration_frame.loc[index]
            valid_f = calibration_frame.loc[oof[str(year)]["row_index"]]
            pieces.append(np.clip(
                raw[year] + residual.mean()
                + W_COUNT * segment_correction(
                    train_f, residual, valid_f, ["balls_before", "strikes_before"], 500)
                + W_PITCHER_COUNT * segment_correction(
                    train_f, residual, valid_f,
                    ["pitcher_id", "balls_before", "strikes_before"], 300)
                + W_EXPERIENCE * segment_correction(
                    train_f, residual, valid_f, ["experience_bin"],
                    EXPERIENCE_SMOOTHING), 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(networks[42], factorizations[42])
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    def show(label, m):
        t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:24s} min {m['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {m['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    results = {}
    print("\nseed averaging (which component gets the averaging, and over how many):",
          flush=True)
    for count in LADDER:
        net_avg, fact_avg = average(networks, count), average(factorizations, count)
        for label, pair in (
                (f"net{count}", (net_avg, factorizations[42])),
                (f"fact{count}", (networks[42], fact_avg)),
                (f"both{count}", (net_avg, fact_avg))):
            results[label] = evaluate(pipeline(*pair))
            results[label]["kind"] = "seed"
            show(label, results[label])

    print("\nBrier objective in place of log loss (seed 42):", flush=True)
    for label, pair in (
            ("mse_net", (networks["mse"], factorizations[42])),
            ("mse_fact", (networks[42], factorizations["mse"])),
            ("mse_both", (networks["mse"], factorizations["mse"]))):
        results[label] = evaluate(pipeline(*pair))
        results[label]["kind"] = "objective"
        show(label, results[label])

    standalone = {}
    for name, store in (("network", networks), ("factorization", factorizations)):
        standalone[name] = {}
        for tag in ["seed42"] + [f"avg{c}" for c in LADDER] + ["mse"]:
            if tag == "seed42":
                got = store[42]
            elif tag == "mse":
                got = store["mse"]
            else:
                got = average(store, int(tag[3:]))
            standalone[name][tag] = {
                str(y_): float(skill(got[str(y_)], targets[season == y_].astype(float)))
                for y_ in YEARS}
    print("\nstandalone skill:", flush=True)
    for name, rows in standalone.items():
        for tag, v in rows.items():
            print(f"  {name:14s} {tag:8s} 2022 {v['2022']:8.0f}  2023 {v['2023']:8.0f}  "
                  f"2024 {v['2024']:8.0f}", flush=True)

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V164_seed_averaging_and_brier_objective",
        "baseline": "V161 (Public 1053.2326413884)",
        "why_untried": {
            "seeds": ("V54 and V85 rejected multi-seed averaging, but both were about tree "
                      "models at the V41 baseline. The only neural measurement is V115's "
                      "epochs12_seeds3, which gained on all three seasons (+421 / +279 / "
                      "+199 standalone) -- and it was measured at 12 epochs while the "
                      "deployed config is 6. The factorization network has never had a "
                      "seed test at all."),
            "objective": ("Both networks train on BCEWithLogitsLoss while the contest "
                          "scores Brier. V73 asked this once, for LightGBM, at the V41 "
                          "baseline. Caveat: the learning rate is held at 2e-3, so a loss "
                          "here is partly a learning-rate result."),
        },
        "seeds": list(SEEDS),
        "ladder": list(LADDER),
        "control_max_abs_difference_vs_v160": control,
        "standalone_skill": standalone,
        "results": {k: {"kind": v["kind"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"networks": networks, "factorizations": factorizations},
                PREDICTIONS, compress=3)
    print(f"\nsafe = {len(survivors)} of {len(results)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
