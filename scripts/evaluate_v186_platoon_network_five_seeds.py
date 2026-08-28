"""V186: one arm of V185 came within 0.17 points -- settle it with five paired seeds.

V185 gave the unused `pitcher_vs_batter_hand` and `pitcher_count` hierarchical encodings to
the models and measured every component separately, paired across three seeds. Nothing was
unanimous, but one arm stands apart -- the platoon encoding given to the **embedding network
alone**:

    arm      scope     seed     2023     2024
    platoon  network     42     -0.17     9.45
    platoon  network   1004      3.39     2.30
    platoon  network   2024      1.50     3.75
    mean                        +1.57    +5.17

2024 is positive at every seed, and the only losing reading in the arm is 2023 at seed 42,
by 0.17 against readings that swing 30 points between seeds. Standalone the network is
better on all three seasons too (2193 / -364 / 690 against the baseline's 2166 / -373 /
629), which is the opposite of V166's restored groups where the factorization network and
Form both degraded.

`all` behaves the same way with a different gap: 2024 positive at every seed (+2.52, +1.23,
+5.36) and 2023 failing once, at seed 1004 by 0.42.

Three seeds cannot separate "+1.57 with one unlucky draw" from "zero." Seeds 777 and 999
already have cached baselines from V164, so extending both arms to five paired seeds costs
one network refit each and no new baseline. The question this answers is narrow and
pre-registered: **across five paired seeds, is 2024 positive every time and is the 2023
mean positive?** The factorization network and Form are left alone -- V185 showed both
losing 2024 with these features, and this is a one-component change.
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


OUTPUT = Path("artifacts/v186_platoon_network_five_seeds_metrics.json")
PREDICTIONS = Path("artifacts/v186_platoon_network_five_seeds_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
NETWORK_EPOCHS = 6
SEEDS = (42, 1004, 2024, 777, 999)
NEW_SEEDS = (777, 999)
ARMS = {
    "platoon": ["pitcher_batter", "pitcher_vs_batter_hand"],
    "all": ["pitcher_batter", "pitcher_vs_batter_hand", "pitcher_count",
            "batter_vs_pitcher_hand"],
}
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    encoded = add_prior_season_target_encodings(data, y)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    global_prior = float(y.mean())

    v185 = joblib.load("artifacts/v185_platoon_encoding_predictions.joblib")
    networks = {arm: {s: dict(v185["networks"][arm][s]) for s in (42, 1004, 2024)}
                for arm in ARMS}
    for arm, groups in ARMS.items():
        started = time.time()
        hierarchical = add_prior_season_hierarchical_encodings(encoded, y, groups)
        frame = select_v2_features(add_row_features(
            add_stable_form_features(hierarchical), global_prior))
        del hierarchical
        gc.collect()
        frame = add_trackman_features(frame, trackman)
        frame = pd.concat([frame, block], axis=1)
        keep = v31_form_columns(frame)
        categorical_columns = [c for c, _ in EMBEDDING_SPECS]
        numeric_columns = [c for c in keep
                           if c not in categorical_columns and c != "season"]
        identity = frame.copy()
        for column in categorical_columns:
            if column not in identity:
                identity[column] = raw_frame[column]
        assert_numeric(identity, numeric_columns)
        del frame
        gc.collect()
        print(f"\narm {arm}: adding seeds {NEW_SEEDS}", flush=True)
        for seed in NEW_SEEDS:
            networks[arm][seed] = {}
        for year in YEARS:
            train_mask, valid_mask = season < year, season == year
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
            del train_identity, valid_identity
            gc.collect()
            actual = targets[valid_mask].astype(float)
            rate = actual.mean()
            for seed in NEW_SEEDS:
                net = train(categorical, numeric, targets[train_mask],
                            cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                            seed=seed, verbose=False)
                networks[arm][seed][str(year)] = predict(
                    net, valid_categorical, valid_numeric)
                del net
                gc.collect()
                got = networks[arm][seed][str(year)]
                print(f"  {year} seed {seed:5d}: netw "
                      f"{100000 * (1 - ((got - actual) ** 2).mean() / (rate * (1 - rate))):7.0f}"
                      f"  [{time.time() - started:.0f}s]", flush=True)
            del categorical, numeric, valid_categorical, valid_numeric
            gc.collect()
        del identity
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
    }
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    calibration_frame["two_strike"] = (
        calibration_frame["strikes_before"] == 2).astype("int64")
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y_: calibration_frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y_])]
        for y_ in YEARS if y_ != 2022}
    valid_frames = {y_: calibration_frame.loc[oof[str(y_)]["row_index"]] for y_ in YEARS}
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(network, factorization):
        parts = dict(fixed)
        parts.update({"network": network, "factorization": factorization})
        blend = {y_: sum(BASE[n] * parts[n][str(y_)] for n in NAMES6) for y_ in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def readings(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    shipping_baseline = pipeline(v164["networks"][42], v164["factorizations"][42])
    shipping_error = (shipping_baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = shipping_baseline
    reference["v41_squared_error"] = shipping_error

    results, standalone = {}, {}
    print(f"\n{'arm':10s} {'seed':>6s} {'2023':>8s} {'2024':>8s}", flush=True)
    for arm in ARMS:
        rows = {}
        for seed in SEEDS:
            paired_base = pipeline(v164["networks"][seed], v164["factorizations"][seed])
            paired_error = (paired_base - target) ** 2
            rows[str(seed)] = readings(
                pipeline(networks[arm][seed], v164["factorizations"][seed]), paired_error)
            print(f"  {arm:10s} {seed:6d} {rows[str(seed)][1]:8.2f} "
                  f"{rows[str(seed)][2]:8.2f}", flush=True)
        matrix = np.array([rows[str(s)] for s in SEEDS])
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        all_2024 = bool((matrix[:, 2] > 0).all())
        print(f"  {arm:10s} {'mean':>6s} {mean[1]:8.2f} {mean[2]:8.2f}   "
              f"sd {sd[1]:6.2f} {sd[2]:6.2f}   2024 positive at every seed: {all_2024}",
              flush=True)
        results[arm] = {"per_seed": rows, "mean": mean.tolist(), "sd": sd.tolist(),
                        "positive_2024_every_seed": all_2024,
                        "mean_2023_positive": bool(mean[1] > 0)}
        standalone[arm] = {
            str(y_): {str(s): float(100000 * (1 - (
                (networks[arm][s][str(y_)] - targets[season == y_].astype(float)) ** 2
            ).mean() / (targets[season == y_].mean()
                        * (1 - targets[season == y_].mean()))))
                for s in SEEDS} for y_ in YEARS}

    print("\nfull metrics at the shipping seed 42:", flush=True)
    finalists = {}
    for arm in ARMS:
        candidate = pipeline(networks[arm][42], v164["factorizations"][42])
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, shipping_baseline, candidate,
                                 masks[year]) for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        finalists[arm] = metrics
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        safe = metrics["min_season_points"] >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {arm:10s} 2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"CI low {low:+7.2f}  blk {metrics['monthly_block_win_rate']:4.0%}  "
              f"{'SAFE' if safe else ''}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V186_platoon_network_five_seeds",
        "baseline": "V175 (Public 1067.8617513573)",
        "question": ("across five paired seeds, is 2024 positive every time and is the "
                     "2023 mean positive, for the unused platoon encoding given to the "
                     "embedding network alone?"),
        "why": ("V185 found 2024 positive at all three seeds with the only losing reading "
                "2023 at seed 42 by 0.17, against readings that swing 30 points between "
                "seeds; three seeds cannot separate that from zero"),
        "seeds": list(SEEDS),
        "arms": ARMS,
        "results": results,
        "standalone_skill": standalone,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True,
                       "prior_season_encodings_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"networks": networks}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
