"""V170: is V166's restored-feature gain the features, or is it the seed?

V166 restored the three groups dropped since V2 and found exactly one safe arm: giving all
eighteen features to the **embedding network alone** gains every fold -- 2022 +2.95,
2023 +6.39, 2024 +2.71 -- while Form and the factorization network both lose with them.

That reading cannot be taken at face value, because V164 measured how far this network
moves on seed alone. Swapping seed 42 for an average of two seeds moved the blend by
+8.11 on 2022 and +16.52 on 2023. Those are three to six times the effect V166 is
claiming, from a change that carries no information at all. A single-seed A/B is not
evidence when the noise is that large relative to the signal.

The fix is pairing. Both arms are re-run at five independent seeds, and the statistic is
the **paired** difference at matched seed:

    effect(s) = points(restored at seed s) - points(baseline at seed s)

which cancels the seed draw exactly. The baseline networks at all five seeds are already
cached from V164, so only the restored side needs training. The feature restoration is
real only if the paired difference is positive on every season for every seed; if it
alternates in sign, V166 found a lucky draw and the arm is dead.
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
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import FEATURE_GROUPS, V2_DROPPED_GROUPS, add_row_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v170_restored_network_seed_metrics.json")
PREDICTIONS = Path("artifacts/v170_restored_network_seed_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
NETWORK_EPOCHS = 6
SEEDS = (42, 1004, 2024, 777, 999)
P = 100000.0 / 0.25


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
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    frame = add_trackman_features(add_row_features(form_raw, float(y.mean())), trackman)
    frame = pd.concat([frame, block], axis=1)
    del form_raw, block
    gc.collect()
    keep = v31_form_columns(frame)
    restored = [c for g in V2_DROPPED_GROUPS for c in FEATURE_GROUPS[g]]
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
    print(f"{len(keep)} columns, of which {len(restored)} are the restored groups",
          flush=True)

    networks = {seed: {} for seed in SEEDS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        train_identity, valid_identity = identity.loc[train_mask], identity.loc[valid_mask]
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
        for seed in SEEDS:
            net = train(categorical, numeric, targets[train_mask],
                        cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                        seed=seed, verbose=False)
            networks[seed][str(year)] = predict(net, valid_categorical, valid_numeric)
            del net
            gc.collect()
            got = networks[seed][str(year)]
            print(f"  {year} seed {seed:5d}: restored network "
                  f"{100000 * (1 - ((got - actual) ** 2).mean() / (rate * (1 - rate))):7.0f}"
                  f"  [{time.time() - started:.0f}s]", flush=True)
        del categorical, numeric, valid_categorical, valid_numeric
        gc.collect()
    del identity
    gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    baseline_networks = joblib.load(
        "artifacts/v164_seed_averaging_predictions.joblib")["networks"]
    v166 = joblib.load("artifacts/v166_restore_dropped_groups_predictions.joblib")
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        "factorization": v160["factorization"][FEATURE_RELIABILITY],
    }
    control = max(float(np.abs(networks[42][str(year)]
                               - v166["networks"]["all_three"][str(year)]).max())
                  for year in YEARS)
    print(f"\ncontrol: seed 42 reproduces V166's restored network to {control:.3e}",
          flush=True)

    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y: calibration_frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y])]
        for y in YEARS if y != 2022}
    valid_frames = {y: calibration_frame.loc[oof[str(y)]["row_index"]] for y in YEARS}

    def pipeline(network):
        local = dict(parts)
        local["network"] = network
        raw = {year: sum(BASE[n] * local[n][str(year)] for n in NAMES6) for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
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

    reference = pipeline(v160["network"][FEATURE_RELIABILITY])
    reference_error = (reference - target) ** 2

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (reference_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    rows = {}
    print(f"\n{'seed':>6s}  {'arm':10s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    for seed in SEEDS:
        rows[seed] = {"baseline": points(pipeline(baseline_networks[seed])),
                      "restored": points(pipeline(networks[seed]))}
        for arm in ("baseline", "restored"):
            v = rows[seed][arm]
            print(f"{seed:6d}  {arm:10s} {v[0]:8.2f} {v[1]:8.2f} {v[2]:8.2f}", flush=True)

    print(f"\npaired effect of the restored features, at matched seed:", flush=True)
    print(f"{'seed':>6s} {'2022':>8s} {'2023':>8s} {'2024':>8s}   all three positive?",
          flush=True)
    effects = {}
    for seed in SEEDS:
        effect = [r - b for r, b in zip(rows[seed]["restored"], rows[seed]["baseline"])]
        effects[seed] = effect
        print(f"{seed:6d} {effect[0]:8.2f} {effect[1]:8.2f} {effect[2]:8.2f}   "
              f"{'yes' if min(effect) > 0 else 'NO'}", flush=True)
    matrix = np.array([effects[s] for s in SEEDS])
    print(f"\n  mean paired effect  {matrix.mean(axis=0)[0]:+7.2f} "
          f"{matrix.mean(axis=0)[1]:+7.2f} {matrix.mean(axis=0)[2]:+7.2f}", flush=True)
    print(f"  sd across seeds     {matrix.std(axis=0, ddof=1)[0]:7.2f} "
          f"{matrix.std(axis=0, ddof=1)[1]:7.2f} "
          f"{matrix.std(axis=0, ddof=1)[2]:7.2f}", flush=True)
    unanimous = all(min(effects[s]) > 0 for s in SEEDS)
    print(f"\n  positive on every season for every seed: {unanimous}", flush=True)
    print(f"  V166 claimed +2.95 / +6.39 / +2.71 from the seed-42 pair alone", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V170_restored_network_seed_robustness",
        "baseline": "V161 (Public 1053.2326413884)",
        "question": ("V166's only safe arm gave the eighteen restored features to the "
                     "embedding network and gained every fold, but V164 measured this "
                     "network moving the blend by +8.11 on 2022 and +16.52 on 2023 from a "
                     "seed change alone -- three to six times the claimed effect"),
        "design": ("paired at matched seed, which cancels the draw; the baseline networks "
                   "at all five seeds are cached from V164"),
        "control_vs_v166_seed42": control,
        "per_seed": {str(s): rows[s] for s in SEEDS},
        "paired_effect": {str(s): effects[s] for s in SEEDS},
        "mean_paired_effect": matrix.mean(axis=0).tolist(),
        "sd_across_seeds": matrix.std(axis=0, ddof=1).tolist(),
        "positive_on_every_season_for_every_seed": bool(unanimous),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"networks": networks}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
