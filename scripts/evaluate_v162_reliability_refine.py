"""V162: refine the reliability scale, and re-ask the drift side now that the feature side moved.

V160's ladder was 75 / 150 / 300 / 600 and V161 shipped 300 for +1.05. Two loose ends.

The ladder was coarse and 600 came close to passing -- 2022 +0.64, 2023 +5.74, blocked only
by 2024 at -0.25 -- with a three-season average of +2.04 against 300's +1.56. So the
optimum may sit between them, and the interval was never sampled.

Refining it is cheap for a reason worth stating: V160 showed Form's predictions to be
**bit-identical** at every scale (0.000e+00 across 75, 150, 300 and 600). A
HistGradientBoosting model with 105 features never splits on `ins_pitcher_reliability` --
it reaches the same information through `ins_log_n_pitcher` and the raw counters. So only
the two networks move, and only they need refitting: about five minutes per scale instead
of twenty-four.

Second loose end. V155 gridded the *drift term's* reliability across 18 combinations and
found 150 optimal -- but that was measured when the feature side was also 150. The feature
side is now 300, so the blend the drift term corrects is a different object and its own
optimum may have moved. That costs nothing to re-ask: the drift term is computed from the
reconstruction, not learned.

Filter: no fold may lose, and at least one must gain. Ranked by weakest season, then the
three-season average. Eight submissions: the minimum predicts the sign 8/8 and nothing
predicts the magnitude, its ratio running 0.13 to 2.15.
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


OUTPUT = Path("artifacts/v162_reliability_refine_metrics.json")
PREDICTIONS = Path("artifacts/v162_reliability_refine_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
CURRENT_FEATURE_RELIABILITY = 300.0
NEW_SCALES = (225.0, 400.0, 500.0)
DRIFT_SHRINKAGE = 3.0
CURRENT_DRIFT_RELIABILITY = 150.0
DRIFT_RELIABILITY_GRID = (75.0, 150.0, 250.0, 400.0)
DRIFT_WEIGHT = 0.10
DRIFT_WEIGHT_GRID = (0.08, 0.10, 0.12)
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT = 8
FACTORIZATION_EPOCHS = 8
FACTORIZATION_HIDDEN = (128, 64)
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
    global_prior = float(y.mean())

    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    networks = {CURRENT_FEATURE_RELIABILITY: v160["network"][CURRENT_FEATURE_RELIABILITY]}
    factorizations = {
        CURRENT_FEATURE_RELIABILITY: v160["factorization"][CURRENT_FEATURE_RELIABILITY]}

    for scale in NEW_SCALES:
        started = time.time()
        block = add_training_inseason_features(
            raw_frame, shrinkage=FEATURE_SHRINKAGE,
            reliability_scale=scale)[feature_names()]
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
        networks[scale], factorizations[scale] = {}, {}
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
            net = train(categorical, numeric, targets[train_mask],
                        cardinalities(vocabularies), epochs=NETWORK_EPOCHS, verbose=False)
            networks[scale][str(year)] = predict(
                net, encode_categorical(valid_identity, vocabularies),
                encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del net
            gc.collect()
            inet.LATENT = FACTORIZATION_LATENT
            inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
            fmodel = inet.train(categorical, numeric, targets[train_mask],
                                inet.field_cardinalities(vocabularies),
                                epochs=FACTORIZATION_EPOCHS,
                                hidden=FACTORIZATION_HIDDEN, verbose=False)
            factorizations[scale][str(year)] = inet.predict(
                fmodel, inet.encode_categorical(valid_identity, vocabularies),
                inet.encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del fmodel, train_identity, valid_identity
            gc.collect()
            actual = targets[valid_mask].astype(float)
            rate = actual.mean()
            skills = [100000 * (1 - ((v[scale][str(year)] - actual) ** 2).mean()
                                / (rate * (1 - rate)))
                      for v in (networks, factorizations)]
            print(f"  scale {scale:5g} {year}: netw {skills[0]:7.0f}  "
                  f"fact {skills[1]:7.0f}  [{time.time() - started:.0f}s]", flush=True)
        del frame, identity, block
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    # Form is bit-identical at every reliability scale (V160: 0.000e+00), so the stored
    # predictions serve for every arm and no refit is needed.
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

    terms = {}
    for scale in DRIFT_RELIABILITY_GRID:
        terms[scale] = drift_correction(
            add_training_inseason_features(
                raw_frame, shrinkage=DRIFT_SHRINKAGE,
                reliability_scale=scale).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(network, factorization, drift_weight, drift_reliability):
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
        return np.clip(np.concatenate(pieces)
                       + drift_weight * terms[drift_reliability], 0, 1)

    baseline = pipeline(networks[CURRENT_FEATURE_RELIABILITY],
                        factorizations[CURRENT_FEATURE_RELIABILITY],
                        DRIFT_WEIGHT, CURRENT_DRIFT_RELIABILITY)
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
        print(f"  {label:26s} min {m['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {m['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    results = {}
    print("\nfeature-side reliability (network and factorization refit):", flush=True)
    for scale in NEW_SCALES:
        label = f"feature_{scale:g}"
        results[label] = evaluate(pipeline(
            networks[scale], factorizations[scale], DRIFT_WEIGHT,
            CURRENT_DRIFT_RELIABILITY))
        results[label]["kind"] = "feature"
        show(label, results[label])

    print("\ndrift-side reliability x weight, re-asked at the new feature scale:",
          flush=True)
    for scale in DRIFT_RELIABILITY_GRID:
        for weight in DRIFT_WEIGHT_GRID:
            if scale == CURRENT_DRIFT_RELIABILITY and weight == DRIFT_WEIGHT:
                continue
            label = f"drift_r{scale:g}_w{weight:.2f}"
            results[label] = evaluate(pipeline(
                networks[CURRENT_FEATURE_RELIABILITY],
                factorizations[CURRENT_FEATURE_RELIABILITY], weight, scale))
            results[label]["kind"] = "drift"
            show(label, results[label])

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V162_reliability_refine",
        "baseline": "V161 (Public 1053.2326413884)",
        "loose_ends": {
            "coarse_ladder": ("V160 sampled 75 / 150 / 300 / 600; 600 was blocked only by "
                              "2024 at -0.25 with a three-season average of +2.04 against "
                              "300's +1.56, so the interval between was never looked at"),
            "drift_side": ("V155 found the drift reliability optimal at 150, but measured "
                           "it when the feature side was also 150; the feature side is now "
                           "300 so the blend it corrects is a different object"),
        },
        "why_cheap": ("V160 showed Form bit-identical at every scale (0.000e+00), so a "
                      "105-feature HistGradientBoosting model never splits on "
                      "ins_pitcher_reliability and only the two networks need refitting"),
        "feature_scales": list(NEW_SCALES),
        "drift_grid": {"reliability": list(DRIFT_RELIABILITY_GRID),
                       "weight": list(DRIFT_WEIGHT_GRID)},
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
