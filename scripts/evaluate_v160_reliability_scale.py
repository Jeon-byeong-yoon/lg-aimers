"""V160: the last untested constant -- RELIABILITY_SCALE on the feature side.

`RELIABILITY_SCALE = 150` builds one of the eighteen in-season columns,

    ins_pitcher_reliability = inside_n / (inside_n + reliability_scale)

so 150 sets the half-trust point at 150 current-season pitches. V92 chose it a priori.
V155 gridded the *drift term's* use of the same constant across eighteen combinations and
every alternative put a fold into a loss, but the drift term reads the reconstruction at
shrinkage 3 while the feature side reads it at shrinkage 20, and the feature is consumed by
Form (0.32), the network (0.20) and the factorization network (0.07) as an input rather
than as a multiplier. Those are different questions and only the first was asked.

Scope. CatBoost also reads the column but already takes its own reconstruction -- V154 gave
it the projected prior while Form and the two networks kept the career one -- so it keeps
reliability 150 and is not refit here. That saves the expensive fits (350s per fold against
30-60s for the others) and costs nothing in deployment: since the constant changes a single
column inside a block that is already computed twice, no fourth reconstruction is needed.

Control. Each component is refit with *its own* historical prior convention -- per-fold for
Form as V102 used, global for the network and the factorization network as V112 and V130
used -- so the control at reliability 150 reproduces the stored predictions exactly and the
comparison isolates the constant. The global prior is a mild leak on a fold model, as V158's
failed control revealed, but it is identical across all four arms here and therefore cannot
move the comparison; correcting it is a separate job about estimate honesty, not score.

Filter: no fold may lose, and at least one must gain. V156 retired the magnitude threshold
by gaining +1.33 on a three-season average of +0.62, four times what V154 returned on a
signal four times larger.
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
from evaluate_v2 import hist_gbdt_pipeline
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


OUTPUT = Path("artifacts/v160_reliability_scale_metrics.json")
PREDICTIONS = Path("artifacts/v160_reliability_scale_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
DRIFT_RELIABILITY = 150.0
SCALES = (75.0, 150.0, 300.0, 600.0)
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
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
    predictions = {"form": {}, "network": {}, "factorization": {}}
    for scale in SCALES:
        started = time.time()
        block = add_training_inseason_features(
            raw_frame, shrinkage=FEATURE_SHRINKAGE,
            reliability_scale=scale)[feature_names()]
        # The network and the factorization network were fitted on a frame built with the
        # global prior (V112, V130); Form on one built with the fold's prior (V102). Each
        # keeps its own convention so the control reproduces exactly.
        shared_global = select_v2_features(add_row_features(form_raw, global_prior))
        shared_global = add_trackman_features(shared_global, trackman)
        shared_global = pd.concat([shared_global, block], axis=1)
        model_columns = v31_form_columns(shared_global)
        categorical_columns = [c for c, _ in EMBEDDING_SPECS]
        numeric_columns = [c for c in model_columns
                           if c not in categorical_columns and c != "season"]
        identity = shared_global.copy()
        for column in categorical_columns:
            if column not in identity:
                identity[column] = raw_frame[column]
        assert_numeric(identity, numeric_columns)
        print(f"\n[reliability {scale:g}] "
              f"ins_pitcher_reliability mean "
              f"{block['ins_pitcher_reliability'].mean():.5f}  "
              f"[features {time.time() - started:.0f}s]", flush=True)

        for key in predictions:
            predictions[key][scale] = {}
        for year in YEARS:
            train_mask = season < year
            valid_mask = season == year

            fold_prior = float(y.loc[train_mask].mean())
            form_frame = select_v2_features(add_row_features(form_raw, fold_prior))
            form_frame = add_trackman_features(form_frame, trackman)
            form_frame = pd.concat([form_frame, block], axis=1)
            model, columns = hist_gbdt_pipeline(form_frame[v31_form_columns(form_frame)])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()})
            model.fit(form_frame.loc[train_mask, columns], targets[train_mask])
            predictions["form"][scale][str(year)] = model.predict_proba(
                form_frame.loc[valid_mask, columns])[:, 1]
            del model, form_frame
            gc.collect()

            train_identity = identity.loc[train_mask]
            valid_identity = identity.loc[valid_mask]
            vocabularies = build_vocabularies(train_identity)
            statistics = numeric_statistics(
                train_identity[numeric_columns].to_numpy(dtype=np.float64))
            net = train(
                encode_categorical(train_identity, vocabularies),
                encode_numeric(
                    train_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics),
                targets[train_mask], cardinalities(vocabularies),
                epochs=NETWORK_EPOCHS, verbose=False)
            predictions["network"][scale][str(year)] = predict(
                net,
                encode_categorical(valid_identity, vocabularies),
                encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del net
            gc.collect()

            inet.LATENT = FACTORIZATION_LATENT
            inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
            fmodel = inet.train(
                inet.encode_categorical(train_identity, vocabularies),
                inet.encode_numeric(
                    train_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics),
                targets[train_mask], inet.field_cardinalities(vocabularies),
                epochs=FACTORIZATION_EPOCHS, hidden=FACTORIZATION_HIDDEN, verbose=False)
            predictions["factorization"][scale][str(year)] = inet.predict(
                fmodel,
                inet.encode_categorical(valid_identity, vocabularies),
                inet.encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del fmodel, train_identity, valid_identity
            gc.collect()

            actual = targets[valid_mask].astype(float)
            rate = actual.mean()
            line = "  ".join(
                f"{k[:4]} {100000 * (1 - ((predictions[k][scale][str(year)] - actual) ** 2).mean() / (rate * (1 - rate))):7.0f}"
                for k in ("form", "network", "factorization"))
            print(f"  {scale:6g} {year}: {line}  "
                  f"[{time.time() - started:.0f}s cumulative]", flush=True)
        del shared_global, identity, block
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    stored = {
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE],
        "network": joblib.load("artifacts/v112_network_weight_predictions.joblib"
                               )["networks"]["without_season"],
        "factorization": joblib.load(
            "artifacts/v130c_interaction_network_predictions.joblib"
        )["predictions"]["latent8"],
    }
    print("\ncontrol at reliability 150 versus the stored predictions:", flush=True)
    for key in stored:
        drift = max(float(np.abs(predictions[key][150.0][str(y_)]
                                 - stored[key][str(y_)]).max()) for y_ in YEARS)
        print(f"  {key:14s} max abs difference {drift:.3e}", flush=True)

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
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE,
            reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(form, network, factorization):
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

    baseline = pipeline(stored["form"], stored["network"], stored["factorization"])
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

    results = {}
    print(f"\n{'reliability':>12} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>7} "
          f"{'2024':>7} {'blk':>5} {'safe':>5}")
    for scale in SCALES:
        label = f"{scale:g}"
        results[label] = evaluate(pipeline(
            predictions["form"][scale], predictions["network"][scale],
            predictions["factorization"][scale]))
        m = results[label]; t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"{label:>12} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:7.2f} {t['season_points'][1]:7.2f} "
              f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:5.0%} "
              f"{'SAFE' if safe else '':>5}", flush=True)

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l) and l != "150"]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V160_reliability_scale",
        "baseline": "V156 (Public 1052.1807428872)",
        "constant": {
            "name": "RELIABILITY_SCALE",
            "value": 150.0,
            "role": ("builds ins_pitcher_reliability = inside_n / (inside_n + scale), so "
                     "150 sets the half-trust point at 150 current-season pitches"),
            "why_untested": ("V155 gridded the drift term's use of the same constant "
                             "across 18 combinations, but the drift term reads the "
                             "reconstruction at shrinkage 3 as a multiplier while the "
                             "feature side reads it at shrinkage 20 as an input"),
        },
        "scope": ("Form, the network and the factorization network are refit; CatBoost "
                  "keeps 150 because it already takes its own reconstruction (V154), "
                  "which also means no fourth reconstruction is needed in deployment"),
        "control": ("each component refit with its own historical prior convention -- "
                    "per-fold for Form as V102 used, global for the two networks as V112 "
                    "and V130 used -- so reliability 150 reproduces the stored "
                    "predictions and the comparison isolates the constant"),
        "scales": list(SCALES),
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)
    print(f"\nsafe = {len(survivors)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
