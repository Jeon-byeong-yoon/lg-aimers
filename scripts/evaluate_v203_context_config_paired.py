"""V203: V157 compared six Context configs against the luckiest draw of six.

V157 rejected every Context hyperparameter variant, and the reason it gave was that all of
them lost the 2024 fold:

    mid          2024 -3.13        no_te form_like      2024 -5.78
    form_like    2024 -5.11        all_features f_like  2024 -5.11
    gentle_long  2024 -4.57
    wide_reg     2024 -2.61

Six readings, all negative, which looks like a consistent direction. But they share one
baseline -- the incumbent Context draw -- and V195 later measured what that draw is:

    Context standalone skill on the 2024 fold, by seed
      42   1004   2024    777    999     13
     725    678    673    677    667    652

**Seed 42 is the best of six and beats the runner-up by 47 points.** Comparing six
candidates against the luckiest draw makes all six look bad, which is exactly the error
V164 made about seed averaging and V190 made about the k curve. The unpaired noise between
two Context draws is sd ~3.2 on the 2024 fold, so readings of -2.6 to -5.8 are one to two
standard deviations against a baseline known to sit at the top of its own distribution.

Same instrument as V200: paired at the same seed, five seeds, with the measured null
computed from V195's cached draws so the readings can be compared to the noise they
replace.

V202's rule predicts what the error bars will look like. These configs move
`max_leaf_nodes`, `min_samples_leaf`, the learning rate and the regularisation together --
tree *structure*, not just tree count -- so the pairing should buy much less than the 8x it
bought for the reliability scale, and the readings may stay unresolvable. That prediction is
part of what is being tested.
"""

import gc
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import (
    add_context_trackman_features, prepare_context_trackman,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v31_feature_removal import columns_after_removal
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v203_context_config_paired_metrics.json")
PREDICTIONS = Path("artifacts/v203_context_config_paired_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
CONFIGS = {
    "mid": dict(max_iter=350, learning_rate=0.045, max_leaf_nodes=23,
                min_samples_leaf=150, l2_regularization=12.0),
    "form_like": dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                      min_samples_leaf=200, l2_regularization=20.0),
    "gentle_long": dict(max_iter=800, learning_rate=0.02, max_leaf_nodes=15,
                        min_samples_leaf=300, l2_regularization=30.0),
    "wide_reg": dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
                     min_samples_leaf=200, l2_regularization=20.0),
    # V202's rule says a change to tree count alone should pair well, so one arm moves
    # only `max_iter` and leaves the structure exactly as the incumbent has it.
    "iter_only_400": dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                          min_samples_leaf=100, l2_regularization=5.0),
}
PAIRED_SEEDS = (42, 1004, 2024, 777, 999)
NULL_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_NETWORK_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
SHIPPED_FM_SEEDS = SHIPPED_NETWORK_SEEDS
SHIPPED_FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_CATBOOST_SEEDS = (42, 1004, 2024, 777)
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
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    context_trackman = prepare_context_trackman(trackman)
    del encoded
    gc.collect()

    arms = {name: {s: {} for s in PAIRED_SEEDS} for name in CONFIGS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(hierarchical, prior))
        features = add_trackman_features(features, trackman)
        features = add_context_trackman_features(features, context_trackman)
        keep = columns_after_removal(list(features.columns), "no_matchup_hte")
        candidate = features[keep]
        del features
        gc.collect()
        for name, config in CONFIGS.items():
            for seed in PAIRED_SEEDS:
                model, model_columns = hist_gbdt_pipeline(candidate)
                model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                    for k, v in config.items()},
                                 histgradientboostingclassifier__random_state=seed)
                model.fit(candidate.loc[train_mask, model_columns], y.loc[train_mask])
                arms[name][seed][str(year)] = model.predict_proba(
                    candidate.loc[valid_mask, model_columns])[:, 1]
                del model
                gc.collect()
            print(f"  {year} {name:14s} done [{time.time() - started:.0f}s]", flush=True)
        del candidate
        gc.collect()
    del hierarchical
    gc.collect()
    joblib.dump({"arms": arms}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    v197 = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")
    incumbent = v195["contexts"]

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": mean_of({**v190["forms"], **v192["forms"]}, SHIPPED_FORM_SEEDS),
        "network": mean_of({**v164["networks"], **v190["networks"]},
                           SHIPPED_NETWORK_SEEDS),
        "factorization": mean_of({**v164["factorizations"], **v194["factorizations"]},
                                 SHIPPED_FM_SEEDS),
        "catboost": mean_of(v197["catboosts"], SHIPPED_CATBOOST_SEEDS),
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
    scale_sum = 1.0 / sum(w for _, w, _ in TERMS)

    def errors(context):
        parts = dict(fixed)
        parts["context"] = context
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
            for cols, weight, smoothing in TERMS:
                value = value + scale_sum * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        prediction = np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)
        return (prediction - target) ** 2

    def points(candidate_error, base_error):
        return [float(P * (base_error[masks[y_]].mean()
                           - candidate_error[masks[y_]].mean())) for y_ in YEARS]

    incumbent_errors = {s: errors({str(y_): incumbent[s][str(y_)] for y_ in YEARS})
                        for s in NULL_SEEDS}
    null = np.array([points(incumbent_errors[a], incumbent_errors[b])
                     for a, b in combinations(NULL_SEEDS, 2)])
    print(f"\nmeasured null -- two incumbent Context draws ({len(null)} pairs):", flush=True)
    print(f"  sd    {null.std(axis=0, ddof=1)[0]:7.2f} "
          f"{null.std(axis=0, ddof=1)[1]:7.2f} {null.std(axis=0, ddof=1)[2]:7.2f}",
          flush=True)
    print(f"  -> V157's readings of -2.6 to -5.8 on 2024 were taken against seed 42, "
          f"the best of six draws", flush=True)

    print("\npaired against the incumbent config at the same seed:", flush=True)
    print(f"  {'arm':15s} {'2022':>17s} {'2023':>17s} {'2024':>17s}   V157 2024",
          flush=True)
    v157 = {"mid": -3.13, "form_like": -5.11, "gentle_long": -4.57, "wide_reg": -2.61,
            "iter_only_400": None}
    results = {}
    for name in CONFIGS:
        rows = [points(errors(arms[name][s]), incumbent_errors[s]) for s in PAIRED_SEEDS]
        block_ = np.array(rows)
        mean = block_.mean(axis=0)
        se = block_.std(axis=0, ddof=1) / np.sqrt(len(PAIRED_SEEDS))
        results[name] = {"config": CONFIGS[name],
                         "per_seed": {str(s): r for s, r in zip(PAIRED_SEEDS, rows)},
                         "paired_mean": mean.tolist(), "paired_se": se.tolist(),
                         "v157_unpaired_2024": v157[name],
                         "all_positive": bool((mean > 0).all())}
        old = f"{v157[name]:+7.2f}" if v157[name] is not None else "   (new)"
        print(f"  {name:15s} " + "  ".join(
            f"{mean[i]:+7.2f}+-{se[i]:5.2f}" for i in range(3))
            + f"   {old}" + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    print(f"\nV202's rule check -- paired SE on 2024 by how much the config moves:",
          flush=True)
    for name in CONFIGS:
        moves = "tree count only" if name == "iter_only_400" else "tree structure"
        print(f"  {name:15s} {moves:16s} SE {results[name]['paired_se'][2]:5.2f}"
              f"   (unpaired would be {null.std(axis=0, ddof=1)[2] / np.sqrt(5):.2f})",
              flush=True)

    survivors = [n for n in results if results[n]["all_positive"]]
    print(f"\narms positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V203_context_config_paired",
        "baseline": "V199 (Public 1083.2461959655)",
        "why": ("V157 rejected six Context configs because all lost the 2024 fold, but all "
                "six shared one baseline draw and V195 later showed that draw -- seed 42 -- "
                "is the best of six on that fold, beating the runner-up by 47 standalone "
                "points"),
        "prediction_being_tested": ("V202 found the paired instrument's power depends on "
                                    "how much of the model the change touches; these "
                                    "configs move tree structure, so pairing should buy "
                                    "much less than the 8x it bought for one column"),
        "configs": CONFIGS,
        "paired_seeds": list(PAIRED_SEEDS),
        "measured_null": {"pairs": len(null), "sd": null.std(axis=0, ddof=1).tolist()},
        "results": results,
        "survivors": survivors,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
