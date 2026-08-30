"""Build V207: the shipped CatBoosts cut to 1,000 trees -- no retraining.

V204 read the iteration curve through prefixes of one fit and the verdict was the first
significant capacity reading of the project: 1,000 trees beats the shipped 1,200 on every
season, +0.93 +- 0.34 on 2024 (2.7 SE), and it is a ridge, not an isolated peak -- 600,
800 and 1,000 agree on 2024 (+0.99 / +0.95 / +0.93) and the right side declines
monotonically (1500 -1.13, 1800 -2.64, 2100 -3.37, 2400 -5.29). V123/V124 chose 1200 on
the old instrument, whose unpaired noise (sd 8.67) could not see a one-point structure.

Because boosting is sequential and the learning rate is fixed, the first 1,000 trees of
the shipped models ARE the 1,000-iteration models -- V204's control reproduced the cached
1200-tree predictions from a 2400-tree fit to 0.000e+00. So the deployment models are the
four shipped `.cbm` files shrunk in place, and the control here demands exact prediction
equality between the shrunk model and the original read at `ntree_end=1000` on the full
training frame.

The calibration layer is refitted because CatBoost's out-of-fold predictions moved, using
V204's cached 1000-prefix fold predictions -- the same construction as build_v199.
"""

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v153_projected_prior import FEATURE_SHRINKAGE
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_inseason_features, build_anchor, feature_names,
)
from inseason_prior_v153 import population_inside_rates, projected_priors
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
RAW_SEGMENT_WEIGHTS = {"count": 0.55, "pitcher_count": 0.25,
                       "experience": 0.20, "platoon_two_strike": 0.80}
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
PLATOON_TWO_STRIKE_SMOOTHING = 1500.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY, CATBOOST_RELIABILITY = 300.0, 150.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
CATBOOST_SEEDS = (42, 1004, 2024, 777)
SHIPPED_TREES, NEW_TREES = 1200, 1000
SHIPPED = {"network": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "form": (42, 1004, 2024, 777, 999, 13),
           "factorization": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "context": (42, 1004, 2024, 777, 999, 13)}


def main():
    assert abs(W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST
               + W_FACTORIZATION - 1.0) < 1e-12
    denominator = sum(RAW_SEGMENT_WEIGHTS.values())
    weights = {k: v / denominator for k, v in RAW_SEGMENT_WEIGHTS.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-12

    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    # The frame build_v154_artifacts builds, reproduced exactly (as build_v199 did).
    table = population_inside_rates(raw_frame)
    pieces = []
    for value in sorted(raw_frame["season"].unique()):
        current = raw_frame.loc[raw_frame["season"] == value]
        history = raw_frame.loc[raw_frame["season"] < value]
        if history.empty:
            pieces.append(pd.DataFrame(np.nan, index=current.index,
                                       columns=feature_names()))
            continue
        pieces.append(add_inseason_features(
            current,
            {g: build_anchor(history, g) for g in ("pitcher", "batter")},
            projected_priors(history, value, table=table.loc[table.index < value]),
            current_season=value, shrinkage=FEATURE_SHRINKAGE)[feature_names()])
    block = pd.concat(pieces).loc[raw_frame.index]
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    features = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical, encoded
    gc.collect()
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encodings = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encodings]
               + [c for c in categorical if c not in model_columns])
    work = features[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    del features
    gc.collect()
    meta = joblib.load("artifacts/v199_catboost_meta.joblib")
    assert list(meta["columns"]) == list(columns), "frame diverged from V154's"
    assert list(meta["categorical"]) == list(categorical)
    print(f"catboost frame: {work.shape[0]:,} rows x {work.shape[1]} columns, "
          f"matches the shipped meta", flush=True)

    # ---- shrink the shipped models; demand exact prediction equality ----
    directory = Path("artifacts/v207_catboosts")
    directory.mkdir(exist_ok=True)
    for seed in CATBOOST_SEEDS:
        source = Path(f"artifacts/v199_catboosts/catboost_seed{seed}.cbm")
        destination = directory / source.name
        original = CatBoostClassifier()
        original.load_model(str(source))
        assert original.tree_count_ == SHIPPED_TREES, original.tree_count_
        reference = original.predict_proba(work, ntree_end=NEW_TREES)[:, 1]
        original.shrink(ntree_end=NEW_TREES)
        assert original.tree_count_ == NEW_TREES, original.tree_count_
        original.save_model(str(destination))
        shrunk = CatBoostClassifier()
        shrunk.load_model(str(destination))
        assert shrunk.tree_count_ == NEW_TREES
        gap = float(np.abs(shrunk.predict_proba(work)[:, 1] - reference).max())
        print(f"  seed {seed}: shrunk {SHIPPED_TREES} -> {NEW_TREES} trees, "
              f"saved-vs-prefix gap {gap:.3e}, "
              f"{destination.stat().st_size / 2**20:.1f} MiB", flush=True)
        assert gap == 0.0, "the shrunk model must be the exact 1000-tree prefix"
        del original, shrunk
        gc.collect()
    del work
    gc.collect()
    meta["catboost_seeds"] = list(CATBOOST_SEEDS)
    meta["catboost_files"] = [f"catboost_seed{seed}.cbm" for seed in CATBOOST_SEEDS]
    meta["catboost_iterations"] = NEW_TREES
    path = Path("artifacts/v207_catboost_meta.joblib")
    joblib.dump(meta, path, compress=3)
    print(f"Saved {path}")

    # ---- calibration, refitted because CatBoost's out-of-fold predictions moved ----
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    v197 = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")
    prefixes = joblib.load(
        "artifacts/v204_catboost_iterations_prefix_predictions.joblib")["prefixes"]
    stores = {
        "network": {**v164["networks"], **v190["networks"]},
        "form": {**v190["forms"], **v192["forms"]},
        "factorization": {**v164["factorizations"], **v194["factorizations"]},
        "context": v195["contexts"],
    }
    control = max(float(np.abs(prefixes[SHIPPED_TREES][s][str(y_)]
                               - v197["catboosts"][s][str(y_)]).max())
                  for s in CATBOOST_SEEDS for y_ in YEARS)
    print(f"control: V204's 1200-prefix matches the shipped folds to {control:.3e}")
    assert control == 0.0, "V204's prefixes must be anchored to the shipped CatBoost"

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "catboost": mean_of(prefixes[NEW_TREES], CATBOOST_SEEDS),
        **{name: mean_of(stores[name], SHIPPED[name]) for name in stores},
    }
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = pd.cut(raw["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
    raw["two_strike"] = (raw["strikes_before"] == 2).astype("int64")
    indices, targets, predictions = [], [], []
    for year in YEARS:
        item = oof[str(year)]
        key = str(year)
        predictions.append(
            W_V17 * parts["v17"][key] + W_FORM * parts["form"][key]
            + W_CONTEXT * parts["context"][key] + W_NETWORK * parts["network"][key]
            + W_CATBOOST * parts["catboost"][key]
            + W_FACTORIZATION * parts["factorization"][key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    fine = make_lookup(frame, centered, ["pitcher_id", "batter_hand", "two_strike"],
                       PLATOON_TWO_STRIKE_SMOOTHING)
    assert fine["two_strike"].dtype == np.int64
    path = Path("artifacts/v207_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": weights["count"],
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": weights["pitcher_count"],
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["experience_bin"], "weight": weights["experience"],
             "lookup": make_lookup(frame, centered, ["experience_bin"],
                                   EXPERIENCE_SMOOTHING)},
            {"columns": ["pitcher_id", "batter_hand", "two_strike"],
             "weight": weights["platoon_two_strike"], "lookup": fine},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "feature_reliability_scale": FEATURE_RELIABILITY,
        "catboost_reliability_scale": CATBOOST_RELIABILITY,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
        "experience_edges": [float(v) for v in EDGES],
        "experience_labels": list(LABELS),
        "catboost_seeds": list(CATBOOST_SEEDS),
        **{f"{name}_seeds": list(seeds) for name, seeds in SHIPPED.items()},
    }, path, compress=3)
    previous = joblib.load("artifacts/v199_calibration.joblib")["global_shift"]
    print(f"Saved {path}, shift={shift:.12f} (V199 was {previous:.12f})")

    Path("artifacts/v207_build_summary.json").write_text(json.dumps({
        "version": "V207",
        "baseline": "V199 (1083.2461959655)",
        "change": "CatBoost cut from 1200 to 1000 trees; every other component unchanged",
        "how": ("the shipped .cbm files shrunk in place -- boosting prefixes are exact, "
                "so no retraining; controls demand prediction equality to 0.0"),
        "validation": {
            "paired_reading_1000_vs_1200": {
                "2022": "+0.80 +- 0.61", "2023": "+0.67 +- 0.67",
                "2024": "+0.93 +- 0.34 (2.7 SE)"},
            "ridge_not_peak": ("600/800/1000 agree on 2024 (+0.99/+0.95/+0.93); "
                               "1500..2400 decline monotonically"),
            "avg_vs_avg_1000": {"2022": "+0.63", "2023": "+0.51", "2024": "+0.72"},
            "minimum_season_expectation": "+0.51 (2023, avg-vs-avg)",
        },
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved artifacts/v207_build_summary.json")


if __name__ == "__main__":
    main()
