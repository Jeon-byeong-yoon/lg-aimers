"""Build V199: the last lottery -- CatBoost, four draws instead of one.

V197 measured it and the reading was half of what I predicted. Standalone, CatBoost's seed
range really is narrow -- 727 to 754 on the 2024 fold against the network's 544 to 629 --
because 1,200 boosting iterations average away most of the permutation randomness behind
ordered target statistics. But at the **blend** level it is the second widest of the five:

    component        weight   standalone range   blend range   averaging is worth
    network            0.20                 85         17.94   +5.30
    CatBoost           0.27                 27          9.23   +0.80 at k=3
    factorization      0.07                137          8.67   +1.62
    Context            0.14                 73          6.54   +0.84
    Form               0.32                 26          2.97   +1.72

A small standalone wobble multiplied by the second-largest weight is still a meaningful
blend wobble. Judging the component before multiplying by its weight would have missed it.

The curve is monotone and positive on all three seasons at both measured points -- k=2 gives
+0.68 / +0.50 / +0.60 and k=3 gives +0.90 / +0.66 / +0.80 -- so this ships the average of
all four measured draws. Nine would be worth perhaps +0.3 more and would cost 75 minutes of
extra fold measurement, which is not the trade to make.

Seed 42's deployment model is `v154_catboost.cbm`, already trained by `build_v154` with the
identical construction, and V197's control confirmed its fold predictions match the shipped
ones to 0.000e+00. Only three models are trained here.

Also settled on the way: **the blend weights stand.** V198 refitted them on the averaged
components using V179's cross-fold transfer test, and the selection bias was +230.68 and
+92.43 points -- each climb helped only the season it was fitted on. V167's conclusion
survives the bad reference it was measured against.
"""

import gc
import json
import shutil
import sys
import time
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
from evaluate_v153_projected_prior import CATBOOST_CONFIG, FEATURE_SHRINKAGE
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_inseason_features, build_anchor, feature_names,
)
import inseason_asof_features_v92 as ins
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
REUSED_SEED = 42
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

    # The frame build_v154_artifacts builds, reproduced exactly.
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
    meta = joblib.load("artifacts/v154_catboost_meta.joblib")
    assert list(meta["columns"]) == list(columns), "frame diverged from V154's"
    assert list(meta["categorical"]) == list(categorical)
    print(f"catboost frame: {work.shape[0]:,} rows x {work.shape[1]} columns, "
          f"matches V154's meta", flush=True)

    directory = Path("artifacts/v199_catboosts")
    directory.mkdir(exist_ok=True)
    paths = []
    for seed in CATBOOST_SEEDS:
        destination = directory / f"catboost_seed{seed}.cbm"
        if seed == REUSED_SEED:
            shutil.copyfile("artifacts/v154_catboost.cbm", destination)
            print(f"  seed {seed} reused from v154_catboost.cbm "
                  f"({destination.stat().st_size / 2**20:.1f} MiB)", flush=True)
        else:
            started = time.time()
            model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=seed, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            model.fit(work, y.to_numpy())
            model.save_model(str(destination))
            print(f"  seed {seed} trained [{time.time() - started:.0f}s], "
                  f"{destination.stat().st_size / 2**20:.1f} MiB", flush=True)
            del model
            gc.collect()
        paths.append(destination)
    del work
    gc.collect()
    meta["catboost_seeds"] = list(CATBOOST_SEEDS)
    meta["catboost_files"] = [p.name for p in paths]
    path = Path("artifacts/v199_catboost_meta.joblib")
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
    stores = {
        "network": {**v164["networks"], **v190["networks"]},
        "form": {**v190["forms"], **v192["forms"]},
        "factorization": {**v164["factorizations"], **v194["factorizations"]},
        "context": v195["contexts"],
    }
    stored_catboost = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                 )["catboost"]["projected"]
    control = max(float(np.abs(v197["catboosts"][42][str(y_)]
                               - stored_catboost[str(y_)]).max()) for y_ in YEARS)
    print(f"control: V197's seed-42 CatBoost matches the shipped one to {control:.3e}")
    assert control == 0.0, "the cached seed-42 CatBoost must be the one V196 shipped"

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "catboost": mean_of(v197["catboosts"], CATBOOST_SEEDS),
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
    path = Path("artifacts/v199_calibration.joblib")
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
    previous = joblib.load("artifacts/v196_calibration.joblib")["global_shift"]
    print(f"Saved {path}, shift={shift:.12f} (V196 was {previous:.12f})")

    Path("artifacts/v199_build_summary.json").write_text(json.dumps({
        "version": "V199",
        "baseline": "V196 (1082.8399355688)",
        "change": "CatBoost averaged over four draws; every other component unchanged",
        "catboost_seeds": list(CATBOOST_SEEDS),
        "reused": "seed 42 is v154_catboost.cbm, identical construction",
        "validation": {
            "catboost_2024_curve": {"1": 0.0, "2": 0.60, "3": 0.80},
            "catboost_lottery_range": {"standalone_2024": 27, "blend_2024": 9.23},
            "why_it_matters_despite_a_narrow_standalone_range": (
                "a small standalone wobble multiplied by the second-largest weight is "
                "still a meaningful blend wobble"),
            "weights_left_alone": ("V198 refitted them on the averaged components under "
                                   "V179's cross-fold transfer test and the selection "
                                   "bias was +230.68 and +92.43 points"),
        },
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
