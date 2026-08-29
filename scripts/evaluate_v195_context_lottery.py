"""V195: the last unmeasured lottery that is cheap -- Context, weight 0.14.

Four of the five learned components have now been measured with the corrected estimator,
which pools over which seeds go inside so the luckiest draw does not sit on both sides:

    component            weight   2024 lottery range   averaging is worth, 2024
    embedding network      0.20                17.94   +5.30 at k=8   (shipped k=9)
    factorization          0.07                 8.67   +1.62 at k=7   (V194)
    Form                   0.32                 2.97   +1.72 at k=5   (shipped k=6)
    Context                0.14                     ?                          ?
    CatBoost               0.27                     ?   deferred: 160s per fold fit

Context is the same model class as Form, so its lottery is presumably as narrow, and its
weight is smaller. The expected number is small. It is measured because it is the last one
that costs minutes rather than an hour, and because Form -- also narrow, also presumed
harmless -- turned out to be worth +1.72.

Reproducing Context is straightforward once located: `evaluate_v31_feature_removal.py`
trains it with `hist_gbdt_pipeline`'s **default** hyperparameters (200 iterations, not
Form's 500), on `select_v2_features(add_row_features(hierarchical, prior))` plus the
Trackman and contextual-Trackman blocks, with the three `hte_pitcher_batter_*` columns
removed, and `prior` is the fold's own training mean rather than the global one.

Baseline is V193 with V194's factorization average folded in, so the curve is read on top
of every averaging decision already made.
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


OUTPUT = Path("artifacts/v195_context_lottery_metrics.json")
PREDICTIONS = Path("artifacts/v195_context_lottery_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_RELIABILITY, FEATURE_SHRINKAGE = 300.0, 20.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
CONTEXT_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_NETWORK_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
SHIPPED_FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)
FACTORIZATION_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
MAX_SUBSETS = 20
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

    contexts = {s: {} for s in CONTEXT_SEEDS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(hierarchical, prior))
        features = add_trackman_features(features, trackman)
        features = add_context_trackman_features(features, context_trackman)
        columns = columns_after_removal(list(features.columns), "no_matchup_hte")
        candidate = features[columns]
        del features
        gc.collect()
        actual = targets[valid_mask].astype(float)
        rate = actual.mean()
        for seed in CONTEXT_SEEDS:
            model, model_columns = hist_gbdt_pipeline(candidate)
            model.set_params(histgradientboostingclassifier__random_state=seed)
            model.fit(candidate.loc[train_mask, model_columns], y.loc[train_mask])
            contexts[seed][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, model_columns])[:, 1]
            del model
            gc.collect()
        skills = [100000 * (1 - ((contexts[s][str(year)] - actual) ** 2).mean()
                            / (rate * (1 - rate))) for s in CONTEXT_SEEDS]
        print(f"  {year} Context by seed: " + "  ".join(f"{v:7.0f}" for v in skills)
              + f"   [{time.time() - started:.0f}s]", flush=True)
        del candidate
        gc.collect()
    del hierarchical
    gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    cached = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                         )["no_matchup_hte"]["context"]
    network_store = dict(v164["networks"]); network_store.update(v190["networks"])
    form_store = dict(v190["forms"]); form_store.update(v192["forms"])
    fm_store = dict(v164["factorizations"]); fm_store.update(v194["factorizations"])

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    print("\ncontrol: which seed reproduces the shipped Context?", flush=True)
    for seed in CONTEXT_SEEDS:
        gap = max(float(np.abs(contexts[seed][str(y_)] - cached[str(y_)]).max())
                  for y_ in YEARS)
        print(f"  seed {seed:5d}: max |new - shipped| = {gap:.3e}", flush=True)

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": mean_of(form_store, SHIPPED_FORM_SEEDS),
        "network": mean_of(network_store, SHIPPED_NETWORK_SEEDS),
        "factorization": mean_of(fm_store, FACTORIZATION_SEEDS),
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

    def pipeline(context):
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
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    base_error = {s: (pipeline(contexts[s]) - target) ** 2 for s in CONTEXT_SEEDS}
    lottery = np.array([points(pipeline(contexts[s]), base_error[CONTEXT_SEEDS[0]])
                        for s in CONTEXT_SEEDS])
    print(f"\nContext's lottery, every seed against seed 42:", flush=True)
    print(f"  sd    {lottery.std(axis=0, ddof=1)[0]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[1]:7.2f} "
          f"{lottery.std(axis=0, ddof=1)[2]:7.2f}", flush=True)
    print(f"  range {lottery[:, 0].ptp():7.2f} {lottery[:, 1].ptp():7.2f} "
          f"{lottery[:, 2].ptp():7.2f}   "
          f"(network 17.94, factorization 8.67, Form 2.97 on 2024)", flush=True)

    generator = np.random.default_rng(20260829)
    print("\naverage of k against the seeds it excludes, pooled over which go inside:",
          flush=True)
    print(f"  {'k':>3s} {'subsets':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    table = {}
    for k in range(1, len(CONTEXT_SEEDS)):
        every = list(combinations(CONTEXT_SEEDS, k))
        picked = ([every[i] for i in generator.choice(len(every), MAX_SUBSETS,
                                                      replace=False)]
                  if len(every) > MAX_SUBSETS else every)
        readings = []
        for inside in picked:
            avg = contexts[inside[0]] if k == 1 else mean_of(contexts, inside)
            candidate = pipeline(avg)
            for s in CONTEXT_SEEDS:
                if s in inside:
                    continue
                readings.append(points(candidate, base_error[s]))
        block = np.array(readings)
        mean = block.mean(axis=0)
        table[k] = {"subsets": len(picked), "readings": len(readings),
                    "expected": mean.tolist(),
                    "sd": block.std(axis=0, ddof=1).tolist()}
        print(f"  {k:3d} {len(picked):8d} {mean[0]:+8.2f} {mean[1]:+8.2f} {mean[2]:+8.2f}"
              + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)

    best = max(table, key=lambda k: table[k]["expected"][2])
    print(f"\nbest k by expected 2024 = {best} ({table[best]['expected'][2]:+.2f})",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V195_context_lottery",
        "baseline": "V193 plus V194's factorization average",
        "seeds": list(CONTEXT_SEEDS),
        "lottery_sd": lottery.std(axis=0, ddof=1).tolist(),
        "lottery_range": [float(lottery[:, i].ptp()) for i in range(3)],
        "k_table": table,
        "best_k_by_2024": best,
        "held_at": {"network": list(SHIPPED_NETWORK_SEEDS),
                    "form": list(SHIPPED_FORM_SEEDS),
                    "factorization": list(FACTORIZATION_SEEDS)},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"contexts": contexts}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
