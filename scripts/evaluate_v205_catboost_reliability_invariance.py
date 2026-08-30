"""V205: the CatBoost reliability scale should be closable by monotone invariance.

The record lists "CatBoost's own reliability scale" as the one unexplored constant: the
CatBoost block is built by `evaluate_v153_projected_prior.build_block`, which leaves
`add_inseason_features` at its module default `RELIABILITY_SCALE = 150` -- never chosen
deliberately, never swept.

But `ins_pitcher_reliability = inside_n / (inside_n + scale)` is a strictly monotone
transform of `inside_n` for every positive scale, so changing the scale preserves the
column's row ordering exactly. A tree model's split candidates partition rows by order;
the only way the scale can reach a CatBoost fit is through value-dependent border
quantization (GreedyLogSum), a second-order effect. This is the same mechanism by which
V160 found Form (HGB) bit-identical across scales.

So instead of a 4-hour sweep, one measurement: rebuild the block at scale 600 (4x the
incumbent), retrain the 2024 fold at seed 42 with the incumbent config, and read the
paired difference against the cached V197 fit. If the reading is small, the constant is
closed by argument plus measurement; only a large reading would justify the sweep.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v153_projected_prior import CATBOOST_CONFIG, FEATURE_SHRINKAGE
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_inseason_features, add_training_inseason_features, build_anchor,
    drift_correction, feature_names,
)
from inseason_prior_v153 import population_inside_rates, projected_priors
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v205_catboost_reliability_invariance_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
INCUMBENT_SCALE = 150.0  # the add_inseason_features module default the block inherits
CANDIDATE_SCALE = 600.0
SEED = 42
FOLD = 2024
SHIPPED = {"network": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "form": (42, 1004, 2024, 777, 999, 13),
           "factorization": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "context": (42, 1004, 2024, 777, 999, 13)}
P = 100000.0 / 0.25


def build_block_with_scale(raw_frame, shrinkage, reliability_scale):
    """`evaluate_v153_projected_prior.build_block` (projected mode) with the
    reliability scale exposed instead of inherited from the module default."""
    table = population_inside_rates(raw_frame)
    pieces = []
    for season in sorted(raw_frame["season"].unique()):
        current = raw_frame.loc[raw_frame["season"] == season]
        history = raw_frame.loc[raw_frame["season"] < season]
        if history.empty:
            pieces.append(pd.DataFrame(np.nan, index=current.index,
                                       columns=feature_names()))
            continue
        anchors = {g: build_anchor(history, g) for g in ("pitcher", "batter")}
        priors = projected_priors(
            history, season, table=table.loc[table.index < season])
        pieces.append(add_inseason_features(
            current, anchors, priors, current_season=season, shrinkage=shrinkage,
            reliability_scale=reliability_scale)[feature_names()])
    return pd.concat(pieces).loc[raw_frame.index]


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
    shared = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    shared = add_trackman_features(shared, trackman)
    del hierarchical, encoded, trackman
    gc.collect()

    started = time.time()
    incumbent_block = build_block_with_scale(
        raw_frame, FEATURE_SHRINKAGE, INCUMBENT_SCALE)
    candidate_block = build_block_with_scale(
        raw_frame, FEATURE_SHRINKAGE, CANDIDATE_SCALE)

    # The scale must touch exactly one column, preserving its row order.
    same = [c for c in feature_names()
            if incumbent_block[c].equals(candidate_block[c])]
    changed = [c for c in feature_names() if c not in same]
    assert changed == ["ins_pitcher_reliability"], changed
    a = incumbent_block["ins_pitcher_reliability"].to_numpy()
    b = candidate_block["ins_pitcher_reliability"].to_numpy()
    both = ~np.isnan(a) & ~np.isnan(b)
    assert (np.isnan(a) == np.isnan(b)).all()
    rho = float(spearmanr(a[both], b[both]).statistic)
    column_gap = float(np.abs(a[both] - b[both]).max())
    print(f"column check: spearman {rho:.9f}, max abs value gap {column_gap:.4f} "
          f"[{time.time() - started:.0f}s]", flush=True)
    del incumbent_block, a, b, both
    gc.collect()

    features = pd.concat([shared, candidate_block], axis=1)
    model_columns = v31_form_columns(features)
    del shared, candidate_block
    gc.collect()
    categorical = [n for n, _ in EMBEDDING_SPECS if n in model_columns]
    work = features.copy()
    for name, _ in EMBEDDING_SPECS:
        if name not in work:
            work[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encodings = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encodings]
               + [c for c in categorical if c not in model_columns])
    work = work[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    del features
    gc.collect()

    model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=SEED, verbose=0,
                               thread_count=6, cat_features=categorical,
                               allow_writing_files=False)
    model.fit(work.loc[season < FOLD], targets[season < FOLD])
    candidate = model.predict_proba(work.loc[season == FOLD])[:, 1]
    print(f"trained {FOLD} fold at scale {CANDIDATE_SCALE:g} "
          f"[{time.time() - started:.0f}s]", flush=True)
    del model, work
    gc.collect()

    cached = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib"
                         )["catboosts"][SEED]
    gap = candidate - cached[str(FOLD)]
    print(f"prediction gap vs cached scale-{INCUMBENT_SCALE:g} fit: "
          f"max {np.abs(gap).max():.3e}, sd {gap.std():.3e}", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    stores = {
        "network": {**v164["networks"], **v190["networks"]},
        "form": {**v190["forms"], **v192["forms"]},
        "factorization": {**v164["factorizations"], **v194["factorizations"]},
        "context": v195["contexts"],
    }

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        **{name: mean_of(stores[name], SHIPPED[name]) for name in stores},
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

    def pipeline(catboost):
        parts = dict(fixed)
        parts["catboost"] = catboost
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
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    incumbent_error = (pipeline(cached) - target) ** 2
    candidate_error = (pipeline({**cached, str(FOLD): candidate}) - target) ** 2
    points = [float(P * (incumbent_error[masks[y_]].mean()
                         - candidate_error[masks[y_]].mean())) for y_ in YEARS]
    print(f"\npaired blend reading, scale {INCUMBENT_SCALE:g} -> {CANDIDATE_SCALE:g} "
          f"(seed {SEED}, {FOLD} fold only):", flush=True)
    print(f"  2022 {points[0]:+8.3f}  2023 {points[1]:+8.3f}  2024 {points[2]:+8.3f}",
          flush=True)
    print("  (2022/2023 are identity checks and must be exactly 0)", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V205_catboost_reliability_invariance",
        "baseline": "V199 (Public 1083.2461959655), CatBoost block reliability scale 150",
        "why": ("the one unexplored constant; but the reliability column is a strictly "
                "monotone transform of inside_n at every scale, so a tree model can see "
                "the scale only through border quantization -- predicted near-null"),
        "design": ("one paired fit: 2024 fold, seed 42, incumbent config, block rebuilt "
                   "at scale 600, read against the cached V197 fit at scale 150"),
        "incumbent_scale": INCUMBENT_SCALE,
        "candidate_scale": CANDIDATE_SCALE,
        "column_check": {"spearman": rho, "max_abs_value_gap": column_gap,
                         "columns_changed": changed},
        "prediction_gap": {"max_abs": float(np.abs(gap).max()),
                           "sd": float(gap.std())},
        "paired_points": {"2022": points[0], "2023": points[1], "2024": points[2]},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
