"""V147: learn the pitcher's forward rate directly, because the binary label wastes the data.

The oracle audit put the frontier precisely. On the 2024 fold the V138 blend scores 903
and an oracle that knows each pitcher's true season rate scores 991 -- an 88 point gap --
while the pitcher-by-count oracle's much larger number is overfit and V146 showed that
interaction cannot be estimated at all. So what is left is estimating each pitcher's
*current* true rate better than the existing features do.

Every component learns from the binary outcome, whose label has a standard deviation of
0.5. But the quantity that matters for the pitcher-level part of the prediction is a
rate, and a rate over the remaining few hundred pitches of a season has a standard error
nearer 0.02. Learning a mapping from (career count, career rate, in-season reconstruction,
recent-game windows) to a *rate* therefore uses the same rows at roughly twenty-five
times the effective precision as learning it from coin flips. The optimal way to combine
a career rate with an in-season rate and a three-game window is a small, smooth function;
a binary-target model has to infer it through the noise, and V102-V104's shrinkage grids
were a hand search over one parameter of that same function.

Construction. For every training row, the stage-one target is the pitcher's success rate
over the rest of that season, counting strictly the rows *after* the current pitch. The
current pitch is excluded so the label never contains its own answer. Rows with fewer
than 100 pitches remaining are dropped from stage-one *training* -- their target is
mostly noise -- while predictions are still produced for all rows. The features are the
row's own official `asof_*` columns plus the in-season block, all per-row, so the fitted
function applies to a 2025 row unchanged and no evaluation row touches another.

Two uses are tested:

`component` puts the forecast straight into the blend. It is an estimate of P(success)
that ignores the batter and the situation, so it is a level estimate and its standalone
skill will be modest, but level is exactly the dimension the oracle gap lives in.

`feature` hands the forecast to CatBoost alongside everything else, so the situation
modelling is kept and only the pitcher-level input improves. This is the more likely of
the two to pay and the more expensive to measure.

Ranked by the weakest season, per the reliability audit: 2023 carries roughly 659 points
recoverable by shrinking the prediction spread against 1.84 on 2024, so its gains are
contaminated by a one-off dispersion anomaly.

Pre-registered gate, unchanged since V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
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
from catboost import CatBoostClassifier, CatBoostRegressor

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, blend6, three_season
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


OUTPUT = Path("artifacts/v147_rate_forecast_metrics.json")
PREDICTIONS = Path("artifacts/v147_rate_forecast_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
MIN_REMAINING = 100
STAGE_ONE = dict(iterations=700, depth=6, learning_rate=0.03, l2_leaf_reg=8.0)
STAGE_TWO = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
COMPONENT_WEIGHTS = (0.05, 0.10, 0.16)
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def remaining_rate(frame, target):
    """Each row's pitcher's success rate over the rest of that season, excluding itself.

    Rows are ordered within (season, pitcher) by the career-cumulative pitch counter, so
    "after" is unambiguous. Excluding the current row is what keeps the label from
    containing its own answer.
    """
    work = pd.DataFrame({
        "season": frame["season"].to_numpy(),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "counter": frame["asof_pitcher_n"].to_numpy(),
        "target": np.asarray(target, dtype=float),
    })
    work["position"] = np.arange(len(work))
    work = work.sort_values(["season", "pitcher_id", "counter"], kind="mergesort")
    grouped = work.groupby(["season", "pitcher_id"], observed=True)["target"]
    total = grouped.transform("sum")
    size = grouped.transform("size")
    cumulative = grouped.cumsum()
    index = grouped.cumcount() + 1
    after_sum = total - cumulative
    after_n = size - index
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(after_n > 0, after_sum / after_n, np.nan)
    output = np.full(len(work), np.nan)
    output[work["position"].to_numpy()] = rate
    counts = np.zeros(len(work))
    counts[work["position"].to_numpy()] = after_n.to_numpy()
    return output, counts


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    targets = y.to_numpy()
    season = raw_frame["season"].to_numpy()

    forward, remaining = remaining_rate(raw_frame, targets)
    usable = np.isfinite(forward) & (remaining >= MIN_REMAINING)
    print(f"stage-one label: {usable.sum():,} of {len(forward):,} rows usable "
          f"(>= {MIN_REMAINING} pitches remaining); "
          f"label sd {np.nanstd(forward[usable]):.4f} against 0.5 for the binary target",
          flush=True)

    inseason = add_training_inseason_features(raw_frame, shrinkage=FEATURE_SHRINKAGE)
    block = inseason[feature_names()]
    asof = [c for c in raw_frame.columns if c.startswith("asof_")]
    stage_one_columns = asof + list(block.columns)
    stage_one_frame = pd.concat(
        [raw_frame[asof], block], axis=1).astype(np.float32)
    print(f"stage-one features: {len(stage_one_columns)} "
          f"({len(asof)} official asof, {len(block.columns)} in-season)", flush=True)

    forecast = {}
    for year in YEARS:
        started = time.time()
        train_mask = (season < year) & usable
        model = CatBoostRegressor(**STAGE_ONE, loss_function="RMSE", random_seed=42,
                                  verbose=0, thread_count=6, allow_writing_files=False)
        model.fit(stage_one_frame.loc[train_mask], forward[train_mask])
        forecast[str(year)] = np.clip(
            model.predict(stage_one_frame.loc[season == year]), 0.0, 1.0)
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        skill = 100000 * (1 - ((forecast[str(year)] - actual) ** 2).mean()
                          / (rate * (1 - rate)))
        truth = forward[season == year]
        valid = np.isfinite(truth)
        correlation = float(np.corrcoef(forecast[str(year)][valid], truth[valid])[0, 1])
        print(f"  stage one {year}: standalone {skill:8.0f}  "
              f"corr with realised forward rate {correlation:.4f}  "
              f"[{time.time() - started:.0f}s]", flush=True)
        del model
        gc.collect()

    # Stage two: the same CatBoost as V117 with the forecast appended.
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encoding_columns]
               + [c for c in categorical if c not in model_columns])
    work = features[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    # One column per fold, since each fold's forecast comes from a different stage-one fit.
    work["v147_forward_rate"] = np.float32(0.0)
    del features, hierarchical, encoded, form_raw, trackman, data, stage_one_frame
    gc.collect()

    stage_two = {}
    for year in YEARS:
        started = time.time()
        column = np.zeros(len(work), dtype=np.float32)
        for other in YEARS:
            column[season == other] = forecast[str(other)]
        # Training rows need a forecast too; the fold's own stage-one model produced
        # predictions only for that fold, so training rows are filled from the same
        # model applied to their own season, which is what the deployed model will do.
        work["v147_forward_rate"] = column
        model = CatBoostClassifier(**STAGE_TWO, random_seed=42, verbose=0,
                                   thread_count=6, cat_features=categorical,
                                   allow_writing_files=False)
        train_slice = work.loc[season < year]
        model.fit(train_slice, targets[season < year])
        del train_slice
        gc.collect()
        stage_two[str(year)] = model.predict_proba(work.loc[season == year])[:, 1]
        actual = targets[season == year].astype(float)
        rate = actual.mean()
        skill = 100000 * (1 - ((stage_two[str(year)] - actual) ** 2).mean()
                          / (rate * (1 - rate)))
        print(f"  stage two {year}: standalone {skill:8.0f}  "
              f"[{time.time() - started:.0f}s]", flush=True)
        del model
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6), common, oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    def passes(m):
        t = m["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and m["monthly_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"\n{'candidate':>24} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    # `feature` swaps the CatBoost slot content; `component` adds the forecast on top,
    # funded from Form as the only component large enough to pay.
    for weight in COMPONENT_WEIGHTS:
        parts = dict(common)
        weights = list(BASE[n] for n in NAMES6)
        weights[1] = round(weights[1] - weight, 6)
        if weights[1] <= 0:
            continue
        merged = {str(v): ((BASE["form"] - weight) * common["form"][str(v)]
                           + weight * forecast[str(v)]) / BASE["form"]
                  for v in YEARS}
        parts["form"] = merged
        candidate = np.clip(
            blend6(tuple(BASE[n] for n in NAMES6), parts, oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        label = f"component_w{weight:.2f}"
        results[label] = evaluate(candidate)
        m = results[label]; t = m["three_season"]
        print(f"{label:>24} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
              f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:7.0%} "
              f"{'YES' if passes(m) else '-':>5}", flush=True)

    candidate = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6), dict(common, catboost=stage_two),
               oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
    results["feature"] = evaluate(candidate)
    m = results["feature"]; t = m["three_season"]
    print(f"{'feature':>24} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
          f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
          f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:7.0%} "
          f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V147_rate_forecast",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "rationale": (
            "On 2024 the blend scores 903 against a pitcher-rate oracle's 991, and V146 "
            "showed the pitcher-by-count interaction cannot be estimated. What is left "
            "is estimating each pitcher's current rate better. Every component learns "
            "from a binary label with standard deviation 0.5, while the quantity that "
            "matters is a rate whose standard error is nearer 0.02."
        ),
        "construction": (
            "Stage-one target is the pitcher's success rate over the rest of that "
            "season, counting strictly the rows after the current pitch so the label "
            "never contains its own answer. Rows with fewer than 100 pitches remaining "
            "are dropped from training. Features are the row's own official asof "
            "columns plus the in-season block, so the function applies to a 2025 row "
            "unchanged and no evaluation row touches another."
        ),
        "min_remaining": MIN_REMAINING,
        "stage_one_config": STAGE_ONE,
        "stage_two_config": STAGE_TWO,
        "component_weights": list(COMPONENT_WEIGHTS),
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True,
                       "row_independent": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forecast": forecast, "stage_two": stage_two}, PREDICTIONS, compress=3)
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
