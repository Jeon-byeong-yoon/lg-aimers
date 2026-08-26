"""V157: refit the Context model, the one component never revisited.

Context is the last component still carrying its original specification. V31 chose it by
dropping three matchup columns from a frame and fitting `hist_gbdt_pipeline` at its
*defaults*, and nothing has touched it since -- through V38's Form retune, V93's in-season
reconstruction, V106's weight refit, V114's network, V117's CatBoost, V138's factorization
network and V154's prior fix. The blend it sits in has been rebuilt four times around it.

Its hyperparameters point the wrong way by this project's own repeated finding:

    component   iterations   learning rate   leaves   min samples   l2
    Form               500            0.03       15           200   20
    Context            200            0.06       31           100    5

Every capacity experiment here has favoured many trees at a low rate with heavy
regularisation for cross-season transfer -- V116's `strong` over `gentle`, V123's rejection
of `deep8`, V38's choice for Form. Context is the opposite on all five numbers, and it is
the only component that was never asked.

Two axes. The configuration grid moves it toward the Form end. The feature variants revisit
V31's choice: `no_te` drops every `te_*` and `hte_*` column, which is what V116 found better
for CatBoost because ordered statistics beat hand-built encodings, and `all_features` puts
the three matchup columns back.

The incumbent configuration is refit as a control. If it does not reproduce the stored
out-of-fold predictions, the harness is wrong and nothing else here means anything.

Ranked by the weakest season, with no fold permitted to lose -- the rule the leaderboard
has validated across six submissions.
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
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v31_feature_removal import REMOVALS
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v157_context_refit_metrics.json")
PREDICTIONS = Path("artifacts/v157_context_refit_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.65, 0.25, 0.10
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 200, 1000, 3000, 8000, np.inf]
LABELS = ["0-200", "200-1k", "1k-3k", "3k-8k", "8k+"]
CONFIGS = {
    "incumbent": dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                      min_samples_leaf=100, l2_regularization=5.0),
    "mid": dict(max_iter=350, learning_rate=0.045, max_leaf_nodes=23,
                min_samples_leaf=150, l2_regularization=12.0),
    "form_like": dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                      min_samples_leaf=200, l2_regularization=20.0),
    "gentle_long": dict(max_iter=800, learning_rate=0.02, max_leaf_nodes=15,
                        min_samples_leaf=300, l2_regularization=30.0),
    "wide_reg": dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
                     min_samples_leaf=200, l2_regularization=20.0),
}
FEATURE_VARIANTS = ("no_matchup_hte", "no_te", "all_features")
BLOCK_FLOOR = 0.75
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
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv",
                               usecols=TRACKMAN_COLUMNS)
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)
    del raw_trackman, encoded
    gc.collect()

    def columns_for(frame, variant):
        if variant == "all_features":
            return list(frame.columns)
        if variant == "no_matchup_hte":
            return [c for c in frame.columns if c not in REMOVALS["no_matchup_hte"]]
        return [c for c in frame.columns if not c.startswith(("te_", "hte_"))]

    # V31 rebuilt the frame per fold because `add_row_features` takes that fold's prior.
    predictions = {}
    for variant in FEATURE_VARIANTS:
        for name, config in CONFIGS.items():
            if variant != "no_matchup_hte" and name != "form_like":
                continue
            key = f"{variant}__{name}"
            predictions[key] = {}
            for year in YEARS:
                started = time.time()
                train_mask = season < year
                prior = float(y.loc[train_mask].mean())
                frame = select_v2_features(add_row_features(hierarchical, prior))
                frame = add_trackman_features(frame, trackman)
                frame = add_context_trackman_features(frame, context_trackman)
                selected = frame[columns_for(frame, variant)]
                model, columns = hist_gbdt_pipeline(selected)
                model.set_params(**{
                    f"histgradientboostingclassifier__{k}": v
                    for k, v in config.items()})
                model.fit(selected.loc[train_mask, columns], targets[train_mask])
                predictions[key][str(year)] = model.predict_proba(
                    selected.loc[season == year, columns])[:, 1]
                actual = targets[season == year].astype(float)
                rate = actual.mean()
                skill = 100000 * (1 - ((predictions[key][str(year)] - actual) ** 2
                                       ).mean() / (rate * (1 - rate)))
                print(f"  {key:28s} {year}: standalone {skill:8.0f}  "
                      f"({len(columns)} cols) [{time.time() - started:.0f}s]", flush=True)
                del model, frame, selected
                gc.collect()

    stored = joblib.load(
        "artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    control = predictions["no_matchup_hte__incumbent"]
    drift = max(float(np.abs(control[str(y_)] - stored[str(y_)]).max()) for y_ in YEARS)
    print(f"\ncontrol reproduces the stored out-of-fold predictions to {drift:.3e}",
          flush=True)
    if drift > 1e-9:
        print("  WARNING: the harness does not reproduce V31; treat results with care",
              flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v153_projected_prior_predictions.joblib")["catboost"]["projected"]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]["latent8"]
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "network": network, "catboost": catboost,
        "factorization": factorization,
    }
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()

    def pipeline(context):
        parts = dict(common, context=context)
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
            train = calibration_frame.loc[index]
            valid = calibration_frame.loc[oof[str(year)]["row_index"]]
            pieces.append(np.clip(
                raw[year] + residual.mean()
                + W_COUNT * segment_correction(
                    train, residual, valid, ["balls_before", "strikes_before"], 500)
                + W_PITCHER_COUNT * segment_correction(
                    train, residual, valid,
                    ["pitcher_id", "balls_before", "strikes_before"], 300)
                + W_EXPERIENCE * segment_correction(
                    train, residual, valid, ["experience_bin"], EXPERIENCE_SMOOTHING),
                0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(
        joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                    )["no_matchup_hte"]["context"])
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, (season_of == 2024))
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    results = {}
    print(f"\n{'candidate':>30} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>7} "
          f"{'2024':>7} {'blk':>5} {'safe':>5}")
    for key in predictions:
        results[key] = evaluate(pipeline(predictions[key]))
        m = results[key]; t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"{key:>30} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:7.2f} {t['season_points'][1]:7.2f} "
              f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:5.0%} "
              f"{'SAFE' if safe else '':>5}", flush=True)

    def safe(key):
        t = results[key]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [k for k in results if safe(k) and k != "no_matchup_hte__incumbent"]
    # Ties on the weakest season are common when 2022 pins at zero, so the average is the
    # stated second key -- the omission that made V155 promote an arbitrary candidate.
    promoted = max(survivors,
                   key=lambda k: (results[k]["min_season_points"],
                                  results[k]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V157_context_refit",
        "baseline": "V154 (Public 1050.8512921822)",
        "rationale": (
            "Context is the last component with its original specification: V31 fitted "
            "`hist_gbdt_pipeline` at its defaults and nothing has touched it through four "
            "rebuilds of the blend around it. Its hyperparameters oppose this project's "
            "repeated finding on all five numbers."
        ),
        "control": {
            "max_abs_difference_from_stored": drift,
            "note": ("the incumbent configuration is refit as a control; if it does not "
                     "reproduce the stored out-of-fold predictions the harness is wrong"),
        },
        "configs": CONFIGS,
        "feature_variants": list(FEATURE_VARIANTS),
        "ranking": "weakest season, then the three-season average",
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)
    print(f"\nsafe (no fold loses) = {len(survivors)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
