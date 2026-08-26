"""V153: retrain the two largest components on in-season features with a season-tracking prior.

The prior inside the in-season reconstruction shrinks toward the mean of the career as-of
rate column, and a career average lags a league that fell from 0.5647 to 0.4861. Measured
against what each fold's target season actually was, the error grows every season -- and
it is visibly in the predictions: on the 2024 fold the over-prediction falls monotonically
with pitcher experience (+0.0157 in F, +0.0077 in R below 200 pitches, crossing zero near
a thousand) which is exactly the shape a bad prior produces, since it carries weight
20/(inside_n + 20).

`inseason_prior_v153` replaces it with the population's *within-season* rate, recovered by
the same anchor differencing V92 built for individual pitchers, then projected linearly to
the target season. That is one uniform rule for all seven reconstructed columns and it
uses nothing from the season being predicted:

    target   V92 prior   V153 prior   actual   V92 gap   V153 gap
      2022    0.547341     0.524781  0.528920   +0.0184    -0.0041
      2023    0.544313     0.517275  0.499957   +0.0444    +0.0173
      2024    0.540175     0.498733  0.486105   +0.0541    +0.0126
      2025    0.535228     0.490536       n/a         -          -

The gap shrinks by 60-77% on every fold, so unlike almost everything else tried lately
this change has a reason to help all three rather than trading one against another.

V152 already patched the symptom: a calibration segment on experience bins was safe on
every fold but worth only +1.71 on 2024, because a post-hoc segment carries weight 0.10
and cannot fix the feature the models learned from. The two components retrained here hold
0.32 and 0.27 of the blend between them, which is where the feature actually enters.

The drift term is left alone deliberately. It reads the same reconstruction at shrinkage 3,
where the prior carries weight 3/(inside_n + 3) and is nearly inert -- rebuilding it with
the new prior moved the folds by +0.01, +0.30 and +0.02, confirming that channel is not
where the error lives.

Ranked by the weakest season, the instrument validated on five submissions.

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
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
import inseason_asof_features_v92 as ins
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, blend6, three_season
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_inseason_features, build_anchor, drift_correction, feature_names,
)
from inseason_prior_v153 import population_inside_rates, projected_priors
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v153_projected_prior_metrics.json")
PREDICTIONS = Path("artifacts/v153_projected_prior_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
CATBOOST_CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def build_block(raw_frame, shrinkage, mode):
    """In-season features per season, with the prior chosen by `mode`."""
    table = population_inside_rates(raw_frame) if mode == "projected" else None
    pieces = []
    for season in sorted(raw_frame["season"].unique()):
        current = raw_frame.loc[raw_frame["season"] == season]
        history = raw_frame.loc[raw_frame["season"] < season]
        if history.empty:
            pieces.append(pd.DataFrame(np.nan, index=current.index,
                                       columns=feature_names()))
            continue
        anchors = {g: build_anchor(history, g) for g in ("pitcher", "batter")}
        if mode == "projected":
            priors = projected_priors(
                history, season, table=table.loc[table.index < season])
        else:
            priors = ins.build_priors(history)
        pieces.append(add_inseason_features(
            current, anchors, priors, current_season=season,
            shrinkage=shrinkage)[feature_names()])
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
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    shared = select_v2_features(add_row_features(form_raw, float(y.mean())))
    shared = add_trackman_features(shared, trackman)
    del hierarchical, encoded, form_raw, trackman
    gc.collect()

    blocks = {mode: build_block(raw_frame, FEATURE_SHRINKAGE, mode)
              for mode in ("career", "projected")}
    difference = (blocks["projected"]["ins_asof_pitcher_success_rate"]
                  - blocks["career"]["ins_asof_pitcher_success_rate"])
    print(f"feature shift from the new prior: mean {difference.mean():+.6f}, "
          f"sd {difference.std():.6f}, "
          f"low-experience mean "
          f"{difference[raw_frame['asof_pitcher_n'] < 1000].mean():+.6f}", flush=True)

    form_predictions, catboost_predictions = {}, {}
    for mode in ("career", "projected"):
        features = pd.concat([shared, blocks[mode]], axis=1)
        model_columns = v31_form_columns(features)
        form_predictions[mode] = {}
        for year in YEARS:
            started = time.time()
            model, columns = hist_gbdt_pipeline(features[model_columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v
                for k, v in FORM_CONFIG.items()})
            model.fit(features.loc[season < year, columns], targets[season < year])
            form_predictions[mode][str(year)] = model.predict_proba(
                features.loc[season == year, columns])[:, 1]
            actual = targets[season == year].astype(float)
            rate = actual.mean()
            skill = 100000 * (1 - ((form_predictions[mode][str(year)] - actual) ** 2
                                   ).mean() / (rate * (1 - rate)))
            print(f"  form {mode:9s} {year}: standalone {skill:8.0f} "
                  f"[{time.time() - started:.0f}s]", flush=True)
            del model
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
        catboost_predictions[mode] = {}
        for year in YEARS:
            started = time.time()
            model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=42, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            train_slice = work.loc[season < year]
            model.fit(train_slice, targets[season < year])
            del train_slice
            gc.collect()
            catboost_predictions[mode][str(year)] = model.predict_proba(
                work.loc[season == year])[:, 1]
            actual = targets[season == year].astype(float)
            rate = actual.mean()
            skill = 100000 * (1 - ((catboost_predictions[mode][str(year)] - actual) ** 2
                                   ).mean() / (rate * (1 - rate)))
            print(f"  catboost {mode:9s} {year}: standalone {skill:8.0f} "
                  f"[{time.time() - started:.0f}s]", flush=True)
            del model
            gc.collect()
        del work, features
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "context": context, "network": network,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        build_block(raw_frame, DRIFT_SHRINKAGE, "career").loc[order], 1.0)
    validation_frame = make_validation_frame()

    # The incumbent uses the stored V102/V116 predictions; the "career" arm retrains the
    # same configuration, so comparing the two arms isolates the prior from any
    # retraining noise. Both are reported against the stored incumbent.
    stored_form = joblib.load(
        "artifacts/v102_inseason_smoothing_predictions.joblib")["forms"][FEATURE_SHRINKAGE]
    stored_catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    baseline = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6),
               dict(common, form=stored_form, catboost=stored_catboost),
               oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(form, catboost):
        candidate = np.clip(
            blend6(tuple(BASE[n] for n in NAMES6),
                   dict(common, form=form, catboost=catboost), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
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
    print(f"\n{'candidate':>26} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    for label, form, catboost in (
            ("career_retrain_both", form_predictions["career"],
             catboost_predictions["career"]),
            ("projected_both", form_predictions["projected"],
             catboost_predictions["projected"]),
            ("projected_form_only", form_predictions["projected"], stored_catboost),
            ("projected_catboost_only", stored_form, catboost_predictions["projected"])):
        results[label] = evaluate(form, catboost)
        m = results[label]; t = m["three_season"]
        print(f"{label:>26} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
              f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:7.0%} "
              f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(l and results[l]) and l != "career_retrain_both"]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V153_projected_prior",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "prior_comparison": {
            "2022": {"career": 0.547341, "projected": 0.524781, "actual": 0.528920},
            "2023": {"career": 0.544313, "projected": 0.517275, "actual": 0.499957},
            "2024": {"career": 0.540175, "projected": 0.498733, "actual": 0.486105},
            "2025": {"career": 0.535228, "projected": 0.490536},
        },
        "rationale": (
            "The prior shrinks a current-season estimate toward a career average, which "
            "lags a league that fell from 0.5647 to 0.4861. The gap grows every season "
            "and shows up as an over-prediction that falls monotonically with pitcher "
            "experience. The replacement recovers the population's within-season rate by "
            "the same anchor differencing V92 built for individuals, then projects it."
        ),
        "why_not_the_drift_term": (
            "It reads the reconstruction at shrinkage 3, where the prior carries weight "
            "3/(inside_n + 3) and is nearly inert; rebuilding it moved the folds by "
            "+0.01, +0.30 and +0.02."
        ),
        "control_arm": (
            "`career_retrain_both` retrains the same configuration with the old prior, so "
            "the difference between the two arms isolates the prior from retraining noise."
        ),
        "form_config": FORM_CONFIG,
        "catboost_config": CATBOOST_CONFIG,
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "row_independent": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"form": form_predictions, "catboost": catboost_predictions},
                PREDICTIONS, compress=3)
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
