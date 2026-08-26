"""V145: change how the components are combined, and add a monotone recalibration.

Every experiment so far has changed *what* goes into the blend or *how much* of it. The
two things never touched are the combination rule and the shape of the calibration.

Combination rule. All six components are averaged in probability space, p = sum(w_i p_i).
The alternative is to average in log-odds space, sigmoid(sum(w_i logit(p_i))), which is
the natural rule for combining independent evidence and produces sharper predictions.
Under a Brier objective, linear averaging minimises variance while log-odds averaging
wins when the components are individually well calibrated but jointly under-confident --
which is exactly what averaging six correlated models tends to produce. The two are
compared, plus mixtures, since nothing forces a corner.

Calibration shape. The layer is a global shift plus two segment averages, so it can move
the level and it can move a count or a pitcher-count cell, but it cannot bend the
probability scale. If the blend is systematically under-confident -- predictions too
close to the base rate at both ends -- no amount of shifting fixes it, and that failure
mode is invisible to every diagnostic run so far, all of which looked at means.

The recalibration used is beta calibration: a logistic regression on log(p) and
log(1-p), three parameters, strictly monotone. Three parameters is the point. V133 tried
a *learned model* on the residuals and it collapsed, moving the 2024 level by -0.0188
because it re-learned league drift that the global shift had already removed. A
three-parameter monotone map has no room to do that; it can only bend the scale.

Both are fitted the way everything else here is fitted: on the seasons before the fold,
never on the fold itself. 2022 gets neither, since no earlier season exists, so its gain
is zero by construction for the recalibration arm -- the same limitation V133 carried.

Ranked by the weakest season, the instrument validated on both sides of zero across four
submissions (+9.9 -> +25.67, +5.01 -> +18.80, +0.64 -> +3.51, -1.86 -> -1.14).

Pre-registered gate, unchanged since V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v145_combination_rule_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
LOGIT_MIX = (0.0, 0.25, 0.50, 0.75, 1.0)
EPSILON = 1e-6
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def logit(p):
    q = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(q / (1.0 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def combine(weights, parts, year, logit_mix):
    """Linear and log-odds averages of the same weights, mixed by `logit_mix`."""
    key = str(year)
    linear = sum(w * parts[n][key] for w, n in zip(weights, NAMES6))
    if logit_mix == 0.0:
        return linear
    odds = sigmoid(sum(w * logit(np.asarray(parts[n][key], dtype=float))
                       for w, n in zip(weights, NAMES6)))
    return (1.0 - logit_mix) * linear + logit_mix * odds


def fit_beta(prediction, target):
    """Beta calibration: monotone, three parameters, fitted by IRLS on log p, log(1-p)."""
    q = np.clip(prediction, EPSILON, 1.0 - EPSILON)
    design = np.column_stack([np.log(q), -np.log(1.0 - q), np.ones(len(q))])
    beta = np.zeros(3)
    for _ in range(40):
        eta = design @ beta
        mu = sigmoid(eta)
        weight = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = design.T @ (target - mu)
        hessian = design.T @ (design * weight[:, None])
        step = np.linalg.solve(hessian + 1e-8 * np.eye(3), gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def apply_beta(prediction, beta):
    q = np.clip(prediction, EPSILON, 1.0 - EPSILON)
    design = np.column_stack([np.log(q), -np.log(1.0 - q), np.ones(len(q))])
    return sigmoid(design @ beta)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id")

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
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    weights = tuple(BASE[n] for n in NAMES6)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()

    def pipeline(logit_mix, recalibrate):
        """Reproduce the deployed pipeline with the combination rule swapped."""
        raw = {year: combine(weights, parts, year, logit_mix) for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            targets = np.concatenate([oof[str(h)]["target"].astype(float)
                                      for h in history])
            predictions = np.concatenate([raw[h] for h in history])
            residual = targets - predictions
            train_frame = raw_frame.loc[index]
            valid_frame = raw_frame.loc[oof[str(year)]["row_index"]]
            count = segment_correction(
                train_frame, residual, valid_frame,
                ["balls_before", "strikes_before"], 500)
            pitcher_count = segment_correction(
                train_frame, residual, valid_frame,
                ["pitcher_id", "balls_before", "strikes_before"], 300)
            current = np.clip(
                raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)
            if recalibrate:
                # Fitted on the prior seasons after their own calibration, so the map
                # sees the same kind of input it will be applied to.
                past = []
                for h in history:
                    if h == 2022:
                        past.append(np.clip(raw[h], 0, 1))
                    else:
                        inner = [g for g in YEARS if g < h]
                        inner_index = np.concatenate(
                            [oof[str(g)]["row_index"] for g in inner])
                        inner_target = np.concatenate(
                            [oof[str(g)]["target"].astype(float) for g in inner])
                        inner_raw = np.concatenate([raw[g] for g in inner])
                        inner_residual = inner_target - inner_raw
                        past.append(np.clip(raw[h] + inner_residual.mean(), 0, 1))
                beta = fit_beta(np.concatenate(past), targets)
                current = apply_beta(current, beta)
            pieces.append(current)
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(0.0, False)
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
    print(f"{'candidate':>22} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    for logit_mix in LOGIT_MIX:
        for recalibrate in (False, True):
            if logit_mix == 0.0 and not recalibrate:
                continue
            label = f"logit{logit_mix:.2f}" + ("_beta" if recalibrate else "")
            candidate = pipeline(logit_mix, recalibrate)
            results[label] = evaluate(candidate)
            results[label]["logit_mix"] = logit_mix
            results[label]["recalibrate"] = recalibrate
            m = results[label]; t = m["three_season"]
            print(f"{label:>22} {m['min_season_points']:7.2f} "
                  f"{t['average_points']:7.2f} {t['season_points'][0]:7.2f} "
                  f"{t['season_points'][1]:8.2f} {t['season_points'][2]:7.2f} "
                  f"{m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V145_combination_rule",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "rationale": (
            "Every experiment so far changed what goes into the blend or how much. The "
            "combination rule (linear in probability) and the calibration shape (shift "
            "plus segment means, which cannot bend the probability scale) have never "
            "been touched. Under-confidence is invisible to every diagnostic run here, "
            "all of which looked at means."
        ),
        "why_three_parameters": (
            "V133's learned residual model collapsed, moving the 2024 level by -0.0188 "
            "because it re-learned league drift the global shift had already removed. A "
            "three-parameter monotone map cannot do that; it can only bend the scale."
        ),
        "logit_mix": list(LOGIT_MIX),
        "ranking": "maximum of the weakest season among survivors",
        "results": {k: {"logit_mix": v["logit_mix"], "recalibrate": v["recalibrate"],
                        "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
