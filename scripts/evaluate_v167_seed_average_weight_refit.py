"""V167: the seed-averaged components are better on every season, and the blend rejects them.

V164 averaged the two neural components over independent seeds. Standalone, every season
of both components improved, with no exception:

    network        seed 42   2166 / -373 /  629      avg of 3   2355 / -200 /  709
    factorization  seed 42   2024 / -841 /  155      avg of 5   2202 / -442 /  423

and yet every seed arm lost 2024 in the blend, by 1.71 to 6.09 points. Components strictly
improve; the blend gets worse. That is not a reason to drop the idea, it is a diagnosis.

Averaging removes variance, which also **contracts the component's spread**. A blend weight
is not paid for accuracy alone -- it is paid for the independent component of a prediction
-- so a smoother, more accurate component wants a *different* weight from the noisy one it
replaces. The current weights were fitted in V119/V135/V139 against single-seed components,
so applying them to averaged components is a mis-specification of exactly the kind that has
paid four times in this project.

V54 diagnosed the same contraction in 2019-era tree models and stopped there, at "rejected".
It never asked the follow-up question.

This hill-climbs the six weights separately for each seed configuration, maximising the
weakest season against the fixed V161 baseline, then breaking ties on the three-season
average. The single-seed components run through the identical climb as a control: V163
showed that space to be locally optimal in all 60 pairwise directions, so the control must
find nothing, and if it finds something the harness is wrong rather than the weights.
"""

import json
import sys
import time
from itertools import permutations
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


OUTPUT = Path("artifacts/v167_seed_average_weight_refit_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SEEDS = (42, 1004, 2024, 777, 999)
MOVERS = ["form", "context", "network", "catboost", "factorization"]
STEPS = (0.05, 0.02, 0.01)
ROUNDS = 5
P = 100000.0 / 0.25
CONFIGS = {
    "control_seed42": (1, 1),
    "net3_fact3": (3, 3),
    "net3_fact5": (3, 5),
    "net5_fact5": (5, 5),
}


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    stored = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")

    def average(store, count):
        return {str(y): np.mean([store[s][str(y)] for s in SEEDS[:count]], axis=0)
                for y in YEARS}

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][20.0],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
    }
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    history_index = {year: np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < year]) for year in YEARS
        if year != 2022}
    train_frames = {y: calibration_frame.loc[i] for y, i in history_index.items()}
    valid_frames = {y: calibration_frame.loc[oof[str(y)]["row_index"]] for y in YEARS}

    def pipeline(parts, weights):
        raw = {year: sum(weights[n] * parts[n][str(year)] for n in NAMES6)
               for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
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

    def season_points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    base_parts = dict(fixed)
    base_parts["network"] = v160["network"][FEATURE_RELIABILITY]
    base_parts["factorization"] = v160["factorization"][FEATURE_RELIABILITY]
    baseline = pipeline(base_parts, BASE)
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    started = time.time()
    pipeline(base_parts, BASE)
    print(f"one evaluation costs {time.time() - started:.2f}s", flush=True)

    def climb(parts, label):
        weights = dict(BASE)
        best = season_points(pipeline(parts, weights), base_error)
        print(f"\n{label}: start  min {min(best):+7.2f}  avg {np.mean(best):+7.2f}  "
              f"{best[0]:+7.2f} {best[1]:+7.2f} {best[2]:+7.2f}", flush=True)
        seen = {tuple(round(weights[n], 4) for n in NAMES6)}
        for round_index in range(ROUNDS):
            improved = False
            for step in STEPS:
                trials = []
                for source, sink in permutations(MOVERS, 2):
                    if weights[source] < step - 1e-12:
                        continue
                    trial = dict(weights)
                    trial[source] = round(trial[source] - step, 4)
                    trial[sink] = round(trial[sink] + step, 4)
                    key = tuple(trial[n] for n in NAMES6)
                    if key in seen:
                        continue
                    seen.add(key)
                    points = season_points(pipeline(parts, trial), base_error)
                    trials.append((min(points), float(np.mean(points)), trial, points))
                if not trials:
                    continue
                trials.sort(key=lambda t: (t[0], t[1]), reverse=True)
                top = trials[0]
                if (top[0], top[1]) > (min(best) + 1e-9, float(np.mean(best)) + 1e-9):
                    weights, best = top[2], top[3]
                    improved = True
                    print(f"  round {round_index} step {step:.2f} -> "
                          f"min {min(best):+7.2f}  avg {np.mean(best):+7.2f}  "
                          f"{ {n: weights[n] for n in NAMES6} }", flush=True)
                    break
            if not improved:
                break
        return weights, best

    def full(parts, weights):
        candidate = pipeline(parts, weights)
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        metrics["weights"] = {n: weights[n] for n in NAMES6}
        return metrics

    results = {}
    for label, (net_count, fact_count) in CONFIGS.items():
        parts = dict(fixed)
        parts["network"] = (base_parts["network"] if net_count == 1
                            else average(stored["networks"], net_count))
        parts["factorization"] = (base_parts["factorization"] if fact_count == 1
                                  else average(stored["factorizations"], fact_count))
        weights, _ = climb(parts, label)
        results[label] = full(parts, weights)
        t = results[label]["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:16s} FINAL min {results[label]['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {results[label]['monthly_block_win_rate']:4.0%}  "
              f"{'SAFE' if safe else ''}", flush=True)

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l) and l != "control_seed42"]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V167_seed_average_weight_refit",
        "baseline": "V161 (Public 1053.2326413884)",
        "premise": ("V164's seed averaging improved every season of both neural components "
                    "standalone and still lost 2024 in the blend, because averaging "
                    "contracts spread and the weights were fitted against single-seed "
                    "components. V54 saw the same contraction in trees and stopped at "
                    "'rejected' without refitting."),
        "control": ("control_seed42 runs the identical climb on the single-seed "
                    "components; V163 showed that space locally optimal in 60 directions, "
                    "so this must find nothing"),
        "steps": list(STEPS),
        "results": {k: {"weights": v["weights"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsafe = {len(survivors)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
