"""V146: strengthen the pitcher-by-count correction, where the oracle says signal remains.

A reliability audit of the V138 blend changed the map. Measuring what oracles at
different levels of knowledge would score on each fold:

    fold    blend    oracle: pitcher rate    oracle: pitcher x count
    2022     2440                    1822                       3514
    2023     -548                    1111                       2817
    2024      903                     991                       2741

On 2024 the blend sits 88 points below an oracle that knows each pitcher's true season
rate. Pitcher-level information is close to exhausted -- which is what six components
built almost entirely out of pitcher summaries would predict. The pitcher-by-count oracle
is far higher, and while it is inflated (roughly 53 rows per cell, so it fits noise), the
pitcher oracle is not (roughly 640 rows per pitcher) and the ordering is unambiguous: the
interaction holds signal the marginals do not.

The pipeline uses that interaction in exactly one place, and timidly. The calibration
layer is

    shift + 0.75 * lookup(balls, strikes | smoothing 500)
          + 0.25 * lookup(pitcher, balls, strikes | smoothing 300)

so the single most informative interaction available gets a quarter of the weight and the
heaviest smoothing relative to its cell size. Those constants were last fitted in V98,
against a three-component blend, before the network, CatBoost and the factorization
network existed -- the same staleness that made weight refits pay four times.

The grid moves both weights and both smoothings, and adds a coarser third segment.
`pitcher x strikes` has three cells per pitcher instead of twelve, so it estimates far
more stably while keeping the part of the interaction that matters most: whether a
pitcher's control holds up when he is behind versus ahead.

Everything runs on cached component predictions; only the calibration is recomputed.

Ranked by the weakest season. The reliability audit also explains why: the 2023 fold
carries about 659 points recoverable by shrinking the prediction spread alone, against
1.84 on 2024, so any 2023 gain is contaminated by a one-off dispersion anomaly and the
minimum is the only reading that ignores it.

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


OUTPUT = Path("artifacts/v146_segment_structure_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
COUNT = ["balls_before", "strikes_before"]
PITCHER_COUNT = ["pitcher_id", "balls_before", "strikes_before"]
PITCHER_STRIKES = ["pitcher_id", "strikes_before"]
# (count weight, pitcher-count weight, pitcher-strikes weight); the incumbent is first.
RECIPES = [
    (0.75, 0.25, 0.00), (0.65, 0.35, 0.00), (0.55, 0.45, 0.00), (0.45, 0.55, 0.00),
    (0.65, 0.25, 0.10), (0.55, 0.25, 0.20), (0.55, 0.35, 0.10), (0.45, 0.35, 0.20),
    (0.35, 0.45, 0.20),
]
PITCHER_COUNT_SMOOTHING = (100.0, 200.0, 300.0)
PITCHER_STRIKES_SMOOTHING = 500.0
COUNT_SMOOTHING = 500.0
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


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
    raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
           for year in YEARS}
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()

    # Segment corrections depend only on (columns, smoothing), so each distinct one is
    # computed once and reused across every recipe that references it.
    cache = {}

    def correction(year, columns, smoothing):
        key = (year, tuple(columns), smoothing)
        if key not in cache:
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            targets = np.concatenate([oof[str(h)]["target"].astype(float)
                                      for h in history])
            residual = targets - np.concatenate([raw[h] for h in history])
            cache[key] = segment_correction(
                raw_frame.loc[index], residual,
                raw_frame.loc[oof[str(year)]["row_index"]], columns, smoothing)
        return cache[key]

    def shift(year):
        history = [h for h in YEARS if h < year]
        targets = np.concatenate([oof[str(h)]["target"].astype(float)
                                  for h in history])
        return float((targets - np.concatenate([raw[h] for h in history])).mean())

    def pipeline(recipe, pitcher_count_smoothing):
        w_count, w_pitcher_count, w_pitcher_strikes = recipe
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            total = raw[year] + shift(year)
            if w_count:
                total = total + w_count * correction(year, COUNT, COUNT_SMOOTHING)
            if w_pitcher_count:
                total = total + w_pitcher_count * correction(
                    year, PITCHER_COUNT, pitcher_count_smoothing)
            if w_pitcher_strikes:
                total = total + w_pitcher_strikes * correction(
                    year, PITCHER_STRIKES, PITCHER_STRIKES_SMOOTHING)
            pieces.append(np.clip(total, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(RECIPES[0], 300.0)
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
    print(f"{'candidate':>28} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    for recipe in RECIPES:
        for smoothing in PITCHER_COUNT_SMOOTHING:
            if recipe == RECIPES[0] and smoothing == 300.0:
                continue
            label = (f"c{recipe[0]:.2f}_pc{recipe[1]:.2f}_ps{recipe[2]:.2f}"
                     f"_s{smoothing:.0f}")
            candidate = pipeline(recipe, smoothing)
            results[label] = evaluate(candidate)
            results[label]["recipe"] = list(recipe)
            results[label]["pitcher_count_smoothing"] = smoothing
            m = results[label]; t = m["three_season"]
            print(f"{label:>28} {m['min_season_points']:7.2f} "
                  f"{t['average_points']:7.2f} {t['season_points'][0]:7.2f} "
                  f"{t['season_points'][1]:8.2f} {t['season_points'][2]:7.2f} "
                  f"{m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V146_segment_structure",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "oracle_audit": {
            "2022": {"blend": 2440, "pitcher": 1822, "pitcher_count": 3514},
            "2023": {"blend": -548, "pitcher": 1111, "pitcher_count": 2817},
            "2024": {"blend": 903, "pitcher": 991, "pitcher_count": 2741},
            "reading": (
                "On 2024 the blend is 88 points below a pitcher-rate oracle, so "
                "pitcher-level information is near exhaustion. The pitcher-by-count "
                "oracle is inflated by roughly 53 rows per cell, but the pitcher oracle "
                "at roughly 640 rows is not, and the ordering is unambiguous."
            ),
        },
        "dispersion_audit": {
            "optimal_shrinkage_lambda": {"2022": 1.096, "2023": 0.3516, "2024": 1.047},
            "recoverable_points": {"2022": 18.85, "2023": 658.99, "2024": 1.84},
            "reading": (
                "2022 and 2024 want no shrinkage; only 2023 does, by a factor of three. "
                "Any 2023 gain is contaminated by a one-off dispersion anomaly worth 659 "
                "points, which is why nine experiments all showed large 2023 gains and "
                "why the minimum season is the only trustworthy reading."
            ),
        },
        "incumbent": "0.75 count (smoothing 500) + 0.25 pitcher-count (smoothing 300)",
        "recipes": [list(r) for r in RECIPES],
        "pitcher_count_smoothing": list(PITCHER_COUNT_SMOOTHING),
        "results": {k: {"recipe": v["recipe"],
                        "pitcher_count_smoothing": v["pitcher_count_smoothing"],
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
