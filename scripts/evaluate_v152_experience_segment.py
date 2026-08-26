"""V152: the calibration has no experience segment, and low-experience rows are biased up.

Splitting the 2024 fold by pitcher experience inside each `game_type` shows the
over-prediction is not an F problem at all -- it is an experience problem that F merely
concentrates:

    type   experience        n     rate      bias
       F        0-200    6,156   0.4472   +0.0157
       F       200-1k    9,504   0.4569   +0.0157
       F        1k-3k    8,694   0.4603   +0.0081
       F        3k-8k    5,428   0.4740   +0.0043
       R        0-200    8,457   0.4738   +0.0077
       R       200-1k   34,617   0.4755   +0.0038
       R        1k-3k   61,301   0.4867   -0.0013
       R          8k+   39,364   0.5080   -0.0006

The bias falls monotonically with experience in both competitions and crosses zero
around a thousand career pitches. V151 corrected `game_type` and lost 2.44 points on the
leaderboard; this says it was correcting the proxy rather than the cause.

The mechanism is in the deployed inference path. The in-season reconstruction shrinks
toward a prior,

    inside_rate = (inside_sum + prior * shrinkage) / (inside_n + shrinkage)

and `build_priors` sets that prior to the mean of the *career* as-of rate column. That is
a career quantity, and it lags a falling league. Measured:

    fold   history      prior    target season      gap
    2022   2019-2021   0.547341     0.528920     +0.0184
    2023   2019-2022   0.544313     0.499957     +0.0444
    2024   2019-2023   0.540175     0.486105     +0.0541
    2025   2019-2024   0.535228     ~0.470       +0.060 to +0.073

The deployed prior is 0.535228 against a projected 2025 level of 0.462-0.475. At
`inside_n = 100` the prior carries weight 20/120 = 0.167, so a +0.07 error produces a
+0.0117 bias -- which is what the low-experience rows show.

Correcting the prior itself means rebuilding the features and retraining every component
that consumes them. This tests the cheaper question first: does a calibration segment on
experience capture the bias? The segments are `asof_pitcher_n` bins, and bins crossed with
`game_type`, since F concentrates the low-experience rows.

An honest note on the evidence this class can produce. 2022 receives no calibration at all
-- no earlier season exists to fit residuals on -- so its gain is exactly zero by
construction, and the pooled monthly block rate is capped near 65-70% (V149). The rule
being applied is therefore "no fold shows a loss and at least one shows a gain", which is
what V144 (-1.86 on 2022) and V151 (-32.36 on 2023) both violated: those overrode folds
carrying real losses. A fold that is structurally zero carries no adverse evidence.

Pre-registered gate:
  * no season mean below zero
  * at least one season mean >= +3 points
  * three-season equally weighted average CI low > 0
  * monthly block win rate over the seasons that can move (2023-2024) >= 75%
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


OUTPUT = Path("artifacts/v152_experience_segment_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
EDGES = [-1, 200, 1000, 3000, 8000, np.inf]
LABELS = ["0-200", "200-1k", "1k-3k", "3k-8k", "8k+"]
SEGMENTS = {
    "count": (["balls_before", "strikes_before"], 500.0),
    "pitcher_count": (["pitcher_id", "balls_before", "strikes_before"], 300.0),
    "experience": (["experience_bin"], 2000.0),
    "experience_type": (["experience_bin", "game_type"], 1000.0),
}
RECIPES = {
    "incumbent": (0.75, 0.25, 0.00, 0.00),
    "exp05": (0.70, 0.25, 0.05, 0.00),
    "exp10": (0.65, 0.25, 0.10, 0.00),
    "exp15": (0.60, 0.25, 0.15, 0.00),
    "exp20": (0.55, 0.25, 0.20, 0.00),
    "expt05": (0.70, 0.25, 0.00, 0.05),
    "expt10": (0.65, 0.25, 0.00, 0.10),
    "expt15": (0.60, 0.25, 0.00, 0.15),
    "expt20": (0.55, 0.25, 0.00, 0.20),
}
BLOCK_FLOOR = 0.75
GAIN_FLOOR = 3.0
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id")
    # A per-row function of an official column, so it is available for a 2025 row and
    # needs nothing from any other evaluation row.
    raw_frame["experience_bin"] = pd.cut(
        raw_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)

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
            raw_frame.drop(columns="experience_bin"),
            shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)

    cache, names = {}, list(SEGMENTS)

    def correction(year, name):
        key = (year, name)
        if key not in cache:
            columns, smoothing = SEGMENTS[name]
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            cache[key] = segment_correction(
                raw_frame.loc[index], residual,
                raw_frame.loc[oof[str(year)]["row_index"]], columns, smoothing)
        return cache[key]

    def shift(year):
        history = [h for h in YEARS if h < year]
        targets = np.concatenate([oof[str(h)]["target"].astype(float)
                                  for h in history])
        return float((targets - np.concatenate([raw[h] for h in history])).mean())

    def pipeline(recipe):
        assert abs(sum(recipe) - 1.0) < 1e-9, recipe
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            total = raw[year] + shift(year)
            for weight, name in zip(recipe, names):
                if weight:
                    total = total + weight * correction(year, name)
            pieces.append(np.clip(total, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(RECIPES["incumbent"])
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2
    season_of = validation_frame["season"].to_numpy()
    month_of = validation_frame["game_month"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    development = validation_frame["validation_role"].to_numpy() == "development"
    bin_of = validation_frame["experience_bin"].to_numpy()

    print("segment values fitted on 2022-2023 (what the 2024 fold uses):")
    print(correction(2024, "experience")[:0].shape and "", end="")
    columns, smoothing = SEGMENTS["experience"]
    history = [2022, 2023]
    index = np.concatenate([oof[str(h)]["row_index"] for h in history])
    residual = np.concatenate(
        [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
    work = raw_frame.loc[index, columns].copy()
    work["_r"] = residual
    stats = work.groupby(columns)["_r"].agg(["mean", "size"])
    print(stats.to_string())

    def recent_block_rate(candidate):
        gain = ((baseline - target) ** 2 - (candidate - target) ** 2)
        keep = development & (season_of >= 2023)
        frame = pd.DataFrame({"s": season_of[keep], "m": month_of[keep],
                              "g": gain[keep]})
        monthly = frame.groupby(["s", "m"], observed=True)["g"].mean()
        return float((monthly > 0).mean())

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
        metrics["recent_block_win_rate"] = recent_block_rate(candidate)
        metrics["low_n_bias_2024"] = float(
            candidate[(season_of == 2024) & np.isin(bin_of, ["0-200", "200-1k"])].mean()
            - target[(season_of == 2024) & np.isin(bin_of, ["0-200", "200-1k"])].mean())
        return metrics

    def passes(m):
        t = m["three_season"]
        return (all(v >= -1e-9 for v in t["season_points"])
                and max(t["season_points"]) >= GAIN_FLOOR
                and t["ci95_low_points"] > 0
                and m["recent_block_win_rate"] >= BLOCK_FLOOR)

    incumbent_low_n = float(
        baseline[(season_of == 2024) & np.isin(bin_of, ["0-200", "200-1k"])].mean()
        - target[(season_of == 2024) & np.isin(bin_of, ["0-200", "200-1k"])].mean())
    print(f"\nincumbent low-experience 2024 bias: {incumbent_low_n:+.6f}")
    print(f"\n{'candidate':>12} {'min':>7} {'avg':>7} {'2022':>6} {'2023':>8} "
          f"{'2024':>7} {'blk23-24':>9} {'lowN bias':>10} {'pass':>5}")
    results = {}
    for label, recipe in RECIPES.items():
        if label == "incumbent":
            continue
        results[label] = evaluate(pipeline(recipe))
        results[label]["recipe"] = list(recipe)
        m = results[label]; t = m["three_season"]
        print(f"{label:>12} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:6.2f} {t['season_points'][1]:8.2f} "
              f"{t['season_points'][2]:7.2f} {m['recent_block_win_rate']:9.0%} "
              f"{m['low_n_bias_2024']:+10.6f} {'YES' if passes(m) else '-':>5}",
              flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V152_experience_segment",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "discovery": {
            "bias_by_experience_2024": {
                "F": {"0-200": 0.0157, "200-1k": 0.0157, "1k-3k": 0.0081,
                      "3k-8k": 0.0043},
                "R": {"0-200": 0.0077, "200-1k": 0.0038, "1k-3k": -0.0013,
                      "8k+": -0.0006}},
            "prior_versus_target": {
                "2022": {"prior": 0.547341, "actual": 0.528920, "gap": 0.0184},
                "2023": {"prior": 0.544313, "actual": 0.499957, "gap": 0.0444},
                "2024": {"prior": 0.540175, "actual": 0.486105, "gap": 0.0541},
                "2025": {"prior": 0.535228, "projected": "0.462-0.475"}},
            "reading": (
                "The in-season reconstruction shrinks toward the mean of the career as-of "
                "rate column, a career quantity that lags a falling league. V151 "
                "corrected game_type and lost 2.44 points; game_type was the proxy and "
                "experience is the cause."
            ),
        },
        "gate": {
            "no_season_below_zero": True,
            "at_least_one_season": f">= {GAIN_FLOOR}",
            "three_season_ci95_low_points": "> 0",
            "block_floor_seasons": "2023-2024",
            "block_floor": BLOCK_FLOOR,
            "rationale": (
                "2022 is exactly zero for any calibration change, so the rule is 'no "
                "fold shows a loss and at least one shows a gain'. V144 (-1.86 on 2022) "
                "and V151 (-32.36 on 2023) both overrode folds carrying real losses; a "
                "structurally zero fold carries no adverse evidence."
            ),
        },
        "segments": {k: {"columns": v[0], "smoothing": v[1]}
                     for k, v in SEGMENTS.items()},
        "recipes": {k: list(v) for k, v in RECIPES.items()},
        "incumbent_low_n_bias_2024": incumbent_low_n,
        "results": {k: {"recipe": v["recipe"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "recent_block_win_rate": v["recent_block_win_rate"],
                        "low_n_bias_2024": v["low_n_bias_2024"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "row_independent": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
