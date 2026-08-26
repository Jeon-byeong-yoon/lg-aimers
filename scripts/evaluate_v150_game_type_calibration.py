"""V150: the calibration layer has no `game_type` segment, and `game_type` broke in 2023.

A per-`game_type` audit of the V138 blend found the whole 2023 anomaly:

    fold  type       n      actual   predicted     bias    skill   points lost
    2022     F  30,448    0.708749    0.688477  -0.0203     -153         20.23
    2023     F  25,686    0.472904    0.649607  +0.1767   -12053       1306.62
    2024     F  30,010    0.459280    0.470490  +0.0112      584          5.95

The F success rate fell from 0.7087 in 2022 to 0.4729 in 2023, a drop of 0.236, and the
2023-fold model -- trained on 2019-2022 where F ran near 0.70 -- predicted 0.6496 for it.
Those 25,686 rows cost 1,307 points on that fold by themselves. That is what the
reliability audit saw as a "dispersion collapse", what the shrinkage audit saw as
lambda* = 0.35, and what all nine rejected experiments were partly repairing: the 2023
top decile at predicted 0.653 against actual 0.485 *is* the F group.

By 2024 the models have a post-break season to learn from and the bias falls to +0.0112.
So the break is history, not headroom. Two things about it are still live.

First, F remains the worst-predicted group: 2024 skill of 584 against R's 902, on 12% of
the rows, and a residual bias of +0.0112 worth about 6 points. And the calibration layer
has no `game_type` term at all -- it holds a count segment at 0.75 and a pitcher-count
segment at 0.25, and nothing that knows F from R. For a subpopulation that is 12% of the
data, a different competition, and 68.5% of October, that is a gap rather than a choice.

Second, F is not evenly spread: 5.7% of March and 68.5% of October in 2024. So a
`game_type` correction is partly a late-season correction, which is the segment of the
season closest in character to what a 2025 model has to extrapolate into.

Recipes keep the convention that segment weights sum to one, since each segment estimates
the same residual and adding them all at full weight would over-correct.

Ranked by the weakest season. Note that 2022 is fixed at exactly zero for any
calibration-layer change, because no earlier season exists to fit residuals on, so the
monthly block win rate is capped near 65-70% for this whole candidate class -- a
measurement artifact identified in V149. Block rates are therefore reported for 2023-2024
separately alongside the pooled figure.

Pre-registered gate, unchanged since V132 except that the block floor is applied to the
2023-2024 blocks, for the stated structural reason:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate over 2023-2024 >= 75%
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


OUTPUT = Path("artifacts/v150_game_type_calibration_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
SEGMENTS = {
    "count": (["balls_before", "strikes_before"], 500.0),
    "pitcher_count": (["pitcher_id", "balls_before", "strikes_before"], 300.0),
    "game_type": (["game_type"], 2000.0),
    "game_type_count": (["game_type", "balls_before", "strikes_before"], 1000.0),
    "game_type_month": (["game_type", "game_month"], 1000.0),
}
# (count, pitcher_count, game_type, game_type_count, game_type_month); incumbent first.
RECIPES = {
    "incumbent": (0.75, 0.25, 0.00, 0.00, 0.00),
    "gt10": (0.65, 0.25, 0.10, 0.00, 0.00),
    "gt20": (0.55, 0.25, 0.20, 0.00, 0.00),
    "gt30": (0.45, 0.25, 0.30, 0.00, 0.00),
    "gtc10": (0.65, 0.25, 0.00, 0.10, 0.00),
    "gtc20": (0.55, 0.25, 0.00, 0.20, 0.00),
    "gtm10": (0.65, 0.25, 0.00, 0.00, 0.10),
    "gtm20": (0.55, 0.25, 0.00, 0.00, 0.20),
    "gt10_gtm10": (0.55, 0.25, 0.10, 0.00, 0.10),
}
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

    cache = {}

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

    names = list(SEGMENTS)

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
    development = validation_frame["validation_role"].to_numpy() == "development"
    season_of = validation_frame["season"].to_numpy()
    month_of = validation_frame["game_month"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    game_type = validation_frame["game_type"].to_numpy()

    def recent_block_rate(candidate):
        """Block win rate over 2023-2024 only; 2022 is exactly zero by construction."""
        gain = ((baseline - target) ** 2 - (candidate - target) ** 2)
        keep = development & (season_of >= 2023)
        frame = pd.DataFrame({"season": season_of[keep], "month": month_of[keep],
                              "gain": gain[keep]})
        monthly = frame.groupby(["season", "month"], observed=True)["gain"].mean()
        return float((monthly > 0).mean()), int(len(monthly))

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
        rate, blocks = recent_block_rate(candidate)
        metrics["recent_block_win_rate"] = rate
        metrics["recent_blocks"] = blocks
        # What the F group specifically does, since that is the motivation.
        f_mask = (game_type == "F") & (season_of == 2024)
        metrics["f_2024_bias"] = float(candidate[f_mask].mean() - target[f_mask].mean())
        return metrics

    def passes(m):
        t = m["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and m["recent_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"{'candidate':>14} {'min':>7} {'avg':>7} {'2022':>6} {'2023':>8} "
          f"{'2024':>7} {'blk23-24':>9} {'blkAll':>7} {'F24 bias':>9} {'pass':>5}")
    for label, recipe in RECIPES.items():
        if label == "incumbent":
            continue
        results[label] = evaluate(pipeline(recipe))
        results[label]["recipe"] = list(recipe)
        m = results[label]; t = m["three_season"]
        print(f"{label:>14} {m['min_season_points']:7.2f} {t['average_points']:7.2f} "
              f"{t['season_points'][0]:6.2f} {t['season_points'][1]:8.2f} "
              f"{t['season_points'][2]:7.2f} {m['recent_block_win_rate']:9.0%} "
              f"{m['monthly_block_win_rate']:7.0%} {m['f_2024_bias']:+9.6f} "
              f"{'YES' if passes(m) else '-':>5}", flush=True)

    incumbent_bias = float(
        baseline[(game_type == "F") & (season_of == 2024)].mean()
        - target[(game_type == "F") & (season_of == 2024)].mean())
    print(f"\nincumbent F 2024 bias {incumbent_bias:+.6f}")

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V150_game_type_calibration",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "discovery": {
            "f_rate_by_season": {"2019": 0.689250, "2020": 0.587774, "2021": 0.703840,
                                 "2022": 0.708749, "2023": 0.472904, "2024": 0.459280},
            "blend_bias_on_f": {"2022": -0.020272, "2023": 0.176703, "2024": 0.011210},
            "points_lost_on_f": {"2022": 20.23, "2023": 1306.62, "2024": 5.95},
            "reading": (
                "The F success rate fell 0.236 between 2022 and 2023 and the 2023-fold "
                "model, trained where F ran near 0.70, predicted 0.6496 for it. Those "
                "25,686 rows cost 1,307 points on that fold alone. That is the "
                "'dispersion collapse', the lambda* of 0.35, and what all nine rejected "
                "experiments were partly repairing."
            ),
        },
        "motivation": (
            "F is still the worst-predicted group on 2024 -- skill 584 against R's 902 on "
            "12% of rows -- and the calibration layer has no game_type term at all. F is "
            "also 5.7% of March and 68.5% of October, so the correction is partly a "
            "late-season one."
        ),
        "block_floor_note": (
            "2022 is exactly zero for any calibration-layer change, so pooled block rate "
            "is capped near 65-70% for this class (V149). The floor is applied to the "
            "2023-2024 blocks; the pooled figure is reported alongside."
        ),
        "segments": {k: {"columns": v[0], "smoothing": v[1]}
                     for k, v in SEGMENTS.items()},
        "recipes": {k: list(v) for k, v in RECIPES.items()},
        "incumbent_f_2024_bias": incumbent_bias,
        "results": {k: {"recipe": v["recipe"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "recent_block_win_rate": v["recent_block_win_rate"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"],
                        "f_2024_bias": v["f_2024_bias"]}
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
