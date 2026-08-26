"""V135: re-open the weight axes V119 closed, using the corrected instrument.

V119 concluded that the v17, Context and network axes were closed: each ladder's
parabola put its vertex within 0.015 of the value already in use. That conclusion was
drawn entirely from the 2024-only reading, which three leaderboard results have since
shown to be biased low by a factor that grows as the estimate shrinks (1.20x, 2.59x,
3.75x). An axis declared closed because its 2024 movement looked like noise is exactly
the kind of finding the correction should be expected to overturn, and re-reading those
axes is overdue.

V134 gave the first hint. Its second stage laddered Context down while also swapping
CatBoost's feature set, and dropping Context from 0.21 to 0.09 moved the three-season
average to +18.08 with a lower bound of +12.42. But that candidate changed two things at
once, so the credit is unattributable -- the gain may be entirely the weight and nothing
to do with the features. This isolates the weights, on the cached incumbent predictions.

Every candidate is a single explicit transfer: take d from one component, give it to
another, leave the rest untouched. V120 is why. There, moving CatBoost up with the other
weights rescaled proportionally broke 2023, while funding the same increase from v17
first improved 2023 monotonically from +4.8 to +23.8 -- same axis, opposite conclusion,
because proportional rescaling quietly bills the largest component. With v17 now retired
to zero there are four live components and twelve ordered pairs, and naming the payer in
each is the only way the result means anything.

Pre-registered gate, unchanged since V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import json
import math
import sys
from itertools import permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v119_five_way_weight_refit import NAMES, blend
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v135_pairwise_transfer_metrics.json")
BASE = dict(zip(NAMES, (0.00, 0.32, 0.21, 0.20, 0.27)))
LIVE = ("form", "context", "network", "catboost")
STEPS = (0.04, 0.08, 0.12)
EXTENSION = (0.16, 0.20, 0.24)
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def three_season(metrics):
    seasons = metrics["season_bootstrap"]
    means = [seasons[str(y)]["mean"] for y in YEARS]
    errors = [(seasons[str(y)]["ci95_high"] - seasons[str(y)]["ci95_low"]) / 3.9199
              for y in YEARS]
    average = sum(means) / 3.0
    combined = math.sqrt(sum(e * e for e in errors)) / 3.0
    return {"average_points": average * P, "se_points": combined * P,
            "ci95_low_points": (average - 1.96 * combined) * P,
            "season_points": [m * P for m in means]}


def transfer(source, target, amount):
    weights = dict(BASE)
    weights[source] = round(weights[source] - amount, 6)
    weights[target] = round(weights[target] + amount, 6)
    if weights[source] < 0:
        return None
    return tuple(weights[name] for name in NAMES)


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
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend(tuple(BASE[n] for n in NAMES), parts, oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(weights):
        candidate = np.clip(
            blend(weights, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["weights"] = {n: w for n, w in zip(NAMES, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    def show(label, metrics):
        t = metrics["three_season"]
        print(f"  {label:26s} avg {t['average_points']:+7.2f}  "
              f"CIlo {t['ci95_low_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+8.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blocks {metrics['monthly_block_win_rate']:5.0%}  "
              f"{'YES' if passes(metrics) else '-'}", flush=True)

    results = {}
    print("pairwise transfers (take from -> give to):", flush=True)
    for source, target in permutations(LIVE, 2):
        for amount in STEPS:
            weights = transfer(source, target, amount)
            if weights is None:
                continue
            label = f"{source}->{target}_{amount:.2f}"
            results[label] = evaluate(weights)
            results[label]["transfer"] = {"from": source, "to": target,
                                          "amount": amount}
            show(label, results[label])

    ranked = sorted(results, key=lambda l: -results[l]["three_season"]["ci95_low_points"])
    best = ranked[0]
    direction = results[best]["transfer"]
    print(f"\nbest direction: {direction['from']} -> {direction['to']}; extending",
          flush=True)
    for amount in EXTENSION:
        weights = transfer(direction["from"], direction["to"], amount)
        if weights is None:
            print(f"  {direction['from']} exhausted at {amount:.2f}", flush=True)
            break
        label = f"{direction['from']}->{direction['to']}_{amount:.2f}"
        results[label] = evaluate(weights)
        results[label]["transfer"] = {"from": direction["from"],
                                      "to": direction["to"], "amount": amount}
        show(label, results[label])

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible,
                   key=lambda l: results[l]["three_season"]["ci95_low_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V135_pairwise_weight_transfer",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27), Public 1047.03653",
        "rationale": (
            "V119 closed the v17, Context and network axes on the 2024-only reading, "
            "which three leaderboard results have since shown biased low. An axis "
            "declared closed because its 2024 movement looked like noise is what the "
            "correction should overturn. V134 hinted at it -- Context from 0.21 to 0.09 "
            "gave a three-season average of +18.08 -- but changed the feature set at the "
            "same time, so credit was unattributable."
        ),
        "design": (
            "Every candidate is one explicit transfer: d from one component to another, "
            "others untouched. V120 showed proportional rescaling quietly bills the "
            "largest component and reversed a conclusion on the same axis."
        ),
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}"},
        "live_components": list(LIVE),
        "steps": list(STEPS),
        "results": {k: {"weights": v["weights"], "transfer": v["transfer"],
                        "three_season": v["three_season"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ntop 10 by three-season lower bound:")
    for label in sorted(results,
                        key=lambda l: -results[l]["three_season"]["ci95_low_points"])[:10]:
        show(label, results[label])
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
