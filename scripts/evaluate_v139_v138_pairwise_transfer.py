"""V139: re-read the weight landscape on the V138 blend, ranking by the weakest season.

V138 scored 1050.5511, up +3.51 from V122. That makes five adopted candidates in a row
that improved the leaderboard, so the gate's *sign* judgment keeps holding. Its magnitude
judgment does not, and the fourth data point says something specific:

    submission        2024 est   3-season avg   actual   3-season ratio   max/min
    V117 CatBoost         +9.9         +31.4    +25.67            0.82        7.2
    V122 weights         +5.01        +10.35    +18.80            1.82        4.1
    V138 FM slot         +0.64        +25.08     +3.51            0.14      107.0

The three-season ratio falls monotonically as the seasonal gains grow more
heterogeneous. Three points is not a law, but the mechanism is plain: a candidate whose
gains differ across folds by a factor of a hundred is exploiting something fold-specific,
while a general improvement shows up similarly everywhere.

So the V131 correction was half right. Banning row-level pooling stopped the *row count*
of a fold from dominating; it did nothing about a fold's *magnitude* dominating. V138 had
2023 at +68.26 against 2024 at +0.64 and still averaged +25.08. Equal weighting fixes one
of the two ways 2023 can take over, and V95's warning applies to the other.

The weakest season behaves far better as a magnitude read:

    candidate    min season   actual   ratio
    V117               9.9     25.67     2.6
    V122              5.01     18.80     3.8
    V138              0.64      3.51     5.5

All four under-predict, within a 2.6-5.5 band, where the average swung 0.14 to 1.82. So
candidates are ranked here by minimum season, understood as a conservative lower bound
rather than an estimate. The gate is unchanged -- it already required every season
non-negative, which is the same quantity being positive; only the *ordering* among
survivors changes.

What is measured: V138 moved Context to 0.14 and added the factorization network at 0.07,
so the landscape V135 mapped no longer applies. If the factorization network covers what
Context was doing, Context should now be reducible further, and the handover ladder was
still rising monotonically when the per-season condition stopped it.

Twelve ordered pairs among the five live components, each an explicit single transfer --
V120's lesson that naming the payer is what makes an axis result mean anything.
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
from evaluate_v137_context_slot_replacement import NAMES6, blend6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v139_v138_transfer_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
LIVE = ("form", "context", "network", "catboost", "factorization")
STEPS = (0.02, 0.04, 0.07)
EXTENSION = (0.10, 0.13)
CATBOOST_SOURCE = "no_te_strong"
FACTORIZATION_SOURCE = "latent8"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
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
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6), parts, oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(weights):
        candidate = np.clip(
            blend6(weights, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["min_season_points"] = min(metrics["three_season"]["season_points"])
        metrics["heterogeneity"] = (
            max(metrics["three_season"]["season_points"])
            / min(metrics["three_season"]["season_points"])
            if min(metrics["three_season"]["season_points"]) > 0 else None)
        metrics["weights"] = {n: round(w, 6) for n, w in zip(NAMES6, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    def show(label, m):
        t = m["three_season"]
        het = "" if m["heterogeneity"] is None else f"{m['heterogeneity']:6.1f}"
        print(f"  {label:30s} min {m['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+8.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"het {het:>6}  blocks {m['monthly_block_win_rate']:5.0%}  "
              f"{'YES' if passes(m) else '-'}", flush=True)

    def make(source, target, amount):
        weights = dict(BASE)
        weights[source] = round(weights[source] - amount, 6)
        weights[target] = round(weights[target] + amount, 6)
        if weights[source] < -1e-12:
            return None
        vector = np.array([weights[n] for n in NAMES6], dtype=float)
        assert abs(vector.sum() - 1.0) < 1e-6, vector
        return tuple(vector / vector.sum())

    results = {}
    print("pairwise transfers on the V138 blend (take from -> give to):", flush=True)
    for source, target in permutations(LIVE, 2):
        for amount in STEPS:
            weights = make(source, target, amount)
            if weights is None:
                continue
            label = f"{source}->{target}_{amount:.2f}"
            results[label] = evaluate(weights)
            results[label]["transfer"] = {"from": source, "to": target,
                                          "amount": amount}
            show(label, results[label])

    eligible = [l for l in results if passes(results[l])]
    if eligible:
        best = max(eligible, key=lambda l: results[l]["min_season_points"])
        direction = results[best]["transfer"]
        print(f"\nbest eligible direction by weakest season: "
              f"{direction['from']} -> {direction['to']}; extending", flush=True)
        for amount in EXTENSION:
            weights = make(direction["from"], direction["to"], amount)
            if weights is None:
                print(f"  {direction['from']} exhausted at {amount:.2f}", flush=True)
                break
            label = f"{direction['from']}->{direction['to']}_{amount:.2f}"
            results[label] = evaluate(weights)
            results[label]["transfer"] = {"from": direction["from"],
                                          "to": direction["to"], "amount": amount}
            show(label, results[label])
        eligible = [l for l in results if passes(results[l])]
    else:
        print("\nno eligible transfer; nothing to extend", flush=True)

    promoted = max(eligible, key=lambda l: results[l]["min_season_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V139_v138_pairwise_transfer",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "instrument_note": {
            "record": [
                {"version": "V117", "est_2024": 9.9, "avg_3season": 31.4,
                 "actual": 25.67, "ratio_3season": 0.82, "heterogeneity": 7.2,
                 "min_season": 9.9},
                {"version": "V122", "est_2024": 5.01, "avg_3season": 10.35,
                 "actual": 18.80, "ratio_3season": 1.82, "heterogeneity": 4.1,
                 "min_season": 5.01},
                {"version": "V138", "est_2024": 0.64, "avg_3season": 25.08,
                 "actual": 3.51, "ratio_3season": 0.14, "heterogeneity": 107.0,
                 "min_season": 0.64},
            ],
            "reading": (
                "The three-season ratio falls monotonically as heterogeneity rises. The "
                "V131 correction stopped a fold's row count from dominating but not its "
                "magnitude: V138 averaged +25.08 with 2023 at +68.26 and 2024 at +0.64. "
                "The weakest season under-predicts consistently (2.6x-5.5x) where the "
                "average swung 0.14x-1.82x, so ranking uses the minimum as a "
                "conservative lower bound. The gate is unchanged -- it already required "
                "every season non-negative, which is that same quantity being positive."
            ),
        },
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}",
                 "ranking": "maximum of the weakest season among survivors"},
        "live_components": list(LIVE),
        "steps": list(STEPS),
        "results": {k: {"weights": v["weights"], "transfer": v["transfer"],
                        "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "heterogeneity": v["heterogeneity"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ntop 8 by weakest season:")
    for label in sorted(results, key=lambda l: -results[l]["min_season_points"])[:8]:
        show(label, results[label])
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
