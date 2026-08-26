"""V136: reduce Context as far as the consistency conditions allow, and no further.

V135 ran twelve ordered pairwise transfers at three step sizes. Eighteen of them touch
Context and all eighteen agree: every transfer *out* of Context is strongly positive on
the three-season average (up to +32.61) and every transfer *in* is strongly negative
(down to -49.09). Unanimity across eighteen independent candidates is not noise.

The diagnosis is that Context is the next v17. Laid side by side:

    component      2022    2023    2024   mean corr. with the rest (2024)
    v17 (retired)  2314   -1386     729   0.867
    form           2392    -754     789   0.875
    context        2224   -1333     725   0.854
    network        2162    -387     616   0.778
    catboost       2397    -844     752   0.864

Context is the second-worst component on 2023, effectively tied with the v17 layer that
V120 retired for precisely that reason, and it is highly correlated with everything
else. It is not distinctively good on 2024 either -- Form beats it there.

So why does cutting it hurt 2024? Because on 2024 every correlation drops (0.778-0.875
against 0.884-0.947 on 2022), so components are more independent there and each one
contributes more diversity. Context still earns something on that fold even while being
redundant on the earlier two.

That is a real conflict between folds, not a magnitude question, and the leaderboard has
nothing to say about it: all three of its lessons concerned candidates whose 2024 sign
was positive and whose actual gain was larger. So the monthly block win rate does the
deciding, as the one condition never impugned, and V135's whole Context family sat at
45-65% against a floor of 75%.

This walks the established direction in one-point steps and stops where consistency
stops. Four recipients are tried for the freed weight, because V120 showed that who
receives -- or pays -- decides the outcome on an axis, not just how far it moves. If
nothing clears the floor at any step, the axis is blocked by consistency rather than by
direction, and that is the finding.

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
from evaluate_v119_five_way_weight_refit import NAMES, blend
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v136_context_reduction_metrics.json")
BASE = dict(zip(NAMES, (0.00, 0.32, 0.21, 0.20, 0.27)))
REDUCTIONS = (0.01, 0.02, 0.03, 0.04)
RECIPIENTS = {
    "catboost": {"catboost": 1.0},
    "network": {"network": 1.0},
    "net_cat": {"network": 0.5, "catboost": 0.5},
    "three_way": {"form": 1 / 3, "network": 1 / 3, "catboost": 1 / 3},
}
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
        metrics["weights"] = {n: round(w, 6) for n, w in zip(NAMES, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"{'candidate':>22} {'avg':>7} {'CIlo':>7} {'2022':>7} {'2023':>8} "
          f"{'2024':>7} {'blocks':>7} {'pass':>5}")
    for name, recipe in RECIPIENTS.items():
        for amount in REDUCTIONS:
            weights = dict(BASE)
            weights["context"] = round(weights["context"] - amount, 6)
            for target, share in recipe.items():
                weights[target] = round(weights[target] + amount * share, 6)
            # Renormalise rather than assert: a 1/3 share rounded to six places
            # leaves the sum a microscopic step off one, which is arithmetic noise
            # and not a specification error.
            vector = np.array([weights[n] for n in NAMES], dtype=float)
            assert abs(vector.sum() - 1.0) < 1e-5, vector
            vector = tuple(vector / vector.sum())
            label = f"{name}_{amount:.2f}"
            results[label] = evaluate(vector)
            results[label]["recipient"] = recipe
            results[label]["reduction"] = amount
            m = results[label]
            t = m["three_season"]
            print(f"{label:>22} {t['average_points']:7.2f} {t['ci95_low_points']:7.2f} "
                  f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
                  f"{t['season_points'][2]:7.2f} "
                  f"{m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible,
                   key=lambda l: results[l]["three_season"]["ci95_low_points"],
                   default=None)

    blocked_by = {}
    for label, m in results.items():
        t = m["three_season"]
        reasons = []
        if t["ci95_low_points"] <= 0:
            reasons.append("sign")
        if t["average_points"] < AVERAGE_FLOOR:
            reasons.append("magnitude")
        if any(v < -1e-9 for v in t["season_points"]):
            reasons.append("season")
        if m["monthly_block_win_rate"] < BLOCK_FLOOR:
            reasons.append("blocks")
        blocked_by[label] = reasons

    OUTPUT.write_text(json.dumps({
        "experiment": "V136_context_fine_reduction",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27), Public 1047.03653",
        "diagnosis": {
            "standalone_skill": {
                "v17_retired": {"2022": 2314, "2023": -1386, "2024": 729},
                "form": {"2022": 2392, "2023": -754, "2024": 789},
                "context": {"2022": 2224, "2023": -1333, "2024": 725},
                "network": {"2022": 2162, "2023": -387, "2024": 616},
                "catboost": {"2022": 2397, "2023": -844, "2024": 752},
            },
            "mean_correlation_2024": {"v17": 0.867, "form": 0.875, "context": 0.854,
                                      "network": 0.778, "catboost": 0.864},
            "reading": (
                "Context is the next v17: second-worst on 2023, effectively tied with "
                "the layer V120 retired for that reason, highly correlated, and not "
                "distinctively good on 2024 either. Cutting it hurts 2024 because every "
                "correlation drops on that fold (0.778-0.875 against 0.884-0.947 on "
                "2022), so components are more independent there and Context still earns "
                "diversity even while redundant earlier."
            ),
        },
        "why_blocks_decide": (
            "The fold conflict here is one of sign, not magnitude. All three leaderboard "
            "lessons concerned candidates whose 2024 sign was positive and whose actual "
            "gain was larger, so they say nothing about a negative 2024. The monthly "
            "block win rate is the one condition never impugned, and it decides."
        ),
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}"},
        "reductions": list(REDUCTIONS),
        "recipients": RECIPIENTS,
        "results": {k: {"weights": v["weights"], "reduction": v["reduction"],
                        "recipient": v["recipient"],
                        "three_season": v["three_season"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "blocked_by": blocked_by,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for reasons in blocked_by.values():
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    print(f"\nblocking condition counts across {len(results)} candidates: {counts}")
    print(f"eligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
