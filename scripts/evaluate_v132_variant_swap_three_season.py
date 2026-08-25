"""V132: choose the CatBoost variant and its weight under the three-season instrument.

V131 re-scored 120 stored candidates using the equally weighted average of the three
per-season gains, after three leaderboard results showed the 2024-only estimate biased
low by a factor that grows as the estimate shrinks (1.20x, 2.59x, 3.75x). The ranking
changed, and it reversed a V116 conclusion.

V116 concluded that `no_te` beats `full`, and it does on 2024: at weight 0.20,
`no_te_strong` gained +12.01 there against `full_gentle`'s +6.74. But on 2023 the order
flips, +95.01 against +107.04, and under equal weighting `full_gentle` leads at every
weight tested. So the earlier finding was really "no_te wins on 2024", which is a
weaker claim than the one recorded.

Those V116 numbers are measured against V114 and cannot be compared to V122, which
already sits at `no_te_strong` weight 0.27. This re-measures the variant choice and the
slot weight against the V122 baseline directly, with every prediction cached.

Pre-registered gate, fixed before looking at any result:

  * three-season equally weighted average CI low > 0     (sign)
  * three-season average >= +3 points                    (magnitude)
  * each of 2022, 2023, 2024 mean >= 0                   (consistency across seasons)
  * monthly block win rate >= 75%                        (consistency within season)

Only the sign and magnitude tests changed, and they now use all three folds instead of
one. The per-season condition is *stricter* than before, which previously bound only
2022 and 2023 and now binds 2024 too, so a candidate cannot buy the average with a
2024 regression. That matters because ordered boosting scored an average of +52.73
here on the strength of 2023 alone while giving up 59 points on 2022, and it must stay
rejected.
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


OUTPUT = Path("artifacts/v132_variant_swap_metrics.json")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
INCUMBENT = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
SINGLES = ("no_te_strong", "full_gentle", "full_strong", "no_te_gentle", "onehot", "long")
MIXTURES = {
    "gentle_strong": {"full_gentle": 0.5, "no_te_strong": 0.5},
    "full_pair": {"full_gentle": 0.5, "full_strong": 0.5},
    "gentle_pair": {"full_gentle": 0.5, "no_te_gentle": 0.5},
    "gentle_strong_onehot": {"full_gentle": 1 / 3, "no_te_strong": 1 / 3,
                             "onehot": 1 / 3},
    "four_plain": {"full_gentle": 0.25, "full_strong": 0.25, "no_te_gentle": 0.25,
                   "no_te_strong": 0.25},
}
SLOT_LADDER = (0.21, 0.24, 0.27, 0.30, 0.33, 0.36)
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
    pool = dict(joblib.load("artifacts/v116_catboost_predictions.joblib")["predictions"])
    pool.update(joblib.load(
        "artifacts/v123_catboost_capacity_predictions.joblib")["predictions"])

    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend(BASE, dict(common, catboost=pool[INCUMBENT]), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(slot, weights):
        candidate = np.clip(
            blend(weights, dict(common, catboost=slot), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["weights"] = {n: round(w, 4) for n, w in zip(NAMES, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0
                and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= 0 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    def show(label, metrics):
        t = metrics["three_season"]
        print(f"  {label:24s} avg {t['average_points']:+7.2f}  "
              f"CIlo {t['ci95_low_points']:+7.2f}  "
              f"2022 {t['season_points'][0]:+7.2f}  2023 {t['season_points'][1]:+8.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  "
              f"blocks {metrics['monthly_block_win_rate']:5.0%}  "
              f"{'YES' if passes(metrics) else '-'}", flush=True)

    slots = {name: pool[name] for name in SINGLES}
    for label, recipe in MIXTURES.items():
        assert abs(sum(recipe.values()) - 1.0) < 1e-9
        slots[label] = {str(year): sum(share * pool[name][str(year)]
                                      for name, share in recipe.items())
                        for year in YEARS}

    print("slot content at weight 0.27 (baseline = no_te_strong at 0.27):", flush=True)
    stage_one = {}
    for label, slot in slots.items():
        stage_one[label] = evaluate(slot, BASE)
        show(label, stage_one[label])

    best = max(stage_one, key=lambda l: stage_one[l]["three_season"]["ci95_low_points"])
    print(f"\nbest slot content: {best}"
          f"{'' if passes(stage_one[best]) else '  (does not pass the gate)'}",
          flush=True)

    print("\nslot weight ladder for that content:", flush=True)
    ladder = {}
    for weight in SLOT_LADDER:
        form_weight = round(BASE[1] - (weight - BASE[4]), 4)
        if form_weight <= 0:
            continue
        weights = (BASE[0], form_weight, BASE[2], BASE[3], round(weight, 4))
        label = f"{best}_w{weight:.2f}"
        ladder[label] = evaluate(slots[best], weights)
        ladder[label]["slot_weight"] = weight
        show(label, ladder[label])

    combined = {**{f"slot:{k}": v for k, v in stage_one.items()},
                **{f"ladder:{k}": v for k, v in ladder.items()}}
    eligible = [k for k, v in combined.items() if passes(v)]
    promoted = max(eligible,
                   key=lambda k: combined[k]["three_season"]["ci95_low_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V132_variant_swap_three_season",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27) with no_te_strong",
        "baseline_public_score": 1047.03653,
        "instrument": (
            "Equally weighted average of the three per-season gains, SE combined from "
            "each season's interval. Not a row-pooled bootstrap: 2023's gains run three "
            "to seven times the others', which is how V95 was misled."
        ),
        "gate": {
            "three_season_ci95_low_points": "> 0",
            "three_season_average_points": f">= {AVERAGE_FLOOR}",
            "each_season_mean_points": ">= 0",
            "monthly_block_win_rate": f">= {BLOCK_FLOOR}",
            "note": ("Only the sign and magnitude tests changed, to use all three folds "
                     "instead of 2024 alone. The per-season condition is stricter than "
                     "before: it previously bound 2022 and 2023, and now binds 2024 too, "
                     "so the average cannot be bought with a 2024 regression."),
        },
        "singles": list(SINGLES),
        "mixtures": MIXTURES,
        "slot_ladder": list(SLOT_LADDER),
        "stage_one": {k: {"weights": v["weights"], "three_season": v["three_season"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"]}
                      for k, v in stage_one.items()},
        "best_slot_content": best,
        "ladder": {k: {"slot_weight": v["slot_weight"], "weights": v["weights"],
                       "three_season": v["three_season"],
                       "monthly_block_win_rate": v["monthly_block_win_rate"]}
                   for k, v in ladder.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
