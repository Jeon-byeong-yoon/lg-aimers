"""V124: mix the CatBoost variants inside one slot instead of picking a winner.

V123 tested four variants as replacements for the incumbent `no_te_strong` and none
passed the gate. Reading them as competitors was the wrong frame. Their season
profiles are close to opposite:

    variant   2022 standalone   2023 standalone   2024 standalone
    strong             2397             -844              752
    onehot             2403             -765              797
    long               2401             -856              771
    ordered            2047             -171              579

`onehot` beats the incumbent on all three seasons standalone yet *loses* 2022 in the
blend (-2.07), which is the signature of reduced diversity: a component that is
individually better but more correlated with Form, Context and the network adds less
than a weaker, less correlated one. `ordered` is the mirror image, dreadful on 2022
standalone but worth +232 on 2023 in the blend, because ordered boosting estimates
gradients on permutations rather than on the whole set and so makes different errors
in kind, not just in size.

A slot filled with an average of several such models is not the same object as any of
them. This costs nothing to test: every prediction is already cached from V116 and
V123, so only the arithmetic is new.

The slot weight is then re-laddered for whichever mixture wins, because a slot whose
contents changed has no reason to keep the 0.27 that V120 fitted for `strong` alone.
"""

import json
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


OUTPUT = Path("artifacts/v124_variant_mixture_metrics.json")
PREDICTIONS = Path("artifacts/v124_variant_mixture_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
SLOT_LADDER = (0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40)
# `ordered` is held to a minority share everywhere it appears. Its 2022 standalone is
# 350 points below the others, so it is included for the kind of error it makes, not
# for its level.
MIXTURES = {
    "strong_onehot": {"no_te_strong": 0.5, "onehot": 0.5},
    "strong_onehot_long": {"no_te_strong": 1 / 3, "onehot": 1 / 3, "long": 1 / 3},
    "strong_onehot_ordered10": {"no_te_strong": 0.45, "onehot": 0.45, "ordered": 0.10},
    "strong_onehot_ordered20": {"no_te_strong": 0.40, "onehot": 0.40, "ordered": 0.20},
    "four_equal": {"no_te_strong": 0.25, "onehot": 0.25, "long": 0.25, "ordered": 0.25},
    "four_ordered15": {"no_te_strong": 0.30, "onehot": 0.30, "long": 0.25,
                       "ordered": 0.15},
    "all_six": {"no_te_strong": 0.22, "onehot": 0.22, "long": 0.20, "deep8": 0.14,
                "no_te_gentle": 0.12, "ordered": 0.10},
    "gentle_strong_onehot": {"no_te_gentle": 1 / 3, "no_te_strong": 1 / 3,
                             "onehot": 1 / 3},
}
P = 100000.0 / 0.25


def mix(pool, recipe):
    assert abs(sum(recipe.values()) - 1.0) < 1e-9, recipe
    return {str(year): sum(share * pool[name][str(year)]
                           for name, share in recipe.items())
            for year in YEARS}


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
    print("pool: " + ", ".join(sorted(pool)), flush=True)

    common = {
        "v17": {str(y): 0.95 * v11_prediction(oof[str(y)]) + 0.05 * logistic[str(y)]
                for y in YEARS},
        "form": form, "context": context, "network": network,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASE, dict(common, catboost=pool["no_te_strong"]), oof, raw_frame)
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
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["weights"] = {n: round(w, 4) for n, w in zip(NAMES, weights)}
        return metrics

    def passes(metrics):
        s = metrics["season_bootstrap"]
        return (metrics["bootstrap_2024"]["ci95_low"] > 0
                and s["2022"]["mean"] > -1e-5 and s["2023"]["mean"] > -1e-5
                and metrics["monthly_block_win_rate"] >= 0.75
                and metrics["bootstrap_2024"]["mean"] * P >= 3.0)

    def show(label, metrics):
        b = metrics["bootstrap_2024"]; s = metrics["season_bootstrap"]
        total = sum(s[str(y)]["mean"] * P for y in YEARS)
        print(f"  {label:24s} CIlo {b['ci95_low']*P:+7.2f}  mean {b['mean']*P:+7.2f}  "
              f"2022 {s['2022']['mean']*P:+7.2f}  2023 {s['2023']['mean']*P:+8.2f}  "
              f"sum {total:+8.2f}  blocks {metrics['monthly_block_win_rate']:5.0%}  "
              f"{'YES' if passes(metrics) else '-'}", flush=True)
        return total

    slots, mixtures, totals = {}, {}, {}
    for label, recipe in MIXTURES.items():
        slots[label] = mix(pool, recipe)
        mixtures[label] = evaluate(slots[label], BASE)
        mixtures[label]["recipe"] = recipe
        totals[label] = show(label, mixtures[label])

    # Ranked by the three-season sum, not the 2024 lower bound: V121 showed the 2024
    # point estimate is a sound sign test but an unreliable magnitude when a candidate's
    # gains differ across seasons, and mixing `ordered` in guarantees exactly that.
    eligible = [label for label in mixtures if passes(mixtures[label])]
    best = max(eligible or list(totals), key=lambda l: totals[l])
    print(f"\nbest mixture by three-season sum: {best} "
          f"({'eligible' if best in eligible else 'NOT eligible'})", flush=True)

    ladder = {}
    for slot_weight in SLOT_LADDER:
        form_weight = round(BASE[1] - (slot_weight - BASE[4]), 4)
        if form_weight <= 0:
            continue
        weights = (BASE[0], form_weight, BASE[2], BASE[3], round(slot_weight, 4))
        label = f"{best}_w{slot_weight:.2f}"
        ladder[label] = evaluate(slots[best], weights)
        ladder[label]["slot_weight"] = slot_weight
        show(label, ladder[label])

    ladder_eligible = [l for l in ladder if passes(ladder[l])]
    promoted = max(ladder_eligible,
                   key=lambda l: sum(ladder[l]["season_bootstrap"][str(y)]["mean"]
                                     for y in YEARS),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V124_catboost_variant_mixture",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27) with no_te_strong alone",
        "rationale": (
            "V123's variants have near-opposite season profiles. `onehot` beats the "
            "incumbent on all three seasons standalone but loses 2022 in the blend, "
            "which is reduced diversity; `ordered` is far worse standalone on 2022 yet "
            "worth +232 on 2023, because permutation-based gradient estimation makes "
            "errors of a different kind. A slot holding their average is not the same "
            "object as any of them, and every prediction is already cached."
        ),
        "selection_rule": (
            "Ranked by the three-season sum. V121 established the 2024 point estimate "
            "as a sound sign test but an unreliable magnitude for candidates whose "
            "gains are heterogeneous across seasons, which mixing `ordered` guarantees."
        ),
        "mixtures": {k: {"recipe": v["recipe"],
                         "bootstrap_2024": v["bootstrap_2024"],
                         "season_bootstrap": v["season_bootstrap"],
                         "monthly_block_win_rate": v["monthly_block_win_rate"],
                         "three_season_sum_points": totals[k]}
                     for k, v in mixtures.items()},
        "mixture_eligible": sorted(eligible),
        "best_mixture": best,
        "slot_ladder": {k: {"slot_weight": v["slot_weight"], "weights": v["weights"],
                            "bootstrap_2024": v["bootstrap_2024"],
                            "season_bootstrap": v["season_bootstrap"],
                            "monthly_block_win_rate": v["monthly_block_win_rate"]}
                        for k, v in ladder.items()},
        "ladder_eligible": sorted(ladder_eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"slots": slots, "recipes": MIXTURES}, PREDICTIONS, compress=3)
    print(f"\nmixture eligible={len(eligible)}  ladder eligible={len(ladder_eligible)}  "
          f"promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
