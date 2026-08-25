"""V120: who pays for CatBoost's weight, at a one-point resolution.

V119 refitted all five weights and found nothing eligible, but its ladders were
informative. The v17, Context and network axes all had vertices within 0.015 of their
current values, so those three are closed. The only live trade is Form against
CatBoost, and moving along it improved 2024 monotonically while breaking 2023 and
dropping the monthly block win rate below the gate.

The reason is which component funds the increase. V119 rescaled the other four
proportionally, so Form -- the strongest single component -- paid most of the bill.
But the v17 ladder shows its weight barely matters any more: every one of its four
rungs, including setting it to zero outright, moved 2024 by less than half a point.
V116 saw the same thing from the other side, where `cb0.20` (which exhausts v17 to
zero) had a *better* 2023 than `cb0.14`, +95.0 against +71.5.

So this is one ladder with a known payer switch. CatBoost rises one point at a time;
v17 funds it until v17 is gone, and Form funds it after that. If the v17 reading is
right, the gate should hold up to `catboost=0.19` and start failing past it. That
also settles V116's weight choice at a resolution its three-point grid could not
reach.
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


OUTPUT = Path("artifacts/v120b_catboost_weight_metrics.json")
BASE = (0.05, 0.40, 0.21, 0.20, 0.14)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"
CATBOOST_LADDER = (0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.23, 0.25,
                   0.27, 0.29, 0.31, 0.33, 0.36, 0.39, 0.42)
P = 100000.0 / 0.25


def weights_for(catboost):
    """v17 funds the increase first, then Form. Context and network never move."""
    increase = catboost - BASE[4]
    from_v17 = min(increase, BASE[0])
    return (round(BASE[0] - from_v17, 4), round(BASE[1] - (increase - from_v17), 4),
            BASE[2], BASE[3], round(catboost, 4))


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
        "v17": {str(y): 0.95 * v11_prediction(oof[str(y)]) + 0.05 * logistic[str(y)]
                for y in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(blend(BASE, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for value in CATBOOST_LADDER:
        weights = weights_for(value)
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
        metrics["weights"] = {n: w for n, w in zip(NAMES, weights)}
        metrics["payer"] = "v17" if value <= BASE[0] + BASE[4] else "form"
        label = f"cb{value:.2f}"
        results[label] = metrics
        b = metrics["bootstrap_2024"]; s = metrics["season_bootstrap"]
        print(f"  {label}  v17={weights[0]:.2f} form={weights[1]:.2f}  "
              f"payer={metrics['payer']:4s}  CIlo {b['ci95_low']*P:+7.2f}  "
              f"mean {b['mean']*P:+7.2f}  2022 {s['2022']['mean']*P:+7.2f}  "
              f"2023 {s['2023']['mean']*P:+7.2f}  blocks "
              f"{metrics['monthly_block_win_rate']:5.0%}", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [label for label in results if passes(label) and label != "cb0.14"]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V120b_catboost_weight_payer_extended",
        "baseline": "V117 (0.05 / 0.40 / 0.21 / 0.20 / 0.14)",
        "rationale": (
            "V119 closed the v17, Context and network axes and showed the only live "
            "trade is Form against CatBoost. Its proportional rescaling made Form pay "
            "for CatBoost's increase, which broke 2023. This ladder makes the payer "
            "explicit: v17 first, whose own ladder showed its weight no longer matters, "
            "and Form only after v17 is exhausted."
        ),
        "ladder": list(CATBOOST_LADDER),
        "payer_switch_at": round(BASE[0] + BASE[4], 4),
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
