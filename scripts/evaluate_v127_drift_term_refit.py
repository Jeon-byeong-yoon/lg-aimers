"""V127: refit the drift term, which has been stale through two blend changes.

The V92 post-hoc drift term is applied after calibration at weight 0.10 with feature
shrinkage 3. That weight was last fitted in V96, when the blend had three components.
Since then the blend gained the embedding network (V114) and CatBoost (V117, reweighted
in V122), so the term is correcting a different prediction vector than the one it was
tuned against -- the same mis-specification that made weight refits pay four times.

V126 makes the case concrete rather than speculative. On the 2024 fold, whose
calibration window sits 0.028 above its target and so most resembles the deployed
0.035, the term moved the level bias from +0.002361 to +0.001024, cutting the level
cost from 2.2 points to 0.4. It is the only mechanism in the pipeline that can react to
a decline steeper than the calibration window has seen, and the 2023 fold shows the
price of having nothing else: a +0.015 bias costing 90 points when only one season was
available to fit a shift on.

Both of its parameters are laddered. The weight is the direct axis. Feature shrinkage
controls how aggressively a thin in-season sample is pulled toward the career rate;
V105 chose 3 for this term against the Form model's 20, so a lower shrinkage was
already known to suit it, but that too was chosen under the old blend.
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


OUTPUT = Path("artifacts/v127_drift_refit_metrics.json")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
FEATURE_SHRINKAGE = 20.0
INCUMBENT_SHRINKAGE = 3.0
INCUMBENT_WEIGHT = 0.10
SHRINKAGES = (2.0, 3.0, 5.0, 8.0)
WEIGHTS = (0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25)
CATBOOST_SOURCE = "no_te_strong"
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
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
    target = np.concatenate([oof[str(year)]["target"].astype(float) for year in YEARS])
    season = raw_frame["season"].to_numpy()[order]

    terms = {}
    for shrinkage in SHRINKAGES:
        terms[shrinkage] = drift_correction(
            add_training_inseason_features(
                raw_frame, shrinkage=shrinkage).loc[order], 1.0)
        print(f"drift term at shrinkage {shrinkage}: mean "
              f"{terms[shrinkage].mean():+.6f}, sd {terms[shrinkage].std():.6f}",
              flush=True)

    calibrated = blend(BASE, parts, oof, raw_frame)
    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(calibrated + INCUMBENT_WEIGHT * terms[INCUMBENT_SHRINKAGE], 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def level_bias(prediction):
        return {str(year): float(prediction[season == year].mean()
                                 - target[season == year].mean())
                for year in YEARS}

    results = {}
    for shrinkage in SHRINKAGES:
        for weight in WEIGHTS:
            label = f"s{shrinkage:g}_w{weight:.2f}"
            candidate = np.clip(calibrated + weight * terms[shrinkage], 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["level_bias"] = level_bias(candidate)
            metrics["drift_shrinkage"] = shrinkage
            metrics["drift_weight"] = weight
            results[label] = metrics

    def passes(label):
        r = results[label]
        s = r["season_bootstrap"]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and s["2022"]["mean"] > -1e-5 and s["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [l for l in results
                if passes(l) and l != f"s{INCUMBENT_SHRINKAGE:g}_w{INCUMBENT_WEIGHT:.2f}"]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    print(f"\n{'candidate':>12} {'CIlo':>7} {'mean':>7} {'2022':>7} {'2023':>8} "
          f"{'sum':>8} {'bias24':>9} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"]):
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        total = sum(s[str(v)]["mean"] * P for v in YEARS)
        print(f"{label:>12} {b['ci95_low']*P:7.2f} {b['mean']*P:7.2f} "
              f"{s['2022']['mean']*P:7.2f} {s['2023']['mean']*P:8.2f} {total:8.2f} "
              f"{r['level_bias']['2024']:+9.6f} {r['monthly_block_win_rate']:7.0%} "
              f"{'YES' if label in eligible else '-':>5}")

    OUTPUT.write_text(json.dumps({
        "experiment": "V127_drift_term_refit",
        "baseline": (f"V122 with drift weight {INCUMBENT_WEIGHT} at shrinkage "
                     f"{INCUMBENT_SHRINKAGE}"),
        "rationale": (
            "The drift weight was last fitted in V96 against a three-component blend; "
            "the network and CatBoost have since been added, so it corrects a different "
            "prediction vector than it was tuned against. V126 showed the term doing "
            "real work on the deployment-like 2024 fold, moving the level bias from "
            "+0.002361 to +0.001024."
        ),
        "shrinkages": list(SHRINKAGES),
        "weights": list(WEIGHTS),
        "results": {k: {"drift_shrinkage": v["drift_shrinkage"],
                        "drift_weight": v["drift_weight"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "season_bootstrap": v["season_bootstrap"],
                        "level_bias": v["level_bias"],
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
