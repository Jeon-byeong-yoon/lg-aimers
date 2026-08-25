"""V129: re-test every cached network variant against the V122 blend.

V112 and V115 chose the network variant and settled its hyperparameters when the blend
was v17 0.19 / Form 0.40 / Context 0.21 / network 0.20 and CatBoost did not exist.
Two things have changed since: CatBoost entered at 0.14 and then 0.27, and v17 was
retired to zero. A component's contribution is not a property of the component, it is a
property of the component *given the others* -- V123 made that concrete when `onehot`
beat the incumbent on all three seasons standalone yet lost 2022 in the blend, because
being individually better while more correlated is worth less than being weaker and
less correlated.

So the variants rejected under the old blend deserve a second reading under the new
one. In particular V115's `epochs12_seeds3` was the only tuning candidate that
matched the incumbent standalone (+638 against +616) and it failed on a blend lower
bound of -1.5, which is inside the range that a blend restructuring can move.

`with_season` is included for completeness even though V114 excluded it on a risk
argument that still holds: the folds cannot measure a 2025 extrapolation, and a
constant calibration shift fitted inside the observed range cannot absorb one. It is
here as a measurement, not a candidate.

Every prediction is cached, so this costs arithmetic only.
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


OUTPUT = Path("artifacts/v129_network_retest_metrics.json")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"
NETWORK_WEIGHTS = (0.14, 0.20, 0.26)
EXCLUDED_FROM_PROMOTION = ("with_season",)
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
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    v112 = joblib.load("artifacts/v112_network_weight_predictions.joblib")["networks"]
    v115 = joblib.load("artifacts/v115_network_tuning_predictions.joblib")["networks"]
    networks = {"incumbent": v112["without_season"], "with_season": v112["with_season"]}
    for name, values in v115.items():
        networks[name] = values
    print("network variants: " + ", ".join(networks), flush=True)

    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    target = validation_frame["target"].to_numpy().astype(float)
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASE, dict(common, network=networks["incumbent"]), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def standalone(values):
        out = {}
        for year in YEARS:
            item = oof[str(year)]
            actual = item["target"].astype(float)
            rate = actual.mean()
            out[str(year)] = float(
                100000 * (1 - ((values[str(year)] - actual) ** 2).mean()
                          / (rate * (1 - rate))))
        return out

    results, alone = {}, {}
    for name, values in networks.items():
        alone[name] = standalone(values)
        for weight in NETWORK_WEIGHTS:
            form_weight = round(BASE[1] - (weight - BASE[3]), 4)
            if form_weight <= 0:
                continue
            weights = (BASE[0], form_weight, BASE[2], round(weight, 4), BASE[4])
            label = f"{name}_w{weight:.2f}"
            candidate = np.clip(
                blend(weights, dict(common, network=values), oof, raw_frame)
                + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["weights"] = {n: w for n, w in zip(NAMES, weights)}
            metrics["variant"] = name
            results[label] = metrics

    def passes(label):
        r = results[label]
        s = r["season_bootstrap"]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and s["2022"]["mean"] > -1e-5 and s["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [l for l in results
                if passes(l) and results[l]["variant"] not in EXCLUDED_FROM_PROMOTION
                and l != "incumbent_w0.20"]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    print("\nstandalone skill by variant:")
    for name, values in alone.items():
        print(f"  {name:18s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print(f"\n{'candidate':>24} {'CIlo':>7} {'mean':>7} {'2022':>7} {'2023':>8} "
          f"{'sum':>8} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"]):
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        total = sum(s[str(v)]["mean"] * P for v in YEARS)
        flag = ("YES" if label in eligible
                else ("excl" if r["variant"] in EXCLUDED_FROM_PROMOTION else "-"))
        print(f"{label:>24} {b['ci95_low']*P:7.2f} {b['mean']*P:7.2f} "
              f"{s['2022']['mean']*P:7.2f} {s['2023']['mean']*P:8.2f} {total:8.2f} "
              f"{r['monthly_block_win_rate']:7.0%} {flag:>5}")

    OUTPUT.write_text(json.dumps({
        "experiment": "V129_network_variant_retest",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27) with the V114 network",
        "rationale": (
            "The network variant and its hyperparameters were settled when v17 held 0.19 "
            "and CatBoost did not exist. A component's contribution depends on the others "
            "it sits beside, so variants rejected under the old blend deserve re-reading. "
            "V115's `epochs12_seeds3` matched the incumbent standalone and failed on a "
            "blend lower bound of -1.5, well inside what a restructuring can move."
        ),
        "excluded_from_promotion": list(EXCLUDED_FROM_PROMOTION),
        "exclusion_reason": (
            "`with_season` cannot be validated: the folds give identical predictions with "
            "and without clipping, so the 2025 extrapolation is unmeasurable, and a "
            "constant shift fitted in range cannot absorb it. Reported, not promotable."
        ),
        "network_weights": list(NETWORK_WEIGHTS),
        "standalone_skill": alone,
        "results": {k: {"variant": v["variant"], "weights": v["weights"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "season_bootstrap": v["season_bootstrap"],
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
