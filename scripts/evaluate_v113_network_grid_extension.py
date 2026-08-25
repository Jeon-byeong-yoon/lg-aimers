"""V113: push the network weight grid past its boundary and settle `season`.

V112 promoted `net 0.20 / form 0.48 / context 0.17`, but Form 0.48 was the lowest
value in that grid and the trend was monotone in that direction. An optimum on a
boundary is not one, so Form is extended down to 0.32 and the network up to 0.35.
Nothing is retrained: both network variants are cached from V112.

`season` is decided here on risk rather than on the local score. Passing it through
and clipping it at 2024 are indistinguishable on every validation fold, so the
extrapolation to 2025 (standardised +2.83 against a training maximum of +2.12) is
unmeasurable. Dropping it costs about 29 points of standalone skill on 2024
(645 -> 616) and 2.5 points of blended CI lower bound, but it removes that risk and
scores better on the robustness indicators: 2022 +22.3 against +13.4 and a monthly
block win rate of 95% against 90%. The calibration shift is a constant fitted where
`season` was in range, so an extrapolated shift at inference would not be absorbed
by it — a level error the local folds cannot show. The dropped variant is therefore
preferred, and both are reported.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import blend, bootstrap
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v113_network_grid_extension_metrics.json")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
NETWORK_WEIGHTS = (0.20, 0.25, 0.30, 0.35)
FORM_WEIGHTS = (0.32, 0.40, 0.48)
CONTEXT_WEIGHTS = (0.17, 0.21)
BASELINE = (0.19, 0.64, 0.17, 0.0)
PREFERRED_VARIANT = "without_season"
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data = data.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    networks = joblib.load("artifacts/v112_network_weight_predictions.joblib")["networks"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(data, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASELINE, oof, logistic, form, context, None, data)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for variant in networks:
        for w_network in NETWORK_WEIGHTS:
            for w_form in FORM_WEIGHTS:
                for w_context in CONTEXT_WEIGHTS:
                    w_v17 = round(1.0 - w_form - w_context - w_network, 4)
                    if w_v17 < 0:
                        continue
                    label = (f"{variant}_net{w_network:.2f}_form{w_form:.2f}"
                             f"_ctx{w_context:.2f}")
                    candidate = np.clip(
                        blend((w_v17, w_form, w_context, w_network), oof, logistic,
                              form, context, networks[variant], data)
                        + DRIFT_WEIGHT * term, 0, 1)
                    metrics = development_metrics(reference, candidate)
                    metrics["bootstrap_2024"] = bootstrap(
                        validation_frame, baseline, candidate, mask_2024)
                    metrics["season_bootstrap"] = {
                        str(year): bootstrap(
                            validation_frame, baseline, candidate,
                            (validation_frame["season"] == year).to_numpy())
                        for year in YEARS}
                    metrics["weights"] = {"v17": w_v17, "form": w_form,
                                          "context": w_context, "network": w_network}
                    results[label] = metrics
        print(f"{variant} extended grid evaluated", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    preferred = [label for label in eligible if label.startswith(PREFERRED_VARIANT)]
    promoted = max(preferred or eligible,
                   key=lambda l: results[l]["bootstrap_2024"]["ci95_low"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V113_network_grid_extension",
        "baseline": "V106 (0.19 / 0.64 / 0.17, no network)",
        "baseline_public_score": 973.0643764994,
        "season_decision": {
            "choice": PREFERRED_VARIANT,
            "reason": (
                "raw and clipped-at-2024 are identical on every validation fold, so the "
                "2025 extrapolation is unmeasurable; dropping costs ~2.5 points of CI "
                "lower bound but improves 2022 and the block win rate, and keeps the "
                "network's level consistent with the fixed calibration shift"
            ),
        },
        "network_weights": list(NETWORK_WEIGHTS),
        "form_weights": list(FORM_WEIGHTS),
        "context_weights": list(CONTEXT_WEIGHTS),
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "nothing_retrained": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\ntop 20 by 2024 bootstrap lower bound (vs V106):")
    print(f"{'candidate':>46} {'v17':>5} {'CIlo':>7} {'mean':>7} {'2022':>7} "
          f"{'2023':>7} {'blocks':>7}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"])[:20]:
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>46} {r['weights']['v17']:5.2f} {b['ci95_low']*P:7.1f} "
              f"{b['mean']*P:7.1f} {s['2022']['mean']*P:7.1f} "
              f"{s['2023']['mean']*P:7.1f} {r['monthly_block_win_rate']:7.1%}")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted]["weights"], indent=2))
        b = results[promoted]["bootstrap_2024"]
        print(f"2024 gain {b['mean']*P:+.1f} CI [{b['ci95_low']*P:+.1f}, "
              f"{b['ci95_high']*P:+.1f}]")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
