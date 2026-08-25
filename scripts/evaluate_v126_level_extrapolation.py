"""V126: does the pipeline land the *level* of a season it has never seen?

Score here is dominated by level. A uniform prediction-mean error of d costs
400000 * d^2 points, so d = 0.005 is 10 points and d = 0.016 is 102.

The league level has fallen every season: 0.564670, 0.532712, 0.532762, 0.528920,
0.499957, 0.486105. Linear fits put 2025 between 0.4622 (last three seasons) and
0.4747 (all six). So a model that lands on 2024's level instead of 2025's is out by
roughly 0.016, which is around a hundred points -- a large share of the 172 that
separate 1028 from the leader.

Two mechanisms are supposed to handle this. The global calibration shift is a constant
fitted on pooled 2022-2024 residuals, a window whose own mean rate is 0.5050 while
2025 should be near 0.470; a constant fitted on higher-level seasons has no way to
know about a further decline. The V92 drift term, applied after calibration at weight
0.10, reads the current season's level from reconstructed in-season counters, and is
the only part of the pipeline that can.

The 2024 fold is the right place to test this, because it reproduces the deployed
situation almost exactly. Its calibration is fitted on 2022-2023, mean rate 0.5144,
and applied to 2024 at 0.4861 -- a gap of 0.028 against the deployed gap of 0.035.

So this measures, for each fold, the mean at every stage: raw blend, after calibration,
after the drift term, against the truth. If the fold-2024 bias is near zero the
machinery extrapolates and there is nothing to win. If it is positive, the deployed
model is making the same error one step further out, and the fix is a scalar.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v126_level_extrapolation.json")
NAMES = ("v17", "form", "context", "network", "catboost")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id")
    season_rate = {int(s): float(v) for s, v in
                   y.groupby(raw_frame["season"]).mean().items()}

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
        "v17": {str(y_): 0.95 * v11_prediction(oof[str(y_)]) + 0.05 * logistic[str(y_)]
                for y_ in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }

    raw = {year: sum(w * parts[n][str(year)] for w, n in zip(BASE, NAMES))
           for year in YEARS}
    drift = {}
    for year in YEARS:
        index = oof[str(year)]["row_index"]
        drift[year] = drift_correction(
            add_training_inseason_features(
                raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[index], 1.0)

    stages, biases = {}, {}
    for year in YEARS:
        item = oof[str(year)]
        target = item["target"].astype(float)
        actual = float(target.mean())
        stage_raw = float(np.clip(raw[year], 0, 1).mean())

        if year == 2022:
            calibrated = np.clip(raw[year], 0, 1)
            window, window_rate, shift = [], None, 0.0
        else:
            window = [h for h in YEARS if h < year]
            index, targets, predictions = [], [], []
            for history in window:
                index.append(oof[str(history)]["row_index"])
                targets.append(oof[str(history)]["target"].astype(float))
                predictions.append(raw[history])
            index = np.concatenate(index)
            residual = np.concatenate(targets) - np.concatenate(predictions)
            shift = float(residual.mean())
            window_rate = float(np.concatenate(targets).mean())
            train_frame = raw_frame.loc[index]
            valid_frame = raw_frame.loc[item["row_index"]]
            count = segment_correction(
                train_frame, residual, valid_frame,
                ["balls_before", "strikes_before"], 500)
            pitcher_count = segment_correction(
                train_frame, residual, valid_frame,
                ["pitcher_id", "balls_before", "strikes_before"], 300)
            calibrated = np.clip(
                raw[year] + shift + 0.75 * count + 0.25 * pitcher_count, 0, 1)
        stage_calibrated = float(calibrated.mean())
        final = np.clip(calibrated + DRIFT_WEIGHT * drift[year], 0, 1)
        stage_final = float(final.mean())

        stages[str(year)] = {
            "actual_rate": actual,
            "calibration_window": [int(h) for h in window],
            "calibration_window_rate": window_rate,
            "window_minus_target": (None if window_rate is None
                                    else window_rate - actual),
            "global_shift": shift,
            "mean_raw": stage_raw,
            "mean_after_calibration": stage_calibrated,
            "mean_after_drift": stage_final,
            "drift_term_mean": float(DRIFT_WEIGHT * drift[year].mean()),
            "bias_raw": stage_raw - actual,
            "bias_after_calibration": stage_calibrated - actual,
            "bias_final": stage_final - actual,
            "points_cost_of_final_bias": 400000 * (stage_final - actual) ** 2,
        }
        biases[year] = stage_final - actual
        s = stages[str(year)]
        print(f"{year}: actual {actual:.6f}  window "
              f"{'-' if window_rate is None else f'{window_rate:.6f}'}  "
              f"shift {shift:+.6f}", flush=True)
        print(f"      raw {stage_raw:.6f} ({s['bias_raw']:+.6f})  "
              f"calibrated {stage_calibrated:.6f} ({s['bias_after_calibration']:+.6f})  "
              f"final {stage_final:.6f} ({s['bias_final']:+.6f})  "
              f"costs {s['points_cost_of_final_bias']:.1f} pts", flush=True)

    years = np.array(sorted(season_rate), dtype=float)
    rates = np.array([season_rate[int(v)] for v in years])
    projections = {}
    for label, sel in (("2019-2024", slice(None)), ("2021-2024", slice(2, None)),
                       ("2022-2024", slice(3, None))):
        a, b = np.polyfit(years[sel], rates[sel], 1)
        projections[label] = {"slope": float(a), "rate_2025": float(a * 2025 + b)}

    # The deployed shift is fitted on 2022-2024 and applied to 2025, so the fold whose
    # window-to-target gap is closest to the deployed one is the honest analogue.
    deployed_gap = float(np.mean([season_rate[y] for y in (2022, 2023, 2024)])
                         - projections["2021-2024"]["rate_2025"])
    print(f"\nleague rate by season: " + "  ".join(
        f"{int(v)} {season_rate[int(v)]:.6f}" for v in years))
    for label, value in projections.items():
        print(f"  fit {label}: slope {value['slope']:+.6f}/yr, "
              f"2025 = {value['rate_2025']:.6f}")
    print(f"deployed calibration window mean 2022-2024 = "
          f"{np.mean([season_rate[y] for y in (2022, 2023, 2024)]):.6f}, "
          f"gap to projected 2025 = {deployed_gap:+.6f}")
    print(f"fold 2024 window-to-target gap = "
          f"{stages['2024']['window_minus_target']:+.6f} (the closest analogue)")
    print(f"\nfold final biases: " + "  ".join(
        f"{y} {biases[y]:+.6f}" for y in YEARS))

    OUTPUT.write_text(json.dumps({
        "experiment": "V126_level_extrapolation",
        "question": (
            "A uniform prediction-mean error of d costs 400000*d^2 points, so the level "
            "is the highest-leverage single number in the submission. The calibration "
            "shift is a constant fitted on seasons whose own level is higher than the "
            "target's, and only the V92 drift term can react to a further decline."
        ),
        "weights": {n: w for n, w in zip(NAMES, BASE)},
        "drift_weight": DRIFT_WEIGHT,
        "season_rate": {str(k): v for k, v in season_rate.items()},
        "projections_2025": projections,
        "deployed_window_gap": deployed_gap,
        "stages": stages,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
