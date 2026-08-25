"""V125: is `ordered`'s 2022 penalty a real regression or an uncalibrated level shift?

V123 and V124 both blocked ordered boosting on the same condition. In the blend it is
worth +232 on 2023 -- the largest single-season movement any candidate has produced
here -- and standalone it takes 2023 from -844 to -171. It fails only because 2022
goes to -59.34, and the gate requires every season to be non-negative.

There is a structural reason to doubt that 2022 reading. The blend gives 2022 no
calibration at all: with no earlier season to fit residuals on, the raw prediction is
used, while 2023 and 2024 each receive a global shift plus the count and pitcher-count
lookups. So 2022 is the one fold where a level error is not absorbed.

That matters because level errors are expensive out of proportion to their size. A
prediction-mean error of d costs 400000 * d^2 points, so d = 0.0122 alone accounts for
-59.5, which is almost exactly the observed penalty. And ordered boosting is precisely
the variant expected to have a different level: it estimates gradients on permutations
of the data rather than on the whole set, so its probability scale need not agree with
plain boosting's.

The test separates the two explanations. If the 2022 damage is a level shift, then
removing the mean bias -- which the deployed model gets for free, since the fitted
calibration always includes a global shift -- should recover most of it, and the
remaining per-row disagreement should be small. If the damage survives mean removal,
ordered boosting really is worse on 2022 and the gate was right.

This is a diagnostic, not a candidate: 2022's shift is estimated on 2022's own rows,
which no deployed model could do. It answers whether the gate is reading an artifact.
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
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v125_2022_calibration_diagnostic.json")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
VARIANTS = ("no_te_strong", "onehot", "long", "deep8", "ordered")
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
    pool = dict(joblib.load("artifacts/v116_catboost_predictions.joblib")["predictions"])
    pool.update(joblib.load(
        "artifacts/v123_catboost_capacity_predictions.joblib")["predictions"])
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
    season = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    mask = season == 2022
    rate = target[mask].mean()
    scale = rate * (1 - rate)

    def build(name):
        return np.clip(
            blend(BASE, dict(common, catboost=pool[name]), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)

    baseline = build("no_te_strong")

    def skill_2022(prediction):
        return float(100000 * (1 - ((prediction[mask] - target[mask]) ** 2).mean()
                               / scale))

    base_skill = skill_2022(baseline)
    base_bias = float(baseline[mask].mean() - rate)
    print(f"2022 actual rate {rate:.6f}, baseline mean {baseline[mask].mean():.6f}, "
          f"bias {base_bias:+.6f}, skill {base_skill:.1f}\n", flush=True)

    results = {}
    for name in VARIANTS:
        candidate = build(name)
        bias = float(candidate[mask].mean() - rate)
        raw_gain = (skill_2022(candidate) - base_skill)
        # Both sides get their own mean removed, so what remains is per-row disagreement
        # with the level difference taken out of the comparison entirely.
        centered_base = np.clip(baseline - base_bias, 0, 1)
        centered_candidate = np.clip(candidate - bias, 0, 1)
        centered_gain = (skill_2022(centered_candidate) - skill_2022(centered_base))
        # What a level error of this size costs on its own, at 400000 * d^2.
        level_cost = 400000 * (bias ** 2 - base_bias ** 2)
        results[name] = {
            "mean_prediction": float(candidate[mask].mean()),
            "bias": bias, "raw_gain_2022": raw_gain,
            "gain_after_mean_removal_2022": centered_gain,
            "level_cost_estimate": float(level_cost),
            "bootstrap_2022_raw": bootstrap(validation_frame, baseline, candidate, mask),
            "bootstrap_2022_centered": bootstrap(
                validation_frame, centered_base, centered_candidate, mask),
        }
        r = results[name]
        print(f"  {name:14s} mean {r['mean_prediction']:.6f}  bias {bias:+.6f}  "
              f"raw {raw_gain:+8.2f}  after mean removal {centered_gain:+8.2f}  "
              f"level accounts for {level_cost:+8.2f}", flush=True)

    ordered = results["ordered"]
    explained = (abs(ordered["level_cost_estimate"])
                 / max(abs(ordered["raw_gain_2022"]), 1e-9))
    verdict = ("level artifact" if ordered["gain_after_mean_removal_2022"] > -1.0
               else "genuine regression")
    print(f"\nordered: level explains {explained:.0%} of the raw 2022 movement; "
          f"verdict = {verdict}")

    OUTPUT.write_text(json.dumps({
        "experiment": "V125_2022_calibration_artifact",
        "question": (
            "The blend gives 2022 no calibration, since no earlier season exists to fit "
            "residuals on, while 2023 and 2024 each get a global shift plus count and "
            "pitcher-count lookups. Level errors cost 400000*d^2, so d=0.0122 alone "
            "accounts for -59.5 -- almost exactly ordered boosting's 2022 penalty."
        ),
        "caveat": (
            "Diagnostic only. The 2022 mean is removed using 2022's own rows, which no "
            "deployed model could do. It establishes whether the gate's 2022 condition "
            "is reading a level artifact, not a candidate to submit."
        ),
        "season": 2022,
        "actual_rate": float(rate),
        "baseline": {"name": "no_te_strong", "skill": base_skill, "bias": base_bias},
        "results": results,
        "ordered_level_share_of_raw_movement": float(explained),
        "verdict": verdict,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
