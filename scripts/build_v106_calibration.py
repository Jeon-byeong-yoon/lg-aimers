"""Build the V106 calibration: blend 0.19 / 0.64 / 0.17, drift shrinkage 3, w 0.10.

V105 changed the Form model's smoothing but kept V96's blend weights. V106 refits
them. The Form model itself is unchanged, so `v105_feature_models.joblib` and
`v105_inseason_anchor.joblib` are reused and only the calibration is rebuilt —
the inference script reads every weight and constant from this bundle.

The gain here is small: the best 2024 point estimate in V106 was +1.4 points
against a per-candidate noise of about +-5. This is a cheap leaderboard check of
a direction, not a confident promotion.

Run with the server-mirror interpreter for consistency; the bundle holds only
DataFrames and floats, so it carries no numpy-version-specific objects.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


OUTPUT = Path("artifacts/v106_calibration.joblib")
W_V17, W_FORM, W_CONTEXT = 0.19, 0.64, 0.17
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        predictions.append(
            W_V17 * v17 + W_FORM * form[str(year)] + W_CONTEXT * context[str(year)])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])

    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": 0.75,
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], 500)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"], "weight": 0.25,
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"], 300)},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, OUTPUT, compress=3)
    print(f"Saved {OUTPUT}")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT}")
    print(f"  feature shrinkage={FEATURE_SHRINKAGE} drift shrinkage={DRIFT_SHRINKAGE} "
          f"drift weight={DRIFT_WEIGHT}")
    print(f"  global_shift={shift:.12f} (V105 was -0.010264743234)")


if __name__ == "__main__":
    main()
