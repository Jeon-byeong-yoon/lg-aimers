"""Build the V108 calibration: V106 blend, drift term broadened to `combined`.

V100 rejected every broadened drift term, but that was at shrinkage 50 with a
weight of 0.15. At V106's shrinkage 3 and weight 0.10 the picture changed, and
`0.5*success + 0.5*failure_modes` at scale 0.20 passed: 2024 mean +1.0 points,
2022 +1.3, 2023 +24.2, monthly block win rate 80%.

That composition keeps the success coefficient at 0.10 — identical to V106 — and
adds the three official failure-mode deltas at 0.10, so it layers a new signal on
top rather than rescaling the existing one.

The blend and the Form model are unchanged from V106, so the calibration lookups
are identical; only the drift metadata differs.

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


OUTPUT = Path("artifacts/v108_calibration.joblib")
W_V17, W_FORM, W_CONTEXT = 0.19, 0.64, 0.17
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.20
DRIFT_COMPOSITION = "combined"


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
        "drift_composition": DRIFT_COMPOSITION,
    }, OUTPUT, compress=3)
    print(f"Saved {OUTPUT}")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT}")
    print(f"  feature shrinkage={FEATURE_SHRINKAGE} drift shrinkage={DRIFT_SHRINKAGE}")
    print(f"  drift weight={DRIFT_WEIGHT} composition={DRIFT_COMPOSITION}")
    print(f"  global_shift={shift:.12f} (V106 was -0.010057027756)")


if __name__ == "__main__":
    main()
