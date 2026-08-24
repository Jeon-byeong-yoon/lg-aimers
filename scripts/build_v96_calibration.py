"""Build the V96 calibration for the refitted blend (0.27 / 0.52 / 0.21).

V93 kept V41's weights even though it had replaced the Form model. V96b refitted
them and selected an interior plateau point. The calibration lookups depend on
the blend, so they are recomputed here. The V92 drift weight was refitted at the
same time and is stored alongside, so inference reads one consistent bundle.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


OUTPUT = Path("artifacts/v96_calibration.joblib")
W_V17, W_FORM, W_CONTEXT = 0.27, 0.52, 0.21
DRIFT_WEIGHT = 0.15


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    form = form["new_form"]["with_2019_nan"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        predictions.append(
            W_V17 * v17 + W_FORM * form[str(year)] + W_CONTEXT * context[str(year)]
        )
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])

    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    corrections = [
        {
            "columns": ["balls_before", "strikes_before"],
            "weight": 0.75,
            "lookup": make_lookup(frame, centered, ["balls_before", "strikes_before"], 500),
        },
        {
            "columns": ["pitcher_id", "balls_before", "strikes_before"],
            "weight": 0.25,
            "lookup": make_lookup(
                frame, centered, ["pitcher_id", "balls_before", "strikes_before"], 300
            ),
        },
    ]
    joblib.dump(
        {
            "global_shift": shift,
            "corrections": corrections,
            "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT},
            "drift_weight": DRIFT_WEIGHT,
        },
        OUTPUT, compress=3,
    )
    print(f"Saved {OUTPUT}")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} drift={DRIFT_WEIGHT}")
    print(f"  global_shift={shift:.12f}  (V93 was -0.010594525297, V41 -0.011287550929)")


if __name__ == "__main__":
    main()
