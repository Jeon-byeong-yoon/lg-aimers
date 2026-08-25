"""Build V122: V117's components with CatBoost reweighted to 0.27 and V17 retired.

V117 scored 1028.2400. V119 then found the v17, Context and network axes closed
(vertices within 0.015 of their values) and the only live trade to be Form against
CatBoost. V120 showed the outcome on that trade depends on *which* component funds
the increase: V119's proportional rescaling made Form pay and broke 2023, whereas
funding it from v17 first improved 2023 monotonically from +4.8 to +23.8 as v17 went
to zero. v17 is the oldest layer, a V6 tree ensemble blended with a logistic model,
and it is the one most mis-fitted to the later seasons.

The weight is 0.27 on two independent readings that agree: the 2024 paired bootstrap
lower bound peaks there at +0.59, and the sum of the 2022, 2023 and 2024 gains peaks
there at +31.05. Agreement matters more than usual here, because V117 over-delivered
against its 2024 point estimate by 2.59x while CatBoost's gains differ across seasons
by a factor of seven, which makes any single-season magnitude unreliable even though
its sign is still sound.

Only the calibration is rebuilt. The CatBoost model itself is unchanged, so
`v117_catboost.cbm` and its metadata are reused verbatim.

The V17 term stays in the inference script multiplied by an exact zero rather than
being deleted. Its feature pipeline still supplies the target-encoding, hierarchical
and Trackman lookups that every other component consumes, and keeping the code path
identical to the one that scored 1028.2400 means a later nonzero weight needs no
edit. Multiplying by 0.0 is exact, so the arithmetic is unaffected.

Run with the server-mirror interpreter.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction


W_V17, W_FORM, W_CONTEXT, W_NETWORK, W_CATBOOST = 0.00, 0.32, 0.21, 0.20, 0.27
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"


def main():
    assert abs(W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST - 1.0) < 1e-12

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_oof = context_oof["no_matchup_hte"]["context"]
    network_oof = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network_oof = network_oof["networks"]["without_season"]
    catboost_oof = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * network_oof[key] + W_CATBOOST * catboost_oof[key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    output = Path("artifacts/v122_calibration.joblib")
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
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, output, compress=3)
    print(f"Saved {output}, shift={shift:.12f} (V117 was -0.010152603992)")
    print(f"  weights v17={W_V17} form={W_FORM} context={W_CONTEXT} "
          f"network={W_NETWORK} catboost={W_CATBOOST}")

    Path("artifacts/v122_build_summary.json").write_text(json.dumps({
        "version": "V122",
        "baseline": "V117 (1028.2400)",
        "change": "catboost 0.14 -> 0.27, v17 0.05 -> 0.00, form 0.40 -> 0.32",
        "weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                    "network": W_NETWORK, "catboost": W_CATBOOST},
        "global_shift": shift,
        "catboost_model": "artifacts/v117_catboost.cbm (unchanged)",
        "validation": {
            "instrument": "2024 paired pitcher-cluster bootstrap plus season sums",
            "ci95_low_points": 0.59, "mean_points": 5.01,
            "season_2022_points": 5.29, "season_2023_points": 20.75,
            "three_season_sum_points": 31.05,
            "monthly_block_win_rate": 0.80,
            "note": ("Both the 2024 lower bound and the three-season sum peak at 0.27. "
                     "The magnitude is held loosely: V117 over-delivered 2.59x and "
                     "CatBoost's gains differ sevenfold across seasons."),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
