"""Build V151: add a `game_type` segment to the calibration, which never had one.

A per-`game_type` audit of the V138 blend found the source of the 2023 anomaly that had
distorted every reading in this project:

    fold  type       n      actual   predicted     bias    skill   points lost
    2022     F  30,448    0.708749    0.688477  -0.0203     -153         20.23
    2023     F  25,686    0.472904    0.649607  +0.1767   -12053       1306.62
    2024     F  30,010    0.459280    0.470490  +0.0112      584          5.95

The F success rate fell from 0.7087 in 2022 to 0.4729 in 2023 -- a break of 0.236 -- and
the 2023-fold model, trained where F ran near 0.70, predicted 0.6496 for it. Those 25,686
rows cost 1,307 points on that fold alone. The "dispersion collapse" of the reliability
audit, the shrinkage optimum of lambda* = 0.35, the 2023 top decile at predicted 0.653
against actual 0.485, and the large 2023 gains that nine rejected experiments all showed:
one cause.

The break itself is history. By 2024 the models have a post-break season and the bias
falls to +0.0112. What is not history is that F remains the worst-predicted group -- 2024
skill of 584 against R's 902, on 12% of the rows -- and that the calibration layer has no
`game_type` term at all. It holds a count segment at 0.75 and a pitcher-count segment at
0.25 and nothing that knows F from R, for a subpopulation that is a different competition
and 68.5% of October.

Adding it at 0.10, taken from the count segment, gains +4.14 on the 2024 fold, halves the
F bias from +0.011210 to +0.005206, and improves **every** monthly block -- 12 of 12, with
the worst month at +0.04. Weights above 0.10 gain slightly more (+4.74 at 0.15, +4.90 for
a game_type/game_type-month pair) but start losing individual months, so 0.10 is the
largest weight at which nothing regresses.

Why the 2023 fold rejects this, and why that does not carry to 2025. That fold fits its
segments on 2022 alone -- entirely pre-break -- so it corrects a 0.47 season using a 0.71
prior and pushes F predictions further up. This was verified directly rather than
assumed: refitting the 2024 fold's segment on 2022 only turns the gain to -2.51 and makes
the F bias *worse* (+0.0206 to +0.0248), reproducing the 2023 behaviour exactly, while
2022+2023 gives +4.14. The deployed model fits on 2022+2023+2024, two thirds post-break,
which is the configuration that works.

So the evidentiary position, stated plainly: 2024 says +4.14 with every month improving,
2022 is exactly zero because it receives no calibration at all, and 2023 is
unfalsifiable-by-construction for a segment that broke inside it. One informative fold.

Only the calibration artifact changes. The inference script iterates over
`calibration["corrections"]`, so `v138_script.py` is reused unmodified and every model
file is byte-identical to V138's.

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


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
# Segment weights sum to one; each estimates the same residual.
W_COUNT, W_PITCHER_COUNT, W_GAME_TYPE = 0.65, 0.25, 0.10
COUNT_SMOOTHING = 500
PITCHER_COUNT_SMOOTHING = 300
GAME_TYPE_SMOOTHING = 2000
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"
FACTORIZATION_SOURCE = "latent8"


def main():
    total = W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST + W_FACTORIZATION
    assert abs(total - 1.0) < 1e-12, total
    assert abs(W_COUNT + W_PITCHER_COUNT + W_GAME_TYPE - 1.0) < 1e-12

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
    factorization_oof = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"][
            FACTORIZATION_SOURCE]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * network_oof[key] + W_CATBOOST * catboost_oof[key]
            + W_FACTORIZATION * factorization_oof[key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    game_type_lookup = make_lookup(frame, centered, ["game_type"],
                                   GAME_TYPE_SMOOTHING)
    print("game_type corrections fitted on 2022-2024:")
    print(game_type_lookup.to_string(index=False))
    shares = frame["game_type"].value_counts(normalize=True)
    print("  fitting-window shares: "
          + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))

    output = Path("artifacts/v151_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": W_COUNT,
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"],
                                   COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": W_PITCHER_COUNT,
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["game_type"], "weight": W_GAME_TYPE,
             "lookup": game_type_lookup},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, output, compress=3)
    print(f"\nSaved {output}, shift={shift:.12f} (V138 was -0.010764370502)")
    print(f"  segments: count {W_COUNT} (s{COUNT_SMOOTHING}) / "
          f"pitcher-count {W_PITCHER_COUNT} (s{PITCHER_COUNT_SMOOTHING}) / "
          f"game_type {W_GAME_TYPE} (s{GAME_TYPE_SMOOTHING})")

    Path("artifacts/v151_build_summary.json").write_text(json.dumps({
        "version": "V151",
        "baseline": "V138 (1050.5510725821)",
        "change": ("a game_type segment at 0.10 added to the calibration, taken from the "
                   "count segment; blend weights and every model file unchanged"),
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "segments": {"count": W_COUNT, "pitcher_count": W_PITCHER_COUNT,
                     "game_type": W_GAME_TYPE},
        "global_shift": shift,
        "discovery": {
            "f_rate_by_season": {"2019": 0.689250, "2020": 0.587774, "2021": 0.703840,
                                 "2022": 0.708749, "2023": 0.472904, "2024": 0.459280},
            "points_lost_on_f": {"2022": 20.23, "2023": 1306.62, "2024": 5.95},
            "note": ("The F break of 0.236 between 2022 and 2023 is the single cause of "
                     "the 2023 anomaly that distorted the dispersion audit, the "
                     "shrinkage optimum and nine rejected experiments."),
        },
        "validation": {
            "fold_2024_gain_points": 4.14,
            "fold_2024_monthly_blocks": "12 of 12, worst month +0.04",
            "f_2024_bias": {"before": 0.011210, "after": 0.005206},
            "fold_2022_gain_points": 0.0,
            "fold_2023_gain_points": -32.36,
            "why_2023_rejects": (
                "That fold fits its segments on 2022 alone, entirely pre-break, so it "
                "corrects a 0.47 season with a 0.71 prior. Verified directly: refitting "
                "the 2024 fold's segment on 2022 only turns +4.14 into -2.51 and makes "
                "the F bias worse (+0.0206 to +0.0248). The deployed model fits on "
                "2022-2024, two thirds post-break."
            ),
            "evidentiary_status": (
                "One informative fold. 2022 is exactly zero because it receives no "
                "calibration, and 2023 is unfalsifiable by construction for a segment "
                "that broke inside it."
            ),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
