"""Build V169: add each pitcher's platoon split to the calibration layer.

Nothing is retrained. The blend is V161 exactly, so the global shift is unchanged; the
only new object is a fourth segment lookup on `pitcher_id x batter_hand`, and the three
existing weights are renormalised so the total correction mass stays at one.

    V161   count 0.55        pitcher x count 0.25        experience 0.20
    V169   count 0.458333    pitcher x count 0.208333    experience 0.166667
           platoon 0.166667  (= 0.20 / 1.20, smoothing 1000)

The inference script iterates `calibration["corrections"]` and merges each lookup with
`fillna(0)`, so a 2025 debutant with no history simply gets no platoon correction and the
script needs no edit -- the same reuse V151 made of `v138_script.py`.
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
PLATOON_SHARE = 0.20
PLATOON_SMOOTHING = 1000.0
RAW_SEGMENT_WEIGHTS = {"count": 0.55, "pitcher_count": 0.25,
                       "experience": 0.20, "platoon": PLATOON_SHARE}
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
CATBOOST_RELIABILITY = 150.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10


def main():
    total = W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST + W_FACTORIZATION
    assert abs(total - 1.0) < 1e-12, total
    denominator = 1.0 + PLATOON_SHARE
    weights = {name: value / denominator for name, value in RAW_SEGMENT_WEIGHTS.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-12, weights

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                           )["forms"][FEATURE_SHRINKAGE]
    network_oof = v160["network"][FEATURE_RELIABILITY]
    factorization_oof = v160["factorization"][FEATURE_RELIABILITY]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                              )["no_matchup_hte"]["context"]
    catboost_oof = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                               )["catboost"]["projected"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = pd.cut(raw["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
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

    platoon = make_lookup(frame, centered, ["pitcher_id", "batter_hand"],
                          PLATOON_SMOOTHING)
    print(f"platoon lookup: {len(platoon)} cells, "
          f"correction range [{platoon['correction'].min():+.6f}, "
          f"{platoon['correction'].max():+.6f}], "
          f"sd {platoon['correction'].std():.6f}")

    path = Path("artifacts/v169_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": weights["count"],
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"],
                                   COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": weights["pitcher_count"],
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["experience_bin"], "weight": weights["experience"],
             "lookup": make_lookup(frame, centered, ["experience_bin"],
                                   EXPERIENCE_SMOOTHING)},
            {"columns": ["pitcher_id", "batter_hand"], "weight": weights["platoon"],
             "lookup": platoon},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "feature_reliability_scale": FEATURE_RELIABILITY,
        "catboost_reliability_scale": CATBOOST_RELIABILITY,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
        "experience_edges": [float(v) for v in EDGES],
        "experience_labels": list(LABELS),
    }, path, compress=3)
    print(f"Saved {path}, shift={shift:.12f} (V161 was -0.010674754699)")

    reference = joblib.load("artifacts/v161_calibration.joblib")
    assert abs(shift - reference["global_shift"]) < 1e-12, (
        "the blend is unchanged, so the shift must be bit-identical to V161")
    print("shift matches V161 exactly, as it must -- the blend did not move")

    Path("artifacts/v169_build_summary.json").write_text(json.dumps({
        "version": "V169",
        "baseline": "V161 (1053.2326413884)",
        "change": ("add a pitcher_id x batter_hand segment -- each pitcher's own platoon "
                   "split -- at 0.20 of the pre-normalisation correction mass with "
                   "smoothing 1000, renormalising the three existing weights"),
        "segment_weights": weights,
        "platoon_smoothing": PLATOON_SMOOTHING,
        "models_retrained": [],
        "validation": {
            "season_points": {"2022": 0.0, "2023": 4.55, "2024": 2.13},
            "note_2022": ("the 2022 fold receives no calibration in the harness, so it "
                          "reads exactly 0.00 by construction; two folds are informative"),
            "controls": {
                "placebo_batter_parity": {"2023": -0.03, "2024": -0.49},
                "placebo_day_parity": {"2023": -0.07, "2024": -0.41},
                "pitcher_only": {"2023": -0.15, "2024": -0.67},
                "handedness_only": {"2023": -0.15, "2024": 1.02},
            },
            "plateau": ("2024 stays between +1.55 and +2.13 across w 0.10-0.30 and "
                        "smoothing 500-2000, so the chosen point is not a spike"),
        },
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
