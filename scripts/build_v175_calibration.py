"""Build V175: resolve the platoon segment by two-strike state.

V169 put `pitcher_id x batter_hand` -- each pitcher's platoon split -- into the
calibration and returned +7.31. V172 asked how finely that split should be resolved and
V173 extended the grid until the optimum stopped being an edge value. The answer is a
three-way key, `pitcher_id x batter_hand x two_strike`, replacing the two-way one:

    2024 points, coarse platoon term removed
        fine w \\ smoothing    1000    1500    2000
        0.65                  4.50    4.39    3.80
        0.80                  4.37   *4.65*   4.17
        1.00                  3.98    4.78    4.47

`0.80 / 1500` is chosen over the nominally higher `1.00 / 1500` because its eight
neighbours are all at or above +3.80 -- a plateau rather than a peak, which is the reading
V168 recorded as the one that survives.

Keeping a coarse platoon term underneath was tested as a free second axis and is worse at
every setting, so it is dropped entirely; the heavy smoothing already supplies the
shrinkage a fallback would.

    V169   count 0.458333  pitcher x count 0.208333  experience 0.166667  platoon 0.166667
    V175   count 0.305556  pitcher x count 0.138889  experience 0.111111  platoon x 2K 0.444444

Nothing is retrained and the blend is untouched, so the global shift must again be
bit-identical to V161's; the build asserts it.
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
RAW_SEGMENT_WEIGHTS = {"count": 0.55, "pitcher_count": 0.25,
                       "experience": 0.20, "platoon_two_strike": 0.80}
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
PLATOON_TWO_STRIKE_SMOOTHING = 1500.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
CATBOOST_RELIABILITY = 150.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10


def main():
    assert abs(W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST
               + W_FACTORIZATION - 1.0) < 1e-12
    denominator = sum(RAW_SEGMENT_WEIGHTS.values())
    weights = {k: v / denominator for k, v in RAW_SEGMENT_WEIGHTS.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-12, weights

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                           )["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                              )["no_matchup_hte"]["context"]
    catboost_oof = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                               )["catboost"]["projected"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = pd.cut(raw["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
    # Identical expression and identical dtype to the one in `v175_script.py`.
    raw["two_strike"] = (raw["strikes_before"] == 2).astype("int64")

    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        key = str(year)
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[key]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[key] + W_CONTEXT * context_oof[key]
            + W_NETWORK * v160["network"][FEATURE_RELIABILITY][key]
            + W_CATBOOST * catboost_oof[key]
            + W_FACTORIZATION * v160["factorization"][FEATURE_RELIABILITY][key])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]

    fine = make_lookup(frame, centered, ["pitcher_id", "batter_hand", "two_strike"],
                       PLATOON_TWO_STRIKE_SMOOTHING)
    assert fine["two_strike"].dtype == np.int64, fine["two_strike"].dtype
    print(f"platoon x two_strike lookup: {len(fine)} cells, "
          f"range [{fine['correction'].min():+.6f}, {fine['correction'].max():+.6f}], "
          f"sd {fine['correction'].std():.6f}")

    path = Path("artifacts/v175_calibration.joblib")
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": weights["count"],
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], COUNT_SMOOTHING)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"],
             "weight": weights["pitcher_count"],
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"],
                                   PITCHER_COUNT_SMOOTHING)},
            {"columns": ["experience_bin"], "weight": weights["experience"],
             "lookup": make_lookup(frame, centered, ["experience_bin"],
                                   EXPERIENCE_SMOOTHING)},
            {"columns": ["pitcher_id", "batter_hand", "two_strike"],
             "weight": weights["platoon_two_strike"], "lookup": fine},
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
    reference = joblib.load("artifacts/v169_calibration.joblib")
    assert abs(shift - reference["global_shift"]) < 1e-12, (
        "the blend is unchanged, so the shift must be bit-identical to V169/V161")
    print(f"Saved {path}, shift={shift:.12f} -- identical to V169, as it must be")

    Path("artifacts/v175_build_summary.json").write_text(json.dumps({
        "version": "V175",
        "baseline": "V169 (1060.5433916316)",
        "change": ("replace pitcher_id x batter_hand with pitcher_id x batter_hand x "
                   "two_strike at 0.80 of the pre-normalisation mass, smoothing 1500"),
        "segment_weights": weights,
        "smoothing": {"count": COUNT_SMOOTHING, "pitcher_count": PITCHER_COUNT_SMOOTHING,
                      "experience": EXPERIENCE_SMOOTHING,
                      "platoon_two_strike": PLATOON_TWO_STRIKE_SMOOTHING},
        "models_retrained": [],
        "validation": {
            "season_points": {"2022": 0.0, "2023": 3.00, "2024": 4.65},
            "note_2022": "the 2022 fold receives no calibration; two folds are informative",
            "plateau_2024": ("the eight neighbouring grid points are all >= +3.80; "
                             "worst +3.80, mean +4.31"),
            "placebos_at_the_shipped_setting": {
                "batter_id parity": {"2023": -0.39, "2024": -3.44},
                "day-of-week parity": {"2023": -1.56, "2024": 1.03},
                "inning parity": {"2023": -0.85, "2024": -0.58},
            },
            "alternatives": {"strikes_before (3 levels)": 1.55,
                             "count_state (12 levels)": -0.54,
                             "experience": 0.65,
                             "keep coarse platoon underneath": "worse at every setting"},
        },
        "global_shift": shift,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
