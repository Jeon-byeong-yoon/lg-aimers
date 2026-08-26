"""Build V156: finer experience bins in the calibration segment.

V155 revisited three constants in the deployed path that no experiment had ever varied.
Two were already at their optimum: every alternative drift `reliability_scale` and every
alternative segment smoothing or weight put at least one fold into a loss. The third was
not, and it was the one with the weakest justification -- the experience bin edges
(200 / 1k / 3k / 8k) were chosen a few hours earlier in V154 to make a diagnostic table
readable, not by fitting anything.

Seven bins (50 / 150 / 400 / 1k / 2.5k / 6k) instead of five, at weight 0.20:

    candidate            2022    2023    2024     avg
    V154 (5 bins, 0.10)    --      --      --      --
    fine 2000 / 0.20     +0.00   +0.01   +1.85   +0.62
    fine  500 / 0.15     +0.00   +0.00   +1.58   +0.53
    fine  500 / 0.10     +0.00   +0.15   +0.86   +0.34

No fold loses, which is the condition the leaderboard has actually validated: V154 gained
while missing two magnitude thresholds, whereas V144 and V151 each overrode a fold
carrying a real loss and both fell. The expected size is small -- V154's three-season
average of +2.34 returned +0.30, so +0.62 suggests something near a tenth of a point.

Only the calibration changes. The blend, every model file and the anchors are untouched,
and `v154_script.py` reads the bin edges and labels straight out of the bundle, so the
inference script is reused unmodified.

Run with the server-mirror interpreter.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
import inseason_asof_features_v92 as ins
from build_v12_calibration import make_lookup
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v38_lr_grid import v31_form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_inseason_features, build_anchor, feature_names,
)
from inseason_prior_v153 import population_inside_rates, projected_priors
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


W_V17, W_FORM, W_CONTEXT = 0.00, 0.32, 0.14
W_NETWORK, W_CATBOOST, W_FACTORIZATION = 0.20, 0.27, 0.07
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
TARGET_SEASON = 2025
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k",
          "2.5k-6k", "6k+"]
CATBOOST_CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
CLOSED_VOCABULARY = ("count_state", "base_state", "hand_matchup", "pitcher_hand",
                     "batter_hand", "top_bottom", "game_type")


def experience_bin(frame):
    """Per-row function of an official column; needs nothing from any other row."""
    return pd.cut(frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)


def main():
    total = W_V17 + W_FORM + W_CONTEXT + W_NETWORK + W_CATBOOST + W_FACTORIZATION
    assert abs(total - 1.0) < 1e-12, total
    assert abs(W_COUNT + W_PITCHER_COUNT + W_EXPERIENCE - 1.0) < 1e-12

    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()

    # The anchor, the CatBoost model and its metadata are identical to V154's and are
    # reused as they stand; only the calibration is rebuilt.
    for required in ("artifacts/v154_inseason_anchor.joblib",
                     "artifacts/v154_catboost.cbm",
                     "artifacts/v154_catboost_meta.joblib"):
        assert Path(required).exists(), required
    print("reusing V154's anchor, CatBoost model and metadata unchanged")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context_oof = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context_oof = context_oof["no_matchup_hte"]["context"]
    network_oof = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network_oof = network_oof["networks"]["without_season"]
    catboost_oof = joblib.load(
        "artifacts/v153_projected_prior_predictions.joblib")["catboost"]["projected"]
    factorization_oof = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]["latent8"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw["experience_bin"] = experience_bin(raw)
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

    experience_lookup = make_lookup(frame, centered, ["experience_bin"],
                                    EXPERIENCE_SMOOTHING)
    print("\nexperience corrections fitted on 2022-2024:")
    print(experience_lookup.to_string(index=False))

    calibration = Path("artifacts/v156_calibration.joblib")
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
            {"columns": ["experience_bin"], "weight": W_EXPERIENCE,
             "lookup": experience_lookup},
        ],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
        "experience_edges": [float(v) for v in EDGES],
        "experience_labels": list(LABELS),
    }, calibration, compress=3)
    print(f"\nSaved {calibration}, shift={shift:.12f} (V138 was -0.010764370502)")

    Path("artifacts/v156_build_summary.json").write_text(json.dumps({
        "version": "V156",
        "baseline": "V154 (1050.8512921822)",
        "change": ("CatBoost retrained on in-season features built with a "
                   "season-tracking prior, plus an experience segment at 0.10 in the "
                   "calibration; Form, the network and the factorization network keep "
                   "the career prior they were trained on"),
        "reused_from_v154": ["inseason_anchor", "catboost.cbm", "catboost_meta"],
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "segments": {"count": W_COUNT, "pitcher_count": W_PITCHER_COUNT,
                     "experience": W_EXPERIENCE},
        "global_shift": shift,
        "validation": {
            "season_points": {"2022": 0.00, "2023": 0.01, "2024": 1.85},
            "min_season_points": 0.00, "average_points": 0.62,
            "monthly_block_win_rate": 0.40,
            "note": ("No fold loses, which is the condition the leaderboard validated "
                     "when V154 gained while missing two magnitude thresholds. The size "
                     "is small: V154's average of +2.34 returned +0.30, so +0.62 "
                     "suggests something near a tenth of a point."),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
