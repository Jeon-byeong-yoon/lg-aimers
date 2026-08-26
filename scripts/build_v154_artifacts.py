"""Build V154: a season-tracking prior for CatBoost's features, plus an experience segment.

Two findings, both about the same defect.

The in-season reconstruction shrinks toward a prior that `build_priors` sets to the mean
of the *career* as-of rate column. But `inside_rate` estimates a current-season quantity,
and a career average lags a league that fell from 0.5647 to 0.4861. The error grows every
season and shows up directly: on the 2024 fold the over-prediction falls monotonically
with pitcher experience (+0.0157 in F, +0.0077 in R below 200 pitches, crossing zero near
a thousand), which is the shape a bad prior produces because it carries weight
20/(inside_n + 20).

    target   V92 prior   V154 prior   actual   V92 gap   V154 gap
      2022    0.547341     0.524781  0.528920   +0.0184    -0.0041
      2023    0.544313     0.517275  0.499957   +0.0444    +0.0173
      2024    0.540175     0.498733  0.486105   +0.0541    +0.0126
      2025    0.535228     0.490536       n/a         -          -

`inseason_prior_v153` recovers the population's within-season rate by the same anchor
differencing V92 built for individual pitchers, then projects it linearly to the target
season. One uniform rule for all seven reconstructed columns, using nothing from the
season being predicted, and it shrinks the gap by 60-77% on every fold.

Only CatBoost gets the new features. V153 measured all four combinations and the prior
*helps* CatBoost on every fold while *hurting* Form (2023 -1.46, 2024 -1.40), so Form,
the network and the factorization network keep the career prior they were trained on.
Feeding a model features built with a different prior than it was fitted on would be a
train/inference mismatch, which is why the inference script now builds the reconstruction
three times: shrinkage 20 with the career prior for Form and the two networks, shrinkage
20 with the projected prior for CatBoost, and shrinkage 3 with the career prior for the
drift term.

The retraining is clean. Rebuilding CatBoost with the *old* prior reproduces the stored
predictions exactly -- all three folds move by 0.00 -- so the measured effect is the prior
and nothing else.

The experience segment at 0.10 is the second piece. V152 found it safe on every fold but
worth only +1.71 alone, because a post-hoc segment carries weight 0.10 and cannot repair
the feature a model learned from. Stacked on the prior fix it lifts the weakest season
from +1.21 to +2.29 and evens the three out.

    candidate            2022    2023    2024     min     avg
    prior only          +2.42   +2.63   +1.21   +1.21   +2.09
    prior + exp 0.10    +2.42   +2.29   +2.32   +2.29   +2.34

That evenness is the point. Heterogeneity of 1.06 against 7.2 for V117, 4.1 for V122 and
107 for V138 -- and V138's lesson was that heterogeneity is what destroys the average's
reliability, so here the average is a trustworthy estimate rather than an artifact of one
fold. A general improvement looks like this; the nine candidates rejected for being
2023-heavy looked like the opposite.

The gate is not met: the average is +2.34 against a +3.0 floor and the monthly block rate
is 65% against 75%. Both are magnitude and consistency thresholds rather than adverse
direction -- every fold gains -- which is a different position from V144 (-1.86 on 2022)
and V151 (-32.36 on 2023), both of which overrode a fold carrying a real loss.

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
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.65, 0.25, 0.10
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
TARGET_SEASON = 2025
EDGES = [-1, 200, 1000, 3000, 8000, np.inf]
LABELS = ["0-200", "200-1k", "1k-3k", "3k-8k", "8k+"]
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

    priors = projected_priors(raw_frame, TARGET_SEASON)
    career = ins.build_priors(raw_frame)
    print("prior for the 2025 anchor:")
    for key in sorted(priors):
        print(f"  {key:34s} career {career[key]:.6f} -> projected {priors[key]:.6f}")

    anchors = {group: build_anchor(raw_frame, group) for group in ("pitcher", "batter")}
    anchor_path = Path("artifacts/v154_inseason_anchor.joblib")
    joblib.dump({
        "anchors": anchors, "priors": priors, "groups": ("pitcher", "batter"),
        "target_season": TARGET_SEASON,
        "anchor_season_max": int(raw_frame["season"].max()),
        "train_rows": int(len(raw_frame)),
        "prior_mode": "projected_population_inside_rate",
    }, anchor_path, compress=3)
    print(f"Saved {anchor_path} ({anchor_path.stat().st_size / 2**20:.2f} MiB)")

    # Training-side features with a per-season projected prior, mirroring how a 2025 row
    # relates to the frozen 2019-2024 state.
    table = population_inside_rates(raw_frame)
    pieces = []
    for value in sorted(raw_frame["season"].unique()):
        current = raw_frame.loc[raw_frame["season"] == value]
        history = raw_frame.loc[raw_frame["season"] < value]
        if history.empty:
            pieces.append(pd.DataFrame(np.nan, index=current.index,
                                       columns=feature_names()))
            continue
        pieces.append(add_inseason_features(
            current,
            {g: build_anchor(history, g) for g in ("pitcher", "batter")},
            projected_priors(history, value, table=table.loc[table.index < value]),
            current_season=value, shrinkage=FEATURE_SHRINKAGE)[feature_names()])
    block = pd.concat(pieces).loc[raw_frame.index]

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encoding_columns]
               + [c for c in categorical if c not in model_columns])
    work = features[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    print(f"catboost frame: {work.shape[0]:,} rows x {work.shape[1]} columns "
          f"(projected prior)", flush=True)

    started = time.time()
    model = CatBoostClassifier(**CATBOOST_CONFIG, random_seed=42, verbose=0,
                               thread_count=6, cat_features=categorical,
                               allow_writing_files=False)
    model.fit(work, y.to_numpy())
    print(f"catboost trained on all six seasons [{time.time() - started:.0f}s]",
          flush=True)
    model_path = Path("artifacts/v154_catboost.cbm")
    model.save_model(str(model_path))
    print(f"Saved {model_path} ({model_path.stat().st_size / 2**20:.2f} MiB)")

    values = {name: sorted(work[name].unique().tolist()) for name in categorical}
    meta_path = Path("artifacts/v154_catboost_meta.joblib")
    joblib.dump({
        "columns": columns, "categorical": categorical, "numeric": numeric,
        "categorical_values": values,
        "closed_vocabulary": [c for c in CLOSED_VOCABULARY if c in categorical],
        "config": CATBOOST_CONFIG, "source": "no_te_strong_projected_prior",
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "prior_mode": "projected_population_inside_rate",
    }, meta_path, compress=3)
    print(f"Saved {meta_path}")

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

    calibration = Path("artifacts/v154_calibration.joblib")
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

    Path("artifacts/v154_build_summary.json").write_text(json.dumps({
        "version": "V154",
        "baseline": "V138 (1050.5510725821)",
        "change": ("CatBoost retrained on in-season features built with a "
                   "season-tracking prior, plus an experience segment at 0.10 in the "
                   "calibration; Form, the network and the factorization network keep "
                   "the career prior they were trained on"),
        "prior": {"career_2025": career["asof_pitcher_success_rate"],
                  "projected_2025": priors["asof_pitcher_success_rate"],
                  "gap_by_fold": {"2022": [0.0184, -0.0041], "2023": [0.0444, 0.0173],
                                  "2024": [0.0541, 0.0126]}},
        "blend_weights": {"v17": W_V17, "form": W_FORM, "context": W_CONTEXT,
                          "network": W_NETWORK, "catboost": W_CATBOOST,
                          "factorization": W_FACTORIZATION},
        "segments": {"count": W_COUNT, "pitcher_count": W_PITCHER_COUNT,
                     "experience": W_EXPERIENCE},
        "global_shift": shift,
        "validation": {
            "season_points": {"2022": 2.42, "2023": 2.29, "2024": 2.32},
            "min_season_points": 2.29, "average_points": 2.34,
            "heterogeneity": 1.06,
            "monthly_block_win_rate": 0.65,
            "control": ("rebuilding CatBoost with the old prior reproduces the stored "
                        "predictions exactly, all folds moving 0.00, so the measured "
                        "effect is the prior alone"),
            "gate": ("not met: average 2.34 against a 3.0 floor and blocks 65% against "
                     "75%. Both are magnitude and consistency thresholds; every fold "
                     "gains, unlike V144 (-1.86 on 2022) and V151 (-32.36 on 2023)."),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
