"""Build V105: feature shrinkage 20, drift shrinkage 10, reliability 150, w 0.15.

V96 used shrinkage 50 in both places. V102-V104 showed 20 is a genuine interior
optimum for the features (10 and 15 were worse, so it is not a boundary artifact)
and that the drift term prefers even less shrinkage. All 25 shortlisted candidates
in V104 had a positive 2024 point estimate; this one was positive on all three
seasons with an 85% monthly block win rate.

The 2024 confidence interval still straddles zero, so this is a leaderboard
experiment rather than a confident promotion — see the record for the reasoning.

Must be run with the server-mirror interpreter: the Form model is newly fitted,
and a numpy 2.x pickle of it cannot be loaded by the server's numpy 1.26.4.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from contextual_trackman_v24 import build_context_lookup, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, build_anchors, build_priors, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 10.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.15
W_V17, W_FORM, W_CONTEXT = 0.27, 0.52, 0.21
GROUPS = ("pitcher", "batter")


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    context_trackman = prepare_context_trackman(trackman)

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    keep = v31_form_columns(features)
    form_model, form_columns = hist_gbdt_pipeline(features[keep])
    form_model.set_params(**{
        f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()})
    form_model.fit(features[form_columns], y)
    missing = [c for c in feature_names() if c not in form_columns]
    if missing:
        raise ValueError(f"in-season features dropped: {missing}")
    print(f"Form model trained: {len(features)} rows, {len(form_columns)} features",
          flush=True)

    v31 = joblib.load("artifacts/v31_feature_models.joblib")
    models = Path("artifacts/v105_feature_models.joblib")
    joblib.dump({
        "form_model": form_model,
        "form_columns": form_columns,
        "context_model": v31["context_model"],
        "context_columns": v31["context_columns"],
        "context_lookup_2025": build_context_lookup(context_trackman, 2025),
    }, models, compress=3)
    print(f"Saved {models} ({models.stat().st_size / 2**20:.2f} MiB)")

    # Calibration for this blend, using the matching shrinkage-20 Form OOF.
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_oof = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form_oof = form_oof["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]

    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        predictions.append(
            W_V17 * v17 + W_FORM * form_oof[str(year)] + W_CONTEXT * context[str(year)])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    calibration = Path("artifacts/v105_calibration.joblib")
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
    }, calibration, compress=3)
    print(f"Saved {calibration}, shift={shift:.12f} (V96 was -0.010036232597)")

    anchor = Path("artifacts/v105_inseason_anchor.joblib")
    joblib.dump({
        "anchors": build_anchors(raw_frame, GROUPS),
        "priors": build_priors(raw_frame, GROUPS),
        "groups": list(GROUPS),
        "target_season": 2025,
        "anchor_season_max": int(raw_frame["season"].max()),
        "train_rows": int(len(raw_frame)),
    }, anchor, compress=3)
    print(f"Saved {anchor} ({anchor.stat().st_size / 2**10:.1f} KiB)")


if __name__ == "__main__":
    main()
