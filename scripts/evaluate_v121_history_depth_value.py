"""V121: how much does CatBoost gain per extra training season, and does its optimal
blend weight move with that?

V117 scored 1028.2400 against V114's 1002.5720, a gain of +25.67 where the 2024
bootstrap predicted +9.9. That instrument had tracked the leaderboard at a cumulative
ratio of 1.082 over six adopted candidates, so a 2.59x under-prediction is not noise
in the instrument; something structural differs between what was measured and what
was deployed.

The candidate explanation is training depth. The 2024 fold's CatBoost saw five
seasons (2019-2023); the deployed model sees six (2019-2024). Both predict the
immediately following season, so depth is the *only* difference between them. A model
whose whole advantage is ordered target statistics over 792 pitchers and 830 batters
is exactly the kind that keeps improving with more history, which would make every
CatBoost measurement taken on the folds an underestimate of the deployed component.

That also has a consequence the weight ladder cannot see on its own: if the deployed
component is stronger than the one V120 weighted, its optimum weight is higher than
V120's peak. The embedding network showed the same sign (predicted +24.5, actual
+29.5), which fits, since it is the other component that learns per-entity parameters.

So this measures the curve directly. Depth is varied while the target stays 2024 and
the training window stays adjacent to it, which is precisely the fold-versus-deployed
contrast: one, two, three, four and five seasons ending in 2023. Two things are read
off it -- how standalone skill grows with depth, and whether the blend weight that
maximises the 2024 fold drifts upward as depth grows. Extrapolating both one step
gives a defensible estimate for the six-season deployed model instead of a guess.
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
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v121_history_depth_metrics.json")
PREDICTIONS = Path("artifacts/v121_history_depth_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
TARGET = 2024
DEPTHS = (1, 2, 3, 4, 5)
CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
# Weights are swept on the 2024 fold alone here, so the ladder can run finely; the
# other four are held at V117's values with Form absorbing the change, as in V120.
CATBOOST_LADDER = tuple(round(0.10 + 0.02 * i, 2) for i in range(16))
BASE = {"v17": 0.0, "form": 0.45, "context": 0.21, "network": 0.20}
P = 100000.0 / 0.25


def skill(prediction, target):
    rate = target.mean()
    return float(100000 * (1 - ((prediction - target) ** 2).mean() / (rate * (1 - rate))))


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

    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    valid_mask = season == TARGET
    valid_target = targets[valid_mask].astype(float)

    predictions, standalone, rows = {}, {}, {}
    for depth in DEPTHS:
        started = time.time()
        train_mask = (season >= TARGET - depth) & (season < TARGET)
        model = CatBoostClassifier(**CONFIG, random_seed=42, verbose=0, thread_count=6,
                                   cat_features=categorical, allow_writing_files=False)
        model.fit(work.loc[train_mask], targets[train_mask])
        prediction = model.predict_proba(work.loc[valid_mask])[:, 1]
        predictions[str(depth)] = prediction
        standalone[str(depth)] = skill(prediction, valid_target)
        rows[str(depth)] = int(train_mask.sum())
        print(f"  depth {depth} ({TARGET - depth}-{TARGET - 1}, "
              f"{train_mask.sum():,} rows): standalone {standalone[str(depth)]:8.0f}  "
              f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    key = str(TARGET)
    parts = {
        "v17": 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key],
        "form": form[key], "context": context[key], "network": network[key],
    }
    item = oof[key]
    fold_target = item["target"].astype(float)
    fold_term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[item["row_index"]], 1.0)

    # No calibration refit here: on a single fold the shift is fitted on the same rows
    # it corrects, which would flatter every candidate equally and blur the ladder.
    # A constant offset cannot move the argmax over the weight, which is all this reads.
    optima = {}
    for depth in DEPTHS:
        curve = {}
        for weight in CATBOOST_LADDER:
            form_weight = BASE["form"] - (weight - 0.14)
            if form_weight <= 0:
                continue
            raw = (BASE["v17"] * parts["v17"] + form_weight * parts["form"]
                   + BASE["context"] * parts["context"]
                   + BASE["network"] * parts["network"]
                   + weight * predictions[str(depth)])
            blended = np.clip(raw + DRIFT_WEIGHT * fold_term, 0, 1)
            curve[f"{weight:.2f}"] = skill(blended, fold_target)
        best = max(curve, key=curve.get)
        optima[str(depth)] = {"curve": curve, "argmax": float(best),
                              "best_skill": curve[best]}
        print(f"  depth {depth}: optimal catboost weight {best}  "
              f"skill {curve[best]:8.0f}", flush=True)

    depths = np.array(DEPTHS, dtype=float)
    skills = np.array([standalone[str(d)] for d in DEPTHS])
    argmaxes = np.array([optima[str(d)]["argmax"] for d in DEPTHS])
    # Skill against log-depth, because each extra season is a smaller relative
    # increase in history than the one before it.
    skill_fit = np.polyfit(np.log(depths), skills, 1)
    weight_fit = np.polyfit(np.log(depths), argmaxes, 1)
    projection = {
        "standalone_skill_at_depth_6": float(np.polyval(skill_fit, np.log(6.0))),
        "standalone_skill_at_depth_5": float(np.polyval(skill_fit, np.log(5.0))),
        "optimal_weight_at_depth_6": float(np.polyval(weight_fit, np.log(6.0))),
        "optimal_weight_at_depth_5": float(np.polyval(weight_fit, np.log(5.0))),
        "skill_per_log_depth": float(skill_fit[0]),
        "weight_per_log_depth": float(weight_fit[0]),
    }

    OUTPUT.write_text(json.dumps({
        "experiment": "V121_history_depth_value",
        "motivation": (
            "V117 gained +25.67 where the 2024 bootstrap predicted +9.9, a 2.59x "
            "under-prediction from an instrument that had tracked at 1.082 over six "
            "candidates. The fold model saw five seasons and the deployed model six, "
            "which is the only structural difference between them."
        ),
        "target_season": TARGET,
        "depths": list(DEPTHS),
        "training_rows": rows,
        "config": CONFIG,
        "standalone_skill": standalone,
        "held_weights": BASE,
        "weight_ladder": list(CATBOOST_LADDER),
        "fold_weight_optima": optima,
        "projection": projection,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)

    print("\ndepth -> standalone 2024 skill, optimal catboost weight")
    for depth in DEPTHS:
        print(f"  {depth}  {standalone[str(depth)]:8.0f}  "
              f"{optima[str(depth)]['argmax']:.2f}")
    print(f"\nper log-depth: skill {projection['skill_per_log_depth']:+.0f}, "
          f"weight {projection['weight_per_log_depth']:+.4f}")
    print(f"projected at depth 6: skill "
          f"{projection['standalone_skill_at_depth_6']:.0f} "
          f"(depth 5 fit {projection['standalone_skill_at_depth_5']:.0f}), "
          f"weight {projection['optimal_weight_at_depth_6']:.3f} "
          f"(depth 5 fit {projection['optimal_weight_at_depth_5']:.3f})")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
