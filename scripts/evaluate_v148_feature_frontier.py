"""V148: is there any signal left in these features, measured as favourably as possible?

The residual audit showed only about 16 points of per-key structure left on the 2024
fold, but that test only asks whether a *per-pitcher, per-batter or per-count additive
correction* would help. It cannot see signal that lives in an interaction of continuous
features no model has captured -- residuals could look flat by pitcher and still be
predictable.

So this asks the question directly. Within a single fold, split the rows in half, train a
regressor on half A to predict the blend's residual from the full 96-column feature set,
and measure the gain on half B. If a model given the fold's own data cannot find
structure in the residual, no model trained on earlier seasons will find it either.

The test is deliberately rigged in favour of finding something:

  * the residual model trains on rows from the *same season* it is scored on, which the
    deployed pipeline can never do -- it must estimate from 2022-2024 and transfer to 2025
  * it sees the same features the blend already used, so anything it finds is signal the
    blend left on the table rather than new information
  * half a season is 125,000 rows, ample for a shallow model

Two capacities are tried, because a null result from an under-powered model would mean
nothing. Both are regularised: residuals of a calibrated ensemble are mostly noise, and
the failure mode to avoid is reporting fitted noise as headroom, which the held-out half
is there to prevent.

A near-zero result on 2022 and 2024 would mean the blend is at the frontier this feature
set supports, and that further modelling effort cannot close the gap to 1100. 2023 is
reported for completeness and expected to be large: the reliability audit found roughly
659 points there recoverable by shrinking the prediction spread alone, a one-off
dispersion anomaly that every one of the nine rejected experiments was partly capturing.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v137_context_slot_replacement import NAMES6, blend6
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v148_feature_frontier.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CONFIGS = {
    "shallow": dict(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=10.0),
    "deep": dict(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=10.0),
}
SPLITS = 3
SEED = 20260826


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
    del features, hierarchical, encoded, form_raw, trackman, data
    gc.collect()
    print(f"feature frame {work.shape[0]:,} x {work.shape[1]}", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    prediction = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6), parts, oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    validation_frame = make_validation_frame()
    fold_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    # `order` maps blend rows back to the training frame, so the feature rows line up.
    aligned = work.loc[order].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    results = {}
    print(f"\n{'fold':>6} {'config':>9} {'split':>6} {'gain(pts)':>10} "
          f"{'resid sd':>9} {'pred sd':>9}")
    for year in YEARS:
        mask = fold_of == year
        p, t = prediction[mask], target[mask]
        frame = aligned.loc[mask].reset_index(drop=True)
        residual = t - p
        variance = t.mean() * (1 - t.mean())
        for name, config in CONFIGS.items():
            gains = []
            for split in range(SPLITS):
                started = time.time()
                half = rng.random(len(p)) < 0.5
                model = CatBoostRegressor(**config, loss_function="RMSE",
                                          random_seed=42, verbose=0, thread_count=6,
                                          cat_features=categorical,
                                          allow_writing_files=False)
                model.fit(frame.loc[half], residual[half])
                correction = model.predict(frame.loc[~half])
                del model
                gc.collect()
                base = ((p[~half] - t[~half]) ** 2).mean()
                after = ((np.clip(p[~half] + correction, 0, 1) - t[~half]) ** 2).mean()
                gain = 100000 * (base - after) / variance
                gains.append(gain)
                print(f"{year:>6} {name:>9} {split:>6} {gain:10.2f} "
                      f"{residual[~half].std():9.5f} {correction.std():9.5f}  "
                      f"[{time.time() - started:.0f}s]", flush=True)
            results[f"{year}_{name}"] = {
                "fold": year, "config": name, "gains": gains,
                "mean_gain_points": float(np.mean(gains)),
                "sd_gain_points": float(np.std(gains)),
            }

    OUTPUT.write_text(json.dumps({
        "experiment": "V148_feature_frontier",
        "question": (
            "The per-key residual audit found about 16 points left on 2024, but it only "
            "tests additive corrections by pitcher, batter or count. This asks whether "
            "any model can predict the blend's residual from the full feature set."
        ),
        "why_favourable": (
            "The residual model trains on rows from the same season it is scored on, "
            "which the deployed pipeline can never do; it sees the features the blend "
            "already used, so anything found is signal left on the table; and half a "
            "season is 125,000 rows. A near-zero result therefore bounds what any "
            "further modelling of this feature set could recover."
        ),
        "configs": CONFIGS,
        "splits": SPLITS,
        "results": results,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "diagnostic_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsummary (mean over splits):")
    for key, value in results.items():
        print(f"  {key:16s} {value['mean_gain_points']:+9.2f} "
              f"+/- {value['sd_gain_points']:.2f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
