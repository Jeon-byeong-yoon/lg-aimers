"""V192: three more Form draws, so the 0.32-weight component gets a real curve.

V190 trained Form at three seeds and V191 showed why three is not enough: an unbiased
curve needs the inside set pooled over subsets, and with three draws only k=1 and k=2 are
available. Form carries 0.32 -- more than the network V189 just fixed -- so it is worth
six.

The network's curve, once seed 42 stopped sitting in every averaged set, came out exactly
as the mechanism predicts:

    k        2022     2023     2024
    1       +0.00    +0.00    +0.00
    2       +5.40    +2.79    +2.19
    3       +6.55    +0.69    +3.37   <- shipped
    5       +6.39    +1.45    +4.34
    8       +7.19    +6.03    +5.30

2024 rises monotonically and saturates, which is what removing `1 - 1/k` of the lottery
variance looks like. Going from the shipped three to eight is worth about +1.9 more.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


PREDICTIONS = Path("artifacts/v192_more_form_seeds_predictions.joblib")
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NEW_SEEDS = (777, 999, 13)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    frame = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical, encoded
    gc.collect()
    frame = add_trackman_features(frame, trackman)
    frame = pd.concat([frame, block], axis=1)
    keep = v31_form_columns(frame)
    assert len(keep) == 105, len(keep)

    forms = {s: {} for s in NEW_SEEDS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        actual = targets[valid_mask].astype(float)
        rate = actual.mean()
        for seed in NEW_SEEDS:
            model, model_columns = hist_gbdt_pipeline(frame[keep])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()},
                             histgradientboostingclassifier__random_state=seed)
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[seed][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            del model
            gc.collect()
        skills = [100000 * (1 - ((forms[s][str(year)] - actual) ** 2).mean()
                            / (rate * (1 - rate))) for s in NEW_SEEDS]
        print(f"  {year} Form seeds {NEW_SEEDS}: "
              + "  ".join(f"{v:7.0f}" for v in skills)
              + f"   [{time.time() - started:.0f}s]", flush=True)
    joblib.dump({"forms": forms}, PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}")


if __name__ == "__main__":
    main()
