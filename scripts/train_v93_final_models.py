"""Train V93 final models: Form HGB with the current-season as-of reconstruction.

Identical to V38 gentle_500 except the Form feature set gains the 18 in-season
reconstruction columns. Training rows build them per season from the previous
season's frozen anchor, which is exactly the relation a 2025 evaluation row has
to the 2019-2024 training data. The Context model is reused unchanged.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import build_context_lookup, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v93_feature_models.joblib")
FORM_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.03,
    "max_iter": 500,
}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)
    prior = float(y.mean())

    inseason_block = add_training_inseason_features(raw_frame)[feature_names()]
    form_raw = add_stable_form_features(hierarchical)
    form_features = select_v2_features(add_row_features(form_raw, prior))
    form_features = add_trackman_features(form_features, trackman)
    form_features = pd.concat([form_features, inseason_block], axis=1)
    keep = [c for c in form_features if not (c.startswith("tm_") and c.endswith("_std"))]
    form_model, form_columns = hist_gbdt_pipeline(form_features[keep])
    form_model.set_params(**{
        f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
    })
    form_model.fit(form_features[form_columns], y)
    missing = [c for c in feature_names() if c not in form_columns]
    if missing:
        raise ValueError(f"In-season features dropped from Form model: {missing}")
    print(f"Form model trained on {len(form_features)} rows, "
          f"{len(form_columns)} features ({len(feature_names())} in-season).", flush=True)

    v31 = joblib.load("artifacts/v31_feature_models.joblib")
    artifact = {
        "form_model": form_model,
        "form_columns": form_columns,
        "context_model": v31["context_model"],
        "context_columns": v31["context_columns"],
        "context_lookup_2025": build_context_lookup(context_trackman, 2025),
    }
    joblib.dump(artifact, OUTPUT, compress=3)
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
