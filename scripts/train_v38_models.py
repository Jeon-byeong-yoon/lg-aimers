"""Train V38 models: form HGB gentle_500 (lr=0.03, iter=500) + V31 context.

Trains on full training data (all seasons) for submission inference.
"""

from pathlib import Path

import joblib
import pandas as pd

from contextual_trackman_v24 import (
    add_context_trackman_features, build_context_lookup, prepare_context_trackman,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

# V38 gentle_500 — the first config to pass the 2024 >5e-6 threshold
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
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)
    prior = float(y.mean())

    # Form model: V38 gentle_500 config
    form_raw = add_stable_form_features(hierarchical)
    form_features = select_v2_features(add_row_features(form_raw, prior))
    form_features = add_trackman_features(form_features, trackman)
    form_columns = [
        c for c in form_features
        if not (c.startswith("tm_") and c.endswith("_std"))
    ]
    form_model, form_columns = hist_gbdt_pipeline(form_features[form_columns])
    form_model.set_params(**{
        f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
    })
    form_model.fit(form_features[form_columns], y)
    print("Form model trained.", flush=True)

    # Context model: V31 original (no_matchup_hte) — reuse from V31 artifact
    v31 = joblib.load("artifacts/v31_feature_models.joblib")
    context_model = v31["context_model"]
    context_columns = v31["context_columns"]
    print("Context model loaded from V31 artifact.", flush=True)

    artifact = {
        "form_model": form_model,
        "form_columns": form_columns,
        "context_model": context_model,
        "context_columns": context_columns,
        "context_lookup_2025": build_context_lookup(context_trackman, 2025),
    }
    output = Path("artifacts/v38_feature_models.joblib")
    joblib.dump(artifact, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
