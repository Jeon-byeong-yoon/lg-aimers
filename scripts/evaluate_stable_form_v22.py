"""Evaluate stable recent-form features in hierarchical Trackman HGB."""

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


DROP_VARIANTS = {
    "all_stable_form": [],
    "success_only": ["stable_recent_middle", "stable_recent_middle_gap"],
}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    hierarchical = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    results, saved = {}, {}
    existing = joblib.load("artifacts/v22_stable_form_predictions.joblib")
    existing_metrics = json.loads(
        Path("artifacts/v22_stable_form_metrics.json").read_text(encoding="utf-8")
    )
    results["all_stable_form"] = existing_metrics
    saved["all_stable_form"] = existing
    for variant, dropped in DROP_VARIANTS.items():
        if variant == "all_stable_form":
            continue
        variant_results, variant_saved = {}, {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            features = select_v2_features(add_row_features(hierarchical, prior))
            features = add_trackman_features(features, trackman).drop(columns=dropped)
            model, columns = hist_gbdt_pipeline(features)
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
            variant_results[str(year)] = score(y.loc[valid_mask], prediction)
            variant_saved[str(year)] = prediction
            print(variant, year, variant_results[str(year)], flush=True)
        results[variant] = variant_results
        saved[variant] = variant_saved
    Path("artifacts/v22_stable_form_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    joblib.dump(saved, "artifacts/v22_stable_form_variants.joblib", compress=3)


if __name__ == "__main__":
    main()
