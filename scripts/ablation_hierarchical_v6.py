"""2024 ablation for hierarchical target-encoding groups on V5 HGB."""

import json
from pathlib import Path

import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import GROUPS, add_prior_season_hierarchical_encodings
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from train_v4_final import TRACKMAN_COLUMNS


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    train_mask = data["season"] < 2024
    valid_mask = data["season"] == 2024
    trackman = prepare_trackman(
        pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    )
    base_encoded = add_prior_season_target_encodings(data, y)
    experiments = {"v5_no_hierarchy": []}
    experiments.update({f"only_{group}": [group] for group in GROUPS})
    experiments["all_hierarchy"] = list(GROUPS)
    results = {}
    for name, groups in experiments.items():
        raw = (
            add_prior_season_hierarchical_encodings(base_encoded, y, groups)
            if groups
            else base_encoded
        )
        features = select_v2_features(
            add_row_features(raw, float(y.loc[train_mask].mean()))
        )
        features = add_trackman_features(features, trackman)
        model, columns = hist_gbdt_pipeline(features)
        model.fit(features.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(features.loc[valid_mask, columns])[:, 1]
        results[name] = score(y.loc[valid_mask], prediction)
        print(name, results[name], flush=True)
    Path("artifacts/v6_hierarchical_ablation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
