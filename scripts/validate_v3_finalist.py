"""Compare the V3 finalist against V2 on earlier chronological folds."""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline, score
from feature_engineering_v2 import add_row_features, select_v2_features
from tune_v3 import replace_estimator


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    results = {}
    for year in (2022, 2023):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        features = select_v2_features(
            add_row_features(data, float(y.loc[train_mask].mean()))
        )
        hist, hist_columns = hist_gbdt_pipeline(features)
        hist.fit(features.loc[train_mask, hist_columns], y.loc[train_mask])
        hist_prediction = hist.predict_proba(features.loc[valid_mask, hist_columns])[:, 1]

        extra, extra_columns = extra_trees_pipeline(features)
        replace_estimator(
            extra,
            ExtraTreesClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=100,
                max_features=0.7,
                n_jobs=-1,
                random_state=42,
            ),
        )
        extra.fit(features.loc[train_mask, extra_columns], y.loc[train_mask])
        extra_prediction = extra.predict_proba(features.loc[valid_mask, extra_columns])[:, 1]
        prediction = 0.60 * extra_prediction + 0.40 * hist_prediction
        results[str(year)] = score(y.loc[valid_mask], prediction)
        print(year, results[str(year)], flush=True)
    Path("artifacts/v3_temporal_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
