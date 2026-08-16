"""Materialize accepted V6 OOF predictions for calibration experiments."""

import joblib
import pandas as pd

from evaluate_v2 import extra_trees_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features


def main() -> None:
    v4 = joblib.load("artifacts/v4_trackman_predictions.joblib")
    v5 = joblib.load("artifacts/v5_target_encoding_predictions.joblib")
    v6 = joblib.load("artifacts/v6_hierarchical_predictions.joblib")
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    saved = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        base = select_v2_features(add_row_features(data, float(y.loc[train_mask].mean())))
        extra, columns = extra_trees_pipeline(base)
        extra.fit(base.loc[train_mask, columns], y.loc[train_mask])
        extra_prediction = extra.predict_proba(base.loc[valid_mask, columns])[:, 1]
        prediction = (
            0.40 * extra_prediction
            + 0.36 * v4[str(year)]["trackman_hgb"]
            + 0.144 * v5[str(year)]["te_trackman_hgb"]
            + 0.096 * v6[str(year)]["hierarchical_hgb"]
        )
        saved[str(year)] = {
            "target": y.loc[valid_mask].to_numpy(),
            "prediction": prediction,
            "row_index": data.index[valid_mask].to_numpy(),
            "components": {
                "extra_trees": extra_prediction,
                "trackman_hgb": v4[str(year)]["trackman_hgb"],
                "te_trackman_hgb": v5[str(year)]["te_trackman_hgb"],
                "hierarchical_hgb": v6[str(year)]["hierarchical_hgb"],
            },
        }
        print(year, prediction.mean(), flush=True)
    joblib.dump(saved, "artifacts/v6_oof_predictions.joblib", compress=3)


if __name__ == "__main__":
    main()
