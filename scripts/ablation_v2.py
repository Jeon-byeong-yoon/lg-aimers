"""Fast HGB ablation of V2 row-wise feature groups on the 2024 holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluate_v2 import hist_gbdt_pipeline, score
from feature_engineering_v2 import FEATURE_GROUPS, add_row_features


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    train_mask = data["season"] < 2024
    valid_mask = data["season"] == 2024
    features = add_row_features(data, float(y.loc[train_mask].mean()))

    experiments = {"all_v2": []}
    experiments.update({f"without_{name}": columns for name, columns in FEATURE_GROUPS.items()})
    results = {}
    for name, dropped in experiments.items():
        candidate = features.drop(columns=dropped)
        model, columns = hist_gbdt_pipeline(candidate)
        model.fit(candidate.loc[train_mask, columns], y.loc[train_mask])
        prediction = model.predict_proba(candidate.loc[valid_mask, columns])[:, 1]
        results[name] = score(y.loc[valid_mask], prediction)
        print(name, results[name], flush=True)

    full_brier = results["all_v2"]["brier"]
    for name, result in results.items():
        result["brier_delta_vs_all"] = result["brier"] - full_brier
    output = Path("artifacts/v2_ablation_metrics.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
