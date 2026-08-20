"""Test small weight transfers inside V31's four-model tree core."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_segment_calibration_v12 import segment_correction


NAMES = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
BASE = np.array([
    0.29483562599237795, 0.2344456574665245,
    0.11617615437521091, 0.35454256216588664,
])
DELTAS = (0.005, 0.01, 0.02)


def tree_prediction(item, weights):
    matrix = np.column_stack([item["components"][name] for name in NAMES])
    return matrix @ weights


def final_prediction(item, logistic, form, context, weights):
    v17 = 0.95 * tree_prediction(item, weights) + 0.05 * logistic
    return 0.75 * v17 + 0.16 * form + 0.09 * context


def fold_score(oof, logistic, form, context, frame, history_years, valid_year, weights):
    train_y, train_p, indices = [], [], []
    for year in history_years:
        item = oof[str(year)]
        train_y.append(item["target"].astype(float))
        train_p.append(final_prediction(
            item, logistic[str(year)], form[str(year)], context[str(year)], weights
        ))
        indices.append(item["row_index"])
    train_y, train_p = np.concatenate(train_y), np.concatenate(train_p)
    residual = train_y - train_p
    train_frame = frame.loc[np.concatenate(indices)]
    valid = oof[str(valid_year)]
    valid_p = final_prediction(
        valid, logistic[str(valid_year)], form[str(valid_year)],
        context[str(valid_year)], weights,
    )
    valid_frame = frame.loc[valid["row_index"]]
    count = segment_correction(
        train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
    )
    pitcher = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(valid_p + residual.mean() + 0.75 * count + 0.25 * pitcher, 0, 1)
    return float(brier_score_loss(valid["target"], prediction))


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    removed = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    form = removed["no_trackman_std"]["form"]
    context = removed["no_matchup_hte"]["context"]
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    candidates = []
    for source in range(len(BASE)):
        for destination in range(len(BASE)):
            if source == destination:
                continue
            for delta in DELTAS:
                weights = BASE.copy()
                weights[source] -= delta
                weights[destination] += delta
                candidates.append((source, destination, delta, weights))
    item22 = oof["2022"]
    base22_prediction = final_prediction(
        item22, logistic["2022"], form["2022"], context["2022"], BASE
    )
    baseline = {
        2022: float(brier_score_loss(item22["target"], base22_prediction)),
        2023: fold_score(oof, logistic, form, context, frame, [2022], 2023, BASE),
        2024: fold_score(
            oof, logistic, form, context, frame, [2022, 2023], 2024, BASE
        ),
    }
    results = []
    for source, destination, delta, weights in candidates:
        prediction22 = final_prediction(
            item22, logistic["2022"], form["2022"], context["2022"], weights
        )
        scores = {
            2022: float(brier_score_loss(item22["target"], prediction22)),
            2023: fold_score(
                oof, logistic, form, context, frame, [2022], 2023, weights
            ),
            2024: fold_score(
                oof, logistic, form, context, frame, [2022, 2023], 2024, weights
            ),
        }
        item = {
            "from": NAMES[source], "to": NAMES[destination], "delta": delta,
            "weights": dict(zip(NAMES, weights.tolist())),
        }
        for year in (2022, 2023, 2024):
            item[f"{year}_brier"] = scores[year]
            item[f"{year}_gain_vs_v31"] = baseline[year] - scores[year]
        results.append(item)
    accepted = [item for item in results if min(
        item["2022_gain_vs_v31"], item["2023_gain_vs_v31"],
        item["2024_gain_vs_v31"],
    ) > 0]
    rank = lambda item: (
        item["2024_gain_vs_v31"], item["2023_gain_vs_v31"],
        item["2022_gain_vs_v31"],
    )
    accepted.sort(key=rank, reverse=True)
    results.sort(key=rank, reverse=True)
    output = {
        "baseline_v31": {str(k): v for k, v in baseline.items()},
        "candidate_count": len(results), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_accepted": accepted[:10], "top5_overall": results[:5],
    }
    Path("artifacts/v34_core_weight_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
