"""Test stable game-state residual corrections on the V27 candidate."""

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction


BLEND = {"v17": 0.69, "form": 0.12, "context_v26": 0.19}
GROUPS = {
    "base_state": ["base_state"],
    "count_hand": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "outs_base": ["outs_before", "base_state"],
    "inning_base": ["inning_bucket", "base_state"],
    "score_base": ["score_bucket", "base_state"],
    "team_count": ["pitcher_team_id", "balls_before", "strikes_before"],
}
SMOOTHING = [100, 300, 1000, 3000, 10000]
STRENGTHS = [0.10, 0.25, 0.50, 0.75, 1.00]
V25_BRIER = {2023: 0.2533208415598194, 2024: 0.24784544408284379}


def add_context(frame):
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], [0, 3, 6, 9, np.inf], labels=False, include_lowest=True
    ).astype("int8")
    output["score_bucket"] = pd.cut(
        output["score_diff_pitcher_team"],
        [-np.inf, -4, -2, 1, 3, np.inf], labels=False,
    ).astype("int8")
    return output


def blend(item, logistic, form, context):
    v17 = 0.95 * v11_prediction(item) + 0.05 * logistic
    return BLEND["v17"] * v17 + BLEND["form"] * form + BLEND["context_v26"] * context


def make_fold(oof, logistic, form, context, frame, history_years, valid_year):
    train_y, train_p, indices = [], [], []
    for year in history_years:
        item = oof[str(year)]
        train_y.append(item["target"].astype(float))
        train_p.append(blend(item, logistic[str(year)], form[str(year)], context[str(year)]))
        indices.append(item["row_index"])
    train_y, train_p = np.concatenate(train_y), np.concatenate(train_p)
    residual = train_y - train_p
    train_frame = frame.loc[np.concatenate(indices)]
    valid = oof[str(valid_year)]
    valid_frame = frame.loc[valid["row_index"]]
    valid_p = blend(valid, logistic[str(valid_year)], form[str(valid_year)], context[str(valid_year)])
    count = segment_correction(
        train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 1000
    )
    pitcher = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 500,
    )
    baseline = np.clip(valid_p + residual.mean() + 0.85 * count + 0.40 * pitcher, 0, 1)
    corrections = {
        (name, smoothing): segment_correction(
            train_frame, residual, valid_frame, columns, smoothing
        ) for name, columns in GROUPS.items() for smoothing in SMOOTHING
    }
    return valid["target"].astype(float), baseline, corrections


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    context = joblib.load("artifacts/v26_contextual_trackman_predictions.joblib")
    raw = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig"
    ).drop(columns=["row_id", "control_success"])
    frame = add_context(raw)
    folds = {
        2023: make_fold(oof, logistic, form, context, frame, [2022], 2023),
        2024: make_fold(oof, logistic, form, context, frame, [2022, 2023], 2024),
    }
    baseline = {
        year: float(brier_score_loss(fold[0], fold[1])) for year, fold in folds.items()
    }
    singles = []
    for name in GROUPS:
        for smoothing in SMOOTHING:
            for strength in STRENGTHS:
                scores = {
                    year: float(brier_score_loss(
                        fold[0], np.clip(
                            fold[1] + strength * fold[2][(name, smoothing)], 0, 1
                        )
                    )) for year, fold in folds.items()
                }
                singles.append({
                    "name": name, "smoothing": smoothing, "strength": strength,
                    "2023_brier": scores[2023], "2024_brier": scores[2024],
                    "2023_gain_vs_v27": baseline[2023] - scores[2023],
                    "2024_gain_vs_v27": baseline[2024] - scores[2024],
                    "2023_gain_vs_v25": V25_BRIER[2023] - scores[2023],
                    "2024_gain_vs_v25": V25_BRIER[2024] - scores[2024],
                })
    stable = [x for x in singles if min(x["2023_gain_vs_v27"], x["2024_gain_vs_v27"]) > 0]
    rank = lambda x: (
        min(x["2023_gain_vs_v27"], x["2024_gain_vs_v27"]),
        x["2023_gain_vs_v27"] + x["2024_gain_vs_v27"],
    )
    stable.sort(key=rank, reverse=True)
    best_by_group = {}
    for item in stable:
        best_by_group.setdefault(item["name"], item)
    mixtures = []
    for left, right in combinations(list(best_by_group)[:5], 2):
        a, b = best_by_group[left], best_by_group[right]
        scores = {}
        for year, fold in folds.items():
            correction = 0.5 * (
                a["strength"] * fold[2][(left, a["smoothing"])]
                + b["strength"] * fold[2][(right, b["smoothing"])]
            )
            scores[year] = float(brier_score_loss(
                fold[0], np.clip(fold[1] + correction, 0, 1)
            ))
        item = {
            "left": a, "right": b,
            "2023_brier": scores[2023], "2024_brier": scores[2024],
            "2023_gain_vs_v27": baseline[2023] - scores[2023],
            "2024_gain_vs_v27": baseline[2024] - scores[2024],
            "2023_gain_vs_v25": V25_BRIER[2023] - scores[2023],
            "2024_gain_vs_v25": V25_BRIER[2024] - scores[2024],
        }
        if min(item["2023_gain_vs_v27"], item["2024_gain_vs_v27"]) > 0:
            mixtures.append(item)
    accepted = stable + mixtures
    accepted.sort(key=rank, reverse=True)
    output = {
        "baseline_v27": {str(k): v for k, v in baseline.items()},
        "baseline_v25": {str(k): v for k, v in V25_BRIER.items()},
        "stable_single_count": len(stable), "stable_mixture_count": len(mixtures),
        "best": accepted[0] if accepted else None, "top10": accepted[:10],
    }
    Path("artifacts/v29_context_residual_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
