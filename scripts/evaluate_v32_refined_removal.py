"""Refine the successful V31 feature-removal strategy."""

import json
from pathlib import Path

import joblib
import pandas as pd

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v19_blend import fold_score
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
ALL_IDS = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
SHORT_FORM = {
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "stable_recent_disagreement",
}


def base_form_columns(frame):
    return [column for column in frame if not (
        column.startswith("tm_") and column.endswith("_std")
    )]


def base_context_columns(frame):
    return [column for column in frame if column not in MATCHUP_HTE]


def form_columns(frame, variant):
    columns = base_form_columns(frame)
    if variant == "no_all_ids":
        return [column for column in columns if column not in ALL_IDS]
    if variant == "no_short_form":
        return [column for column in columns if column not in SHORT_FORM]
    if variant == "no_pitchmix":
        return [column for column in columns if not (
            column.startswith("asof_pitcher_") and "pitchmix" in column
            or column.startswith("asof_pitcher_") and column.endswith((
                "fastball_rate", "breaking_rate", "offspeed_rate"
            ))
            or column.startswith("tm_pitch_group_")
        )]
    raise ValueError(variant)


def context_columns(frame, variant):
    columns = base_context_columns(frame)
    if variant == "no_all_ids":
        return [column for column in columns if column not in ALL_IDS]
    if variant == "s500_only":
        return [column for column in columns if not column.endswith("_s100")]
    if variant == "no_context_pitchmix":
        return [column for column in columns if not column.startswith("tm_ctx_pitch_group_")]
    raise ValueError(variant)


def train_predict(features, columns, train_mask, valid_mask, target):
    candidate = features[columns]
    model, columns = hist_gbdt_pipeline(candidate)
    model.fit(candidate.loc[train_mask, columns], target.loc[train_mask])
    return model.predict_proba(candidate.loc[valid_mask, columns])[:, 1]


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
    form_variants = ("no_all_ids", "no_short_form", "no_pitchmix")
    context_variants = ("no_all_ids", "s500_only", "no_context_pitchmix")
    predictions = {
        "form": {name: {} for name in form_variants},
        "context": {name: {} for name in context_variants},
    }
    for year in (2022, 2023, 2024):
        train_mask, valid_mask = data["season"] < year, data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        context_features = select_v2_features(add_row_features(hierarchical, prior))
        context_features = add_trackman_features(context_features, trackman)
        context_features = add_context_trackman_features(context_features, context_trackman)
        for name in form_variants:
            predictions["form"][name][str(year)] = train_predict(
                form_features, form_columns(form_features, name), train_mask, valid_mask, y
            )
            print(year, "form", name, flush=True)
        for name in context_variants:
            predictions["context"][name][str(year)] = train_predict(
                context_features, context_columns(context_features, name),
                train_mask, valid_mask, y,
            )
            print(year, "context", name, flush=True)
    joblib.dump(predictions, "artifacts/v32_refined_removal_predictions.joblib", compress=3)

    v31 = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    base_form = v31["no_trackman_std"]["form"]
    base_context = v31["no_matchup_hte"]["context"]
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_choices = ["v31"] + list(form_variants)
    context_choices = ["v31"] + list(context_variants)
    candidates = []
    for form_name in form_choices:
        for context_name in context_choices:
            if form_name == context_name == "v31":
                continue
            form = base_form if form_name == "v31" else predictions["form"][form_name]
            context = (
                base_context if context_name == "v31"
                else predictions["context"][context_name]
            )
            component = {
                year: (0.16 * form[year] + 0.09 * context[year]) / 0.25
                for year in form
            }
            scores = {
                2023: fold_score(oof, logistic, component, data, [2022], 2023, 0.25),
                2024: fold_score(oof, logistic, component, data, [2022, 2023], 2024, 0.25),
            }
            candidates.append({
                "form": form_name, "context": context_name,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
            })
    baseline = {2023: 0.25326829650157456, 2024: 0.24783690211517923}
    for item in candidates:
        for year in (2023, 2024):
            item[f"{year}_gain_vs_v31"] = baseline[year] - item[f"{year}_brier"]
    accepted = [item for item in candidates if min(
        item["2023_gain_vs_v31"], item["2024_gain_vs_v31"]
    ) > 0]
    rank = lambda item: (item["2024_gain_vs_v31"], item["2023_gain_vs_v31"])
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v31": {str(k): v for k, v in baseline.items()},
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_overall": candidates[:10],
    }
    Path("artifacts/v32_refined_removal_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
