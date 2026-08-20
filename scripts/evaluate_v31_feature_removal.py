"""Chronologically test conservative feature removals from V25 HGB components."""

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


REMOVALS = {
    "no_player_ids": {"pitcher_id", "batter_id"},
    "no_all_ids": {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"},
    "no_matchup_hte": {
        "hte_pitcher_batter_100", "hte_pitcher_batter_500",
        "hte_pitcher_batter_log_count",
    },
    "no_trackman_std": set(),
}


def columns_after_removal(columns, name):
    if name == "no_trackman_std":
        return [column for column in columns if not (
            column.startswith("tm_") and column.endswith("_std")
        )]
    return [column for column in columns if column not in REMOVALS[name]]


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
    predictions = {
        name: {"form": {}, "context": {}} for name in REMOVALS
    }
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        context_features = select_v2_features(add_row_features(hierarchical, prior))
        context_features = add_trackman_features(context_features, trackman)
        context_features = add_context_trackman_features(context_features, context_trackman)
        for name in REMOVALS:
            for component, features in (
                ("form", form_features), ("context", context_features)
            ):
                columns = columns_after_removal(list(features.columns), name)
                candidate_features = features[columns]
                model, columns = hist_gbdt_pipeline(candidate_features)
                model.fit(candidate_features.loc[train_mask, columns], y.loc[train_mask])
                predictions[name][component][str(year)] = model.predict_proba(
                    candidate_features.loc[valid_mask, columns]
                )[:, 1]
            print(year, name, flush=True)
    joblib.dump(
        predictions, "artifacts/v31_feature_removal_predictions.joblib", compress=3
    )

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    base_form = joblib.load("artifacts/v22_stable_form_variants.joblib")["all_stable_form"]
    base_context = joblib.load("artifacts/v24_contextual_trackman_predictions.joblib")
    frame = data
    choices = ["base"] + list(REMOVALS)
    candidates = []
    for form_name in choices:
        for context_name in choices:
            if form_name == context_name == "base":
                continue
            form = base_form if form_name == "base" else predictions[form_name]["form"]
            context = (
                base_context if context_name == "base"
                else predictions[context_name]["context"]
            )
            component = {
                year: (0.16 * form[year] + 0.09 * context[year]) / 0.25
                for year in form
            }
            scores = {
                2023: fold_score(oof, logistic, component, frame, [2022], 2023, 0.25),
                2024: fold_score(oof, logistic, component, frame, [2022, 2023], 2024, 0.25),
            }
            candidates.append({
                "form": form_name, "context": context_name,
                "2023_brier": scores[2023], "2024_brier": scores[2024],
            })
    baseline = {2023: 0.2533208415598194, 2024: 0.24784544408284379}
    for item in candidates:
        for year in (2023, 2024):
            item[f"{year}_gain_vs_v25"] = baseline[year] - item[f"{year}_brier"]
    accepted = [item for item in candidates if min(
        item["2023_gain_vs_v25"], item["2024_gain_vs_v25"]
    ) > 0]
    rank = lambda item: (
        item["2024_gain_vs_v25"], item["2023_gain_vs_v25"],
    )
    accepted.sort(key=rank, reverse=True)
    candidates.sort(key=rank, reverse=True)
    output = {
        "baseline_v25": {str(k): v for k, v in baseline.items()},
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "best": accepted[0] if accepted else None,
        "top10_overall": candidates[:10],
    }
    Path("artifacts/v31_feature_removal_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
