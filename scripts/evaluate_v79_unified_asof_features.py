"""V79: 통합 As-of 신뢰도 피처의 3시즌·12블록 검증."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, "scripts")
from asof_features_v79 import add_asof_reliability_features, learn_asof_priors
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v78_bayesian_form_delta import BASELINE, FORM_CONFIG, YEARS, evaluate, form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


VARIANTS = ("rates", "hierarchy", "pitchmix")


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    _, baseline_predictions = evaluate(baseline_form, oof, logistic, context, data)

    candidate_predictions = {variant: {} for variant in VARIANTS}
    for year in YEARS:
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        priors = learn_asof_priors(data.loc[train_mask])
        for variant in VARIANTS:
            asof = add_asof_reliability_features(hierarchical, priors, variant)
            form_raw = add_stable_form_features(asof)
            features = select_v2_features(add_row_features(form_raw, prior))
            features = add_trackman_features(features, trackman)
            columns = form_columns(features)
            model, columns = hist_gbdt_pipeline(features[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{name}": value
                for name, value in FORM_CONFIG.items()
            })
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            candidate_predictions[variant][str(year)] = model.predict_proba(
                features.loc[valid_mask, columns]
            )[:, 1]
            del asof, form_raw, features, model
        print(f"year={year} done", flush=True)

    results = []
    for variant in VARIANTS:
        scores, predictions = evaluate(candidate_predictions[variant], oof, logistic, context, data)
        gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
        blocks = []
        for year in YEARS:
            key = str(year)
            index = oof[key]["row_index"]
            months = data.loc[index, "game_month"].to_numpy()
            target = oof[key]["target"].astype(float)
            candidate_error = (predictions[key] - target) ** 2
            baseline_error = (baseline_predictions[key] - target) ** 2
            for label, mask in (
                ("early_3_5", months <= 5),
                ("mid_6_7", (months >= 6) & (months <= 7)),
                ("late_8", months == 8),
                ("finish_9_10", months >= 9),
            ):
                blocks.append({
                    "season": year, "block": label, "n": int(mask.sum()),
                    "gain": float(baseline_error[mask].mean() - candidate_error[mask].mean()),
                })
        wins = sum(block["gain"] > 0 for block in blocks)
        results.append({
            "variant": variant,
            "scores": scores,
            "gains_vs_v41": gains,
            "block_wins": wins,
            "block_count": len(blocks),
            "block_win_rate": wins / len(blocks),
            "blocks": blocks,
            "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
            "submit_ready": (
                all(gains[str(year)] > 0 for year in YEARS)
                and gains["2024"] >= 5e-5
                and wins / len(blocks) >= 0.70
            ),
        })
    results.sort(key=lambda row: (row["gains_vs_v41"]["2024"], row["block_win_rate"]), reverse=True)
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V79_unified_asof_features",
        "compliance": {
            "official_data_only": True,
            "priors_fit_on_past_training_seasons_only": True,
            "test_row_aggregation_used": False,
            "post_pitch_information_used": False,
        },
        "candidate_count": len(results),
        "submit_ready_count": len(ready),
        "best": results[0],
        "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v79_unified_asof_features_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(candidate_predictions, "artifacts/v79_unified_asof_predictions.joblib", compress=3)
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
