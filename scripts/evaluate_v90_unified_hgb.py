"""V90: server-compatible unified Form + Context Trackman HGB.

The unified model is trained directly on all permitted row-wise, historical
form, hierarchical encoding, Trackman, and contextual Trackman features. It
partially replaces V41's separate Form/Context component. Candidate selection
excludes the sealed 2024 September-October confirmation window.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v90_unified_hgb_metrics.json")
PREDICTIONS = Path("artifacts/v90_unified_hgb_predictions.joblib")
CONFIGS = {
    "gentle": {
        "max_leaf_nodes": 15,
        "min_samples_leaf": 200,
        "l2_regularization": 20.0,
        "learning_rate": 0.03,
        "max_iter": 500,
    },
    "strong": {
        "max_leaf_nodes": 15,
        "min_samples_leaf": 300,
        "l2_regularization": 30.0,
        "learning_rate": 0.03,
        "max_iter": 400,
    },
}
REPLACEMENT_RATIOS = (0.25, 0.50, 1.00)
MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}


def unified_columns(frame):
    return [
        column for column in frame.columns
        if column not in MATCHUP_HTE
        and not (column.startswith("tm_") and column.endswith("_std"))
    ]


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form = add_stable_form_features(hierarchical)

    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    baseline_context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    baseline_context = baseline_context["no_matchup_hte"]["context"]

    predictions = {name: {} for name in CONFIGS}
    feature_counts = {}
    for year in YEARS:
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        features = select_v2_features(add_row_features(form, prior))
        features = add_trackman_features(features, trackman)
        features = add_context_trackman_features(features, context_trackman)
        columns = unified_columns(features)
        candidate = features[columns]
        feature_counts[str(year)] = int(len(columns))

        for name, config in CONFIGS.items():
            model, model_columns = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{key}": value
                for key, value in config.items()
            })
            model.fit(candidate.loc[train_mask, model_columns], y.loc[train_mask])
            predictions[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, model_columns]
            )[:, 1]
        print(f"year={year} unified HGB models done", flush=True)

    validation_frame = make_validation_frame()
    candidate_predictions = {}
    development = {}
    for config_name, unified in predictions.items():
        for ratio in REPLACEMENT_RATIOS:
            candidate_name = f"{config_name}_replace_{ratio:.2f}"
            blended_form = {
                key: (1.0 - ratio) * baseline_form[key] + ratio * unified[key]
                for key in baseline_form
            }
            blended_context = {
                key: (1.0 - ratio) * baseline_context[key] + ratio * unified[key]
                for key in baseline_context
            }
            parts = []
            for year in YEARS:
                prediction, _, _, _ = calibrated_prediction(
                    year, oof, logistic, blended_form, blended_context, raw_frame
                )
                parts.append(prediction)
            candidate_predictions[candidate_name] = np.concatenate(parts)
            development[candidate_name] = development_metrics(
                validation_frame, candidate_predictions[candidate_name]
            )

    eligible = [
        name for name, result in development.items()
        if result["season_gain_development"]["2022"] >= -1e-5
        and result["season_gain_development"]["2023"] >= -1e-5
        and result["gain_2024_mar_aug"] > 0
        and result["gain_2024_jul_aug"] > 0
        and result["monthly_block_win_rate"] >= 0.75
    ]
    promoted = max(
        eligible,
        key=lambda name: (
            development[name]["gain_2024_mar_aug"],
            development[name]["gain_2024_jul_aug"],
        ),
        default=None,
    )
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {
            "candidate": promoted,
            "comparison": compare_candidate(
                validation_frame, candidate_predictions[promoted]
            ),
        }

    output = {
        "experiment": "V90_server_compatible_unified_hgb",
        "baseline": "V41",
        "server_compatibility": {
            "python": "3.11.15",
            "numpy": "1.26.4",
            "pandas": "2.3.3",
            "scikit_learn": "1.8.0",
            "external_model_package_required": False,
            "catboost_excluded": "not installed on local or documented server",
            "lightgbm_excluded": "not part of documented server base packages",
        },
        "configs": CONFIGS,
        "replacement_ratios": REPLACEMENT_RATIOS,
        "feature_counts": feature_counts,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "current_pitch_post_event_information_used": False,
            "prior_season_and_asof_features_only": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(candidate_predictions, PREDICTIONS, compress=3)
    print(json.dumps({
        "feature_counts": feature_counts,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation": final_confirmation,
    }, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
