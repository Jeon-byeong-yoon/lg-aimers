"""V89: recency sample weights for the V41 Form HGB layer.

Target encodings and as-of histories are unchanged. Only training-row loss
weights vary. Candidate selection excludes 2024 September-October; that window
is evaluated once for the single promoted candidate.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import BASE_CONFIG, v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS


OUTPUT = Path("artifacts/v89_recency_sample_weight_metrics.json")
PREDICTIONS = Path("artifacts/v89_recency_sample_weight_predictions.joblib")
FORM_CONFIG = {**BASE_CONFIG, "learning_rate": 0.03, "max_iter": 500}
DECAYS = {
    "unweighted_control": 1.0,
    "mild_decay_090": 0.90,
    "moderate_decay_075": 0.75,
}


def recency_weights(seasons, valid_year, decay):
    if decay == 1.0:
        return np.ones(len(seasons), dtype=float)
    age = (valid_year - 1 - seasons.astype(int)).clip(lower=0)
    weight = np.power(decay, age.to_numpy(dtype=float))
    return weight / weight.mean()


def development_metrics(frame, candidate_prediction):
    work = frame.copy()
    work["candidate"] = np.clip(np.asarray(candidate_prediction), 0, 1)
    work["gain"] = (
        (work["v41_prediction"] - work["target"]) ** 2
        - (work["candidate"] - work["target"]) ** 2
    )
    development = work[work["validation_role"] == "development"]
    monthly = development.groupby(["season", "game_month"], observed=True)["gain"].mean()
    season_gain = {
        str(year): float(development.loc[development["season"] == year, "gain"].mean())
        for year in YEARS
    }
    mask_2024_dev = (development["season"] == 2024)
    mask_2024_jul_aug = mask_2024_dev & (development["game_month"] >= 7)
    return {
        "season_gain_development": season_gain,
        "gain_2024_mar_aug": float(development.loc[mask_2024_dev, "gain"].mean()),
        "gain_2024_jul_aug": float(development.loc[mask_2024_jul_aug, "gain"].mean()),
        "monthly_blocks_won": int((monthly > 0).sum()),
        "monthly_blocks_total": int(len(monthly)),
        "monthly_block_win_rate": float((monthly > 0).mean()),
        "worst_monthly_gain": float(monthly.min()),
    }


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]

    form_raw = add_stable_form_features(hierarchical)
    form_predictions = {name: {} for name in DECAYS}
    weight_summaries = {name: {} for name in DECAYS}
    for year in YEARS:
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        columns = v31_form_columns(form_features)
        candidate = form_features[columns]

        for name, decay in DECAYS.items():
            if name == "unweighted_control":
                form_predictions[name][str(year)] = baseline_form[str(year)]
                weight_summaries[name][str(year)] = {"min": 1.0, "max": 1.0, "mean": 1.0}
                continue
            model, model_columns = hist_gbdt_pipeline(candidate)
            model.set_params(**{
                f"histgradientboostingclassifier__{key}": value
                for key, value in FORM_CONFIG.items()
            })
            weight = recency_weights(data.loc[train_mask, "season"], year, decay)
            model.fit(
                candidate.loc[train_mask, model_columns],
                y.loc[train_mask],
                histgradientboostingclassifier__sample_weight=weight,
            )
            form_predictions[name][str(year)] = model.predict_proba(
                candidate.loc[valid_mask, model_columns]
            )[:, 1]
            weight_summaries[name][str(year)] = {
                "min": float(weight.min()),
                "max": float(weight.max()),
                "mean": float(weight.mean()),
            }
        print(f"year={year} weighted Form models done", flush=True)

    validation_frame = make_validation_frame()
    candidate_predictions = {}
    development = {}
    for name, form in form_predictions.items():
        parts = []
        for year in YEARS:
            prediction, _, _, _ = calibrated_prediction(
                year, oof, logistic, form, context, raw_frame
            )
            parts.append(prediction)
        candidate_predictions[name] = np.concatenate(parts)
        development[name] = development_metrics(
            validation_frame, candidate_predictions[name]
        )

    eligible = [
        name for name in DECAYS if name != "unweighted_control"
        and development[name]["season_gain_development"]["2022"] >= -1e-5
        and development[name]["season_gain_development"]["2023"] >= -1e-5
        and development[name]["gain_2024_mar_aug"] > 0
        and development[name]["gain_2024_jul_aug"] > 0
        and development[name]["monthly_block_win_rate"] >= 0.75
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
        "experiment": "V89_recency_sample_weight_form_hgb",
        "baseline": "V41",
        "form_config": FORM_CONFIG,
        "decays": DECAYS,
        "weight_summaries": weight_summaries,
        "development_results": development,
        "promotion_rule": {
            "2022_and_2023_gain_min": -1e-5,
            "2024_mar_aug_gain_positive": True,
            "2024_jul_aug_gain_positive": True,
            "development_monthly_win_rate_min": 0.75,
        },
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "target_encoding_changed": False,
            "only_training_loss_weights_changed": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(
        {"row_index": validation_frame.index.to_numpy(), "predictions": candidate_predictions},
        PREDICTIONS,
        compress=3,
    )
    print(json.dumps({
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation": final_confirmation,
    }, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
