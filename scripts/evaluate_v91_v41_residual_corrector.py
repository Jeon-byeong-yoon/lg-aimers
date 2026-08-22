"""V91: conservative chronological residual correction on calibrated V41 OOF.

Only previous-season V41 OOF residuals train each fold. Player/team IDs,
season, and month are excluded. Corrections are clipped and weakly applied.
The sealed 2024 September-October window is inspected only for one candidate
that passes development criteria.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, "scripts")
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL


OUTPUT = Path("artifacts/v91_v41_residual_corrector_metrics.json")
PREDICTIONS = Path("artifacts/v91_v41_residual_corrector_predictions.joblib")
CORRECTION_CLIP = 0.02
STRENGTHS = (0.10, 0.25, 0.50)
CONFIGS = {
    "shallow": {
        "max_leaf_nodes": 7,
        "min_samples_leaf": 1000,
        "l2_regularization": 50.0,
        "learning_rate": 0.03,
        "max_iter": 120,
    },
    "very_shallow": {
        "max_leaf_nodes": 5,
        "min_samples_leaf": 2000,
        "l2_regularization": 100.0,
        "learning_rate": 0.025,
        "max_iter": 100,
    },
}
FEATURES = [
    "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def make_regressor(frame, config):
    categorical = [column for column in LOW_CARDINAL_CATEGORICAL if column in frame]
    numeric = [column for column in frame.columns if column not in categorical]
    return make_pipeline(
        ColumnTransformer([
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ]),
        HistGradientBoostingRegressor(
            **config,
            loss="squared_error",
            random_state=42,
        ),
    )


def active_development_metrics(frame, prediction):
    work = frame.copy()
    work["candidate"] = np.clip(prediction, 0, 1)
    work["gain"] = (
        (work["v41_prediction"] - work["target"]) ** 2
        - (work["candidate"] - work["target"]) ** 2
    )
    active = work[
        (work["season"] >= 2023)
        & (work["validation_role"] == "development")
    ]
    monthly = active.groupby(["season", "game_month"], observed=True)["gain"].mean()
    return {
        "active_monthly_blocks_won": int((monthly > 0).sum()),
        "active_monthly_blocks_total": int(len(monthly)),
        "active_monthly_win_rate": float((monthly > 0).mean()),
        "active_worst_monthly_gain": float(monthly.min()),
    }


def main():
    frame = make_validation_frame()
    missing = [column for column in FEATURES if column not in frame]
    if missing:
        raise ValueError(f"Missing residual features: {missing}")
    frame["v41_residual"] = frame["target"] - frame["v41_prediction"]

    corrections = {}
    correction_summaries = {}
    for name, config in CONFIGS.items():
        correction = np.zeros(len(frame), dtype=float)
        correction_summaries[name] = {}
        for valid_year in (2023, 2024):
            train_mask = frame["season"] < valid_year
            valid_mask = frame["season"] == valid_year
            model = make_regressor(frame[FEATURES], config)
            model.fit(
                frame.loc[train_mask, FEATURES],
                frame.loc[train_mask, "v41_residual"],
            )
            raw = model.predict(frame.loc[valid_mask, FEATURES])
            clipped = np.clip(raw, -CORRECTION_CLIP, CORRECTION_CLIP)
            correction[valid_mask.to_numpy()] = clipped
            correction_summaries[name][str(valid_year)] = {
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std()),
                "clipped_fraction": float((np.abs(raw) > CORRECTION_CLIP).mean()),
            }
        corrections[name] = correction
        print(f"{name} residual folds done", flush=True)

    candidate_predictions = {}
    development = {}
    for config_name, correction in corrections.items():
        for strength in STRENGTHS:
            candidate_name = f"{config_name}_strength_{strength:.2f}"
            prediction = np.clip(
                frame["v41_prediction"].to_numpy() + strength * correction,
                0,
                1,
            )
            candidate_predictions[candidate_name] = prediction
            result = development_metrics(frame, prediction)
            result.update(active_development_metrics(frame, prediction))
            development[candidate_name] = result

    eligible = [
        name for name, result in development.items()
        if result["season_gain_development"]["2022"] >= -1e-12
        and result["season_gain_development"]["2023"] > 0
        and result["gain_2024_mar_aug"] > 0
        and result["gain_2024_jul_aug"] > 0
        and result["active_monthly_win_rate"] >= 0.75
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
            "comparison": compare_candidate(frame, candidate_predictions[promoted]),
        }

    output = {
        "experiment": "V91_conservative_v41_oof_residual_corrector",
        "baseline": "V41",
        "configs": CONFIGS,
        "features": FEATURES,
        "excluded_feature_groups": [
            "player IDs", "team IDs", "season", "game_month", "Trackman current pitch",
        ],
        "correction_clip": CORRECTION_CLIP,
        "strengths": STRENGTHS,
        "correction_summaries": correction_summaries,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_train_data_only": True,
            "test_csv_read": False,
            "previous_season_oof_residuals_only": True,
            "player_or_team_id_memorization": False,
            "current_pitch_post_event_information_used": False,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(candidate_predictions, PREDICTIONS, compress=3)
    print(json.dumps({
        "correction_summaries": correction_summaries,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation": final_confirmation,
    }, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
