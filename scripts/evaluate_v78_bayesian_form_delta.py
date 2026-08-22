"""V78: 공식 as-of 폼 변화량을 투수 이력 표본 수로 축소한 Form HGB 검증."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


YEARS = (2022, 2023, 2024)
BASELINE = {2022: 0.24338879436541092, 2023: 0.25315389703549307, 2024: 0.24782554051940017}
SHRINK_STRENGTHS = (50.0, 200.0, 500.0, 1000.0)
FORM_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.03,
    "max_iter": 500,
}


def add_shrunk_deltas(frame, strength):
    output = add_stable_form_features(frame)
    n = output["asof_pitcher_n"].fillna(0).clip(lower=0)
    reliability = n / (n + strength)
    success_long = output["asof_pitcher_success_rate"]
    middle_long = output["asof_pitcher_middle_rate"]
    for window in (1, 3, 5):
        success = output[f"asof_pitcher_prev{window}_game_success_rate"]
        middle = output[f"asof_pitcher_prev{window}_game_middle_rate"]
        output[f"v78_success_delta_{window}"] = (success - success_long) * reliability
        output[f"v78_middle_delta_{window}"] = (middle - middle_long) * reliability
    output["v78_success_trend_1_5"] = (
        output["asof_pitcher_prev1_game_success_rate"]
        - output["asof_pitcher_prev5_game_success_rate"]
    ) * reliability
    output["v78_middle_trend_1_5"] = (
        output["asof_pitcher_prev1_game_middle_rate"]
        - output["asof_pitcher_prev5_game_middle_rate"]
    ) * reliability
    output["v78_reliability"] = reliability
    return output


def form_columns(frame):
    return [column for column in frame if not (column.startswith("tm_") and column.endswith("_std"))]


def evaluate(form_prediction, oof, logistic, context, raw_frame):
    scores, predictions, block_gains = {}, {}, []
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        raw = 0.55 * v17 + 0.32 * form_prediction[key] + 0.13 * context[key]
        if year == 2022:
            prediction = np.clip(raw, 0, 1)
        else:
            indices, targets, history_predictions = [], [], []
            for history_year in YEARS:
                if history_year >= year:
                    continue
                history_key = str(history_year)
                history_v17 = 0.95 * v11_prediction(oof[history_key]) + 0.05 * logistic[history_key]
                history_raw = (
                    0.55 * history_v17
                    + 0.32 * form_prediction[history_key]
                    + 0.13 * context[history_key]
                )
                indices.append(oof[history_key]["row_index"])
                targets.append(oof[history_key]["target"].astype(float))
                history_predictions.append(history_raw)
            index = np.concatenate(indices)
            residual = np.concatenate(targets) - np.concatenate(history_predictions)
            train_frame = raw_frame.loc[index]
            valid_frame = raw_frame.loc[oof[key]["row_index"]]
            count = segment_correction(
                train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
            )
            pitcher_count = segment_correction(
                train_frame, residual, valid_frame,
                ["pitcher_id", "balls_before", "strikes_before"], 300,
            )
            prediction = np.clip(raw + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)
        target = np.asarray(oof[key]["target"], dtype=float)
        scores[key] = float(brier_score_loss(target, prediction))
        predictions[key] = prediction
    return scores, predictions


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

    candidate_predictions = {str(strength): {} for strength in SHRINK_STRENGTHS}
    for strength in SHRINK_STRENGTHS:
        form_raw = add_shrunk_deltas(hierarchical, strength)
        # add_row_features의 prior 의존 컬럼은 폴드별로 재생성한다.
        for year in YEARS:
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            features = select_v2_features(add_row_features(form_raw, prior))
            features = add_trackman_features(features, trackman)
            columns = form_columns(features)
            model, columns = hist_gbdt_pipeline(features[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{name}": value
                for name, value in FORM_CONFIG.items()
            })
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            candidate_predictions[str(strength)][str(year)] = model.predict_proba(
                features.loc[valid_mask, columns]
            )[:, 1]
            del features, model
        print(f"strength={strength} done", flush=True)

    results = []
    for strength in SHRINK_STRENGTHS:
        key = str(strength)
        scores, predictions = evaluate(candidate_predictions[key], oof, logistic, context, data)
        gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
        blocks = []
        for year in YEARS:
            year_key = str(year)
            indices = oof[year_key]["row_index"]
            months = data.loc[indices, "game_month"].to_numpy()
            target = np.asarray(oof[year_key]["target"], dtype=float)
            candidate_error = (predictions[year_key] - target) ** 2
            baseline_error = (baseline_predictions[year_key] - target) ** 2
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
        block_wins = sum(block["gain"] > 0 for block in blocks)
        results.append({
            "shrink_strength": strength,
            "scores": scores,
            "gains_vs_v41": gains,
            "block_wins": block_wins,
            "block_count": len(blocks),
            "block_win_rate": block_wins / len(blocks),
            "blocks": blocks,
            "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
            "submit_ready": (
                all(gains[str(year)] > 0 for year in YEARS)
                and gains["2024"] >= 5e-5
                and block_wins / len(blocks) >= 0.70
            ),
        })
    results.sort(key=lambda row: (row["gains_vs_v41"]["2024"], row["block_win_rate"]), reverse=True)
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V78_bayesian_form_delta",
        "compliance": {
            "official_asof_features_only": True,
            "test_row_aggregation_used": False,
            "post_pitch_information_used": False,
        },
        "baseline": {str(k): v for k, v in BASELINE.items()},
        "form_config": FORM_CONFIG,
        "candidate_count": len(results),
        "submit_ready_count": len(ready),
        "best": results[0],
        "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v78_bayesian_form_delta_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(candidate_predictions, "artifacts/v78_bayesian_form_delta_predictions.joblib", compress=3)
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
