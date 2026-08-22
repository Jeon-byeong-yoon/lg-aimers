"""V81: 공식 As-of 표본 수로 V17/Form/Context 행별 가중치를 조정."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS


STRENGTHS = (200.0, 500.0, 1000.0)
AMPLITUDES = (0.02, 0.05, 0.08)
GATES = ("form_only", "dual")


def reliability(values, strength):
    values = values.fillna(0).clip(lower=0)
    return values / (values + strength)


def raw_components(oof, logistic, form, context):
    components = {}
    for year in YEARS:
        key = str(year)
        components[key] = {
            "v17": 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key],
            "form": np.asarray(form[key]),
            "context": np.asarray(context[key]),
        }
    return components


def dynamic_raw_predictions(candidate, components, oof, data):
    strength = candidate["strength"]
    amplitude = candidate["amplitude"]
    gate = candidate["gate"]
    predictions, weight_ranges = {}, {}
    for year in YEARS:
        key = str(year)
        history = data.loc[data["season"] < year]
        valid = data.loc[oof[key]["row_index"]]
        pitcher_center = float(reliability(history["asof_pitcher_n"], strength).mean())
        batter_center = float(reliability(history["asof_batter_n"], strength).mean())
        pitcher_signal = reliability(valid["asof_pitcher_n"], strength).to_numpy() - pitcher_center
        batter_signal = reliability(valid["asof_batter_n"], strength).to_numpy() - batter_center
        w_form = 0.32 + amplitude * pitcher_signal
        if gate == "dual":
            w_context = 0.13 + 0.5 * amplitude * batter_signal
        else:
            w_context = np.full(len(valid), 0.13)
        w_v17 = 1.0 - w_form - w_context
        prediction = (
            w_v17 * components[key]["v17"]
            + w_form * components[key]["form"]
            + w_context * components[key]["context"]
        )
        predictions[key] = prediction
        weight_ranges[key] = {
            "v17": [float(w_v17.min()), float(w_v17.max())],
            "form": [float(w_form.min()), float(w_form.max())],
            "context": [float(w_context.min()), float(w_context.max())],
        }
    return predictions, weight_ranges


def calibrate(raw_predictions, oof, data):
    calibrated, scores = {}, {}
    for year in YEARS:
        key = str(year)
        raw = raw_predictions[key]
        if year == 2022:
            prediction = np.clip(raw, 0, 1)
        else:
            history_years = [history_year for history_year in YEARS if history_year < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history_years])
            target = np.concatenate([oof[str(h)]["target"].astype(float) for h in history_years])
            history_prediction = np.concatenate([raw_predictions[str(h)] for h in history_years])
            residual = target - history_prediction
            train_frame = data.loc[index]
            valid_frame = data.loc[oof[key]["row_index"]]
            count = segment_correction(
                train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500
            )
            pitcher_count = segment_correction(
                train_frame, residual, valid_frame,
                ["pitcher_id", "balls_before", "strikes_before"], 300,
            )
            prediction = np.clip(raw + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1)
        calibrated[key] = prediction
        scores[key] = float(brier_score_loss(oof[key]["target"].astype(float), prediction))
    return scores, calibrated


def block_results(candidate, baseline, oof, data):
    rows = []
    for year in YEARS:
        key = str(year)
        index = oof[key]["row_index"]
        months = data.loc[index, "game_month"].to_numpy()
        target = np.asarray(oof[key]["target"], dtype=float)
        candidate_error = (candidate[key] - target) ** 2
        baseline_error = (baseline[key] - target) ** 2
        for label, mask in (
            ("early_3_5", months <= 5),
            ("mid_6_7", (months >= 6) & (months <= 7)),
            ("late_8", months == 8),
            ("finish_9_10", months >= 9),
        ):
            rows.append({
                "season": year, "block": label, "n": int(mask.sum()),
                "gain": float(baseline_error[mask].mean() - candidate_error[mask].mean()),
            })
    return rows


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw.pop("control_success")
    data = raw.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    components = raw_components(oof, logistic, form, context)
    baseline_raw = {
        str(year): (
            0.55 * components[str(year)]["v17"]
            + 0.32 * components[str(year)]["form"]
            + 0.13 * components[str(year)]["context"]
        )
        for year in YEARS
    }
    _, baseline_predictions = calibrate(baseline_raw, oof, data)

    results = []
    for strength in STRENGTHS:
        for amplitude in AMPLITUDES:
            for gate in GATES:
                candidate = {"strength": strength, "amplitude": amplitude, "gate": gate}
                candidate_raw, ranges = dynamic_raw_predictions(candidate, components, oof, data)
                scores, predictions = calibrate(candidate_raw, oof, data)
                gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
                blocks = block_results(predictions, baseline_predictions, oof, data)
                wins = sum(row["gain"] > 0 for row in blocks)
                results.append({
                    **candidate, "scores": scores, "gains_vs_v41": gains,
                    "weight_ranges": ranges, "block_wins": wins,
                    "block_count": len(blocks), "block_win_rate": wins / len(blocks),
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
        "experiment": "V81_dynamic_expert_weights",
        "compliance": {
            "official_asof_counts_only": True,
            "gate_centers_fit_on_past_training_seasons_only": True,
            "test_row_aggregation_used": False,
            "weights_sum_to_one": True,
        },
        "candidate_count": len(results), "submit_ready_count": len(ready),
        "best": results[0], "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v81_dynamic_expert_weights_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
