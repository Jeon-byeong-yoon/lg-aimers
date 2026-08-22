"""V80: 공식 As-of 제구 실패 경향별 규제 HGB 전문 모델 검증."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS, evaluate


EXPERT_FEATURES = {
    "middle": [
        "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
        "batter_hand", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n", "asof_batter_middle_rate",
    ],
    "reverse": [
        "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
        "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_batter_n",
        "asof_batter_success_rate",
    ],
    "ball": [
        "season", "game_month", "inning", "balls_before", "strikes_before",
        "outs_before", "base_state", "li", "pitcher_hand", "batter_hand",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_batter_n", "asof_batter_success_rate",
    ],
}
EXPERT_CONFIG = {
    "max_leaf_nodes": 7,
    "min_samples_leaf": 500,
    "l2_regularization": 30.0,
    "learning_rate": 0.03,
    "max_iter": 300,
}
WEIGHTS = (0.01, 0.025, 0.05, 0.075, 0.10)


def add_expert_deltas(frame, expert):
    output = frame[EXPERT_FEATURES[expert]].copy()
    n = output["asof_pitcher_n"].fillna(0).clip(lower=0)
    reliability = n / (n + 500.0)
    if expert == "middle":
        recent = (
            0.6 * output["asof_pitcher_prev3_game_middle_rate"]
            + 0.4 * output["asof_pitcher_prev5_game_middle_rate"]
        )
        output["v80_middle_delta"] = (
            recent - output["asof_pitcher_middle_rate"]
        ) * reliability
    elif expert == "reverse":
        output["v80_reverse_reliable"] = output["asof_pitcher_reverse_rate"] * reliability
    else:
        output["v80_ball_minus_strike"] = (
            output["asof_pitcher_ball_rate"] - output["asof_pitcher_strike_rate"]
        ) * reliability
    output["v80_reliability"] = reliability
    return output


def block_metrics(candidate, baseline, oof, data):
    blocks = []
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
            blocks.append({
                "season": year, "block": label, "n": int(mask.sum()),
                "gain": float(baseline_error[mask].mean() - candidate_error[mask].mean()),
            })
    return blocks


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = raw.pop("control_success").astype("uint8")
    data = raw.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    _, baseline_predictions = evaluate(baseline_form, oof, logistic, context, data)

    experts = {expert: {} for expert in EXPERT_FEATURES}
    for year in YEARS:
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        for expert in EXPERT_FEATURES:
            features = add_expert_deltas(data, expert)
            model, columns = hist_gbdt_pipeline(features)
            model.set_params(**{
                f"histgradientboostingclassifier__{name}": value
                for name, value in EXPERT_CONFIG.items()
            })
            model.fit(features.loc[train_mask, columns], y.loc[train_mask])
            experts[expert][str(year)] = model.predict_proba(
                features.loc[valid_mask, columns]
            )[:, 1]
            del features, model
        print(f"year={year} done", flush=True)

    expert_sets = {**experts}
    expert_sets["mean"] = {
        str(year): np.mean([experts[name][str(year)] for name in EXPERT_FEATURES], axis=0)
        for year in YEARS
    }
    results = []
    for expert, expert_prediction in expert_sets.items():
        for weight in WEIGHTS:
            blended_form = {
                str(year): (1.0 - weight) * baseline_form[str(year)] + weight * expert_prediction[str(year)]
                for year in YEARS
            }
            scores, predictions = evaluate(blended_form, oof, logistic, context, data)
            gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
            blocks = block_metrics(predictions, baseline_predictions, oof, data)
            wins = sum(block["gain"] > 0 for block in blocks)
            results.append({
                "expert": expert, "weight_within_form": weight,
                "effective_total_weight": 0.32 * weight,
                "scores": scores, "gains_vs_v41": gains,
                "block_wins": wins, "block_count": len(blocks),
                "block_win_rate": wins / len(blocks), "blocks": blocks,
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
        "experiment": "V80_failure_risk_experts",
        "compliance": {
            "official_asof_and_current_row_features_only": True,
            "proxy_targets_invented": False,
            "current_pitch_result_or_location_used": False,
            "test_row_aggregation_used": False,
        },
        "expert_config": EXPERT_CONFIG,
        "candidate_count": len(results),
        "submit_ready_count": len(ready),
        "best": results[0],
        "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v80_failure_risk_experts_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(experts, "artifacts/v80_failure_risk_expert_predictions.joblib", compress=3)
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
