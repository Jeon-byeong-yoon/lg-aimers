"""V84: V83 세 신호의 제한 simplex 가중치 안정 영역 검증."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS, evaluate
from evaluate_v83_stable_final_ensemble import block_results, pitcher_cluster_bootstrap


SCALES = (0.25, 0.50, 0.75, 1.00)


def signal_predictions(oof, logistic, context, baseline_form, data):
    _, baseline = evaluate(baseline_form, oof, logistic, context, data)
    v79 = joblib.load("artifacts/v79_unified_asof_predictions.joblib")
    _, rates = evaluate(v79["rates"], oof, logistic, context, data)
    _, hierarchy = evaluate(v79["hierarchy"], oof, logistic, context, data)
    v80 = joblib.load("artifacts/v80_failure_risk_expert_predictions.joblib")
    middle_form = {
        str(year): 0.99 * baseline_form[str(year)] + 0.01 * v80["middle"][str(year)]
        for year in YEARS
    }
    _, middle = evaluate(middle_form, oof, logistic, context, data)
    return baseline, {"rates": rates, "hierarchy": hierarchy, "middle": middle}


def blend(weights, scale, signals, baseline):
    output = {}
    for year in YEARS:
        key = str(year)
        signal = sum(weights[name] * signals[name][key] for name in weights)
        output[key] = np.clip(baseline[key] + scale * (signal - baseline[key]), 0, 1)
    return output


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw.pop("control_success")
    data = raw.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    baseline, signals = signal_predictions(oof, logistic, context, baseline_form, data)

    recipes = []
    for rates_i in range(1, 9):
        for hierarchy_i in range(1, 9):
            middle_i = 10 - rates_i - hierarchy_i
            if middle_i < 1:
                continue
            weights = {
                "rates": rates_i / 10,
                "hierarchy": hierarchy_i / 10,
                "middle": middle_i / 10,
            }
            for scale in SCALES:
                recipes.append((weights, scale))

    results, prediction_cache = [], {}
    for index, (weights, scale) in enumerate(recipes):
        prediction = blend(weights, scale, signals, baseline)
        prediction_cache[index] = prediction
        scores = {
            str(year): float(brier_score_loss(oof[str(year)]["target"], prediction[str(year)]))
            for year in YEARS
        }
        gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
        blocks = block_results(prediction, baseline, oof, data)
        wins = sum(row["gain"] > 0 for row in blocks)
        results.append({
            "recipe_index": index, "weights": weights, "scale": scale,
            "scores": scores, "gains_vs_v41": gains,
            "minimum_season_gain": float(min(gains.values())),
            "mean_gain": float(np.mean(list(gains.values()))),
            "block_wins": wins, "block_count": len(blocks),
            "block_win_rate": wins / len(blocks), "blocks": blocks,
            "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
        })

    # 같은 scale에서 가중치 L1 거리 0.2인 인접 격자의 안정성.
    for row in results:
        neighbors = [
            other for other in results
            if other["scale"] == row["scale"]
            and sum(abs(other["weights"][name] - row["weights"][name]) for name in row["weights"]) <= 0.2000001
        ]
        stable = [other for other in neighbors if other["all_seasons_improved"]]
        row["neighbor_count"] = len(neighbors)
        row["positive_neighbor_count"] = len(stable)
        row["positive_neighbor_rate"] = len(stable) / len(neighbors)

    results.sort(key=lambda row: (
        row["all_seasons_improved"], row["block_win_rate"],
        row["positive_neighbor_rate"], row["minimum_season_gain"],
        row["gains_vs_v41"]["2024"],
    ), reverse=True)

    # 상위 10개만 투수 군집 bootstrap으로 추가 확인.
    for rank, row in enumerate(results[:10]):
        confidence = pitcher_cluster_bootstrap(
            prediction_cache[row["recipe_index"]], baseline, oof, data,
            seed=8400 + rank, iterations=1000,
        )
        row["pitcher_cluster_bootstrap_95_ci"] = confidence
        row["all_bootstrap_lower_bounds_positive"] = all(
            confidence[str(year)]["lower"] > 0 for year in YEARS
        )
    for row in results[10:]:
        row["pitcher_cluster_bootstrap_95_ci"] = None
        row["all_bootstrap_lower_bounds_positive"] = False

    for row in results:
        row["submit_ready"] = (
            row["all_seasons_improved"]
            and row["block_win_rate"] >= 0.70
            and row["positive_neighbor_rate"] >= 0.70
            and row["gains_vs_v41"]["2024"] >= 5e-6
        )
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V84_robust_signal_weights",
        "policy": {
            "weight_step": 0.1, "minimum_signal_weight": 0.1,
            "scales": list(SCALES), "continuous_optimization": False,
            "neighbor_l1_radius": 0.2,
        },
        "candidate_count": len(results), "submit_ready_count": len(ready),
        "best": results[0], "best_submit_ready": ready[0] if ready else None,
        "top_20": results[:20],
    }
    Path("artifacts/v84_robust_signal_weights_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best", "best_submit_ready")}, indent=2))


if __name__ == "__main__":
    main()
