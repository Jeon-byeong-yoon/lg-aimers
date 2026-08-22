"""V83: V78–V82 중 서로 다른 시즌 역할을 보인 신호의 제한 균등 결합."""

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS, evaluate


SIGNALS = ("rates_2024", "hierarchy_2022", "middle_2023")


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


def blend_predictions(names, scale, signals, baseline):
    output = {}
    for year in YEARS:
        key = str(year)
        equal_signal = np.mean([signals[name][key] for name in names], axis=0)
        output[key] = np.clip(baseline[key] + scale * (equal_signal - baseline[key]), 0, 1)
    return output


def pitcher_cluster_bootstrap(candidate, baseline, oof, data, seed, iterations=1000):
    """투수를 군집으로 샘플링한 paired Brier gain 95% 신뢰구간."""
    rng = np.random.default_rng(seed)
    intervals = {}
    for year in YEARS:
        key = str(year)
        index = oof[key]["row_index"]
        target = np.asarray(oof[key]["target"], dtype=float)
        delta = (baseline[key] - target) ** 2 - (candidate[key] - target) ** 2
        grouped = pd.DataFrame({
            "pitcher_id": data.loc[index, "pitcher_id"].to_numpy(),
            "delta": delta,
        }).groupby("pitcher_id", observed=True)["delta"].agg(["sum", "count"])
        sums = grouped["sum"].to_numpy()
        counts = grouped["count"].to_numpy()
        sampled = rng.integers(0, len(grouped), size=(iterations, len(grouped)))
        boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        intervals[key] = {
            "lower": float(np.quantile(boot, 0.025)),
            "median": float(np.quantile(boot, 0.5)),
            "upper": float(np.quantile(boot, 0.975)),
            "pitcher_clusters": int(len(grouped)),
            "iterations": iterations,
        }
    return intervals


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw.pop("control_success")
    data = raw.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
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
    signals = {
        "rates_2024": rates,
        "hierarchy_2022": hierarchy,
        "middle_2023": middle,
    }

    # 실험 결과를 보고 연속 가중치를 튜닝하지 않는다.
    recipes = []
    for size in (2, 3):
        for names in itertools.combinations(SIGNALS, size):
            for scale in (0.5, 1.0):
                recipes.append((names, scale))

    results = []
    for recipe_index, (names, scale) in enumerate(recipes):
        prediction = blend_predictions(names, scale, signals, baseline)
        scores = {
            str(year): float(brier_score_loss(oof[str(year)]["target"], prediction[str(year)]))
            for year in YEARS
        }
        gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
        blocks = block_results(prediction, baseline, oof, data)
        wins = sum(row["gain"] > 0 for row in blocks)
        confidence = pitcher_cluster_bootstrap(
            prediction, baseline, oof, data, seed=8300 + recipe_index
        )
        all_ci_positive = all(confidence[str(year)]["lower"] > 0 for year in YEARS)
        positive_gains = [gain for gain in gains.values() if gain > 0]
        negative_gains = [gain for gain in gains.values() if gain < 0]
        results.append({
            "signals": list(names), "scale_toward_signal_mean": scale,
            "scores": scores, "gains_vs_v41": gains,
            "mean_gain": float(np.mean(list(gains.values()))),
            "minimum_season_gain": float(min(gains.values())),
            "positive_to_negative_gain_ratio": (
                float(sum(positive_gains) / abs(sum(negative_gains))) if negative_gains else None
            ),
            "block_wins": wins, "block_count": len(blocks),
            "block_win_rate": wins / len(blocks), "blocks": blocks,
            "pitcher_cluster_bootstrap_95_ci": confidence,
            "all_bootstrap_lower_bounds_positive": all_ci_positive,
            "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
            "submit_ready": (
                all(gains[str(year)] > 0 for year in YEARS)
                and gains["2024"] >= 5e-5
                and wins / len(blocks) >= 0.70
                and all_ci_positive
            ),
        })
    results.sort(
        key=lambda row: (
            row["all_seasons_improved"], row["minimum_season_gain"],
            row["block_win_rate"], row["mean_gain"],
        ), reverse=True,
    )
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V83_stable_final_ensemble",
        "selection_policy": {
            "signals": {
                "rates_2024": "V79 rates; strongest 2024 gain",
                "hierarchy_2022": "V79 hierarchy; strongest 2022 gain",
                "middle_2023": "V80 middle at 0.01; strongest stable 2023 block pattern",
            },
            "equal_weights_only": True,
            "scales": [0.5, 1.0],
            "continuous_weight_optimization": False,
        },
        "compliance": {
            "official_data_only": True,
            "test_row_aggregation_used": False,
            "post_pitch_information_used": False,
        },
        "candidate_count": len(results), "submit_ready_count": len(ready),
        "best": results[0], "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v83_stable_final_ensemble_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
