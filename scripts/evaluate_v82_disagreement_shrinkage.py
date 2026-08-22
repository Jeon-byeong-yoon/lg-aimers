"""V82: 과거 OOF에서 고정한 모델 불일치 임계값으로만 국소 확률 수축."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS
from evaluate_v81_dynamic_expert_weights import calibrate, raw_components


QUANTILES = (0.80, 0.90, 0.95)
SHRINKAGES = (0.005, 0.01, 0.02)


def disagreement(components, year):
    key = str(year)
    return np.column_stack([
        components[key]["v17"], components[key]["form"], components[key]["context"]
    ]).std(axis=1)


def block_results(candidate, baseline, oof, data):
    rows = []
    for year in (2023, 2024):
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
    y_all = raw.pop("control_success").astype(float)
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
    _, baseline = calibrate(baseline_raw, oof, data)
    disagreements = {str(year): disagreement(components, year) for year in YEARS}

    results = []
    for quantile in QUANTILES:
        for shrinkage in SHRINKAGES:
            predictions = {"2022": baseline["2022"].copy()}
            thresholds = {}
            affected = {}
            for year in (2023, 2024):
                key = str(year)
                history_years = [history_year for history_year in YEARS if history_year < year]
                history_disagreement = np.concatenate([
                    disagreements[str(history_year)] for history_year in history_years
                ])
                threshold = float(np.quantile(history_disagreement, quantile))
                upper = float(np.quantile(history_disagreement, 0.99))
                history_index = np.concatenate([
                    oof[str(history_year)]["row_index"] for history_year in history_years
                ])
                prior = float(y_all.loc[history_index].mean())
                gate = np.clip(
                    (disagreements[key] - threshold) / max(upper - threshold, 1e-12), 0, 1
                )
                predictions[key] = np.clip(
                    baseline[key] + shrinkage * gate * (prior - baseline[key]), 0, 1
                )
                thresholds[key] = {"threshold": threshold, "upper_99": upper, "prior": prior}
                affected[key] = {
                    "positive_gate_rate": float((gate > 0).mean()),
                    "mean_gate": float(gate.mean()),
                }
            scores = {
                str(year): float(brier_score_loss(oof[str(year)]["target"], predictions[str(year)]))
                for year in YEARS
            }
            gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
            blocks = block_results(predictions, baseline, oof, data)
            wins = sum(row["gain"] > 0 for row in blocks)
            results.append({
                "quantile": quantile, "shrinkage": shrinkage,
                "thresholds_from_past_oof": thresholds, "affected": affected,
                "scores": scores, "gains_vs_v41": gains,
                "active_block_wins": wins, "active_block_count": len(blocks),
                "active_block_win_rate": wins / len(blocks), "blocks": blocks,
                "no_season_degraded": all(gains[str(year)] >= 0 for year in YEARS),
                "submit_ready": (
                    all(gains[str(year)] >= 0 for year in YEARS)
                    and gains["2024"] >= 5e-5
                    and wins / len(blocks) >= 0.70
                ),
            })
    results.sort(key=lambda row: (row["gains_vs_v41"]["2024"], row["active_block_win_rate"]), reverse=True)
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V82_disagreement_shrinkage",
        "note": "2022 is unchanged because no earlier V41 OOF distribution exists.",
        "compliance": {
            "thresholds_and_priors_fit_on_past_oof_only": True,
            "validation_or_test_distribution_used_for_thresholds": False,
            "test_row_aggregation_used": False,
        },
        "candidate_count": len(results), "submit_ready_count": len(ready),
        "best": results[0], "best_submit_ready": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v82_disagreement_shrinkage_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
