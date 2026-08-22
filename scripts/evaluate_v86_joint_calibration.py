"""V86: V85 신호를 원시 예측에서 먼저 결합한 후 전용 보정 재구축."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from evaluate_v78_bayesian_form_delta import BASELINE, YEARS, evaluate
from evaluate_v81_dynamic_expert_weights import calibrate, raw_components
from evaluate_v83_stable_final_ensemble import block_results, pitcher_cluster_bootstrap


SEEDS = (42, 1042, 2042)


def raw_signal(component, form_prediction):
    return {
        str(year): (
            0.55 * component[str(year)]["v17"]
            + 0.32 * form_prediction[str(year)]
            + 0.13 * component[str(year)]["context"]
        )
        for year in YEARS
    }


def combine_raw(seed_components, seed, baseline_form, component):
    baseline_raw = raw_signal(component, baseline_form)
    rates_raw = raw_signal(component, seed_components[seed]["rates"])
    hierarchy_raw = raw_signal(component, seed_components[seed]["hierarchy"])
    middle_form = {
        str(year): (
            0.99 * baseline_form[str(year)]
            + 0.01 * seed_components[seed]["middle"][str(year)]
        )
        for year in YEARS
    }
    middle_raw = raw_signal(component, middle_form)
    return {
        str(year): (
            baseline_raw[str(year)]
            + 0.2 * (rates_raw[str(year)] - baseline_raw[str(year)])
            + 0.2 * (hierarchy_raw[str(year)] - baseline_raw[str(year)])
            + 0.6 * (middle_raw[str(year)] - baseline_raw[str(year)])
        )
        for year in YEARS
    }


def late_blend(seed_components, seed, baseline_form, oof, logistic, context, data, baseline):
    _, rates = evaluate(seed_components[seed]["rates"], oof, logistic, context, data)
    _, hierarchy = evaluate(seed_components[seed]["hierarchy"], oof, logistic, context, data)
    middle_form = {
        str(year): 0.99 * baseline_form[str(year)] + 0.01 * seed_components[seed]["middle"][str(year)]
        for year in YEARS
    }
    _, middle = evaluate(middle_form, oof, logistic, context, data)
    return {
        str(year): (
            baseline[str(year)]
            + 0.2 * (rates[str(year)] - baseline[str(year)])
            + 0.2 * (hierarchy[str(year)] - baseline[str(year)])
            + 0.6 * (middle[str(year)] - baseline[str(year)])
        )
        for year in YEARS
    }


def assess(name, prediction, baseline, oof, data, seed):
    scores = {
        str(year): float(brier_score_loss(oof[str(year)]["target"], prediction[str(year)]))
        for year in YEARS
    }
    gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
    blocks = block_results(prediction, baseline, oof, data)
    wins = sum(row["gain"] > 0 for row in blocks)
    confidence = pitcher_cluster_bootstrap(
        prediction, baseline, oof, data, seed=seed, iterations=1000
    )
    return {
        "candidate": name, "scores": scores, "gains_vs_v41": gains,
        "block_wins": wins, "block_count": len(blocks),
        "block_win_rate": wins / len(blocks), "blocks": blocks,
        "pitcher_cluster_bootstrap_95_ci": confidence,
        "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
        "submission_candidate": (
            all(gains[str(year)] > 0 for year in YEARS)
            and wins / len(blocks) >= 0.70
        ),
    }


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw.pop("control_success")
    data = raw.drop(columns="row_id")
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    seed_components = joblib.load("artifacts/v85_multiseed_component_predictions.joblib")
    component = raw_components(oof, logistic, baseline_form, context)
    baseline_raw = raw_signal(component, baseline_form)
    _, baseline = calibrate(baseline_raw, oof, data)

    joint_by_seed, late_by_seed = {}, {}
    for seed in SEEDS:
        ensemble_raw = combine_raw(seed_components, seed, baseline_form, component)
        _, joint_by_seed[seed] = calibrate(ensemble_raw, oof, data)
        late_by_seed[seed] = late_blend(
            seed_components, seed, baseline_form, oof, logistic, context, data, baseline
        )
    joint_mean_raw = {
        str(year): np.mean([
            combine_raw(seed_components, seed, baseline_form, component)[str(year)]
            for seed in SEEDS
        ], axis=0)
        for year in YEARS
    }
    _, joint_mean = calibrate(joint_mean_raw, oof, data)
    late_mean = {
        str(year): np.mean([late_by_seed[seed][str(year)] for seed in SEEDS], axis=0)
        for year in YEARS
    }

    candidates = {
        "late_blend_seed_2042": late_by_seed[2042],
        "joint_calibration_seed_2042": joint_by_seed[2042],
        "late_blend_seed_mean": late_mean,
        "joint_calibration_seed_mean": joint_mean,
    }
    results = [
        assess(name, prediction, baseline, oof, data, seed=8600 + index)
        for index, (name, prediction) in enumerate(candidates.items())
    ]
    results.sort(key=lambda row: (
        row["submission_candidate"], row["block_win_rate"],
        min(row["gains_vs_v41"].values()), row["gains_vs_v41"]["2024"],
    ), reverse=True)
    ready = [row for row in results if row["submission_candidate"]]
    output = {
        "experiment": "V86_joint_calibration",
        "fixed_calibration": {
            "global_shift": True, "count_min_samples": 500,
            "pitcher_count_min_samples": 300,
            "count_weight": 0.75, "pitcher_count_weight": 0.25,
        },
        "candidate_count": len(results), "submission_candidate_count": len(ready),
        "best": results[0], "best_submission_candidate": ready[0] if ready else None,
        "results": results,
    }
    Path("artifacts/v86_joint_calibration_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"joint_seed_2042": joint_by_seed[2042], "joint_seed_mean": joint_mean},
        "artifacts/v86_joint_calibrated_predictions.joblib", compress=3,
    )
    print(json.dumps({k: output[k] for k in (
        "candidate_count", "submission_candidate_count", "best"
    )}, indent=2))


if __name__ == "__main__":
    main()
