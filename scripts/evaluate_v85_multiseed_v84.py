"""V85: V84 0.2/0.2/0.6 신호 가중치를 고정한 HGB 3시드 안정성 검증."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
from asof_features_v79 import add_asof_reliability_features, learn_asof_priors
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v78_bayesian_form_delta import BASELINE, FORM_CONFIG, YEARS, evaluate, form_columns
from evaluate_v80_failure_risk_experts import EXPERT_CONFIG, add_expert_deltas
from evaluate_v83_stable_final_ensemble import block_results, pitcher_cluster_bootstrap
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


SEEDS = (42, 1042, 2042)


def train_seed_models(features, train_mask, valid_mask, y, config, seeds):
    predictions = {}
    columns = form_columns(features)
    for seed in seeds:
        model, model_columns = hist_gbdt_pipeline(features[columns])
        params = {**config, "random_state": seed}
        model.set_params(**{
            f"histgradientboostingclassifier__{name}": value
            for name, value in params.items()
        })
        model.fit(features.loc[train_mask, model_columns], y.loc[train_mask])
        predictions[seed] = model.predict_proba(features.loc[valid_mask, model_columns])[:, 1]
    return predictions


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = raw.pop("control_success").astype("uint8")
    data = raw.drop(columns="row_id")
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    baseline_form = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    _, baseline = evaluate(baseline_form, oof, logistic, context, data)

    raw_predictions = {
        seed: {"rates": {}, "hierarchy": {}, "middle": {}} for seed in SEEDS
    }
    for year in YEARS:
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        target_prior = float(y.loc[train_mask].mean())
        priors = learn_asof_priors(data.loc[train_mask])

        for variant in ("rates", "hierarchy"):
            asof = add_asof_reliability_features(hierarchical, priors, variant)
            form_raw = add_stable_form_features(asof)
            features = select_v2_features(add_row_features(form_raw, target_prior))
            features = add_trackman_features(features, trackman)
            seeded = train_seed_models(
                features, train_mask, valid_mask, y, FORM_CONFIG, SEEDS
            )
            for seed in SEEDS:
                raw_predictions[seed][variant][str(year)] = seeded[seed]
            del asof, form_raw, features, seeded

        middle_features = add_expert_deltas(data, "middle")
        middle_columns = list(middle_features.columns)
        middle_seeded = {}
        for seed in SEEDS:
            model, columns = hist_gbdt_pipeline(middle_features[middle_columns])
            params = {**EXPERT_CONFIG, "random_state": seed}
            model.set_params(**{
                f"histgradientboostingclassifier__{name}": value
                for name, value in params.items()
            })
            model.fit(middle_features.loc[train_mask, columns], y.loc[train_mask])
            middle_seeded[seed] = model.predict_proba(
                middle_features.loc[valid_mask, columns]
            )[:, 1]
        for seed in SEEDS:
            raw_predictions[seed]["middle"][str(year)] = middle_seeded[seed]
        del middle_features, middle_seeded
        print(f"year={year} done", flush=True)

    seed_final = {}
    for seed in SEEDS:
        _, rates = evaluate(raw_predictions[seed]["rates"], oof, logistic, context, data)
        _, hierarchy = evaluate(raw_predictions[seed]["hierarchy"], oof, logistic, context, data)
        middle_form = {
            str(year): 0.99 * baseline_form[str(year)] + 0.01 * raw_predictions[seed]["middle"][str(year)]
            for year in YEARS
        }
        _, middle = evaluate(middle_form, oof, logistic, context, data)
        seed_final[seed] = {
            str(year): np.clip(
                baseline[str(year)]
                + 0.2 * (rates[str(year)] - baseline[str(year)])
                + 0.2 * (hierarchy[str(year)] - baseline[str(year)])
                + 0.6 * (middle[str(year)] - baseline[str(year)]),
                0, 1,
            )
            for year in YEARS
        }

    candidates = {f"seed_{seed}": seed_final[seed] for seed in SEEDS}
    candidates["seed_mean"] = {
        str(year): np.mean([seed_final[seed][str(year)] for seed in SEEDS], axis=0)
        for year in YEARS
    }
    results = []
    for index, (name, prediction) in enumerate(candidates.items()):
        scores = {
            str(year): float(brier_score_loss(oof[str(year)]["target"], prediction[str(year)]))
            for year in YEARS
        }
        gains = {str(year): BASELINE[year] - scores[str(year)] for year in YEARS}
        blocks = block_results(prediction, baseline, oof, data)
        wins = sum(row["gain"] > 0 for row in blocks)
        confidence = pitcher_cluster_bootstrap(
            prediction, baseline, oof, data, seed=8500 + index, iterations=1000
        )
        results.append({
            "candidate": name, "scores": scores, "gains_vs_v41": gains,
            "block_wins": wins, "block_count": len(blocks),
            "block_win_rate": wins / len(blocks), "blocks": blocks,
            "pitcher_cluster_bootstrap_95_ci": confidence,
            "all_seasons_improved": all(gains[str(year)] > 0 for year in YEARS),
            "submit_ready": (
                all(gains[str(year)] > 0 for year in YEARS)
                and wins / len(blocks) >= 0.70
            ),
        })
    results.sort(key=lambda row: (
        row["all_seasons_improved"], row["block_win_rate"],
        min(row["gains_vs_v41"].values()), row["gains_vs_v41"]["2024"],
    ), reverse=True)
    ready = [row for row in results if row["submit_ready"]]
    output = {
        "experiment": "V85_multiseed_v84",
        "fixed_signal_weights": {"rates": 0.2, "hierarchy": 0.2, "middle": 0.6},
        "seeds": list(SEEDS), "candidate_count": len(results),
        "submit_ready_count": len(ready), "best": results[0],
        "best_submit_ready": ready[0] if ready else None, "results": results,
    }
    Path("artifacts/v85_multiseed_v84_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    joblib.dump(raw_predictions, "artifacts/v85_multiseed_component_predictions.joblib", compress=3)
    print(json.dumps({k: output[k] for k in ("candidate_count", "submit_ready_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
