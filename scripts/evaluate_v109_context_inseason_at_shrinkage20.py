"""V109: last re-validation — Context with the shrinkage-20 reconstruction.

V94a gave the Context HGB the in-season reconstruction and it reached +6.5
development points before failing on the transfer window. Two premises have moved
since: the reconstruction is now shrunk at 20 rather than 50, and Context carries
0.17 of the blend rather than 0.13. This is the last item on the re-validation
list; Form hyperparameters (V97, V107) and drift composition (V100, V108) are both
closed in both directions.

The prior is weak. Context is the smallest component, and V94a suggested the
signal is largely already expressed through Form's 0.64 share. A better Context
would justify more weight, so the context axis is swept alongside.

Promotion now requires a 2024 point estimate of at least +3 points. V108 passed
every earlier criterion on a +1.0 estimate and lost 0.47 on the leaderboard, which
is exactly what a per-candidate noise of +-5 predicts for an estimate that small.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v109_context_inseason_metrics.json")
PREDICTIONS = Path("artifacts/v109_context_inseason_predictions.joblib")
MATCHUP_HTE = {"hte_pitcher_batter_100", "hte_pitcher_batter_500",
               "hte_pitcher_batter_log_count"}
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
FORM_WEIGHTS = (0.60, 0.64)
CONTEXT_WEIGHTS = (0.17, 0.21, 0.25)
BASELINE = (0.64, 0.17)
MIN_POINT_ESTIMATE = 3.0 / (100000.0 / 0.25)
ROUNDS = 3000
SEED = 20260825
P = 100000.0 / 0.25


def bootstrap(frame, base, candidate, mask, rounds=ROUNDS, seed=SEED):
    part = frame.loc[mask]
    target = part["target"].to_numpy()
    gain = (base[mask] - target) ** 2 - (candidate[mask] - target) ** 2
    grouped = pd.DataFrame({"p": part["pitcher_id"].to_numpy(), "g": gain}) \
        .groupby("p")["g"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(sums), size=(rounds, len(sums)))
    draws = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return {"mean": float(gain.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975))}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    baseline_context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    baseline_context = baseline_context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    drift_frame = add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order]
    term = drift_correction(drift_frame, 1.0)

    new_context = {}
    feature_count = None
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        frame = select_v2_features(add_row_features(hierarchical, prior))
        frame = add_trackman_features(frame, trackman)
        frame = add_context_trackman_features(frame, context_trackman)
        frame = pd.concat([frame, block], axis=1)
        columns = [c for c in frame.columns if c not in MATCHUP_HTE]
        feature_count = len(columns)
        model, model_columns = hist_gbdt_pipeline(frame[columns])
        model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
        new_context[str(year)] = model.predict_proba(
            frame.loc[valid_mask, model_columns])[:, 1]
        print(f"year={year} context trained ({feature_count} feat)", flush=True)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend_and_calibrate(BASELINE[0], BASELINE[1], oof, logistic,
                            form, baseline_context, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for w_form in FORM_WEIGHTS:
        for w_context in CONTEXT_WEIGHTS:
            label = f"ctx_inseason_form{w_form:.2f}_ctx{w_context:.2f}"
            candidate = np.clip(
                blend_and_calibrate(w_form, w_context, oof, logistic,
                                    form, new_context, raw_frame)
                + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            results[label] = metrics
            print(f"evaluated {label}", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["mean"] >= MIN_POINT_ESTIMATE
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["mean"],
                   default=None)

    output = {
        "experiment": "V109_context_inseason_at_shrinkage20",
        "baseline": "V106 (0.19 / 0.64 / 0.17, drift shrinkage 3, w 0.10)",
        "baseline_public_score": 973.0643764994,
        "rationale": (
            "V94a gave Context the reconstruction at shrinkage 50 with a 0.13 weight and "
            "it failed on transfer. Shrinkage is now 20 and the weight 0.17. Last item on "
            "the re-validation list."
        ),
        "promotion_rule": "2024 point estimate >= +3 points, per the V108 correction",
        "context_feature_count": feature_count,
        "feature_shrinkage": FEATURE_SHRINKAGE, "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_weight": DRIFT_WEIGHT,
        "form_weights": list(FORM_WEIGHTS), "context_weights": list(CONTEXT_WEIGHTS),
        "results": results, "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "calibration_refitted_per_blend": True},
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"context": new_context}, PREDICTIONS, compress=3)

    print(f"\ncontext features = {feature_count} (in-season {len(feature_names())} added)")
    print("gains vs V106 baseline, by 2024 bootstrap:")
    print(f"{'candidate':>36} {'CIlo':>7} {'mean':>7} {'CIhi':>7} {'2022':>7} {'2023':>7} "
          f"{'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["mean"]):
        r = results[label]; b = r["bootstrap_2024"]; sb = r["season_bootstrap"]
        print(f"{label:>36} {b['ci95_low']*P:7.1f} {b['mean']*P:7.1f} {b['ci95_high']*P:7.1f} "
              f"{sb['2022']['mean']*P:7.1f} {sb['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)} (needs 2024 mean >= +3.0)\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
