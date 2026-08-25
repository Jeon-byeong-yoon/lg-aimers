"""V107: re-test the Form configuration now that its features changed.

V97 rejected every alternative Form configuration, but that was against features
shrunk at 50 and a Form weight of 0.52. Both have moved: V105 set the shrinkage to
20 and V106 raised the weight to 0.64, so this layer is now two thirds of the blend
and its inputs are noisier than when the configuration was chosen in V38.

V97 only tried *more* capacity (max_leaf 23 and 31) and found it worse. Lowering
the shrinkage from 50 to 20 means each ins_/dlt_ column carries more sampling
noise, which argues the opposite way — toward tighter trees and heavier
regularisation. That direction was never tested. `wide_23` is kept as a control to
confirm the sign really did flip.

Ranked by the 2024 paired pitcher bootstrap, whose cumulative ratio to the
leaderboard now stands at 1.039 over five adopted candidates.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v107_form_config_metrics.json")
PREDICTIONS = Path("artifacts/v107_form_config_predictions.joblib")
BASE = {"max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 20.0,
        "learning_rate": 0.03, "max_iter": 500}
CONFIGS = {
    "tight_11": {**BASE, "max_leaf_nodes": 11},
    "tight_11_reg": {**BASE, "max_leaf_nodes": 11, "min_samples_leaf": 300,
                     "l2_regularization": 30.0},
    "reg_l2_40": {**BASE, "l2_regularization": 40.0},
    "reg_min_400": {**BASE, "min_samples_leaf": 400},
    "tight_slow": {**BASE, "max_leaf_nodes": 11, "learning_rate": 0.02, "max_iter": 750},
    "wide_23": {**BASE, "max_leaf_nodes": 23},
}
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
FORM_WEIGHTS = (0.60, 0.64, 0.68)
CONTEXT_WEIGHT = 0.17
BASELINE_FORM = 0.64
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
    form_raw = add_stable_form_features(hierarchical)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    baseline_form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    baseline_form = baseline_form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    drift_block = add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order]
    delta = np.nan_to_num(
        drift_block["dlt_asof_pitcher_success_rate"].to_numpy(dtype=float), nan=0.0)
    inside = np.expm1(drift_block["ins_log_n_pitcher"].to_numpy(dtype=float))
    term = delta * (inside / (inside + RELIABILITY))

    forms = {name: {} for name in CONFIGS}
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        frame = select_v2_features(add_row_features(form_raw, prior))
        frame = add_trackman_features(frame, trackman)
        frame = pd.concat([frame, block], axis=1)
        columns = v31_form_columns(frame)
        for name, config in CONFIGS.items():
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in config.items()})
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[name][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"year={year} config={name} trained", flush=True)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend_and_calibrate(BASELINE_FORM, CONTEXT_WEIGHT, oof, logistic,
                            baseline_form, context, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for name, form in forms.items():
        for w_form in FORM_WEIGHTS:
            label = f"{name}_form{w_form:.2f}"
            candidate = np.clip(
                blend_and_calibrate(w_form, CONTEXT_WEIGHT, oof, logistic,
                                    form, context, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
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
        return (r["bootstrap_2024"]["mean"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["mean"],
                   default=None)

    output = {
        "experiment": "V107_form_config_at_shrinkage20",
        "baseline": "V106 (form config V38 gentle_500, form weight 0.64)",
        "baseline_public_score": 973.0643764994,
        "rationale": (
            "V97 rejected alternative Form configurations against features shrunk at 50 "
            "and a 0.52 weight. Shrinkage is now 20 and the weight 0.64, and V97 only "
            "tried more capacity; less smoothing argues for tighter, more regularised trees."
        ),
        "base_config": BASE, "configs": CONFIGS,
        "feature_shrinkage": FEATURE_SHRINKAGE, "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_weight": DRIFT_WEIGHT, "form_weights": list(FORM_WEIGHTS),
        "context_weight": CONTEXT_WEIGHT,
        "results": results, "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "calibration_refitted_per_blend": True},
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms}, PREDICTIONS, compress=3)

    print("\ngains vs V106 baseline, by 2024 bootstrap:")
    print(f"{'candidate':>24} {'CIlo':>7} {'mean':>7} {'CIhi':>7} {'2022':>7} {'2023':>7} "
          f"{'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["mean"]):
        r = results[label]; b = r["bootstrap_2024"]; sb = r["season_bootstrap"]
        print(f"{label:>24} {b['ci95_low']*P:7.1f} {b['mean']*P:7.1f} {b['ci95_high']*P:7.1f} "
              f"{sb['2022']['mean']*P:7.1f} {sb['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted], indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
