"""V104: extend the shrinkage grid below its boundary and rank by the 2024 window.

Two problems with V103 are fixed here.

First, its grid started at shrinkage 20, which is where the best candidates sat.
An optimum on the boundary is not an optimum — the same mistake V96 caught in its
own first grid — so 10 and 15 are added and the Form model is retrained for them.

Second, V103 ranked candidates by the pooled-across-seasons bootstrap, which is
dominated by 2023. Its promoted candidate scored +29.0 pooled but +0.7 on 2024
alone. Ranking here is by the 2024 paired pitcher bootstrap lower bound, the
instrument that separated V96 (+4.4) from V103 (-9.2). Candidates are screened on
the cheap 2024 mean first, then the top slice is bootstrapped.
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


OUTPUT = Path("artifacts/v104_shrinkage_extension_metrics.json")
PREDICTIONS = Path("artifacts/v104_shrinkage_extension_predictions.joblib")
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NEW_SHRINKAGES = (10.0, 15.0)
FEATURE_SHRINKAGES = (10.0, 15.0, 20.0, 50.0)
DRIFT_SHRINKAGES = (10.0, 15.0, 20.0, 50.0)
RELIABILITY_SCALES = (25.0, 50.0, 150.0)
DRIFT_SCALES = (0.10, 0.15, 0.20, 0.25)
BASELINE = (50.0, 50.0, 150.0, 0.15)
W_FORM, W_CONTEXT = 0.52, 0.21
SCREEN_TOP = 25
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
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    forms = dict(joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")["forms"])
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    for shrinkage in NEW_SHRINKAGES:
        block = add_training_inseason_features(
            raw_frame, shrinkage=shrinkage)[feature_names()]
        forms[shrinkage] = {}
        for year in YEARS:
            train_mask = raw_frame["season"] < year
            valid_mask = raw_frame["season"] == year
            prior = float(y.loc[train_mask].mean())
            frame = select_v2_features(add_row_features(form_raw, prior))
            frame = add_trackman_features(frame, trackman)
            frame = pd.concat([frame, block], axis=1)
            columns = v31_form_columns(frame)
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
            })
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[shrinkage][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"shrinkage={shrinkage} year={year} form trained", flush=True)

    terms = {}
    for shrinkage in DRIFT_SHRINKAGES:
        block = add_training_inseason_features(raw_frame, shrinkage=shrinkage).loc[order]
        delta = np.nan_to_num(
            block["dlt_asof_pitcher_success_rate"].to_numpy(dtype=float), nan=0.0)
        inside = np.expm1(block["ins_log_n_pitcher"].to_numpy(dtype=float))
        terms[shrinkage] = {
            scale: delta * (inside / (inside + scale)) for scale in RELIABILITY_SCALES
        }

    blends = {
        shrinkage: blend_and_calibrate(
            W_FORM, W_CONTEXT, oof, logistic, forms[shrinkage], context, raw_frame)
        for shrinkage in FEATURE_SHRINKAGES
    }
    print("blends calibrated", flush=True)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    target_2024 = validation_frame.loc[mask_2024, "target"].to_numpy()
    baseline = np.clip(
        blends[BASELINE[0]] + BASELINE[3] * terms[BASELINE[1]][BASELINE[2]], 0, 1)
    base_error = (baseline[mask_2024] - target_2024) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    screen = {}
    built = {}
    for k_feature in FEATURE_SHRINKAGES:
        for k_drift in DRIFT_SHRINKAGES:
            for scale in RELIABILITY_SCALES:
                for drift in DRIFT_SCALES:
                    if (k_feature, k_drift, scale, drift) == BASELINE:
                        continue
                    label = (f"feat{k_feature:.0f}_drift{k_drift:.0f}"
                             f"_rel{scale:.0f}_w{drift:.2f}")
                    candidate = np.clip(
                        blends[k_feature] + drift * terms[k_drift][scale], 0, 1)
                    built[label] = candidate
                    screen[label] = float(
                        (base_error - (candidate[mask_2024] - target_2024) ** 2).mean())
        print(f"screened feature shrinkage {k_feature}", flush=True)

    shortlist = sorted(screen, key=screen.get, reverse=True)[:SCREEN_TOP]
    print(f"\nscreened {len(screen)} candidates, bootstrapping top {len(shortlist)}",
          flush=True)

    results = {}
    for label in shortlist:
        candidate = built[label]
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS
        }
        metrics["gain_2024_full"] = screen[label]
        results[label] = metrics

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    output = {
        "experiment": "V104_shrinkage_extension",
        "baseline": "V96 (feature 50, drift 50, reliability 150, w 0.15)",
        "baseline_public_score": 962.8787800874,
        "fixes_over_v103": [
            "shrinkage grid extended below its boundary (10, 15 added)",
            "ranking by the 2024 bootstrap lower bound instead of the pooled one",
        ],
        "feature_shrinkages": list(FEATURE_SHRINKAGES),
        "drift_shrinkages": list(DRIFT_SHRINKAGES),
        "reliability_scales": list(RELIABILITY_SCALES),
        "drift_scales": list(DRIFT_SCALES),
        "screened": len(screen),
        "bootstrapped": len(shortlist),
        "bootstrap": {"rounds": ROUNDS, "seed": SEED, "unit": "pitcher_id"},
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "constants_frozen_at_training_time": True},
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": {k: v for k, v in forms.items() if k in NEW_SHRINKAGES},
                 "baseline": baseline}, PREDICTIONS, compress=3)

    print(f"\ntop {len(shortlist)} by 2024 bootstrap lower bound (vs V96):")
    print(f"{'candidate':>34} {'2024CIlo':>9} {'2024mean':>9} {'2024hi':>8} "
          f"{'2022':>7} {'2023':>7} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"]):
        r = results[label]; b = r["bootstrap_2024"]; sb = r["season_bootstrap"]
        print(f"{label:>34} {b['ci95_low']*P:9.1f} {b['mean']*P:9.1f} {b['ci95_high']*P:8.1f} "
              f"{sb['2022']['mean']*P:7.1f} {sb['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted], indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
