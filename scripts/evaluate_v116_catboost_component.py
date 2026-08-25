"""V116: add CatBoost as a fifth blend component.

V90 excluded CatBoost and LightGBM on the grounds that they are "not part of the
documented server base packages". Re-reading `docs/05`, that was a self-imposed
constraint rather than a rule: `requirements.txt` is installed with pip, the server
allows internet access for package installation, and an installation failure does
not count against the daily submission quota. So trying costs almost nothing.

The history points here too. V65 and V71 were the only experiments that ever
improved all three seasons at once, and both were LightGBM; they were rejected for
falling under the 5e-6 threshold in force at the time, then dropped entirely for
packaging risk. Since then the instrument improved (the 2024 bootstrap now tracks
the leaderboard at a cumulative ratio of 1.08 over six candidates), the blend was
restructured, and model diversity was shown to pay when the embedding network added
+29.5.

CatBoost specifically fits this problem: ordered target statistics are built for
high-cardinality categoricals, which is exactly `pitcher_id` (792) and `batter_id`
(830). Our hand-built `te_*`/`hte_*` encodings do the same job worse — V94b showed
they carry league-level contamination.

Two feature variants are compared: the full Form feature set with native categorical
handling, and the same set with `te_*`/`hte_*` removed so CatBoost derives its own
statistics instead of consuming ours.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import blend as blend4, bootstrap
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v116_catboost_metrics.json")
PREDICTIONS = Path("artifacts/v116_catboost_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
# V114 baseline: v17 / form / context / network
BASE = (0.19, 0.40, 0.21, 0.20)
CONFIGS = {
    "gentle": dict(iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=6.0),
    "strong": dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0),
}
FEATURE_VARIANTS = ("full", "no_te")
CATBOOST_WEIGHTS = (0.08, 0.14, 0.20)
ROUNDS = 3000
SEED = 20260825
P = 100000.0 / 0.25


def blend5(weights, oof, logistic, form, context, network, extra, frame):
    """Five-way blend; calibration refitted exactly as every earlier step did."""
    w_v17, w_form, w_context, w_network, w_extra = weights
    raw = {}
    for year in YEARS:
        key = str(year)
        v17 = 0.95 * v11_prediction(oof[key]) + 0.05 * logistic[key]
        raw[year] = (w_v17 * v17 + w_form * form[key] + w_context * context[key]
                     + w_network * network[key] + w_extra * extra[key])
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            target.append(oof[str(history)]["target"].astype(float))
            prediction.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(target) - np.concatenate(prediction)
        train_frame = frame.loc[index]
        valid_frame = frame.loc[oof[str(year)]["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        pieces.append(np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1))
    return np.concatenate(pieces)


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

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns
                        if c.startswith(("te_", "hte_"))]
    column_sets = {
        "full": model_columns + [c for c in categorical if c not in model_columns],
        "no_te": [c for c in model_columns if c not in encoding_columns]
                 + [c for c in categorical if c not in model_columns],
    }
    print(f"categorical: {len(categorical)}  encodings removed in no_te: "
          f"{len(encoding_columns)}", flush=True)

    frames = {}
    for variant, columns in column_sets.items():
        work = features[columns].copy()
        for name in categorical:
            work[name] = work[name].astype(str)
        numeric = [c for c in columns if c not in categorical]
        work[numeric] = work[numeric].astype(np.float32)
        frames[variant] = work
        print(f"  {variant}: {work.shape[1]} columns", flush=True)

    targets = y.to_numpy()
    predictions, standalone = {}, {}
    for variant in FEATURE_VARIANTS:
        for name, config in CONFIGS.items():
            key = f"{variant}_{name}"
            predictions[key], standalone[key] = {}, {}
            for year in YEARS:
                started = time.time()
                train_mask = (raw_frame["season"] < year).to_numpy()
                valid_mask = (raw_frame["season"] == year).to_numpy()
                model = CatBoostClassifier(
                    **config, random_seed=42, verbose=0, thread_count=6,
                    cat_features=categorical, allow_writing_files=False)
                model.fit(frames[variant].loc[train_mask], targets[train_mask])
                prediction = model.predict_proba(frames[variant].loc[valid_mask])[:, 1]
                predictions[key][str(year)] = prediction
                target = targets[valid_mask].astype(float)
                rate = target.mean()
                standalone[key][str(year)] = float(
                    100000 * (1 - ((prediction - target) ** 2).mean()
                              / (rate * (1 - rate))))
                print(f"  {key} {year}: standalone "
                      f"{standalone[key][str(year)]:8.0f}  [{time.time() - started:.0f}s]",
                      flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend4(BASE, oof, logistic, form, context, network, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for key in predictions:
        for w_extra in CATBOOST_WEIGHTS:
            # Take the new component's share from the V17 layer first, then Form.
            w_v17 = max(0.0, BASE[0] - w_extra)
            w_form = round(BASE[1] - max(0.0, w_extra - BASE[0]), 4)
            label = f"{key}_cb{w_extra:.2f}"
            candidate = np.clip(
                blend5((round(w_v17, 4), w_form, BASE[2], BASE[3], w_extra),
                       oof, logistic, form, context, network, predictions[key], raw_frame)
                + DRIFT_WEIGHT * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            metrics["weights"] = {"v17": round(w_v17, 4), "form": w_form,
                                  "context": BASE[2], "network": BASE[3],
                                  "catboost": w_extra}
            results[label] = metrics
        print(f"{key} evaluated", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V116_catboost_component",
        "baseline": "V114 (0.19 / 0.40 / 0.21 / 0.20)",
        "baseline_public_score": 1002.5719943737,
        "rationale": (
            "V90 excluded CatBoost as not being a documented base package, but "
            "requirements.txt is pip-installed, the server permits internet for package "
            "installation, and installation failures do not count against the daily quota."
        ),
        "configs": CONFIGS,
        "feature_variants": list(FEATURE_VARIANTS),
        "categorical_features": categorical,
        "encodings_removed_in_no_te": encoding_columns,
        "catboost_weights": list(CATBOOST_WEIGHTS),
        "standalone_skill": standalone,
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)

    print("\nstandalone skill:")
    for key, values in standalone.items():
        print(f"  {key:18s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print("\ntop 15 by 2024 bootstrap lower bound (vs V114):")
    print(f"{'candidate':>26} {'v17':>5} {'CIlo':>7} {'mean':>7} {'2022':>7} "
          f"{'2023':>7} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"])[:15]:
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>26} {r['weights']['v17']:5.2f} {b['ci95_low']*P:7.1f} "
              f"{b['mean']*P:7.1f} {s['2022']['mean']*P:7.1f} {s['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
