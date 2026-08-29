"""V190: measure the lottery on the largest weight, and find the network's best k.

V189 gained +3.59 by replacing one network draw with an average of three, and the reasoning
that got it there generalises: every learned component is one draw from a lottery, and the
fold reading of any single draw is not that component's value.

The lottery has now been measured for exactly one component. Ranked by blend weight:

    Form                 0.32   HistGradientBoosting, random_state=42   unmeasured
    CatBoost             0.27   random_seed=42, ordered permutations     unmeasured
    embedding network    0.20   averaged over three seeds (V189)        sd 7.76 on 2024
    Context              0.14   HistGradientBoosting, random_state=42   unmeasured
    factorization        0.07   single draw; V188 expected 2022/2023 negative

**Form carries more weight than the component that just paid.** Its randomness is narrower
than a network's -- `random_state` only picks the 200,000-row subsample that sets the bin
thresholds -- so the lottery may be small, but "may be small" is what V164 assumed about
the reference seed and it cost this project five versions of a wrong conclusion. One CatBoost
fit takes 160 seconds on the smallest fold, so that one is deferred; Form is cheap.

Second question. `net3` shipped because V188 put it ahead of `net5` on expected 2023
(+9.91 against +1.97), which is odd -- more draws should not be worse -- and suggests the
estimate itself was noisy at five baselines. Four more seeds are trained so the average of
k can be scored against **held-out** seeds only, which is the clean form of the estimator:
comparing an average against a seed inside it reuses the same draw on both sides.

Baseline throughout is V189 exactly: the three-seed network average, the factorization
network at seed 42, and V175's four calibration segments.
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
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


OUTPUT = Path("artifacts/v190_form_lottery_and_k_metrics.json")
PREDICTIONS = Path("artifacts/v190_form_lottery_and_k_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
FORM_SEEDS = (42, 1004, 2024)
NETWORK_EPOCHS = 6
SHIPPED_NETWORK_SEEDS = (42, 1004, 2024)
NEW_NETWORK_SEEDS = (13, 314, 2718, 65537)
ALL_NETWORK_SEEDS = SHIPPED_NETWORK_SEEDS + (777, 999) + NEW_NETWORK_SEEDS
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    frame = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    del hierarchical, encoded
    gc.collect()
    frame = add_trackman_features(frame, trackman)
    frame = pd.concat([frame, block], axis=1)
    keep = v31_form_columns(frame)
    assert len(keep) == 105, len(keep)
    categorical_columns = [c for c, _ in EMBEDDING_SPECS]
    numeric_columns = [c for c in keep
                       if c not in categorical_columns and c != "season"]
    identity = frame.copy()
    for column in categorical_columns:
        if column not in identity:
            identity[column] = raw_frame[column]
    assert_numeric(identity, numeric_columns)

    forms = {s: {} for s in FORM_SEEDS}
    networks = {s: {} for s in NEW_NETWORK_SEEDS}
    started = time.time()
    for year in YEARS:
        train_mask, valid_mask = season < year, season == year
        actual = targets[valid_mask].astype(float)
        rate = actual.mean()
        for seed in FORM_SEEDS:
            model, model_columns = hist_gbdt_pipeline(frame[keep])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()},
                             histgradientboostingclassifier__random_state=seed)
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[seed][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            del model
            gc.collect()
        skills = [100000 * (1 - ((forms[s][str(year)] - actual) ** 2).mean()
                            / (rate * (1 - rate))) for s in FORM_SEEDS]
        print(f"  {year} Form by seed: " + "  ".join(f"{v:7.0f}" for v in skills)
              + f"   [{time.time() - started:.0f}s]", flush=True)

        train_identity, valid_identity = identity.loc[train_mask], identity.loc[valid_mask]
        vocabularies = build_vocabularies(train_identity)
        statistics = numeric_statistics(
            train_identity[numeric_columns].to_numpy(dtype=np.float64))
        categorical = encode_categorical(train_identity, vocabularies)
        numeric = encode_numeric(
            train_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        valid_categorical = encode_categorical(valid_identity, vocabularies)
        valid_numeric = encode_numeric(
            valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
        del train_identity, valid_identity
        gc.collect()
        for seed in NEW_NETWORK_SEEDS:
            net = train(categorical, numeric, targets[train_mask],
                        cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                        seed=seed, verbose=False)
            networks[seed][str(year)] = predict(net, valid_categorical, valid_numeric)
            del net
            gc.collect()
        skills = [100000 * (1 - ((networks[s][str(year)] - actual) ** 2).mean()
                            / (rate * (1 - rate))) for s in NEW_NETWORK_SEEDS]
        print(f"  {year} new network seeds: " + "  ".join(f"{v:7.0f}" for v in skills)
              + f"   [{time.time() - started:.0f}s]", flush=True)
        del categorical, numeric, valid_categorical, valid_numeric
        gc.collect()
    del frame, identity
    gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    cached_form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                              )["forms"][FEATURE_SHRINKAGE]
    all_networks = dict(v164["networks"])
    all_networks.update(networks)
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        "factorization": v160["factorization"][FEATURE_RELIABILITY],
    }
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    calibration_frame["two_strike"] = (
        calibration_frame["strikes_before"] == 2).astype("int64")
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y_: calibration_frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y_])]
        for y_ in YEARS if y_ != 2022}
    valid_frames = {y_: calibration_frame.loc[oof[str(y_)]["row_index"]] for y_ in YEARS}
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(form, network):
        parts = dict(fixed)
        parts.update({"form": form, "network": network})
        blend = {y_: sum(BASE[n] * parts[n][str(y_)] for n in NAMES6) for y_ in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def mean_of(store, seeds):
        return {str(y_): np.mean([store[s][str(y_)] for s in seeds], axis=0)
                for y_ in YEARS}

    shipped_network = mean_of(v164["networks"], SHIPPED_NETWORK_SEEDS)

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    # ---- 1. does the cached Form match one of the seeds? ----
    print("\ncontrol: which seed reproduces the shipped Form?", flush=True)
    for seed in FORM_SEEDS:
        gap = max(float(np.abs(forms[seed][str(y_)] - cached_form[str(y_)]).max())
                  for y_ in YEARS)
        print(f"  seed {seed:5d}: max |new - shipped| = {gap:.3e}", flush=True)

    # ---- 2. Form's lottery ----
    print("\nForm's lottery: each seed as the blend baseline, others measured against it",
          flush=True)
    form_singles = {s: pipeline(forms[s], shipped_network) for s in FORM_SEEDS}
    form_errors = {s: (p - target) ** 2 for s, p in form_singles.items()}
    form_lottery = {}
    for s in FORM_SEEDS:
        row = points(form_singles[s], form_errors[FORM_SEEDS[0]])
        form_lottery[str(s)] = row
        print(f"  seed {s:5d} vs seed 42: {row[0]:+8.2f} {row[1]:+8.2f} {row[2]:+8.2f}",
              flush=True)
    matrix = np.array([form_lottery[str(s)] for s in FORM_SEEDS])
    print(f"  range: {matrix[:, 0].ptp():8.2f} {matrix[:, 1].ptp():8.2f} "
          f"{matrix[:, 2].ptp():8.2f}   (the network's 2024 range was 17.94)", flush=True)

    form_average = mean_of(forms, FORM_SEEDS)
    candidate = pipeline(form_average, shipped_network)
    expected = np.array([points(candidate, form_errors[s]) for s in FORM_SEEDS]).mean(axis=0)
    print(f"  three-seed Form average, expected vs a typical draw: "
          f"{expected[0]:+7.2f} {expected[1]:+7.2f} {expected[2]:+7.2f}   "
          f"{'ALL POSITIVE' if (expected > 0).all() else ''}", flush=True)

    # ---- 3. the network's best k, scored on held-out seeds only ----
    print("\nthe network's k, each average scored only against seeds it does not contain:",
          flush=True)
    base_error = {s: ((pipeline(cached_form, all_networks[s]) - target) ** 2)
                  for s in ALL_NETWORK_SEEDS}
    k_table = {}
    for k in range(1, 8):
        inside = ALL_NETWORK_SEEDS[:k]
        outside = [s for s in ALL_NETWORK_SEEDS if s not in inside]
        avg = (all_networks[inside[0]] if k == 1
               else mean_of(all_networks, inside))
        cand = pipeline(cached_form, avg)
        rows = np.array([points(cand, base_error[s]) for s in outside])
        k_table[k] = {"inside": list(inside), "held_out": outside,
                      "expected": rows.mean(axis=0).tolist(),
                      "sd": rows.std(axis=0, ddof=1).tolist()}
        m = rows.mean(axis=0)
        print(f"  k={k} ({len(outside)} held out): {m[0]:+7.2f} {m[1]:+7.2f} "
              f"{m[2]:+7.2f}   sd {rows.std(axis=0, ddof=1)[2]:5.2f}"
              + ("   <- shipped" if k == 3 else ""), flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V190_form_lottery_and_k",
        "baseline": "V189 (Public 1071.4549348488)",
        "questions": {
            "form": ("Form carries 0.32, more than the network that just paid, and its "
                     "lottery has never been measured; HistGradientBoosting's "
                     "random_state picks the 200,000-row subsample that sets the bin "
                     "thresholds"),
            "k": ("V188 put net3 ahead of net5 on expected 2023, which more draws should "
                  "not do, so the estimate was probably noisy at five baselines; the "
                  "average of k is now scored only against seeds it does not contain"),
        },
        "form_seeds": list(FORM_SEEDS),
        "form_lottery_vs_seed42": form_lottery,
        "form_lottery_range": [float(matrix[:, i].ptp()) for i in range(3)],
        "form_average_expected": expected.tolist(),
        "network_k_table": k_table,
        "deferred": ("CatBoost carries 0.27 and one fit takes 160s on the smallest fold, "
                     "so three seeds over three folds is about 45 minutes, and three "
                     "shipped models would put the archive near 220 MB"),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "networks": networks}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
