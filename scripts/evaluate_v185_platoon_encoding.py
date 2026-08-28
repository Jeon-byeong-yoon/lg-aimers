"""V185: the platoon encoding exists in the code and the pipeline has never used it.

`hierarchical_target_encoding_v6.py` defines four groups. Every build and evaluation script
in this repository calls it the same way:

    add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])

so `pitcher_vs_batter_hand`, `pitcher_count` and `batter_vs_pitcher_hand` have been written,
tested once, and unused ever since. The decision that dropped them is on record, from V6:

    HGB 후보          2024 Brier   판정
    V5 기준            0.248105    기준
    투수 x 타자 손     0.248135    악화
    투수 x 카운트      0.248134    악화
    타자 x 투수 손     0.248118    악화
    투수 x 타자        0.248079    개선

The platoon group was rejected on a **3e-5 Brier difference** -- noise -- measured on a
single HistGradientBoosting model at the V5 baseline, when the score was in the 800s and
before the in-season reconstruction, the embedding network, CatBoost and the factorization
network existed.

Since then the leaderboard has paid **+14.65** for exactly that content, delivered through
the calibration layer instead: `pitcher_id x batter_hand` (V169, +7.31) and then split by
two-strike state (V175, +7.32). The content is no longer a hypothesis. What is open is
whether the *models* can use it, because the calibration layer only ever applies it as one
additive offset per cell, while a feature can interact with everything else the model sees.

`pitcher_count` comes along for the same reason: V183 put the most remaining structure
there, and it was dropped in the same table on a 2.9e-5 difference.

Four arms, each adding groups to the encoding list: the platoon group, the pitcher-count
group, both, and all three. `batter_vs_pitcher_hand` does not get its own arm because V183
scored it *below* its matched placebo (208.7 against 230.3) and V171 already lost it out of
fold; it appears only inside `all`.

**Networks are measured paired across three seeds.** V170 is the reason: V166 found a
feature change that gained all three folds at seed 42 and reversed at four other seeds,
because this network moves the blend by 8 to 16 points on seed alone. Form is deterministic
and needs one run. The baseline networks at all three seeds are cached from V164.
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
import interaction_network_v130 as inet
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
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


OUTPUT = Path("artifacts/v185_platoon_encoding_metrics.json")
PREDICTIONS = Path("artifacts/v185_platoon_encoding_predictions.joblib")
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
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT, FACTORIZATION_EPOCHS = 8, 8
FACTORIZATION_HIDDEN = (128, 64)
SEEDS = (42, 1004, 2024)
ARMS = {
    "platoon": ["pitcher_batter", "pitcher_vs_batter_hand"],
    "pitcher_count": ["pitcher_batter", "pitcher_count"],
    "both_pitcher": ["pitcher_batter", "pitcher_vs_batter_hand", "pitcher_count"],
    "all": ["pitcher_batter", "pitcher_vs_batter_hand", "pitcher_count",
            "batter_vs_pitcher_hand"],
}
P = 100000.0 / 0.25


def skill(prediction, actual):
    rate = actual.mean()
    return 100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate)))


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
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    global_prior = float(y.mean())
    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]

    forms, networks, factorizations, widths = {}, {}, {}, {}
    for arm, groups in ARMS.items():
        started = time.time()
        hierarchical = add_prior_season_hierarchical_encodings(encoded, y, groups)
        frame = select_v2_features(add_row_features(
            add_stable_form_features(hierarchical), global_prior))
        del hierarchical
        gc.collect()
        frame = add_trackman_features(frame, trackman)
        frame = pd.concat([frame, block], axis=1)
        keep = v31_form_columns(frame)
        widths[arm] = len(keep)
        categorical_columns = [c for c, _ in EMBEDDING_SPECS]
        numeric_columns = [c for c in keep
                           if c not in categorical_columns and c != "season"]
        identity = frame.copy()
        for column in categorical_columns:
            if column not in identity:
                identity[column] = raw_frame[column]
        assert_numeric(identity, numeric_columns)
        print(f"\narm {arm}: {widths[arm]} columns, groups {groups}", flush=True)
        forms[arm] = {}
        networks[arm] = {s: {} for s in SEEDS}
        factorizations[arm] = {s: {} for s in SEEDS}
        for year in YEARS:
            train_mask, valid_mask = season < year, season == year
            model, model_columns = hist_gbdt_pipeline(frame[keep])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()})
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[arm][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            del model
            gc.collect()
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
            actual = targets[valid_mask].astype(float)
            for seed in SEEDS:
                net = train(categorical, numeric, targets[train_mask],
                            cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                            seed=seed, verbose=False)
                networks[arm][seed][str(year)] = predict(
                    net, valid_categorical, valid_numeric)
                del net
                gc.collect()
                fmodel = inet.train(categorical, numeric, targets[train_mask],
                                    inet.field_cardinalities(vocabularies),
                                    epochs=FACTORIZATION_EPOCHS, seed=seed,
                                    hidden=FACTORIZATION_HIDDEN, verbose=False)
                factorizations[arm][seed][str(year)] = inet.predict(
                    fmodel, valid_categorical, valid_numeric)
                del fmodel
                gc.collect()
            print(f"  {year}: form {skill(forms[arm][str(year)], actual):7.0f}  "
                  f"netw {skill(networks[arm][42][str(year)], actual):7.0f}  "
                  f"fact {skill(factorizations[arm][42][str(year)], actual):7.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)
            del categorical, numeric, valid_categorical, valid_numeric
            gc.collect()
        del frame, identity
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    base_form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE]
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
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

    def pipeline(form, network, factorization):
        parts = dict(fixed)
        parts.update({"form": form, "network": network, "factorization": factorization})
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

    def readings(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y_]].mean() - error[masks[y_]].mean()))
                for y_ in YEARS]

    baseline = pipeline(base_form, v164["networks"][42], v164["factorizations"][42])
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    results = {}
    print(f"\n{'arm':16s} {'scope':14s} {'seed':>6s} {'2023':>8s} {'2024':>8s}", flush=True)
    for arm in ARMS:
        results[arm] = {}
        r = readings(pipeline(forms[arm], v164["networks"][42],
                              v164["factorizations"][42]), base_error)
        results[arm]["form"] = {"42": r}
        print(f"  {arm:16s} {'form only':14s} {'-':>6s} {r[1]:8.2f} {r[2]:8.2f}",
              flush=True)
        for scope in ("network", "factorization", "both_nets", "everything"):
            results[arm][scope] = {}
            for seed in SEEDS:
                net = (networks[arm][seed] if scope in ("network", "both_nets",
                                                        "everything")
                       else v164["networks"][seed])
                fact = (factorizations[arm][seed]
                        if scope in ("factorization", "both_nets", "everything")
                        else v164["factorizations"][seed])
                form = forms[arm] if scope == "everything" else base_form
                paired_base = pipeline(base_form, v164["networks"][seed],
                                       v164["factorizations"][seed])
                paired_error = (paired_base - target) ** 2
                r = readings(pipeline(form, net, fact), paired_error)
                results[arm][scope][str(seed)] = r
                print(f"  {arm:16s} {scope:14s} {seed:6d} {r[1]:8.2f} {r[2]:8.2f}",
                      flush=True)

    def unanimous(arm, scope):
        rows = results[arm][scope]
        return all(r[1] > 0 and r[2] > 0 for r in rows.values())

    winners = [(a, s) for a in ARMS for s in results[a]
               if unanimous(a, s)]
    print(f"\npositive on both informative folds for every seed: "
          f"{winners if winners else 'none'}", flush=True)

    finalists = {}
    for arm, scope in winners:
        net = (networks[arm][42] if scope in ("network", "both_nets", "everything")
               else v164["networks"][42])
        fact = (factorizations[arm][42]
                if scope in ("factorization", "both_nets", "everything")
                else v164["factorizations"][42])
        form = forms[arm] if scope in ("form", "everything") else base_form
        candidate = pipeline(form, net, fact)
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        finalists[f"{arm}|{scope}"] = metrics
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        print(f"  {arm}|{scope:14s} 2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"CI low {low:+7.2f}  blk {metrics['monthly_block_win_rate']:4.0%}",
              flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V185_platoon_encoding",
        "baseline": "V175 (Public 1067.8617513573)",
        "why_untried": ("hierarchical_target_encoding_v6.py defines four groups and every "
                        "script in the repository passes only ['pitcher_batter']. The "
                        "platoon group was dropped at V6 on a 3e-5 Brier difference from a "
                        "single HGB at the V5 baseline, before the in-season "
                        "reconstruction, the network, CatBoost and the factorization "
                        "network existed -- and the leaderboard has since paid +14.65 for "
                        "the same content through the calibration layer."),
        "seed_pairing": ("V170: a feature change gained all three folds at seed 42 and "
                         "reversed at four other seeds, so networks are measured paired"),
        "arms": {k: v for k, v in ARMS.items()},
        "widths": widths,
        "results": results,
        "unanimous": [f"{a}|{s}" for a, s in winners],
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True,
                       "prior_season_encodings_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "networks": networks,
                 "factorizations": factorizations}, PREDICTIONS, compress=3)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
