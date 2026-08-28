"""V166: three feature groups have been dropped since V2 and never re-asked.

`feature_engineering_v2.py` carries this, and the comment dates the decision:

    # Selected after the 2024 HGB ablation. These hand-crafted groups added noise
    # beyond the original official as-of columns and are omitted from the candidate.
    V2_DROPPED_GROUPS = ["game_context", "recent_form", "failure_proxies"]

Eighteen features -- eight game-context flags, eight recent-form gaps, two failure
proxies -- have been withheld from Form, the network and the factorization network for
the entire project, on evidence from a single HistGradientBoosting model at V2, when the
score was in the 800s. Since then the pipeline gained the in-season reconstruction (V92),
the embedding network (V111), CatBoost (V117) and the factorization network (V130), and
weight refits against exactly this kind of staleness have paid four times.

`recent_form` is the group worth naming. Its eight columns are gaps between the pitcher's
last one, three and five games and his career rate -- built from the official
`asof_pitcher_prev{1,3,5}_game_*` columns, which are present in `test.csv` and therefore
legal per row. "Is this pitcher's command below his own baseline right now" is the one
question the blend's six pitcher-summary components are least able to answer, and V146's
oracle audit put pitcher-level information close to exhausted while the interaction terms
still held signal. A within-pitcher deviation is that kind of term.

Four arms: each group restored alone, then all three. Baseline is the cached Form,
network and factorization predictions -- which are by definition the all-dropped arm, so
no reproduction control is needed for it. Context and CatBoost are untouched; V157 showed
refitting Context is not worth it, and CatBoost builds its own encodings.

Everything else is V161 exactly. Filter: no fold may lose, and at least one must gain;
ranked by weakest season, then the three-season average.
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
from feature_engineering_v2 import (
    FEATURE_GROUPS, V2_DROPPED_GROUPS, add_row_features, select_v2_features,
)
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v166_restore_dropped_groups_metrics.json")
PREDICTIONS = Path("artifacts/v166_restore_dropped_groups_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT, FACTORIZATION_EPOCHS = 8, 8
FACTORIZATION_HIDDEN = (128, 64)
ARMS = {
    "game_context": ["game_context"],
    "recent_form": ["recent_form"],
    "failure_proxies": ["failure_proxies"],
    "all_three": ["game_context", "recent_form", "failure_proxies"],
}


def skill(prediction, actual):
    rate = actual.mean()
    return 100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate)))


def select_with_restored(frame, restore):
    dropped = [column for group in V2_DROPPED_GROUPS if group not in restore
               for column in FEATURE_GROUPS[group]]
    return frame.drop(columns=dropped)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    del encoded, hierarchical
    gc.collect()
    global_prior = float(y.mean())
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE,
        reliability_scale=FEATURE_RELIABILITY)[feature_names()]
    rows = add_row_features(form_raw, global_prior)
    del form_raw
    gc.collect()
    baseline_width = len(v31_form_columns(pd.concat(
        [add_trackman_features(select_v2_features(rows), trackman), block], axis=1)))
    print(f"baseline model width: {baseline_width} columns", flush=True)

    forms, networks, factorizations, widths = {}, {}, {}, {}
    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]

    for arm, restore in ARMS.items():
        started = time.time()
        frame = add_trackman_features(select_with_restored(rows, restore), trackman)
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
        print(f"\narm {arm}: {widths[arm]} columns "
              f"(+{widths[arm] - baseline_width} restored)", flush=True)
        forms[arm], networks[arm], factorizations[arm] = {}, {}, {}
        for year in YEARS:
            train_mask = season < year
            valid_mask = season == year
            model, model_columns = hist_gbdt_pipeline(frame[keep])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()})
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[arm][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            del model
            gc.collect()
            train_identity = identity.loc[train_mask]
            valid_identity = identity.loc[valid_mask]
            vocabularies = build_vocabularies(train_identity)
            statistics = numeric_statistics(
                train_identity[numeric_columns].to_numpy(dtype=np.float64))
            categorical = encode_categorical(train_identity, vocabularies)
            numeric = encode_numeric(
                train_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
            valid_categorical = encode_categorical(valid_identity, vocabularies)
            valid_numeric = encode_numeric(
                valid_identity[numeric_columns].to_numpy(dtype=np.float64), statistics)
            net = train(categorical, numeric, targets[train_mask],
                        cardinalities(vocabularies), epochs=NETWORK_EPOCHS, verbose=False)
            networks[arm][str(year)] = predict(net, valid_categorical, valid_numeric)
            del net
            gc.collect()
            fmodel = inet.train(categorical, numeric, targets[train_mask],
                                inet.field_cardinalities(vocabularies),
                                epochs=FACTORIZATION_EPOCHS,
                                hidden=FACTORIZATION_HIDDEN, verbose=False)
            factorizations[arm][str(year)] = inet.predict(
                fmodel, valid_categorical, valid_numeric)
            del fmodel, train_identity, valid_identity, categorical, numeric
            del valid_categorical, valid_numeric
            gc.collect()
            actual = targets[valid_mask].astype(float)
            print(f"  {year}: form {skill(forms[arm][str(year)], actual):7.0f}  "
                  f"netw {skill(networks[arm][str(year)], actual):7.0f}  "
                  f"fact {skill(factorizations[arm][str(year)], actual):7.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)
        del frame, identity
        gc.collect()

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    base_form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE]
    base_network = v160["network"][FEATURE_RELIABILITY]
    base_factorization = v160["factorization"][FEATURE_RELIABILITY]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                          )["no_matchup_hte"]["context"]
    catboost = joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                           )["catboost"]["projected"]
    v17 = {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
           for v in YEARS}
    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(form, network, factorization):
        parts = {"v17": v17, "form": form, "context": context, "network": network,
                 "catboost": catboost, "factorization": factorization}
        weights = tuple(BASE[n] for n in NAMES6)
        raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
               for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            train_f = calibration_frame.loc[index]
            valid_f = calibration_frame.loc[oof[str(year)]["row_index"]]
            pieces.append(np.clip(
                raw[year] + residual.mean()
                + W_COUNT * segment_correction(
                    train_f, residual, valid_f, ["balls_before", "strikes_before"], 500)
                + W_PITCHER_COUNT * segment_correction(
                    train_f, residual, valid_f,
                    ["pitcher_id", "balls_before", "strikes_before"], 300)
                + W_EXPERIENCE * segment_correction(
                    train_f, residual, valid_f, ["experience_bin"],
                    EXPERIENCE_SMOOTHING), 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(base_form, base_network, base_factorization)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    def show(label, m):
        t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:34s} min {m['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {m['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    results = {}
    print("\nrestored groups, all three components refit together:", flush=True)
    for arm in ARMS:
        results[arm] = evaluate(pipeline(forms[arm], networks[arm], factorizations[arm]))
        results[arm]["scope"] = "all"
        show(arm, results[arm])

    print("\nand one component at a time, to locate any gain:", flush=True)
    for arm in ARMS:
        for scope, triple in (
                ("form", (forms[arm], base_network, base_factorization)),
                ("network", (base_form, networks[arm], base_factorization)),
                ("factorization", (base_form, base_network, factorizations[arm]))):
            label = f"{arm}|{scope}"
            results[label] = evaluate(pipeline(*triple))
            results[label]["scope"] = scope
            show(label, results[label])

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V166_restore_dropped_groups",
        "baseline": "V161 (Public 1053.2326413884)",
        "why_untried": ("V2_DROPPED_GROUPS was set after the V2 single-HGB ablation, when "
                        "the score was in the 800s, and has never been re-asked. Eighteen "
                        "features have been withheld from Form, the network and the "
                        "factorization network for the entire project. The in-season "
                        "reconstruction, the embedding network, CatBoost and the "
                        "factorization network all arrived afterwards."),
        "groups": {g: FEATURE_GROUPS[g] for g in V2_DROPPED_GROUPS},
        "baseline_width": baseline_width,
        "widths": widths,
        "results": {k: {"scope": v["scope"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True,
                       "row_level_features_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "networks": networks,
                 "factorizations": factorizations}, PREDICTIONS, compress=3)
    print(f"\nsafe = {len(survivors)} of {len(results)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
