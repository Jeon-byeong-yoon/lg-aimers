"""V158: the target-encoding smoothing strengths, set in V5 and never revisited.

The encodings shrink each entity's history toward a prior by a fixed strength:

    target_encoding_v5.STRENGTHS                     = (50, 200)
    hierarchical GROUPS['pitcher_batter']['strengths'] = (50, 200)

Both were chosen in V5 and V6, before the Form model existed in its current form, before
the in-season reconstruction, before the network, CatBoost, the factorization network and
four rebuilds of the blend. Nothing since has touched them, and the shrinkage constant one
layer down -- the in-season reconstruction's -- was gridded four times (V102, V103, V104,
V107) and moved from 50 to 20, so there is precedent for the a priori value being wrong.

Who consumes them: Form (0.32), Context (0.14), the network (0.20) and the factorization
network (0.07), which is 0.73 of the blend. CatBoost does not -- V116's winning `no_te`
variant drops every `te_*` and `hte_*` column -- so the expensive fits are not needed and
all four affected components can be retrained per variant.

That matters for correctness as much as cost: every component that reads these columns is
refit on each variant, so no model is ever handed encodings built with a strength it was
not fitted on. V154 had to split the in-season reconstruction in two for exactly that
reason; here the split is unnecessary because the one component that would have needed it
does not read the columns at all.

The incumbent strengths are refit as a control. If they do not reproduce the stored
out-of-fold predictions, the harness is wrong.

Ranked by the weakest season with the three-season average as the stated tiebreak, and no
fold permitted to lose -- the rule the leaderboard validated across six submissions, and
the tiebreak V155 was missing when seven candidates tied at zero.
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import hierarchical_target_encoding_v6 as hte
import interaction_network_v130 as inet
import target_encoding_v5 as te
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, cardinalities,
    encode_categorical, encode_numeric, numeric_statistics, predict, train,
)
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v31_feature_removal import REMOVALS
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v158_encoding_strengths_metrics.json")
PREDICTIONS = Path("artifacts/v158_encoding_strengths_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.65, 0.25, 0.10
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 200, 1000, 3000, 8000, np.inf]
LABELS = ["0-200", "200-1k", "1k-3k", "3k-8k", "8k+"]
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT = 8
FACTORIZATION_EPOCHS = 8
FACTORIZATION_HIDDEN = (128, 64)
# Multipliers on both the flat and the hierarchical strengths.
SCALES = {"incumbent": 1.0, "half": 0.5, "double": 2.0, "quadruple": 4.0}
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()

    base_flat = tuple(te.STRENGTHS)
    base_groups = {g: tuple(spec["strengths"]) for g, spec in hte.GROUPS.items()}
    print(f"incumbent flat strengths {base_flat}")
    print(f"incumbent hierarchical strengths {base_groups['pitcher_batter']} "
          f"(pitcher_batter)", flush=True)

    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv",
                              usecols=TRACKMAN_COLUMNS)
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)
    del raw_trackman
    gc.collect()
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]

    predictions = {name: {} for name in ("form", "context", "network", "factorization")}
    for scale_name, scale in SCALES.items():
        started = time.time()
        te.STRENGTHS = tuple(s * scale for s in base_flat)
        for group, spec in hte.GROUPS.items():
            spec["strengths"] = tuple(s * scale for s in base_groups[group])
        encoded = te.add_prior_season_target_encodings(data, y)
        hierarchical = hte.add_prior_season_hierarchical_encodings(
            encoded, y, ["pitcher_batter"])
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, float(y.mean())))
        form_features = add_trackman_features(form_features, trackman)
        form_features = pd.concat([form_features, block], axis=1)
        model_columns = v31_form_columns(form_features)
        print(f"\n[{scale_name}] flat {te.STRENGTHS}, "
              f"hierarchical {hte.GROUPS['pitcher_batter']['strengths']}  "
              f"[features {time.time() - started:.0f}s]", flush=True)

        for key in predictions:
            predictions[key][scale_name] = {}
        for year in YEARS:
            train_mask = season < year
            valid_mask = season == year

            model, columns = hist_gbdt_pipeline(form_features[model_columns])
            model.set_params(**{f"histgradientboostingclassifier__{k}": v
                                for k, v in FORM_CONFIG.items()})
            model.fit(form_features.loc[train_mask, columns], targets[train_mask])
            predictions["form"][scale_name][str(year)] = model.predict_proba(
                form_features.loc[valid_mask, columns])[:, 1]
            del model
            gc.collect()

            prior = float(y.loc[train_mask].mean())
            ctx = select_v2_features(add_row_features(hierarchical, prior))
            ctx = add_trackman_features(ctx, trackman)
            ctx = add_context_trackman_features(ctx, context_trackman)
            keep = [c for c in ctx.columns if c not in REMOVALS["no_matchup_hte"]]
            model, columns = hist_gbdt_pipeline(ctx[keep])
            model.fit(ctx.loc[train_mask, columns], targets[train_mask])
            predictions["context"][scale_name][str(year)] = model.predict_proba(
                ctx.loc[valid_mask, columns])[:, 1]
            del model, ctx
            gc.collect()

            categorical_columns = [c for c, _ in EMBEDDING_SPECS]
            numeric_columns = [c for c in model_columns
                               if c not in categorical_columns and c != "season"]
            identity = form_features.copy()
            for column in categorical_columns:
                if column not in identity:
                    identity[column] = raw_frame[column]
            assert_numeric(identity, numeric_columns)
            train_identity = identity.loc[train_mask]
            vocabularies = build_vocabularies(train_identity)
            statistics = numeric_statistics(
                train_identity[numeric_columns].to_numpy(dtype=np.float64))
            net = train(
                encode_categorical(train_identity, vocabularies),
                encode_numeric(
                    train_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics),
                targets[train_mask], cardinalities(vocabularies),
                epochs=NETWORK_EPOCHS, verbose=False)
            valid_identity = identity.loc[valid_mask]
            predictions["network"][scale_name][str(year)] = predict(
                net,
                encode_categorical(valid_identity, vocabularies),
                encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del net
            gc.collect()

            inet.LATENT = FACTORIZATION_LATENT
            inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]
            spec = inet.field_cardinalities(vocabularies)
            fmodel = inet.train(
                inet.encode_categorical(train_identity, vocabularies),
                inet.encode_numeric(
                    train_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics),
                targets[train_mask], spec, epochs=FACTORIZATION_EPOCHS,
                hidden=FACTORIZATION_HIDDEN, verbose=False)
            predictions["factorization"][scale_name][str(year)] = inet.predict(
                fmodel,
                inet.encode_categorical(valid_identity, vocabularies),
                inet.encode_numeric(
                    valid_identity[numeric_columns].to_numpy(dtype=np.float64),
                    statistics))
            del fmodel, identity, train_identity, valid_identity
            gc.collect()

            actual = targets[valid_mask].astype(float)
            rate = actual.mean()
            line = "  ".join(
                f"{k[:4]} {100000 * (1 - ((predictions[k][scale_name][str(year)] - actual) ** 2).mean() / (rate * (1 - rate))):7.0f}"
                for k in ("form", "context", "network", "factorization"))
            print(f"  {scale_name:10s} {year}: {line}  "
                  f"[{time.time() - started:.0f}s cumulative]", flush=True)
        del encoded, hierarchical, form_raw, form_features
        gc.collect()

    te.STRENGTHS = base_flat
    for group, spec in hte.GROUPS.items():
        spec["strengths"] = base_groups[group]

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    stored_form = joblib.load(
        "artifacts/v102_inseason_smoothing_predictions.joblib")["forms"][FEATURE_SHRINKAGE]
    stored_context = joblib.load(
        "artifacts/v31_feature_removal_predictions.joblib")["no_matchup_hte"]["context"]
    stored_network = joblib.load(
        "artifacts/v112_network_weight_predictions.joblib")["networks"]["without_season"]
    stored_factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]["latent8"]
    catboost = joblib.load(
        "artifacts/v153_projected_prior_predictions.joblib")["catboost"]["projected"]
    drift = {"v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                     for v in YEARS}}

    control_drift = max(
        float(np.abs(predictions["form"]["incumbent"][str(y_)]
                     - stored_form[str(y_)]).max()) for y_ in YEARS)
    print(f"\ncontrol Form reproduces the stored predictions to {control_drift:.3e}",
          flush=True)

    calibration_frame = raw_frame.copy()
    calibration_frame["experience_bin"] = pd.cut(
        calibration_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()

    def pipeline(form, context, network, factorization):
        parts = dict(drift, form=form, context=context, network=network,
                     catboost=catboost, factorization=factorization)
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

    baseline = pipeline(stored_form, stored_context, stored_network,
                        stored_factorization)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, (season_of == 2024))
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    results = {}
    print(f"\n{'candidate':>14} {'min':>7} {'avg':>7} {'2022':>7} {'2023':>7} "
          f"{'2024':>7} {'blk':>5} {'safe':>5}")
    for scale_name in SCALES:
        results[scale_name] = evaluate(pipeline(
            predictions["form"][scale_name], predictions["context"][scale_name],
            predictions["network"][scale_name],
            predictions["factorization"][scale_name]))
        m = results[scale_name]; t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"{scale_name:>14} {m['min_season_points']:7.2f} "
              f"{t['average_points']:7.2f} {t['season_points'][0]:7.2f} "
              f"{t['season_points'][1]:7.2f} {t['season_points'][2]:7.2f} "
              f"{m['monthly_block_win_rate']:5.0%} {'SAFE' if safe else '':>5}",
              flush=True)

    def safe(name):
        t = results[name]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [n for n in results if safe(n) and n != "incumbent"]
    promoted = max(survivors,
                   key=lambda n: (results[n]["min_season_points"],
                                  results[n]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V158_encoding_strengths",
        "baseline": "V154 (Public 1050.8512921822)",
        "rationale": (
            "The target-encoding strengths were set in V5 and V6 and never revisited, "
            "while the shrinkage one layer down was gridded four times and moved from 50 "
            "to 20. Form, Context, the network and the factorization network read them, "
            "0.73 of the blend; CatBoost's winning no_te variant does not, so every "
            "affected component can be refit per variant and no model is handed "
            "encodings built with a strength it was not fitted on."
        ),
        "incumbent_strengths": {"flat": list(base_flat),
                                "hierarchical": {k: list(v)
                                                 for k, v in base_groups.items()}},
        "scales": SCALES,
        "control": {"form_max_abs_difference": control_drift},
        "ranking": "weakest season, then the three-season average",
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)
    print(f"\nsafe (no fold loses) = {len(survivors)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
