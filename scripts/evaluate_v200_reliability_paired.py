"""V200: the reliability scale was decided on readings worth 0.15 of their own noise.

V160 chose 300 over 75, 150 and 600; V162 then rejected 225, 400 and 500. Every one of
those readings required **retraining the two networks**, so candidate and baseline carried
different seed draws, and V191 measured the noise in a single-draw blend reading at
sd 5.37 on the 2024 fold. The difference of two independent draws therefore has
sd ~7.6 -- against which the readings were:

    V160   75  2024 -0.72     300  2024 +0.77     600  2024 -0.25
    V162  225  2024 +0.07     400  2024 -1.10     500  2024 -1.05

**All of them inside 0.15 standard deviations.** The same V162 run shows what a trustworthy
reading looks like: its drift-side arms are computed from the reconstruction rather than
learned, so they share the network draws, and they came back at +30.66 and -32.02 on 2023 --
large and structured. The separation is exact:

> A reading is trustworthy when candidate and baseline share the same draws.

This re-asks the question with the noise removed two ways.

**Pairing.** For each seed s, the candidate at scale X is compared to scale 300 **at the
same seed s**. The reliability scale changes one column out of 105 -- Form is bit-identical
across scales, which is how V160 found the column is never split on -- so at a fixed seed
the two network fits start from identical weights and see almost identical data, and most of
the lottery cancels. Five seeds give a mean and a standard error.

**A measured null.** Nine scale-300 draws are cached, so the *unpaired* noise is computed
directly from them: scale 300 at seed s against scale 300 at seed s'. That says how much the
pairing actually bought, rather than assuming it.

Form is not retrained -- it is bit-identical at every scale. Context, CatBoost and the
calibration layer are held at what V199 ships.
"""

import gc
import json
import sys
import time
from itertools import combinations
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


OUTPUT = Path("artifacts/v200_reliability_paired_metrics.json")
PREDICTIONS = Path("artifacts/v200_reliability_paired_predictions.joblib")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE = 20.0
INCUMBENT_SCALE = 300.0
NEW_SCALES = (150.0, 225.0, 450.0, 600.0)
PAIRED_SEEDS = (42, 1004, 2024, 777, 999)
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
NETWORK_EPOCHS = 6
FACTORIZATION_LATENT, FACTORIZATION_EPOCHS = 8, 8
FACTORIZATION_HIDDEN = (128, 64)
SHIPPED_FORM_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_CONTEXT_SEEDS = (42, 1004, 2024, 777, 999, 13)
SHIPPED_CATBOOST_SEEDS = (42, 1004, 2024, 777)
NULL_SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
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
    shared = select_v2_features(add_row_features(
        add_stable_form_features(hierarchical), float(y.mean())))
    shared = add_trackman_features(shared, trackman)
    del hierarchical, encoded, trackman
    gc.collect()
    inet.LATENT = FACTORIZATION_LATENT
    inet.FIELD_SPECS = [(c, FACTORIZATION_LATENT) for c in inet.FIELDS]

    networks = {scale: {s: {} for s in PAIRED_SEEDS} for scale in NEW_SCALES}
    factorizations = {scale: {s: {} for s in PAIRED_SEEDS} for scale in NEW_SCALES}
    for scale in NEW_SCALES:
        started = time.time()
        block = add_training_inseason_features(
            raw_frame, shrinkage=FEATURE_SHRINKAGE,
            reliability_scale=scale)[feature_names()]
        frame = pd.concat([shared, block], axis=1)
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
        del frame, block
        gc.collect()
        print(f"\nscale {scale:g}", flush=True)
        for year in YEARS:
            train_mask, valid_mask = season < year, season == year
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
            spec = inet.field_cardinalities(vocabularies)
            del train_identity, valid_identity
            gc.collect()
            for seed in PAIRED_SEEDS:
                net = train(categorical, numeric, targets[train_mask],
                            cardinalities(vocabularies), epochs=NETWORK_EPOCHS,
                            seed=seed, verbose=False)
                networks[scale][seed][str(year)] = predict(
                    net, valid_categorical, valid_numeric)
                del net
                gc.collect()
                model = inet.train(categorical, numeric, targets[train_mask], spec,
                                   epochs=FACTORIZATION_EPOCHS, seed=seed,
                                   hidden=FACTORIZATION_HIDDEN, verbose=False)
                factorizations[scale][seed][str(year)] = inet.predict(
                    model, valid_categorical, valid_numeric)
                del model
                gc.collect()
            print(f"  {year} done [{time.time() - started:.0f}s]", flush=True)
            del categorical, numeric, valid_categorical, valid_numeric
            gc.collect()
        del identity
        gc.collect()
    del shared
    gc.collect()
    joblib.dump({"networks": networks, "factorizations": factorizations},
                PREDICTIONS, compress=3)
    print(f"Saved {PREDICTIONS}", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    v197 = joblib.load("artifacts/v197_catboost_lottery_predictions.joblib")
    incumbent_net = {**v164["networks"], **v190["networks"]}
    incumbent_fm = {**v164["factorizations"], **v194["factorizations"]}
    form_store = {**v190["forms"], **v192["forms"]}

    def mean_of(store, chosen):
        return {str(y_): np.mean([store[s][str(y_)] for s in chosen], axis=0)
                for y_ in YEARS}

    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": mean_of(form_store, SHIPPED_FORM_SEEDS),
        "context": mean_of(v195["contexts"], SHIPPED_CONTEXT_SEEDS),
        "catboost": mean_of(v197["catboosts"], SHIPPED_CATBOOST_SEEDS),
    }
    frame = raw_frame.copy()
    frame["experience_bin"] = pd.cut(frame["asof_pitcher_n"], EDGES,
                                     labels=LABELS).astype(str)
    frame["two_strike"] = (frame["strikes_before"] == 2).astype("int64")
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y_: frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y_])]
        for y_ in YEARS if y_ != 2022}
    valid_frames = {y_: frame.loc[oof[str(y_)]["row_index"]] for y_ in YEARS}
    scale_sum = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(network, factorization):
        parts = dict(fixed)
        parts.update({"network": network, "factorization": factorization})
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
            for cols, weight, smoothing in TERMS:
                value = value + scale_sum * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def errors(network, factorization):
        return (pipeline(network, factorization) - target) ** 2

    def points(candidate_error, base_error):
        return [float(P * (base_error[masks[y_]].mean() - candidate_error[masks[y_]].mean()))
                for y_ in YEARS]

    # --- the measured null: unpaired noise between two scale-300 draws ---
    incumbent_errors = {s: errors({str(y_): incumbent_net[s][str(y_)] for y_ in YEARS},
                                  {str(y_): incumbent_fm[s][str(y_)] for y_ in YEARS})
                        for s in NULL_SEEDS}
    null = np.array([points(incumbent_errors[a], incumbent_errors[b])
                     for a, b in combinations(NULL_SEEDS, 2)])
    print(f"\nmeasured null -- two scale-300 draws against each other "
          f"({len(null)} pairs):", flush=True)
    print(f"  sd    {null.std(axis=0, ddof=1)[0]:7.2f} "
          f"{null.std(axis=0, ddof=1)[1]:7.2f} {null.std(axis=0, ddof=1)[2]:7.2f}",
          flush=True)
    print(f"  range {null[:, 0].ptp():7.2f} {null[:, 1].ptp():7.2f} "
          f"{null[:, 2].ptp():7.2f}", flush=True)
    print("  -> this is the noise every V160 and V162 reading was taken against",
          flush=True)

    # --- paired readings ---
    print(f"\npaired against scale {INCUMBENT_SCALE:g} at the same seed:", flush=True)
    print(f"  {'scale':>6s} {'2022':>17s} {'2023':>17s} {'2024':>17s}", flush=True)
    results = {}
    for scale in NEW_SCALES:
        rows = []
        for seed in PAIRED_SEEDS:
            candidate = errors(networks[scale][seed], factorizations[scale][seed])
            rows.append(points(candidate, incumbent_errors[seed]))
        block_ = np.array(rows)
        mean = block_.mean(axis=0)
        se = block_.std(axis=0, ddof=1) / np.sqrt(len(PAIRED_SEEDS))
        # And the average-of-five against the average-of-five.
        avg = points(errors(mean_of(networks[scale], PAIRED_SEEDS),
                            mean_of(factorizations[scale], PAIRED_SEEDS)),
                     errors(mean_of(incumbent_net, PAIRED_SEEDS),
                            mean_of(incumbent_fm, PAIRED_SEEDS)))
        results[f"{scale:g}"] = {
            "per_seed": {str(s): r for s, r in zip(PAIRED_SEEDS, rows)},
            "paired_mean": mean.tolist(), "paired_se": se.tolist(),
            "average_vs_average": avg,
            "all_positive": bool((mean > 0).all())}
        print(f"  {scale:6g} " + "  ".join(
            f"{mean[i]:+7.2f}+-{se[i]:5.2f}" for i in range(3))
            + ("   ALL POSITIVE" if (mean > 0).all() else ""), flush=True)
        print(f"         avg-vs-avg  {avg[0]:+7.2f}        {avg[1]:+7.2f}        "
              f"{avg[2]:+7.2f}", flush=True)

    paired_sd = np.mean([np.array(results[k]["paired_se"])[2]
                         * np.sqrt(len(PAIRED_SEEDS)) for k in results])
    print(f"\npairing cut the 2024 noise from sd {null.std(axis=0, ddof=1)[2]:.2f} "
          f"(unpaired) to sd {paired_sd:.2f} (paired, single seed) -- a factor of "
          f"{null.std(axis=0, ddof=1)[2] / paired_sd:.1f}", flush=True)
    survivors = [k for k in results if results[k]["all_positive"]]
    print(f"scales positive on every season: {survivors if survivors else 'none'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V200_reliability_paired",
        "baseline": "V199 (Public 1083.2461959655), feature reliability scale 300",
        "why": ("V160 and V162 decided this constant on readings inside 0.15 standard "
                "deviations of their own noise, because every arm retrained the two "
                "networks and so carried different seed draws"),
        "design": ("paired at the same seed, five seeds per scale, plus a measured null "
                   "from the nine cached scale-300 draws"),
        "incumbent_scale": INCUMBENT_SCALE,
        "new_scales": list(NEW_SCALES),
        "paired_seeds": list(PAIRED_SEEDS),
        "measured_null": {"pairs": len(null),
                          "sd": null.std(axis=0, ddof=1).tolist(),
                          "range": [float(null[:, i].ptp()) for i in range(3)]},
        "results": results,
        "survivors": survivors,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
