"""V198: the blend weights were fitted against single draws, and the components are averages now.

Four components are now averages of six to nine draws, and the blend weights they enter
with -- v17 0.00, Form 0.32, Context 0.14, network 0.20, CatBoost 0.27, factorization 0.07 --
were fitted in V119, V135 and V139 against **single-draw** components. A weight is paid for
the independent part of a prediction, and averaging changes exactly that: it removes the
lottery variance while keeping the signal, so every averaged component is now a different
object from the one its weight was fitted to. This is the staleness that has paid four
times in this project.

V167 already tried refitting weights on averaged components and found nothing. That was
measured against seed 42, the reference V188 showed to be the luckiest draw of five on the
2024 fold -- the same bias that made V164 reject averaging for five versions and made V190's
k table read backwards. So the question is open again.

**The danger is equally clear.** V179 climbed the eight calibration parameters and measured
+4 to +7 points of pure selection bias: each climb helped only the season it was fitted on.
Six weights is the same kind of search. So the answer here is not a hill-climb score, it is
the **cross-fold transfer test** V179 established as the valid instrument:

  * climb the weights against 2023 alone, with 2024 never consulted, then read 2024
  * climb against 2024 alone, with 2023 never consulted, then read 2023
  * in-fold minus out-of-fold is the selection bias, in points

If a climb transfers, the weights really are mis-specified for averaged components. If each
climb only helps its own fold, the weights stand and V167's conclusion survives its bad
reference.
"""

import json
import sys
from itertools import permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v198_weights_on_averages_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SHIPPED = {"network": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "form": (42, 1004, 2024, 777, 999, 13),
           "factorization": (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537),
           "context": (42, 1004, 2024, 777, 999, 13)}
MOVERS = ["form", "context", "network", "catboost", "factorization"]
STEPS = (0.05, 0.03, 0.02, 0.01)
ROUNDS = 6
P = 100000.0 / 0.25


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    v192 = joblib.load("artifacts/v192_more_form_seeds_predictions.joblib")
    v194 = joblib.load("artifacts/v194_factorization_lottery_predictions.joblib")
    v195 = joblib.load("artifacts/v195_context_lottery_predictions.joblib")
    stores = {
        "network": {**v164["networks"], **v190["networks"]},
        "form": {**v190["forms"], **v192["forms"]},
        "factorization": {**v164["factorizations"], **v194["factorizations"]},
        "context": v195["contexts"],
    }

    def mean_of(store, chosen):
        return {str(y): np.mean([store[s][str(y)] for s in chosen], axis=0)
                for y in YEARS}

    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        **{name: mean_of(stores[name], SHIPPED[name]) for name in stores},
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
    train_frames = {y: frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y])]
        for y in YEARS if y != 2022}
    valid_frames = {y: frame.loc[oof[str(y)]["row_index"]] for y in YEARS}
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(weights):
        blend = {y: sum(weights[n] * parts[n][str(y)] for n in NAMES6) for y in YEARS}
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
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, cols, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = pipeline(BASE)
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def readings(weights):
        error = (pipeline(weights) - target) ** 2
        return {str(y): float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS}

    def climb(fit_year):
        weights = dict(BASE)
        best = readings(weights)
        seen = {tuple(round(weights[n], 4) for n in NAMES6)}
        count = 1
        for _ in range(ROUNDS):
            improved = False
            for step in STEPS:
                trials = []
                for source, sink in permutations(MOVERS, 2):
                    if weights[source] < step - 1e-12:
                        continue
                    trial = dict(weights)
                    trial[source] = round(trial[source] - step, 4)
                    trial[sink] = round(trial[sink] + step, 4)
                    key = tuple(trial[n] for n in NAMES6)
                    if key in seen:
                        continue
                    seen.add(key)
                    r = readings(trial)
                    count += 1
                    trials.append((r[str(fit_year)], trial, r))
                if not trials:
                    continue
                trials.sort(key=lambda t: t[0], reverse=True)
                if trials[0][0] > best[str(fit_year)] + 1e-9:
                    weights, best = trials[0][1], trials[0][2]
                    improved = True
                    break
            if not improved:
                break
        return weights, best, count

    results = {}
    for fit_year, read_year in ((2023, 2024), (2024, 2023)):
        weights, best, count = climb(fit_year)
        bias = best[str(fit_year)] - best[str(read_year)]
        print(f"\nclimb on {fit_year} alone ({count} evaluations, "
              f"{read_year} never consulted)", flush=True)
        print("  " + "  ".join(f"{n} {weights[n]:.2f}" for n in NAMES6), flush=True)
        print(f"  fitted fold {fit_year}: {best[str(fit_year)]:+7.2f}      "
              f"held-out fold {read_year}: {best[str(read_year)]:+7.2f}", flush=True)
        print(f"  selection bias = {bias:+7.2f} points", flush=True)
        results[f"fit_{fit_year}"] = {
            "weights": {n: weights[n] for n in NAMES6}, "readings": best,
            "evaluations": count, "selection_bias": bias,
            "out_of_fold": best[str(read_year)]}

    transfers = all(results[f"fit_{y}"]["out_of_fold"] > 0 for y in (2023, 2024))
    print(f"\nboth climbs positive on the season they never saw: {transfers}", flush=True)
    if transfers:
        print("  -> the weights really are mis-specified for averaged components",
              flush=True)
    else:
        print("  -> each climb helps only its own fold; the weights stand, and V167's "
              "conclusion survives its bad reference", flush=True)

    finalists = {}
    for label in ("fit_2023", "fit_2024"):
        weights = {n: results[label]["weights"][n] for n in NAMES6}
        candidate = pipeline(weights)
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        finalists[label] = {"three_season": t,
                            "monthly_block_win_rate": metrics["monthly_block_win_rate"],
                            "bootstrap_2024": metrics["bootstrap_2024"]}
        print(f"  {label}: 2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V198_weights_on_averages",
        "baseline": "V196 (Public 1082.8399355688)",
        "premise": ("the blend weights were fitted in V119/V135/V139 against single-draw "
                    "components, and four components are now averages of six to nine "
                    "draws; a weight is paid for the independent part of a prediction and "
                    "averaging changes exactly that"),
        "why_v167_does_not_settle_it": ("V167 refitted weights on averaged components and "
                                        "found nothing, measured against seed 42 -- the "
                                        "luckiest draw of five on the 2024 fold"),
        "instrument": ("V179's cross-fold transfer test, because six weights is the same "
                       "kind of search that produced +4 to +7 points of selection bias on "
                       "the eight calibration parameters"),
        "results": results,
        "transfers_across_seasons": bool(transfers),
        "finalists": finalists,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
