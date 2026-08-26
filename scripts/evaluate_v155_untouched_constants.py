"""V155: three constants in the deployed path that no experiment has ever varied.

V154 is the current best at 1050.8513. Going back through the pipeline for constants
that were set once and never revisited turns up three.

`RELIABILITY_SCALE = 150`. The drift term is `weight * delta * reliability` where
`reliability = inside_n / (inside_n + reliability_scale)`, so this number decides how many
current-season pitches a pitcher needs before the drift signal is believed -- 150 puts the
half-trust point there. V127 gridded the drift weight and its feature shrinkage across 28
combinations and left this one alone; the calibration bundle has carried
`drift_reliability_scale: 150.0` since V92 chose it a priori.

The experience segment's smoothing, 2000, and its bin edges. Both were picked when the
segment was introduced in V154 a few hours ago, on no evidence at all -- the edges
(200 / 1k / 3k / 8k) came from wanting readable diagnostic output, not from fitting.

None of these need a model retrained. The drift term is recomputed from the
reconstruction, and the segment is a lookup.

Ranked by the weakest season. The rule as the leaderboard has now settled it, across six
submissions: the minimum season predicts the *sign* without exception (6/6), nothing
predicts the magnitude, and a candidate is worth submitting when no fold shows a loss --
V144 (-1.86 on 2022) and V151 (-32.36 on 2023) both overrode folds carrying real losses
and both failed, while V154 cleared every fold and gained despite missing two magnitude
thresholds.
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
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v155_untouched_constants_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
DRIFT_RELIABILITY = 150.0
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.65, 0.25, 0.10
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 200, 1000, 3000, 8000, np.inf]
LABELS = ["0-200", "200-1k", "1k-3k", "3k-8k", "8k+"]
FINE_EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
FINE_LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
RELIABILITY_GRID = (50.0, 100.0, 150.0, 250.0, 400.0, 700.0)
DRIFT_WEIGHT_GRID = (0.08, 0.10, 0.13)
SEGMENT_SMOOTHING_GRID = (500.0, 1000.0, 2000.0, 4000.0)
SEGMENT_WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20)
P = 100000.0 / 0.25


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id")
    raw_frame["experience_bin"] = pd.cut(
        raw_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    raw_frame["experience_fine"] = pd.cut(
        raw_frame["asof_pitcher_n"], FINE_EDGES, labels=FINE_LABELS).astype(str)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v153_projected_prior_predictions.joblib")["catboost"]["projected"]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]["latent8"]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization,
    }
    weights = tuple(BASE[n] for n in NAMES6)
    raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
           for year in YEARS}
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    # One reconstruction per reliability scale; the drift term is the only consumer.
    terms = {}
    plain = raw_frame.drop(columns=["experience_bin", "experience_fine"])
    for scale in RELIABILITY_GRID:
        started = time.time()
        block = add_training_inseason_features(
            plain, shrinkage=DRIFT_SHRINKAGE, reliability_scale=scale).loc[order]
        terms[scale] = drift_correction(block, 1.0)
        print(f"  reliability {scale:6.0f}: term mean {terms[scale].mean():+.6f}, "
              f"sd {terms[scale].std():.6f}  [{time.time() - started:.0f}s]", flush=True)
        del block
        gc.collect()

    cache = {}

    def correction(year, columns, smoothing):
        key = (year, tuple(columns), smoothing)
        if key not in cache:
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            cache[key] = segment_correction(
                raw_frame.loc[index], residual,
                raw_frame.loc[oof[str(year)]["row_index"]], columns, smoothing)
        return cache[key]

    def shift(year):
        history = [h for h in YEARS if h < year]
        targets = np.concatenate([oof[str(h)]["target"].astype(float)
                                  for h in history])
        return float((targets - np.concatenate([raw[h] for h in history])).mean())

    def pipeline(segment_column, segment_smoothing, segment_weight,
                 drift_weight, reliability):
        count_weight = round(1.0 - W_PITCHER_COUNT - segment_weight, 6)
        assert count_weight > 0
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            total = (raw[year] + shift(year)
                     + count_weight * correction(
                         year, ["balls_before", "strikes_before"], 500.0)
                     + W_PITCHER_COUNT * correction(
                         year, ["pitcher_id", "balls_before", "strikes_before"], 300.0))
            if segment_weight:
                total = total + segment_weight * correction(
                    year, [segment_column], segment_smoothing)
            pieces.append(np.clip(total, 0, 1))
        return np.clip(np.concatenate(pieces) + drift_weight * terms[reliability], 0, 1)

    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()
    baseline = pipeline("experience_bin", EXPERIENCE_SMOOTHING, W_EXPERIENCE,
                        DRIFT_WEIGHT, DRIFT_RELIABILITY)
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

    def show(label, m):
        t = m["three_season"]
        safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:30s} min {m['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+7.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blk {m['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    results = {}
    print("\ndrift reliability scale x weight (segment held at V154's values):",
          flush=True)
    for scale in RELIABILITY_GRID:
        for drift_weight in DRIFT_WEIGHT_GRID:
            if scale == DRIFT_RELIABILITY and drift_weight == DRIFT_WEIGHT:
                continue
            label = f"rel{scale:.0f}_dw{drift_weight:.2f}"
            results[label] = evaluate(pipeline(
                "experience_bin", EXPERIENCE_SMOOTHING, W_EXPERIENCE,
                drift_weight, scale))
            results[label]["kind"] = "drift"
            show(label, results[label])

    print("\nexperience segment smoothing x weight (drift held at V154's values):",
          flush=True)
    for smoothing in SEGMENT_SMOOTHING_GRID:
        for segment_weight in SEGMENT_WEIGHT_GRID:
            if smoothing == EXPERIENCE_SMOOTHING and segment_weight == W_EXPERIENCE:
                continue
            label = f"seg_s{smoothing:.0f}_w{segment_weight:.2f}"
            results[label] = evaluate(pipeline(
                "experience_bin", smoothing, segment_weight,
                DRIFT_WEIGHT, DRIFT_RELIABILITY))
            results[label]["kind"] = "segment"
            show(label, results[label])

    print("\nfiner experience bins (7 instead of 5):", flush=True)
    for smoothing in (500.0, 1000.0, 2000.0):
        for segment_weight in (0.10, 0.15, 0.20):
            label = f"fine_s{smoothing:.0f}_w{segment_weight:.2f}"
            results[label] = evaluate(pipeline(
                "experience_fine", smoothing, segment_weight,
                DRIFT_WEIGHT, DRIFT_RELIABILITY))
            results[label]["kind"] = "fine_bins"
            show(label, results[label])

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors, key=lambda l: results[l]["min_season_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V155_untouched_constants",
        "baseline": "V154 (Public 1050.8512921822)",
        "constants_examined": {
            "drift_reliability_scale": {
                "value": DRIFT_RELIABILITY,
                "note": ("half-trust point of the drift signal in current-season "
                         "pitches; V127 gridded the drift weight and shrinkage across 28 "
                         "combinations and never touched this")},
            "experience_segment_smoothing": {
                "value": EXPERIENCE_SMOOTHING,
                "note": "picked when the segment was introduced in V154, on no evidence"},
            "experience_bin_edges": {
                "value": [200, 1000, 3000, 8000],
                "note": "chosen for readable diagnostics, not fitted"},
        },
        "selection_rule": (
            "No fold may show a loss; among survivors the weakest season decides. Six "
            "submissions: the minimum predicts the sign 6/6, nothing predicts the "
            "magnitude, and V154 gained while missing two magnitude thresholds whereas "
            "V144 and V151 overrode folds carrying real losses and both failed."
        ),
        "results": {k: {"kind": v["kind"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "row_independent": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsafe (no fold loses) = {len(survivors)} of {len(results)}  "
          f"promoted={promoted}")
    if promoted:
        show("PROMOTED " + promoted, results[promoted])
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
