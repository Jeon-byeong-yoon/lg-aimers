"""V179: measure the selection bias with a whole season, not three months of one.

V177 rejected the joint calibration refit on a within-2024 holdout. V178 then scored that
holdout against the two changes whose leaderboard answers are known and it failed:

    V169 over V161   holdout -1.14   leaderboard +7.31
    V175 over V169   holdout +5.82   leaderboard +7.32

A device that rejects the biggest gain the project has recorded is not a device. The
month-by-month readings say why -- single months swing 25 points on 2024 -- so three
months and 77,000 rows is simply too small to resolve effects of this size, and V177's
verdict cannot stand on it.

The suspicion it was testing is still real. V176 ran 195 coordinate-descent steps and V177
ran 381, all with the 2024 fold as the objective, and both landed on `pitcher_count`
weight exactly 0 and `count` smoothing 75 -- values that look like fitted noise rather
than a refit.

So measure the selection bias with the one instrument this project has that is neither
small nor shared: **the other season.** Climb the eight parameters against 2023 alone,
with 2024 never consulted, and then read 2024. Climb against 2024 alone and read 2023. The
in-fold gain minus the out-of-fold gain *is* the selection bias, in points, measured rather
than argued.

If a climb transfers across seasons the surface has real structure and the refit is worth
shipping. If each climb only helps the fold it was fitted on, the parameters are being
fitted to fold-specific noise and V175's layer stands.
"""

import json
import sys
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


OUTPUT = Path("artifacts/v179_cross_fold_transfer_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SEGMENTS = {
    "count": ["balls_before", "strikes_before"],
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
    "experience": ["experience_bin"],
    "platoon2k": ["pitcher_id", "batter_hand", "two_strike"],
}
V175 = {"count": (0.55, 500.0), "pitcher_count": (0.25, 300.0),
        "experience": (0.20, 2000.0), "platoon2k": (0.80, 1500.0)}
WEIGHT_FACTORS = (0.0, 0.35, 0.5, 0.7, 0.85, 1.0, 1.18, 1.4, 2.0, 3.0)
SMOOTHING_FACTORS = (0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0)
ROUNDS = 6
P = 100000.0 / 0.25


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][20.0],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "network": v160["network"][FEATURE_RELIABILITY],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        "factorization": v160["factorization"][FEATURE_RELIABILITY],
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
    blend = {year: sum(BASE[n] * parts[n][str(year)] for n in NAMES6) for year in YEARS}

    def calibrated(config):
        terms = [(SEGMENTS[k], w, s) for k, (w, s) in config.items() if w > 0]
        scale = 1.0 / sum(w for _, w, _ in terms)
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in terms:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = calibrated(V175)
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def readings(config):
        error = (calibrated(config) - target) ** 2
        return {str(y): float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS}

    def climb(fit_year):
        """Coordinate descent maximising one season, with the other never consulted."""
        config = dict(V175)
        best = readings(config)
        count = 1
        for _ in range(ROUNDS):
            improved = False
            for name in SEGMENTS:
                for axis, ladder in (("weight", WEIGHT_FACTORS),
                                     ("smoothing", SMOOTHING_FACTORS)):
                    trials = []
                    for factor in ladder:
                        weight, smoothing = config[name]
                        trial = dict(config)
                        trial[name] = ((round(V175[name][0] * factor, 4), smoothing)
                                       if axis == "weight"
                                       else (weight, round(V175[name][1] * factor, 1)))
                        if trial[name] == config[name]:
                            continue
                        if sum(w for w, _ in trial.values()) <= 0:
                            continue
                        r = readings(trial)
                        count += 1
                        trials.append((r[str(fit_year)], trial, r))
                    if not trials:
                        continue
                    trials.sort(key=lambda t: t[0], reverse=True)
                    if trials[0][0] > best[str(fit_year)] + 1e-9:
                        config, best = trials[0][1], trials[0][2]
                        improved = True
            if not improved:
                break
        return config, best, count

    results = {}
    for fit_year, read_year in ((2023, 2024), (2024, 2023)):
        config, best, count = climb(fit_year)
        bias = best[str(fit_year)] - best[str(read_year)]
        print(f"\nclimb on {fit_year} alone ({count} evaluations, {read_year} never seen)",
              flush=True)
        for name, (weight, smoothing) in config.items():
            print(f"  {name:14s} weight {V175[name][0]:5.2f} -> {weight:6.3f}   "
                  f"smoothing {V175[name][1]:6.0f} -> {smoothing:7.1f}", flush=True)
        print(f"  fitted fold {fit_year}: {best[str(fit_year)]:+7.2f}      "
              f"held-out fold {read_year}: {best[str(read_year)]:+7.2f}", flush=True)
        print(f"  selection bias = {bias:+7.2f} points", flush=True)
        results[f"fit_{fit_year}"] = {
            "config": {k: list(v) for k, v in config.items()},
            "readings": best, "evaluations": count,
            "in_fold": best[str(fit_year)], "out_of_fold": best[str(read_year)],
            "selection_bias": bias}

    transfers = all(results[f"fit_{y}"]["out_of_fold"] > 0 for y in (2023, 2024))
    print(f"\nboth climbs positive on the season they never saw: {transfers}", flush=True)

    print("\nfull metrics for both climbed configurations:", flush=True)
    finalists = {}
    for label in ("fit_2023", "fit_2024"):
        config = {k: tuple(v) for k, v in results[label]["config"].items()}
        candidate = calibrated(config)
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        finalists[label] = metrics
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        safe = metrics["min_season_points"] >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:10s} 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V179_cross_fold_transfer",
        "baseline": "V175 (Public 1067.8617513573)",
        "why_not_the_holdout": ("V178 scored V177's within-2024 holdout against two known "
                                "answers and it rejected V169, which returned +7.31; "
                                "single months swing 25 points on that fold so 77,000 rows "
                                "cannot resolve effects of this size"),
        "design": ("climb the eight parameters against one season with the other never "
                   "consulted, then read the other; in-fold minus out-of-fold is the "
                   "selection bias in points"),
        "results": results,
        "transfers_across_seasons": bool(transfers),
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
