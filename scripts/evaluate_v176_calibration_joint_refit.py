"""V176: refit the whole calibration layer, which V175 left mis-specified.

V175 dropped a term carrying 44% of the correction mass into a layer whose other three
weights and all four smoothings were fitted without it:

    count            weight 0.55  smoothing  500   <- V98, three-component blend
    pitcher x count  weight 0.25  smoothing  300   <- V98, refined V146
    experience       weight 0.20  smoothing 2000   <- V155/V156
    platoon x 2K     weight 0.80  smoothing 1500   <- V173, the only one fitted with the rest

Every one of those was chosen against a layer that did not contain the term now doing most
of the work. This is the exact staleness that made blend-weight refits pay four times, and
it has never been asked of the calibration layer as a whole -- V146 moved the count pair,
V155 moved the experience pair, always one at a time and always with the others frozen.

The experience term is the one to watch. Its job was to correct low-experience pitchers
who are biased up (V152), and a per-pitcher term at 44% mass now reaches those same rows
directly. Its weight is allowed to reach zero.

Three parts.

**Joint refit.** Coordinate descent over all eight parameters. The objective is the
shipping rule stated exactly: a candidate is feasible only if no fold loses, and among
feasible candidates the 2024 fold decides. 2022 is +0.00 by construction for any
calibration-only change, so feasibility reduces to 2023 >= 0.

**Re-audit.** The noise-corrected recoverable-points map is redrawn on the V175 residual,
because a 44% term changes what is left.

**Fifth segment.** Whatever the re-audit puts on top is swept, with the placebo that has
now decided two submissions in a row.
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


OUTPUT = Path("artifacts/v176_calibration_joint_refit_metrics.json")
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
WEIGHT_FACTORS = (0.0, 0.5, 0.7, 0.85, 1.0, 1.18, 1.4, 2.0)
SMOOTHING_FACTORS = (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5)
ROUNDS = 4
P = 100000.0 / 0.25

CANDIDATE_FIFTH = {
    "batter_platoon": ["batter_id", "pitcher_hand"],
    "batter_two_strike": ["batter_id", "two_strike"],
    "batter_count": ["batter_id", "balls_before", "strikes_before"],
    "platoon_experience": ["pitcher_id", "batter_hand", "experience_bin"],
    "pitcher_count_hand": ["pitcher_id", "batter_hand", "balls_before"],
    "pitcher_inning": ["pitcher_id", "inning_bin"],
    "placebo_batter_parity": ["pitcher_id", "batter_hand", "batter_parity"],
    "placebo_day_parity": ["pitcher_id", "batter_hand", "day_parity"],
}


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
    frame["inning_bin"] = np.minimum(frame["inning"], 10).astype(str)
    frame["batter_parity"] = frame["batter_id"] % 2
    frame["day_parity"] = frame["game_dayofweek"] % 2
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

    def calibrated(config, extra=None):
        terms = [(SEGMENTS[k], w, s) for k, (w, s) in config.items() if w > 0]
        if extra is not None:
            terms.append(extra)
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

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    def score(config, extra=None):
        """The shipping rule as a sort key: feasible first, then the 2024 fold."""
        season = points(calibrated(config, extra))
        feasible = min(season) >= -1e-9
        return (feasible, season[2] if feasible else min(season)), season

    # ---------- 1. joint refit ----------
    config = dict(V175)
    best_key, best_season = score(config)
    print(f"start (V175)  2023 {best_season[1]:+7.2f}  2024 {best_season[2]:+7.2f}",
          flush=True)
    evaluations = 1
    for round_index in range(ROUNDS):
        improved = False
        for name in SEGMENTS:
            for axis, ladder in (("weight", WEIGHT_FACTORS),
                                 ("smoothing", SMOOTHING_FACTORS)):
                trials = []
                for factor in ladder:
                    if factor == 1.0:
                        continue
                    weight, smoothing = config[name]
                    trial = dict(config)
                    trial[name] = ((round(V175[name][0] * factor, 4), smoothing)
                                   if axis == "weight"
                                   else (weight, round(V175[name][1] * factor, 1)))
                    if trial[name] == config[name]:
                        continue
                    if sum(w for w, _ in trial.values()) <= 0:
                        continue
                    key, season = score(trial)
                    evaluations += 1
                    trials.append((key, trial, season))
                if not trials:
                    continue
                trials.sort(key=lambda t: t[0], reverse=True)
                if trials[0][0] > best_key:
                    best_key, config, best_season = trials[0][0], trials[0][1], trials[0][2]
                    improved = True
                    print(f"  r{round_index} {name:14s} {axis:9s} -> "
                          f"2023 {best_season[1]:+7.2f}  2024 {best_season[2]:+7.2f}   "
                          f"{ {k: v for k, v in config.items()} }", flush=True)
        if not improved:
            break
    refit = dict(config)
    print(f"\njoint refit after {evaluations} evaluations:", flush=True)
    for name, (weight, smoothing) in refit.items():
        was = V175[name]
        print(f"  {name:14s} weight {was[0]:5.2f} -> {weight:5.2f}   "
              f"smoothing {was[1]:6.0f} -> {smoothing:6.0f}", flush=True)
    total = sum(w for w, _ in refit.values())
    print("  normalised: " + "  ".join(
        f"{k} {w / total:.4f}" for k, (w, _) in refit.items()), flush=True)

    # ---------- 2. re-audit ----------
    residual_now = target - baseline
    audit_frame = frame.loc[order].reset_index(drop=True)

    def recoverable(columns, year):
        mask = masks[year]
        sub = audit_frame.loc[mask]
        key = sub[columns[0]].astype(str)
        for column in columns[1:]:
            key = key + "|" + sub[column].astype(str)
        grouped = pd.DataFrame({"r": residual_now[mask],
                                "k": key.to_numpy()}).groupby("k")["r"]
        n = grouped.size().to_numpy().astype(float)
        mean = grouped.mean().to_numpy()
        var = np.nan_to_num(grouped.var(ddof=1).to_numpy())
        unbiased = np.maximum(0.0, mean ** 2 - var / np.maximum(n, 1.0))
        return float(P * (n * unbiased).sum() / n.sum()), int(len(n))

    print("\nwhat is left after V175 (noise-corrected recoverable points on 2024):",
          flush=True)
    audit = {}
    for label, columns in list(CANDIDATE_FIFTH.items()) + [
            (f"{k} (in use)", v) for k, v in SEGMENTS.items()]:
        value, cells = recoverable(columns, 2024)
        audit[label] = {"2024": value, "cells": cells}
        print(f"  {label:26s} {cells:6d} cells  {value:8.1f}", flush=True)

    # ---------- 3. fifth segment on top of the refit ----------
    print("\nfifth segment on top of the refit "
          "(2024 by smoothing 500 1000 2000 4000):", flush=True)
    fifth = {}
    for label, columns in CANDIDATE_FIFTH.items():
        rows = []
        for weight in (0.10, 0.20, 0.40):
            row = []
            for smoothing in (500.0, 1000.0, 2000.0, 4000.0):
                tag = f"{label}|w{weight:.2f}|s{smoothing:g}"
                season = points(calibrated(refit, (columns, weight, smoothing)))
                fifth[tag] = season
                row.append(f"{season[2]:6.2f}")
            rows.append(f"w={weight:.2f} {' '.join(row)}")
        best = max((t for t in fifth if t.startswith(f"{label}|")),
                   key=lambda t: fifth[t][2])
        flag = "SAFE" if min(fifth[best]) >= -1e-9 and max(fifth[best]) > 0 else ""
        print(f"  {label:24s} {' | '.join(rows)}   best 2023 {fifth[best][1]:+6.2f} "
              f"2024 {fifth[best][2]:+6.2f} {flag}", flush=True)

    # ---------- finalists ----------
    def full(config, extra=None, label=""):
        candidate = calibrated(config, extra)
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        safe = metrics["min_season_points"] >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:34s} 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)
        return metrics

    print("\nfinalists:", flush=True)
    results = {"joint_refit": full(refit, None, "joint refit")}
    ranked = sorted((t for t in fifth
                     if min(fifth[t]) >= -1e-9 and max(fifth[t]) > 0
                     and not t.startswith("placebo")),
                    key=lambda t: fifth[t][2], reverse=True)
    for tag in ranked[:3]:
        label, weight, smoothing = tag.split("|")
        results[f"refit+{tag}"] = full(
            refit, (CANDIDATE_FIFTH[label], float(weight[1:]), float(smoothing[1:])),
            f"refit + {tag}")

    OUTPUT.write_text(json.dumps({
        "experiment": "V176_calibration_joint_refit",
        "baseline": "V175 (Public 1067.8617513573)",
        "premise": ("V175 added a term carrying 44% of the correction mass to a layer "
                    "whose other three weights and all four smoothings were fitted "
                    "without it; the layer has never been refitted as a whole"),
        "v175_config": {k: list(v) for k, v in V175.items()},
        "refit_config": {k: list(v) for k, v in refit.items()},
        "refit_season_points": best_season,
        "evaluations": evaluations,
        "audit_after_v175": audit,
        "fifth_segment_grid": fifth,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in results.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
