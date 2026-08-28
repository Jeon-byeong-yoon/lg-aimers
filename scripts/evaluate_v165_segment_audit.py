"""V165: audit every available segment for structure the V161 calibration still leaves.

The calibration layer has produced two of the last three gains -- V152's experience
segment shipped in V156 (+1.33) -- but the segments in it were chosen one at a time, by
hypothesis. Nobody has ever swept the whole column list and asked which segments still
carry bias after the current layer runs. V146 audited pitcher and pitcher-by-count
oracles; V152 audited experience inside `game_type`. Everything else is unmeasured.

For each candidate segment this reports the points recoverable by knowing that segment's
offsets exactly, **corrected for noise**. A group's mean residual squares up to

    E[mean_g^2] = true_g^2 + var_g / n_g

so the raw oracle is inflated by one variance per cell, badly for fine segments. The
estimate subtracts it:

    points = 400000 * sum_g (n_g / N) * max(0, mean_g^2 - var_g / n_g)

which is unbiased for the real structure and goes to zero for a segment that is only
noise. Cheap: everything runs on cached component predictions.

This is a map, not a candidate list. A segment only becomes a candidate if the structure
survives being fitted out-of-fold, which is the second half of the script: for the top
segments, fit the offsets on prior seasons only and score the fold under the usual rule.
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


OUTPUT = Path("artifacts/v165_segment_audit_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
P = 100000.0 / 0.25


def annotate(frame):
    """Derived segment columns. Every one is a property of the row alone."""
    out = frame.copy()
    out["experience_bin"] = pd.cut(out["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    out["batter_experience_bin"] = pd.cut(
        out["asof_batter_n"], [-1, 100, 400, 1200, 3000, 8000, np.inf],
        labels=["b0-100", "b100-400", "b400-1.2k", "b1.2k-3k", "b3k-8k", "b8k+"]).astype(str)
    out["inning_bin"] = np.minimum(out["inning"], 10).astype(str)
    out["li_bin"] = pd.cut(out["li"], [-np.inf, 0.4, 0.8, 1.2, 2.0, 3.5, np.inf],
                           labels=["l0", "l1", "l2", "l3", "l4", "l5"]).astype(str)
    out["score_bin"] = pd.cut(out["score_diff_pitcher_team"],
                              [-np.inf, -6, -3, -1, 0, 1, 3, 6, np.inf],
                              labels=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]).astype(str)
    out["hands"] = out["pitcher_hand"].astype(str) + "_" + out["batter_hand"].astype(str)
    out["fastball_bin"] = pd.cut(out["asof_pitcher_fastball_rate"],
                                 [-np.inf, .3, .4, .5, .6, .7, np.inf],
                                 labels=["f0", "f1", "f2", "f3", "f4", "f5"]).astype(str)
    out["count_state"] = (out["balls_before"].astype(str) + "-"
                          + out["strikes_before"].astype(str))
    return out


SEGMENTS = {
    "count (in use)": ["balls_before", "strikes_before"],
    "pitcher x count (in use)": ["pitcher_id", "balls_before", "strikes_before"],
    "experience (in use)": ["experience_bin"],
    "game_type (V151, lost)": ["game_type"],
    "inning": ["inning_bin"],
    "outs": ["outs_before"],
    "runners": ["num_runners_on"],
    "base_state": ["base_state"],
    "leverage": ["li_bin"],
    "score_gap": ["score_bin"],
    "handedness": ["hands"],
    "game_month": ["game_month"],
    "dayofweek": ["game_dayofweek"],
    "top_bottom": ["top_bottom"],
    "batter_experience": ["batter_experience_bin"],
    "pitch_mix": ["fastball_bin"],
    "pitcher_team": ["pitcher_team_id"],
    "batter_team": ["batter_team_id"],
    "batter_id": ["batter_id"],
    "pitcher_id": ["pitcher_id"],
    "pitcher x strikes": ["pitcher_id", "strikes_before"],
    "experience x count": ["experience_bin", "count_state"],
    "experience x game_type": ["experience_bin", "game_type"],
    "count x handedness": ["count_state", "hands"],
    "count x inning": ["count_state", "inning_bin"],
    "experience x leverage": ["experience_bin", "li_bin"],
    "pitcher x experience-free": ["pitcher_id", "hands"],
}


def recoverable(frame, residual, columns):
    """Noise-corrected points a perfect segment offset would recover, in-fold."""
    key = frame[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "|" + frame[column].astype(str)
    grouped = pd.DataFrame({"r": residual, "k": key.to_numpy()}).groupby("k")["r"]
    n = grouped.size().to_numpy().astype(float)
    mean = grouped.mean().to_numpy()
    var = grouped.var(ddof=1).to_numpy()
    var = np.where(np.isfinite(var), var, 0.0)
    unbiased = np.maximum(0.0, mean ** 2 - var / np.maximum(n, 1.0))
    return float(P * (n * unbiased).sum() / n.sum()), int(len(n)), float(n.mean())


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
    calibration_frame = annotate(raw_frame)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()

    weights = tuple(BASE[n] for n in NAMES6)
    blend = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
             for year in YEARS}

    def calibrated(extra=None):
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in history])
            train_f = calibration_frame.loc[index]
            valid_f = calibration_frame.loc[oof[str(year)]["row_index"]]
            value = (blend[year] + residual.mean()
                     + W_COUNT * segment_correction(
                         train_f, residual, valid_f,
                         ["balls_before", "strikes_before"], 500)
                     + W_PITCHER_COUNT * segment_correction(
                         train_f, residual, valid_f,
                         ["pitcher_id", "balls_before", "strikes_before"], 300)
                     + W_EXPERIENCE * segment_correction(
                         train_f, residual, valid_f, ["experience_bin"],
                         EXPERIENCE_SMOOTHING))
            if extra is not None:
                columns, weight, smoothing = extra
                value = value + weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = calibrated()
    target = validation_frame["target"].to_numpy().astype(float)
    residual = target - baseline
    audit_frame = calibration_frame.loc[order].reset_index(drop=True)

    print(f"{'segment':30s} {'cells':>7s} {'n/cell':>8s} "
          f"{'2022':>8s} {'2023':>8s} {'2024':>8s}")
    audit = {}
    for label, columns in SEGMENTS.items():
        row = {}
        for year in YEARS:
            mask = season_of == year
            points, cells, per = recoverable(audit_frame.loc[mask], residual[mask], columns)
            row[str(year)] = points
            if year == 2024:
                row["cells_2024"], row["rows_per_cell_2024"] = cells, per
        audit[label] = row
        print(f"{label:30s} {row['cells_2024']:7d} {row['rows_per_cell_2024']:8.0f} "
              f"{row['2022']:8.1f} {row['2023']:8.1f} {row['2024']:8.1f}", flush=True)

    in_use = {"count (in use)", "pitcher x count (in use)", "experience (in use)",
              "game_type (V151, lost)"}
    ranked = sorted((l for l in audit if l not in in_use),
                    key=lambda l: min(audit[l]["2022"], audit[l]["2023"], audit[l]["2024"]),
                    reverse=True)
    print("\nranked by weakest season (unused segments only):", flush=True)
    for label in ranked[:8]:
        row = audit[label]
        print(f"  {label:30s} min {min(row['2022'], row['2023'], row['2024']):8.1f}",
              flush=True)

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

    print("\nout-of-fold test of the top segments (offsets fitted on prior seasons only):",
          flush=True)
    results = {}
    for label in ranked[:6]:
        for weight in (0.05, 0.10, 0.20):
            for smoothing in (500.0, 2000.0):
                tag = f"{label}|w{weight:.2f}|s{smoothing:g}"
                metrics = evaluate(calibrated((SEGMENTS[label], weight, smoothing)))
                results[tag] = metrics
                t = metrics["three_season"]
                safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
                print(f"  {tag:46s} min {metrics['min_season_points']:+7.2f}  "
                      f"avg {t['average_points']:+7.2f}  "
                      f"2022 {t['season_points'][0]:+7.2f}  "
                      f"2023 {t['season_points'][1]:+7.2f}  "
                      f"2024 {t['season_points'][2]:+7.2f}  {'SAFE' if safe else ''}",
                      flush=True)

    def safe(tag):
        t = results[tag]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [t for t in results if safe(t)]
    promoted = max(survivors,
                   key=lambda t: (results[t]["min_season_points"],
                                  results[t]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V165_segment_audit",
        "baseline": "V161 (Public 1053.2326413884)",
        "statistic": ("400000 * sum_g (n_g/N) * max(0, mean_g^2 - var_g/n_g); the "
                      "subtraction removes the one-variance-per-cell inflation that makes "
                      "fine segments look informative when they are noise"),
        "audit": audit,
        "ranked_unused": ranked,
        "results": {k: {"three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsafe = {len(survivors)} of {len(results)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
