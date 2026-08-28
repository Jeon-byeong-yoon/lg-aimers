"""V180: shrink the platoon cells toward their parent instead of toward zero.

V176-V179 established what the calibration layer will and will not accept. Adding one
structural term with real content, verified against a matched placebo, paid +7.31 and
+7.32. Tuning the eight existing parameters pays nothing: V179 climbed them against one
season with the other never consulted and neither climb transferred --

    climb on 2023   fitted +3.70   held-out 2024  -3.15   selection bias +6.84
    climb on 2024   fitted +3.77   held-out 2023  -0.44   selection bias +4.21

-- so the free-parameter count, not the idea, is what decides. This experiment is one
structural change with the parameter count held down.

Every lookup in the layer estimates a cell the same way,

    correction(cell) = sum(cell) / (n(cell) + smoothing)

which shrinks a thin cell toward **zero**: the league. For `pitcher x batter_hand x
two_strike` that is the wrong destination. A pitcher with forty two-strike pitches against
left-handers should fall back on *his own* platoon split, and then on *his own* overall
residual, before he falls back on the league. Zero is three levels away and the estimator
jumps straight there.

The hierarchical form shrinks each level toward the level above:

    e0     = 0                                              the league
    e1(p)  = (sum(p)   + s1 * e0)     / (n(p)   + s1)       the pitcher
    e2(p,h)= (sum(p,h) + s2 * e1(p))  / (n(p,h) + s2)       his platoon split
    e3(...)= (sum(...) + s3 * e2(p,h))/ (n(...) + s3)       his two-strike approach

This is ordinary partial pooling and it costs one extra smoothing constant over the flat
term, not eight. It also ships unchanged: the hierarchy is a training-time estimation
detail and the artifact is still one flat table, so `v175_script.py` needs no edit.

Discipline, given what V179 just showed. One pre-registered point is read first -- all
three smoothings at V175's 1500. Then a 3x3x3 grid. Then whatever wins must pass the
cross-fold transfer test before it is allowed near a submission, and a matched placebo
hierarchy (the same three levels with the last split replaced by a meaningless bit) runs
alongside.
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


OUTPUT = Path("artifacts/v180_hierarchical_shrinkage_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
COUNT = (["balls_before", "strikes_before"], 0.55, 500.0)
PITCHER_COUNT = (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0)
EXPERIENCE = (["experience_bin"], 0.20, 2000.0)
PLATOON2K_WEIGHT = 0.80
FLAT_SMOOTHING = 1500.0
LEVELS = [["pitcher_id"], ["pitcher_id", "batter_hand"]]
PRE_REGISTERED = (1500.0, 1500.0, 1500.0)
LADDER = (500.0, 1500.0, 4000.0)
P = 100000.0 / 0.25


def hierarchical_correction(train_frame, train_residual, valid_frame, levels, smoothings):
    """Partial pooling: each level shrinks toward the level above, not toward zero."""
    centered = train_residual - train_residual.mean()
    parent_train = np.zeros(len(train_frame), dtype=float)
    parent_valid = np.zeros(len(valid_frame), dtype=float)
    for columns, smoothing in zip(levels, smoothings):
        work = train_frame[columns].copy()
        work["_residual"] = centered - parent_train
        stats = work.groupby(columns, dropna=False, sort=False)["_residual"].agg(
            ["sum", "count"])
        stats["_estimate"] = stats["sum"] / (stats["count"] + smoothing)
        table = stats[["_estimate"]].reset_index()
        for frame, parent in ((train_frame, "train"), (valid_frame, "valid")):
            merged = frame[columns].merge(table, how="left", on=columns, sort=False)
            values = merged["_estimate"].fillna(0.0).to_numpy()
            if parent == "train":
                parent_train = parent_train + values
            else:
                parent_valid = parent_valid + values
    return parent_valid


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
    frame["day_parity"] = (frame["game_dayofweek"] % 2).astype("int64")
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

    def calibrated(leaf=None, smoothings=None):
        """V175's layer; `leaf` replaces the flat platoon term with a hierarchy."""
        flat = [COUNT, PITCHER_COUNT, EXPERIENCE]
        total = sum(w for _, w, _ in flat) + PLATOON2K_WEIGHT
        scale = 1.0 / total
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in flat:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            if leaf is None:
                fine = segment_correction(
                    train_f, residual, valid_f,
                    ["pitcher_id", "batter_hand", "two_strike"], FLAT_SMOOTHING)
            else:
                fine = hierarchical_correction(
                    train_f, residual, valid_f, LEVELS + [leaf], smoothings)
            value = value + scale * PLATOON2K_WEIGHT * fine
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = calibrated()
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    control = float(np.abs(baseline - calibrated()).max())
    print(f"control: the flat baseline reproduces itself to {control:.3e}", flush=True)

    real_leaf = ["pitcher_id", "batter_hand", "two_strike"]
    placebo_leaf = ["pitcher_id", "batter_hand", "day_parity"]

    print(f"\npre-registered point, all three smoothings at {PRE_REGISTERED[0]:g}:",
          flush=True)
    results = {}
    for label, leaf in (("hierarchical", real_leaf), ("placebo", placebo_leaf)):
        season = points(calibrated(leaf, PRE_REGISTERED))
        results[f"prereg_{label}"] = season
        flag = "SAFE" if min(season) >= -1e-9 and max(season) > 0 else ""
        print(f"  {label:14s} 2023 {season[1]:+7.2f}  2024 {season[2]:+7.2f}  {flag}",
              flush=True)

    print("\n3x3x3 grid on the real hierarchy (2024 points):", flush=True)
    grid = {}
    for s1 in LADDER:
        for s2 in LADDER:
            row = []
            for s3 in LADDER:
                season = points(calibrated(real_leaf, (s1, s2, s3)))
                grid[f"s{s1:g}_{s2:g}_{s3:g}"] = season
                row.append(f"{season[2]:6.2f}")
            print(f"  pitcher {s1:5g}  platoon {s2:5g}   by leaf "
                  f"{'  '.join(f'{s:g}' for s in LADDER)}: {' '.join(row)}", flush=True)

    safe = {k: v for k, v in grid.items() if min(v) >= -1e-9 and max(v) > 0}
    best = max(safe, key=lambda k: safe[k][2], default=None)
    print(f"\n{len(safe)} safe of {len(grid)}; best by 2024: {best}", flush=True)

    verdict = {}
    if best is not None:
        s1, s2, s3 = (float(v) for v in best[1:].split("_"))
        placebo_season = points(calibrated(placebo_leaf, (s1, s2, s3)))
        print(f"  real    2023 {grid[best][1]:+7.2f}  2024 {grid[best][2]:+7.2f}",
              flush=True)
        print(f"  placebo 2023 {placebo_season[1]:+7.2f}  "
              f"2024 {placebo_season[2]:+7.2f}", flush=True)
        candidate = calibrated(real_leaf, (s1, s2, s3))
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        print(f"  full metrics: 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)
        verdict = {"best": best, "real": grid[best], "placebo": placebo_season,
                   "three_season": t, "monthly_block_win_rate":
                       metrics["monthly_block_win_rate"],
                   "bootstrap_2024": metrics["bootstrap_2024"]}

    OUTPUT.write_text(json.dumps({
        "experiment": "V180_hierarchical_shrinkage",
        "baseline": "V175 (Public 1067.8617513573)",
        "idea": ("every lookup shrinks a thin cell toward zero, the league; a pitcher's "
                 "two-strike platoon cell should fall back on his own platoon split and "
                 "then his own residual first. One extra constant, not eight."),
        "ships_unchanged": ("the hierarchy is a training-time estimation detail and the "
                            "artifact is still one flat table, so the inference script "
                            "needs no edit"),
        "control_max_abs_difference": control,
        "pre_registered": {"smoothings": list(PRE_REGISTERED),
                           "hierarchical": results.get("prereg_hierarchical"),
                           "placebo": results.get("prereg_placebo")},
        "grid": grid,
        "safe": sorted(safe),
        "verdict": verdict,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
