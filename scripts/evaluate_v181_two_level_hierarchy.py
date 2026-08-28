"""V181: the hierarchy failed on a level we already knew was poison -- drop it and retry.

V180 shrank the two-strike platoon cells toward their parents instead of toward the league
and every one of 27 settings lost 2024, by 10 to 47 points. The reason is not subtle and
it was already on record.

The three-level hierarchy is

    pitcher  ->  pitcher x batter_hand  ->  pitcher x batter_hand x two_strike

and its first level is the **pitcher marginal**, which V165 measured losing out of fold at
every weight it was given (-0.98 to -11.44) and which V168 used as one of the controls
showing the platoon effect was an interaction and not a marginal. Shrinking toward a
harmful parent is worse than shrinking toward zero, and the grid agrees: the loss shrinks
monotonically as every smoothing rises, that is, as the whole hierarchy is switched off.

The repair is to start the hierarchy at the level that works. Two levels,

    pitcher x batter_hand  ->  pitcher x batter_hand x two_strike

so a thin two-strike cell falls back on that pitcher's platoon split -- the quantity V169
shipped and the leaderboard paid +7.31 for -- and no further. The pitcher marginal never
enters.

Same discipline as V180: a pre-registered point first, then a small grid, a matched
placebo throughout, and the cross-fold transfer test on anything that survives.
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
from evaluate_v180_hierarchical_shrinkage import hierarchical_correction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v181_two_level_hierarchy_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
FLAT = [(["balls_before", "strikes_before"], 0.55, 500.0),
        (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
        (["experience_bin"], 0.20, 2000.0)]
PLATOON2K_WEIGHT = 0.80
FLAT_SMOOTHING = 1500.0
PARENT = ["pitcher_id", "batter_hand"]
PRE_REGISTERED = (1000.0, 1500.0)
PARENT_LADDER = (300.0, 1000.0, 3000.0, 10000.0)
LEAF_LADDER = (500.0, 1000.0, 1500.0, 3000.0)
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
    scale = 1.0 / (sum(w for _, w, _ in FLAT) + PLATOON2K_WEIGHT)

    def calibrated(leaf=None, smoothings=None):
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in FLAT:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            if leaf is None:
                fine = segment_correction(
                    train_f, residual, valid_f,
                    ["pitcher_id", "batter_hand", "two_strike"], FLAT_SMOOTHING)
            else:
                fine = hierarchical_correction(train_f, residual, valid_f,
                                               [PARENT, leaf], smoothings)
            pieces.append(np.clip(value + scale * PLATOON2K_WEIGHT * fine, 0, 1))
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

    real = ["pitcher_id", "batter_hand", "two_strike"]
    placebo = ["pitcher_id", "batter_hand", "day_parity"]

    print(f"pre-registered point, parent {PRE_REGISTERED[0]:g} / "
          f"leaf {PRE_REGISTERED[1]:g}:", flush=True)
    prereg = {}
    for label, leaf in (("hierarchical", real), ("placebo", placebo)):
        season = points(calibrated(leaf, PRE_REGISTERED))
        prereg[label] = season
        flag = "SAFE" if min(season) >= -1e-9 and max(season) > 0 else ""
        print(f"  {label:14s} 2023 {season[1]:+7.2f}  2024 {season[2]:+7.2f}  {flag}",
              flush=True)

    print("\ngrid (2024 points), parent smoothing down the side:", flush=True)
    print(f"  {'':16s}leaf " + " ".join(f"{s:7g}" for s in LEAF_LADDER), flush=True)
    grid = {}
    for parent_s in PARENT_LADDER:
        row = []
        for leaf_s in LEAF_LADDER:
            season = points(calibrated(real, (parent_s, leaf_s)))
            grid[f"p{parent_s:g}_l{leaf_s:g}"] = season
            row.append(f"{season[2]:7.2f}")
        print(f"  parent {parent_s:7g}     {' '.join(row)}", flush=True)

    safe = {k: v for k, v in grid.items() if min(v) >= -1e-9 and max(v) > 0}
    best = max(safe, key=lambda k: safe[k][2], default=None)
    print(f"\n{len(safe)} safe of {len(grid)}; best by 2024: {best}", flush=True)

    verdict = {}
    if best is not None:
        parent_s, leaf_s = (float(v[1:]) for v in best.split("_"))
        placebo_season = points(calibrated(placebo, (parent_s, leaf_s)))
        candidate = calibrated(real, (parent_s, leaf_s))
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        print(f"  real    2023 {grid[best][1]:+7.2f}  2024 {grid[best][2]:+7.2f}  "
              f"CI low {low:+7.2f}  blk {metrics['monthly_block_win_rate']:4.0%}",
              flush=True)
        print(f"  placebo 2023 {placebo_season[1]:+7.2f}  "
              f"2024 {placebo_season[2]:+7.2f}", flush=True)
        verdict = {"best": best, "real": grid[best], "placebo": placebo_season,
                   "three_season": t,
                   "monthly_block_win_rate": metrics["monthly_block_win_rate"],
                   "bootstrap_2024": metrics["bootstrap_2024"]}

    OUTPUT.write_text(json.dumps({
        "experiment": "V181_two_level_hierarchy",
        "baseline": "V175 (Public 1067.8617513573)",
        "why_v180_failed": ("its first level was the pitcher marginal, which V165 measured "
                            "losing out of fold at every weight and which V168 used as a "
                            "control showing the platoon effect is an interaction; "
                            "shrinking toward a harmful parent is worse than shrinking "
                            "toward zero, and the loss fell monotonically as the "
                            "hierarchy was smoothed away"),
        "repair": ("start at pitcher x batter_hand -- the quantity the leaderboard paid "
                   "+7.31 for -- so the pitcher marginal never enters"),
        "pre_registered": {"smoothings": list(PRE_REGISTERED), **prereg},
        "grid": grid,
        "safe": sorted(safe),
        "verdict": verdict,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
