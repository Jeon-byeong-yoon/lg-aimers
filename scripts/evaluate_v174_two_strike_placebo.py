"""V174: one placebo before shipping the two-strike split.

V173 put the optimum at `pitcher_id x batter_hand x two_strike`, weight 0.80, smoothing
1500, with the coarse platoon term removed entirely -- 2024 +4.65 on a plateau whose eight
neighbours are all at or above +3.80.

But the shipped geometry is very different from V169's. The fine term carries 0.80/1.80 =
44% of the correction mass where the platoon term carried 17%, and smoothing rose from
1,000 to 1,500 against cells that are half the size. Two things changed at once: the split
became three-way, and the term became heavy and heavily smoothed. Only the first is a
claim about baseball.

So the same control V168 used runs again at the shipped setting. Split each
pitcher-by-handedness cell by a meaningless row-derived bit instead of the two-strike
flag -- batter-id parity and day-of-week parity -- which reproduces the cell count, the
cell size and the shrinkage exactly, with no content. If those gain too, V173 found a
geometry and not an effect, and the honest move is to keep V169.

V170 is why this is worth two minutes: V166's arm looked safe on all three folds and was a
seed draw. A control that costs nothing is cheaper than a submission that costs a day.
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


OUTPUT = Path("artifacts/v174_two_strike_placebo_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
PLATOON = ["pitcher_id", "batter_hand"]
SHIPPED = {"weight": 0.80, "smoothing": 1500.0}
SPLITS = {"two_strike": "two_strike", "placebo_batter_parity": "batter_parity",
          "placebo_day_parity": "day_parity", "placebo_inning_parity": "inning_parity"}
WEIGHTS = (0.40, 0.65, 0.80, 1.00)
SMOOTHINGS = (1000.0, 1500.0, 2000.0)
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
    frame["two_strike"] = (frame["strikes_before"] == 2).astype(int)
    frame["batter_parity"] = frame["batter_id"] % 2
    frame["day_parity"] = frame["game_dayofweek"] % 2
    frame["inning_parity"] = frame["inning"] % 2
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

    def calibrated(terms):
        scale = 1.0 / sum(weight for _, weight, _ in terms)
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
                if weight == 0.0:
                    continue
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def build(split, weight, smoothing, coarse=0.0):
        return [(["balls_before", "strikes_before"], W_COUNT, COUNT_SMOOTHING),
                (["pitcher_id", "balls_before", "strikes_before"], W_PITCHER_COUNT,
                 PITCHER_COUNT_SMOOTHING),
                (["experience_bin"], W_EXPERIENCE, EXPERIENCE_SMOOTHING),
                (PLATOON, coarse, 1000.0),
                (PLATOON + [split], weight, smoothing)]

    baseline = calibrated(build("two_strike", 0.0, 1500.0, coarse=0.20))   # V169
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    valid = valid_frames[2024]
    print("2024 fold cell geometry -- the placebos must match:", flush=True)
    for label, split in SPLITS.items():
        key = (valid["pitcher_id"].astype(str) + "|" + valid["batter_hand"].astype(str)
               + "|" + valid[split].astype(str))
        print(f"  {label:24s} {key.nunique():6d} cells, "
              f"{len(key) / key.nunique():6.0f} rows each", flush=True)

    grid, results = {}, {}
    print(f"\n2024 points across the plateau "
          f"(smoothing {' '.join(f'{s:6g}' for s in SMOOTHINGS)}):", flush=True)
    for label, split in SPLITS.items():
        print(f"  {label}", flush=True)
        for weight in WEIGHTS:
            row = []
            for smoothing in SMOOTHINGS:
                tag = f"{label}|w{weight:.2f}|s{smoothing:g}"
                season = points(calibrated(build(split, weight, smoothing)))
                grid[tag] = {"season_points": season}
                row.append(f"{season[2]:6.2f}")
            print(f"    w={weight:.2f}   {' '.join(row)}", flush=True)

    print(f"\nat the shipped setting w={SHIPPED['weight']:.2f} "
          f"s={SHIPPED['smoothing']:g}:", flush=True)
    for label, split in SPLITS.items():
        candidate = calibrated(build(split, SHIPPED["weight"], SHIPPED["smoothing"]))
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        results[label] = metrics
        flag = "SAFE" if metrics["min_season_points"] >= -1e-9 else ""
        print(f"  {label:24s} 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}  {flag}", flush=True)

    real = results["two_strike"]["three_season"]["season_points"][2]
    worst_placebo = max(results[l]["three_season"]["season_points"][2]
                        for l in SPLITS if l != "two_strike")
    verdict = real > 0 and worst_placebo < 0.5 * real
    print(f"\n  two_strike {real:+.2f} against the best placebo {worst_placebo:+.2f}"
          f"  -> {'content, not geometry' if verdict else 'GEOMETRY -- do not ship'}",
          flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V174_two_strike_placebo",
        "baseline": "V169 (Public 1060.5433916316)",
        "shipped_setting": SHIPPED,
        "why": ("V173's optimum changed two things at once -- the split became three-way, "
                "and the term went from 17% to 44% of correction mass at heavier "
                "smoothing. Only the first is a claim about baseball, so the placebos "
                "reproduce the geometry exactly and carry no content."),
        "grid": grid,
        "at_shipped_setting": {
            k: {"three_season": v["three_season"],
                "min_season_points": v["min_season_points"],
                "monthly_block_win_rate": v["monthly_block_win_rate"],
                "bootstrap_2024": v["bootstrap_2024"]} for k, v in results.items()},
        "verdict_content_not_geometry": bool(verdict),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
