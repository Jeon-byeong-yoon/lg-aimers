"""V168: the calibration has no per-pitcher platoon term, and that is where the map points.

V165 swept every segment available and measured, per fold, the points a perfect set of
offsets would recover -- corrected for the one-variance-per-cell inflation that makes fine
segments look informative when they are noise. Read the 2024 column, because every 2023
reading is contaminated by the `game_type = F` break (even `top_bottom`, which cannot
possibly matter, shows 93 there):

    segment                       cells   n/cell     2024
    pitcher x handedness            777      326    374.4     <- largest unused
    pitcher x strikes              1168      217    325.8
    pitcher_id                      391      648    139.3
    batter_id                       424      598     97.8
    count x inning                  120     2113     25.9
    ... every other segment                          < 16

Out of fold the ordering inverts for two of them: `pitcher x strikes` loses 2024 at every
weight, and `pitcher_id` alone loses it badly (-0.98 to -11.44). Only `pitcher x
handedness` gains both informative folds, at all six settings tried.

`pitcher_hand` is determined by `pitcher_id`, so that segment is exactly **each pitcher's
own platoon split** -- how much his command changes against same- versus opposite-handed
batters. It is one of the most established effects in baseball and the calibration layer
has never carried it. The models see `pitcher_hand`, `batter_hand` and `hand_matchup` as
features, but a per-pitcher residual on a two-way split is the kind of fine interaction a
tree or an embedding smooths away.

Three ways this could be an illusion, and a control for each.

1. **Mass, not segment.** The extra term is added on top of three weights that already sum
   to one, so it raises total correction mass. The renormalised mode takes the mass from
   the existing terms instead, holding the total at one.
2. **Shrinkage, not handedness.** Splitting each pitcher in two halves the cell size, so at
   fixed smoothing the correction is more shrunk than a plain `pitcher_id` term. Two
   placebos split each pitcher by something meaningless and row-derived -- batter-id parity
   and day-of-week parity -- giving the same cell count and the same shrinkage with none of
   the content. If they gain as much, the handedness is decoration.
3. **Marginal, not interaction.** `handedness` alone scores 5.9 on 2024. It runs here too.

2022 reads exactly +0.00 for every candidate because that fold receives no calibration at
all, so the weakest-season rule has only two informative folds here. That was equally true
of V156's experience segment, which shipped and returned +1.33.
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


OUTPUT = Path("artifacts/v168_pitcher_platoon_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
SMOOTHINGS = (200.0, 500.0, 1000.0, 2000.0, 4000.0)
SEGMENTS = {
    "platoon": ["pitcher_id", "batter_hand"],
    "placebo_batter_parity": ["pitcher_id", "batter_parity"],
    "placebo_day_parity": ["pitcher_id", "day_parity"],
    "pitcher_only": ["pitcher_id"],
    "handedness_only": ["hand_matchup"],
}
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
    frame["hand_matchup"] = (frame["pitcher_hand"].astype(str) + "-"
                             + frame["batter_hand"].astype(str))
    frame["batter_parity"] = (frame["batter_id"] % 2).astype(str)
    frame["day_parity"] = (frame["game_dayofweek"] % 2).astype(str)
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    masks = {year: season_of == year for year in YEARS}
    history_index = {year: np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < year])
        for year in YEARS if year != 2022}
    train_frames = {y: frame.loc[i] for y, i in history_index.items()}
    valid_frames = {y: frame.loc[oof[str(y)]["row_index"]] for y in YEARS}

    weights = tuple(BASE[n] for n in NAMES6)
    blend = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
             for year in YEARS}

    def calibrated(extra=None, renormalise=False):
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            scale = 1.0
            if extra is not None and renormalise:
                scale = 1.0 / (1.0 + extra[1])
            value = (blend[year] + residual.mean()
                     + scale * W_COUNT * segment_correction(
                         train_f, residual, valid_f,
                         ["balls_before", "strikes_before"], 500)
                     + scale * W_PITCHER_COUNT * segment_correction(
                         train_f, residual, valid_f,
                         ["pitcher_id", "balls_before", "strikes_before"], 300)
                     + scale * W_EXPERIENCE * segment_correction(
                         train_f, residual, valid_f, ["experience_bin"],
                         EXPERIENCE_SMOOTHING))
            if extra is not None:
                columns, weight, smoothing = extra
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = calibrated()
    target = validation_frame["target"].to_numpy().astype(float)
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    print("cells per segment on the 2024 fold:", flush=True)
    valid_2024 = valid_frames[2024]
    for label, columns in SEGMENTS.items():
        key = valid_2024[columns[0]].astype(str)
        for column in columns[1:]:
            key = key + "|" + valid_2024[column].astype(str)
        print(f"  {label:24s} {key.nunique():6d} cells, "
              f"{len(key) / key.nunique():6.0f} rows each", flush=True)

    grid, best_per_segment = {}, {}
    for label, columns in SEGMENTS.items():
        print(f"\n{label}:", flush=True)
        for renormalise in (False, True):
            for weight in WEIGHTS:
                for smoothing in SMOOTHINGS:
                    tag = (f"{label}|{'renorm' if renormalise else 'add'}"
                           f"|w{weight:.2f}|s{smoothing:g}")
                    season = points(calibrated((columns, weight, smoothing), renormalise))
                    grid[tag] = {"season_points": season, "min": min(season),
                                 "average": float(np.mean(season))}
                    flag = "SAFE" if min(season) >= -1e-9 and max(season) > 0 else ""
                    print(f"  {tag:44s} min {min(season):+7.2f}  "
                          f"avg {np.mean(season):+7.2f}  2023 {season[1]:+7.2f}  "
                          f"2024 {season[2]:+7.2f}  {flag}", flush=True)
        candidates = [t for t in grid if t.startswith(f"{label}|")]
        best_per_segment[label] = max(
            candidates, key=lambda t: (grid[t]["season_points"][2], grid[t]["average"]))

    print("\nbest 2024 reading per segment -- the placebo comparison:", flush=True)
    for label, tag in best_per_segment.items():
        g = grid[tag]
        print(f"  {label:24s} {tag.split('|', 1)[1]:26s} "
              f"2023 {g['season_points'][1]:+7.2f}  2024 {g['season_points'][2]:+7.2f}",
              flush=True)

    def safe(tag):
        s = grid[tag]["season_points"]
        return min(s) >= -1e-9 and max(s) > 0

    survivors = [t for t in grid if safe(t) and t.startswith("platoon|")]
    ranked = sorted(survivors, key=lambda t: (grid[t]["min"], grid[t]["average"]),
                    reverse=True)
    finalists = ranked[:5]
    results = {}
    print("\nfull metrics for the five best platoon settings:", flush=True)
    for tag in finalists:
        label, mode, weight, smoothing = tag.split("|")
        candidate = calibrated((SEGMENTS[label], float(weight[1:]),
                                float(smoothing[1:])), mode == "renorm")
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        results[tag] = metrics
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        print(f"  {tag:44s} min {metrics['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  "
              f"2024 CI low {low:+7.2f}  blk {metrics['monthly_block_win_rate']:4.0%}",
              flush=True)

    promoted = finalists[0] if finalists else None
    OUTPUT.write_text(json.dumps({
        "experiment": "V168_pitcher_platoon_segment",
        "baseline": "V161 (Public 1053.2326413884)",
        "segment": ("pitcher_id x batter_hand -- pitcher_hand is determined by pitcher_id, "
                    "so this is each pitcher's own platoon split"),
        "controls": {
            "mass": "renormalise mode holds the total correction weight at one",
            "shrinkage": ("placebo_batter_parity and placebo_day_parity split each pitcher "
                          "by a meaningless row-derived bit, matching cell count and "
                          "shrinkage with no content"),
            "marginal": "handedness_only and pitcher_only run the same grid",
        },
        "grid": grid,
        "best_per_segment": {k: {"tag": v, **grid[v]} for k, v in best_per_segment.items()},
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in results.items()},
        "promoted_candidate": promoted,
        "caveat_2022": ("the 2022 fold receives no calibration, so it reads exactly +0.00 "
                        "for every candidate and only two folds are informative; the same "
                        "was true of V156's experience segment, which returned +1.33"),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsafe platoon settings = {len(survivors)} of {len(WEIGHTS) * len(SMOOTHINGS) * 2}"
          f"  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
