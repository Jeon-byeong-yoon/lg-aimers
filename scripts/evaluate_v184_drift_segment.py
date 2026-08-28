"""V184: the calibration layer has never seen the within-season signal.

V183 redrew the frontier and the placebo line ate almost everything. At 1,500 cells the
shipped platoon term's oracle (887.9) and a meaningless split's oracle (869.1) are
indistinguishable, so every fine partition on that map is inflation. What survives the
placebo comparison at honest cell sizes is one thing:

    partition        cells   n/cell   perfect correction   de-noised
    pitcher            391      648                228.0       143.6

Real, coarse enough to trust -- and V165 measured `pitcher` alone **losing** out of fold at
every weight. Both are true at once, and together they say what the structure is: each
season's pitchers deviate from prediction by about 144 points' worth, and that deviation
does not carry over from the seasons before it. It is *this* season's form. V148 found the
same wall from a different direction and called it unreachable.

It is not unreachable, because the pipeline already reconstructs it. The V92/V153 in-season
block recovers each pitcher's within-season rate from the career-cumulative `asof_*`
columns by anchor differencing, and it reaches the prediction two ways: as features to
Form, the network and the factorization network, and as a post-hoc drift term

    correction = weight * delta * reliability          weight = 0.10

which is a **fixed straight line** through the shrunk deviation. One weight, fitted at V96
against a three-component blend, for the entire relationship between a pitcher's current
form and how wrong the blend is about him.

The calibration layer, which has just paid +14.65 twice by learning a segment's shape
instead of assuming it, contains no term on this signal at all. Its four segments are all
static -- count, pitcher-by-count, experience, handedness. Binning the drift signal and
giving it a lookup lets that relationship be an arbitrary learned function instead of a
line, and crossing it with reliability learns the two-dimensional surface the linear form
collapses into a product.

The cells are coarse on purpose -- ten bins, or ten by five -- which is the range where
V183 showed the placebo line is still far below the real one and a reading can be believed.

Arms are pre-registered: the drift bin alone, crossed with reliability, each with the
linear drift term kept and switched off, over a 3x2 weight-by-smoothing grid, and a matched
placebo throughout. Bin edges come from each fold's training seasons only.
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
from inseason_asof_features_v92 import (
    COMPOSITIONS, add_training_inseason_features, drift_correction,
)


OUTPUT = Path("artifacts/v184_drift_segment_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
DRIFT_BINS = 10
RELIABILITY_BINS = 5
WEIGHTS = (0.10, 0.20, 0.40)
SMOOTHINGS = (500.0, 2000.0)
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
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    block = add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order]
    term = drift_correction(block, 1.0)
    delta = np.nan_to_num(np.asarray(COMPOSITIONS["success"](block), dtype=float),
                          nan=0.0)
    reliability = np.nan_to_num(
        block["ins_pitcher_reliability"].to_numpy(dtype=float), nan=0.0)
    signal = delta * reliability

    frame = raw_frame.loc[order].reset_index(drop=True)
    frame["experience_bin"] = pd.cut(frame["asof_pitcher_n"], EDGES,
                                     labels=LABELS).astype(str)
    frame["two_strike"] = (frame["strikes_before"] == 2).astype("int64")
    frame["_signal"] = signal
    frame["_reliability"] = reliability
    mixed = ((frame["pitcher_id"].to_numpy() * 2654435761
              + frame["batter_id"].to_numpy()) % 10007) / 10007.0
    frame["_placebo"] = mixed
    frame.index = pd.Index(order)

    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    history_index = {y: np.concatenate([oof[str(h)]["row_index"] for h in YEARS if h < y])
                     for y in YEARS if y != 2022}
    valid_index = {y: oof[str(y)]["row_index"] for y in YEARS}
    blend = {year: sum(BASE[n] * parts[n][str(year)] for n in NAMES6) for year in YEARS}

    def binned(column, bins, year, source):
        """Quantile bins whose edges come from the fold's training seasons only."""
        train_values = frame.loc[history_index[year], column].to_numpy()
        edges = np.unique(np.quantile(train_values, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        return pd.cut(frame.loc[source, column], edges, labels=False).astype(str)

    print(f"drift signal: mean {signal.mean():+.6f}, sd {signal.std():.6f}, "
          f"non-zero on {100 * (signal != 0).mean():.1f}% of rows", flush=True)

    def calibrated(extra=None, drift_weight=DRIFT_WEIGHT, placebo=False, cross=False):
        weights = [w for _, w, _ in TERMS] + ([extra[0]] if extra else [])
        scale = 1.0 / sum(weights)
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f = frame.loc[history_index[year]].copy()
            valid_f = frame.loc[valid_index[year]].copy()
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            if extra is not None:
                weight, smoothing = extra
                column = "_placebo" if placebo else "_signal"
                for f, index in ((train_f, history_index[year]),
                                 (valid_f, valid_index[year])):
                    f["_bin"] = binned(column, DRIFT_BINS, year, index).to_numpy()
                    if cross:
                        f["_rbin"] = binned("_reliability", RELIABILITY_BINS,
                                            year, index).to_numpy()
                key = ["_bin", "_rbin"] if cross else ["_bin"]
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, key, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + drift_weight * term, 0, 1)

    baseline = calibrated()
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    print(f"\ncontrol: dropping the linear drift term entirely -> "
          f"{points(calibrated(drift_weight=0.0))}", flush=True)

    results = {}
    print(f"\n{'arm':40s} {'2023':>8s} {'2024':>8s}", flush=True)
    for cross in (False, True):
        for keep_linear in (True, False):
            for placebo in (False, True):
                name = ("bin x reliability" if cross else "bin")
                name += " + linear" if keep_linear else " (linear off)"
                if placebo:
                    name = "PLACEBO " + name
                for weight in WEIGHTS:
                    for smoothing in SMOOTHINGS:
                        label = f"{name}|w{weight:.2f}|s{smoothing:g}"
                        season = points(calibrated(
                            (weight, smoothing),
                            drift_weight=DRIFT_WEIGHT if keep_linear else 0.0,
                            placebo=placebo, cross=cross))
                        results[label] = season
                        flag = ("SAFE" if min(season) >= -1e-9 and max(season) > 0
                                else "")
                        print(f"  {label:38s} {season[1]:8.2f} {season[2]:8.2f}  {flag}",
                              flush=True)

    real_safe = [l for l in results if not l.startswith("PLACEBO")
                 and min(results[l]) >= -1e-9 and max(results[l]) > 0]
    placebo_best = max((results[l][2] for l in results if l.startswith("PLACEBO")),
                       default=float("-inf"))
    promoted = max(real_safe, key=lambda l: results[l][2], default=None)
    print(f"\nreal safe = {len(real_safe)}; best real 2024 = "
          f"{results[promoted][2]:+.2f}" if promoted else "\nreal safe = 0", flush=True)
    print(f"best placebo 2024 = {placebo_best:+.2f}", flush=True)

    finalists = {}
    if promoted:
        for label in sorted(real_safe, key=lambda l: results[l][2], reverse=True)[:3]:
            name, weight, smoothing = label.split("|")
            candidate = calibrated(
                (float(weight[1:]), float(smoothing[1:])),
                drift_weight=0.0 if "(linear off)" in name else DRIFT_WEIGHT,
                placebo=False, cross="reliability" in name)
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
            print(f"  {label:38s} 2023 {t['season_points'][1]:+7.2f}  "
                  f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
                  f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V184_drift_segment",
        "baseline": "V175 (Public 1067.8617513573)",
        "premise": ("V183 left one honest reading -- pitcher-level structure worth 144 "
                    "de-noised points on 2024 that V165 showed does not transfer from "
                    "prior seasons, i.e. this season's form. The pipeline reconstructs "
                    "exactly that, but delivers it to the prediction as a fixed straight "
                    "line, `weight * delta * reliability`, with one weight fitted at V96 "
                    "against a three-component blend. The calibration layer, which has "
                    "twice paid for learning a segment's shape instead of assuming it, "
                    "has no term on this signal."),
        "design": ("ten quantile bins of the shrunk drift signal, optionally crossed with "
                   "five reliability bins; edges from each fold's training seasons only; "
                   "the linear term kept and switched off; matched placebo throughout"),
        "arms": {k: {"season_points": v,
                     "safe": bool(min(v) >= -1e-9 and max(v) > 0)}
                 for k, v in results.items()},
        "real_safe": sorted(real_safe),
        "best_placebo_2024": placebo_best,
        "promoted_candidate": promoted,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True,
                       "bin_edges_from_training_seasons_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
