"""V182: two structural changes to the correction algebra that add no free parameters.

V179 drew the line: the calibration layer accepts one structural addition with content and
rejects refitting its parameters, because eight-parameter hill climbing manufactures +4 to
+7 points of selection bias. Both changes here sit on the accepting side of that line --
each replaces a formula with a better-motivated one and introduces **no new constant**, so
there is nothing to overfit and the safe gate is the whole test.

**A. The correction should not be flat across the probability range.** Every term is
applied as `p + w * c`, the same shift whether the blend says 0.25 or 0.72. A pitcher's
platoon effect is not like that: a shift is bounded by how much room there is, and near
the ends there is less. That is what a log-odds shift does, and to first order a log-odds
shift of d moves the probability by `d * p(1-p)`. So the same fitted `c` is applied scaled
by the row's own variance relative to its cell's average variance,

    p + w * c * [p(1-p)] / [mean p(1-p) over the cell]

which leaves the cell's average shift exactly `w * c` -- the fitted quantity is unchanged
and only its distribution within the cell moves. No constant is added. V145 tested
log-odds *blending* of components and rejected it; the calibration layer has never been
asked.

**B. The shrinkage denominator counts the wrong thing.** Every lookup is

    c = sum(cell) / (n(cell) + smoothing)

which treats `n` as the information in the cell. For a binary outcome the information in a
row is `p(1-p)`, so a hundred rows at p = 0.72 carry about a fifth less than a hundred at
p = 0.50 and should be shrunk harder. Replacing `n` with `n_eff = sum of 4p(1-p)` -- scaled
so that a cell at p = 0.5 is unchanged, which keeps every existing smoothing constant
meaning what it meant -- is the empirical-Bayes form. Again no new constant.

Six arms, pre-registered: each change on the platoon term alone, each on all four terms,
and the two combined. The safe gate is unchanged and both informative folds are reported,
which for a zero-parameter change is already the cross-fold transfer test V179 established.
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
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v182_zero_parameter_structure_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FINE = 3
P = 100000.0 / 0.25


def correction(train_frame, train_residual, train_p, valid_frame, valid_p,
               columns, smoothing, effective_n=False, variance_scaled=False):
    """One segment lookup, optionally with an information-weighted denominator and a
    within-cell shift that follows the local variance."""
    centered = train_residual - train_residual.mean()
    work = train_frame[columns].copy()
    work["_r"] = centered
    work["_v"] = 4.0 * train_p * (1.0 - train_p)
    grouped = work.groupby(columns, dropna=False, sort=False)
    stats = grouped.agg(_sum=("_r", "sum"), _n=("_r", "size"), _vsum=("_v", "sum"))
    denominator = stats["_vsum"] if effective_n else stats["_n"]
    stats["_c"] = stats["_sum"] / (denominator + smoothing)
    stats["_vbar"] = stats["_vsum"] / stats["_n"]
    table = stats[["_c", "_vbar"]].reset_index()
    merged = valid_frame[columns].merge(table, how="left", on=columns, sort=False)
    values = merged["_c"].fillna(0.0).to_numpy()
    if not variance_scaled:
        return values
    vbar = merged["_vbar"].to_numpy()
    ratio = np.where(np.isfinite(vbar) & (vbar > 1e-9),
                     (4.0 * valid_p * (1.0 - valid_p)) / vbar, 1.0)
    return values * ratio


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
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def calibrated(effective_n=(), variance_scaled=()):
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in history])
            train_p = np.concatenate([blend[h] for h in history])
            train_f, valid_f = train_frames[year], valid_frames[year]
            valid_p = blend[year]
            value = valid_p + residual.mean()
            for index, (columns, weight, smoothing) in enumerate(TERMS):
                value = value + scale * weight * correction(
                    train_f, residual, train_p, valid_f, valid_p, columns, smoothing,
                    effective_n=index in effective_n,
                    variance_scaled=index in variance_scaled)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    baseline = calibrated()
    stored = joblib.load("artifacts/v175_calibration.joblib")
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error
    print(f"control: rebuilt V175 layer, shift {stored['global_shift']:.12f} in the "
          f"artifact; prediction mean {baseline.mean():.6f}", flush=True)

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    everything = tuple(range(len(TERMS)))
    arms = {
        "A_variance_scaled_fine": {"variance_scaled": (FINE,)},
        "A_variance_scaled_all": {"variance_scaled": everything},
        "B_effective_n_fine": {"effective_n": (FINE,)},
        "B_effective_n_all": {"effective_n": everything},
        "AB_fine": {"variance_scaled": (FINE,), "effective_n": (FINE,)},
        "AB_all": {"variance_scaled": everything, "effective_n": everything},
    }
    results = {}
    print("\narms (2022 is +0.00 by construction; both other folds must gain):", flush=True)
    for label, kwargs in arms.items():
        season = points(calibrated(**kwargs))
        results[label] = season
        flag = "SAFE" if min(season) >= -1e-9 and max(season) > 0 else ""
        print(f"  {label:24s} 2023 {season[1]:+7.2f}  2024 {season[2]:+7.2f}  {flag}",
              flush=True)

    safe = [l for l in results if min(results[l]) >= -1e-9 and max(results[l]) > 0]
    promoted = max(safe, key=lambda l: results[l][2], default=None)
    finalists = {}
    if safe:
        print("\nfull metrics for the safe arms:", flush=True)
        for label in sorted(safe, key=lambda l: results[l][2], reverse=True):
            candidate = calibrated(**arms[label])
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
            print(f"  {label:24s} 2023 {t['season_points'][1]:+7.2f}  "
                  f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
                  f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V182_zero_parameter_structure",
        "baseline": "V175 (Public 1067.8617513573)",
        "why_allowed": ("V179 showed the layer rejects parameter refits because an "
                        "eight-parameter climb manufactures +4 to +7 points of selection "
                        "bias; both changes here replace a formula and add no constant, "
                        "so there is nothing to overfit"),
        "A": ("apply each correction scaled by the row's p(1-p) relative to its cell "
              "average -- the first-order effect of a log-odds shift, leaving the cell's "
              "average shift exactly as fitted"),
        "B": ("replace n with the sum of 4p(1-p) in the shrinkage denominator, the "
              "empirical-Bayes information count, scaled so a cell at p=0.5 is unchanged "
              "and every existing smoothing constant keeps its meaning"),
        "arms": {k: {"season_points": v,
                     "safe": bool(min(v) >= -1e-9 and max(v) > 0)}
                 for k, v in results.items()},
        "safe": sorted(safe),
        "promoted_candidate": promoted,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
