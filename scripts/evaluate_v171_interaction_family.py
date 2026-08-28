"""V171: the platoon term paid 7.31 -- sweep the rest of its family against V169.

V169 added `pitcher_id x batter_hand` to the calibration and the leaderboard went
1053.2326 -> 1060.5434. The 2024 fold had predicted +2.13, so the return was 3.4 times the
estimate -- the largest ratio this project has recorded, and far outside the 0.13-2.15
band the eight previous submissions traced.

That is worth taking seriously as a signal about *where* the remaining structure is. The
term that paid is a per-pitcher interaction with a small-cardinality row property: not the
pitcher marginal (which loses out of fold), not the property marginal (which loses 2023),
but the two crossed. V146 said the same thing from the oracle side and this is the first
time it has been cashed.

So the family gets swept properly, against the V169 baseline this time:

  * pitcher x every other small-cardinality row property -- outs, runners, base state,
    inning, leverage, half-inning, score state, month, competition
  * `batter_id x pitcher_hand`, the mirror image: the batter's own platoon split. V165
    found `batter_id` carrying 98 recoverable points on 2024 and gaining that fold out of
    fold (+0.96 to +2.56) while losing 2023 -- exactly the pattern `pitcher_id` showed
    before crossing it with handedness fixed it.
  * the platoon term itself split finer, by experience and by count.

Two things are re-measured first. The noise-corrected recoverable-points audit is re-run
on the **V169** residual, so the map reflects what the platoon term already took. And the
platoon weight and smoothing are re-swept at the new baseline, since adding a fifth
segment renormalises the fourth.

Every candidate is renormalised so total correction mass stays at one, which V168 showed
dominates adding on top. The 2022 fold receives no calibration and reads exactly +0.00, so
two folds are informative; that was true of V156 (+1.33) and of V169 (+7.31).
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


OUTPUT = Path("artifacts/v171_interaction_family_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
RAW = {"count": 0.55, "pitcher_count": 0.25, "experience": 0.20, "platoon": 0.20}
PLATOON_SMOOTHING = 1000.0
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
WEIGHTS = (0.05, 0.10, 0.20, 0.30)
SMOOTHINGS = (500.0, 1000.0, 2000.0)
PLATOON_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
PLATOON_SMOOTHINGS = (300.0, 500.0, 1000.0, 2000.0)
P = 100000.0 / 0.25

FAMILY = {
    "batter_platoon": ["batter_id", "pitcher_hand"],
    "pitcher_x_outs": ["pitcher_id", "outs_before"],
    "pitcher_x_runners": ["pitcher_id", "num_runners_on"],
    "pitcher_x_base": ["pitcher_id", "base_state"],
    "pitcher_x_inning": ["pitcher_id", "inning_bin"],
    "pitcher_x_leverage": ["pitcher_id", "li_bin"],
    "pitcher_x_half": ["pitcher_id", "top_bottom"],
    "pitcher_x_score": ["pitcher_id", "score_bin"],
    "pitcher_x_month": ["pitcher_id", "month_bin"],
    "pitcher_x_gametype": ["pitcher_id", "game_type"],
    "pitcher_x_batter_exp": ["pitcher_id", "batter_experience_bin"],
    "platoon_x_experience": ["pitcher_id", "batter_hand", "experience_bin"],
    "platoon_x_twostrike": ["pitcher_id", "batter_hand", "two_strike"],
    "batter_x_count": ["batter_id", "balls_before", "strikes_before"],
}


def annotate(frame):
    out = frame.copy()
    out["experience_bin"] = pd.cut(out["asof_pitcher_n"], EDGES,
                                   labels=LABELS).astype(str)
    out["batter_experience_bin"] = pd.cut(
        out["asof_batter_n"], [-1, 100, 400, 1200, 3000, 8000, np.inf],
        labels=["b0", "b1", "b2", "b3", "b4", "b5"]).astype(str)
    out["inning_bin"] = np.minimum(out["inning"], 10).astype(str)
    out["li_bin"] = pd.cut(out["li"], [-np.inf, 0.4, 0.8, 1.2, 2.0, 3.5, np.inf],
                           labels=["l0", "l1", "l2", "l3", "l4", "l5"]).astype(str)
    out["score_bin"] = pd.cut(out["score_diff_pitcher_team"],
                              [-np.inf, -3, -1, 0, 1, 3, np.inf],
                              labels=["s0", "s1", "s2", "s3", "s4", "s5"]).astype(str)
    out["month_bin"] = out["game_month"].astype(str)
    out["two_strike"] = (out["strikes_before"] == 2).astype(int).astype(str)
    return out


def recoverable(frame, residual, columns):
    key = frame[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "|" + frame[column].astype(str)
    grouped = pd.DataFrame({"r": residual, "k": key.to_numpy()}).groupby("k")["r"]
    n = grouped.size().to_numpy().astype(float)
    mean = grouped.mean().to_numpy()
    var = np.nan_to_num(grouped.var(ddof=1).to_numpy())
    unbiased = np.maximum(0.0, mean ** 2 - var / np.maximum(n, 1.0))
    return float(P * (n * unbiased).sum() / n.sum()), int(len(n))


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
    frame = annotate(raw_frame)
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

    def calibrated(platoon_weight=RAW["platoon"], platoon_smoothing=PLATOON_SMOOTHING,
                   extra=None):
        """V169's four segments, renormalised, optionally plus a fifth."""
        raw_weights = dict(RAW)
        raw_weights["platoon"] = platoon_weight
        terms = [(["balls_before", "strikes_before"], raw_weights["count"],
                  COUNT_SMOOTHING),
                 (["pitcher_id", "balls_before", "strikes_before"],
                  raw_weights["pitcher_count"], PITCHER_COUNT_SMOOTHING),
                 (["experience_bin"], raw_weights["experience"], EXPERIENCE_SMOOTHING),
                 (["pitcher_id", "batter_hand"], raw_weights["platoon"],
                  platoon_smoothing)]
        if extra is not None:
            terms.append(extra)
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
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
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

    residual_now = target - baseline
    audit_frame = frame.loc[order].reset_index(drop=True)
    print("what is left after V169's platoon term "
          "(noise-corrected recoverable points):", flush=True)
    print(f"  {'segment':26s} {'cells':>7s} {'2023':>9s} {'2024':>9s}", flush=True)
    audit = {}
    for label, columns in list(FAMILY.items()) + [
            ("platoon (in use)", ["pitcher_id", "batter_hand"]),
            ("pitcher x count (in use)", ["pitcher_id", "balls_before", "strikes_before"])]:
        row = {}
        for year in (2023, 2024):
            mask = masks[year]
            value, cells = recoverable(audit_frame.loc[mask], residual_now[mask], columns)
            row[str(year)], row["cells"] = value, cells
        audit[label] = row
        print(f"  {label:26s} {row['cells']:7d} {row['2023']:9.1f} {row['2024']:9.1f}",
              flush=True)

    print("\nre-sweep the platoon weight and smoothing at the V169 baseline:", flush=True)
    platoon_grid = {}
    for weight in PLATOON_WEIGHTS:
        row = []
        for smoothing in PLATOON_SMOOTHINGS:
            season = points(calibrated(weight, smoothing))
            platoon_grid[f"w{weight:.2f}_s{smoothing:g}"] = season
            row.append(f"{season[2]:7.2f}")
        print(f"  w={weight:.2f}  2024 by smoothing "
              f"{'  '.join(f'{s:g}' for s in PLATOON_SMOOTHINGS)}: {' '.join(row)}",
              flush=True)

    print("\nfifth segment, renormalised (2022 is +0.00 by construction):", flush=True)
    grid = {}
    for label, columns in FAMILY.items():
        best = None
        for weight in WEIGHTS:
            for smoothing in SMOOTHINGS:
                tag = f"{label}|w{weight:.2f}|s{smoothing:g}"
                season = points(calibrated(extra=(columns, weight, smoothing)))
                grid[tag] = {"season_points": season, "min": min(season),
                             "average": float(np.mean(season))}
                if best is None or season[2] > grid[best]["season_points"][2]:
                    best = tag
        g = grid[best]
        flag = "SAFE" if g["min"] >= -1e-9 and max(g["season_points"]) > 0 else ""
        print(f"  {label:22s} best-2024 {best.split('|', 1)[1]:18s} "
              f"2023 {g['season_points'][1]:+7.2f}  2024 {g['season_points'][2]:+7.2f}  "
              f"{flag}", flush=True)

    def safe(tag):
        s = grid[tag]["season_points"]
        return min(s) >= -1e-9 and max(s) > 0

    survivors = sorted((t for t in grid if safe(t)),
                       key=lambda t: (grid[t]["season_points"][2], grid[t]["average"]),
                       reverse=True)
    results = {}
    print(f"\n{len(survivors)} safe of {len(grid)}; full metrics for the best five by 2024:",
          flush=True)
    for tag in survivors[:5]:
        label, weight, smoothing = tag.split("|")
        candidate = calibrated(extra=(FAMILY[label], float(weight[1:]),
                                      float(smoothing[1:])))
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        results[tag] = metrics
        print(f"  {tag:40s} min {metrics['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)

    promoted = survivors[0] if survivors else None
    OUTPUT.write_text(json.dumps({
        "experiment": "V171_interaction_family",
        "baseline": "V169 (Public 1060.5433916316)",
        "premise": ("V169's pitcher_id x batter_hand term returned 7.31 against a 2024 "
                    "fold estimate of 2.13 -- a ratio of 3.4, outside the 0.13-2.15 band "
                    "of the eight previous submissions. The family it belongs to is "
                    "per-pitcher interactions with small-cardinality row properties."),
        "audit_after_v169": audit,
        "platoon_resweep": platoon_grid,
        "grid": grid,
        "safe": survivors,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"]}
                      for k, v in results.items()},
        "promoted_candidate": promoted,
        "ranking": ("among safe candidates, ranked by the 2024 fold -- 2022 receives no "
                    "calibration and reads exactly 0.00, and 2023 is contaminated by the "
                    "game_type = F break, which V165 showed inflating every segment there"),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
