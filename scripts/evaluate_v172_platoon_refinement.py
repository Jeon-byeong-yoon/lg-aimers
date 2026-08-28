"""V172: refine the platoon term itself, since only the platoon family survives.

V171 swept eleven `pitcher x situation` segments against the V169 baseline and every one
lost 2024. The batter's own platoon split gained 2024 (+2.33) and lost 2023 (-5.71). What
survived was only the platoon term split finer:

    pitcher x batter_hand x two_strike    2023 +3.29   2024 +2.52   at w0.30 / s1000
    pitcher x batter_hand x experience    2023 +3.43   2024 +0.67   at w0.10 / s500

So the effect is handedness, not pitcher interactions in general, and the open question is
how finely handedness should be resolved. Four things are asked here.

**Which split.** `two_strike` is a binary guess. `strikes_before` has three levels and
`count_state` has twelve; the count is where a pitcher's approach against a given
handedness actually changes (an 0-2 slider away versus a 3-0 fastball). All three run.

**Add or replace.** V171 added the refinement as a fifth term beside the plain platoon
term, which is hierarchical shrinkage -- coarse first, fine on top. Replacing the coarse
term with the fine one is the alternative and was never tried; it has fewer parameters but
loses the fallback for pitchers with few pitches against one hand.

**How many.** A sixth term (platoon, platoon x strikes, platoon x experience together) is
the natural extension if two work.

**Where it stops.** Every arm is swept over weight and smoothing so the answer is a
surface, not a point. V168's lesson was that the safe candidate ranked first by the
three-season average sat on a cliff while the 2024 ridge was broad and flat; the same
reading applies here.

Ranked among safe candidates by the 2024 fold. 2022 receives no calibration and reads
exactly +0.00; 2023 is contaminated by the `game_type = F` break, which V165 showed
inflating every segment on that fold.
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


OUTPUT = Path("artifacts/v172_platoon_refinement_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE, W_PLATOON = 0.55, 0.25, 0.20, 0.20
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
PLATOON_SMOOTHING = 1000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
WEIGHTS = (0.10, 0.15, 0.20, 0.30, 0.40)
SMOOTHINGS = (500.0, 1000.0, 2000.0, 4000.0)
PLATOON = ["pitcher_id", "batter_hand"]
REFINEMENTS = {
    "two_strike": PLATOON + ["two_strike"],
    "strikes": PLATOON + ["strikes_before"],
    "count": PLATOON + ["count_state"],
    "experience": PLATOON + ["experience_bin"],
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
    frame["two_strike"] = (frame["strikes_before"] == 2).astype(int)
    frame["count_state"] = (frame["balls_before"].astype(str) + "-"
                            + frame["strikes_before"].astype(str))
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
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def core(with_platoon=True):
        terms = [(["balls_before", "strikes_before"], W_COUNT, COUNT_SMOOTHING),
                 (["pitcher_id", "balls_before", "strikes_before"], W_PITCHER_COUNT,
                  PITCHER_COUNT_SMOOTHING),
                 (["experience_bin"], W_EXPERIENCE, EXPERIENCE_SMOOTHING)]
        if with_platoon:
            terms.append((PLATOON, W_PLATOON, PLATOON_SMOOTHING))
        return terms

    baseline = calibrated(core())
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    print("cells on the 2024 fold:", flush=True)
    valid = valid_frames[2024]
    for label, columns in REFINEMENTS.items():
        key = valid[columns[0]].astype(str)
        for column in columns[1:]:
            key = key + "|" + valid[column].astype(str)
        print(f"  platoon x {label:12s} {key.nunique():6d} cells, "
              f"{len(key) / key.nunique():6.0f} rows each", flush=True)

    grid = {}

    def sweep(prefix, build):
        best = None
        for weight in WEIGHTS:
            row = []
            for smoothing in SMOOTHINGS:
                tag = f"{prefix}|w{weight:.2f}|s{smoothing:g}"
                season = points(calibrated(build(weight, smoothing)))
                grid[tag] = {"season_points": season, "min": min(season),
                             "average": float(np.mean(season))}
                row.append(f"{season[2]:7.2f}")
                if best is None or season[2] > grid[best]["season_points"][2]:
                    best = tag
            print(f"    w={weight:.2f}  2024: {' '.join(row)}", flush=True)
        g = grid[best]
        flag = "SAFE" if g["min"] >= -1e-9 and max(g["season_points"]) > 0 else ""
        print(f"    best {best.split('|', 1)[1]:18s} "
              f"2023 {g['season_points'][1]:+7.2f}  2024 {g['season_points'][2]:+7.2f}  "
              f"{flag}", flush=True)

    header = "        2024 by smoothing " + " ".join(f"{s:6g}" for s in SMOOTHINGS)
    for label, columns in REFINEMENTS.items():
        print(f"\nADD  platoon + platoon x {label}   (V169's term kept)\n{header}",
              flush=True)
        sweep(f"add_{label}",
              lambda w, s, c=columns: core() + [(c, w, s)])
        print(f"\nREPLACE  platoon x {label} instead of platoon\n{header}", flush=True)
        sweep(f"replace_{label}",
              lambda w, s, c=columns: core(with_platoon=False) + [(c, w, s)])

    print(f"\nSIX terms: platoon + platoon x strikes + platoon x experience\n{header}",
          flush=True)
    sweep("six",
          lambda w, s: core() + [(REFINEMENTS["strikes"], w, s),
                                 (REFINEMENTS["experience"], 0.10, 500.0)])

    def safe(tag):
        s = grid[tag]["season_points"]
        return min(s) >= -1e-9 and max(s) > 0

    survivors = sorted((t for t in grid if safe(t)),
                       key=lambda t: (grid[t]["season_points"][2], grid[t]["average"]),
                       reverse=True)
    results = {}
    print(f"\n{len(survivors)} safe of {len(grid)}; full metrics for the best six by 2024:",
          flush=True)
    for tag in survivors[:6]:
        prefix, weight, smoothing = tag.split("|")
        weight, smoothing = float(weight[1:]), float(smoothing[1:])
        if prefix == "six":
            terms = core() + [(REFINEMENTS["strikes"], weight, smoothing),
                              (REFINEMENTS["experience"], 0.10, 500.0)]
        else:
            mode, label = prefix.split("_", 1)
            terms = core(with_platoon=(mode == "add")) + [
                (REFINEMENTS[label], weight, smoothing)]
        candidate = calibrated(terms)
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
        print(f"  {tag:34s} min {metrics['min_season_points']:+7.2f}  "
              f"avg {t['average_points']:+7.2f}  2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  2024 CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)

    promoted = survivors[0] if survivors else None
    OUTPUT.write_text(json.dumps({
        "experiment": "V172_platoon_refinement",
        "baseline": "V169 (Public 1060.5433916316)",
        "premise": ("V171 found eleven pitcher x situation segments all losing 2024 while "
                    "the platoon term split finer survived, so the effect is handedness "
                    "and the open question is how finely to resolve it"),
        "arms": ("four splits (two_strike, strikes, count, experience) x two modes (add "
                 "beside V169's term, or replace it) plus a six-term arm"),
        "grid": grid,
        "safe": survivors,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in results.items()},
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
