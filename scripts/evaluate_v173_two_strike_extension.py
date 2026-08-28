"""V173: V172's optimum sat on the edge of the grid, so extend it before shipping.

V172 asked how finely the platoon term should be resolved. Replacing `pitcher_id x
batter_hand` with `pitcher_id x batter_hand x two_strike` beat every alternative on the
2024 fold, and the binary split beat the three-level `strikes_before` and the twelve-level
`count_state` -- the effect is a pitcher's *two-strike approach* against a given
handedness, not a general count dependence. Cell counts say why the finest split fails:
8,492 cells at 30 rows each cannot be estimated.

    REPLACE platoon x two_strike, 2024 points
        w \\ smoothing    500    1000    2000    4000
        0.20            3.08    2.32    1.04   -0.23
        0.30            3.66    3.39    2.00    0.38
        0.40            3.53   *4.02*   2.71    0.85

The maximum is at `w = 0.40`, which is the largest weight the grid contained, and the
column is still rising there. Shipping an edge value would be shipping an unfinished
search, and V168's lesson was that the safe reading and the stable reading are not the
same point.

Two extensions. The weight runs to 1.30, where the renormalised term carries more mass
than the count and pitcher-by-count terms combined and a turnover must exist. And the
plain platoon term is restored as a free second axis: "replace" and "add" were two corners
of a plane, and the interior -- a coarse fallback at low weight under a strong fine term --
was never looked at. That fallback matters for pitchers with few pitches against one hand,
which is exactly where a three-way split runs out of data.

Ranked among safe candidates by the 2024 fold, with the neighbourhood printed so the
choice can be read off a plateau rather than a peak.
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


OUTPUT = Path("artifacts/v173_two_strike_extension_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
COUNT_SMOOTHING, PITCHER_COUNT_SMOOTHING, EXPERIENCE_SMOOTHING = 500, 300, 2000
PLATOON_SMOOTHING = 1000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
PLATOON = ["pitcher_id", "batter_hand"]
FINE = PLATOON + ["two_strike"]
FINE_WEIGHTS = (0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.30)
FINE_SMOOTHINGS = (500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0)
COARSE_WEIGHTS = (0.00, 0.05, 0.10, 0.20)
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

    def build(coarse, fine, smoothing):
        return [(["balls_before", "strikes_before"], W_COUNT, COUNT_SMOOTHING),
                (["pitcher_id", "balls_before", "strikes_before"], W_PITCHER_COUNT,
                 PITCHER_COUNT_SMOOTHING),
                (["experience_bin"], W_EXPERIENCE, EXPERIENCE_SMOOTHING),
                (PLATOON, coarse, PLATOON_SMOOTHING),
                (FINE, fine, smoothing)]

    baseline = calibrated(build(0.20, 0.0, PLATOON_SMOOTHING))   # V169 exactly
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def points(candidate):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    grid = {}
    for coarse in COARSE_WEIGHTS:
        print(f"\ncoarse platoon weight {coarse:.2f}   2024 by fine smoothing "
              f"{' '.join(f'{s:6g}' for s in FINE_SMOOTHINGS)}", flush=True)
        for fine in FINE_WEIGHTS:
            row = []
            for smoothing in FINE_SMOOTHINGS:
                tag = f"c{coarse:.2f}|f{fine:.2f}|s{smoothing:g}"
                season = points(calibrated(build(coarse, fine, smoothing)))
                grid[tag] = {"season_points": season, "min": min(season),
                             "average": float(np.mean(season))}
                row.append(f"{season[2]:6.2f}")
            print(f"  fine={fine:.2f}   {' '.join(row)}", flush=True)

    def safe(tag):
        s = grid[tag]["season_points"]
        return min(s) >= -1e-9 and max(s) > 0

    survivors = sorted((t for t in grid if safe(t)),
                       key=lambda t: (grid[t]["season_points"][2], grid[t]["average"]),
                       reverse=True)
    print(f"\n{len(survivors)} safe of {len(grid)}", flush=True)

    def neighbourhood(tag):
        """The eight surrounding grid points, to tell a plateau from a spike."""
        coarse, fine, smoothing = tag.split("|")
        coarse, fine = float(coarse[1:]), float(fine[1:])
        smoothing = float(smoothing[1:])
        fi, si = FINE_WEIGHTS.index(fine), FINE_SMOOTHINGS.index(smoothing)
        values = []
        for df in (-1, 0, 1):
            for ds in (-1, 0, 1):
                if df == 0 and ds == 0:
                    continue
                if 0 <= fi + df < len(FINE_WEIGHTS) and 0 <= si + ds < len(FINE_SMOOTHINGS):
                    key = (f"c{coarse:.2f}|f{FINE_WEIGHTS[fi + df]:.2f}"
                           f"|s{FINE_SMOOTHINGS[si + ds]:g}")
                    values.append(grid[key]["season_points"][2])
        return min(values), float(np.mean(values))

    results = {}
    print("\nfull metrics for the best eight by 2024, with the neighbourhood:", flush=True)
    for tag in survivors[:8]:
        coarse, fine, smoothing = tag.split("|")
        candidate = calibrated(build(float(coarse[1:]), float(fine[1:]),
                                     float(smoothing[1:])))
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate, masks[year])
            for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        worst, mean = neighbourhood(tag)
        metrics["neighbourhood_2024"] = {"worst": worst, "mean": mean}
        results[tag] = metrics
        low = metrics["season_bootstrap"]["2024"]["ci95_low"] * P
        print(f"  {tag:26s} 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}  "
              f"neighbours worst {worst:+6.2f} mean {mean:+6.2f}", flush=True)

    promoted = max(
        results, key=lambda t: (results[t]["neighbourhood_2024"]["worst"],
                                results[t]["three_season"]["season_points"][2]),
        default=None)
    OUTPUT.write_text(json.dumps({
        "experiment": "V173_two_strike_extension",
        "baseline": "V169 (Public 1060.5433916316)",
        "premise": ("V172's best arm sat at w = 0.40, the largest weight in that grid, "
                    "with the column still rising -- an edge optimum is an unfinished "
                    "search. The coarse platoon term is restored as a second axis because "
                    "'replace' and 'add' were only two corners of a plane."),
        "grid": grid,
        "safe": survivors,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"],
                          "neighbourhood_2024": v["neighbourhood_2024"]}
                      for k, v in results.items()},
        "promoted_candidate": promoted,
        "ranking": ("among the safe finalists, ranked by the worst of the eight "
                    "neighbouring grid points on 2024, then by 2024 itself -- a plateau "
                    "beats a peak, which is the lesson V168 recorded"),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
