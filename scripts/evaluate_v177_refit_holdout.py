"""V177: V176's refit gained, but 195 hill-climbing steps on the 2024 fold is selection.

V176 refitted all eight calibration parameters jointly and reached 2023 +1.45 / 2024
+1.67. Two things in that run say do not ship it yet.

**The climb selected on the fold it is scored by.** Every earlier accepted change was a
single parameter or a small pre-registered grid; this was 195 evaluations of coordinate
descent whose objective *is* the 2024 fold. Some of +1.67 is real and some is the search.
The only way to know the split is to hold data out, so the climb here runs on 2024 months
3-7 and is scored on months 8-10, which it never sees. A gain that survives the holdout is
a gain; one that does not was the search talking.

**The placebos started passing.** On top of V176's refit, a fifth segment splitting each
pitcher-handedness cell by a *meaningless* bit scored 2024 +1.58 and +1.39 -- as high as
every genuine candidate except one. At V168 and V174 the real term beat its placebo by
4.5x; here the margin is gone. Adding almost any lightly-weighted, heavily-smoothed fifth
term now helps, which makes it geometry rather than content, and the whole fifth-segment
family is dead unless a candidate clears its own matched placebo by a wide margin.

The one exception is worth a proper test. `batter_id x pitcher_hand` -- the batter's own
platoon split, the mirror of what V169 shipped -- scored 2024 +2.68 against the placebo's
+1.58 at the same setting. That is 1.7x, not 4.5x, so it gets the matched-placebo
comparison run across the whole grid rather than at one point.

Also note what V176's audit exposed: the noise-corrected recoverable-points statistic gave
the *placebos* 549 and 558 points on 2024, against the real platoon term's 573. At roughly
1,500 cells and 167 rows the correction stops working, so the audit map was only ever
trustworthy for coarse segments and out-of-fold testing is the sole authority for fine ones.

The ladder is also extended, because V176's `count` weight stopped at the largest factor
its ladder contained.
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


OUTPUT = Path("artifacts/v177_refit_holdout_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SEGMENTS = {
    "count": ["balls_before", "strikes_before"],
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
    "experience": ["experience_bin"],
    "platoon2k": ["pitcher_id", "batter_hand", "two_strike"],
}
V175 = {"count": (0.55, 500.0), "pitcher_count": (0.25, 300.0),
        "experience": (0.20, 2000.0), "platoon2k": (0.80, 1500.0)}
V176 = {"count": (1.10, 200.0), "pitcher_count": (0.125, 510.0),
        "experience": (0.20, 800.0), "platoon2k": (1.12, 900.0)}
WEIGHT_FACTORS = (0.0, 0.35, 0.5, 0.7, 0.85, 1.0, 1.18, 1.4, 2.0, 3.0, 4.5)
SMOOTHING_FACTORS = (0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0)
ROUNDS = 6
SPLIT_MONTH = 8
P = 100000.0 / 0.25
MIRROR = ["batter_id", "pitcher_hand"]
MIRROR_PLACEBO = {"placebo_parity": ["batter_id", "day_parity"],
                  "placebo_inning": ["batter_id", "inning_parity"]}


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
    frame["day_parity"] = frame["game_dayofweek"] % 2
    frame["inning_parity"] = frame["inning"] % 2
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    month_of = frame.loc[order, "game_month"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    early = masks[2024] & (month_of < SPLIT_MONTH)
    late = masks[2024] & (month_of >= SPLIT_MONTH)
    print(f"2024 fold split: months <{SPLIT_MONTH} = {early.sum():,} rows, "
          f">={SPLIT_MONTH} = {late.sum():,} rows", flush=True)
    train_frames = {y: frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y])]
        for y in YEARS if y != 2022}
    valid_frames = {y: frame.loc[oof[str(y)]["row_index"]] for y in YEARS}
    blend = {year: sum(BASE[n] * parts[n][str(year)] for n in NAMES6) for year in YEARS}

    def calibrated(config, extra=None):
        terms = [(SEGMENTS[k], w, s) for k, (w, s) in config.items() if w > 0]
        if extra is not None:
            terms.append(extra)
        scale = 1.0 / sum(w for _, w, _ in terms)
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

    baseline = calibrated(V175)
    base_error = (baseline - target) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = base_error

    def block(candidate, mask):
        error = (candidate - target) ** 2
        return float(P * (base_error[mask].mean() - error[mask].mean()))

    def readings(candidate):
        return {"2022": block(candidate, masks[2022]),
                "2023": block(candidate, masks[2023]),
                "2024": block(candidate, masks[2024]),
                "2024_early": block(candidate, early),
                "2024_late": block(candidate, late)}

    def climb(objective, label):
        """Coordinate descent; feasible means no fold loses, then `objective` decides."""
        config = dict(V175)
        candidate = calibrated(config)
        best = readings(candidate)
        best_key = (min(best["2022"], best["2023"], best["2024"]) >= -1e-9,
                    best[objective])
        count = 1
        for round_index in range(ROUNDS):
            improved = False
            for name in SEGMENTS:
                for axis, ladder in (("weight", WEIGHT_FACTORS),
                                     ("smoothing", SMOOTHING_FACTORS)):
                    trials = []
                    for factor in ladder:
                        weight, smoothing = config[name]
                        trial = dict(config)
                        trial[name] = ((round(V175[name][0] * factor, 4), smoothing)
                                       if axis == "weight"
                                       else (weight, round(V175[name][1] * factor, 1)))
                        if trial[name] == config[name]:
                            continue
                        if sum(w for w, _ in trial.values()) <= 0:
                            continue
                        r = readings(calibrated(trial))
                        count += 1
                        key = (min(r["2022"], r["2023"], r["2024"]) >= -1e-9, r[objective])
                        trials.append((key, trial, r))
                    if not trials:
                        continue
                    trials.sort(key=lambda t: t[0], reverse=True)
                    if trials[0][0] > best_key:
                        best_key, config, best = trials[0][0], trials[0][1], trials[0][2]
                        improved = True
            if not improved:
                break
        print(f"\n{label}: {count} evaluations", flush=True)
        for name, (weight, smoothing) in config.items():
            print(f"  {name:14s} weight {V175[name][0]:5.2f} -> {weight:6.3f}   "
                  f"smoothing {V175[name][1]:6.0f} -> {smoothing:7.1f}", flush=True)
        return config, best

    def show(label, r):
        print(f"  {label:26s} 2023 {r['2023']:+7.2f}  2024 {r['2024']:+7.2f}   "
              f"[early {r['2024_early']:+7.2f}  HOLDOUT {r['2024_late']:+7.2f}]",
              flush=True)

    results = {}
    print("\n=== the honest test: climb on 2024 months 3-7, score on 8-10 ===", flush=True)
    honest_config, honest = climb("2024_early", "climb on 2024 early only")
    show("honest refit", honest)
    results["honest_refit"] = {"config": {k: list(v) for k, v in honest_config.items()},
                               "readings": honest}

    print("\n=== for comparison: climb on the whole 2024 fold ===", flush=True)
    greedy_config, greedy = climb("2024", "climb on all of 2024")
    show("greedy refit", greedy)
    results["greedy_refit"] = {"config": {k: list(v) for k, v in greedy_config.items()},
                               "readings": greedy}

    print("\n=== V176's stopping point, re-read on the holdout ===", flush=True)
    v176 = readings(calibrated(V176))
    show("V176 config", v176)
    results["v176_config"] = {"config": {k: list(v) for k, v in V176.items()},
                              "readings": v176}

    print("\n=== the mirror term against its own matched placebos ===", flush=True)
    mirror = {}
    for label, columns in [("batter_platoon", MIRROR)] + list(MIRROR_PLACEBO.items()):
        cells = valid_frames[2024][columns].astype(str).agg("|".join, axis=1).nunique()
        best_tag, best_value = None, -1e9
        for weight in (0.05, 0.10, 0.20, 0.40):
            for smoothing in (500.0, 1000.0, 2000.0):
                r = readings(calibrated(honest_config, (columns, weight, smoothing)))
                mirror[f"{label}|w{weight:.2f}|s{smoothing:g}"] = r
                if r["2024_late"] > best_value:
                    best_tag, best_value = f"w{weight:.2f}|s{smoothing:g}", r["2024_late"]
        r = mirror[f"{label}|{best_tag}"]
        print(f"  {label:20s} {cells:5d} cells  best-by-holdout {best_tag:16s} "
              f"2023 {r['2023']:+6.2f}  2024 {r['2024']:+6.2f}  "
              f"HOLDOUT {r['2024_late']:+6.2f}", flush=True)

    print("\nfull metrics:", flush=True)
    finalists = {}
    for label, config in (("honest_refit", honest_config), ("greedy_refit", greedy_config),
                          ("v176_config", V176)):
        candidate = calibrated(config)
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
        safe = metrics["min_season_points"] >= -1e-9 and max(t["season_points"]) > 0
        print(f"  {label:16s} 2023 {t['season_points'][1]:+7.2f}  "
              f"2024 {t['season_points'][2]:+7.2f}  CI low {low:+7.2f}  "
              f"blk {metrics['monthly_block_win_rate']:4.0%}  {'SAFE' if safe else ''}",
              flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V177_refit_holdout",
        "baseline": "V175 (Public 1067.8617513573)",
        "concerns": {
            "selection": ("V176 ran 195 coordinate-descent steps whose objective was the "
                          "2024 fold it is scored by; the holdout separates the gain from "
                          "the search"),
            "placebos_passing": ("on top of V176's refit a meaningless fifth split scored "
                                 "2024 +1.58, as high as every genuine candidate but one, "
                                 "so the fifth-segment family is geometry not content"),
            "audit_statistic": ("the noise correction gave the placebos 549 and 558 points "
                                "on 2024 against the real term's 573, so the map is only "
                                "trustworthy for coarse segments"),
        },
        "split_month": SPLIT_MONTH,
        "rows": {"2024_early": int(early.sum()), "2024_late": int(late.sum())},
        "results": results,
        "mirror_grid": mirror,
        "finalists": {k: {"three_season": v["three_season"],
                          "min_season_points": v["min_season_points"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"],
                          "bootstrap_2024": v["bootstrap_2024"]}
                      for k, v in finalists.items()},
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
