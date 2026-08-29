"""V188: V164 rejected seed averaging against the luckiest draw on the deciding fold.

V187 lost 2.59 points after passing the gate, and the post-mortem named the reason: for a
stochastic component the fold reading is not the value of a change, it is one draw. That
same objection applies backwards, to a conclusion this project has been carrying since
V164.

V164 averaged the two neural components over independent seeds, found every season of both
components better standalone, and rejected it because the blend lost 2024 in all nine
configurations. Every one of those readings was taken **against seed 42**. On the 2024
fold:

    network        by seed   629   579   621   544   560     mean 587   seed 42 is 1st of 5
    factorization  by seed   155   202    65    82   151     mean 131   seed 42 is 2nd of 5

Seed 42 is the best draw of five on the fold that decides. So "averaging loses 2024" was
measured as (average of k draws) minus (the luckiest draw), which is not the comparison
deployment faces. Deployment ships **one arbitrary draw**, and the honest question is
whether an average beats a *typical* one.

This recomputes every reading against each of the five single-seed baselines in turn and
reports the mean -- the expected effect of shipping an average instead of an arbitrary
seed. Nothing is retrained; V164's predictions are cached.

Two things come out of it. The expected gain, if positive, revives a candidate that also
does something no other candidate does: it **shrinks the deployment lottery**. The spread
of blend points across the five single-seed configurations measures that lottery directly,
and V187 just paid 2.59 points to it.
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


OUTPUT = Path("artifacts/v188_seed_average_vs_expected_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SEEDS = (42, 1004, 2024, 777, 999)
LADDER = (2, 3, 5)
P = 100000.0 / 0.25


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    store = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
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
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(network, factorization):
        parts = dict(fixed)
        parts.update({"network": network, "factorization": factorization})
        blend = {y: sum(BASE[n] * parts[n][str(y)] for n in NAMES6) for y in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def average(kind, count):
        return {str(y): np.mean([store[kind][s][str(y)] for s in SEEDS[:count]], axis=0)
                for y in YEARS}

    singles = {s: pipeline(store["networks"][s], store["factorizations"][s])
               for s in SEEDS}
    errors = {s: (p - target) ** 2 for s, p in singles.items()}

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    print("the deployment lottery -- each single seed against seed 42:", flush=True)
    print(f"  {'seed':>6s} {'2022':>8s} {'2023':>8s} {'2024':>8s}", flush=True)
    lottery = {}
    for s in SEEDS:
        row = points(singles[s], errors[42])
        lottery[str(s)] = row
        print(f"  {s:6d} {row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}", flush=True)
    matrix = np.array([lottery[str(s)] for s in SEEDS])
    print(f"  {'sd':>6s} {matrix.std(axis=0, ddof=1)[0]:8.2f} "
          f"{matrix.std(axis=0, ddof=1)[1]:8.2f} "
          f"{matrix.std(axis=0, ddof=1)[2]:8.2f}", flush=True)
    print(f"  {'range':>6s} {matrix[:, 0].ptp():8.2f} {matrix[:, 1].ptp():8.2f} "
          f"{matrix[:, 2].ptp():8.2f}   <- what shipping an arbitrary seed costs or wins",
          flush=True)

    print("\nseed averaging, measured against every single-seed baseline in turn:",
          flush=True)
    results = {}
    for count in LADDER:
        for label, pair in (
                (f"net{count}", (average("networks", count),
                                 store["factorizations"][42])),
                (f"fact{count}", (store["networks"][42],
                                  average("factorizations", count))),
                (f"both{count}", (average("networks", count),
                                  average("factorizations", count)))):
            candidate = pipeline(*pair)
            rows = {str(s): points(candidate, errors[s]) for s in SEEDS}
            block = np.array([rows[str(s)] for s in SEEDS])
            expected = block.mean(axis=0)
            against42 = rows["42"]
            results[label] = {"per_baseline": rows, "expected": expected.tolist(),
                              "against_seed42": against42,
                              "expected_all_positive": bool((expected > 0).all())}
            print(f"  {label:10s} vs seed42  {against42[0]:+7.2f} {against42[1]:+7.2f} "
                  f"{against42[2]:+7.2f}     EXPECTED  {expected[0]:+7.2f} "
                  f"{expected[1]:+7.2f} {expected[2]:+7.2f}   "
                  f"{'ALL POSITIVE' if (expected > 0).all() else ''}", flush=True)

    # For a candidate to be shipped it must also beat the *typical* draw, and shipping an
    # average is only meaningful if it reduces the spread the lottery table just measured.
    print("\nspread of the blend across single seeds, and what an average would leave:",
          flush=True)
    for count in LADDER:
        avg = pipeline(average("networks", count), average("factorizations", count))
        gaps = np.array([points(avg, errors[s]) for s in SEEDS])
        print(f"  average of {count}: expected {gaps.mean(axis=0)[2]:+7.2f} on 2024, "
              f"and the single-seed sd it replaces was "
              f"{matrix.std(axis=0, ddof=1)[2]:.2f}", flush=True)

    best = max((l for l in results if results[l]["expected_all_positive"]),
               key=lambda l: results[l]["expected"][2], default=None)
    finalists = {}
    if best is not None:
        print(f"\nfull metrics for {best}, against the mean single-seed baseline:",
              flush=True)
        count = int(best[-1])
        pair = ((average("networks", count), store["factorizations"][42])
                if best.startswith("net") else
                (store["networks"][42], average("factorizations", count))
                if best.startswith("fact") else
                (average("networks", count), average("factorizations", count)))
        candidate = pipeline(*pair)
        for s in SEEDS:
            reference = validation_frame.copy()
            reference["v41_prediction"] = singles[s]
            reference["v41_squared_error"] = errors[s]
            metrics = development_metrics(reference, candidate)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, singles[s], candidate, masks[year])
                for year in YEARS}
            metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
            t = three_season(metrics)
            print(f"  vs seed {s:5d}: 2022 {t['season_points'][0]:+7.2f}  "
                  f"2023 {t['season_points'][1]:+7.2f}  "
                  f"2024 {t['season_points'][2]:+7.2f}  "
                  f"blk {metrics['monthly_block_win_rate']:4.0%}", flush=True)
            finalists[str(s)] = {"three_season": t,
                                 "monthly_block_win_rate": metrics["monthly_block_win_rate"]}

    OUTPUT.write_text(json.dumps({
        "experiment": "V188_seed_average_vs_expected",
        "baseline": "V175 (Public 1067.8617513573)",
        "premise": ("V164 rejected seed averaging on readings taken against seed 42, which "
                    "is the best of five draws on the 2024 fold (629 against a mean of "
                    "587 for the network, second best for the factorization network). "
                    "Deployment ships one arbitrary draw, so the comparison that matters "
                    "is against a typical seed, not the luckiest one."),
        "seeds": list(SEEDS),
        "deployment_lottery_vs_seed42": lottery,
        "lottery_sd": matrix.std(axis=0, ddof=1).tolist(),
        "lottery_range": [float(matrix[:, i].ptp()) for i in range(3)],
        "results": results,
        "best_expected": best,
        "finalists_vs_each_seed": finalists,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nbest by expected 2024 = {best}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
