"""V137: put the factorization network in the Context slot instead of removing weight.

V136 established that the Context axis is unanimous in direction and completely blocked
in practice: eighteen pairwise transfers all say Context is over-weighted, yet reducing
it by even one point sends 2024 negative and leaves the monthly block win rate at 65-70%
against a floor of 75%. Moving its weight to components that already exist does not work,
because the reason Context still earns its place is diversity on the 2024 fold, where
every inter-component correlation drops.

V120 solved the same shape of problem for v17. It did not merely cut v17's weight -- it
*replaced* v17 with CatBoost, and that is what passed. The analogue here is to find
something for the slot rather than to empty it.

V130's factorization network was rejected as a sixth component, and rightly: standalone
2024 skill of +338 against the V111 network's +616. But rejected-as-an-addition is a
different question from rejected-as-a-replacement, and its profile against Context is
striking:

    candidate     2022 skill  corr   2023 skill  corr   2024 skill  corr
    context             2224  0.919        -1333  0.903         725  0.817
    FM latent6          2150  0.888         -558  0.841         338  0.777

It fixes exactly what is wrong with Context. On 2023, where Context is second-worst in
the ensemble at -1333 and effectively tied with the retired v17, the factorization
network scores -558 -- 775 points better. It is less correlated with the rest on all
three seasons. And it is worse on 2024, which is the risk that has blocked every
Context candidate so far.

So this is not a weaker component being smuggled in. It is the V123 lesson run in
reverse: there, `onehot` was individually stronger on all three seasons yet lost 2022 in
the blend because it was more correlated. Here the trade is inverted, and whether it pays
is precisely what the blend has to decide.

Splits from a five-point handover up to full replacement, three latent dimensions, all
on cached predictions.

Pre-registered gate, unchanged since V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v137b_context_slot_metrics.json")
NAMES6 = ("v17", "form", "context", "network", "catboost", "factorization")
BASE = (0.00, 0.32, 0.21, 0.20, 0.27, 0.00)
CONTEXT_TOTAL = 0.21
HANDOVERS = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
CATBOOST_SOURCE = "no_te_strong"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def blend6(weights, parts, oof, frame):
    raw = {}
    for year in YEARS:
        key = str(year)
        raw[year] = sum(w * parts[n][key] for w, n in zip(weights, NAMES6))
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, targets, predictions = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            targets.append(oof[str(history)]["target"].astype(float))
            predictions.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(targets) - np.concatenate(predictions)
        train_frame = frame.loc[index]
        valid_frame = frame.loc[oof[str(year)]["row_index"]]
        count = segment_correction(
            train_frame, residual, valid_frame, ["balls_before", "strikes_before"], 500)
        pitcher_count = segment_correction(
            train_frame, residual, valid_frame,
            ["pitcher_id", "balls_before", "strikes_before"], 300)
        pieces.append(np.clip(
            raw[year] + residual.mean() + 0.75 * count + 0.25 * pitcher_count, 0, 1))
    return np.concatenate(pieces)


def three_season(metrics):
    seasons = metrics["season_bootstrap"]
    means = [seasons[str(y)]["mean"] for y in YEARS]
    errors = [(seasons[str(y)]["ci95_high"] - seasons[str(y)]["ci95_low"]) / 3.9199
              for y in YEARS]
    average = sum(means) / 3.0
    combined = math.sqrt(sum(e * e for e in errors)) / 3.0
    return {"average_points": average * P, "se_points": combined * P,
            "ci95_low_points": (average - 1.96 * combined) * P,
            "season_points": [m * P for m in means]}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id")

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    factorization = dict(joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"])
    factorization["latent6_8"] = {
        str(year): 0.5 * (factorization["latent6"][str(year)]
                          + factorization["latent8"][str(year)])
        for year in YEARS}
    zero = {str(year): 0.0 for year in YEARS}
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend6(BASE, dict(common, factorization=zero), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(weights, slot):
        candidate = np.clip(
            blend6(weights, dict(common, factorization=slot), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["weights"] = {n: round(w, 4) for n, w in zip(NAMES6, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"{'candidate':>22} {'ctx':>5} {'fm':>5} {'avg':>7} {'CIlo':>7} {'2022':>7} "
          f"{'2023':>8} {'2024':>7} {'blocks':>7} {'pass':>5}")
    for variant in ('latent6', 'latent8', 'latent6_8'):
        for handover in HANDOVERS:
            context_weight = round(CONTEXT_TOTAL - handover, 4)
            weights = (BASE[0], BASE[1], context_weight, BASE[3], BASE[4],
                       round(handover, 4))
            assert abs(sum(weights) - 1.0) < 1e-9, weights
            label = f"{variant}_fm{handover:.2f}"
            results[label] = evaluate(weights, factorization[variant])
            results[label]["handover"] = handover
            results[label]["variant"] = variant
            m = results[label]; t = m["three_season"]
            print(f"{label:>22} {context_weight:5.2f} {handover:5.2f} "
                  f"{t['average_points']:7.2f} {t['ci95_low_points']:7.2f} "
                  f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
                  f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible,
                   key=lambda l: results[l]["three_season"]["ci95_low_points"],
                   default=None)

    blocked_by = {}
    for label, m in results.items():
        t = m["three_season"]
        reasons = []
        if t["ci95_low_points"] <= 0:
            reasons.append("sign")
        if t["average_points"] < AVERAGE_FLOOR:
            reasons.append("magnitude")
        if any(v < -1e-9 for v in t["season_points"]):
            reasons.append("season")
        if m["monthly_block_win_rate"] < BLOCK_FLOOR:
            reasons.append("blocks")
        blocked_by[label] = reasons

    OUTPUT.write_text(json.dumps({
        "experiment": "V137b_context_slot_fine_handover",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27), Public 1047.03653",
        "rationale": (
            "V136 showed the Context axis unanimous in direction and blocked in "
            "practice, because Context still earns diversity on 2024 where every "
            "correlation drops. V120 solved the same shape for v17 by replacing it "
            "rather than emptying its weight. The factorization network was rejected as "
            "an addition, but it fixes precisely Context's weakness: -558 against -1333 "
            "on 2023, and lower correlation on all three seasons."
        ),
        "profile": {
            "context": {"2022": 2224, "2023": -1333, "2024": 725,
                        "corr": [0.919, 0.903, 0.817]},
            "factorization_latent6": {"2022": 2150, "2023": -558, "2024": 338,
                                      "corr": [0.888, 0.841, 0.777]},
        },
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}"},
        "handovers": list(HANDOVERS),
        "results": {k: {"weights": v["weights"], "handover": v["handover"],
                        "variant": v["variant"], "three_season": v["three_season"],
                        "bootstrap_2024": v["bootstrap_2024"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "blocked_by": blocked_by,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for reasons in blocked_by.values():
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    print(f"\nblocking condition counts across {len(results)} candidates: {counts}")
    print(f"eligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
