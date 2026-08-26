"""V143: deliberately violate the gate once, to find out which fold to believe.

Nine independent experiments -- V123, V124, V127, V132, V135, V136, V139, V141, V142 --
were rejected with the identical signature: 2023 improves substantially, 2024 (or in one
case 2022) goes negative, and the monthly block win rate falls short. All the remaining
headroom lives in that one trade, and the four leaderboard results cannot settle it,
because every one of them concerned a candidate whose folds *agreed* in sign. They
establish that the magnitude reading is unreliable; they say nothing about a fold
conflict.

Two readings of the situation remain, and they are not distinguishable from the
development data:

  1. 2024 represents 2025 faithfully, the 2023 gains are fold-specific, and the gate is
     right to veto.
  2. 2024 is anomalous, and nine candidates' worth of real improvement is being refused.

One submission decides it. That is worth spending, because it does not merely test one
candidate -- it tests the veto that has blocked an entire family.

Which candidate carries the most information is a separate question, and the answer is
not the largest 2023 gain. The nine split into two families:

  * 2024 negative: V123 (+232.06 / -14.52), V135 (+111.69 / -11.13), V142 (+79.48 /
    -21.54), and others. These test whether 2024 deserves a veto.
  * 2024 positive, only 2022 negative: V132's `onehot_w0.36` -- 2022 -1.27, 2023 +23.97,
    2024 +5.24, blocks 80%. This clears three of the four conditions, blocks included.

The second family is the better bet, and not only because it is milder. The fold that
objects there is 2022, and 2022 is the one fold that receives *no calibration at all* --
no earlier season exists to fit residuals on, so the raw blend is used, as V126
documented while measuring the level. It is simultaneously the oldest fold, the furthest
from 2025, and the one whose measurement conditions least resemble deployment. A veto
from 2022 is the weakest veto in the set.

Everything is re-measured against V138 rather than V122, since that is what a submission
would actually be compared to.

PRE-REGISTERED INTERPRETATION, fixed before any result is seen:

  * gain > +2 points  -> the objecting fold's veto is too strict; reopen that family and
                        re-rank the nine under a gate that drops it.
  * loss             -> the veto is validated; keep V138 and close this line for good.
  * within +/-2      -> inconclusive, which counts as validated. No evidence to loosen a
                        condition is not evidence to loosen it.

The gate is not being changed here. It is being *tested*, once, with the result written
down in advance so that whatever comes back cannot be reinterpreted after the fact.
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
from evaluate_v137_context_slot_replacement import NAMES6, blend6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v143_gate_violation_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
FACTORIZATION_SOURCE = "latent8"
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
# Form funds any increase, as in V132: it is the only component large enough, and V139
# showed the others are all at their local optimum.
CATBOOST_WEIGHTS = (0.27, 0.31, 0.36, 0.40)
P = 100000.0 / 0.25


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
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    pool = dict(joblib.load("artifacts/v116_catboost_predictions.joblib")["predictions"])
    pool.update(joblib.load(
        "artifacts/v123_catboost_capacity_predictions.joblib")["predictions"])
    pool.update(joblib.load(
        "artifacts/v142_environment_removal_predictions.joblib")["predictions"])
    print("slot candidates: " + ", ".join(sorted(pool)), flush=True)

    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6),
               dict(common, catboost=pool["no_te_strong"]), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(slot, catboost_weight):
        form_weight = round(BASE["form"] - (catboost_weight - BASE["catboost"]), 6)
        if form_weight <= 0:
            return None
        weights = (BASE["v17"], form_weight, BASE["context"], BASE["network"],
                   round(catboost_weight, 6), BASE["factorization"])
        assert abs(sum(weights) - 1.0) < 1e-9, weights
        candidate = np.clip(
            blend6(weights, dict(common, catboost=slot), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        metrics["failing_seasons"] = [
            str(year) for year, value in zip(YEARS, t["season_points"]) if value < 0]
        metrics["weights"] = {n: round(w, 6) for n, w in zip(NAMES6, weights)}
        return metrics

    results = {}
    print(f"\n{'candidate':>20} {'cb':>5} {'form':>5} {'min':>7} {'avg':>7} "
          f"{'2022':>7} {'2023':>8} {'2024':>7} {'blocks':>7} {'fails':>10}")
    for name in ("onehot", "no_env", "no_season", "no_te_strong", "long"):
        if name not in pool:
            continue
        for weight in CATBOOST_WEIGHTS:
            metrics = evaluate(pool[name], weight)
            if metrics is None:
                continue
            label = f"{name}_cb{weight:.2f}"
            results[label] = metrics
            t = metrics["three_season"]
            print(f"{label:>20} {weight:5.2f} {metrics['weights']['form']:5.2f} "
                  f"{metrics['min_season_points']:7.2f} {t['average_points']:7.2f} "
                  f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
                  f"{t['season_points'][2]:7.2f} "
                  f"{metrics['monthly_block_win_rate']:7.0%} "
                  f"{','.join(metrics['failing_seasons']) or '-':>10}", flush=True)

    # The probe wants the candidate whose two readings disagree most while keeping the
    # objection confined to 2022 -- the uncalibrated fold, and so the weakest veto.
    def probe_score(label):
        m = results[label]
        if m["failing_seasons"] != ["2022"]:
            return -1e9
        if m["monthly_block_win_rate"] < 0.75:
            return -1e9
        return m["three_season"]["average_points"]

    ranked = sorted(results, key=probe_score, reverse=True)
    chosen = ranked[0] if probe_score(ranked[0]) > -1e9 else None
    if chosen is None:
        print("\nno candidate confines its objection to 2022 while clearing blocks; "
              "the probe would have to spend its veto test on a 2024 objection instead",
              flush=True)
        fallback = max(results, key=lambda l: results[l]["three_season"]["average_points"])
        print(f"  widest-disagreement fallback: {fallback}", flush=True)
    else:
        m = results[chosen]
        print(f"\nchosen probe: {chosen}", flush=True)
        print(f"  weights: " + "  ".join(f"{k}={v}" for k, v in m["weights"].items()),
              flush=True)
        print(f"  three-season average says +{m['three_season']['average_points']:.2f}; "
              f"weakest season says {m['min_season_points']:+.2f}", flush=True)
        print(f"  objection confined to {m['failing_seasons']}, blocks "
              f"{m['monthly_block_win_rate']:.0%}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V143_gate_violation_probe",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "purpose": (
            "Nine experiments were rejected with one signature: 2023 up, another fold "
            "down, blocks short. The four leaderboard results cannot settle it because "
            "all four concerned candidates whose folds agreed in sign. One submission "
            "tests the veto that blocked the whole family."
        ),
        "why_a_2022_objection": (
            "2022 receives no calibration at all -- no earlier season exists to fit "
            "residuals on, so the raw blend is used (V126). It is also the oldest fold "
            "and the furthest from 2025, so its measurement conditions least resemble "
            "deployment. A veto from 2022 is the weakest in the set."
        ),
        "pre_registered_interpretation": {
            "gain_above_2_points": ("the objecting fold's veto is too strict; reopen the "
                                    "family and re-rank the nine without it"),
            "loss": "the veto is validated; keep V138 and close this line",
            "within_2_points": ("inconclusive, counted as validated -- no evidence to "
                                "loosen a condition is not evidence to loosen it"),
        },
        "gate_status": "deliberately violated, once, with the reading fixed in advance",
        "catboost_weights": list(CATBOOST_WEIGHTS),
        "results": {k: {"weights": v["weights"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "failing_seasons": v["failing_seasons"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "chosen_probe": chosen,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
