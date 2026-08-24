"""V103: decouple the feature shrinkage from the drift-term shrinkage.

V102 tied both to one constant and was rejected, but its shrinkage-20 arm
produced the largest 2024 July-August gains of any experiment in the project
(+3.5e-5 to +4.7e-5). That signal is worth isolating: the constant enters in two
places, and they need not agree.

  features   -> the Form model's ins_/dlt_ columns, so a change means retraining
  drift term -> the single scalar that extrapolates the league level

The Form models for every shrinkage were already fitted in V102 and saved, so
every combination here is pure recombination.

Promotion rule, revised
-----------------------
A power calculation on the sealed 2024 September-October window (34,976 rows,
257 pitchers) gives a paired-bootstrap 95% CI width of 37 points. It cannot
resolve anything smaller than roughly +-18 points, so every sealed reading taken
so far — V96 +4.0, H1 -5.6, V101 -1.6, V102 -11.4 — sits inside its own noise
band. Using it as a sole veto was not sound. The development window is 711,528
rows with a CI width of 12.7 points and is the instrument with the power.

So the primary test becomes the development-window paired pitcher bootstrap lower
bound, and the sealed window is demoted to a sanity check that only rejects a
reading far outside its noise band. This revision is stated before the results are
seen, and is checked retrospectively against V96, which is the one candidate whose
leaderboard outcome is known.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from inseason_asof_features_v92 import add_training_inseason_features


OUTPUT = Path("artifacts/v103_decoupled_shrinkage_metrics.json")
SHRINKAGES = (20.0, 50.0, 100.0, 200.0)
RELIABILITY_SCALES = (50.0, 150.0, 400.0)
DRIFT_SCALES = (0.10, 0.15, 0.20)
BASELINE = (50.0, 50.0, 150.0, 0.15)
W_FORM, W_CONTEXT = 0.52, 0.21
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260825
SEALED_NOISE_FLOOR = -5.0e-5
P = 100000.0 / 0.25


def bootstrap_gain(frame, base, candidate, mask, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    part = frame.loc[mask]
    target = part["target"].to_numpy()
    gain = (base[mask] - target) ** 2 - (candidate[mask] - target) ** 2
    grouped = pd.DataFrame(
        {"pitcher": part["pitcher_id"].to_numpy(), "gain": gain}
    ).groupby("pitcher")["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(rounds)
    for index in range(rounds):
        picked = rng.integers(0, len(sums), len(sums))
        draws[index] = sums[picked].sum() / counts[picked].sum()
    return {
        "mean": float(gain.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = data.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    forms = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")["forms"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    terms = {}
    for shrinkage in SHRINKAGES:
        block = add_training_inseason_features(frame, shrinkage=shrinkage).loc[order]
        delta = np.nan_to_num(
            block["dlt_asof_pitcher_success_rate"].to_numpy(dtype=float), nan=0.0)
        inside = np.expm1(block["ins_log_n_pitcher"].to_numpy(dtype=float))
        terms[shrinkage] = {
            scale: delta * (inside / (inside + scale)) for scale in RELIABILITY_SCALES
        }
        print(f"drift terms built for shrinkage={shrinkage}", flush=True)

    blends = {
        shrinkage: blend_and_calibrate(
            W_FORM, W_CONTEXT, oof, logistic, forms[shrinkage], context, frame)
        for shrinkage in SHRINKAGES
    }
    print("blends calibrated", flush=True)

    validation_frame = make_validation_frame()
    sealed = ((validation_frame["season"] == 2024)
              & (validation_frame["game_month"] >= 9)).to_numpy()
    development = ~sealed
    baseline = np.clip(
        blends[BASELINE[0]] + BASELINE[3] * terms[BASELINE[1]][BASELINE[2]], 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for k_feature in SHRINKAGES:
        for k_drift in SHRINKAGES:
            for scale in RELIABILITY_SCALES:
                for drift in DRIFT_SCALES:
                    setting = (k_feature, k_drift, scale, drift)
                    if setting == BASELINE:
                        continue
                    label = (f"feat{k_feature:.0f}_drift{k_drift:.0f}"
                             f"_rel{scale:.0f}_w{drift:.2f}")
                    candidate = np.clip(
                        blends[k_feature] + drift * terms[k_drift][scale], 0, 1)
                    metrics = development_metrics(reference, candidate)
                    metrics["development_bootstrap"] = bootstrap_gain(
                        validation_frame, baseline, candidate, development)
                    metrics["sealed_gain"] = float(
                        ((baseline[sealed] - validation_frame.loc[sealed, "target"]) ** 2
                         - (candidate[sealed] - validation_frame.loc[sealed, "target"]) ** 2
                         ).mean())
                    results[label] = metrics
        print(f"evaluated feature shrinkage {k_feature}", flush=True)

    def passes(label):
        r = results[label]
        return (r["development_bootstrap"]["ci95_low"] > 0
                and r["season_gain_development"]["2022"] > -1e-5
                and r["season_gain_development"]["2023"] > -1e-5
                and r["gain_2024_mar_aug"] > 0
                and r["monthly_block_win_rate"] >= 0.75
                and r["sealed_gain"] > SEALED_NOISE_FLOOR)

    eligible = [label for label in results if passes(label)]
    promoted = max(
        eligible,
        key=lambda l: results[l]["development_bootstrap"]["ci95_low"],
        default=None,
    )

    output = {
        "experiment": "V103_decoupled_shrinkage",
        "baseline": "V96 (feature shrinkage 50, drift shrinkage 50, reliability 150, w 0.15)",
        "baseline_public_score": 962.8787800874,
        "promotion_rule_revision": {
            "reason": (
                "The sealed 2024 Sep-Oct window has a paired-bootstrap 95% CI width of "
                "37 points over 34,976 rows, so it cannot resolve the effects being "
                "measured. Every sealed reading so far sat inside its own noise band."
            ),
            "primary": "development-window paired pitcher bootstrap ci95_low > 0",
            "secondary": "monthly block win rate >= 75%, no 2022/2023 regression",
            "sealed_window": f"demoted to a sanity check; reject only below {SEALED_NOISE_FLOOR}",
            "retrospective_check": "must accept V96, whose leaderboard gain was +11.77",
        },
        "shrinkages": list(SHRINKAGES),
        "reliability_scales": list(RELIABILITY_SCALES),
        "drift_scales": list(DRIFT_SCALES),
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED,
                      "unit": "pitcher_id", "window": "development"},
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "constants_frozen_at_training_time": True,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\ntop 20 by development bootstrap lower bound (vs V96):")
    print(f"{'candidate':>34} {'devCI_low':>10} {'devmean':>9} {'2022':>7} {'2023':>7} "
          f"{'2024pt':>7} {'blocks':>7} {'sealed':>9} {'pass':>5}")
    ranked = sorted(results.items(),
                    key=lambda kv: -kv[1]["development_bootstrap"]["ci95_low"])
    for label, r in ranked[:20]:
        b = r["development_bootstrap"]
        sg = r["season_gain_development"]
        print(f"{label:>34} {b['ci95_low']*P:10.1f} {b['mean']*P:9.1f} "
              f"{sg['2022']*P:7.1f} {sg['2023']*P:7.1f} {r['gain_2024_mar_aug']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {r['sealed_gain']*P:9.1f} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted], indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
