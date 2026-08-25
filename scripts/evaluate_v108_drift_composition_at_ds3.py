"""V108: re-test the drift term's composition at its new shrinkage and weight.

V100 tried broadening the drift term beyond the success-rate delta and every
combination failed. That was against shrinkage 50 and a weight of 0.15. Both have
moved: V106 runs the term at shrinkage 3 with weight 0.10, so the delta it reads is
far less smoothed and enters with a smaller coefficient. Whether the other failure
modes help is therefore an open question again.

The three official failure definitions map onto three deltas:

  middle_rate  -> "스트라이크존 가운데 부근"
  ball_rate    -> "스트라이크존에서 크게 벗어남"
  reverse_rate -> "포수 요구 방향과 반대"

Sub-weights are fixed a priori as in V100 — fitting them on the validation seasons
is what sank V78 and V80. Only the overall scale is swept. No model is retrained,
so this is the cheapest remaining check.
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


OUTPUT = Path("artifacts/v108_drift_composition_metrics.json")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
RELIABILITY = 150.0
W_FORM, W_CONTEXT = 0.64, 0.17
BASELINE_SCALE = 0.10
SCALES = (0.05, 0.10, 0.15, 0.20, 0.30)
ROUNDS = 3000
SEED = 20260825
P = 100000.0 / 0.25


def bootstrap(frame, base, candidate, mask, rounds=ROUNDS, seed=SEED):
    part = frame.loc[mask]
    target = part["target"].to_numpy()
    gain = (base[mask] - target) ** 2 - (candidate[mask] - target) ** 2
    grouped = pd.DataFrame({"p": part["pitcher_id"].to_numpy(), "g": gain}) \
        .groupby("p")["g"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(sums), size=(rounds, len(sums)))
    draws = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    return {"mean": float(gain.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975))}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = data.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    block = add_training_inseason_features(frame, shrinkage=DRIFT_SHRINKAGE).loc[order]
    inside = np.expm1(block["ins_log_n_pitcher"].to_numpy(dtype=float))
    reliability = inside / (inside + RELIABILITY)

    def d(name):
        return np.nan_to_num(
            block[f"dlt_asof_{name}"].to_numpy(dtype=float), nan=0.0)

    success, middle = d("pitcher_success_rate"), d("pitcher_middle_rate")
    ball, reverse = d("pitcher_ball_rate"), d("pitcher_reverse_rate")
    batter = d("batter_success_rate")
    failure = -(middle + ball + reverse) / 3.0
    composites = {
        "success": success,
        "success_middle": 0.5 * success - 0.5 * middle,
        "failure_modes": failure,
        "combined": 0.5 * success + 0.5 * failure,
        "pitcher_batter": 0.75 * success + 0.25 * batter,
        "success_reverse": 0.5 * success - 0.5 * reverse,
    }
    terms = {name: value * reliability for name, value in composites.items()}

    blend = blend_and_calibrate(W_FORM, W_CONTEXT, oof, logistic, form, context, frame)
    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(blend + BASELINE_SCALE * terms["success"], 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for name, term in terms.items():
        for scale in SCALES:
            if name == "success" and scale == BASELINE_SCALE:
                continue
            label = f"{name}_w{scale:.2f}"
            candidate = np.clip(blend + scale * term, 0, 1)
            metrics = development_metrics(reference, candidate)
            metrics["bootstrap_2024"] = bootstrap(
                validation_frame, baseline, candidate, mask_2024)
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
            results[label] = metrics

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["mean"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["mean"],
                   default=None)

    output = {
        "experiment": "V108_drift_composition_at_ds3",
        "baseline": "V106 (success delta only, shrinkage 3, scale 0.10)",
        "baseline_public_score": 973.0643764994,
        "drift_shrinkage": DRIFT_SHRINKAGE, "reliability_scale": RELIABILITY,
        "blend_weights": {"v17": round(1 - W_FORM - W_CONTEXT, 2),
                          "form": W_FORM, "context": W_CONTEXT},
        "composites": {
            "success": "dlt_success",
            "success_middle": "0.5*success - 0.5*middle",
            "failure_modes": "-(middle + ball + reverse)/3",
            "combined": "0.5*success + 0.5*failure_modes",
            "pitcher_batter": "0.75*success + 0.25*batter",
            "success_reverse": "0.5*success - 0.5*reverse",
        },
        "scales": list(SCALES),
        "results": results, "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "sub_weights_fixed_a_priori": True},
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("gains vs V106 baseline, by 2024 bootstrap:")
    print(f"{'candidate':>24} {'CIlo':>7} {'mean':>7} {'CIhi':>7} {'2022':>7} {'2023':>7} "
          f"{'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["mean"])[:18]:
        r = results[label]; b = r["bootstrap_2024"]; sb = r["season_bootstrap"]
        print(f"{label:>24} {b['ci95_low']*P:7.1f} {b['mean']*P:7.1f} {b['ci95_high']*P:7.1f} "
              f"{sb['2022']['mean']*P:7.1f} {sb['2023']['mean']*P:7.1f} "
              f"{r['monthly_block_win_rate']:7.1%} {'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted], indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
