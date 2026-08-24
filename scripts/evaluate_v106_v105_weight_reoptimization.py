"""V106: refit the blend, drift weight and drift shrinkage for V105's Form model.

Two plays that have already worked, combined.

The first is the V96 play. V105 replaced the Form model (its features are now
shrunk at 20 instead of 50) but kept V96's 0.27 / 0.52 / 0.21 split and its 0.15
drift weight, both fitted against the shrinkage-50 model. The same staleness was
worth +6.54 points at V41 and +11.77 at V96, each time after the Form model
improved.

The second is the V104 play. V105's drift shrinkage of 10 sat on the boundary of
that grid, so 3 and 5 are added here. V104 showed the feature axis was genuinely
interior at 20, but the drift axis was never tested below its edge.

Ranking is by the 2024 paired pitcher bootstrap lower bound, then by its mean.
Across four adopted candidates the cumulative 2024-bootstrap estimate now tracks
the leaderboard at a ratio of 1.012, so it is treated as an unbiased predictor
with roughly +-5 points of per-candidate noise.
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


OUTPUT = Path("artifacts/v106_weight_reoptimization_metrics.json")
FEATURE_SHRINKAGE = 20.0
FORM_WEIGHTS = (0.48, 0.52, 0.56, 0.60, 0.64)
CONTEXT_WEIGHTS = (0.17, 0.21, 0.25)
DRIFT_SHRINKAGES = (3.0, 5.0, 10.0, 15.0)
DRIFT_WEIGHTS = (0.10, 0.15, 0.20, 0.25)
RELIABILITY = 150.0
BASELINE = (0.52, 0.21, 10.0, 0.15)
SCREEN_TOP = 25
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

    terms = {}
    for shrinkage in DRIFT_SHRINKAGES:
        block = add_training_inseason_features(frame, shrinkage=shrinkage).loc[order]
        delta = np.nan_to_num(
            block["dlt_asof_pitcher_success_rate"].to_numpy(dtype=float), nan=0.0)
        inside = np.expm1(block["ins_log_n_pitcher"].to_numpy(dtype=float))
        terms[shrinkage] = delta * (inside / (inside + RELIABILITY))
    print(f"drift terms built for {DRIFT_SHRINKAGES}", flush=True)

    blends = {}
    for w_form in FORM_WEIGHTS:
        for w_context in CONTEXT_WEIGHTS:
            blends[(w_form, w_context)] = blend_and_calibrate(
                w_form, w_context, oof, logistic, form, context, frame)
            print(f"blend v17={1 - w_form - w_context:.2f} form={w_form:.2f} "
                  f"context={w_context:.2f} calibrated", flush=True)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    target_2024 = validation_frame.loc[mask_2024, "target"].to_numpy()
    baseline = np.clip(
        blends[(BASELINE[0], BASELINE[1])] + BASELINE[3] * terms[BASELINE[2]], 0, 1)
    base_error = (baseline[mask_2024] - target_2024) ** 2
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    screen, built = {}, {}
    for (w_form, w_context), blend in blends.items():
        for shrinkage in DRIFT_SHRINKAGES:
            for drift in DRIFT_WEIGHTS:
                setting = (w_form, w_context, shrinkage, drift)
                if setting == BASELINE:
                    continue
                label = (f"form{w_form:.2f}_ctx{w_context:.2f}"
                         f"_ds{shrinkage:.0f}_w{drift:.2f}")
                candidate = np.clip(blend + drift * terms[shrinkage], 0, 1)
                built[label] = candidate
                screen[label] = float(
                    (base_error - (candidate[mask_2024] - target_2024) ** 2).mean())
    shortlist = sorted(screen, key=screen.get, reverse=True)[:SCREEN_TOP]
    print(f"\nscreened {len(screen)}, bootstrapping top {len(shortlist)}", flush=True)

    results = {}
    for label in shortlist:
        candidate = built[label]
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS
        }
        results[label] = metrics

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["mean"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    output = {
        "experiment": "V106_v105_weight_reoptimization",
        "baseline": "V105 (0.27/0.52/0.21, drift shrinkage 10, w 0.15)",
        "baseline_public_score": 969.9639164589,
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "form_weights": list(FORM_WEIGHTS),
        "context_weights": list(CONTEXT_WEIGHTS),
        "drift_shrinkages": list(DRIFT_SHRINKAGES),
        "drift_weights": list(DRIFT_WEIGHTS),
        "reliability_scale": RELIABILITY,
        "screened": len(screen),
        "bootstrapped": len(shortlist),
        "cumulative_prediction_ratio": 1.012,
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "calibration_refitted_per_blend": True},
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ntop {len(shortlist)} by 2024 bootstrap (vs V105):")
    print(f"{'candidate':>34} {'v17':>5} {'CIlo':>7} {'mean':>7} {'CIhi':>7} "
          f"{'2022':>7} {'2023':>7} {'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["mean"]):
        r = results[label]; b = r["bootstrap_2024"]; sb = r["season_bootstrap"]
        wf = float(label.split("form")[1][:4]); wc = float(label.split("ctx")[1][:4])
        print(f"{label:>34} {1 - wf - wc:5.2f} {b['ci95_low']*P:7.1f} {b['mean']*P:7.1f} "
              f"{b['ci95_high']*P:7.1f} {sb['2022']['mean']*P:7.1f} "
              f"{sb['2023']['mean']*P:7.1f} {r['monthly_block_win_rate']:7.1%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}\npromoted={promoted}")
    if promoted:
        print(json.dumps(results[promoted], indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
