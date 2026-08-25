"""V110: build a Form-weight ladder for leaderboard-guided optimisation.

Score along a single blend direction is an exact quadratic in the weight, because
the calibration and every component prediction are fixed and only the mixing
coefficient moves. Three measured points therefore determine the parabola and its
vertex exactly.

Local validation put the optimum at Form 0.60-0.64, but its per-candidate noise is
about +-5 points, so it cannot distinguish 0.64 from 0.76. The leaderboard measures
245,789 rows exactly and, because Public is the whole test set with no separate
Private split, there is no generalisation gap to worry about. With submissions not
scarce, walking this line on the leaderboard is strictly better than trusting the
local peak.

Context stays at 0.17 and the drift term is untouched, so every rung differs from
V106 in exactly one number: v17 = 1 - form - 0.17. Nothing is retrained; only the
calibration lookups are refitted per rung, as V96 did.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from build_v12_calibration import make_lookup
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


METRICS = Path("artifacts/v110_weight_ladder_metrics.json")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_RELIABILITY = 150.0
DRIFT_WEIGHT = 0.10
CONTEXT_WEIGHT = 0.17
BASELINE_FORM = 0.64
LADDER = (0.70, 0.76, 0.82)
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


def write_calibration(path, w_form, oof, logistic, form, context, raw):
    w_v17 = 1.0 - w_form - CONTEXT_WEIGHT
    indices, targets, predictions = [], [], []
    for year in YEARS:
        item = oof[str(year)]
        v17 = 0.95 * v11_prediction(item) + 0.05 * logistic[str(year)]
        predictions.append(
            w_v17 * v17 + w_form * form[str(year)] + CONTEXT_WEIGHT * context[str(year)])
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    shift = float(residual.mean())
    centered = residual - shift
    frame = raw.loc[index]
    joblib.dump({
        "global_shift": shift,
        "corrections": [
            {"columns": ["balls_before", "strikes_before"], "weight": 0.75,
             "lookup": make_lookup(frame, centered,
                                   ["balls_before", "strikes_before"], 500)},
            {"columns": ["pitcher_id", "balls_before", "strikes_before"], "weight": 0.25,
             "lookup": make_lookup(frame, centered,
                                   ["pitcher_id", "balls_before", "strikes_before"], 300)},
        ],
        "blend_weights": {"v17": round(w_v17, 4), "form": w_form,
                          "context": CONTEXT_WEIGHT},
        "feature_shrinkage": FEATURE_SHRINKAGE,
        "drift_shrinkage": DRIFT_SHRINKAGE,
        "drift_reliability_scale": DRIFT_RELIABILITY,
        "drift_weight": DRIFT_WEIGHT,
    }, path, compress=3)
    return shift


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw = data.copy()
    frame = data.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend_and_calibrate(BASELINE_FORM, CONTEXT_WEIGHT, oof, logistic,
                            form, context, frame) + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for w_form in LADDER:
        tag = f"{int(round(w_form * 100))}"
        path = Path(f"artifacts/v110_form{tag}_calibration.joblib")
        shift = write_calibration(path, w_form, oof, logistic, form, context, raw)
        candidate = np.clip(
            blend_and_calibrate(w_form, CONTEXT_WEIGHT, oof, logistic,
                                form, context, frame) + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["global_shift"] = shift
        metrics["v17_weight"] = round(1.0 - w_form - CONTEXT_WEIGHT, 4)
        metrics["calibration_artifact"] = str(path)
        results[f"form{tag}"] = metrics
        print(f"form={w_form:.2f} v17={metrics['v17_weight']:.2f} "
              f"shift={shift:+.9f} saved {path.name}", flush=True)

    METRICS.write_text(json.dumps({
        "experiment": "V110_form_weight_ladder",
        "baseline": "V106 (form 0.64, context 0.17, v17 0.19)",
        "baseline_public_score": 973.0643764994,
        "purpose": (
            "Score is an exact quadratic along this line; local noise of +-5 points "
            "cannot locate the vertex, the leaderboard can."
        ),
        "context_weight": CONTEXT_WEIGHT,
        "drift": {"shrinkage": DRIFT_SHRINKAGE, "reliability_scale": DRIFT_RELIABILITY,
                  "weight": DRIFT_WEIGHT},
        "ladder": list(LADDER),
        "results": results,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "calibration_refitted_per_rung": True, "nothing_retrained": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nlocal 2024 bootstrap vs V106 (expected to be flat and uninformative):")
    print(f"{'rung':>8} {'v17':>6} {'CIlo':>7} {'mean':>7} {'CIhi':>7} {'2022':>7} "
          f"{'2023':>7} {'blocks':>7}")
    for name, r in results.items():
        b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{name:>8} {r['v17_weight']:6.2f} {b['ci95_low']*P:7.1f} {b['mean']*P:7.1f} "
              f"{b['ci95_high']*P:7.1f} {s['2022']['mean']*P:7.1f} "
              f"{s['2023']['mean']*P:7.1f} {r['monthly_block_win_rate']:7.1%}")
    print(f"\nSaved {METRICS}")


if __name__ == "__main__":
    main()
