"""V119: refit all five blend weights now that CatBoost is a component.

V116 carved CatBoost's 0.14 out of the V17 share alone, because that keeps the change
to one axis and made the component's own effect readable. It is almost certainly not
the optimum. Refitting the whole weight vector after adding a component has paid four
times in a row here (V96 +6.5, V105, V106 +7.1, V110 closed an axis without a
submission), and the reason is always the same mis-specification: the old weights were
fitted when the components were different, so every one of them is now answering a
question that no longer holds.

The search follows V110's method rather than a brute-force simplex. V110 established
that the +-5 point local noise is largely *common* across candidates built from the
same out-of-fold predictions, so absolute confidence intervals are wide while the
*shape* of a ladder is well determined. So each axis gets a ladder, a parabola is
fitted to it to locate the vertex, and only the joint step along the combined vertex
direction is then measured against the full pre-registered gate. That costs about
sixty evaluations instead of several hundred, and spends the precision where it
actually exists.

Weights are moved one axis at a time with the remaining four rescaled
proportionally, so every candidate is a valid convex combination and no axis change
silently reassigns another component's share.
"""

import json
import sys
import time
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


OUTPUT = Path("artifacts/v119_weight_refit_metrics.json")
NAMES = ("v17", "form", "context", "network", "catboost")
BASE = (0.05, 0.40, 0.21, 0.20, 0.14)
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
CATBOOST_SOURCE = "no_te_strong"
LADDER = (-0.08, -0.04, 0.04, 0.08)
JOINT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
P = 100000.0 / 0.25


def shift_axis(base, axis, delta):
    """Move one weight by delta and rescale the others proportionally."""
    target = max(0.0, base[axis] + delta)
    rest = 1.0 - base[axis]
    scale = (1.0 - target) / rest if rest > 0 else 0.0
    return tuple(target if i == axis else w * scale for i, w in enumerate(base))


def blend(weights, parts, oof, frame):
    raw = {}
    for year in YEARS:
        key = str(year)
        raw[year] = sum(w * parts[name][key] for w, name in zip(weights, NAMES))
    pieces = []
    for year in YEARS:
        if year == 2022:
            pieces.append(np.clip(raw[year], 0, 1))
            continue
        index, target, prediction = [], [], []
        for history in [y for y in YEARS if y < year]:
            index.append(oof[str(history)]["row_index"])
            target.append(oof[str(history)]["target"].astype(float))
            prediction.append(raw[history])
        index = np.concatenate(index)
        residual = np.concatenate(target) - np.concatenate(prediction)
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


def vertex(deltas, gains):
    """Parabola vertex through (0, 0) and the ladder points, or None if not concave."""
    x = np.array((0.0,) + tuple(deltas))
    g = np.array((0.0,) + tuple(gains))
    a, b, _ = np.polyfit(x, g, 2)
    if a >= 0:
        return None, float(a), float(b)
    return float(-b / (2 * a)), float(a), float(b)


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
    parts = {
        "v17": {str(y): 0.95 * v11_prediction(oof[str(y)]) + 0.05 * logistic[str(y)]
                for y in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    started = time.time()
    baseline = np.clip(blend(BASE, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
    print(f"baseline evaluated in {time.time() - started:.0f}s per candidate", flush=True)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(weights, full=False):
        candidate = np.clip(
            blend(weights, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        if full:
            metrics["season_bootstrap"] = {
                str(year): bootstrap(validation_frame, baseline, candidate,
                                     (validation_frame["season"] == year).to_numpy())
                for year in YEARS}
        metrics["weights"] = {n: round(w, 4) for n, w in zip(NAMES, weights)}
        return metrics

    ladders, vertices = {}, {}
    for axis, name in enumerate(NAMES):
        gains, rows = [], {}
        for delta in LADDER:
            weights = shift_axis(BASE, axis, delta)
            metrics = evaluate(weights)
            gain = metrics["bootstrap_2024"]["mean"] * P
            gains.append(gain)
            rows[f"{delta:+.2f}"] = metrics
            print(f"  {name:9s} {delta:+.2f} -> {name}={weights[axis]:.4f}  "
                  f"gain {gain:+7.2f}", flush=True)
        ladders[name] = rows
        peak, a, b = vertex(LADDER, gains)
        vertices[name] = {"delta": peak, "quadratic_a": a, "quadratic_b": b,
                          "concave": peak is not None}
        # A convex ladder has no interior optimum, so the best sampled point is used
        # as the direction instead of an extrapolated vertex that would run away.
        if peak is None:
            peak = LADDER[int(np.argmax(gains))]
            vertices[name]["delta"] = float(peak)
        clipped = float(np.clip(peak, min(LADDER), max(LADDER)))
        vertices[name]["clipped_delta"] = clipped
        print(f"  {name:9s} vertex {peak:+.4f} "
              f"({'concave' if a < 0 else 'convex, best point used'}), "
              f"clipped {clipped:+.4f}", flush=True)

    # The per-axis vertices are gradients measured independently, so applying all of
    # them at once overshoots; the joint direction is walked at several scales and the
    # step is chosen by the same gate every earlier candidate faced.
    direction = np.array([vertices[name]["clipped_delta"] for name in NAMES])
    print(f"\njoint direction: " + "  ".join(
        f"{n}{d:+.3f}" for n, d in zip(NAMES, direction)), flush=True)

    results = {}
    for scale in JOINT_SCALES:
        weights = np.clip(np.array(BASE) + scale * direction, 0.0, None)
        weights = weights / weights.sum()
        label = f"joint_{scale:.2f}"
        results[label] = evaluate(tuple(weights), full=True)
        b = results[label]["bootstrap_2024"]
        print(f"  {label}  " + " ".join(
            f"{n}={w:.3f}" for n, w in zip(NAMES, weights))
            + f"  CIlo {b['ci95_low']*P:+7.2f}  mean {b['mean']*P:+7.2f}", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V119_five_way_weight_refit",
        "baseline": "V117 (0.05 / 0.40 / 0.21 / 0.20 / 0.14)",
        "baseline_weights": {n: w for n, w in zip(NAMES, BASE)},
        "rationale": (
            "V116 took CatBoost's share from the V17 layer alone to keep the component's "
            "own effect readable. Refitting the whole vector after adding a component "
            "has paid four times here, because the old weights were fitted against a "
            "different set of components."
        ),
        "method": (
            "Per-axis ladders with the other four weights rescaled proportionally, a "
            "parabola fitted to each ladder for its vertex, then the joint step along "
            "the combined direction measured at several scales. Follows V110, which "
            "found local noise to be common across candidates so ladder shape is far "
            "better determined than any absolute interval."
        ),
        "ladder_offsets": list(LADDER),
        "joint_scales": list(JOINT_SCALES),
        "ladders": ladders,
        "vertices": vertices,
        "joint_direction": {n: float(d) for n, d in zip(NAMES, direction)},
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nranked by 2024 bootstrap lower bound (vs V117):")
    print(f"{'candidate':>12} {'CIlo':>8} {'mean':>8} {'2022':>8} {'2023':>8} "
          f"{'blocks':>7} {'pass':>5}")
    for label in sorted(results, key=lambda l: -results[l]["bootstrap_2024"]["ci95_low"]):
        r = results[label]; b = r["bootstrap_2024"]; s = r["season_bootstrap"]
        print(f"{label:>12} {b['ci95_low']*P:8.2f} {b['mean']*P:8.2f} "
              f"{s['2022']['mean']*P:8.2f} {s['2023']['mean']*P:8.2f} "
              f"{r['monthly_block_win_rate']:7.1%} "
              f"{'YES' if label in eligible else '-':>5}")
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
