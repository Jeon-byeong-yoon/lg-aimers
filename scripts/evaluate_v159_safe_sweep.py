"""V159: sweep everything cached against V156, filtering only on "no fold loses".

V156 retired the magnitude threshold. Its development signal was a three-season average of
+0.62 with a 40% monthly block rate -- far below the +3.0 floor pre-registered in V132 --
and it returned +1.33, four times what V154 returned on a signal four times larger. The
seven-submission record now reads:

    submission   min season   3-season avg   actual
    V117              +9.9         +31.4    +25.67
    V122             +5.01        +10.35    +18.80
    V138             +0.64        +25.08     +3.51
    V144             -1.86         +8.43     -1.14
    V151            -32.36             -     -2.44
    V154             +2.29         +2.34     +0.30
    V156             +0.00         +0.62     +1.33

The minimum season predicts the sign without exception. Nothing predicts the magnitude --
the average's ratio runs from 0.13 to 2.15. So a magnitude floor is not a filter, it is a
way of discarding expected gain, and with submissions renewing daily there is no reason to
pay it.

That reopens everything previously dismissed as too small. Every component in the blend was
selected under an earlier configuration -- Form's smoothing in V102, Context in V31,
the network in V112, CatBoost's variant in V116, the factorization latent in V137 -- and
the blend has been rebuilt around each of them several times since. This tries every
cached alternative in every slot, plus small pairwise weight transfers, and keeps whatever
puts no fold into a loss.

Excluded: the V158 variants. They were built with a global target prior where V102 used a
per-fold one, so their control failed to reproduce the stored predictions (Form differing
by up to 3.8e-02) and their absolute level is not comparable to this baseline. V157 is
included because it used the per-fold prior and its control reproduced exactly.
"""

import json
import math
import sys
import time
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


OUTPUT = Path("artifacts/v159_safe_sweep_metrics.json")
# V156 exactly.
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
W_COUNT, W_PITCHER_COUNT, W_EXPERIENCE = 0.55, 0.25, 0.20
EXPERIENCE_SMOOTHING = 2000.0
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
LIVE = ("form", "context", "network", "catboost", "factorization")
TRANSFER_STEPS = (0.02, 0.04)
P = 100000.0 / 0.25


def load_alternatives():
    """Every cached alternative per slot, from sources with a clean per-fold prior."""
    out = {slot: {} for slot in LIVE}
    forms = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")["forms"]
    for key, value in forms.items():
        out["form"][f"v102:{key:g}"] = value
    for key, value in joblib.load(
            "artifacts/v31_feature_removal_predictions.joblib").items():
        out["context"][f"v31:{key}"] = value["context"]
    for key, value in joblib.load(
            "artifacts/v157_context_refit_predictions.joblib")["predictions"].items():
        out["context"][f"v157:{key}"] = value
    for key, value in joblib.load(
            "artifacts/v112_network_weight_predictions.joblib")["networks"].items():
        out["network"][f"v112:{key}"] = value
    for key, value in joblib.load(
            "artifacts/v115_network_tuning_predictions.joblib")["networks"].items():
        out["network"][f"v115:{key}"] = value
    for stem, inner in (("v116_catboost", "predictions"),
                        ("v123_catboost_capacity", "predictions"),
                        ("v142_environment_removal", "predictions"),
                        ("v141_disjoint_subset", "predictions"),
                        ("v153_projected_prior", "catboost")):
        payload = joblib.load(f"artifacts/{stem}_predictions.joblib")[inner]
        tag = stem.split("_")[0]
        for key, value in payload.items():
            out["catboost"][f"{tag}:{key}"] = value
    for stem in ("v130b_interaction_network", "v130c_interaction_network"):
        payload = joblib.load(f"artifacts/{stem}_predictions.joblib")["predictions"]
        tag = stem.split("_")[0]
        for key, value in payload.items():
            out["factorization"][f"{tag}:{key}"] = value
    return out


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    data.pop("control_success")
    raw_frame = data.drop(columns="row_id")
    raw_frame["experience_bin"] = pd.cut(
        raw_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    alternatives = load_alternatives()
    incumbent = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": alternatives["form"]["v102:20"],
        "context": alternatives["context"]["v31:no_matchup_hte"],
        "network": alternatives["network"]["v112:without_season"],
        "catboost": alternatives["catboost"]["v153:projected"],
        "factorization": alternatives["factorization"]["v130c:latent8"],
    }
    print("cached alternatives per slot: " + ", ".join(
        f"{s} {len(alternatives[s])}" for s in LIVE), flush=True)

    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame.drop(columns="experience_bin"),
            shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()
    validation_frame["experience_bin"] = pd.cut(
        validation_frame["asof_pitcher_n"], EDGES, labels=LABELS).astype(str)
    season_of = validation_frame["season"].to_numpy()

    def pipeline(parts, weights):
        raw = {year: sum(w * parts[n][str(year)] for w, n in zip(weights, NAMES6))
               for year in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(raw[year], 0, 1))
                continue
            history = [h for h in YEARS if h < year]
            index = np.concatenate([oof[str(h)]["row_index"] for h in history])
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - raw[h] for h in history])
            train = raw_frame.loc[index]
            valid = raw_frame.loc[oof[str(year)]["row_index"]]
            pieces.append(np.clip(
                raw[year] + residual.mean()
                + W_COUNT * segment_correction(
                    train, residual, valid, ["balls_before", "strikes_before"], 500)
                + W_PITCHER_COUNT * segment_correction(
                    train, residual, valid,
                    ["pitcher_id", "balls_before", "strikes_before"], 300)
                + W_EXPERIENCE * segment_correction(
                    train, residual, valid, ["experience_bin"], EXPERIENCE_SMOOTHING),
                0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    weights_base = tuple(BASE[n] for n in NAMES6)
    baseline = pipeline(incumbent, weights_base)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(candidate):
        metrics = development_metrics(reference, candidate)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (season_of == year)) for year in YEARS}
        metrics["bootstrap_2024"] = metrics["season_bootstrap"]["2024"]
        t = three_season(metrics)
        metrics["three_season"] = t
        metrics["min_season_points"] = min(t["season_points"])
        return metrics

    results, started = {}, time.time()
    print("\nsingle-slot swaps:", flush=True)
    for slot in LIVE:
        for name, values in alternatives[slot].items():
            if values is incumbent[slot]:
                continue
            parts = dict(incumbent)
            parts[slot] = values
            label = f"{slot}={name}"
            results[label] = evaluate(pipeline(parts, weights_base))
            results[label]["kind"] = "slot"
            m = results[label]; t = m["three_season"]
            safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
            if safe:
                print(f"  {label:36s} 2022 {t['season_points'][0]:+7.2f}  "
                      f"2023 {t['season_points'][1]:+7.2f}  "
                      f"2024 {t['season_points'][2]:+7.2f}  "
                      f"avg {t['average_points']:+7.2f}  "
                      f"blk {m['monthly_block_win_rate']:4.0%}  SAFE", flush=True)
    print(f"  ({len(results)} swaps in {time.time() - started:.0f}s)", flush=True)

    print("\npairwise weight transfers:", flush=True)
    from itertools import permutations
    for source, target in permutations(LIVE, 2):
        for step in TRANSFER_STEPS:
            weights = dict(BASE)
            if weights[source] - step < -1e-12:
                continue
            weights[source] = round(weights[source] - step, 6)
            weights[target] = round(weights[target] + step, 6)
            vector = tuple(weights[n] for n in NAMES6)
            label = f"{source}->{target}_{step:.2f}"
            results[label] = evaluate(pipeline(incumbent, vector))
            results[label]["kind"] = "weight"
            m = results[label]; t = m["three_season"]
            safe = min(t["season_points"]) >= -1e-9 and max(t["season_points"]) > 0
            if safe:
                print(f"  {label:36s} 2022 {t['season_points'][0]:+7.2f}  "
                      f"2023 {t['season_points'][1]:+7.2f}  "
                      f"2024 {t['season_points'][2]:+7.2f}  "
                      f"avg {t['average_points']:+7.2f}  "
                      f"blk {m['monthly_block_win_rate']:4.0%}  SAFE", flush=True)

    def safe(label):
        t = results[label]["three_season"]["season_points"]
        return min(t) >= -1e-9 and max(t) > 0

    survivors = [l for l in results if safe(l)]
    promoted = max(survivors,
                   key=lambda l: (results[l]["min_season_points"],
                                  results[l]["three_season"]["average_points"]),
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V159_safe_sweep",
        "baseline": "V156 (Public 1052.1807428872)",
        "filter": ("no fold may lose, and at least one must gain; no magnitude threshold, "
                   "which V156 showed to be a way of discarding expected gain rather than "
                   "a filter"),
        "excluded": ("V158 variants: built with a global target prior where V102 used a "
                     "per-fold one, so their control failed to reproduce the stored "
                     "predictions and their level is not comparable to this baseline"),
        "alternatives_per_slot": {s: sorted(alternatives[s]) for s in LIVE},
        "transfer_steps": list(TRANSFER_STEPS),
        "results": {k: {"kind": v["kind"], "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "safe_candidates": sorted(survivors),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(results)} candidates, {len(survivors)} safe")
    if survivors:
        print(f"\nall safe candidates, ranked by weakest season then average:")
        for label in sorted(survivors,
                            key=lambda l: (-results[l]["min_season_points"],
                                           -results[l]["three_season"]["average_points"])):
            m = results[label]; t = m["three_season"]
            print(f"  {label:36s} min {m['min_season_points']:+7.2f}  "
                  f"avg {t['average_points']:+7.2f}  "
                  f"2022 {t['season_points'][0]:+7.2f}  "
                  f"2023 {t['season_points'][1]:+7.2f}  "
                  f"2024 {t['season_points'][2]:+7.2f}  "
                  f"blk {m['monthly_block_win_rate']:4.0%}")
    print(f"\npromoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
